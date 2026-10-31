// p100-series MoE expert cache (moe-lru), CUDA side. See moe-lru.cuh and p100-docs/MOE.md.
#include "moe-lru.cuh"
#include "ggml-impl.h"
#include "mmvq.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

// wispoffates' A/B switch and pass timer are not ported: every feature runs its default arm, no per-pass fill counter
static bool ggml_cuda_ab_listed(const char *) { return false; }
static int ggml_cuda_ab_arm() { return 0; }
static unsigned long long * ggml_cuda_pass_fill_counter(int) { return nullptr; }

// p100-series: host mirrors of the LRU states, so the host can issue copy-engine transfers for the experts a prefill assembly
// misses. A state changes only when LRU_DECIDE runs on it (marked dirty there); a dirty mirror is re-read after one stream sync.
struct ggml_cuda_lru_mirror {
    struct entry { const ggml_tensor * state; std::vector<int32_t> h; bool dirty; };
    std::vector<entry> entries;
    entry * find(const ggml_tensor * s) {
        for (auto & e : entries) {
            if (e.state == s) {
                return &e;
            }
        }
        return nullptr;
    }
};

static __device__ unsigned long long lru_tail_skipped = 0, lru_tail_filled = 0, lru_tail_calls = 0;   // GGML_CUDA_LRU_TAIL_STATS

static __global__ void k_lru_decide(
        const int32_t * __restrict__ ids, int32_t * slot_ids,   // no __restrict__: lanes read what lane 0 wrote
        int32_t * state, int32_t * fill_list,
        int n_used, int n_tokens, int n_expert, int n_slot, int ids_stride, int exclusive,
        ggml_lru_ctrl * ctrl, int layer, int log_dev, int log_dev_index, int remap,
        int tail, int32_t * keep, int tail_stats, unsigned long long * pass_fills,
        lru_stash_ctl * stash, int stash_alloc,
        int host_max, int host_every, ggml_lru_host_req * hreq, const float * hx, int hx_stride, int n_embd) {
    const int lane = threadIdx.x;
    int32_t * owner    = state;                     // expert -> slot
    int32_t * slot_of  = state + n_expert;          // slot -> expert
    int32_t * last_use = state + n_expert + n_slot; // tick of the last use per slot
    int32_t * hslot    = state + n_expert + 2 * n_slot + 3; // exclusive: expert -> host slot, -1 while in a pool
    int32_t * hexpert  = state + 2 * n_expert + 2 * n_slot + 4; // exclusive: host slot -> expert
    const int32_t t0   = state[n_expert + 2 * n_slot];
    const int n_total = n_used * n_tokens;

    if (remap) {   // exclusive, large batch: ids in the assembled [pool slots | host slots] layout, state untouched
        for (int i = lane; i < n_total; i += WARP_SIZE) {
            const int32_t e = ids[(i / n_used) * ids_stride + i % n_used];
            slot_ids[i] = (e < 0 || e >= n_expert) ? e : (owner[e] >= 0 ? owner[e] : n_slot + hslot[e]);
        }
        return;
    }

    if (stash != nullptr) {
        // victim stash: every fill launched before this kernel has completed -> the flushes may take its entries; this layer's
        // host slots must hold what the state says before a fill reads them -> wait until its newest entry has been written back
        // (normally a pass ago; the flush for it was issued before this kernel, so it can always run)
        if (lane == 0) {
            volatile lru_stash_ctl * vs = stash;
            vs->snap = vs->counter;
            const uint32_t need = layer < 1024 ? vs->last[layer] : 0;
            if (need != 0) {
                while ((int32_t) (vs->flushed - need) < 0) {
                }
            }
        }
        __syncwarp();
    }

    for (int i = lane; i < n_total; i += WARP_SIZE) {
        const int32_t e = ids[(i / n_used) * ids_stride + i % n_used];
        if (keep) keep[i] = 1;
        if (e < 0 || e >= n_expert) continue;
        const int32_t s = owner[e];
        slot_ids[i] = s;
        if (s >= 0) last_use[s] = t0;
    }
    __syncwarp();

    int n_fill = 0;
    int n_skip = 0;
    int n_host_ent = 0;
    for (int i = 0; i < n_total; i++) {
        if (slot_ids[i] >= 0) continue;
        const int32_t e = ids[(i / n_used) * ids_stride + i % n_used];
        if (e < 0 || e >= n_expert) continue;
        const int32_t own = owner[e];
        if (own >= 0) {                              // a duplicate of an expert filled earlier in this batch
            if (lane == 0) slot_ids[i] = own;
            __syncwarp();
            continue;
        }
        if (tail > 0) {
            // p100-series route-tail skip (Strata's rule): a missing expert that every entry of the batch ranks >= tail is neither
            // filled nor computed; its entries read the token's rank-0 slot (resolved earlier in this loop) with keep = 0
            bool tail_only = true;
            for (int j = lane; j < n_total; j += WARP_SIZE) {
                if (ids[(j / n_used) * ids_stride + j % n_used] == e && j % n_used < tail) tail_only = false;
            }
            tail_only = __all_sync(0xffffffff, tail_only);
            if (tail_only) {
                if (lane == 0) {
                    // the entry must read an owned slot (finite data; weight 0) that is NOT already in this token's row: duplicate ids
                    // in a row break the multi-token MUL_MAT_ID ids helper (RTD: illegal address in tile_mm_id with the rank-0 slot).
                    // Refreshing its recency keeps a later fill of this batch from evicting it into another duplicate.
                    const int base = (i / n_used) * n_used;
                    int c = -1;
                    for (int m = 0; m < n_slot && c < 0; ++m) {
                        const int cand = (slot_ids[base] + 1 + m) % n_slot;
                        if (slot_of[cand] < 0) continue;
                        bool used = false;
                        for (int q = 0; q < n_used; ++q) {
                            if (slot_ids[base + q] == cand) { used = true; break; }
                        }
                        if (!used) c = cand;
                    }
                    if (c < 0) c = slot_ids[base];   // unreachable with n_slot > n_used owned slots
                    last_use[c] = t0;
                    slot_ids[i] = c;
                    keep[i] = 0;
                }
                n_skip++;
                __syncwarp();
                continue;
            }
        }
        // host: past the fill budget, and (LLAMA_MOE_LRU_HOST_EVERY=D) every D-th miss, so the CPU and the copies share them
        const int n_miss = n_fill + n_host_ent;
        if (host_max >= 0 && (n_fill >= host_max || (host_every > 0 && n_miss % host_every == host_every - 1))) {
            // host-computed miss: no fill, no state change; the matmuls skip it (slot -1) and the merge writes the host's result.
            // Every device decides the same, only the writer lists it.
            if (lane == 0) {
                slot_ids[i] = -1;
                if (hreq != nullptr) {
                    int32_t * en = hreq->ent + 3 * n_host_ent;
                    en[0] = i; en[1] = e; en[2] = exclusive ? hslot[e] : e;
                }
            }
            n_host_ent++;
            __syncwarp();
            continue;
        }
        int32_t bv = INT_MAX; int bs = n_slot;
        for (int s = lane; s < n_slot; s += WARP_SIZE) {
            const int32_t v = last_use[s];
            if (v < bv) { bv = v; bs = s; }
        }
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            const int32_t ov = __shfl_down_sync(0xffffffff, bv, off);
            const int     os = __shfl_down_sync(0xffffffff, bs, off);
            if (ov < bv || (ov == bv && os < bs)) { bv = ov; bs = os; }
        }
        const int best = __shfl_sync(0xffffffff, bs, 0);
        if (lane == 0) {
            const int32_t prev = slot_of[best];
            if (prev >= 0) owner[prev] = -1;
            // exclusive: the incoming expert's host slot receives the victim (the fill swaps the two)
            int32_t h = e;
            if (exclusive) {
                h = hslot[e];
                hslot[e] = -1;
                if (prev >= 0) { hslot[prev] = h; hexpert[h] = prev; }
                // disk refill: tell the host which slot the victim gets. Hybrid: when the host is behind (ring backlog >=
                // swap_backlog) this device writes the victim back itself (swap) instead of leaving it to the host
                if (ctrl != nullptr && prev >= 0) {
                    volatile uint32_t * lcount = ctrl->lcount + log_dev_index;  // this device's ring index (all layers)
                    const uint32_t idx = *lcount;
                    uint32_t tl = ctrl->tail;
                    while (idx - tl >= ctrl->cap) {         // ring full: wait for the refill thread
                        tl = ctrl->tail;
                    }
                    const int swap = (idx - tl) >= ctrl->swap_backlog;
                    ctrl->mode[(size_t) log_dev_index * ctrl->cap + idx % ctrl->cap] = swap;
                    if (log_dev) {
                        int32_t * en = ctrl->entries + 4 * (size_t) (idx % ctrl->cap);
                        en[0] = layer; en[1] = prev; en[2] = h; en[3] = t0;
                    }
                    __threadfence_system();
                    if (log_dev) {
                        ctrl->head = idx + 1;
                        __threadfence_system();
                    }
                    *lcount = idx + 1;
                    if (swap) {
                        h |= 1 << 30;                      // the fill swaps this entry
                    }
                }
                // victim stash: take a ring entry (none free -> the fill writes this victim back itself); host slot < 4096,
                // ring slot + 1 in bits 12..29
                if (stash_alloc && prev >= 0) {
                    volatile lru_stash_ctl * vs = stash;
                    const uint32_t k = vs->counter;
                    if (k - vs->flushed < vs->cap) {
                        vs->counter = k + 1;
                        vs->last[layer] = k + 1;
                        h |= (int32_t) ((k % vs->cap + 1) << 12);
                    }
                }
            }
            slot_of[best] = e;
            owner[e] = best;
            last_use[best] = t0;
            slot_ids[i] = best;
            fill_list[3 * n_fill + 0] = e;
            fill_list[3 * n_fill + 1] = best;
            fill_list[3 * n_fill + 2] = h;
        }
        n_fill++;
        __syncwarp();
    }
    if (hreq != nullptr) {
        // the request: activations, then the counts, then seq (the host reads nothing before it sees seq move)
        const int n4 = n_embd / 4;
        for (int t = 0; t < n_tokens; t++) {
            const float4 * src4 = (const float4 *) (hx + (size_t) t * hx_stride);
            float4 * dst4 = (float4 *) (hreq->x + (size_t) t * n_embd);
            for (int j = lane; j < n4; j += WARP_SIZE) {
                dst4[j] = src4[j];
            }
        }
        __syncwarp();
        if (lane == 0) {
            hreq->n_tok = n_tokens;
            hreq->n_ent = n_host_ent;
            __threadfence_system();
            const uint32_t sq = hreq->seq + 1;
            hreq->seq = sq;
            if (n_host_ent == 0) {
                hreq->done = sq;   // nothing for the host: the merge passes at once
            }
            __threadfence_system();
        }
    }
    if (lane == 0) {
        state[n_expert + 2 * n_slot] = t0 + 1;
        state[n_expert + 2 * n_slot + 1] += n_total;
        state[n_expert + 2 * n_slot + 2] += n_fill;
        fill_list[3 * n_slot] = n_fill;
        if (pass_fills != nullptr) {   // GGML_CUDA_PASS_TIMER: fills per pass
            atomicAdd(pass_fills, (unsigned long long) n_fill);
        }
        if (tail_stats) {   // GGML_CUDA_LRU_TAIL_STATS=1: cumulative skipped / filled entries (device 0, layer 0 calls)
            lru_tail_skipped += n_skip; lru_tail_filled += n_fill;
            if (log_dev && (++lru_tail_calls % 4096) == 0) {
                printf("LRU_TAIL calls=%llu skipped=%llu filled=%llu\n", lru_tail_calls, lru_tail_skipped, lru_tail_filled);
            }
        }
    }
}


// copy n_rows row slices of seg_bytes each (rstride apart in src, packed in dst) when the slices are not 16-byte aligned, e.g.
// the per-device half of a Q2_0 ffn_down row (640 values = 180 B -> 90 B per device). Byte-wide reads of pinned host memory
// become 32-byte PCIe requests (~1/4 of the bandwidth); instead every thread reads aligned 16-byte words covering the slices
// (words with no needed byte are skipped) and scatters the needed bytes into device memory. The words stay inside the source
// tensor's allocation: its base is 16-byte aligned and ggml pads tensor sizes.
static __device__ __forceinline__ void lru_copy_rows_unaligned(
        char * __restrict__ dst, const char * __restrict__ src, size_t n_rows, size_t seg, size_t rstride,
        size_t tid, size_t nthr) {
    if (rstride == seg) {                       // packed rows: one contiguous run
        seg = n_rows * seg; rstride = seg; n_rows = 1;
    }
    const uintptr_t a0   = (uintptr_t) src & ~(uintptr_t) 15;
    const int64_t   head = (int64_t) ((uintptr_t) src - a0);
    const int64_t   len  = (int64_t) ((n_rows - 1) * rstride + seg);
    const int64_t   n_w  = (head + len + 15) / 16;
    for (int64_t w = tid; w < n_w; w += nthr) {
        const int64_t p = 16 * w - head;        // offset of the word's first byte relative to src
        int64_t q0 = p < 0 ? 0 : p;
        int64_t row = q0 / (int64_t) rstride;
        int64_t col = q0 - row * (int64_t) rstride;
        if (col >= (int64_t) seg && col + (16 - (q0 - p)) <= (int64_t) rstride) {
            continue;                           // the word lies in the gap between two slices
        }
        const uint4 v = *(const uint4 *) (a0 + 16 * w);
        const uint8_t * b = (const uint8_t *) &v;
        for (int i = (int) (q0 - p); i < 16; i++) {
            const int64_t q = p + i;
            if (q >= len) break;
            if (col < (int64_t) seg) {
                dst[row * seg + col] = (char) b[i];
            }
            if (++col == (int64_t) rstride) { col = 0; row++; }
        }
    }
}

// disk refill: block-wide wait until host slot h of layer `layer` holds expert e (the refill thread sets content after the copy)
static __device__ __forceinline__ void lru_wait_content(const ggml_lru_ctrl * ctrl, int layer, int32_t h, int32_t e) {
    if (threadIdx.x == 0) {
        const volatile int32_t * c = ctrl->content + (size_t) layer * ctrl->n_host + h;
        while (*c != e) {
        }
        __threadfence_system();
    }
    __syncthreads();
}

// one fill descriptor: the expert slice of one pool. Row mode: n_rows = 1, seg_bytes = the slice. Column split (ne[0]): every
// local row is a seg_bytes slice at src_off of a source row (src_row_stride apart). vec: all strides/offsets 16-byte aligned.


// victim stash for one fill launch. decode: the fill list carries ring slots (bits 12..29 of the host slot); use: victims go to
// the ring (otherwise the fill swaps in place and marks the entry empty for the flush)
struct lru_stash_args {
    lru_stash_rec * rec;
    char * buf;
    size_t stride;
    uint32_t koff[3];
    int decode, use;
};

// blockIdx.z = descriptor, block rows walk the fill list (the grid is sized by the batch's ids, not by the slot count:
// most decode steps fill nothing, and 16K blocks that only read the count cost ~45 us per launch on sm_60)
static __global__ void k_lru_fill(const int32_t * __restrict__ fill_list, int n_slot, const lru_fill_descs descs,
                                  ggml_lru_ctrl * ctrl, int layer, int dev, unsigned int * counter, const lru_stash_args st) {
    const lru_fill_desc & d = descs.d[blockIdx.z];
    const int n_fill = fill_list[3 * n_slot];
    for (int j = blockIdx.y; j < n_fill; j += gridDim.y) {
        const int32_t s = fill_list[3 * j + 1];
        const int32_t hraw = fill_list[3 * j + 2];  // inclusive: the expert id (row of the model tensor)
        int32_t h = hraw & 0x3fffffff;
        int sl = 0;                                 // victim stash: ring slot + 1
        if (st.decode) {
            sl = (h >> 12) & 0x3ffff;
            h &= 0xfff;
        }
        const bool swap = d.swap || ((hraw >> 30) & 1);   // hybrid refill: this device writes the victim back itself
        if (ctrl != nullptr) {
            lru_wait_content(ctrl, layer, h, fill_list[3 * j + 0]);
        }
        char * src_e = d.src + (size_t) h * d.src_estride + d.src_off;
        char * dst_e = d.pool + (size_t) s * d.estride;
        char * wb_e  = nullptr;                     // victim stash: this slice's place in the ring entry
        if (sl != 0) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                lru_stash_rec & r = st.rec[sl - 1];
                r.dst[blockIdx.z]   = src_e;
                r.off[blockIdx.z]   = st.koff[blockIdx.z];
                r.bytes[blockIdx.z] = st.use ? (uint32_t) (d.n_rows * d.seg_bytes) : 0;
                if (blockIdx.z == 0) {
                    for (int z = gridDim.z; z < 3; ++z) {
                        r.bytes[z] = 0;
                    }
                }
            }
            if (st.use) {
                wb_e = st.buf + (size_t) (sl - 1) * st.stride + st.koff[blockIdx.z];
            }
        }
        if (d.vec) {
            const size_t seg16 = d.seg_bytes / 16, total = d.n_rows * seg16;
            for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
                const size_t row = i / seg16, c = i % seg16;
                uint4 * pd = (uint4 *) (dst_e + row * d.seg_bytes) + c;
                uint4 * ps = (uint4 *) (src_e + row * d.src_row_stride) + c;
                const uint4 incoming = *ps;
                if (wb_e != nullptr) {
                    ((uint4 *) (wb_e + row * d.seg_bytes))[c] = *pd;
                } else if (swap) {
                    *ps = *pd;
                }
                *pd = incoming;
            }
        } else if (swap) {
            // byte-wise swap of unaligned slices (decode moves only a few experts per token)
            const size_t rs = d.n_rows > 1 ? d.src_row_stride : d.seg_bytes, total = d.n_rows * d.seg_bytes;
            for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
                const size_t row = i / d.seg_bytes, b = i % d.seg_bytes;
                char * pd = dst_e + row * d.seg_bytes + b;
                char * ps = src_e + row * rs + b;
                const char incoming = *ps;
                *ps = *pd;
                *pd = incoming;
            }
        } else {
            lru_copy_rows_unaligned(dst_e, src_e, d.n_rows, d.seg_bytes, d.n_rows > 1 ? d.src_row_stride : d.seg_bytes,
                                    blockIdx.x * (size_t) blockDim.x + threadIdx.x, (size_t) gridDim.x * blockDim.x);
        }
    }
    // disk refill: the last block to finish tells the host that this device is done reading the step's host slots
    if (ctrl != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            const unsigned int total = gridDim.x * gridDim.y * gridDim.z;
            if (atomicAdd(counter, 1u) == total - 1) {
                *counter = 0;
                volatile uint32_t * dn = ctrl->done + (size_t) dev * ctrl->n_layer + layer;
                *dn = *dn + gridDim.z;
                __threadfence_system();
            }
        }
    }
}

// victim stash: snapshot of the completed entries at the end of a pass, latch of the next flush range, and the flush itself
// (each block takes whole slices; the last block publishes the new flushed count after a system fence)
static __global__ void k_lru_stash_snap(lru_stash_ctl * s, uint32_t * hsnap) {
    volatile lru_stash_ctl * v = s;
    v->snap = v->counter;
    if (hsnap != nullptr) {   // copy-engine mode: the host reads it once the pass has completed
        *(volatile uint32_t *) hsnap = v->snap;
        __threadfence_system();
    }
}

// copy-engine mode: after the write-back copies on this stream
static __global__ void k_lru_stash_set(lru_stash_ctl * s, uint32_t flushed) {
    __threadfence_system();
    ((volatile lru_stash_ctl *) s)->flushed = flushed;
}

static __global__ void k_lru_stash_latch(lru_stash_ctl * s) {
    volatile lru_stash_ctl * v = s;
    v->a = v->flushed;
    v->b = v->snap;
}

static __global__ void k_lru_stash_flush(lru_stash_ctl * s, const lru_stash_rec * rec, const char * __restrict__ buf, size_t stride) {
    const uint32_t a = s->a, b = s->b, cap = s->cap;
    const uint32_t n = b - a;
    for (uint32_t w = blockIdx.x; w < 3 * n; w += gridDim.x) {
        const uint32_t sl = (a + w / 3) % cap;
        const int z = w % 3;
        const volatile lru_stash_rec * r = rec + sl;
        const uint32_t bytes = r->bytes[z];
        if (bytes == 0) {
            continue;
        }
        const uint4 * src = (const uint4 *) (buf + (size_t) sl * stride + r->off[z]);
        uint4 * dst = (uint4 *) r->dst[z];
        for (uint32_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) {
            dst[i] = __ldcg(src + i);
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(&s->done, 1u) == gridDim.x - 1) {
        s->done = 0;
        __threadfence_system();
        ((volatile lru_stash_ctl *) s)->flushed = b;
    }
}

// prefill through the pools: one block row per expert of the assembled tensor. A pooled expert (owner[e] = slot) is copied
// from the pool in device memory, a missing one is read from the pinned host tensor in place. Under a device split the
// pool, dst and the src offsets are this device's slice (row split: src_off = first row; column split: every local row is a
// seg_bytes column slice at src_off of a source row). vec: 16-byte copies (all strides / offsets 16-byte aligned).
template <bool vec>
static __global__ void k_lru_assemble(
        const int32_t * __restrict__ owner, int n_slot,
        const char * __restrict__ src, const char * __restrict__ pool, char * __restrict__ dst,
        size_t src_estride, size_t estride, size_t seg_bytes, size_t n_rows, size_t src_row_stride, size_t src_off,
        const int32_t * __restrict__ hslot, int pooled_only, const ggml_lru_ctrl * ctrl, int layer) {
    const int e = blockIdx.y;
    const int32_t s = owner[e];
    char * dst_e = dst + (size_t) e * estride;
    const bool pooled = s >= 0 && s < n_slot;
    if (pooled_only && !pooled) {
        return;                                  // the copy engine brings this one
    }
    const size_t h = hslot ? (size_t) hslot[e] : (size_t) e;   // exclusive: the expert's host slot
    if (!pooled && ctrl != nullptr) {
        lru_wait_content(ctrl, layer, (int32_t) h, e);
    }
    const char * src_e = pooled ? pool + (size_t) s * estride : src + h * src_estride + src_off;
    const size_t rstride = pooled ? seg_bytes : src_row_stride;   // pool slices are packed
    if (vec) {
        const size_t seg16 = seg_bytes / 16, total = n_rows * seg16;
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
            const size_t row = i / seg16, c = i % seg16;
            ((uint4 *) (dst_e + row * seg_bytes))[c] = ((const uint4 *) (src_e + row * rstride))[c];
        }
    } else {
        lru_copy_rows_unaligned(dst_e, src_e, n_rows, seg_bytes, n_rows > 1 ? rstride : seg_bytes,
                                blockIdx.x * (size_t) blockDim.x + threadIdx.x, (size_t) gridDim.x * blockDim.x);
    }
}

// exclusive cache, large batch: the assembled tensor is [pool slots | host slots] (the ids are remapped to match), so entry j
// is pool slot j (j < n_slot) or host slot j - n_slot of this device's contiguous host region; every copy is one contiguous
// slice. Disk refill: a host slot is read once it holds the expert the state says it holds.
template <bool vec>
static __global__ void k_lru_assemble_sliced(
        const char * __restrict__ pool, const char * __restrict__ hbase, char * __restrict__ dst, size_t es, int n_slot, int j0,
        const int32_t * __restrict__ hexpert, const ggml_lru_ctrl * ctrl, int layer) {
    const int j = j0 + blockIdx.y;
    const char * src_e;
    if (j < n_slot) {
        src_e = pool + (size_t) j * es;
    } else {
        const int hs = j - n_slot;
        if (ctrl != nullptr) {
            lru_wait_content(ctrl, layer, hs, hexpert[hs]);
        }
        src_e = hbase + (size_t) hs * es;
    }
    char * dst_e = dst + (size_t) j * es;
    if (vec) {
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < es / 16; i += (size_t) gridDim.x * blockDim.x) {
            ((uint4 *) dst_e)[i] = ((const uint4 *) src_e)[i];
        }
    } else {
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < es; i += (size_t) gridDim.x * blockDim.x) {
            dst_e[i] = src_e[i];
        }
    }
}

// exclusive LRU: a stream about to read host slots waits until the pending victim write-back has landed
static void ggml_cuda_lru_wb_wait(ggml_backend_cuda_context & ctx, cudaStream_t stream) {
    if (ctx.lru_wb_pending) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, ctx.lru_wb_event, 0));
    }
}

// p100-series LRU victim stash (GGML_CUDA_LRU_STASH=<MiB> of VRAM per device, exclusive cache, decode fills inside meta passes).
// PTM2: a fill moves 755 KB in 166 us; read-only it would run at link speed, the in-kernel victim write-back adds ~48 us (+40 %),
// ~1.8 ms of a ~55 ms Flash-Next verify pass. The 10-13 staged write-back (GGML_CUDA_LRU_STAGED) lost on a main/side event round
// trip per layer. Here nothing on the main stream waits for a write-back: decides allocate ring entries and fills park victims
// in them; every GGML_CUDA_LRU_STASH_CHUNK fill calls (default 8) and at the end of the pass a latch + flush kernel pair on
// lru_wb_stream writes the completed entries back (GGML_CUDA_LRU_STASH_BLOCKS blocks, default 16). A decide spins only while its
// own layer's newest entry is unflushed (a fill would read a stale host slot); prefill readers wait for the last flush event.
// Off with GGML_CUDA_LRU_STAGED / GGML_CUDA_LRU_OVERLAP; A/B feature "stash".
static size_t ggml_cuda_lru_stash_bytes() {
    static const size_t b = [] {
        const char * e = getenv("GGML_CUDA_LRU_STASH");
        const char * st = getenv("GGML_CUDA_LRU_STAGED");
        const char * ov = getenv("GGML_CUDA_LRU_OVERLAP");
        if (e == nullptr || atoi(e) <= 0 || (st != nullptr && atoi(st) != 0) || (ov != nullptr && atoi(ov) != 0)) {
            return (size_t) 0;
        }
        return (size_t) atoi(e) << 20;
    }();
    return b;
}

// final: at the end of a pass, after k_lru_stash_snap (takes every entry); a mid-pass flush only takes the entries of the fills
// before the current layer (its decide's snapshot), so the pass stays dirty until the final flush
static void ggml_cuda_lru_stash_flush(ggml_backend_cuda_context & ctx, bool final) {
    static const int blocks = [] { const char * e = getenv("GGML_CUDA_LRU_STASH_BLOCKS"); return e && atoi(e) > 0 ? atoi(e) : 16; }();
    CUDA_CHECK(cudaEventRecord(ctx.lru_stash_main_event, ctx.stream()));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.lru_wb_stream, ctx.lru_stash_main_event, 0));
    k_lru_stash_latch<<<1, 1, 0, ctx.lru_wb_stream>>>(ctx.lru_stash_ctl);
    k_lru_stash_flush<<<blocks, 256, 0, ctx.lru_wb_stream>>>(ctx.lru_stash_ctl, ctx.lru_stash_rec, ctx.lru_stash, ctx.lru_stash_stride);
    CUDA_CHECK(cudaEventRecord(ctx.lru_stash_event, ctx.lru_wb_stream));
    if (final) {
        ctx.lru_stash_dirty = false;
    }
    ctx.lru_stash_flushed = true;
    ctx.lru_stash_calls   = 0;
}

// a stream about to read host slots outside the decode fills (assembly, DMA) waits for the last flush
static void ggml_cuda_lru_stash_worker_wait(ggml_backend_cuda_context & ctx);

static void ggml_cuda_lru_stash_wait(ggml_backend_cuda_context & ctx, cudaStream_t stream) {
    if (ctx.lru_stash_flushed) {
        ggml_cuda_lru_stash_worker_wait(ctx);   // copy-engine mode: the event is recorded once the copies are issued
        CUDA_CHECK(cudaStreamWaitEvent(stream, ctx.lru_stash_event, 0));
    }
}

// GGML_CUDA_LRU_STASH_CE=1 (PTS2: flush kernels slow whatever runs beside them almost 1:1 -> test DMA instead of SM stores):
// no flush kernels; at the end of a pass the snapshot also goes to mapped host memory, and at the next pass (the host has
// read the logits, so the pass has completed) the host reads the ring records and issues one D2H copy per slice on
// lru_wb_stream, then a 1-thread kernel publishes the flushed count
static bool ggml_cuda_lru_stash_ce() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_LRU_STASH_CE"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

// PTS4: issuing ~125 copies + publishes inside synchronize() cost ~0.9 ms of host time per cycle -> a worker thread per device
// issues them (GGML_CUDA_LRU_STASH_CE_THREAD=0: inline). Everything else that touches the ring's host side (readers of host
// slots, the next pass's inline flush, a resize, teardown) first waits until the worker is idle.
struct ggml_cuda_stash_worker {
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    bool job = false, quit = false;
    uint32_t b = 0;
};

static void ggml_cuda_lru_stash_flush_ce_issue(ggml_backend_cuda_context & ctx, uint32_t b);

static void ggml_cuda_lru_stash_worker_wait(ggml_backend_cuda_context & ctx) {
    auto * w = (ggml_cuda_stash_worker *) ctx.lru_stash_worker;
    if (w != nullptr) {
        std::unique_lock<std::mutex> lk(w->m);
        w->cv.wait(lk, [w] { return !w->job; });
    }
}

void ggml_cuda_lru_stash_worker_stop(ggml_backend_cuda_context & ctx) {
    auto * w = (ggml_cuda_stash_worker *) ctx.lru_stash_worker;
    if (w != nullptr) {
        {
            std::lock_guard<std::mutex> lk(w->m);
            w->quit = true;
        }
        w->cv.notify_all();
        w->th.join();
        delete w;
        ctx.lru_stash_worker = nullptr;
    }
}

// hand [hflushed, b) to the worker (started on first use); false = inline
bool ggml_cuda_lru_stash_worker_post(ggml_backend_cuda_context & ctx, uint32_t b) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_LRU_STASH_CE_THREAD"); return e == nullptr || atoi(e) != 0; }();
    if (!on) {
        return false;
    }
    auto * w = (ggml_cuda_stash_worker *) ctx.lru_stash_worker;
    if (w == nullptr) {
        w = new ggml_cuda_stash_worker;
        ctx.lru_stash_worker = w;
        ggml_backend_cuda_context * c = &ctx;
        w->th = std::thread([w, c] {
            ggml_cuda_set_device(c->device);
            std::unique_lock<std::mutex> lk(w->m);
            while (true) {
                w->cv.wait(lk, [w] { return w->job || w->quit; });
                if (!w->job) {
                    return;
                }
                const uint32_t jb = w->b;
                lk.unlock();
                ggml_cuda_lru_stash_flush_ce_issue(*c, jb);
                lk.lock();
                w->job = false;
                w->cv.notify_all();
            }
        });
    }
    {
        std::unique_lock<std::mutex> lk(w->m);
        w->cv.wait(lk, [w] { return !w->job; });
        w->b = b;
        w->job = true;
    }
    w->cv.notify_all();
    return true;
}

// inline flush (next pass, teardown): the pass must have completed
void ggml_cuda_lru_stash_flush_ce(ggml_backend_cuda_context & ctx) {
    ggml_cuda_lru_stash_worker_wait(ctx);
    if (cudaEventQuery(ctx.lru_stash_ce_event) != cudaSuccess) {
        CUDA_CHECK(cudaEventSynchronize(ctx.lru_stash_ce_event));   // no sync since the pass (not the decode loop): wait for it
    }
    ggml_cuda_lru_stash_flush_ce_issue(ctx, *(volatile uint32_t *) ctx.lru_stash_hsnap);
    ctx.lru_stash_flushed    = true;
    ctx.lru_stash_ce_pending = false;
}

static void ggml_cuda_lru_stash_flush_ce_issue(ggml_backend_cuda_context & ctx, uint32_t b) {
    // PTS3: one flushed count after the whole flush made layer 0 of the next pass wait for all of it -> publish every
    // GGML_CUDA_LRU_STASH_CE_STEP entries (default 4; entries are in layer order). GGML_CUDA_LRU_STASH_STATS=1: flush timing
    static const uint32_t step = [] { const char * e = getenv("GGML_CUDA_LRU_STASH_CE_STEP"); return e && atoi(e) > 0 ? (uint32_t) atoi(e) : 4u; }();
    static const bool stats = [] { const char * e = getenv("GGML_CUDA_LRU_STASH_STATS"); return e != nullptr && atoi(e) != 0; }();
    struct ce_stats { cudaEvent_t e0 = nullptr, e1 = nullptr; bool live = false; double ms = 0, bytes = 0, n = 0, entries = 0; };
    static thread_local ce_stats cs[GGML_CUDA_MAX_DEVICES];
    ce_stats & S = cs[ctx.device];
    if (stats) {
        if (S.e0 == nullptr) {
            CUDA_CHECK(cudaEventCreate(&S.e0));
            CUDA_CHECK(cudaEventCreate(&S.e1));
        }
        if (S.live && cudaEventQuery(S.e1) == cudaSuccess) {   // the previous flush, if done (never blocks)
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, S.e0, S.e1));
            S.ms += ms; S.n += 1;
            if ((int) S.n % 200 == 0) {
                GGML_LOG_INFO("%s: device %d: %.0f flushes, mean %.2f ms, %.1f entries, %.2f MB, %.2f GB/s\n", __func__, ctx.device, S.n,
                              S.ms / S.n, S.entries / S.n, S.bytes / S.n / 1e6, S.bytes / (S.ms * 1e6));
            }
            S.live = false;
        }
    }
    const volatile lru_stash_rec * rec = ctx.lru_stash_rec;
    const bool time_it = stats && !S.live && b != ctx.lru_stash_hflushed;
    if (time_it) {
        CUDA_CHECK(cudaEventRecord(S.e0, ctx.lru_wb_stream));
    }
    double bytes_all = 0;
    for (uint32_t k = ctx.lru_stash_hflushed; k != b; ++k) {
        const uint32_t sl = k % ctx.lru_stash_cap;
        for (int z = 0; z < 3; ++z) {
            const uint32_t bytes = rec[sl].bytes[z];
            if (bytes != 0) {
                CUDA_CHECK(cudaMemcpyAsync((void *) rec[sl].dst[z], ctx.lru_stash + (size_t) sl * ctx.lru_stash_stride + rec[sl].off[z],
                                           bytes, cudaMemcpyDeviceToHost, ctx.lru_wb_stream));
                bytes_all += bytes;
            }
        }
        if (k + 1 != b && (k + 1 - ctx.lru_stash_hflushed) % step == 0) {
            k_lru_stash_set<<<1, 1, 0, ctx.lru_wb_stream>>>(ctx.lru_stash_ctl, k + 1);
        }
    }
    k_lru_stash_set<<<1, 1, 0, ctx.lru_wb_stream>>>(ctx.lru_stash_ctl, b);
    if (time_it) {
        CUDA_CHECK(cudaEventRecord(S.e1, ctx.lru_wb_stream));
        S.live = true; S.bytes += bytes_all; S.entries += b - ctx.lru_stash_hflushed;
    }
    CUDA_CHECK(cudaEventRecord(ctx.lru_stash_event, ctx.lru_wb_stream));
    ctx.lru_stash_hflushed = b;
}

// meta backend pass hook (threaded dispatch, on the device's thread): k = 0 before the device issues the pass, 1 after
void ggml_cuda_lru_stash_pass(ggml_backend_cuda_context & ctx, int k) {
    const size_t bytes = ggml_cuda_lru_stash_bytes();
    if (bytes == 0) {
        return;
    }
    if (k == 1) {
        if (ctx.lru_stash_dirty && ctx.lru_stash_hsnap != nullptr) {
            k_lru_stash_snap<<<1, 1, 0, ctx.stream()>>>(ctx.lru_stash_ctl, ctx.lru_stash_hsnap);
            CUDA_CHECK(cudaEventRecord(ctx.lru_stash_ce_event, ctx.stream()));
            ctx.lru_stash_ce_pending = true;
            ctx.lru_stash_dirty      = false;
        } else if (ctx.lru_stash_dirty) {
            k_lru_stash_snap<<<1, 1, 0, ctx.stream()>>>(ctx.lru_stash_ctl, nullptr);
            ggml_cuda_lru_stash_flush(ctx, true);
        }
        ctx.lru_stash_armed = false;
        return;
    }
    if (ctx.lru_stash_ctl == nullptr) {
        ggml_cuda_set_device(ctx.device);
        if (ctx.lru_wb_stream == nullptr) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.lru_wb_stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_wb_event, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_fill_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_stash_main_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_stash_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaMalloc(&ctx.lru_stash_ctl, sizeof(lru_stash_ctl)));
        CUDA_CHECK(cudaMemset(ctx.lru_stash_ctl, 0, sizeof(lru_stash_ctl)));   // cap 0: no entries until a fill size is known
        if (ggml_cuda_lru_stash_ce()) {
            CUDA_CHECK(cudaHostAlloc((void **) &ctx.lru_stash_hsnap, sizeof(uint32_t), cudaHostAllocMapped | cudaHostAllocPortable));
            *ctx.lru_stash_hsnap = 0;
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_stash_ce_event, cudaEventDisableTiming));
        }
    }
    if (ctx.lru_stash_ce_pending) {
        ggml_cuda_lru_stash_flush_ce(ctx);   // the previous pass's victims (before any resize below)
    }
    if (ctx.lru_stash_need > ctx.lru_stash_stride) {
        // (re)size the ring for the largest fill seen (the first passes visit every layer): everything in flight must be done
        ggml_cuda_lru_stash_worker_wait(ctx);
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.lru_wb_stream));
        if (ctx.lru_stash_rec != nullptr) {
            if (ctx.lru_stash_hsnap != nullptr) {
                CUDA_CHECK(cudaFreeHost(ctx.lru_stash_rec));
            } else {
                CUDA_CHECK(cudaFree(ctx.lru_stash_rec));
            }
            CUDA_CHECK(cudaFree(ctx.lru_stash));
            ctx.lru_stash_rec = nullptr;
            ctx.lru_stash = nullptr;
        }
        const size_t stride = (ctx.lru_stash_need + 255) & ~(size_t) 255;
        const uint32_t cap = (uint32_t) std::min<size_t>(bytes / stride, (1u << 18) - 2);
        if (cap > 0) {
            CUDA_CHECK(cudaMalloc(&ctx.lru_stash, (size_t) cap * stride));
            if (ctx.lru_stash_hsnap != nullptr) {   // copy-engine mode: the host reads the records
                CUDA_CHECK(cudaHostAlloc((void **) &ctx.lru_stash_rec, (size_t) cap * sizeof(lru_stash_rec), cudaHostAllocMapped | cudaHostAllocPortable));
                memset((void *) ctx.lru_stash_rec, 0, (size_t) cap * sizeof(lru_stash_rec));
            } else {
                CUDA_CHECK(cudaMalloc(&ctx.lru_stash_rec, (size_t) cap * sizeof(lru_stash_rec)));
                CUDA_CHECK(cudaMemset(ctx.lru_stash_rec, 0, (size_t) cap * sizeof(lru_stash_rec)));
            }
        }
        ctx.lru_stash_stride = stride;
        ctx.lru_stash_cap = cap;
        CUDA_CHECK(cudaMemcpy((char *) ctx.lru_stash_ctl + offsetof(lru_stash_ctl, cap), &cap, sizeof(cap), cudaMemcpyHostToDevice));
        GGML_LOG_INFO("%s: device %d: LRU victim stash %u entries x %zu KiB\n", __func__, ctx.device, cap, stride >> 10);
    }
    static const bool stash_ab = ggml_cuda_ab_listed("stash");
    ctx.lru_stash_armed = ctx.lru_stash_cap > 0 && !(stash_ab && ggml_cuda_ab_arm() == 1);
    ctx.lru_stash_calls = 0;
}

void ggml_cuda_lru_assemble(ggml_backend_cuda_context & ctx, ggml_tensor * dst, cudaStream_t stream, int gx, int pooled_only) {
    if (stream == nullptr) {
        stream = ctx.stream();
    }
    if (!pooled_only) {
        ggml_cuda_lru_wb_wait(ctx, stream);
        ggml_cuda_lru_stash_wait(ctx, stream);
    }
    const ggml_tensor * state = dst->src[0];
    const ggml_tensor * pool  = dst->src[1];

    ggml_tensor * src;
    memcpy(&src, dst->op_params, sizeof(src));
    GGML_ASSERT(ggml_get_op_params_i32(dst, 5) == 0 && "tail assembly (ggml_lru_assemble_tail) needs GGML_CUDA_LRU_DMA=1");

    {   // exclusive: [pool slots | host slots] from the per-device host layout
        const int n_expert_x = (int) dst->ne[2], n_slot_x = (int) pool->ne[2];
        if (ggml_nelements(state) >= 2 * (int64_t) n_expert_x + 2 * (int64_t) n_slot_x + 3) {
            ggml_lru_ctrl * xctrl;
            memcpy(&xctrl, (const char *) dst->op_params + 8, sizeof(xctrl));
            const int xlayer = ggml_get_op_params_i32(dst, 4);
            const size_t es = pool->nb[2];
            int dev = 0;
            if (src->nb[2] != es) {
                cudaGetDevice(&dev);
            }
            const char * hbase = (const char *) src->data + (size_t) dev * (size_t) src->ne[2] * es;
            const int32_t * hexpert = (const int32_t *) state->data + 2 * n_expert_x + 2 * n_slot_x + 4;
            const int j0 = 0, nj = pooled_only ? n_slot_x : n_expert_x;
            const dim3 grid(gx, nj), block(256);
            if (es % 16 == 0) {
                k_lru_assemble_sliced<true><<<grid, block, 0, stream>>>((const char *) pool->data, hbase, (char *) dst->data, es,
                    n_slot_x, j0, hexpert, xctrl, xlayer);
            } else {
                k_lru_assemble_sliced<false><<<grid, block, 0, stream>>>((const char *) pool->data, hbase, (char *) dst->data, es,
                    n_slot_x, j0, hexpert, xctrl, xlayer);
            }
            return;
        }
    }

    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(pool));
    GGML_ASSERT(dst->nb[1] == pool->nb[1] && dst->nb[2] == pool->nb[2] && dst->ne[1] == pool->ne[1]);
    const int n_expert = (int) dst->ne[2];
    const int n_slot   = (int) pool->ne[2];
    const bool exclusive = ggml_nelements(state) >= 2 * (int64_t) n_expert + 2 * (int64_t) n_slot + 3;
    const int32_t * hslot = exclusive ? (const int32_t *) state->data + n_expert + 2 * n_slot + 3 : nullptr;
    ggml_lru_ctrl * ctrl;
    memcpy(&ctrl, (const char *) dst->op_params + 8, sizeof(ctrl));
    const int layer = ggml_get_op_params_i32(dst, 4);

    // same device-index rule as ggml_cuda_lru_fill: device i holds slice i of an even split
    int dev = 0;
    size_t seg_bytes, n_rows, src_row_stride, src_off;
    if (pool->nb[1] != src->nb[1]) {          // column split (ne[0]): each local row is a slice of a source row
        cudaGetDevice(&dev);
        seg_bytes = pool->nb[1]; n_rows = pool->ne[1]; src_row_stride = src->nb[1]; src_off = (size_t) dev * pool->nb[1];
    } else {                                  // whole rows: one contiguous run per expert
        if (pool->ne[1] != src->ne[1]) {
            cudaGetDevice(&dev);
        }
        seg_bytes = pool->nb[2]; n_rows = 1; src_row_stride = 0; src_off = (size_t) dev * pool->ne[1] * pool->nb[1];
    }
    const bool vec = seg_bytes % 16 == 0 && src_row_stride % 16 == 0 && src_off % 16 == 0 && src->nb[2] % 16 == 0 &&
                     pool->nb[2] % 16 == 0 && (uintptr_t) src->data % 16 == 0;
    const dim3 grid(gx, n_expert), block(256);
    if (vec) {
        k_lru_assemble<true><<<grid, block, 0, stream>>>((const int32_t *) state->data, n_slot,
            (const char *) src->data, (const char *) pool->data, (char *) dst->data,
            src->nb[2], dst->nb[2], seg_bytes, n_rows, src_row_stride, src_off, hslot, pooled_only, ctrl, layer);
    } else {
        k_lru_assemble<false><<<grid, block, 0, stream>>>((const int32_t *) state->data, n_slot,
            (const char *) src->data, (const char *) pool->data, (char *) dst->data,
            src->nb[2], dst->nb[2], seg_bytes, n_rows, src_row_stride, src_off, hslot, pooled_only, ctrl, layer);
    }
}

// p100-series: copy-engine assembly (GGML_CUDA_LRU_DMA=1). The zero-copy assembly kernel reads host memory on the SMs, so it
// cannot overlap the matmuls (FZD); here the experts a prefill misses are moved by DMA (cudaMemcpy2DAsync per run of
// consecutive experts) on copy_stream, and only the pooled experts are gathered by the kernel on the main stream.
bool ggml_cuda_lru_dma_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_LRU_DMA"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

static void ggml_cuda_lru_mirror_mark_dirty(ggml_backend_cuda_context & ctx, const ggml_tensor * state) {
    auto * m = (ggml_cuda_lru_mirror *) ctx.lru_mirror;
    if (m != nullptr) {
        if (auto * e = m->find(state)) {
            e->dirty = true;
        }
    }
}

// the host copy of a state: missing or dirty -> one sync of the main stream, then every dirty mirror is re-read
static const std::vector<int32_t> & ggml_cuda_lru_mirror_get(ggml_backend_cuda_context & ctx, const ggml_tensor * state) {
    if (ctx.lru_mirror == nullptr) {
        ctx.lru_mirror = new ggml_cuda_lru_mirror;
    }
    auto * m = (ggml_cuda_lru_mirror *) ctx.lru_mirror;
    auto * me = m->find(state);
    if (me == nullptr) {
        m->entries.push_back({ state, std::vector<int32_t>(ggml_nelements(state)), true });
        me = &m->entries.back();
    }
    if (me->dirty) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        for (auto & e : m->entries) {
            if (e.dirty) {
                CUDA_CHECK(cudaMemcpy(e.h.data(), e.state->data, ggml_nbytes(e.state), cudaMemcpyDeviceToHost));
                e.dirty = false;
            }
        }
    }
    return me->h;
}

void ggml_cuda_lru_assemble_dma(ggml_backend_cuda_context & ctx, ggml_tensor * dst, cudaStream_t copy_stream) {
    const ggml_tensor * state = dst->src[0];
    const ggml_tensor * pool  = dst->src[1];
    ggml_tensor * src;
    memcpy(&src, dst->op_params, sizeof(src));

    if (ggml_get_op_params_i32(dst, 6) != 0) {
        // zero-copy tail: the matmul reads the host pool in place; only the host slots must be current (write-backs done)
        ggml_cuda_lru_wb_wait(ctx, copy_stream);
        ggml_cuda_lru_stash_wait(ctx, copy_stream);
        return;
    }
    const bool tail    = ggml_get_op_params_i32(dst, 5) != 0;   // host part only (ggml_lru_assemble_tail)
    const int n_slot   = (int) pool->ne[2];
    const int n_expert = (int) dst->ne[2] + (tail ? n_slot : 0);
    const bool exclusive = ggml_nelements(state) >= 2 * (int64_t) n_expert + 2 * (int64_t) n_slot + 3;
    GGML_ASSERT(!tail || exclusive);

    int dev = 0;
    size_t seg_bytes, n_rows, src_row_stride, src_off;
    if (pool->nb[1] != src->nb[1]) {
        cudaGetDevice(&dev);
        seg_bytes = pool->nb[1]; n_rows = pool->ne[1]; src_row_stride = src->nb[1]; src_off = (size_t) dev * pool->nb[1];
        GGML_ASSERT(src->nb[2] == (size_t) src->ne[1] * src->nb[1]);   // rows continue across experts
    } else {
        if (pool->ne[1] != src->ne[1]) {
            cudaGetDevice(&dev);
        }
        seg_bytes = pool->nb[2]; n_rows = 1; src_row_stride = 0; src_off = (size_t) dev * pool->ne[1] * pool->nb[1];
    }

    const std::vector<int32_t> & m = ggml_cuda_lru_mirror_get(ctx, state);
    ggml_cuda_lru_wb_wait(ctx, copy_stream);
    ggml_cuda_lru_stash_wait(ctx, copy_stream);
    ggml_lru_ctrl * dctrl;
    memcpy(&dctrl, (const char *) dst->op_params + 8, sizeof(dctrl));
    const int dlayer = ggml_get_op_params_i32(dst, 4);

    if (exclusive) {
        // [pool slots | host slots]: the host part is this device's contiguous host region -> ONE copy-engine transfer; the
        // pool part one device-to-device copy (DS: the scattered path issued ~127 copies per assembly at ~250 us each)
        const size_t es = pool->nb[2];
        const int64_t n_host = src->ne[2];
        int xdev = 0;
        if (src->nb[2] != es) {
            cudaGetDevice(&xdev);
        }
        if (dctrl != nullptr) {   // disk refill: every host slot must hold the expert the state says it holds
            const int32_t * hexp = m.data() + 2 * n_expert + 2 * n_slot + 4;
            for (int64_t hs = 0; hs < n_host; hs++) {
                const volatile int32_t * c = dctrl->content + (size_t) dlayer * dctrl->n_host + hs;
                while (*c != hexp[hs]) {
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + (tail ? 0 : (size_t) n_slot * es),
            (const char *) src->data + (size_t) xdev * (size_t) n_host * es, (size_t) n_host * es, cudaMemcpyHostToDevice, copy_stream));
        if (!tail) {   // tail: the matmul reads the pooled experts from the pool itself
            CUDA_CHECK(cudaMemcpyAsync(dst->data, pool->data, (size_t) n_slot * es, cudaMemcpyDeviceToDevice, ctx.stream()));
        }
        return;
    }
    const int32_t * owner = m.data();
    const int32_t * hslot = exclusive ? m.data() + n_expert + 2 * n_slot + 3 : nullptr;
    auto pooled = [&](int e) { return owner[e] >= 0 && owner[e] < n_slot; };
    auto hidx   = [&](int e) { return hslot ? (int64_t) hslot[e] : (int64_t) e; };

    // GGML_CUDA_LRU_DMA_STATS=1: host time spent issuing the copies (submission cost on the dispatching thread)
    static const bool dma_stats = [] { const char * e = getenv("GGML_CUDA_LRU_DMA_STATS"); return e != nullptr && atoi(e) != 0; }();
    static std::atomic<int64_t> st_ns{0}, st_calls{0}, st_copies{0}, st_bytes{0};
    const auto st_t0 = std::chrono::steady_clock::now();
    int64_t n_copies = 0, n_bytes = 0;
    for (int e = 0; e < n_expert; ) {
        if (pooled(e)) {
            e++;
            continue;
        }
        const int64_t h0 = hidx(e);
        int k = 1;
        while (e + k < n_expert && !pooled(e + k) && hidx(e + k) == h0 + k) {
            k++;
        }
        if (dctrl != nullptr) {   // disk refill: every slot of the run must hold its expert
            for (int q = 0; q < k; q++) {
                const volatile int32_t * c = dctrl->content + (size_t) dlayer * dctrl->n_host + (h0 + q);
                while (*c != e + q) {
                }
            }
        }
        const char * s0 = (const char *) src->data + h0 * src->nb[2] + src_off;
        char * d0 = (char *) dst->data + (size_t) e * dst->nb[2];
        if (n_rows == 1) {
            CUDA_CHECK(cudaMemcpy2DAsync(d0, dst->nb[2], s0, src->nb[2], seg_bytes, k, cudaMemcpyHostToDevice, copy_stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(d0, seg_bytes, s0, src_row_stride, seg_bytes, (size_t) k * n_rows,
                                         cudaMemcpyHostToDevice, copy_stream));
        }
        e += k;
        n_copies++;
        n_bytes += (int64_t) k * (n_rows == 1 ? (int64_t) seg_bytes : (int64_t) (seg_bytes * n_rows));
    }
    if (dma_stats) {
        const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - st_t0).count();
        st_ns += ns; st_copies += n_copies; st_bytes += n_bytes;
        const int64_t c = ++st_calls;
        if (c % 2000 == 0) {
            fprintf(stderr, "lru dma stats: %lld assemblies, %lld copies (%.1f per assembly), %.2f GB, issue time %.2f s (%.1f us per copy)\n",
                (long long) c, (long long) st_copies.load(), (double) st_copies.load() / c, st_bytes.load() / 1e9, st_ns.load() / 1e9,
                st_ns.load() / 1e3 / std::max<int64_t>(1, st_copies.load()));
        }
    }
    ggml_cuda_lru_assemble(ctx, dst, ctx.stream(), 32, /*pooled_only =*/ 1);
}


// p100-series LRU prefetch (LLAMA_MOE_LRU_PF): a decide marked tail < 0 and its fills run on a per-device side stream, so the
// next layer's predicted experts load while this layer computes. The real decide of that layer waits for the side stream's
// event first (state and pool are touched by one stream at a time), and the graph keeps the prefetch's buffers alive until then.
struct ggml_cuda_lru_pf_state {
    cudaStream_t stream = nullptr;
    cudaEvent_t  ev_main = nullptr;
    std::vector<cudaEvent_t> ev_layer;
    std::vector<uint8_t>     pending;
};
static ggml_cuda_lru_pf_state g_lru_pf[GGML_CUDA_MAX_DEVICES];

static ggml_cuda_lru_pf_state & ggml_cuda_lru_pf(int device, int layer) {
    ggml_cuda_lru_pf_state & p = g_lru_pf[device];
    if (p.stream == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&p.ev_main, cudaEventDisableTiming));
    }
    while ((int) p.ev_layer.size() <= layer) {
        cudaEvent_t e;
        CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        p.ev_layer.push_back(e);
        p.pending.push_back(0);
    }
    return p;
}

static bool ggml_cuda_lru_is_prefetch(const ggml_tensor * dec) {
    return dec != nullptr && dec->op == GGML_OP_LRU_DECIDE && ggml_get_op_params_i32(dec, 5) == 0 && ggml_get_op_params_i32(dec, 6) < 0;
}

bool ggml_cuda_lru_ce_enabled();

void ggml_cuda_lru_decide(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids  = dst->src[0];
    ggml_tensor * state      = dst->src[1];

    const int n_expert = ggml_get_op_params_i32(dst, 0);
    const int n_slot   = ggml_get_op_params_i32(dst, 1);

    GGML_ASSERT(ids->nb[0] == ggml_type_size(ids->type));
    const int ids_stride = (int) (ids->nb[1] / ggml_type_size(ids->type));

    const int n_used   = (int) ids->ne[0];
    const int n_tokens = (int) ids->ne[1];
    int32_t * fill_list = (int32_t *) dst->data + (size_t) n_used * n_tokens;
    const int exclusive = ggml_nelements(state) >= 2 * (int64_t) n_expert + 2 * (int64_t) n_slot + 3;
    const int remap = ggml_get_op_params_i32(dst, 5);
    if (!remap) {
        ggml_cuda_lru_mirror_mark_dirty(ctx, state);
    }
    ggml_lru_ctrl * ctrl;
    memcpy(&ctrl, (const char *) dst->op_params + 8, sizeof(ctrl));
    const int layer = ggml_get_op_params_i32(dst, 4);
    const bool prefetch = !remap && ggml_get_op_params_i32(dst, 6) < 0;
    const int tail  = remap || prefetch ? 0 : ggml_get_op_params_i32(dst, 6);
    int32_t * keep  = tail > 0 ? fill_list + 3 * (size_t) n_slot + 1 : nullptr;
    static const int tail_stats = [] { const char * e = getenv("GGML_CUDA_LRU_TAIL_STATS"); return e && atoi(e) != 0; }();
    int dev = 0;
    cudaGetDevice(&dev);
    // victim stash: every decide of an exclusive cache keeps the ring's hazard rule once a ring exists; allocation only in a pass
    lru_stash_ctl * stash = exclusive && !remap && ctrl == nullptr && n_expert <= 4096 ? ctx.lru_stash_ctl : nullptr;
    const int stash_alloc = stash != nullptr && ctx.lru_stash_armed && layer < 1024;
    if (stash_alloc) {
        ctx.lru_stash_dirty = true;
    }

    // host-computed misses (ggml_lru_decide_host)
    const int host_max = remap ? -1 : ggml_get_op_params_i32(dst, 8) - 1;
    const int host_every = ggml_get_op_params_i32(dst, 13);
    ggml_lru_host_req * hreq = nullptr;
    const float * hx = nullptr;
    int hx_stride = 0, n_embd = 0;
    if (host_max >= 0) {
        memcpy(&hreq, (const char *) dst->op_params + 40, sizeof(hreq));
        const int writer = ggml_get_op_params_i32(dst, 12);
        if (writer == 1 && dev != 0) {
            hreq = nullptr;   // tensor split: device 0 hands the misses to the host, the others only skip them
        }
        const ggml_tensor * x = dst->src[2];
        GGML_ASSERT(x != nullptr && x->ne[0] % 4 == 0 && x->nb[1] % 16 == 0 && (uintptr_t) x->data % 16 == 0);
        GGML_ASSERT(hreq == nullptr || (x->ne[0] == hreq->n_embd && n_tokens <= GGML_LRU_HOST_MAX_TOK &&
                                        n_used * n_tokens <= hreq->max_ent));
        hx = (const float *) x->data;
        hx_stride = (int) (x->nb[1] / sizeof(float));
        n_embd = (int) x->ne[0];
    }

    if (prefetch && !ggml_cuda_lru_ce_enabled()) {
        // after the predicted ids exist on the main stream; no stash, no host misses, no disk refill. The victim stash parks
        // write-backs in a VRAM ring that a prefetch fill would not wait for (it could read a stale host slot): not together.
        GGML_ASSERT(ctx.lru_stash_ctl == nullptr && ctrl == nullptr && "LLAMA_MOE_LRU_PF needs GGML_CUDA_LRU_STASH and disk refill off");
        ggml_cuda_lru_pf_state & pf = ggml_cuda_lru_pf(ctx.device, layer);
        CUDA_CHECK(cudaEventRecord(pf.ev_main, ctx.stream()));
        CUDA_CHECK(cudaStreamWaitEvent(pf.stream, pf.ev_main, 0));
        k_lru_decide<<<1, WARP_SIZE, 0, pf.stream>>>(
            (const int32_t *) ids->data, (int32_t *) dst->data, (int32_t *) state->data, fill_list,
            n_used, n_tokens, n_expert, n_slot, ids_stride, exclusive, nullptr, layer, dev == 0, dev, 0,
            0, nullptr, 0, nullptr, nullptr, 0, -1, 0, nullptr, nullptr, 0, 0);
        pf.pending[layer] = 1;
        return;
    }
    if (prefetch) {
        GGML_ASSERT(ctx.lru_stash_ctl == nullptr && ctrl == nullptr && "LLAMA_MOE_LRU_PF needs GGML_CUDA_LRU_STASH and disk refill off");
        k_lru_decide<<<1, WARP_SIZE, 0, ctx.stream()>>>(
            (const int32_t *) ids->data, (int32_t *) dst->data, (int32_t *) state->data, fill_list,
            n_used, n_tokens, n_expert, n_slot, ids_stride, exclusive, nullptr, layer, dev == 0, dev, 0,
            0, nullptr, 0, nullptr, nullptr, 0, -1, 0, nullptr, nullptr, 0, 0);
        return;
    }
    if (!remap && layer < (int) g_lru_pf[ctx.device].pending.size() && g_lru_pf[ctx.device].pending[layer]) {
        ggml_cuda_lru_pf_state & pf = g_lru_pf[ctx.device];
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), pf.ev_layer[layer], 0));
        pf.pending[layer] = 0;
    }
    k_lru_decide<<<1, WARP_SIZE, 0, ctx.stream()>>>(
        (const int32_t *) ids->data, (int32_t *) dst->data,
        (int32_t *) state->data, fill_list,
        n_used, n_tokens, n_expert, n_slot, ids_stride, exclusive, remap ? nullptr : ctrl, layer, dev == 0, dev, remap,
        tail, keep, tail > 0 && tail_stats, ggml_cuda_pass_fill_counter(dev), stash, stash_alloc,
        host_max, host_every, hreq, hx, hx_stride, n_embd);
}

// host-computed misses: wait until the host has finished the request, then write each listed expert output into its entry of
// the expert outputs [n_embd, n_used, n_tokens] (zeros there: the matmuls skipped slot -1). The host's results are read from
// device-readable host memory with volatile loads (written by the CPU after the request was published).
static __global__ void k_lru_host_merge(ggml_lru_host_req * req, float * dst, int n_embd, size_t nb1, size_t nb2, int n_used) {
    __shared__ int n_ent;
    if (threadIdx.x == 0) {
        volatile ggml_lru_host_req * v = req;
        const uint32_t sq = v->seq;
        while (v->done != sq) {
        }
        __threadfence_system();
        n_ent = v->n_ent;
    }
    __syncthreads();
    for (int j = blockIdx.x; j < n_ent; j += gridDim.x) {
        const int i = ((volatile int32_t *) req->ent)[3 * j];
        float * d = (float *) ((char *) dst + (size_t) (i / n_used) * nb2 + (size_t) (i % n_used) * nb1);
        const volatile float * o = req->out + (size_t) j * n_embd;
        for (int c = threadIdx.x; c < n_embd; c += blockDim.x) {
            d[c] = o[c];
        }
    }
}

void ggml_cuda_lru_host_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_lru_host_req * req;
    memcpy(&req, dst->op_params, sizeof(req));
    const int writer = ggml_get_op_params_i32(dst, 2);
    int dev = 0;
    cudaGetDevice(&dev);
    if (writer == 1 && dev != 0) {
        return;   // tensor split: the host's results enter the partial sums on device 0 only
    }
    GGML_ASSERT(dst->data == dst->src[0]->data && dst->nb[0] == sizeof(float));
    k_lru_host_merge<<<8, 256, 0, ctx.stream()>>>(req, (float *) dst->data, (int) dst->ne[0], dst->nb[1], dst->nb[2], (int) dst->ne[1]);
}

lru_fill_desc ggml_cuda_lru_fill_desc(const ggml_tensor * dst) {
    const ggml_tensor * pool = dst->src[1];
    ggml_tensor * src;
    memcpy(&src, dst->op_params, sizeof(src));
    GGML_ASSERT(pool->nb[2] % 16 == 0);
    GGML_ASSERT(src->nb[2] % 16 == 0);

    // under a device split this dispatch runs once per device with a pool view of that device's slice, so the source offset
    // needs the device index (device i holds slice i of an even split)
    lru_fill_desc d{};
    d.src = (char *) src->data; d.pool = (char *) pool->data;
    d.swap = ggml_get_op_params_i32(dst, 3);
    d.src_estride = src->nb[2]; d.estride = pool->nb[2];
    int dev = 0;
    if (ggml_get_op_params_i32(dst, 7)) {
        // exclusive: the host pool holds each device's slices contiguously ([device][host slot][slice]), slice = this
        // device's pool slot bytes; the number of devices = full expert bytes / slice
        const size_t es = pool->nb[2];
        if (src->nb[2] != es) {
            cudaGetDevice(&dev);
        }
        d.src_estride = es; d.seg_bytes = es; d.n_rows = 1; d.src_row_stride = 0;
        d.src_off = (size_t) dev * (size_t) src->ne[2] * es;
        d.vec = es % 16 == 0 && (uintptr_t) src->data % 16 == 0;
        return d;
    }
    if (pool->nb[1] != src->nb[1]) {          // columnar: local rows are column slices of the source rows (ne[0] split)
        cudaGetDevice(&dev);
        d.seg_bytes = pool->nb[1]; d.n_rows = pool->ne[1]; d.src_row_stride = src->nb[1]; d.src_off = (size_t) dev * pool->nb[1];
    } else {                                  // whole rows: this device holds a slice of the row set (or the whole pool)
        if (pool->ne[1] != src->ne[1]) {
            cudaGetDevice(&dev);
        }
        d.seg_bytes = pool->nb[2]; d.n_rows = 1; d.src_row_stride = 0; d.src_off = (size_t) dev * pool->ne[1] * pool->nb[1];
    }
    d.vec = d.seg_bytes % 16 == 0 && d.src_row_stride % 16 == 0 && d.src_off % 16 == 0 && (uintptr_t) src->data % 16 == 0;
    return d;
}

// exclusive LRU, staged write-back: save each fill's victim (pool slot, this device's slice) to staging[k][j] (device memory),
// before the fill overwrites the slot ...
static __global__ void k_lru_save_victims(const int32_t * __restrict__ fill_list, int n_slot, const lru_fill_descs descs,
                                          char * __restrict__ staging, const size_t koff0, const size_t koff1, const size_t koff2) {
    const lru_fill_desc & d = descs.d[blockIdx.z];
    const size_t koff = blockIdx.z == 0 ? koff0 : blockIdx.z == 1 ? koff1 : koff2;
    const int n_fill = fill_list[3 * n_slot];
    for (int j = blockIdx.y; j < n_fill; j += gridDim.y) {
        const int32_t s = fill_list[3 * j + 1];
        const uint4 * ps = (const uint4 *) (d.pool + (size_t) s * d.estride);
        uint4 * pd = (uint4 *) (staging + koff + (size_t) j * d.estride);
        for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < d.estride / 16; i += (size_t) gridDim.x * blockDim.x) {
            pd[i] = ps[i];
        }
    }
}

// ... and write them to their host slots on a side stream while the matmuls run (each device writes its own slice bytes)
static __global__ void k_lru_write_back(const int32_t * __restrict__ fill_list, int n_slot, const lru_fill_descs descs,
                                        const char * __restrict__ staging, const size_t koff0, const size_t koff1, const size_t koff2) {
    const lru_fill_desc & d = descs.d[blockIdx.z];
    const size_t koff = blockIdx.z == 0 ? koff0 : blockIdx.z == 1 ? koff1 : koff2;
    const int n_fill = fill_list[3 * n_slot];
    for (int j = blockIdx.y; j < n_fill; j += gridDim.y) {
        const int32_t h = fill_list[3 * j + 2];
        const char * st = staging + koff + (size_t) j * d.estride;
        char * hd = d.src + (size_t) h * d.src_estride + d.src_off;
        const size_t rs = d.n_rows > 1 ? d.src_row_stride : d.seg_bytes;
        if (d.vec) {
            const size_t seg16 = d.seg_bytes / 16, total = d.n_rows * seg16;
            for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
                const size_t row = i / seg16, c = i % seg16;
                ((uint4 *) (hd + row * rs))[c] = ((const uint4 *) (st + row * d.seg_bytes))[c];
            }
        } else {
            const size_t total = d.n_rows * d.seg_bytes;
            for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < total; i += (size_t) gridDim.x * blockDim.x) {
                const size_t row = i / d.seg_bytes, b = i % d.seg_bytes;
                hd[row * rs + b] = st[row * d.seg_bytes + b];
            }
        }
    }
}

static constexpr int LRU_STAGING_FILLS = 32;   // fills per call the staging buffer holds; more -> in-place swap

// fills nodes[0..n) share one fill list (the gate/up/down fills of a layer): one launch
// p100-series LRU fill overlap (GGML_CUDA_LRU_OVERLAP=1, opt-in): the decode fill (in-place swap) runs on a high-priority side
// stream; the expert matvec on that pool computes its resident entries first, then waits for the fill (mmvq.cu); every other reader
// of a filled pool, LRU_ASSEMBLE, the end of the graph and synchronize wait for the fill first. Off with GGML_CUDA_MMVQ_LB (that path
// could take the matvec without the split).
// wispoffates' fill overlap (GGML_CUDA_LRU_OVERLAP) needs his matvec's split resident/pending pass (mmvq-lb), not ported: off
static bool ggml_cuda_lru_overlap_enabled() {
    return false;
}

void ggml_cuda_lru_ovl_join(ggml_backend_cuda_context & ctx) {
    if (ctx.lru_ovl_active) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ctx.lru_ovl_fill_event, 0));
        ctx.lru_ovl_active = false;
    }
}

bool ggml_cuda_lru_ovl_reads_pool(const ggml_backend_cuda_context & ctx, const ggml_tensor * t) {
    if (!ctx.lru_ovl_active || t == nullptr || t->data == nullptr) {
        return false;
    }
    const char * d = (const char *) t->data;
    for (int k = 0; k < ctx.lru_ovl_npools; ++k) {
        if (d >= ctx.lru_ovl_pool[k] && d < ctx.lru_ovl_pool[k] + ctx.lru_ovl_pool_bytes[k]) {
            return true;
        }
    }
    return false;
}

// before node i (and the nodes a fusion may take with it): wait for an in-flight fill unless the only pool readers are expert
// matvecs that split themselves (MUL_MAT_ID with <= MMVQ_MAX_BATCH_SIZE tokens)
void ggml_cuda_lru_ovl_guard(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int last = std::min(cgraph->n_nodes, i + 4);
    for (int n = i; n < last && ctx.lru_ovl_active; ++n) {
        const ggml_tensor * node = cgraph->nodes[n];
        if (node->op == GGML_OP_LRU_ASSEMBLE) {
            ggml_cuda_lru_ovl_join(ctx);
            return;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (ggml_cuda_lru_ovl_reads_pool(ctx, node->src[j]) &&
                    !(node->op == GGML_OP_MUL_MAT_ID && node->ne[2] <= MMVQ_MAX_BATCH_SIZE)) {
                ggml_cuda_lru_ovl_join(ctx);
                return;
            }
        }
    }
}


// p100-series copy-engine LRU fills (GGML_CUDA_LRU_CE=1, exclusive cache, no victim stash, no disk refill). The decide kernels
// stay on the GPU (the state changes in stream order); a fill op does not run k_lru_fill on the SMs but publishes its fill
// list to device-readable host memory, and a host thread per device replays it with the copy engines on a copy stream:
// for every (expert, slot, host slot) and kind, pool slot -> VRAM scratch (the victim), host slot -> pool slot (the incoming
// expert), scratch -> host slot (the swap). Copies run in decide order, so they follow the same state transitions. A real
// fill makes the main stream wait (a one-thread spin on a device counter) until its job and every earlier one are done; a
// prefetch fill (LLAMA_MOE_LRU_PF) does not wait, its copies overlap the compute, and the next layer's real fill waits for them.
bool ggml_cuda_lru_ce_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_LRU_CE"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

// the first fills of a device go through k_lru_fill: buffers that grow on first use (cudaMalloc/cudaFree wait for the whole
// device) must not do so while a stream waits for host-issued copies. GGML_CUDA_LRU_CE_AFTER fill ops (default 2000 ~ 40 tokens).
static bool ggml_cuda_lru_ce_ready(int device) {
    static const int after = [] { const char * e = getenv("GGML_CUDA_LRU_CE_AFTER"); return e ? atoi(e) : 2000; }();
    static int count[GGML_CUDA_MAX_DEVICES] = {};
    return ++count[device] > after;
}

static constexpr int LRU_CE_RING = 256;
static constexpr int LRU_CE_MAXF = 256;
struct lru_ce_slot {
    volatile uint32_t seq;
    volatile int32_t  n;
    int32_t list[3 * LRU_CE_MAXF];
};
struct lru_ce_job {
    uint32_t seq;
    int n_kinds;
    int layer;
    lru_fill_desc d[3];
};
static constexpr int LRU_CE_REGIONS = 16;   // victim scratch regions (one fill entry each), reused after their write-back
struct lru_ce_dev {
    bool init = false;
    int device = 0;
    cudaStream_t copy = nullptr;
    lru_ce_slot * ring = nullptr;          // mapped pinned host memory
    unsigned int * done = nullptr;         // device memory: last job whose pool fills are done
    unsigned int * wb_done = nullptr;      // device memory: last job whose victim write-backs are done
    unsigned int * done_k = nullptr;       // device memory [2]: last job whose first / second copied kind is done (staged release)
    const void * pend_pool[3] = {};        // dispatch thread: pools still being filled for job pend_seq
    unsigned int * pend_ctr[3] = {};
    uint32_t pend_seq[3] = {};
    int pend_unconsumed = 0;
    char * scratch = nullptr;              // device memory: LRU_CE_REGIONS regions, one victim of each kind per region
    size_t scratch_size = 0;
    size_t region_size = 0;
    cudaStream_t wb = nullptr;             // write-backs of the victims to their host slots, off the critical path
    cudaEvent_t region_ev[LRU_CE_REGIONS] = {};
    bool region_used[LRU_CE_REGIONS] = {};
    int region_pos = 0;
    cudaEvent_t after_ev = nullptr;
    std::vector<cudaEvent_t> layer_ev;     // per pooled layer: its last write-back done
    std::vector<uint8_t> layer_used;
    uint32_t issued = 0;                   // host: last published job
    std::atomic<uint32_t> handled{0};      // worker: last job whose copies are issued
    std::condition_variable handled_cv;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<lru_ce_job> queue;
    std::thread worker;
};
static lru_ce_dev g_lru_ce[GGML_CUDA_MAX_DEVICES];

static __global__ void k_lru_ce_publish(const int32_t * __restrict__ fill_list, int n_slot, lru_ce_slot * slot, uint32_t seq) {
    const int n = min(fill_list[3 * n_slot], LRU_CE_MAXF);
    for (int i = threadIdx.x; i < 3 * n; i += blockDim.x) {
        slot->list[i] = fill_list[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        slot->n = n;
        __threadfence_system();
        slot->seq = seq;
        __threadfence_system();
    }
}

static __global__ void k_lru_ce_wait(const unsigned int * done, uint32_t seq) {
    while ((int32_t) (*(volatile const unsigned int *) done - seq) < 0) {
    }
}

static __global__ void k_lru_ce_done(unsigned int * done, uint32_t seq) {
    *(volatile unsigned int *) done = seq;
}

static bool ggml_cuda_lru_ce_debug() {
    static const bool on = getenv("GGML_CUDA_LRU_CE_DEBUG") != nullptr;
    return on;
}

static bool ggml_cuda_lru_ce_staged() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_LRU_CE_STAGED"); return !e || atoi(e) != 0; }();
    return on;
}

// the main stream waits for a pool kind still being filled before the first op that reads it (MUL_MAT_ID, mmvq incl. its fused
// gate); every pending kind is waited at the next fill / host sync at the latest
void ggml_cuda_lru_ce_pending_wait(ggml_backend_cuda_context & ctx, const void * data) {
    lru_ce_dev & cd = g_lru_ce[ctx.device];
    for (int k = 0; k < 3; k++) {
        if (cd.pend_pool[k] != nullptr && (data == nullptr || cd.pend_pool[k] == data)) {
            CU_CHECK(cuStreamWaitValue32((CUstream) ctx.stream(), (CUdeviceptr) cd.pend_ctr[k], cd.pend_seq[k], CU_STREAM_WAIT_VALUE_GEQ));
            if (data == nullptr && ++cd.pend_unconsumed == 1) {
                GGML_LOG_WARN("%s: device %d: a staged LRU fill was not consumed by a matvec (pool pointer mismatch?)\n", __func__, ctx.device);
            }
            cd.pend_pool[k] = nullptr;
        }
    }
}

static void ggml_cuda_lru_ce_worker(lru_ce_dev * cd) {
    ggml_cuda_set_device(cd->device);
    for (;;) {
        lru_ce_job job;
        {
            std::unique_lock<std::mutex> lk(cd->mtx);
            cd->cv.wait(lk, [cd] { return !cd->queue.empty(); });
            job = cd->queue.front();
            cd->queue.erase(cd->queue.begin());
        }
        static const bool prof = getenv("GGML_CUDA_LRU_CE_PROF") != nullptr;
        static thread_local double p_spin = 0, p_issue = 0, p_fills = 0; static thread_local int p_jobs = 0;
        const auto tq = std::chrono::steady_clock::now();
        lru_ce_slot * sl = cd->ring + job.seq % LRU_CE_RING;
        while (sl->seq != job.seq) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (prof) {
            p_spin += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tq).count();
        }
        const int n = sl->n;
        if (ggml_cuda_lru_ce_debug()) {
            fprintf(stderr, "lru_ce dev %d: job %u published, %d fills\n", cd->device, job.seq, n);
        }
        while ((int) cd->layer_ev.size() <= job.layer) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            cd->layer_ev.push_back(e);
            cd->layer_used.push_back(0);
        }
        // host slots of this layer hold what the state says only after the layer's earlier write-backs
        if (cd->layer_used[job.layer]) {
            CUDA_CHECK(cudaStreamWaitEvent(cd->copy, cd->layer_ev[job.layer], 0));
        }
        // staged release (GGML_CUDA_LRU_CE_STAGED, default on): one chunk, kind by kind, each kind's counter written as soon as its
        // copies are done, so the gate matvec starts while up and down still copy (the main stream waits for kind 0 at the fill
        // and for kinds 1 / 2 right before the op that reads their pool, ggml_cuda_lru_ce_pending_wait)
        bool released = false;
        if (ggml_cuda_lru_ce_staged() && n > 0 && n <= LRU_CE_REGIONS && job.n_kinds == 3) {
            int regions[LRU_CE_REGIONS];
            for (int j = 0; j < n; j++) {
                const int r = cd->region_pos;
                cd->region_pos = (cd->region_pos + 1) % LRU_CE_REGIONS;
                regions[j] = r;
                if (cd->region_used[r]) {
                    CUDA_CHECK(cudaStreamWaitEvent(cd->copy, cd->region_ev[r], 0));
                }
            }
            // in the order the graph reads the pools (ffn_moe_up, then gate, then down): kinds 1, 0, 2
            static constexpr int korder[3] = {1, 0, 2};
            size_t koffs[3];
            for (int k = 0, off = 0; k < 3; k++) {
                koffs[k] = off;
                off += (job.d[k].n_rows * job.d[k].seg_bytes + 255) & ~(size_t) 255;
            }
            for (int ki = 0; ki < 3; ki++) {
                const int k = korder[ki];
                const lru_fill_desc & d = job.d[k];
                const size_t bytes = d.n_rows * d.seg_bytes;
                const size_t koff = koffs[k];
                for (int j = 0; j < n; j++) {
                    const int32_t s = sl->list[3 * j + 1];
                    const int32_t h = sl->list[3 * j + 2] & 0x3fffffff;
                    char * pool_e = d.pool + (size_t) s * d.estride;
                    char * host_e = d.src + (size_t) h * d.src_estride + d.src_off;
                    if (d.swap) {
                        CUDA_CHECK(cudaMemcpyAsync(cd->scratch + (size_t) regions[j] * cd->region_size + koff, pool_e, bytes,
                                                   cudaMemcpyDeviceToDevice, cd->copy));
                    }
                    CUDA_CHECK(cudaMemcpyAsync(pool_e, host_e, bytes, cudaMemcpyHostToDevice, cd->copy));
                }
                CU_CHECK(cuStreamWriteValue32((CUstream) cd->copy, (CUdeviceptr) (ki < 2 ? cd->done_k + ki : cd->done), job.seq,
                                              CU_STREAM_WRITE_VALUE_DEFAULT));
            }
            released = true;
            CUDA_CHECK(cudaEventRecord(cd->after_ev, cd->copy));
            CUDA_CHECK(cudaStreamWaitEvent(cd->wb, cd->after_ev, 0));
            for (int j = 0; j < n; j++) {
                const int32_t h = sl->list[3 * j + 2] & 0x3fffffff;
                const int r = regions[j];
                size_t off = (size_t) r * cd->region_size;
                for (int k = 0; k < 3; k++) {
                    const lru_fill_desc & d = job.d[k];
                    const size_t bytes = d.n_rows * d.seg_bytes;
                    if (d.swap) {
                        char * host_e = d.src + (size_t) h * d.src_estride + d.src_off;
                        CUDA_CHECK(cudaMemcpyAsync(host_e, cd->scratch + off, bytes, cudaMemcpyDeviceToHost, cd->wb));
                    }
                    off += (bytes + 255) & ~(size_t) 255;
                }
                CUDA_CHECK(cudaEventRecord(cd->region_ev[r], cd->wb));
                cd->region_used[r] = true;
            }
        }
        // chunks of at most LRU_CE_REGIONS entries: a chunk's victims are written back before the next chunk reuses the regions
        for (int j0 = 0; !released && (j0 < n || (n == 0 && j0 == 0)); j0 += LRU_CE_REGIONS) {
            const int j1 = std::min(n, j0 + LRU_CE_REGIONS);
            int regions[LRU_CE_REGIONS];
            for (int j = j0; j < j1; j++) {
                const int32_t s = sl->list[3 * j + 1];
                const int32_t h = sl->list[3 * j + 2] & 0x3fffffff;
                const int r = cd->region_pos;
                cd->region_pos = (cd->region_pos + 1) % LRU_CE_REGIONS;
                regions[j - j0] = r;
                if (cd->region_used[r]) {
                    CUDA_CHECK(cudaStreamWaitEvent(cd->copy, cd->region_ev[r], 0));   // its previous victim is written back
                }
                size_t off = (size_t) r * cd->region_size;
                for (int k = 0; k < job.n_kinds; k++) {
                    const lru_fill_desc & d = job.d[k];
                    char * pool_e = d.pool + (size_t) s * d.estride;
                    char * host_e = d.src + (size_t) h * d.src_estride + d.src_off;
                    const size_t bytes = d.n_rows * d.seg_bytes;
                    if (d.swap) {
                        CUDA_CHECK(cudaMemcpyAsync(cd->scratch + off, pool_e, bytes, cudaMemcpyDeviceToDevice, cd->copy));
                    }
                    CUDA_CHECK(cudaMemcpyAsync(pool_e, host_e, bytes, cudaMemcpyHostToDevice, cd->copy));
                    off += (bytes + 255) & ~(size_t) 255;
                }
            }
            if (j1 >= n) {
                // the pool is ready: release the waiting stream first, then write the last victims back on the other stream
                CU_CHECK(cuStreamWriteValue32((CUstream) cd->copy, (CUdeviceptr) cd->done_k, job.seq, CU_STREAM_WRITE_VALUE_DEFAULT));
                CU_CHECK(cuStreamWriteValue32((CUstream) cd->copy, (CUdeviceptr) (cd->done_k + 1), job.seq, CU_STREAM_WRITE_VALUE_DEFAULT));
                CU_CHECK(cuStreamWriteValue32((CUstream) cd->copy, (CUdeviceptr) cd->done, job.seq, CU_STREAM_WRITE_VALUE_DEFAULT));
                released = true;
            }
            CUDA_CHECK(cudaEventRecord(cd->after_ev, cd->copy));
            CUDA_CHECK(cudaStreamWaitEvent(cd->wb, cd->after_ev, 0));
            for (int j = j0; j < j1; j++) {
                const int32_t h = sl->list[3 * j + 2] & 0x3fffffff;
                const int r = regions[j - j0];
                size_t off = (size_t) r * cd->region_size;
                for (int k = 0; k < job.n_kinds; k++) {
                    const lru_fill_desc & d = job.d[k];
                    const size_t bytes = d.n_rows * d.seg_bytes;
                    if (d.swap) {
                        char * host_e = d.src + (size_t) h * d.src_estride + d.src_off;
                        CUDA_CHECK(cudaMemcpyAsync(host_e, cd->scratch + off, bytes, cudaMemcpyDeviceToHost, cd->wb));
                    }
                    off += (bytes + 255) & ~(size_t) 255;
                }
                CUDA_CHECK(cudaEventRecord(cd->region_ev[r], cd->wb));
                cd->region_used[r] = true;
            }
            if (n == 0) break;
        }
        GGML_ASSERT(released);
        if (prof) {
            const auto t2 = std::chrono::steady_clock::now();
            p_issue += std::chrono::duration<double, std::micro>(t2 - tq).count();
            p_fills += n; p_jobs++;
            if (p_jobs % 4000 == 0) {
                fprintf(stderr, "lru_ce prof dev %d: %d jobs, %.2f fills/job, host us/job: publish wait + issue %.1f (spin %.1f)\n",
                        cd->device, p_jobs, p_fills / p_jobs, p_issue / p_jobs, p_spin / p_jobs);
                p_issue = p_spin = p_fills = 0; p_jobs = 0;
            }
        }
        {
            std::lock_guard<std::mutex> lk(cd->mtx);
            cd->handled.store(job.seq);
        }
        cd->handled_cv.notify_all();
        if (n > 0) {
            CUDA_CHECK(cudaEventRecord(cd->layer_ev[job.layer], cd->wb));
            cd->layer_used[job.layer] = 1;
        }
        // in job order: every reader of host slots outside this path (prefill assembly, remap) waits for it
        CU_CHECK(cuStreamWriteValue32((CUstream) cd->wb, (CUdeviceptr) cd->wb_done, job.seq, CU_STREAM_WRITE_VALUE_DEFAULT));
        if (ggml_cuda_lru_ce_debug() && job.seq < 30) {
            fprintf(stderr, "lru_ce dev %d: job %u copies issued\n", cd->device, job.seq);
        }
    }
}

static lru_ce_dev & ggml_cuda_lru_ce_dev(int device, size_t scratch_need) {
    lru_ce_dev & cd = g_lru_ce[device];
    if (!cd.init) {
        ggml_cuda_set_device(device);
        cd.device = device;
        CUDA_CHECK(cudaStreamCreateWithFlags(&cd.copy, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&cd.wb, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&cd.after_ev, cudaEventDisableTiming));
        for (int r = 0; r < LRU_CE_REGIONS; r++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&cd.region_ev[r], cudaEventDisableTiming));
        }
        // per-layer events up front: the worker makes no allocating CUDA calls while it runs
        for (int l = 0; l < 256; l++) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            cd.layer_ev.push_back(e);
            cd.layer_used.push_back(0);
        }
        CUDA_CHECK(cudaHostAlloc((void **) &cd.ring, sizeof(lru_ce_slot) * LRU_CE_RING, cudaHostAllocMapped | cudaHostAllocPortable));
        memset((void *) cd.ring, 0, sizeof(lru_ce_slot) * LRU_CE_RING);
        CUDA_CHECK(cudaMalloc((void **) &cd.done, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(cd.done, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc((void **) &cd.wb_done, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(cd.wb_done, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc((void **) &cd.done_k, 2 * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(cd.done_k, 0, 2 * sizeof(unsigned int)));
        cd.init = true;
        cd.worker = std::thread(ggml_cuda_lru_ce_worker, &cd);
        cd.worker.detach();
    }
    if (scratch_need > cd.scratch_size) {
        // only grows between jobs of a new shape: wait for the worker to drain before swapping the buffer
        CUDA_CHECK(cudaStreamSynchronize(cd.copy));
        for (;;) {
            std::lock_guard<std::mutex> lk(cd.mtx);
            if (cd.queue.empty()) break;
        }
        CUDA_CHECK(cudaStreamSynchronize(cd.copy));
        // on this device: the calling dispatch thread's current device may be another one (a scratch there turned every
        // victim copy into a peer copy that waited on the other GPU's all-reduce)
        ggml_cuda_set_device(device);
        if (cd.scratch) CUDA_CHECK(cudaFree(cd.scratch));
        // regions of at least 2 MiB: a later layer with larger experts must not regrow it (cudaFree waits for the whole device)
        cd.region_size = std::max<size_t>(scratch_need, 2u << 20);
        scratch_need = cd.region_size;
        CUDA_CHECK(cudaMalloc((void **) &cd.scratch, cd.region_size * LRU_CE_REGIONS));
        cd.scratch_size = scratch_need;
        // CUDA 12 loads kernel modules lazily, and a first launch cannot load while a kernel spins on the device (the main
        // stream's k_lru_ce_wait): every kernel and copy kind the worker uses runs once here, before any wait exists

        CUDA_CHECK(cudaMemcpyAsync(cd.scratch + 256, cd.scratch, 256, cudaMemcpyDeviceToDevice, cd.copy));
        CUDA_CHECK(cudaMemcpyAsync(cd.scratch, (const void *) cd.ring, 256, cudaMemcpyHostToDevice, cd.copy));
        CUDA_CHECK(cudaMemcpyAsync((void *) (cd.ring + 1), cd.scratch, 256, cudaMemcpyDeviceToHost, cd.copy));
        CUDA_CHECK(cudaStreamSynchronize(cd.copy));
        memset((void *) cd.ring, 0, sizeof(lru_ce_slot) * 2);
    }
    return cd;
}

// true if handled by the copy engines
static bool ggml_cuda_lru_ce_fill(ggml_backend_cuda_context & ctx, const int32_t * fl, int n_slot, const lru_fill_descs & descs, int n,
                                  bool prefetch, int layer) {
    size_t need = 0;
    for (int k = 0; k < n; k++) {
        if (!descs.d[k].swap && !prefetch && false) return false;
        need += ((descs.d[k].n_rows * descs.d[k].seg_bytes) + 255) & ~(size_t) 255;
    }
    lru_ce_dev & cd = ggml_cuda_lru_ce_dev(ctx.device, need);
    ggml_cuda_lru_ce_pending_wait(ctx, nullptr);
    // bound the host's run-ahead: a stream waiting for copies stops the device, and a dispatch thread that keeps launching fills
    // the driver's queue and then blocks inside a launch holding the context lock, which the worker needs for its copies.
    // Waiting here (no CUDA call held) until the worker is at most 16 jobs behind keeps the queue short.
    {
        std::unique_lock<std::mutex> lk(cd.mtx);
        cd.handled_cv.wait(lk, [&cd] { return (int32_t) (cd.issued - cd.handled.load()) < 16; });
    }
    lru_ce_job job{};
    job.seq = ++cd.issued;
    job.n_kinds = n;
    job.layer = layer;
    for (int k = 0; k < n; k++) job.d[k] = descs.d[k];
    k_lru_ce_publish<<<1, 256, 0, ctx.stream()>>>(fl, n_slot, cd.ring + job.seq % LRU_CE_RING, job.seq);
    if (ggml_cuda_lru_ce_debug()) {
        fprintf(stderr, "lru_ce dev %d: queued job %u (prefetch %d) layer %d\n", ctx.device, job.seq, (int) prefetch, layer);
    }
    {
        std::lock_guard<std::mutex> lk(cd.mtx);
        cd.queue.push_back(job);
    }
    cd.cv.notify_one();
    if (!prefetch) {
        // the GPU front end waits for the counter (no kernel occupies an SM, no module load needed)
        if (ggml_cuda_lru_ce_staged() && n == 3) {
            // nothing waits here: each pool is waited for right before its first reader (up, gate, down), so the ops in between
            // (the shared expert) run while the copies land
            cd.pend_pool[0] = descs.d[1].pool; cd.pend_ctr[0] = cd.done_k;     cd.pend_seq[0] = job.seq;
            cd.pend_pool[1] = descs.d[0].pool; cd.pend_ctr[1] = cd.done_k + 1; cd.pend_seq[1] = job.seq;
            cd.pend_pool[2] = descs.d[2].pool; cd.pend_ctr[2] = cd.done;       cd.pend_seq[2] = job.seq;
        } else {
            CU_CHECK(cuStreamWaitValue32((CUstream) ctx.stream(), (CUdeviceptr) cd.done, job.seq, CU_STREAM_WAIT_VALUE_GEQ));
        }
    }
    return true;
}

// host slots are current only after the write-backs of every job published so far
void ggml_cuda_lru_ce_sync_host(ggml_backend_cuda_context & ctx) {
    ggml_cuda_lru_ce_pending_wait(ctx, nullptr);
    lru_ce_dev & cd = g_lru_ce[ctx.device];
    if (cd.init && cd.issued > 0) {
        if (ggml_cuda_lru_ce_debug()) {
            fprintf(stderr, "lru_ce dev %d: host-slot sync waits for job %u\n", ctx.device, cd.issued);
        }
        CU_CHECK(cuStreamWaitValue32((CUstream) ctx.stream(), (CUdeviceptr) cd.wb_done, cd.issued, CU_STREAM_WAIT_VALUE_GEQ));
    }
}

void ggml_cuda_lru_fill_n(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n) {
    const ggml_tensor * fill_list = nodes[0]->src[0];
    const int n_slot = (int) ((fill_list->ne[0] - 1) / 3);
    const int n_ids  = ggml_get_op_params_i32(nodes[0], 2);   // at most this many fills (0: unknown -> n_slot)
    int rows = std::max(1, std::min(n_slot, n_ids > 0 ? n_ids : n_slot));
    lru_fill_descs descs{};
    for (int k = 0; k < n; k++) {
        descs.d[k] = ggml_cuda_lru_fill_desc(nodes[k]);
    }
    // p100-series fill grid (PTM2): blocks whose fill-list row is past n_fill only read the count and exit, but sm_60 still
    // dispatches every one (~36 ns per block): 64 x 40 x 3 blocks at a 4-token verify = ~277 us per layer with nothing to fill.
    // The block rows walk the fill list, so a few rows cover any count; GGML_CUDA_LRU_FILL_ROWS (default 4, 0 = one row per id as
    // before), GGML_CUDA_LRU_FILL_BX blocks per slice (default 32; fillbench: 32 moves a fill faster than 64 or 128)
    static const int fill_rows = [] { const char * e = getenv("GGML_CUDA_LRU_FILL_ROWS"); return e ? atoi(e) : 4; }();
    static const int fill_bx   = [] { const char * e = getenv("GGML_CUDA_LRU_FILL_BX"); return e && atoi(e) > 0 ? atoi(e) : 32; }();
    static const bool fill_ab  = ggml_cuda_ab_listed("fillgrid");
    const bool old_grid = fill_rows <= 0 || (fill_ab && ggml_cuda_ab_arm() == 1);
    if (!old_grid) {
        rows = std::min(rows, fill_rows);
    }
    const dim3 grid(old_grid ? 64 : fill_bx, rows, n), block(256);
    const int32_t * fl = (const int32_t *) fill_list->data;
    ggml_lru_ctrl * ctrl;
    memcpy(&ctrl, (const char *) nodes[0]->op_params + 16, sizeof(ctrl));
    const int layer = ggml_get_op_params_i32(nodes[0], 6);

    if (ggml_cuda_lru_ce_enabled() && ctrl == nullptr && ctx.lru_stash_ctl == nullptr && descs.d[0].n_rows == 1) {
        if (ggml_cuda_lru_ce_ready(ctx.device)) {
            GGML_ASSERT(n == 3 && "GGML_CUDA_LRU_CE needs the fused gate/up/down fill");
            ggml_cuda_lru_ce_fill(ctx, fl, n_slot, descs, n, ggml_cuda_lru_is_prefetch(fill_list->view_src), layer);
            return;
        }
        // warm-up (before the copy engines take over): every fill, prefetches too, on the main stream where its decide ran
        ggml_cuda_lru_ce_sync_host(ctx);
        k_lru_fill<<<grid, block, 0, ctx.stream()>>>(fl, n_slot, descs, nullptr, layer, 0, nullptr, lru_stash_args{});
        return;
    }
    if (ggml_cuda_lru_is_prefetch(fill_list->view_src)) {
        ggml_cuda_lru_pf_state & pf = ggml_cuda_lru_pf(ctx.device, layer);
        k_lru_fill<<<grid, block, 0, pf.stream>>>(fl, n_slot, descs, nullptr, layer, 0, nullptr, lru_stash_args{});
        CUDA_CHECK(cudaEventRecord(pf.ev_layer[layer], pf.stream));
        return;
    }

    // victim stash: the decide encoded ring slots when the cache is exclusive (same condition as in ggml_cuda_lru_decide)
    lru_stash_args st{};
    {
        ggml_tensor * src0;
        memcpy(&src0, nodes[0]->op_params, sizeof(src0));
        st.decode = ctx.lru_stash_ctl != nullptr && ctrl == nullptr && ggml_get_op_params_i32(nodes[0], 7) && src0->ne[2] <= 4096;
        if (st.decode) {
            size_t sum = 0;
            bool ok = true;
            for (int k = 0; k < n; k++) {
                st.koff[k] = (uint32_t) sum;
                sum += descs.d[k].n_rows * descs.d[k].seg_bytes;
                ok = ok && descs.d[k].vec && descs.d[k].swap && descs.d[k].n_rows == 1;
            }
            ctx.lru_stash_need = std::max(ctx.lru_stash_need, sum);
            st.use    = ok && ctx.lru_stash != nullptr && sum <= ctx.lru_stash_stride;
            st.rec    = ctx.lru_stash_rec;
            st.buf    = ctx.lru_stash;
            st.stride = ctx.lru_stash_stride;
        }
    }
    int dev = 0;
    if (ctrl != nullptr) {
        cudaGetDevice(&dev);
        if (ctx.lru_counter == nullptr) {
            CUDA_CHECK(cudaMalloc(&ctx.lru_counter, sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(ctx.lru_counter, 0, sizeof(unsigned int)));
        }
    }

    // exclusive: staged write-back, opt-in (GGML_CUDA_LRU_STAGED=1). WB (10-13): identical text but SLOWER than the in-place
    // swap (Flash-Next 288 slots 22.5 vs 28.4 t/s, Qwen Q5 96 slots 28.1 vs 34.3): two extra kernels + a main/side-stream
    // event round trip per layer cost more than the write-back they hide on the decode path
    static const bool staged_on = [] { const char * e = getenv("GGML_CUDA_LRU_STAGED"); return e != nullptr && atoi(e) != 0; }();
    bool staged = staged_on && descs.d[0].swap && rows <= LRU_STAGING_FILLS;
    for (int k = 0; k < n && staged; k++) {
        staged = descs.d[k].estride % 16 == 0;
    }
    if (!staged) {
        if (ggml_cuda_lru_overlap_enabled() && ctrl == nullptr) {
            if (ctx.lru_ovl_stream == nullptr) {
                ggml_cuda_set_device(ctx.device);
                int lo = 0, hi = 0;
                CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
                CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.lru_ovl_stream, cudaStreamNonBlocking, hi));
                CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_ovl_dec_event, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_ovl_fill_event, cudaEventDisableTiming));
            }
            ggml_cuda_lru_ovl_join(ctx);   // one fill in flight at a time
            CUDA_CHECK(cudaEventRecord(ctx.lru_ovl_dec_event, ctx.stream()));
            CUDA_CHECK(cudaStreamWaitEvent(ctx.lru_ovl_stream, ctx.lru_ovl_dec_event, 0));
            ggml_cuda_lru_wb_wait(ctx, ctx.lru_ovl_stream);
            k_lru_fill<<<grid, block, 0, ctx.lru_ovl_stream>>>(fl, n_slot, descs, ctrl, layer, dev, ctx.lru_counter, st);
            CUDA_CHECK(cudaEventRecord(ctx.lru_ovl_fill_event, ctx.lru_ovl_stream));
            ctx.lru_ovl_active = true;
            ctx.lru_ovl_fill   = fl;
            ctx.lru_ovl_nslot  = n_slot;
            ctx.lru_ovl_npools = n;
            for (int k = 0; k < n; k++) {
                ctx.lru_ovl_pool[k]       = descs.d[k].pool;
                ctx.lru_ovl_pool_bytes[k] = (size_t) n_slot * descs.d[k].estride;
            }
            return;
        }
        ggml_cuda_lru_wb_wait(ctx, ctx.stream());
        k_lru_fill<<<grid, block, 0, ctx.stream()>>>(fl, n_slot, descs, ctrl, layer, dev, ctx.lru_counter, st);
        static const int stash_chunk = [] { const char * e = getenv("GGML_CUDA_LRU_STASH_CHUNK"); return e && atoi(e) > 0 ? atoi(e) : 8; }();
        if (ctx.lru_stash_armed && ctx.lru_stash_dirty && ctx.lru_stash_hsnap == nullptr && ++ctx.lru_stash_calls >= stash_chunk) {
            ggml_cuda_lru_stash_flush(ctx, false);   // takes the entries of the fills before this one (this decide's snapshot)
        }
        return;
    }
    size_t koff[3] = { 0, 0, 0 }, need = 0;
    for (int k = 0; k < n; k++) {
        koff[k] = need;
        need += (size_t) LRU_STAGING_FILLS * descs.d[k].estride;
    }
    if (ctx.lru_wb_stream == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.lru_wb_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_wb_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_fill_event, cudaEventDisableTiming));
    }
    if (need > ctx.lru_staging_size) {
        if (ctx.lru_staging != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(ctx.lru_wb_stream));
            CUDA_CHECK(cudaFree(ctx.lru_staging));
        }
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc(&ctx.lru_staging, need));
        ctx.lru_staging_size = need;
    }
    // the staging buffer and the host slots of the previous write-back must be free
    ggml_cuda_lru_wb_wait(ctx, ctx.stream());
    k_lru_save_victims<<<grid, block, 0, ctx.stream()>>>(fl, n_slot, descs, (char *) ctx.lru_staging, koff[0], koff[1], koff[2]);
    lru_fill_descs plain = descs;
    for (int k = 0; k < n; k++) {
        plain.d[k].swap = 0;
    }
    k_lru_fill<<<grid, block, 0, ctx.stream()>>>(fl, n_slot, plain, nullptr, 0, 0, nullptr, st);
    CUDA_CHECK(cudaEventRecord(ctx.lru_fill_event, ctx.stream()));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.lru_wb_stream, ctx.lru_fill_event, 0));
    const dim3 grid_wb(8, rows, n);
    k_lru_write_back<<<grid_wb, block, 0, ctx.lru_wb_stream>>>(fl, n_slot, descs, (const char *) ctx.lru_staging, koff[0], koff[1], koff[2]);
    CUDA_CHECK(cudaEventRecord(ctx.lru_wb_event, ctx.lru_wb_stream));
    ctx.lru_wb_pending = true;
}

void ggml_cuda_lru_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_lru_fill_n(ctx, &dst, 1);
}

// per-pass timer op marks (GGML_CUDA_PASS_TIMER_OPS): layer from the "<name>-<il>" suffix, fill bytes per fill on this device

// p100-series: prefill copy/compute overlap for LRU_ASSEMBLE (GGML_CUDA_LRU_ASYNC=0: on the main stream). The side stream gets
// a narrow grid (GGML_CUDA_LRU_ASYNC_GX, default 4 blocks per expert): zero-copy reads need only a few warps in flight to fill
// PCIe, and every SM the assembly takes is one the overlapped compute does not get.
bool ggml_cuda_lru_async_enabled() {
    // opt-in (FZD: no gain; useful only with LLAMA_MOE_LRU_PREFETCH=1)
    static const bool on = [] { const char * e = getenv("GGML_CUDA_LRU_ASYNC"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

static int ggml_cuda_lru_async_gx() {
    static const int gx = [] { const char * e = getenv("GGML_CUDA_LRU_ASYNC_GX"); return e ? std::max(1, atoi(e)) : 4; }();
    return gx;
}

void ggml_cuda_lru_assemble_async(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &cap));
    int slot = -1;
    for (int k = 0; k < ggml_backend_cuda_context::LRU_MAX_PENDING && cap == cudaStreamCaptureStatusNone; ++k) {
        if (ctx.lru_pending[k] == nullptr) {
            slot = k;
            break;
        }
    }
    if (slot < 0) {                      // capturing a CUDA graph, or too many in flight: in order on the main stream
        ggml_cuda_lru_assemble(ctx, dst);
        return;
    }
    if (ctx.lru_stream == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.lru_stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_fork_event, cudaEventDisableTiming));
        for (int k = 0; k < ggml_backend_cuda_context::LRU_MAX_PENDING; ++k) {
            CUDA_CHECK(cudaEventCreateWithFlags(&ctx.lru_events[k], cudaEventDisableTiming));
        }
    }
    ggml_cuda_lru_ce_sync_host(ctx);   // the side stream inherits it through the fork event
    CUDA_CHECK(cudaEventRecord(ctx.lru_fork_event, ctx.stream()));
    CUDA_CHECK(cudaStreamWaitEvent(ctx.lru_stream, ctx.lru_fork_event, 0));
    if (ggml_cuda_lru_dma_enabled()) {
        ggml_cuda_lru_assemble_dma(ctx, dst, ctx.lru_stream);   // missing experts by DMA on the side stream, pooled gather on main
    } else {
        ggml_cuda_lru_assemble(ctx, dst, ctx.lru_stream, ggml_cuda_lru_async_gx());
    }
    CUDA_CHECK(cudaEventRecord(ctx.lru_events[slot], ctx.lru_stream));
    ctx.lru_pending[slot] = dst;
    ctx.n_lru_pending++;
}

// before computing node i (and the nodes a fusion may take with it, at most a few): wait for any pending assembly they read.
// The window must stay small: the wait lands at the first node whose window reaches the reader, and a wide window (64)
// moved it to the start of the next layer, halving the overlap (LPF: 32.6 s of GPU idle in a 36K prefill). FZD showed the
// FZ run 2 mismatch was the slot count, not a race.
void ggml_cuda_lru_wait_srcs(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int last = std::min(cgraph->n_nodes, i + 8);
    for (int n = i; n < last && ctx.n_lru_pending > 0; ++n) {
        const ggml_tensor * node = cgraph->nodes[n];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            for (int k = 0; k < ggml_backend_cuda_context::LRU_MAX_PENDING; ++k) {
                const ggml_tensor * p = ctx.lru_pending[k];
                if (p != nullptr && (src == p || src->view_src == p)) {
                    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), ctx.lru_events[k], 0));
                    ctx.lru_pending[k] = nullptr;
                    ctx.n_lru_pending--;
                }
            }
        }
    }
}

void ggml_cuda_lru_drain(ggml_backend_cuda_context & ctx) {
    for (int k = 0; k < ggml_backend_cuda_context::LRU_MAX_PENDING && ctx.n_lru_pending > 0; ++k) {
        if (ctx.lru_pending[k] != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(ctx.device, 0), ctx.lru_events[k], 0));
            ctx.lru_pending[k] = nullptr;
            ctx.n_lru_pending--;
        }
    }
}


void ggml_cuda_lru_ctx_free(ggml_backend_cuda_context & ctx) {
    ggml_cuda_lru_stash_worker_stop(ctx);
    delete (ggml_cuda_lru_mirror *) ctx.lru_mirror;
    if (ctx.lru_wb_stream != nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaStreamSynchronize(ctx.lru_wb_stream));
        CUDA_CHECK(cudaStreamDestroy(ctx.lru_wb_stream));
        CUDA_CHECK(cudaEventDestroy(ctx.lru_wb_event));
        CUDA_CHECK(cudaEventDestroy(ctx.lru_fill_event));
    }
    if (ctx.lru_staging != nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaFree(ctx.lru_staging));
    }
    if (ctx.lru_stash_ctl != nullptr) {   // the flushes ran on ctx.lru_wb_stream (synchronized above)
        ggml_cuda_set_device(ctx.device);
        if (ctx.lru_stash_ce_pending) {    // copy-engine mode: the last pass's victims are still in the ring
            ggml_cuda_lru_stash_flush_ce(ctx);
            CUDA_CHECK(cudaStreamSynchronize(ctx.lru_wb_stream));
        }
        CUDA_CHECK(cudaFree(ctx.lru_stash_ctl));
        if (ctx.lru_stash_rec != nullptr) {
            if (ctx.lru_stash_hsnap != nullptr) {
                CUDA_CHECK(cudaFreeHost(ctx.lru_stash_rec));
            } else {
                CUDA_CHECK(cudaFree(ctx.lru_stash_rec));
            }
            CUDA_CHECK(cudaFree(ctx.lru_stash));
        }
        if (ctx.lru_stash_hsnap != nullptr) {
            CUDA_CHECK(cudaFreeHost(ctx.lru_stash_hsnap));
            CUDA_CHECK(cudaEventDestroy(ctx.lru_stash_ce_event));
        }
        CUDA_CHECK(cudaEventDestroy(ctx.lru_stash_main_event));
        CUDA_CHECK(cudaEventDestroy(ctx.lru_stash_event));
    }
    if (ctx.lru_counter != nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaFree(ctx.lru_counter));
    }
}
