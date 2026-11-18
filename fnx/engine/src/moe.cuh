// The routed experts in three launches (included by kernels.cu, after k_mmvq's helpers):
//   router + route: the logits, top-10 per token, their weights, and the job list (one job per distinct expert);
//   gate|up: every job's stacked [gate; up] rows for its tokens;
//   down: each block makes its job's silu(gate) * up rows as q8_1 in shared memory, then the down rows.
// Output rows: pair t*TOPK+k. The shared expert runs beside them as dense launches into rows T*TOPK+t; the weighted
// sum happens in the ring.
namespace {
#include "p100_dma_box.cuh"

// The router and the routing in one launch. logits rows of NEXP + 1: the router's, then the shared expert's gate,
// a warp per row. The last block to finish selects each token's top 10 by logit (softmax is monotonic; ties: the lower
// index), a warp per token, the weights the selected softmax renormalized (exp(l - l_max) over the ten), and builds
// the jobs; with an ExpertCache (ec.n_slots) its first warp then plans the layer's slots (cache_plan).
constexpr int RR_THREADS = 256, RR_WPR = 2;   // two warps per logit row
// router rows a warp by the window (split42; every logit's sum takes the same steps in the same order whatever the choice): 2, from
// 14 rows 4. A warp loads each window row's x chunk once for its rows (one row a warp re-read it for every row: ~83 MB from L2 at 16)
template <int T> constexpr int rr_rq() { return T >= 14 ? 4 : 2; }
template <int T> constexpr int rr_rows() { return RR_THREADS / 32 / RR_WPR * rr_rq<T>(); }

// one warp: the layer's jobs whose experts are in a slot keep it (ref 2: not to be taken); each other one takes the
// slot of the layer's pool that the pool's CLOCK hand reaches first with ref 0 (a ref 1 it passes drops to 0), its old
// expert pointing back at RAM. The hand goes round the pool's slots past its lent ones. list: how many took a slot,
// then their ids
__device__ void cache_plan(const Job* jobs, int nj, int layer, const ExpertCache& ec, int* list) {
    const int lane = threadIdx.x & 31, base = layer * NEXP, p = ec.pool_of[layer], lent = ec.lent[p];
    // an arena's pools hold every expert until it lends, and then only its own pool's experts can be missing
    if (ec.arena_full && (p != ec.arena_pool || lent == 0)) {
        if (lane == 0) list[0] = 0;
        return;
    }
    // FNX_PF_REFILL: the prompt path's last slots, back from a read but still lent, join the pool as the card's copier fills them: the first
    // fill[p] (the host's figure for this window, the same on every card, so the cards' slots stay alike). One whose expert came in by a
    // miss meanwhile joins empty
    if (ec.fill && ec.fill[2 * EC_POOLS + p] > 0) {
        const int cnt = ec.fill[2 * EC_POOLS + p], from = ec.fill[EC_POOLS + p], to = min(ec.fill[p], cnt);
        if (to > from) {
            const int t0 = ec.s0[p] + ec.n[p] - cnt;
            for (int i = from + lane; i < to; i += 32) {
                const int s = t0 + i, g = ec.fill_list[s];
                if (g >= 0 && ec.slot_of[g] < 0) {
                    char* at = ec.pool[p] + (size_t) (s - ec.s0[p]) * ec.slot_bytes[p];
                    ec.owner[s] = g;
                    ec.slot_of[g] = s;
                    ec.ep_gu[g] = at;
                    ec.ep_dn[g] = at + ec.dn_off[p];
                    ec.ref[s] = 1;
                } else {
                    ec.owner[s] = -1;
                    ec.ref[s] = 0;
                }
            }
            __syncwarp();
            if (lane == 0) {
                ec.fill[EC_POOLS + p] = to;
                ec.lent[EC_POOLS + p] = cnt - to;
                if (to == cnt) ec.fill[2 * EC_POOLS + p] = 0;
            }
            __syncwarp();
        }
    }
    const int s0 = ec.s0[p] + lent, n = ec.n[p] - lent - ec.lent[EC_POOLS + p];   // lent[EC_POOLS + p]: the pool's last slots, the prompt path's (FNX_PF_LEND)
    for (int j = lane; j < nj; j += 32)
        if (const int s = ec.slot_of[base + jobs[j].w]; s >= 0) ec.ref[s] = 2;
    __syncwarp();
    int hand = ec.hand[p] - s0, k = 0;   // within the slots not lent
    if (hand < 0 || hand >= n) hand = 0;
    if (n > 32) {
        // the layer's misses first (job order), then victims for up to 32 of them a window of 32 slots: the window's free slots in
        // order take the next misses, its ref-1 slots before the last one taken drop to 0, the hand moves past it, each lane does its
        // victim's bookkeeping (split41: the same victims, refs, hand and lists as one miss at a time, which waited on a few
        // dependent loads of lane 0 a miss)
        __shared__ int mlist[MAXT * TOPK];
        int m = 0;
        for (int j0 = 0; j0 < nj; j0 += 32) {
            const bool mi = j0 + lane < nj && ec.slot_of[base + jobs[j0 + lane].w] < 0;
            const unsigned miss = __ballot_sync(0xffffffffu, mi);
            if (mi) mlist[m + __popc(miss & ((1u << lane) - 1u))] = base + jobs[j0 + lane].w;
            m += __popc(miss);
        }
        __syncwarp();
        while (k < m) {
            const int idx = s0 + (hand + lane) % n;
            const unsigned char r = ec.ref[idx];
            const unsigned free = __ballot_sync(0xffffffffu, r == 0);
            const int take = min(__popc(free), m - k);
            if (take == 0) {
                if (r == 1) ec.ref[idx] = 0;
                hand = (hand + 32) % n;
                __syncwarp();
                continue;
            }
            unsigned fb = free;
            for (int q = 1; q < take; ++q) fb &= fb - 1u;
            const int last = __ffs(fb) - 1;
            const int rank = __popc(free & ((1u << lane) - 1u));
            if (lane < last && r == 1) ec.ref[idx] = 0;
            if ((free >> lane & 1u) && rank < take) {
                const int g = mlist[k + rank];
                if (const int o = ec.owner[idx]; o >= 0) {
                    ec.slot_of[o] = -1;
                    ec.ep_gu[o] = ec.host_gu[o];
                    ec.ep_dn[o] = ec.host_dn[o];
                }
                ec.owner[idx] = g;
                ec.slot_of[g] = idx;
                ec.ref[idx] = 2;
                char* at = ec.pool[p] + (size_t) (idx - ec.s0[p]) * ec.slot_bytes[p];
                ec.ep_gu[g] = at;
                ec.ep_dn[g] = at + ec.dn_off[p];
                list[1 + k + rank] = g;
            }
            hand = (hand + last + 1) % n;
            k += take;
            __syncwarp();
        }
    } else {
        // the misses found 32 jobs at a time (a load each, side by side), then taken in job order
        for (int j0 = 0; j0 < nj; j0 += 32) {
            unsigned miss = __ballot_sync(0xffffffffu, j0 + lane < nj && ec.slot_of[base + jobs[j0 + lane].w] < 0);
            while (miss) {
                const int g = base + jobs[j0 + __ffs(miss) - 1].w;
                miss &= miss - 1;
                int victim = -1;
                while (victim < 0) {
                    const int idx = s0 + (hand + lane) % n;
                    const unsigned char r = ec.ref[idx];
                    const unsigned free = __ballot_sync(0xffffffffu, r == 0);
                    const int f = free ? __ffs(free) - 1 : 32;
                    if (lane < f && r == 1) ec.ref[idx] = 0;
                    if (free) victim = s0 + (hand + f) % n;
                    hand = (hand + (free ? f + 1 : 32)) % n;
                }
                if (lane == 0) {
                    if (const int o = ec.owner[victim]; o >= 0) {
                        ec.slot_of[o] = -1;
                        ec.ep_gu[o] = ec.host_gu[o];
                        ec.ep_dn[o] = ec.host_dn[o];
                    }
                    ec.owner[victim] = g;
                    ec.slot_of[g] = victim;
                    ec.ref[victim] = 2;
                    char* at = ec.pool[p] + (size_t) (victim - ec.s0[p]) * ec.slot_bytes[p];
                    ec.ep_gu[g] = at;
                    ec.ep_dn[g] = at + ec.dn_off[p];
                    list[1 + k] = g;
                }
                ++k;
                __syncwarp();
            }
        }
    }
    for (int j = lane; j < nj; j += 32) ec.ref[ec.slot_of[base + jobs[j].w]] = 1;
    if (lane == 0) {
        ec.hand[p] = s0 + hand;
        list[0] = k;
    }
}

// FNX_RR_FAST (topk_patch.py): the top K of a warp's NEXP scores (lane + 32 i in s[i] of lane `lane`; those in `skip` left out) by score,
// ties to the lower index, as k_router_route's loop of warp maxima picks them: every top K score is at or above the K-th largest of the
// lanes' maxima, so those are the candidates, ranked against each other. out[r] = the index of rank r < K. False (warp-uniform):
// over RRF_CAP candidates or fewer than K finite scores
constexpr int RRF_CAP = 64;
__device__ unsigned rr_bad;   // FNX_RR_FAST=2: tokens whose fast picks or weights differed from the loop's
__device__ __forceinline__ float rr_sel16(const float (&a)[NEXP / 32], const int i) {
    float r = a[0];
#pragma unroll
    for (int q = 1; q < NEXP / 32; ++q) r = i == q ? a[q] : r;
    return r;
}
__device__ __forceinline__ bool rr_topk_fast(const float (&s)[NEXP / 32], const unsigned skip, const int K, const int lane, float* cs,
                                             int* ci, int* out) {
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NEXP / 32; ++i)
        if (!(skip >> i & 1u) && s[i] > m) m = s[i];
    float x = m;   // the lanes' maxima sorted in descending order across the warp (bitonic)
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1)
#pragma unroll
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            const float y = __shfl_xor_sync(0xffffffffu, x, stride);
            const bool desc = (lane & size) == 0, low = (lane & stride) == 0;
            x = low == desc ? fmaxf(x, y) : fminf(x, y);
        }
    const float th = __shfl_sync(0xffffffffu, x, K - 1);
    if (!(th > -INFINITY)) return false;
    int n = 0;
#pragma unroll
    for (int i = 0; i < NEXP / 32; ++i) {
        const bool c = !(skip >> i & 1u) && s[i] >= th;
        const unsigned b = __ballot_sync(0xffffffffu, c);
        const int pos = n + __popc(b & ((1u << lane) - 1u));
        if (c && pos < RRF_CAP) {
            cs[pos] = s[i];
            ci[pos] = lane + 32 * i;
        }
        n += __popc(b);
    }
    if (n > RRF_CAP) return false;
    __syncwarp();
    for (int j = lane; j < n; j += 32) {
        const float v = cs[j];
        const int id = ci[j];
        int r = 0;
        for (int q = 0; q < n; ++q) {
            const float w = cs[q];
            r += w > v || (w == v && ci[q] < id);
        }
        if (r < K) out[r] = id;
    }
    __syncwarp();
    return true;
}
// a token's picks into o[0..TOPK) and their weights' numerators into e[0..TOPK) (the warp's shared memory), z summed in pick order
__device__ __forceinline__ bool rr_fast_pick(const float (&v)[NEXP / 32], const float lam, const ExpertCache& ec, const int layer,
                                             const int lane, float* cs, int* ci, int* o, float* e, float& z) {
    if (!rr_topk_fast(v, 0u, TOPK, lane, cs, ci, o)) return false;
    if (lam > 0.f) {   // picks 0 and 1 stay the unraised top 2; 2..9 by the raised scores, as the loop's second pass
        const int a0 = o[0], a1 = o[1], a9 = o[TOPK - 1];
        const float b0 = __shfl_sync(0xffffffffu, rr_sel16(v, a0 >> 5), a0 & 31);
        const float b9 = __shfl_sync(0xffffffffu, rr_sel16(v, a9 >> 5), a9 & 31);
        const float up = lam * (b0 - b9);
        float vb[NEXP / 32];
#pragma unroll
        for (int i = 0; i < NEXP / 32; ++i) vb[i] = v[i] + (ec.slot_of[(size_t) layer * NEXP + lane + 32 * i] >= 0 ? up : 0.f);
        unsigned skip = 0;
        if ((a0 & 31) == lane) skip |= 1u << (a0 >> 5);
        if ((a1 & 31) == lane) skip |= 1u << (a1 >> 5);
        __syncwarp();
        if (!rr_topk_fast(vb, skip, TOPK - 2, lane, cs, ci, o + 2)) return false;
    }
    float top0 = 0.f;
    z = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        const int id = o[k];
        const float bo = __shfl_sync(0xffffffffu, rr_sel16(v, id >> 5), id & 31);
        if (k == 0) top0 = bo;
        const float ek = __expf(bo - top0);
        z += ek;
        if (lane == 0) e[k] = ek;
    }
    __syncwarp();
    return true;
}

// FNX_RR_TIMES (rrt_patch.py): the launch's stamps [first start, tail, top 10, jobs, plan, end] in a ring of 4096 launches
__device__ unsigned long long rr_tt[4096][8];   // [6]: the last block start, [7]: the last block done with its rows
__device__ unsigned rr_tn;
__device__ int rr_on;
__device__ __forceinline__ unsigned long long rr_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <int T, int RQ = rr_rq<T>()>   // RQ: FNX_RR_RQ4 (rr_rq4_patch.py) takes 4 at every T
__global__ void __launch_bounds__(RR_THREADS) k_router_route(const uint16_t* __restrict__ W, const float* __restrict__ x,
                                                            float* __restrict__ logits, int* __restrict__ topi,
                                                            float* __restrict__ topw, Job* __restrict__ gu, Job* __restrict__ dn,
                                                            int* __restrict__ njobs, unsigned* __restrict__ counter,
                                                            const ExpertCache ec, int layer, int* list, const ShexpUp su,
                                                            const RouteTail rt, const int fast) {
    static_assert(RR_THREADS == Q8S_THREADS, "the shared expert's q8s blocks in the router's launch");
    constexpr int NC = N / 4 / 32 / RR_WPR, L = NEXP + 1;   // float4 chunks per lane
    constexpr int RR_RQ = RQ, RR_ROWS = RR_THREADS / 32 / RR_WPR * RQ;
    const int nrr = (int) gridDim.x - su.blocks;              // the router's own blocks; the shared expert's after them
    const bool rr_t = rr_on != 0;
    if (rr_t && threadIdx.x == 0) {
        const unsigned long long t0 = rr_now();
        atomicMin(&rr_tt[rr_tn & 4095u][0], t0);
        if ((int) blockIdx.x < nrr) atomicMax(&rr_tt[rr_tn & 4095u][6], t0);
    }
    if ((int) blockIdx.x >= nrr) {
        shexp_up_block<T>(su, blockIdx.x - nrr);
        return;
    }
    const int e = threadIdx.x, warp = e >> 5, lane = e & 31, rw = warp / RR_WPR, half = warp % RR_WPR;
    // FNX_FOLD 2: the window's rows to f16 for the row kernels (k_f32_f16_rows's launch), a share a router block
    if (rt.xh)
        for (int i = (int) blockIdx.x * RR_THREADS + e; i < T * N; i += nrr * RR_THREADS) reinterpret_cast<__half*>(rt.xh)[i] = __float2half_rn(x[i]);
    // this warp's RR_RQ rows of its half (bf16 weights, four a lane per chunk: the bits of f32 with the low half zero), each x chunk
    // loaded once for them; every (row, lane) sum adds chunk i = 0..NC-1 in order, then the butterfly, then half 0 + half 1
    const int row0 = blockIdx.x * RR_ROWS + rw * RR_RQ;
    __shared__ float rsum[T][RR_ROWS][RR_WPR];
    {
        const uint2* w2[RR_RQ];
#pragma unroll
        for (int r = 0; r < RR_RQ; ++r) w2[r] = reinterpret_cast<const uint2*>(W + (size_t) min(row0 + r, L - 1) * N) + half * NC * 32;
        float acc[RR_RQ][T];
#pragma unroll
        for (int r = 0; r < RR_RQ; ++r)
#pragma unroll
            for (int t = 0; t < T; ++t) acc[r][t] = 0.f;
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            float4 wv[RR_RQ];
#pragma unroll
            for (int r = 0; r < RR_RQ; ++r) {
                const uint2 u = __ldg(w2[r] + i * 32 + lane);
                wv[r] = make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u));
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float4 b = (reinterpret_cast<const float4*>(x + (size_t) t * N) + half * NC * 32)[i * 32 + lane];
#pragma unroll
                for (int r = 0; r < RR_RQ; ++r) acc[r][t] += wv[r].x * b.x + wv[r].y * b.y + wv[r].z * b.z + wv[r].w * b.w;
            }
        }
#pragma unroll
        for (int r = 0; r < RR_RQ; ++r)
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float v = wsum(acc[r][t]);
                if (lane == 0) rsum[t][rw * RR_RQ + r][half] = v;
            }
    }
    __syncthreads();
    for (int q = e; q < T * RR_ROWS; q += RR_THREADS) {
        const int t = q / RR_ROWS, r = q % RR_ROWS, lrow = blockIdx.x * RR_ROWS + r;
        if (lrow < L)   // FNX_EXPERT_MASK (maska_patch.py): a dropped expert is never picked
            logits[(size_t) t * L + lrow] = lrow < NEXP && rt.keep && !(rt.keep[lrow >> 5] >> (lrow & 31) & 1u) ? -INFINITY : rsum[t][r][0] + rsum[t][r][1];
    }
    __shared__ bool last;
    if (rr_t && e == 0) atomicMax(&rr_tt[rr_tn & 4095u][7], rr_now());
    __threadfence();
    __syncthreads();
    if (e == 0) last = atomicAdd(counter, 1u) == (unsigned) nrr - 1;
    __syncthreads();
    if (!last) return;
    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][1] = rr_now();
    __threadfence();
    // top 10 per token: warp w takes tokens w, w + 8, ...; lane l holds experts l + 32 i
    __shared__ int ex[MAXT * TOPK];
    __shared__ float rf_s[RR_THREADS / 32][RRF_CAP], rf_e[RR_THREADS / 32][TOPK];   // FNX_RR_FAST: a warp's candidates, weights
    __shared__ int rf_i[RR_THREADS / 32][RRF_CAP], rf_o[RR_THREADS / 32][TOPK];
    static_assert(T * TOPK < RR_THREADS, "a thread per pick and one more (the job count) below");
#pragma unroll 1
    for (int t = warp; t < T; t += RR_THREADS / 32) {
        float v[NEXP / 32];
#pragma unroll
        for (int i = 0; i < NEXP / 32; ++i) v[i] = __ldcg(logits + (size_t) t * L + lane + 32 * i);
        // FNX_ROUTE_PRIOR (lossy): with lambda > 0, past the original top 2 the picks go by the logits raised by lambda * (top - 10th)
        // for the experts in a slot (Skliar et al. 2024); a pick's weight is still that of its original logit (bo)
        const float lam = ec.n_slots && ec.prior ? *ec.prior : 0.f;
        float fz = 0.f;
        const bool fok = fast && rr_fast_pick(v, lam, ec, layer, lane, rf_s[warp], rf_i[warp], rf_o[warp], rf_e[warp], fz);
        if (fok && fast == 1) {
            if (lane < TOPK) {
                const int id = rf_o[warp][lane];
                topi[t * TOPK + lane] = id;
                ex[t * TOPK + lane] = id;
                topw[t * TOPK + lane] = rf_e[warp][lane] / fz;
            }
            continue;
        }
        float vb[NEXP / 32];
        if (lam > 0.f) {
            unsigned tk = 0;
            float b0 = 0.f, b9 = 0.f;
#pragma unroll 1
            for (int k = 0; k < TOPK; ++k) {
                float bv = -INFINITY;
                int bi = NEXP;
#pragma unroll
                for (int i = 0; i < NEXP / 32; ++i)
                    if (!(tk >> i & 1u) && v[i] > bv) { bv = v[i]; bi = lane + 32 * i; }
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                if ((bi & 31) == lane) tk |= 1u << (bi >> 5);
                if (k == 0) b0 = bv;
                b9 = bv;
            }
            const float up = lam * (b0 - b9);
#pragma unroll
            for (int i = 0; i < NEXP / 32; ++i) vb[i] = v[i] + (ec.slot_of[(size_t) layer * NEXP + lane + 32 * i] >= 0 ? up : 0.f);
        }
        unsigned taken = 0;
        float top0 = 0.f, z = 0.f, wk = 0.f;
#pragma unroll 1
        for (int k = 0; k < TOPK; ++k) {
            const bool raised = lam > 0.f && k >= 2;
            float bv = -INFINITY, bo = 0.f;
            int bi = NEXP;
#pragma unroll
            for (int i = 0; i < NEXP / 32; ++i) {
                const float s = raised ? vb[i] : v[i];
                if (!(taken >> i & 1u) && s > bv) { bv = s; bo = v[i]; bi = lane + 32 * i; }
            }
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const float oo = __shfl_xor_sync(0xffffffffu, bo, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bo = oo; bi = oi; }
            }
            if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
            if (k == 0) top0 = bo;
            const float ek = __expf(bo - top0);
            z += ek;
            if (lane == k) {
                wk = ek;
                topi[t * TOPK + k] = bi;
                ex[t * TOPK + k] = bi;
            }
        }
        if (lane < TOPK) topw[t * TOPK + lane] = wk / z;
        if (fok) {   // FNX_RR_FAST=2: the fast path against the loop
            const bool bad = lane < TOPK && (rf_o[warp][lane] != ex[t * TOPK + lane] ||
                                             __float_as_uint(rf_e[warp][lane] / fz) != __float_as_uint(wk / z));
            if (__any_sync(0xffffffffu, bad) && lane == 0) atomicAdd(&rr_bad, 1u);
        }
    }
    __syncthreads();
    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][2] = rr_now();
    // jobs: one per distinct expert, numbered in the order of each expert's first pick; a pair takes the next slot of
    // its expert's job (the order of a job's rows does not matter)
    constexpr int P = T * TOPK;
    __shared__ int firstpair[NEXP], slots[NEXP], jobof[NEXP];
    __shared__ unsigned char first[MAXT * TOPK];
    for (int x2 = e; x2 < NEXP; x2 += RR_THREADS) {
        firstpair[x2] = P;
        slots[x2] = 0;
    }
    __syncthreads();
    if (e < P) atomicMin(&firstpair[ex[e]], e);
    __syncthreads();
    if (e < P) first[e] = firstpair[ex[e]] == e;
    __syncthreads();
    // a first pick's job number from its warp's ballot and the warps' counts before it (split41: the numbers a loop over first[] gave)
    __shared__ int wcnt[RR_THREADS / 32];
    {
        const bool fp = e < P && first[e];
        const unsigned b = __ballot_sync(0xffffffffu, fp);
        if (lane == 0) wcnt[warp] = __popc(b);
        __syncthreads();
        if (fp) {
            int j = __popc(b & ((1u << lane) - 1u));
            for (int w = 0; w < warp; ++w) j += wcnt[w];
            jobof[ex[e]] = j;
            gu[j].w = dn[j].w = ex[e];
        }
    }
    __syncthreads();
    if (e < P) {
        const int j = jobof[ex[e]], s = atomicAdd(&slots[ex[e]], 1);
        gu[j].y[s] = e / TOPK;
        gu[j].out[s] = dn[j].y[s] = dn[j].out[s] = e;
    }
    __syncthreads();
    if (e < P && first[e]) gu[jobof[ex[e]]].n = dn[jobof[ex[e]]].n = slots[ex[e]];
    __shared__ int nj;
    if (e == P) {
        int n = 0;
        for (int w = 0; w < RR_THREADS / 32; ++w) n += wcnt[w];
        njobs[0] = nj = n;
        *counter = 0u;
    }
    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][3] = rr_now();
    if (!ec.n_slots) return;
    __syncthreads();
    if (warp == 0) cache_plan(gu, nj, layer, ec, list);
    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][4] = rr_now();
    if (!rt.box) return;
    // FNX_FOLD 1: the list to the copy engine (k_dma_publish) and the listed experts' jobs marked (k_jobs_split), launches of their own before
    __syncthreads();
    const int k = list[0];
    if (warp == 0) {
        DmaBox* box = (DmaBox*) rt.box;
        DmaDev* dv = (DmaDev*) rt.dev;
        const int n = min(k, DMA_MAX), p = ec.pool_of[layer];
        for (int q = lane; q < n; q += 32) {
            const int g = list[1 + q];
            char* at = ec.pool[p] + (size_t) (ec.slot_of[g] - ec.s0[p]) * ec.slot_bytes[p];
            box->ent[q] = DmaEnt{ec.host_gu[g], at, ec.host_dn[g], at + ec.dn_off[p]};
        }
        __syncwarp();
        if (lane == 0) {
            if (n == 0) dv->want = 0;
            else {
                box->n = n;
                box->gu_bytes = rt.gu_bytes;
                box->dn_bytes = rt.dn_bytes;
                const unsigned sq = ++dv->seq;
                dv->want = sq;
                __threadfence_system();
                box->seq = sq;
                __threadfence_system();
            }
        }
    }
    if (e < nj && k) {
        const int g = layer * NEXP + gu[e].w;
        for (int i = 1; i <= k; ++i)
            if (list[i] == g) {
                gu[e].w = ~gu[e].w;
                dn[e].w = ~dn[e].w;
                break;
            }
    }
    if (rr_t) {
        __syncthreads();
        if (e == 0) {
            const unsigned n = rr_tn & 4095u;
            rr_tt[n][5] = rr_now();
            rr_tt[(n + 1) & 4095u][0] = ~0ull;
            rr_tt[(n + 1) & 4095u][6] = rr_tt[(n + 1) & 4095u][7] = 0ull;
            rr_tn = rr_tn + 1;
        }
    }
}

// rows of one job: tmp[j] = W row . y[ys[j]] for j < n <= TT, G lanes per row
template <ggml_type type, int G, int TT, bool X = false>
__device__ __forceinline__ void job_rows(const char* wr, bool live, int ncols, int n, const block_q8_1* y, const int* ys,
                                         int y_stride, float* tmp) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = vdr_of<type>();
    constexpr int per_block = qi / vdr;
    static_assert(G >= per_block && G % per_block == 0, "G must cover a block");
    constexpr int bpi = G / per_block;
    const int gl = threadIdx.x % G, bpr = ncols / qk;
    if (live)
#pragma unroll 8
        for (int kbx = gl / per_block; kbx < bpr; kbx += bpi) {
            const int kby = kbx * (qk / QK8_1), kqs = vdr * (gl % per_block);
#pragma unroll
            for (int j = 0; j < TT; ++j)
                if (j < n) tmp[j] += vdot<type, X>(wr, y + (size_t) ys[j] * y_stride + kby, kbx, kqs);
        }
}

// TT: the most rows a job has in this launch (the window's T): the loop unrolls that far, not to MAXT
template <int G, int TT = MAXT>
__device__ __forceinline__ void job_store(const float* tmp, int n, bool live, int row, const int* outs, float* out, int out_stride) {
    const int gl = threadIdx.x % G;
#pragma unroll
    for (int j = 0; j < TT; ++j) {
        if (j >= n) break;
        float v = tmp[j];
#pragma unroll
        for (int o = G / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (live && gl == 0) out[(size_t) outs[j] * out_stride + row] = v;
    }
}

// one thread's dot of U columns of a weight row (its unit `u`) with one q8_1 activation row
template <ggml_type type, int U, bool X = false>
__device__ __forceinline__ float unit_dot(const char* wrow, int u, const block_q8_1* y) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk, qi = ggml_cuda_type_traits<type>::qi, vdr = vdr_of<type>();
    float s = 0.f;
    if constexpr (U % qk == 0) {
#pragma unroll
        for (int b = 0; b < U / qk; ++b) {
            const int kbx = u * (U / qk) + b;
#pragma unroll
            for (int i = 0; i < qi / vdr; ++i) s += vdot<type, X>(wrow, y + kbx * (qk / QK8_1), kbx, vdr * i);
        }
    } else {
        __trap();   // a type whose blocks are wider than the unit: the launcher refuses it
    }
    return s;
}

#include "moe_w.cuh"

// gate|up: rows [0, 2 * ffs) of each job [0, *njobs)
template <ggml_type TA, int G, int TT, bool X = false>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                       const block_q8_1* __restrict__ y, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ Job jb;
    if (threadIdx.x == 0) jb = jobs[ji];
    __syncthreads();
    const int row = blockIdx.x * (MMVQ_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
    float tmp[TT] = {};
    job_rows<TA, G, TT, X>(moe_expert(w, jb.w) + (size_t) r * w.row_a, live, w.ncols, jb.n, y, jb.y, w.ncols / 32, tmp);
    job_store<G, TT>(tmp, jb.n, live, row, jb.out, out, w.nrows);
}

// down: one block per (job, rows_per_block rows). The block first makes its tokens' h = silu(gate) * up (gu rows
// jb.y[j] of [gate (ffs) | up (ffs)]) as q8_1 in shared memory; thread (row, u) then dots its 32-column unit u of the
// row with each token's h, and the block sums a row's units into out row jb.out[j]
constexpr int DN_THREADS = 256, DN_U = 32, DN_MAXB = FF_CARD_MAX / 32 > 8 ? FF_CARD_MAX / 32 : 8;   // DN_MAXB: ffs / 32 on any card

// every pair's h for k_moe_down, made once by k_moe_h (each of the ~70 down blocks of a job made its own). One MoE runs at
// a time on a card (its window's stream), so one buffer per card
__device__ block_q8_1 g_moe_hq[MAXT * TOPK][DN_MAXB];

// h = silu(gate) * up of pair row e of gu ([gate (ffs) | up (ffs)]) as q8_1, a warp per (pair, 32 columns)
__global__ void k_moe_h(const float* __restrict__ gu, int ffs, int npairs) {
    const int nb = ffs / DN_U, q = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (q >= npairs * nb) return;
    const int e = q / nb, b = q % nb;
    const float* g = gu + (size_t) e * 2 * ffs;
    const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
    const float amax = wmax(fabsf(xi)), sm = wsum(xi);
    const float d = amax / 127.0f;
    block_q8_1& h = g_moe_hq[e][b];
    h.qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
    if (lane == 0) h.ds = make_half2(__float2half(d), __float2half(sm));
}

// PRE: the pairs' h made by k_moe_h (past 8 rows: kbench 16 rows 82.9 -> 70.7 us); else each block makes its own (the
// extra launch costs more than it saves: 5 rows 30.7 -> 31.7)
template <ggml_type TA, bool PRE, bool X = false>
__global__ void __launch_bounds__(DN_THREADS) k_moe_down(MoeW w, const Job* __restrict__ jobs, const int* __restrict__ njobs,
                                                        const float* __restrict__ gu, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ Job jb;
    __shared__ block_q8_1 hq[MAXT][DN_MAXB];
    extern __shared__ float dn_part[];   // [the window's T][DN_THREADS] (launched with that much): MAXT's would halve the
    float(*part)[DN_THREADS] = reinterpret_cast<float(*)[DN_THREADS]>(dn_part);   // blocks an SM holds
    if (threadIdx.x == 0) jb = jobs[ji];
    __syncthreads();
    const int ffs = w.ncols, nb = ffs / DN_U, n = jb.n;
    if constexpr (PRE) {
        static_assert(sizeof(block_q8_1) == 36, "a q8_1 block is 9 words");
        for (int q = threadIdx.x; q < n * nb * 9; q += DN_THREADS) {
            const int jb9 = q / 9, j = jb9 / nb, b = jb9 % nb;
            reinterpret_cast<int*>(&hq[j][b])[q % 9] = reinterpret_cast<const int*>(&g_moe_hq[jb.y[j]][b])[q % 9];
        }
    } else {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int q = warp; q < n * nb; q += DN_THREADS / 32) {
            const int j = q / nb, b = q % nb;
            const float* g = gu + (size_t) jb.y[j] * 2 * ffs;
            const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
            const float amax = wmax(fabsf(xi)), sm = wsum(xi);
            const float d = amax / 127.0f;
            hq[j][b].qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
            if (lane == 0) hq[j][b].ds = make_half2(__float2half(d), __float2half(sm));
        }
    }
    __syncthreads();
    const int rpb = DN_THREADS / nb, r = threadIdx.x / nb, u = threadIdx.x % nb;
    const int row = blockIdx.x * rpb + r;
    const bool live = r < rpb && row < w.nrows;
    if (live) {
        const char* wr = moe_expert(w, jb.w) + (size_t) row * w.row_a;
        for (int j = 0; j < n; ++j) part[j][threadIdx.x] = unit_dot<TA, DN_U, X>(wr, u, hq[j]);
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n * rpb; e += DN_THREADS) {
        const int j = e / rpb, rr = e % rpb, orow = blockIdx.x * rpb + rr;
        if (orow >= w.nrows) continue;
        float s = 0.f;
        for (int k = 0; k < nb; ++k) s += part[j][rr * nb + k];
        out[(size_t) jb.out[j] * w.nrows + orow] = s;
    }
}

template <ggml_type type> constexpr int per_block_of() { return ggml_cuda_type_traits<type>::qi / vdr_of<type>(); }
template <ggml_type type> constexpr bool unit_ok(int U) { return U % ggml_cuda_type_traits<type>::qk == 0; }

// lanes per row: enough to keep every block of a row busy (as mmvq picks them)
template <ggml_type TA> int moe_g(int ncols) {
    constexpr int pa = per_block_of<TA>();
    const int lanes = (ncols / ggml_cuda_type_traits<TA>::qk) * pa;
    int G = pa;
    while (G < lanes && G < 32) G *= 2;
    return G;
}

// k_moe_gu for BigJobs: the job's pairs MAXT at a time through the row
template <ggml_type TA, int G>
__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                           const block_q8_1* __restrict__ y, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    load_bigjob(jb, jobs + ji);
    const int row = blockIdx.x * (MMVQ_THREADS / G) + threadIdx.x / G;
    const bool live = row < w.nrows;
    const int r = live ? row : 0;
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int nb = min(MAXT, jb.n - b0);
        float tmp[MAXT] = {};
        job_rows<TA, G, MAXT>(moe_expert(w, jb.w) + (size_t) r * w.row_a, live, w.ncols, nb, y, jb.y + b0, w.ncols / 32, tmp);
        job_store<G>(tmp, nb, live, row, jb.out + b0, out, w.nrows);
    }
}

// k_moe_down for BigJobs: per MAXT pairs, their h into shared memory, the dots, the sums
template <ggml_type TA>
__global__ void __launch_bounds__(DN_THREADS) k_moe_down_big(MoeW w, const BigJob* __restrict__ jobs, const int* __restrict__ njobs,
                                                            const float* __restrict__ gu, float* __restrict__ out) {
    const int ji = blockIdx.y;
    if (ji >= *njobs) return;
    __shared__ BigJob jb;
    __shared__ block_q8_1 hq[MAXT][DN_MAXB];
    __shared__ float part[MAXT][DN_THREADS];
    load_bigjob(jb, jobs + ji);
    const int ffs = w.ncols, nb = ffs / DN_U, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rpb = DN_THREADS / nb, r = threadIdx.x / nb, u = threadIdx.x % nb;
    const int row = blockIdx.x * rpb + r;
    const bool live = r < rpb && row < w.nrows;
    const char* wr = moe_expert(w, jb.w) + (size_t) (live ? row : 0) * w.row_a;
#pragma unroll 1
    for (int b0 = 0; b0 < jb.n; b0 += MAXT) {
        const int n = min(MAXT, jb.n - b0);
        for (int q = warp; q < n * nb; q += DN_THREADS / 32) {
            const int j = q / nb, b = q % nb;
            const float* g = gu + (size_t) jb.y[b0 + j] * 2 * ffs;
            const float xi = silu_(g[b * 32 + lane]) * g[ffs + b * 32 + lane];
            const float amax = wmax(fabsf(xi)), sm = wsum(xi);
            const float d = amax / 127.0f;
            hq[j][b].qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
            if (lane == 0) hq[j][b].ds = make_half2(__float2half(d), __float2half(sm));
        }
        __syncthreads();
        if (live)
            for (int j = 0; j < n; ++j) part[j][threadIdx.x] = unit_dot<TA, DN_U>(wr, u, hq[j]);
        __syncthreads();
        for (int e = threadIdx.x; e < n * rpb; e += DN_THREADS) {
            const int j = e / rpb, rr = e % rpb, orow = blockIdx.x * rpb + rr;
            if (orow >= w.nrows) continue;
            float s = 0.f;
            for (int k = 0; k < nb; ++k) s += part[j][rr * nb + k];
            out[(size_t) jb.out[b0 + j] * w.nrows + orow] = s;
        }
        __syncthreads();
    }
}

template <ggml_type TA>
void moe_launch_big(bool down, const MoeW& w, const BigJob* jobs, const int* njobs, int njobs_cap, const void* in, float* out,
                    cudaStream_t s) {
    if (down) {
        if (!unit_ok<TA>(DN_U) || w.ncols / DN_U > DN_MAXB)
            throw std::runtime_error("moe_down: shape or type does not fit the units");
        const int rpb = DN_THREADS / (w.ncols / DN_U);
        k_moe_down_big<TA><<<dim3((w.nrows + rpb - 1) / rpb, njobs_cap), DN_THREADS, 0, s>>>(w, jobs, njobs, (const float*) in, out);
        return;
    }
    const int G = moe_g<TA>(w.ncols);
#define FNX_MB(g)                                                                                                        \
    {                                                                                                                    \
        const dim3 grid((w.nrows + MMVQ_THREADS / g - 1) / (MMVQ_THREADS / g), njobs_cap);                              \
        k_moe_gu_big<TA, g><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, (const block_q8_1*) in, out);                 \
        return;                                                                                                          \
    }
    if constexpr (per_block_of<TA>() <= 4) { if (G == 4) FNX_MB(4) }
    if constexpr (per_block_of<TA>() <= 8) { if (G == 8) FNX_MB(8) }
    if constexpr (per_block_of<TA>() <= 16) { if (G == 16) FNX_MB(16) }
    FNX_MB(32)
#undef FNX_MB
}

// idea 67: IQ3_S's writing kernels with the dots in the xmad form (iq3s_x.cuh). FNX_MOE_X: bit T: the gate|up kernel of T rows
// of window; FNX_MOE_XD: 1: the down kernel to 8 rows of window, 2: past 8
#ifdef FNX_DP4A_FORMS
inline unsigned moe_x() { static const unsigned m = (unsigned) env_int("FNX_MOE_X", 0); return m; }
inline unsigned moe_xd() { static const unsigned m = (unsigned) env_int("FNX_MOE_XD", 0); return m; }
#endif
template <ggml_type TA, bool PRE>
void dn_launch(dim3 grid, size_t smem, cudaStream_t s, const MoeW& w, const Job* jobs, const int* njobs, const float* gu,
               float* out) {
#ifdef FNX_DP4A_FORMS
    if constexpr (TA == GGML_TYPE_IQ3_S) {
        if (moe_xd() & (PRE ? 2u : 1u)) {
            k_moe_down<TA, PRE, true><<<grid, DN_THREADS, smem, s>>>(w, jobs, njobs, gu, out);
            return;
        }
    }
#endif
    k_moe_down<TA, PRE><<<grid, DN_THREADS, smem, s>>>(w, jobs, njobs, gu, out);
}

// the gate|up kernel for a window of T rows (a job holds at most T)
template <ggml_type TA, int G>
void gu_launch(int T, dim3 grid, cudaStream_t s, const MoeW& w, const Job* jobs, const int* njobs, const block_q8_1* y, float* out) {
#ifdef FNX_DP4A_FORMS
    if constexpr (TA == GGML_TYPE_IQ3_S) {
        if (T >= 1 && T <= 16 && ((moe_x() >> T) & 1u)) {
            switch (T) {
                case 1: k_moe_gu<TA, G, 1, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 2: k_moe_gu<TA, G, 2, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 3: k_moe_gu<TA, G, 3, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 4: k_moe_gu<TA, G, 4, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 5: k_moe_gu<TA, G, 5, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 6: k_moe_gu<TA, G, 6, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 7: k_moe_gu<TA, G, 7, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 8: k_moe_gu<TA, G, 8, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 9: k_moe_gu<TA, G, 9, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 10: k_moe_gu<TA, G, 10, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 11: k_moe_gu<TA, G, 11, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 12: k_moe_gu<TA, G, 12, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 13: k_moe_gu<TA, G, 13, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 14: k_moe_gu<TA, G, 14, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 15: k_moe_gu<TA, G, 15, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                case 16: k_moe_gu<TA, G, 16, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;
                default: break;
            }
        }
    }
#endif
    switch (T) {
        case 1: k_moe_gu<TA, G, 1><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 2: k_moe_gu<TA, G, 2><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 3: k_moe_gu<TA, G, 3><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 4: k_moe_gu<TA, G, 4><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 5: k_moe_gu<TA, G, 5><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 6: k_moe_gu<TA, G, 6><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 7: k_moe_gu<TA, G, 7><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 8: k_moe_gu<TA, G, 8><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 9: k_moe_gu<TA, G, 9><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 10: k_moe_gu<TA, G, 10><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 11: k_moe_gu<TA, G, 11><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 12: k_moe_gu<TA, G, 12><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 13: k_moe_gu<TA, G, 13><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 14: k_moe_gu<TA, G, 14><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 15: k_moe_gu<TA, G, 15><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        case 16: k_moe_gu<TA, G, 16><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); break;
        default: throw std::runtime_error("moe gate|up: T out of range");
    }
}

template <ggml_type TA>
void moe_launch(bool down, const MoeW& w, const Job* jobs, const int* njobs, int njobs_max, const void* in, float* out,
                cudaStream_t s, int T) {
    if (down) {
        if (!unit_ok<TA>(DN_U) || w.ncols / DN_U > DN_MAXB)
            throw std::runtime_error("moe_down: shape or type does not fit the units");
        const int rpb = DN_THREADS / (w.ncols / DN_U), nh = njobs_max * (w.ncols / DN_U);   // njobs_max: the window's pairs
        const dim3 grid((w.nrows + rpb - 1) / rpb, njobs_max);
        const size_t smem = sizeof(float) * T * DN_THREADS;
        if (T > 8) {
            if (njobs_max > MAXT * TOPK) throw std::runtime_error("moe_down: more pairs than g_moe_hq holds");
            k_moe_h<<<(nh + 7) / 8, 256, 0, s>>>((const float*) in, w.ncols, njobs_max);
            dn_launch<TA, true>(grid, smem, s, w, jobs, njobs, (const float*) in, out);
        } else {
            dn_launch<TA, false>(grid, smem, s, w, jobs, njobs, (const float*) in, out);
        }
        return;
    }
    // gate|up rows have N columns, where every type's lanes per row come to 32: only that variant is compiled (the
    // others, never used, were 192 of the file's kernels). Q4_K's gate|up has a kernel of its own (moe_q4k.cuh)
    if constexpr (TA == GGML_TYPE_Q4_K) {
        throw std::runtime_error("moe gate|up: Q4_K runs gu_q4k_launch");
    } else {
        if (moe_g<TA>(w.ncols) != 32) throw std::runtime_error("moe gate|up: only 32 lanes a row are compiled (rows of N columns)");
        const dim3 grid((w.nrows + MMVQ_THREADS / 32 - 1) / (MMVQ_THREADS / 32), njobs_max);
        gu_launch<TA, 32>(T, grid, s, w, jobs, njobs, (const block_q8_1*) in, out);
    }
}


}  // namespace
