#include "ar-rank.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>

// p100-series: two-GPU AllReduce launched PER DEVICE, for the meta backend's threaded dispatch (one host thread per device).
//
// The threaded path's own AllReduce (peer copy + ADD) needs two host barriers per reduction: the copy may only be enqueued once
// the peer's partial is enqueued, the ADD only once the peer's pull is enqueued. On gemma4 decode the GPU then idles ~37 us per
// reduction waiting for the host threads (60 reductions per token). Here each device's thread enqueues one kernel and moves on;
// the GPUs synchronize among themselves:
//   phase 1: write my partial into the PEER's inbox (P2P stores into its device memory)
//   phase 2: thread 0 of each block writes the call token into the peer's arrival slot, then spins on its OWN slot (local read)
//   phase 3: sum = mine + inbox (my device memory), in place
// Both devices compute every element (x0 + x1 on device 0, x1 + x0 on device 1: the same IEEE result), exactly what the copy +
// ADD path computes. Two inbox / arrival slots alternate by call parity: a device can only start call N after it saw the peer's
// token for N - 1, which the peer writes after its call N - 2 kernel (the last reader of slot N % 2) has completed.
// Based on ggml-cuda/allreduce.cu's chunked kernel (pinned-host staging, sm70+ for __nanosleep); this one is P2P-only and spins
// with a plain volatile read (Pascal).

static constexpr int    AR_RANK_SLOTS        = 2;
static constexpr int    AR_RANK_MAX_BLOCKS   = 8;    // blocks for a small reduction (decode: latency-bound)
static constexpr int    AR_RANK_ALLOC_BLOCKS = 128;  // arrival flags allocated per slot (large reductions use more blocks)
// a reduction above 1 MiB (prefill activations with GGML_META_AR_MAX_BYTES raised) is bandwidth-bound: up to
// GGML_CUDA_AR_BLOCKS_LARGE blocks (default 64) stripe the P2P writes and the sum
static int ar_blocks_large() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_AR_BLOCKS_LARGE"); return e ? std::max(1, std::min(AR_RANK_ALLOC_BLOCKS, atoi(e))) : 64; }();
    return v;
}
static constexpr int    AR_RANK_THREADS      = 256;
static constexpr size_t AR_RANK_STRIDE       = 64; // bytes between arrival ints (one cache line each)

// p100-series AllReduce profiler (GGML_CUDA_AR_PROF=<path prefix>, exp/ar-prof): block 0 / thread 0 of every AllReduce kernel records
// SM clock64 at  0 entry, 1 stores issued (block barrier), 2 system fence done, 3 peer token seen, 4 exit  (LL path: 1 = 2 = stores
// issued, 3 = all peer words received) + %globaltimer at entry, into a per-rank ring indexed by call token; dumped to
// <prefix>.<pid>.<n>.csv when the context is freed. The extra block barrier before the fence exists only in profiling runs.
static constexpr uint32_t AR_PROF_RING = 1u << 16;
static __device__ __forceinline__ unsigned long long ar_prof_gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
static __device__ __forceinline__ void ar_prof_mark(unsigned long long * prof, const int token, const int k) {
    unsigned long long * e = prof + (size_t) ((uint32_t) token & (AR_PROF_RING - 1)) * 8;
    e[k] = (unsigned long long) clock64();
    if (k == 0) {
        e[5] = ar_prof_gtime();
    }
}

// LL ("low latency", as NCCL's LL protocol): every 8-byte word carries one float and the call token. An aligned 8-byte store becomes
// visible at the peer as a unit, so the receiver polls the data words themselves: no system fence and no separate arrival flag
// (the signal path costs ~3 us of the ~10 us per decode reduction on P100 / PCIe). Stale words hold older tokens, never equal to the
// current one; slot reuse is safe by the same parity argument as the flag path. Taken for n <= AR_LL_MAX_ELEMS (decode and small
// batches); GGML_CUDA_AR_LL=0 disables it.
static constexpr int AR_LL_MAX_ELEMS = 32768;
static __device__ __forceinline__ void ar_ll_put(uint2 * p, const float v, const int token) {
    asm volatile("st.volatile.global.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(__float_as_uint(v)), "r"(token) : "memory");
}
// two LL words in one 16-byte store: {v0, token, v1, token} is exactly the layout of words 2k and 2k+1. A 16-byte store is not
// guaranteed to land as a unit, but each 8-byte half carries its own token, which is all the reader relies on.
static __device__ __forceinline__ void ar_ll_put2(uint2 * p, const float v0, const float v1, const int token) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1, %2, %3, %2};" :: "l"(p), "r"(__float_as_uint(v0)), "r"(token), "r"(__float_as_uint(v1)) : "memory");
}
static __device__ __forceinline__ float2 ar_ll_get2(const uint2 * p, const int token) {
    unsigned x0, y0, x1, y1;
    do {
        asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x0), "=r"(y0), "=r"(x1), "=r"(y1) : "l"(p) : "memory");
    } while (y0 != (unsigned) token || y1 != (unsigned) token);
    return make_float2(__uint_as_float(x0), __uint_as_float(x1));
}
static __device__ __forceinline__ float ar_ll_get(const uint2 * p, const int token) {
    unsigned x, y;
    do {
        asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "l"(p) : "memory");
    } while (y != (unsigned) token);
    return __uint_as_float(x);
}
// p100-series 10-25: compressed wire for LARGE reductions (prefill activations, > 1 MiB, GGML_META_AR_MAX_BYTES raised): the PCIe P2P
// bytes are the cost there (gemma4 prefill: 20 % of GPU time in AllReduce copies; the in-kernel path at f32 measured even with the peer
// copy), so send 16 bits per element. GGML_CUDA_AR_WIRE: 2 = default, 1 = bf16 (fp32 range, 8-bit mantissa), 0 = f32; 2 = fp16 with a
// power-of-two scale per chunk / row (11-bit mantissa; the scale keeps gemma4's large activations in range). Both devices must hold the
// SAME sum (the next subgraph reads mirrored activations): each device writes its own partial back ROUNDED the same way before adding the
// peer's decoded one, so both compute round(x0) + round(x1). Decode and small reductions (<= 1 MiB, LL path) are not touched.
// Phases AW1-AW3: gemma4 prefill +10..13 % (serving 1K / 4K / 11.5K prompts +13 / +12 / +11 %), qwen36 +7 %, Swift +7 %, glimmer +10 %;
// KLD gemma4 16-chunk +6 % on average over -ub 512 / 1024 / 1536 (inside the batching noise), ornith even, qwen36 better, Swift / glimmer
// bit-identical (their q6_K prefill matmuls already produce fp16 values, which the scaled wire carries exactly).
static int ar_wire() {
    static const int w = [] { const char * e = getenv("GGML_CUDA_AR_WIRE"); return e ? std::max(0, std::min(2, atoi(e))) : 0; }();   // p100-optimizations: exact fp32 wire by default (16-bit wire changes prefill sums)
    return w;
}
static __device__ __forceinline__ uint32_t ar_bf16_enc(const float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7FFFu + ((u >> 16) & 1u);   // round to nearest even (activations are finite)
    return u >> 16;
}
static __device__ __forceinline__ float ar_bf16_dec(const uint32_t b) {
    return __uint_as_float(b << 16);
}
// scale for a chunk / row with absolute maximum m: values * inv land below 2^15 (fp16 max 65504), power of two = exact
static __device__ __forceinline__ void ar_f16_scale(const float m, float & inv, float & scale) {
    int e = 0;
    if (m > 0.0f) {
        frexpf(m, &e);                 // m = f * 2^e, f in [0.5, 1)
    }
    const int sh = max(-30, min(30, 15 - e));
    inv   = ldexpf(1.0f, sh);
    scale = ldexpf(1.0f, -sh);
}
// 16-bit values at the start of the slot, the fp32 scales (WIRE 2) after them
static __host__ __device__ __forceinline__ size_t ar_wire_scales_off(const int64_t n) {
    return ((size_t) n*2 + 255) & ~(size_t) 255;
}

struct ar_prof_meta { uint32_t token; uint8_t kind; int32_t n; int32_t nrows; };

struct ggml_cuda_ar_rank_ctx {
    int      devices[2];
    size_t   max_bytes;   // largest reduction taken
    size_t   slot_bytes;  // inbox bytes per slot: max_bytes at f32, about half with the 16-bit wire (reductions > 1 MiB)
    char   * inbox[2];    // device i: AR_RANK_SLOTS * slot_bytes, written by the peer
    char   * arrival[2];  // device i: AR_RANK_SLOTS * AR_RANK_ALLOC_BLOCKS * AR_RANK_STRIDE, written by the peer
    uint32_t calls[2];    // per rank; both ranks issue the same sequence of reductions
    unsigned long long * prof[2] = {nullptr, nullptr};  // GGML_CUDA_AR_PROF: device ring per rank
    uint2 * ll[2] = {nullptr, nullptr};                 // device i: AR_RANK_SLOTS * AR_LL_MAX_ELEMS LL words, written by the peer
    std::vector<ar_prof_meta> prof_meta[2];
};

static __global__ void ggml_cuda_ar_rank_kernel(float * data, const int n, float * __restrict__ wire_out, const float * __restrict__ wire_in,
                                                int * signal_out, const int * signal_in, const int token,
                                                unsigned long long * prof) {
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const bool pr  = prof != nullptr && bid == 0 && tid == 0;
    if (pr) {
        ar_prof_mark(prof, token, 0);
    }
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;
    const int n4   = n / 4;
    const bool vec = ((uintptr_t) data & 15) == 0;

    // phase 1: my partial -> peer inbox (groups of 4 elements striped over the grid; tail by block 0)
    for (int i = gtid; i < n4; i += gnt) {
        float4 v;
        if (vec) {
            v = ((const float4 *) data)[i];
        } else {
            v = make_float4(data[4*i + 0], data[4*i + 1], data[4*i + 2], data[4*i + 3]);
        }
        ((float4 *) wire_out)[i] = v;
    }
    if (bid == 0 && tid < n - 4*n4) {
        wire_out[4*n4 + tid] = data[4*n4 + tid];
    }
    if (prof != nullptr) {
        __syncthreads();
        if (pr) {
            ar_prof_mark(prof, token, 1);
        }
    }
    __threadfence_system();
    __syncthreads();

    // phase 2: per-block handshake (the peer's block bid reads exactly the stripe our block bid wrote)
    if (tid == 0) {
        if (pr) {
            ar_prof_mark(prof, token, 2);
        }
        *(volatile int *) (signal_out + bid * (AR_RANK_STRIDE / sizeof(int))) = token;
        __threadfence_system();
        const volatile int * in = (const volatile int *) (signal_in + bid * (AR_RANK_STRIDE / sizeof(int)));
        while (*in != token) {
        }
        if (pr) {
            ar_prof_mark(prof, token, 3);
        }
    }
    __syncthreads();
    __threadfence();

    // phase 3: in-place sum with the peer's partial
    for (int i = gtid; i < n4; i += gnt) {
        const float4 o = __ldcg((const float4 *) wire_in + i);
        if (vec) {
            float4 v = ((float4 *) data)[i];
            v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
            ((float4 *) data)[i] = v;
        } else {
            data[4*i + 0] += o.x; data[4*i + 1] += o.y; data[4*i + 2] += o.z; data[4*i + 3] += o.w;
        }
    }
    if (bid == 0 && tid < n - 4*n4) {
        data[4*n4 + tid] += __ldcg(wire_in + 4*n4 + tid);
    }
    if (prof != nullptr) {
        __syncthreads();
        if (pr) {
            ar_prof_mark(prof, token, 4);
        }
    }
}

// LL variant of ggml_cuda_ar_rank_kernel: same sum and result. V4: pairs of elements per 16-byte store / load (n even).
template <bool V4>
static __global__ void ggml_cuda_ar_rank_ll_kernel(float * data, const int n, uint2 * __restrict__ ll_out, const uint2 * __restrict__ ll_in,
                                                   const int token, unsigned long long * prof) {
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;
    const bool pr  = prof != nullptr && bid == 0 && tid == 0;
    if (pr) {
        ar_prof_mark(prof, token, 0);
    }
    if constexpr (V4) {
        for (int i = gtid; i < n/2; i += gnt) {
            ar_ll_put2(ll_out + 2*i, data[2*i], data[2*i + 1], token);
        }
    } else {
        for (int i = gtid; i < n; i += gnt) {
            ar_ll_put(ll_out + i, data[i], token);
        }
    }
    if (prof != nullptr) {
        __syncthreads();
        if (pr) {
            ar_prof_mark(prof, token, 1);
            ar_prof_mark(prof, token, 2);
        }
    }
    if constexpr (V4) {
        for (int i = gtid; i < n/2; i += gnt) {
            const float2 o = ar_ll_get2(ll_in + 2*i, token);
            data[2*i]     = data[2*i]     + o.x;
            data[2*i + 1] = data[2*i + 1] + o.y;
        }
    } else {
        for (int i = gtid; i < n; i += gnt) {
            data[i] = data[i] + ar_ll_get(ll_in + i, token);
        }
    }
    if (prof != nullptr) {
        __syncthreads();
        if (pr) {
            ar_prof_mark(prof, token, 3);
            ar_prof_mark(prof, token, 4);
        }
    }
}

// AllReduce fused with the RMS norm that opens the next subgraph (exactly rms_norm_f32<BS, true[, true]>'s register path: same
// thread -> column map, accumulation order, block_reduce and output expression, so the result is bit-identical):
//   ADD_FIRST:  x = (mine + peer) + res, stored to x_out (the residual ADD node), then norm      (qwen35moe: AR -> ADD -> RMS_NORM -> MUL)
//   otherwise:  x = mine + peer, then norm                                                     (gemma4:    AR -> RMS_NORM -> MUL [-> ADD])
//   y = rsqrt(mean(x^2) + eps) * x * w [+ add2]
// One block per row (rows striped over the grid); the handshake partition is by row on both devices.
template <int BS, bool ADD_FIRST, bool ADD_AFTER, bool LL, int WIRE = 0>
static __global__ void __launch_bounds__(BS) ggml_cuda_ar_rank_norm_kernel(
        float * data, const int ncols, const int nrows, float * __restrict__ wire_out, const float * __restrict__ wire_in,
        uint2 * __restrict__ ll_out, const uint2 * __restrict__ ll_in, const bool ll_v4,
        int * signal_out, const int * signal_in, const int token,
        const float * res, float * x_out, const float * w, const float * add2, float * y, const float eps,
        unsigned long long * prof) {
    // no __restrict__ on the fused outputs: the allocator may place an output in an input's buffer (in-place ADD / norm). Every
    // element is read and written by the same thread, y only after the row's reduction, so aliasing is safe.
    constexpr int max_regs = 8;
    __shared__ float s_sum[32];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const bool pr = prof != nullptr && bid == 0 && tid == 0;
    if (pr) {
        ar_prof_mark(prof, token, 0);
    }

    for (int r = bid; r < nrows; r += gridDim.x) {
        if constexpr (LL) {
            if (ll_v4) {  // the reader below keeps its per-element map (same words), only the stores pair up
                for (int c2 = tid; c2 < ncols/2; c2 += BS) {
                    const int64_t o = (int64_t) r*ncols + 2*c2;
                    ar_ll_put2(ll_out + o, data[o], data[o + 1], token);
                }
                continue;
            }
        }
        if constexpr (!LL && WIRE != 0) {
            // 16-bit wire: encode the row into the peer's inbox, keep the rounded row in place (see ar_wire)
            uint16_t * w16 = (uint16_t *) wire_out;
            float inv = 1.0f, scale = 1.0f;
            if constexpr (WIRE == 2) {
                float m = 0.0f;
                for (int col = tid; col < ncols; col += BS) {
                    m = fmaxf(m, fabsf(data[(int64_t) r*ncols + col]));
                }
                m = block_reduce<block_reduce_method::MAX, BS>(m, s_sum);
                __syncthreads();
                ar_f16_scale(m, inv, scale);
                if (tid == 0) {
                    ((float *) ((char *) wire_out + ar_wire_scales_off((int64_t) ncols*nrows)))[r] = scale;
                }
            }
            for (int col = tid; col < ncols; col += BS) {
                const int64_t o = (int64_t) r*ncols + col;
                const float x = data[o];
                if constexpr (WIRE == 1) {
                    const uint32_t b = ar_bf16_enc(x);
                    w16[o]  = (uint16_t) b;
                    data[o] = ar_bf16_dec(b);
                } else {
                    const half h = __float2half_rn(x*inv);
                    w16[o]  = __half_as_ushort(h);
                    data[o] = __half2float(h)*scale;
                }
            }
            continue;
        }
        for (int col = tid; col < ncols; col += BS) {
            if constexpr (LL) {
                ar_ll_put(ll_out + (int64_t) r*ncols + col, data[(int64_t) r*ncols + col], token);
            } else {
                wire_out[(int64_t) r*ncols + col] = data[(int64_t) r*ncols + col];
            }
        }
    }
    if (prof != nullptr) {
        __syncthreads();
        if (pr) {
            ar_prof_mark(prof, token, 1);
        }
    }
    if constexpr (LL) {
        if (pr) {
            ar_prof_mark(prof, token, 2);
        }
    } else {
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            if (pr) {
                ar_prof_mark(prof, token, 2);
            }
            *(volatile int *) (signal_out + bid * (AR_RANK_STRIDE / sizeof(int))) = token;
            __threadfence_system();
            const volatile int * in = (const volatile int *) (signal_in + bid * (AR_RANK_STRIDE / sizeof(int)));
            while (*in != token) {
            }
            if (pr) {
                ar_prof_mark(prof, token, 3);
            }
        }
        __syncthreads();
        __threadfence();
    }

    for (int r = bid; r < nrows; r += gridDim.x) {
        const int64_t o = (int64_t) r*ncols;
        float xv[max_regs];
        float tmp = 0.0f;
        float wscale = 1.0f;
        if constexpr (!LL && WIRE == 2) {
            wscale = __ldcg((const float *) ((const char *) wire_in + ar_wire_scales_off((int64_t) ncols*nrows)) + r);
        }
#pragma unroll
        for (int u = 0; u < max_regs; ++u) {
            const int col = tid + u*BS;
            float v = 0.0f;
            if (col < ncols) {
                if constexpr (LL) {
                    v = data[o + col] + ar_ll_get(ll_in + o + col, token);
                } else if constexpr (WIRE == 1) {
                    v = data[o + col] + ar_bf16_dec(__ldcg((const unsigned short *) wire_in + o + col));
                } else if constexpr (WIRE == 2) {
                    v = data[o + col] + __half2float(__ushort_as_half(__ldcg((const unsigned short *) wire_in + o + col)))*wscale;
                } else {
                    v = data[o + col] + __ldcg(wire_in + o + col);
                }
                data[o + col] = v;
                if constexpr (ADD_FIRST) {
                    v = v + res[o + col];
                    x_out[o + col] = v;
                }
            }
            xv[u] = v;
            tmp += xv[u] * xv[u];
        }
        tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);
#pragma unroll
        for (int u = 0; u < max_regs; ++u) {
            const int col = tid + u*BS;
            if (col < ncols) {
                if constexpr (ADD_AFTER) {
                    y[o + col] = scale * xv[u] * w[col] + add2[o + col];
                } else {
                    y[o + col] = scale * xv[u] * w[col];
                }
            }
        }
        __syncthreads(); // s_sum reuse by the next row
    }
    if (pr) {
        if constexpr (LL) {
            ar_prof_mark(prof, token, 3);
        }
        ar_prof_mark(prof, token, 4);
    }
}

// ggml_cuda_ar_rank_kernel with a 16-bit wire (n % 4 == 0, data 16-byte aligned). Chunk = the 1024 elements one block covers per
// iteration (256 threads x float4); WIRE 2 scales per chunk.
template <int WIRE>
static __global__ void __launch_bounds__(256) ggml_cuda_ar_rank_wire_kernel(float * data, const int n, char * __restrict__ wire_out,
        const char * __restrict__ wire_in, int * signal_out, const int * signal_in, const int token) {
    __shared__ float s_red[32];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n4  = n / 4;
    const int gnt = gridDim.x * blockDim.x;
    float4 * d4 = (float4 *) data;
    uint2  * w_out = (uint2 *) wire_out;
    const uint2 * w_in = (const uint2 *) wire_in;
    float * s_out = (float *) (wire_out + ar_wire_scales_off(n));
    const float * s_in = (const float *) (wire_in + ar_wire_scales_off(n));

    // phase 1: encode my partial into the peer's inbox and keep the rounded value in place
    for (int base = bid*blockDim.x; base < n4; base += gnt) {
        const int i = base + tid;
        float4 v = i < n4 ? d4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        uint2 w;
        if constexpr (WIRE == 1) {
            w.x = ar_bf16_enc(v.x) | (ar_bf16_enc(v.y) << 16);
            w.y = ar_bf16_enc(v.z) | (ar_bf16_enc(v.w) << 16);
            v = make_float4(ar_bf16_dec(w.x & 0xFFFFu), ar_bf16_dec(w.x >> 16), ar_bf16_dec(w.y & 0xFFFFu), ar_bf16_dec(w.y >> 16));
        } else {
            float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
            m = block_reduce<block_reduce_method::MAX, 256>(m, s_red);
            float inv, scale;
            ar_f16_scale(m, inv, scale);
            const half2 h0 = __floats2half2_rn(v.x*inv, v.y*inv), h1 = __floats2half2_rn(v.z*inv, v.w*inv);
            w.x = *(const uint32_t *) &h0;
            w.y = *(const uint32_t *) &h1;
            const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
            v = make_float4(f0.x*scale, f0.y*scale, f1.x*scale, f1.y*scale);
            if (tid == 0) {
                s_out[base / blockDim.x] = scale;
            }
            __syncthreads();             // s_red reuse by the next chunk
        }
        if (i < n4) {
            w_out[i] = w;
            d4[i]    = v;
        }
    }
    __threadfence_system();
    __syncthreads();
    // phase 2: per-block handshake (same striping on both devices)
    if (tid == 0) {
        *(volatile int *) (signal_out + bid * (AR_RANK_STRIDE / sizeof(int))) = token;
        __threadfence_system();
        const volatile int * in = (const volatile int *) (signal_in + bid * (AR_RANK_STRIDE / sizeof(int)));
        while (*in != token) {
        }
    }
    __syncthreads();
    __threadfence();
    // phase 3: rounded mine (in place) + decoded peer
    for (int base = bid*blockDim.x; base < n4; base += gnt) {
        const int i = base + tid;
        if (i >= n4) {
            continue;
        }
        const uint2 w = __ldcg(w_in + i);
        float4 o;
        if constexpr (WIRE == 1) {
            o = make_float4(ar_bf16_dec(w.x & 0xFFFFu), ar_bf16_dec(w.x >> 16), ar_bf16_dec(w.y & 0xFFFFu), ar_bf16_dec(w.y >> 16));
        } else {
            const float scale = __ldcg(s_in + base / blockDim.x);
            const float2 f0 = __half22float2(*(const half2 *) &w.x), f1 = __half22float2(*(const half2 *) &w.y);
            o = make_float4(f0.x*scale, f0.y*scale, f1.x*scale, f1.y*scale);
        }
        float4 v = d4[i];
        v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
        d4[i] = v;
    }
}

static bool ar_ll_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AR_LL");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}
// largest reduction (elements) taken by the LL path; GGML_CUDA_AR_LL_MAX overrides (capped at the buffer size). Default 8192 =
// single-token decode (gemma4 2816, qwen36 2048, Swift 5120): the LL words double the bytes on the wire, which loses to the flag
// path once the payload is bandwidth-bound -- Swift's MTP verify reductions (5120 x 2..5 tokens) served 1-2 % slower at 32K (R14S).
static int64_t ar_ll_max() {
    static const int64_t m = [] {
        const char * e = getenv("GGML_CUDA_AR_LL_MAX");
        const int64_t v = e ? (int64_t) atoll(e) : (int64_t) 8192;
        return std::min<int64_t>(v, AR_LL_MAX_ELEMS);
    }();
    return m;
}
// 16-byte LL stores (two words); GGML_CUDA_AR_LL_V4=0 for 8-byte stores
static bool ar_ll_v4() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AR_LL_V4");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

ggml_cuda_ar_rank_ctx * ggml_cuda_ar_rank_init(int dev0, int dev1, size_t max_bytes) {
    if (getenv("GGML_CUDA_P2P") == nullptr) {
        return nullptr; // peer access is only enabled (ggml_cuda_init) under GGML_CUDA_P2P
    }
    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, dev0, dev1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, dev1, dev0));
    if (!can01 || !can10) {
        return nullptr;
    }
    auto * c = new ggml_cuda_ar_rank_ctx{};
    c->devices[0] = dev0;
    c->devices[1] = dev1;
    c->max_bytes  = max_bytes;
    // with the 16-bit wire every reduction above 1 MiB travels as 16-bit values + fp32 scales (one per 1024-element chunk or per row, rows
    // <= 65535 on the fused-norm path), so a slot needs about half of max_bytes
    c->slot_bytes = ar_wire() != 0 && max_bytes > ((size_t) 1 << 20)
        ? std::max((size_t) 1 << 20, ar_wire_scales_off((int64_t) (max_bytes / sizeof(float))) + (size_t) 65536 * sizeof(float))
        : max_bytes;
    const size_t arrival_bytes = (size_t) AR_RANK_SLOTS * AR_RANK_ALLOC_BLOCKS * AR_RANK_STRIDE;
    bool ok = true;
    for (int i = 0; i < 2 && ok; ++i) {
        ggml_cuda_set_device(c->devices[i]);
        ok = cudaMalloc((void **) &c->inbox[i], AR_RANK_SLOTS * c->slot_bytes) == cudaSuccess &&
             cudaMalloc((void **) &c->arrival[i], arrival_bytes) == cudaSuccess &&
             cudaMemset(c->arrival[i], 0, arrival_bytes) == cudaSuccess &&
             cudaDeviceSynchronize() == cudaSuccess;
    }
    for (int i = 0; i < 2 && ok; ++i) {
        // LL words start at token 0: call tokens start at 1, so a fresh buffer never matches
        ggml_cuda_set_device(c->devices[i]);
        const size_t lb = (size_t) AR_RANK_SLOTS * AR_LL_MAX_ELEMS * sizeof(uint2);
        ok = cudaMalloc((void **) &c->ll[i], lb) == cudaSuccess && cudaMemset(c->ll[i], 0, lb) == cudaSuccess &&
             cudaDeviceSynchronize() == cudaSuccess;
    }
    if (ok && getenv("GGML_CUDA_AR_PROF") != nullptr) {
        for (int i = 0; i < 2 && ok; ++i) {
            ggml_cuda_set_device(c->devices[i]);
            const size_t pb = (size_t) AR_PROF_RING * 8 * sizeof(unsigned long long);
            ok = cudaMalloc((void **) &c->prof[i], pb) == cudaSuccess && cudaMemset(c->prof[i], 0, pb) == cudaSuccess &&
                 cudaDeviceSynchronize() == cudaSuccess;
            c->prof_meta[i].reserve(AR_PROF_RING);
        }
        GGML_LOG_INFO("%s: AllReduce profiler on (GGML_CUDA_AR_PROF)\n", __func__);
    }
    if (!ok) {
        (void) cudaGetLastError();
        ggml_cuda_ar_rank_free(c);
        return nullptr;
    }
    return c;
}

void ggml_cuda_ar_rank_free(ggml_cuda_ar_rank_ctx * c) {
    if (c == nullptr) {
        return;
    }
    if (c->prof[0] != nullptr && c->prof[1] != nullptr && !c->prof_meta[0].empty()) {
        static int seq = 0;
        const char * pre = getenv("GGML_CUDA_AR_PROF");
        char path[1024];
        snprintf(path, sizeof(path), "%s.%d.%d.csv", pre, (int) getpid(), seq++);
        std::vector<unsigned long long> h[2];
        for (int i = 0; i < 2; ++i) {
            ggml_cuda_set_device(c->devices[i]);
            CUDA_CHECK(cudaDeviceSynchronize());
            h[i].resize((size_t) AR_PROF_RING * 8);
            CUDA_CHECK(cudaMemcpy(h[i].data(), c->prof[i], h[i].size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        }
        FILE * f = fopen(path, "w");
        if (f) {
            fprintf(f, "rank,token,kind,n,nrows,c0,c1,c2,c3,c4,g0\n");
            for (int i = 0; i < 2; ++i) {
                const size_t nm = c->prof_meta[i].size();
                const size_t first = nm > AR_PROF_RING ? nm - AR_PROF_RING : 0;
                for (size_t j = first; j < nm; ++j) {
                    const ar_prof_meta & m = c->prof_meta[i][j];
                    const unsigned long long * e = h[i].data() + (size_t) (m.token & (AR_PROF_RING - 1)) * 8;
                    fprintf(f, "%d,%u,%d,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu\n", i, m.token, (int) m.kind, m.n, m.nrows,
                            e[0], e[1], e[2], e[3], e[4], e[5]);
                }
            }
            fclose(f);
            GGML_LOG_INFO("%s: AllReduce profile -> %s\n", __func__, path);
        }
    }
    for (int i = 0; i < 2; ++i) {
        ggml_cuda_set_device(c->devices[i]);
        if (c->prof[i])    { CUDA_CHECK(cudaFree(c->prof[i])); }
        if (c->ll[i])      { CUDA_CHECK(cudaFree(c->ll[i])); }
        if (c->inbox[i])   { CUDA_CHECK(cudaFree(c->inbox[i])); }
        if (c->arrival[i]) { CUDA_CHECK(cudaFree(c->arrival[i])); }
    }
    delete c;
}

int ggml_cuda_ar_rank_wire() {
    return ar_wire();
}

bool ggml_cuda_ar_rank_eligible(const ggml_cuda_ar_rank_ctx * c, const ggml_tensor * t) {
    if (c == nullptr || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nbytes(t) > c->max_bytes || ggml_nelements(t) <= 0) {
        return false;
    }
    if (ar_wire() == 0 || ggml_nbytes(t) <= ((size_t) 1 << 20)) {
        return ggml_nbytes(t) <= c->slot_bytes;
    }
    // above 1 MiB only the 16-bit wire fits a slot: the plain wire kernel needs n % 4 == 0 and 16-byte alignment (the fused-norm kernel
    // has neither restriction, but it is only tried after this check)
    const int64_t n = ggml_nelements(t);
    return n % 4 == 0 && ((uintptr_t) t->data & 15) == 0 &&
           ar_wire_scales_off(n) + (size_t) (n / 1024 + 1) * sizeof(float) <= c->slot_bytes;
}

void ggml_cuda_ar_rank_reduce(ggml_cuda_ar_rank_ctx * c, int rank, cudaStream_t stream, ggml_tensor * t) {
    const int      peer  = rank ^ 1;
    const uint32_t token = ++c->calls[rank];
    const int      slot  = token % AR_RANK_SLOTS;
    const int      n     = (int) ggml_nelements(t);
    const int      n4    = n / 4;
    const int      maxb  = (size_t) n * sizeof(float) > ((size_t) 1 << 20) ? ar_blocks_large() : AR_RANK_MAX_BLOCKS;
    const int      nb    = std::max(1, std::min(maxb, (n4 + AR_RANK_THREADS - 1) / AR_RANK_THREADS));

    float * wire_out   = (float *) (c->inbox[peer] + (size_t) slot * c->slot_bytes);
    float * wire_in    = (float *) (c->inbox[rank] + (size_t) slot * c->slot_bytes);
    int   * signal_out = (int *) (c->arrival[peer] + (size_t) slot * AR_RANK_ALLOC_BLOCKS * AR_RANK_STRIDE);
    int   * signal_in  = (int *) (c->arrival[rank] + (size_t) slot * AR_RANK_ALLOC_BLOCKS * AR_RANK_STRIDE);

    const bool ll = ar_ll_enabled() && n <= ar_ll_max();
    if (c->prof[rank] != nullptr) {
        c->prof_meta[rank].push_back({token, (uint8_t) (ll ? 8 : 0), n, 1});
    }
    if (ll) {
        uint2 * ll_out = c->ll[peer] + (size_t) slot * AR_LL_MAX_ELEMS;
        uint2 * ll_in  = c->ll[rank] + (size_t) slot * AR_LL_MAX_ELEMS;
        const int nbl  = std::max(1, std::min(AR_RANK_MAX_BLOCKS, (n + 4*AR_RANK_THREADS - 1) / (4*AR_RANK_THREADS)));
        if (ar_ll_v4() && n % 2 == 0) {
            ggml_cuda_ar_rank_ll_kernel<true><<<nbl, AR_RANK_THREADS, 0, stream>>>((float *) t->data, n, ll_out, ll_in, (int) token, c->prof[rank]);
        } else {
            ggml_cuda_ar_rank_ll_kernel<false><<<nbl, AR_RANK_THREADS, 0, stream>>>((float *) t->data, n, ll_out, ll_in, (int) token, c->prof[rank]);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (ar_wire() != 0 && (size_t) n * sizeof(float) > ((size_t) 1 << 20) && n % 4 == 0 && ((uintptr_t) t->data & 15) == 0 &&
            ar_wire_scales_off(n) + (size_t) ((n4 + AR_RANK_THREADS - 1) / AR_RANK_THREADS) * sizeof(float) <= c->slot_bytes) {
        if (ar_wire() == 1) {
            ggml_cuda_ar_rank_wire_kernel<1><<<nb, AR_RANK_THREADS, 0, stream>>>((float *) t->data, n, (char *) wire_out, (const char *) wire_in,
                                                                                 signal_out, signal_in, (int) token);
        } else {
            ggml_cuda_ar_rank_wire_kernel<2><<<nb, AR_RANK_THREADS, 0, stream>>>((float *) t->data, n, (char *) wire_out, (const char *) wire_in,
                                                                                 signal_out, signal_in, (int) token);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    ggml_cuda_ar_rank_kernel<<<nb, AR_RANK_THREADS, 0, stream>>>((float *) t->data, n, wire_out, wire_in, signal_out, signal_in, (int) token,
                                                                  c->prof[rank]);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_ar_rank_norm_eligible(const ggml_cuda_ar_rank_ctx * c, const ggml_tensor * t) {
    if (!ggml_cuda_ar_rank_eligible(c, t)) {
        return false;
    }
    const int64_t ncols = t->ne[0];
    const int     bs    = ncols < 1024 ? 256 : 1024;
    return ncols <= bs * 8 && ggml_nrows(t) <= 65535;
}

void ggml_cuda_ar_rank_reduce_norm(ggml_cuda_ar_rank_ctx * c, int rank, cudaStream_t stream, ggml_tensor * t, const ggml_cuda_ar_rank_norm_args & a) {
    const int      peer  = rank ^ 1;
    const uint32_t token = ++c->calls[rank];
    const int      slot  = token % AR_RANK_SLOTS;
    const int      ncols = (int) t->ne[0];
    const int      nrows = (int) ggml_nrows(t);
    const int      nb    = std::min((size_t) ggml_nbytes(t) > ((size_t) 1 << 20) ? ar_blocks_large() : AR_RANK_MAX_BLOCKS, nrows);

    float * wire_out   = (float *) (c->inbox[peer] + (size_t) slot * c->slot_bytes);
    float * wire_in    = (float *) (c->inbox[rank] + (size_t) slot * c->slot_bytes);
    int   * signal_out = (int *) (c->arrival[peer] + (size_t) slot * AR_RANK_ALLOC_BLOCKS * AR_RANK_STRIDE);
    int   * signal_in  = (int *) (c->arrival[rank] + (size_t) slot * AR_RANK_ALLOC_BLOCKS * AR_RANK_STRIDE);

    const bool ll = ar_ll_enabled() && (int64_t) ncols * nrows <= ar_ll_max();
    const bool v4 = ar_ll_v4() && ncols % 2 == 0;
    uint2 * ll_out = c->ll[peer] + (size_t) slot * AR_LL_MAX_ELEMS;
    uint2 * ll_in  = c->ll[rank] + (size_t) slot * AR_LL_MAX_ELEMS;
#define AR_NORM_LAUNCH1(BS, AF, AA, LLV) ggml_cuda_ar_rank_norm_kernel<BS, AF, AA, LLV><<<nb, BS, 0, stream>>>((float *) t->data, ncols, nrows, \
        wire_out, wire_in, ll_out, ll_in, v4, signal_out, signal_in, (int) token, a.res, a.x_out, a.w, a.add2, a.y, a.eps, c->prof[rank])
#define AR_NORM_LAUNCHW(BS, AF, AA, W) ggml_cuda_ar_rank_norm_kernel<BS, AF, AA, false, W><<<nb, BS, 0, stream>>>((float *) t->data, ncols, nrows, \
        wire_out, wire_in, ll_out, ll_in, v4, signal_out, signal_in, (int) token, a.res, a.x_out, a.w, a.add2, a.y, a.eps, c->prof[rank])
    // compressed wire only for large reductions (see ar_wire); the f16 scales (one per row) must fit after the 16-bit values
    const int wire = !ll && ar_wire() != 0 && (size_t) ggml_nbytes(t) > ((size_t) 1 << 20) &&
                     ar_wire_scales_off((int64_t) ncols*nrows) + (size_t) nrows*sizeof(float) <= c->slot_bytes ? ar_wire() : 0;
#define AR_NORM_LAUNCH(BS, AF, AA) do { if (ll) AR_NORM_LAUNCH1(BS, AF, AA, true); else if (wire == 1) AR_NORM_LAUNCHW(BS, AF, AA, 1); \
        else if (wire == 2) AR_NORM_LAUNCHW(BS, AF, AA, 2); else AR_NORM_LAUNCH1(BS, AF, AA, false); } while (0)
    const bool af = a.res != nullptr;
    const bool aa = a.add2 != nullptr;
    if (c->prof[rank] != nullptr) {
        c->prof_meta[rank].push_back({token, (uint8_t) ((ll ? 8 : 0) + 1 + (af ? 1 : 0) + (aa ? 2 : 0)), ncols, nrows});
    }
    if (ncols < 1024) {
        if (af) { if (aa) AR_NORM_LAUNCH(256, true, true); else AR_NORM_LAUNCH(256, true, false); }
        else    { if (aa) AR_NORM_LAUNCH(256, false, true); else AR_NORM_LAUNCH(256, false, false); }
    } else {
        if (af) { if (aa) AR_NORM_LAUNCH(1024, true, true); else AR_NORM_LAUNCH(1024, true, false); }
        else    { if (aa) AR_NORM_LAUNCH(1024, false, true); else AR_NORM_LAUNCH(1024, false, false); }
    }
#undef AR_NORM_LAUNCH
#undef AR_NORM_LAUNCHW
#undef AR_NORM_LAUNCH1
    CUDA_CHECK(cudaGetLastError());
}
