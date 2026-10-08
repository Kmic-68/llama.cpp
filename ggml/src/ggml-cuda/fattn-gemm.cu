// Flash attention via cuBLAS GEMMs, for pre-Volta NVIDIA (P100 / sm_60).
//
// Why this exists: on Pascal there are no tensor cores, so long-context prefill falls to
// flash_attn_tile, which measures 3.55 TFLOPS of a 19.05 peak (18.6%). The cuBLAS hgemm
// next to it in the same model reaches 15.7, and still reaches 13-15 at attention shapes
// (measured: QK^T k=256 -> 14.65, PV n=256 -> 13.10). Attention is ~86% of prefill at
// 262144 context on qwen3.5, so closing that gap is worth ~2x on long-context prefill.
//
// Structure is standard flash attention with an online softmax, except the two matmuls
// are cuBLAS calls instead of hand-written tiles:
//
//   for each KV chunk:
//       S = K^T Q                      (GEMM, f16 accumulate by default -- NOT what the tile
//                                       kernel does; see the note at the QK^T call)
//       m_new = max(m, rowmax(S+mask))
//       corr  = exp(m - m_new);  P = exp(S + mask - m_new)   (P in its own buffer, not in S)
//       l     = l*corr + rowsum(P)
//       Otmp  = V P                    (GEMM, f16 accumulate over one chunk)
//       O     = O*corr + Otmp          (fused rescale + f32 accumulate across chunks, out of place)
//   dst = O / l
//
// S is computed TRANSPOSED ([n_kv_chunk x n_tokens], column-major) so that one query's
// scores are contiguous: that makes the softmax kernel coalesced and turns PV into a
// plain V*P with no transpose.
//
// K/V are dequantized one chunk at a time, so this path never materializes the whole
// cache in f16 -- unlike the tile path, which converts all of K and V on every call and
// costs 512 MiB per GPU at 262144 context.

#include "common.cuh"
#include "fattn-common.cuh"   // FATTN_KQ_MAX_OFFSET
#include "fattn-gemm.cuh"
#include "convert.cuh"

#include <cublas_v2.h>
#include <cuda.h>
#include "fattn-fold-sass.h"
#include <algorithm>
#include <mutex>
#include <vector>

// A compact causal mask (I32 [nt], ggml_flash_attn_ext) keeps the first L_t keys of query t, with
// L_t = L_0 + t (llama_kv_cache::kq_mask_prefix). That f16 mask is a staircase: row t, key j is
// 0 iff j - t < L_0. So one vector B[k] = (k - (nt-1) < L_0 ? 0 : -inf), read with a row stride of
// -1 from B + nt - 1, gives every kernel below exactly the values of the full [nkv x nt] mask at the
// same (row, key) positions, in nkv + nt halfs instead of nkv*nt.
static __global__ void fattn_gemm_mask_stair(const int32_t * __restrict__ L, half * __restrict__ B, const int64_t n, const int64_t nt) {
    const int64_t k = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (k < n) {
        B[k] = k - (nt - 1) < (int64_t) L[0] ? __float2half(0.0f) : __float2half(-INFINITY);
    }
}

struct fattn_gemm_mask_view {
    const half * data;
    int64_t      s;      // row stride in halfs (may be negative)
};

static fattn_gemm_mask_view fattn_gemm_mask_get(const ggml_tensor * mask, const int64_t nt, const int64_t nkv,
        ggml_cuda_pool_alloc<half> & stair, cudaStream_t stream) {
    if (mask->type != GGML_TYPE_I32) {
        return { (const half *) mask->data, (int64_t) (mask->nb[1]/sizeof(half)) };
    }
    const int64_t n = nkv + nt - 1;
    stair.alloc(n);
    fattn_gemm_mask_stair<<<(n + 255)/256, 256, 0, stream>>>((const int32_t *) mask->data, stair.get(), n, nt);
    CUDA_CHECK(cudaGetLastError());
    return { stair.get() + (nt - 1), -1 };
}


// Precision, chosen at runtime.
//
//   default -- COMPUTE_16F for both GEMMs: fp16 scores and probabilities, and an fp16 PV partial
//     per chunk folded into an fp32 running output. The QK^T dot product (k = D = 256)
//     accumulates in fp16, where upstream's tile kernel accumulates KQ in fp32
//     (fattn-tile.cuh:604): ~4x more error per attention logit.
//   GGML_CUDA_FA_GEMM_PREC=32 -- fp32 accumulation everywhere, and more precise than upstream:
//     QK^T runs COMPUTE_32F over the same fp16 K and Q (a product of two halves needs 22
//     significant bits, so every product is EXACT in fp32 -- upstream rounds each product to
//     half first); scores and probabilities are fp32 (upstream keeps them in half); V is
//     dequantized straight to fp32 (exact for q4_0, where upstream rounds V to half); PV is
//     all-fp32. Costs 11% of pp2048 at depth 16384 and 27% at 65536.
//
// What the default costs in model output is measured by paired per-chunk perplexity against
// PREC=32 (OPTLOG attempt 152). PREC=32 is kept as the reference.
static bool ggml_cuda_fa_gemm_prec32() {
    static const bool v = [] {
        const char * s = getenv("GGML_CUDA_FA_GEMM_PREC");
        return s && atoi(s) == 32;
    }();
    return v;
}

static __device__ __forceinline__ float fattn_gemm_to_f(const half  x) { return __half2float(x); }
static __device__ __forceinline__ float fattn_gemm_to_f(const float x) { return x; }
template <typename T> static __device__ __forceinline__ T fattn_gemm_store(const float x);
template <> __device__ __forceinline__ half  fattn_gemm_store<half >(const float x) { return __float2half(x); }
template <> __device__ __forceinline__ float fattn_gemm_store<float>(const float x) { return x; }

// One block per (query token, head). Applies the mask, advances the running softmax
// statistics, writes the chunk's probabilities, and reports the rescale factor for O.
//
// OUT OF PLACE, deliberately: S and P must be separate buffers. From 0f5b88954 until 2026-09-13 P
// was written over S (saving 50 MB), and that is not safe here. Pass 2 loads S[j] and then
// stores P[j] at the same address, and on this toolchain the store can land before the load, so
// the load reads back a probability as a score: exp(4*P - m) with P <= 1/8 and a very negative
// row max m gives values up to ~1e7. Measured by running the kernel twice on identical inputs
// inside the op (fp32, 16384 context, ~2240 softmax launches per run): in place with the two
// aliased __restrict__ parameters of the original, with one restrict pointer, and with no
// restrict at all, all mismatch (4, 6 and 3 launches); perplexity varies run to run (3.3165,
// 3.3189, 3.3199; 3.3144-3.3272). Out of place: 0 mismatches in ~6700 launches, 3.3165 every
// run. The fp16 instantiation showed no mismatch in 15360 checked launches, but it is the same
// pattern, and there a read-back probability overflows half to inf and NaNs the output -- a
// 4096-context perplexity run once went NaN in exactly that way and did not reproduce.
// Cost of the separate buffer: ~1% of the op, and nkv_c*nt*gqa more elements of scratch.
template <int block_size, typename T>
static __global__ void fattn_gemm_softmax(
        const T      * __restrict__ S,          // [nkv_c x nt] per head, column-major: scores
        T            * __restrict__ P,          // out, same layout, a separate buffer: probabilities
        const half   * __restrict__ mask,       // [nkv_pad x nt], contiguous
        const float  * __restrict__ mask_first, // [nt] first nonzero mask column of each row
        float        * __restrict__ m_state,    // [nt x nh]
        float        * __restrict__ l_state,    // [nt x nh]
        float        * __restrict__ corr_out,   // [nt x nh]
        const int nkv_c,      // keys in this chunk
        const int nkv_off,    // offset of this chunk within the full KV
        const int nt,
        const int64_t s_mask, // mask row stride in elements
        const int64_t s_head) // per-head stride of S and P in elements
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    const T * Sh = S + h*s_head + (int64_t) t*nkv_c;
    T       * Ph = P + h*s_head + (int64_t) t*nkv_c;
    // A chunk that ends at or before this row's first nonzero mask entry sees only +-0 there, and
    // adding +-0 to a logit changes neither the row max nor exp(v - m): skip those loads. Under a
    // causal mask that is every chunk but the last, which is worth ~7% of the op at 65536 context.
    const half * mh = (float) (nkv_off + nkv_c) > mask_first[t] ? mask + (int64_t) t*s_mask + nkv_off : nullptr;

    __shared__ float red[block_size/WARP_SIZE];

    // pass 1: row max of (4*S + mask); Q already carried scale*0.25 into the GEMM
    float vmax = -FLT_MAX/2.0f;
    for (int j = tid; j < nkv_c; j += block_size) {
        float v = 4.0f*fattn_gemm_to_f(Sh[j]);
        if (mh) {
            v += __half2float(mh[j]);
        }
        // + FATTN_KQ_MAX_OFFSET, as upstream does in tile (:816), vec (:342) and mma
        // (:723/:800): it raises the running max by 3*ln2 so every probability comes out
        // <= 1/8, giving the f16 probabilities and the f16 PV accumulator 3 bits of headroom.
        // Cancels exactly in the final divide by the row sum.
        vmax = fmaxf(vmax, v + FATTN_KQ_MAX_OFFSET);
    }
    vmax = warp_reduce_max(vmax);
    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            red[tid/WARP_SIZE] = vmax;
        }
        __syncthreads();
        vmax = tid < block_size/WARP_SIZE ? red[tid] : -FLT_MAX/2.0f;
        vmax = warp_reduce_max(vmax);
        if (tid == 0) {
            red[0] = vmax;
        }
        __syncthreads();
        vmax = red[0];
        // every thread must have read the max out of red[0] before pass 2 reuses red for the row sum:
        // without this barrier warp 0 can store its partial sum into red[0] first, a late warp then
        // takes that sum as the row max, and exp(v - m) is wrong for its keys -- run-to-run drift in
        // fp32, and in fp16 a probability that overflows to inf and NaNs the output
        __syncthreads();
    }

    const float m_old = m_state[h*nt + t];
    const float m_new = fmaxf(m_old, vmax);
    // m_old == -inf on the first chunk; exp(-inf - m_new) is 0, which is what we want,
    // but guard against inf-inf producing NaN when the whole row is masked out.
    const float corr  = m_old <= -FLT_MAX/4.0f ? 0.0f : expf(m_old - m_new);

    // pass 2: P = exp(v - m_new), and its row sum
    float sum = 0.0f;
    for (int j = tid; j < nkv_c; j += block_size) {
        float v = 4.0f*fattn_gemm_to_f(Sh[j]);
        if (mh) {
            v += __half2float(mh[j]);
        }
        const float p = (v <= -FLT_MAX/4.0f || m_new <= -FLT_MAX/4.0f) ? 0.0f : expf(v - m_new);
        Ph[j] = fattn_gemm_store<T>(p);
        sum += p;
    }
    sum = warp_reduce_sum(sum);
    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            red[tid/WARP_SIZE] = sum;
        }
        __syncthreads();
        sum = tid < block_size/WARP_SIZE ? red[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            red[0] = sum;
        }
        __syncthreads();
        sum = red[0];
    }

    if (tid == 0) {
        m_state[h*nt + t]  = m_new;
        l_state[h*nt + t]  = l_state[h*nt + t]*corr + sum;
        corr_out[h*nt + t] = corr;
    }
}

// first[t] = the smallest key index j with mask[t][j] != 0, or nkv if the row has none. The
// softmax skips the mask for every chunk of row t that ends at or before it. General, not
// causal-specific: a mask with nonzero entries early (sliding window, other sequences) just
// skips less. -0.0 counts as zero (adding it is a no-op for the max and the exp); NaN counts as
// nonzero, the conservative direction. Stays on the GPU: a host read would synchronize, and
// under -sm tensor the host thread feeding the other GPU would stall behind it.
template <int block_size>
static __global__ void fattn_gemm_mask_first_nz(
        const half * __restrict__ mask, float * __restrict__ first,
        const int64_t s_mask, const int nkv) {
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    const half * mh = mask + (int64_t) t*s_mask;

    __shared__ float red[block_size/WARP_SIZE];

    // track max(-j) over nonzero entries, i.e. the smallest nonzero j, as an exact float (j < 2^24)
    float neg = -(float) nkv;
    for (int j = tid; j < nkv; j += block_size) {
        if (__half2float(mh[j]) != 0.0f) {
            neg = fmaxf(neg, -(float) j);
        }
    }
    neg = warp_reduce_max(neg);
    if (block_size > WARP_SIZE) {
        if (tid % WARP_SIZE == 0) {
            red[tid/WARP_SIZE] = neg;
        }
        __syncthreads();
        neg = tid < block_size/WARP_SIZE ? red[tid] : -(float) nkv;
        neg = warp_reduce_max(neg);
        if (tid == 0) {
            red[0] = neg;
        }
        __syncthreads();
        neg = red[0];
    }
    if (tid == 0) {
        first[t] = -neg;
    }
}

static __global__ void fattn_gemm_fill(float * __restrict__ p, const float v, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = v;
    }
}

// O = O*corr + Otmp, applied after the PV GEMM.
//
// The PV GEMM runs f16-accumulate (2:1 rate on Pascal) into a per-chunk f16 partial Otmp,
// and this kernel folds it into the f32 running output. So f16 summation spans only one
// chunk (k <= chunk) while accumulation ACROSS chunks stays f32 -- strictly better than
// upstream's tile kernel, which keeps VKQ in half2 over the whole cache
// (fattn-tile.cuh, FAST_FP16_AVAILABLE).
//
// Fusing the rescale into the accumulate is also cheaper than the rescale-only kernel it
// replaces: that one read+wrote O and then the beta=1 GEMM read+wrote O again (50 MB per
// chunk at nt=2048, gqa=6); this reads O+Otmp and writes the next O once (38 MB).
//
// Out of place, like the softmax: the running O is read from one buffer and written to the
// other, and the caller swaps them per chunk. The in-place form loaded and stored the same
// address inside the thread loop -- the pattern that raced in the softmax. At DV=256 the loop
// runs once per thread and no run-to-run difference was ever observed, but nothing proves it
// safe either; the second buffer costs DV*nt*gqa floats (12 MB per GPU at -ub 2048).
template <typename T>
static __global__ void fattn_gemm_accum_O(
        const float * __restrict__ O_in, float * __restrict__ O_out, const T * __restrict__ Otmp,
        const float * __restrict__ corr,
        const int DV, const int nt) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float c = corr[h*nt + t];
    const int64_t off = ((int64_t) h*nt + t)*DV;
    const float * Ih = O_in  + off;
    float       * Oh = O_out + off;
    const T     * Th = Otmp  + off;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        Oh[d] = Ih[d]*c + fattn_gemm_to_f(Th[d]);
    }
}

// dst[d, h, t] = O[d, t, h] / l[t, h]  -- note dst is permuted: [DV, n_head, n_tokens, n_seq]
static __global__ void fattn_gemm_finalize(
        const float * __restrict__ O, const float * __restrict__ l_state,
        float * __restrict__ dst,
        const int DV, const int nt, const int nh, const int head0,
        const int64_t dst_s1, const int64_t dst_s2) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float l = l_state[h*nt + t];
    const float inv = l > 0.0f ? 1.0f/l : 0.0f;
    const float * Oh = O + ((int64_t) h*nt + t)*DV;
    float * dh = dst + (int64_t) t*dst_s2 + (int64_t)(head0 + h)*dst_s1;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        dh[d] = Oh[d]*inv;
    }
}

// Convert one head's Q from strided f32 to contiguous f16 [D x nt].
// Q is pre-scaled by `qscale` = scale*0.25 here, BEFORE the f16 conversion and therefore
// before the fp16 QK^T accumulation, and the softmax multiplies the logits back by 4.
// This mirrors upstream fattn-tile.cuh:925-937, whose comment names this hardware:
// "Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in
// the KQ calculation." Both factors are exact powers of two at D=256 (scale = 1/16), so
// the pre-scale costs nothing in precision and buys 64x of fp16 overflow headroom in the
// accumulator. It must be applied here and not via cuBLAS `alpha`, because alpha is applied
// AFTER the accumulation and so cannot prevent a partial sum from overflowing.
static __global__ void fattn_gemm_q_to_f16(
        const char * __restrict__ Q, half * __restrict__ Qf16,
        const int D, const int nt, const int64_t nbq1, const int64_t nbq2,
        const int head0, const float qscale) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float * q = (const float *) (Q + (int64_t) t*nbq1 + (int64_t)(head0 + h)*nbq2);
    half * o = Qf16 + ((int64_t) h*nt + t)*D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        o[d] = __float2half(q[d]*qscale);
    }
}

// ---------------------------------------------------------------------------------------------
// Fold path (GGML_CUDA_FA_FOLD, default on for a q4_0 K/V cache): two hand-written GEMM kernels in
// the style of gemm-fold.cu (fp16 HFMA2 products, half2 chains folded into fp32) replace
// dequant + cuBLAS QK + softmax + cuBLAS PV + rescale.
//
//   fa_fold_qk:  S tile = K Q^T for 128 keys x 128 query columns, k = D. The epilogue applies the
//                mask, takes the max of the tile per column (m_t), writes P = exp(4S + mask - m_t)
//                in f16 and the tile's fp32 row sum l_t. S never reaches memory.
//   fa_fold_pv:  O = O*corr + sum_t exp(m_t - M) * (V P_t), with M the new running max. The fp16
//                chains restart at every 128-key tile and are folded into fp32 with the tile's
//                factor, so the rescale costs one FMUL per fold.
//
// Precision against the cuBLAS path: QK^T chains are 128 (not 256) long and summed in fp32; P is
// relative to the tile max instead of the chunk max (more bits); PV accumulates fp16 over 128 keys
// (not over the 2048-key chunk) and fp32 across tiles.
namespace fa_fold {

// The KV range is cut into nsplit independent streams of L keys each (own O, m, l), merged at the
// end: the PV grid is only (DV/128) x (N/128) x heads, 3.4 waves on 56 SMs with one stream.
// Virtual head z = kv_head*nsplit + split; chunk c of every stream runs in the same launch.
struct fa_fold_split {
    int nsplit, L, c, nkv;
};
static __device__ __forceinline__ int fa_fold_kv_off(const fa_fold_split & g, const int z) {
    return (z % g.nsplit)*g.L + g.c;
}
static __device__ __forceinline__ int fa_fold_nkv_c(const fa_fold_split & g, const int z, const int C) {
    const int off = fa_fold_kv_off(g, z);
    return max(0, min(min(C, g.L - g.c), g.nkv - off));
}

constexpr int BM  = 128;
constexpr int BN  = 128;
constexpr int BK2 = 16;   // k2 steps per smem tile = 32 values of K
constexpr int TK  = 128;  // keys per softmax tile (= QK BM, and the PV fold period)
constexpr int MAXTILE = 16; // tiles per chunk: chunk <= 2048 keys

static __device__ __forceinline__ float fold_h(const half2 p) {
    const half2    s = __hadd2(p, __lowhigh2highlow(p));  // high lane = lo + hi
    const int32_t  v = ((int32_t) *(const uint32_t *) &s) >> 3;
    return __int_as_float(v & 0x8FFFE000);                 // = (lo + hi) * 2^-112
}

// Dequantize one chunk of a q4_0 K and V cache for all KV heads:
//   K16[h][key][D]              (row per key, D contiguous)
//   Vp [h][key/2][DV] as half2  (lo lane = even key, hi lane = odd key: the PV A operand)
// Keys from nkv_c up to the padded chunk are written as zeros (P is zero there too, and 0*garbage
// could be NaN). Exact nibble values times the block scale, rounded once to half: identical to
// the generic dequantizer.
static __global__ void fa_fold_dequant(
        const char * __restrict__ K, const char * __restrict__ V,
        half * __restrict__ K16, half2 * __restrict__ Vp,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
        const int D, const fa_fold_split g, const int C) {
    const int nb   = D/QK4_0;
    const int b    = threadIdx.x % nb;          // block within a row
    const int kp   = blockIdx.x*(blockDim.x/nb) + threadIdx.x/nb;  // key pair
    const int z    = blockIdx.y;
    const int h    = z / g.nsplit;
    if (2*kp >= C) {
        return;
    }
    const int nkv_c  = fa_fold_nkv_c(g, z, C);
    const int kv_off = fa_fold_kv_off(g, z);
    half * k16 = K16 + ((int64_t) z*C + 2*kp)*D + b*QK4_0;
    half2 * vp = Vp  + ((int64_t) z*(C/2) + kp)*D + b*QK4_0;
    float vlo[QK4_0], vhi[QK4_0];
#pragma unroll
    for (int r = 0; r < 2; r++) {
        const int key = 2*kp + r;
        float * vv = r == 0 ? vlo : vhi;
        if (key < nkv_c) {
            const block_q4_0 * bk = (const block_q4_0 *) (K + h*nbk2 + (int64_t) (kv_off + key)*nbk1) + b;
            const block_q4_0 * bv = (const block_q4_0 *) (V + h*nbv2 + (int64_t) (kv_off + key)*nbv1) + b;
            const float dk = __half2float(bk->d);
            const float dv = __half2float(bv->d);
            half kk[QK4_0];
#pragma unroll
            for (int j = 0; j < QK4_0/2; j++) {
                const int qk = bk->qs[j];
                const int qv = bv->qs[j];
                kk[j]           = __float2half(dk*(float) ((qk & 0xF) - 8));
                kk[j + QK4_0/2] = __float2half(dk*(float) ((qk >> 4)  - 8));
                vv[j]           = dv*(float) ((qv & 0xF) - 8);
                vv[j + QK4_0/2] = dv*(float) ((qv >> 4)  - 8);
            }
#pragma unroll
            for (int j = 0; j < QK4_0; j += 8) {
                *(uint4 *) (k16 + r*D + j) = *(const uint4 *) (kk + j);
            }
        } else {
#pragma unroll
            for (int j = 0; j < QK4_0; j++) {
                vv[j] = 0.0f;
            }
#pragma unroll
            for (int j = 0; j < QK4_0; j += 8) {
                *(uint4 *) (k16 + r*D + j) = make_uint4(0, 0, 0, 0);
            }
        }
    }
    half2 pv[QK4_0];
#pragma unroll
    for (int j = 0; j < QK4_0; j++) {
        pv[j] = __halves2half2(__float2half(vlo[j]), __float2half(vhi[j]));
    }
#pragma unroll
    for (int j = 0; j < QK4_0; j += 4) {
        *(uint4 *) (vp + j) = *(const uint4 *) (pv + j);
    }
}

// The shared GEMM main loop of gemm-fold.cu: C[m][n] = sum_k A[m][k] B[n][k], both operands
// k-contiguous f16 (A rows are `lda` halves apart, B rows `ldb`), or, with A_PAIRS, A given as
// half2 k-pairs [k/2][m] (m contiguous, `lda` words per k2 row). fold_k2 half2 steps per fp16
// chain; at every fold `on_fold(fold_index, acc, h)` moves the chains into fp32.
template <bool A_PAIRS>
struct fa_fold_tiles {
    uint32_t As[2][BK2][BM];
    uint32_t Bs[2][BK2][BN];
};

struct fa_fold_nop { __device__ void operator()() const {} };

template <bool A_PAIRS, int fold_k2, bool B_BLOCKED = false, typename F, typename PRE = fa_fold_nop>
static __device__ __forceinline__ void fa_fold_mainloop(
        fa_fold_tiles<A_PAIRS> & sm,
        const half * __restrict__ A, const int64_t lda, const int mA,   // valid rows of A (m)
        const half * __restrict__ B, const int64_t ldb, const int nB,   // valid rows of B (n)
        const int K, float (&acc)[8][8], F on_fold, PRE pre = PRE()) {
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;

    half2 h[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            h[i][j] = make_half2(0.0f, 0.0f);
        }
    }
    uint4 ra[2];
    uint4 rb[2];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            if (A_PAIRS) {
                const int r = l >> 5;   // k2 row
                const int c = l & 31;   // uint4 column: m = 4c
                ra[i] = *(const uint4 *) ((const uint32_t *) A + (int64_t) (k0/2 + r)*lda + c*4);
            } else {
                const int r = l >> 2;
                const int c = l & 3;
                ra[i] = r < mA ? *(const uint4 *) (A + (int64_t) r*lda + k0 + c*8) : make_uint4(0, 0, 0, 0);
            }
            const int r = l >> 2;
            const int c = l & 3;
            if (B_BLOCKED) {
                // [k/32][128 rows][32]: one B tile is one contiguous 8 KiB block
                rb[i] = *(const uint4 *) (B + (int64_t) (k0/32)*(BN*32) + r*32 + c*8);
            } else {
                rb[i] = r < nB ? *(const uint4 *) (B + (int64_t) r*ldb + k0 + c*8) : make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            if (A_PAIRS) {
                const int r  = l >> 5;
                const int c  = l & 31;
                const int sw = (r >> 2) << 3;
                *(uint4 *) &sm.As[buf][r][(c*4) ^ sw] = ra[i];
            } else {
                const int r  = l >> 2;
                const int c  = l & 3;
                const int rs = r ^ (c << 3);
                sm.As[buf][c*4 + 0][rs] = ra[i].x; sm.As[buf][c*4 + 1][rs] = ra[i].y;
                sm.As[buf][c*4 + 2][rs] = ra[i].z; sm.As[buf][c*4 + 3][rs] = ra[i].w;
            }
            const int r  = l >> 2;
            const int c  = l & 3;
            const int rs = r ^ (c << 3);
            sm.Bs[buf][c*4 + 0][rs] = rb[i].x; sm.Bs[buf][c*4 + 1][rs] = rb[i].y;
            sm.Bs[buf][c*4 + 2][rs] = rb[i].z; sm.Bs[buf][c*4 + 3][rs] = rb[i].w;
        }
    };

    const int nit = K / (2*BK2);
    gload(0);
    pre();   // runs while the first tile is in flight
    sstore(0);
    __syncthreads();

    for (int it = 0; it < nit; it++) {
        const int buf = it & 1;
        if (it + 1 < nit) {
            gload((it + 1)*2*BK2);
        }
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int   sw = (k2 >> 2) << 3;
            const uint4 a0 = *(const uint4 *) &sm.As[buf][k2][(ty*4) ^ sw];
            const uint4 a1 = *(const uint4 *) &sm.As[buf][k2][(64 + ty*4) ^ sw];
            const uint4 b0 = *(const uint4 *) &sm.Bs[buf][k2][(tx*4) ^ sw];
            const uint4 b1 = *(const uint4 *) &sm.Bs[buf][k2][(64 + tx*4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = __hfma2(ai, bj, h[i][j]);
                }
            }
        }
        if (((it + 1)*BK2) % fold_k2 == 0 || it + 1 == nit) {
            on_fold(((it + 1)*BK2 - 1) / fold_k2, acc, h);
            // restart the chains (zeroing here, once per fold, is cheaper than an HMUL2 select
            // inside the loop, which compiled to both products plus a MOV per chain)
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    h[i][j] = make_half2(0.0f, 0.0f);
                }
            }
        }
        if (it + 1 < nit) {
            sstore(buf ^ 1);
        }
        __syncthreads();
    }
}

// Row (A / m) and column (B / n) of a thread's i-th / j-th output.
static __device__ __forceinline__ int fa_fold_row(const int i) { return (i < 4 ? 0 : 64) + (threadIdx.x >> 4)*4 + (i & 3); }
static __device__ __forceinline__ int fa_fold_col(const int j) { return (j < 4 ? 0 : 64) + (threadIdx.x & 15)*4 + (j & 3); }

// QK + softmax, two CTAs per SM. k = D is short (8 smem tiles at D = 256), so with one CTA per SM
// the prologue load and the softmax epilogue sat exposed (main loop 72% of peak, epilogue +20%).
// Here the fp16 chains run over the whole of D (two lanes of D/4 products each, then one fp32
// fold -- still shorter chains than cuBLAS COMPUTE_16F's single D-long one), so no fp32
// accumulators live through the loop, and the smem tiles are single-buffered: <= 128 registers
// and ~17 KiB of smem, and a second CTA covers each one's latency.
static __global__ void __launch_bounds__(256, 2) fa_fold_qk2(
        const half * __restrict__ K16, const half * __restrict__ Q16,
        const half * __restrict__ mask, const float * __restrict__ mask_first, const int64_t s_mask,
        half * __restrict__ P, float * __restrict__ mt, float * __restrict__ lt,
        const int D, const int N, const int nt, const int C, const fa_fold_split g, const bool prefix = false) {
    union smem_t {
        struct { uint32_t As[BK2][BM]; uint32_t Bs[BK2][BN]; } t;
        float red[16][BN];
        uint2 Ps[BN*16];
    };
    __shared__ __align__(16) smem_t sm;
    __shared__ float colv[BN];
    __shared__ int   colq[BN];

    const int h   = blockIdx.z;
    const int m0  = blockIdx.x*TK;
    const int n0  = blockIdx.y*BN;
    const int t   = threadIdx.x;
    const int tx  = t & 15;
    const int ty  = t >> 4;
    const int nkv_c  = fa_fold_nkv_c(g, h, C);
    const int kv_off = fa_fold_kv_off(g, h);

    int cq = -1;
    if (t < BN) {
        const int n  = n0 + t;
        const int tq = n < N ? n % nt : 0;
        cq = n < N && (float) (kv_off + m0 + TK) > mask_first[tq] ? tq : -1;
        colq[t] = cq;
    }
    // block-uniform: does any column need its mask here, is this the ragged last tile?
    // (under a causal mask only the diagonal chunk does; everything else takes the plain path)
    const bool any_mask = __syncthreads_or(cq >= 0);
    const bool ragged   = m0 + TK > nkv_c;

    // prefix mask (compact causal): a key at or past mask_first[t] is masked for query t, so a tile
    // whose first key is past every column's mask_first computes to exactly P = 0, m_t = -inf,
    // l_t = 0 (see the end of this kernel). Write those without the GEMM.
    if (prefix) {
        bool cskip = true;
        if (t < BN) {
            const int n = n0 + t;
            cskip = n >= N || mask_first[n % nt] <= (float) (kv_off + m0);
        }
        if (__syncthreads_and(cskip)) {
#pragma unroll
            for (int ih = 0; ih < 2; ih++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int idx = t + 256*k;
                    const int kbl = idx >> 9;
                    const int r   = (idx >> 2) & 127;
                    *(uint4 *) (P + (((int64_t) (h*gridDim.y + blockIdx.y)*(C/32) + m0/32 + ih*2 + kbl)*BN + r)*32 + (idx & 3)*8) =
                        make_uint4(0, 0, 0, 0);
                }
            }
            if (t < BN && n0 + t < N) {
                const int64_t o = ((int64_t) h*(C/TK) + blockIdx.x)*N + n0 + t;
                mt[o] = -INFINITY;
                lt[o] = 0.0f;
            }
            return;
        }
    }

    const half * A = K16 + ((int64_t) h*C + m0)*D;
    const half * B = Q16 + ((int64_t) (h / g.nsplit)*N + n0)*D;
    const int nB = N - n0;

    half2 hh[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            hh[i][j] = make_half2(0.0f, 0.0f);
        }
    }
    uint4 ra[2], rb[2];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            ra[i] = *(const uint4 *) (A + (int64_t) r*D + k0 + c*8);
            rb[i] = r < nB ? *(const uint4 *) (B + (int64_t) r*D + k0 + c*8) : make_uint4(0, 0, 0, 0);
        }
    };
    const int nit = D / (2*BK2);
    gload(0);
    for (int it = 0; it < nit; it++) {
        if (it > 0) {
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l  = t + 256*i;
            const int r  = l >> 2;
            const int c  = l & 3;
            const int rs = r ^ (c << 3);
            sm.t.As[c*4 + 0][rs] = ra[i].x; sm.t.As[c*4 + 1][rs] = ra[i].y;
            sm.t.As[c*4 + 2][rs] = ra[i].z; sm.t.As[c*4 + 3][rs] = ra[i].w;
            sm.t.Bs[c*4 + 0][rs] = rb[i].x; sm.t.Bs[c*4 + 1][rs] = rb[i].y;
            sm.t.Bs[c*4 + 2][rs] = rb[i].z; sm.t.Bs[c*4 + 3][rs] = rb[i].w;
        }
        __syncthreads();
        if (it + 1 < nit) {
            gload((it + 1)*2*BK2);
        }
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int   sw = (k2 >> 2) << 3;
            const uint4 a0 = *(const uint4 *) &sm.t.As[k2][(ty*4) ^ sw];
            const uint4 a1 = *(const uint4 *) &sm.t.As[k2][(64 + ty*4) ^ sw];
            const uint4 b0 = *(const uint4 *) &sm.t.Bs[k2][(tx*4) ^ sw];
            const uint4 b1 = *(const uint4 *) &sm.t.Bs[k2][(64 + tx*4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            // j outer and b in the first slot: same products per chain (bit-identical), but ptxas then leaves
            // 163 instead of 300 same-bank source pairs after bankfix (qk2 181.8 -> 177.7 ms on the model shape)
#pragma unroll
            for (int j = 0; j < 8; j++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    hh[i][j] = __hfma2(bj, ai, hh[i][j]);
                }
            }
        }
    }

    // logits in log2 units: log2(e) * (4*S + mask) (Q carried scale/4), from the one fold; keys
    // past nkv_c are -inf. The whole softmax state (m_t, M, the running max) is kept in log2
    // units so every exponential is a bare exp2f.
    float acc[8][8];
    float cmax[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        cmax[j] = -FLT_MAX/2.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            acc[i][j] = fold_h(hh[i][j]) * (0x1p114f*1.44269504088896341f);
        }
        if (any_mask || ragged) {
            const int tq = colq[fa_fold_col(j)];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int key = fa_fold_row(i);
                if (m0 + key >= nkv_c) {
                    acc[i][j] = -INFINITY;
                } else if (tq >= 0) {
                    acc[i][j] += 1.44269504088896341f*__half2float(mask[(int64_t) tq*s_mask + kv_off + m0 + key]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            cmax[j] = fmaxf(cmax[j], acc[i][j]);
        }
    }
    __syncthreads();   // all reads of the tiles are done: red aliases them
#pragma unroll
    for (int j = 0; j < 8; j++) {
        sm.red[ty][fa_fold_col(j)] = cmax[j];
    }
    __syncthreads();
    if (t < BN) {
        float m = -FLT_MAX/2.0f;
#pragma unroll
        for (int r = 0; r < 16; r++) {
            m = fmaxf(m, sm.red[r][t]);
        }
        colv[t] = m + 3.0f;   // FATTN_KQ_MAX_OFFSET (3 ln 2) in log2 units: every p <= 1/8
    }
    __syncthreads();
    float csum[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        csum[j] = 0.0f;
    }
    // P = exp(v - m_t) in f16, staged through smem in two 64-key halves and written as whole
    // 8 KiB blocks of the [h][n/128][key/32][128][32] layout that fa_fold_pv reads
#pragma unroll
    for (int ih = 0; ih < 2; ih++) {
        if (ih > 0) {
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int c = fa_fold_col(j);
            const float m = colv[c];
            half pp[4];
#pragma unroll
            for (int q = 0; q < 4; q++) {
                // v = -inf gives 0; a fully masked column has m ~ -FLT_MAX/2 and v = -inf, also 0
                const float p = exp2f(acc[ih*4 + q][j] - m);
                pp[q] = __float2half(p);
                csum[j] += p;
            }
            sm.Ps[c*16 + (ty ^ (((c >> 2) & 7) << 1))] = *(const uint2 *) pp;
        }
        __syncthreads();
        const uint4 * Ps = (const uint4 *) sm.Ps;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int idx = t + 256*k;
            const int kbl = idx >> 9;
            const int r   = (idx >> 2) & 127;
            const int q   = kbl*4 + (idx & 3);
            *(uint4 *) (P + (((int64_t) (h*gridDim.y + blockIdx.y)*(C/32) + m0/32 + ih*2 + kbl)*BN + r)*32 + (idx & 3)*8) =
                Ps[r*8 + (q ^ ((r >> 2) & 7))];
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; j++) {
        sm.red[ty][fa_fold_col(j)] = csum[j];
    }
    __syncthreads();
    if (t < BN) {
        const int n = n0 + t;
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < 16; r++) {
            s += sm.red[r][t];
        }
        if (n < N) {
            const int64_t o = ((int64_t) h*(C/TK) + blockIdx.x)*N + n;
            mt[o] = colv[t] <= -FLT_MAX/8.0f ? -INFINITY : colv[t];
            lt[o] = s;
        }
    }
}

// grid (DV/BM, ceil(N/BN), nhkv).  O_out = O_in*corr + sum_t exp(m_t - M) * V P_t
static __global__ void __launch_bounds__(256, 1) fa_fold_pv(
        const half2 * __restrict__ Vp, const half * __restrict__ P,
        const float * __restrict__ mt, const float * __restrict__ lt,
        const float * __restrict__ m_in, float * __restrict__ m_out, float * __restrict__ l_state,
        const float * __restrict__ O_in, float * __restrict__ O_out,
        const int DV, const int N, const int C, const int ntile,
        const float * __restrict__ mask_first = nullptr, const int nt = 0, const fa_fold_split g = {}, const bool prefix = false) {
    __shared__ __align__(16) fa_fold_tiles<true> sm;
    __shared__ float corr[BN];
    __shared__ float fac[MAXTILE][BN];   // exp(m_t - M) per tile and column

    const int h  = blockIdx.z;
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BN;
    const int ty = threadIdx.x >> 4;

    auto prologue = [&]() {
        if (threadIdx.x < BN) {
            const int n = n0 + threadIdx.x;
            float M = -INFINITY, c = 0.0f;
            if (n < N) {
                const float mo = m_in[(int64_t) h*N + n];
                M = mo;
                for (int tt = 0; tt < ntile; tt++) {
                    M = fmaxf(M, mt[((int64_t) h*(C/TK) + tt)*N + n]);
                }
                c = mo == -INFINITY ? 0.0f : exp2f(mo - M);
                if (blockIdx.x == 0) {
                    float l = l_state[(int64_t) h*N + n]*c;
                    if (M != -INFINITY) {
                        for (int tt = 0; tt < ntile; tt++) {
                            const int64_t o = ((int64_t) h*(C/TK) + tt)*N + n;
                            const float mm = mt[o];
                            l += mm == -INFINITY ? 0.0f : lt[o]*exp2f(mm - M);
                        }
                    }
                    l_state[(int64_t) h*N + n] = l;
                    m_out[(int64_t) h*N + n]   = M;
                }
                for (int tt = 0; tt < ntile; tt++) {
                    const float mm = mt[((int64_t) h*(C/TK) + tt)*N + n];
                    fac[tt][threadIdx.x] = (mm == -INFINITY || M == -INFINITY) ? 0.0f : exp2f(mm - M);
                }
            } else {
                for (int tt = 0; tt < ntile; tt++) {
                    fac[tt][threadIdx.x] = 0.0f;
                }
            }
            corr[threadIdx.x] = c;
        }
    };

    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
        }
    }
    // prefix mask: tiles past this column block's last visible key hold P = 0 (fa_fold_qk2), and
    // their factor is 0: stop before them (at least one tile, so the loop shape is unchanged)
    int ntile_run = ntile;
    if (prefix) {
        __shared__ int kmax;
        if (threadIdx.x == 0) {
            kmax = 0;
        }
        __syncthreads();
        if (threadIdx.x < BN && n0 + (int) threadIdx.x < N) {
            atomicMax(&kmax, (int) mask_first[(n0 + threadIdx.x) % nt]);
        }
        __syncthreads();
        const int vis = kmax - fa_fold_kv_off(g, h);   // visible keys of this chunk for the block
        ntile_run = max(1, min(ntile, (vis + TK - 1)/TK));
    }
    fa_fold_mainloop<true, TK/2, true>(sm,
        (const half *) (Vp + (int64_t) h*(C/2)*DV + m0), DV, BM,
        P + ((int64_t) h*gridDim.y + blockIdx.y)*C*BN, C, N - n0,
        ntile_run*TK, acc, [&](int tt, float (&a)[8][8], half2 (&hh)[8][8]) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const float f = fac[tt][fa_fold_col(j)];
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    a[i][j] += fold_h(hh[i][j]) * f;
                }
            }
        }, prologue);   // the mainloop's first __syncthreads orders fac/corr for everyone

    // all 16 loads of the running output first (the fp16 chains' registers are free now), so their
    // latencies overlap instead of queueing behind each store
    float4 oin[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = min(n0 + fa_fold_col(j), N - 1);
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            oin[j][ih] = *(const float4 *) (O_in + ((int64_t) h*N + n)*DV + m0 + ih*64 + ty*4);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int c = fa_fold_col(j);
        const int n = n0 + c;
        if (n >= N) {
            continue;
        }
        const float cr = corr[c];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int64_t o = ((int64_t) h*N + n)*DV + m0 + ih*64 + ty*4;
            const float4 oi = oin[j][ih];
            float4 v;
            v.x = oi.x*cr + acc[ih*4 + 0][j]*0x1p112f;
            v.y = oi.y*cr + acc[ih*4 + 1][j]*0x1p112f;
            v.z = oi.z*cr + acc[ih*4 + 2][j]*0x1p112f;
            v.w = oi.w*cr + acc[ih*4 + 3][j]*0x1p112f;
            *(float4 *) (O_out + o) = v;
        }
    }
}

// fa_fold_pv with gemm_fold_kernel_u2's main loop (OPTLOG 238): load/store offsets computed once, the
// tile loop unrolled by two with compile-time smem buffers, chains restarted by HMUL2, and the fold as
// HADD2 -> f32 without the 2^-112 bit trick (subnormal chain sums and tiny products are kept instead of
// flushed). Same products, chains (128 keys) and fold points. GGML_CUDA_FA_PV2=0: fa_fold_pv.
static __global__ void __launch_bounds__(256, 1) fa_fold_pv2(
        const half2 * __restrict__ Vp, const half * __restrict__ P,
        const float * __restrict__ mt, const float * __restrict__ lt,
        const float * __restrict__ m_in, float * __restrict__ m_out, float * __restrict__ l_state,
        const float * __restrict__ O_in, float * __restrict__ O_out,
        const int DV, const int N, const int C, const int ntile,
        const float * __restrict__ mask_first = nullptr, const int nt = 0, const fa_fold_split g = {}, const bool prefix = false) {
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BN];
    __shared__ float corr[BN];
    __shared__ float fac[MAXTILE][BN];   // exp(m_t - M) per tile and column

    const int h  = blockIdx.z;
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BN;
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;

    auto prologue = [&]() {
        if (t < BN) {
            const int n = n0 + t;
            float M = -INFINITY, c = 0.0f;
            if (n < N) {
                // all tile maxima loaded at once (independent loads; ntile <= MAXTILE)
                float mm[MAXTILE];
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) {
                    mm[tt] = tt < ntile ? mt[((int64_t) h*(C/TK) + tt)*N + n] : -INFINITY;
                }
                const float mo = m_in[(int64_t) h*N + n];
                M = mo;
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) {
                    M = fmaxf(M, mm[tt]);
                }
                c = mo == -INFINITY ? 0.0f : exp2f(mo - M);
                if (blockIdx.x == 0) {
                    float l = l_state[(int64_t) h*N + n]*c;
                    if (M != -INFINITY) {
                        float lv[MAXTILE];
#pragma unroll
                        for (int tt = 0; tt < MAXTILE; tt++) {
                            lv[tt] = tt < ntile ? lt[((int64_t) h*(C/TK) + tt)*N + n] : 0.0f;
                        }
                        for (int tt = 0; tt < ntile; tt++) {
                            l += mm[tt] == -INFINITY ? 0.0f : lv[tt]*exp2f(mm[tt] - M);
                        }
                    }
                    l_state[(int64_t) h*N + n] = l;
                    m_out[(int64_t) h*N + n]   = M;
                }
#pragma unroll
                for (int tt = 0; tt < MAXTILE; tt++) {
                    if (tt < ntile) {
                        fac[tt][t] = (mm[tt] == -INFINITY || M == -INFINITY) ? 0.0f : exp2f(mm[tt] - M);
                    }
                }
            } else {
                for (int tt = 0; tt < ntile; tt++) {
                    fac[tt][t] = 0.0f;
                }
            }
            corr[t] = c;
        }
    };

    int ntile_run = ntile;
    if (prefix) {
        __shared__ int kmax;
        if (t == 0) {
            kmax = 0;
        }
        __syncthreads();
        if (t < BN && n0 + t < N) {
            atomicMax(&kmax, (int) mask_first[(n0 + t) % nt]);
        }
        __syncthreads();
        const int vis = kmax - fa_fold_kv_off(g, h);
        ntile_run = max(1, min(ntile, (vis + TK - 1)/TK));
    }

    float acc[8][8];
    half2 hh[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
            hh[i][j]  = make_half2(0.0f, 0.0f);
        }
    }

    // A: V as half2 key pairs [k2][DV] (DV words per k2 row); B: P blocked [key/32][128][32]
    const uint32_t * A = (const uint32_t *) (Vp + (int64_t) h*(C/2)*DV + m0);
    const half     * B = P + ((int64_t) h*gridDim.y + blockIdx.y)*C*BN;
    const uint4 * pa[2];
    const uint4 * pb[2];
    int sa[2], sb[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int l = t + 256*i;
        const int ra_ = l >> 5, ca = l & 31;
        pa[i] = (const uint4 *) (A + (int64_t) ra_*DV + ca*4);
        sa[i] = ra_*BM + ((ca*4) ^ ((ra_ >> 2) << 3));
        const int rb_ = l >> 2, cb = l & 3;
        pb[i] = (const uint4 *) (B + rb_*32 + cb*8);
        sb[i] = (cb*4)*BN + (rb_ ^ (cb << 3));
    }
    const int64_t da = (int64_t) BK2*DV/4;   // uint4 per tile along A
    constexpr int db = BN*32/8;              // uint4 per tile along B
    int ao[4], bo[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
        ao[q] = (ty*4) ^ (q << 3);
        bo[q] = (tx*4) ^ (q << 3);
    }

    uint4 ra[2], rb[2];
    auto gload = [&](const int it) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            ra[i] = pa[i][it*da];
            rb[i] = pb[i][it*db];
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            *(uint4 *) (&As[buf][0][0] + sa[i]) = ra[i];
            uint32_t * b = &Bs[buf][0][0] + sb[i];
            b[0] = rb[i].x; b[BN] = rb[i].y; b[2*BN] = rb[i].z; b[3*BN] = rb[i].w;
        }
    };
    auto tile = [&](const int buf, const bool restart) {
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const uint4 a0 = *(const uint4 *) &As[buf][k2][ao[k2 >> 2]];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][ao[k2 >> 2] + 64];
            const uint4 b0 = *(const uint4 *) &Bs[buf][k2][bo[k2 >> 2]];
            const uint4 b1 = *(const uint4 *) &Bs[buf][k2][bo[k2 >> 2] + 64];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    hh[i][j] = k2 == 0 && restart ? __hmul2(ai, bj) : __hfma2(ai, bj, hh[i][j]);
                }
            }
        }
    };
    auto fold = [&](const int tt) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float f = fac[tt][fa_fold_col(j)];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                acc[i][j] += __half2float(__hadd(__low2half(hh[i][j]), __high2half(hh[i][j]))) * f;
            }
        }
    };

    constexpr int TPF = TK/2/BK2;         // tiles per fold (128 keys)
    const int ntl = ntile_run*TPF;        // tiles of 32 keys, a multiple of 4
    gload(0);
    prologue();
    sstore(0);
    __syncthreads();
    for (int it = 0; it < ntl; it += 2) {
        gload(it + 1);
        tile(0, it % TPF == 0);
        sstore(1);
        __syncthreads();
        if (it + 2 < ntl) {
            gload(it + 2);
        }
        tile(1, false);
        if ((it + 2) % TPF == 0) {
            fold((it + 2)/TPF - 1);
        }
        if (it + 2 < ntl) {
            sstore(0);
        }
        __syncthreads();
    }

    float4 oin[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = min(n0 + fa_fold_col(j), N - 1);
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            oin[j][ih] = *(const float4 *) (O_in + ((int64_t) h*N + n)*DV + m0 + ih*64 + ty*4);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int c = fa_fold_col(j);
        const int n = n0 + c;
        if (n >= N) {
            continue;
        }
        const float cr = corr[c];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int64_t o = ((int64_t) h*N + n)*DV + m0 + ih*64 + ty*4;
            const float4 oi = oin[j][ih];
            float4 v;
            v.x = oi.x*cr + acc[ih*4 + 0][j];
            v.y = oi.y*cr + acc[ih*4 + 1][j];
            v.z = oi.z*cr + acc[ih*4 + 2][j];
            v.w = oi.w*cr + acc[ih*4 + 3][j];
            *(float4 *) (O_out + o) = v;
        }
    }
}

// Oracle sparse attention (diagnostic, GGML_CUDA_FA_ORACLE_DELTA): Mq[kvh][n] = max over this
// chunk's tiles and splits of mt (the per-tile max, log2 units), accumulated across chunks.
static __global__ void fa_oracle_max(const float * __restrict__ mt, float * __restrict__ Mq,
        const int N, const int C, const int ntile, const int nsplit, const int nz) {
    const int n = blockIdx.x*blockDim.x + threadIdx.x;
    const int kvh = blockIdx.y;
    if (n >= N) {
        return;
    }
    float M = Mq[(int64_t) kvh*N + n];
    for (int sp = 0; sp < nsplit; sp++) {
        const int z = kvh*nsplit + sp;
        for (int tt = 0; tt < ntile; tt++) {
            M = fmaxf(M, mt[((int64_t) z*(C/TK) + tt)*N + n]);
        }
    }
    Mq[(int64_t) kvh*N + n] = M;
}

// Zero every P entry whose logit is more than dlog2 below its query's global max and recompute the
// tile row sums lt over the kept keys, so the PV pass computes softmax over the kept set only.
// grid (ntile, ceil(N/BN), nz), block BN: one thread per query column, 128 keys each.
static __global__ void fa_oracle_prune(half * __restrict__ P, const float * __restrict__ mt, float * __restrict__ lt,
        const float * __restrict__ Mq, const float dlog2, const int N, const int C, const int nsplit,
        unsigned long long * __restrict__ kept, unsigned long long * __restrict__ total) {
    const int tt = blockIdx.x;
    const int nb = blockIdx.y;
    const int z  = blockIdx.z;
    const int r  = threadIdx.x;
    const int n  = nb*BN + r;
    unsigned long long k = 0, tot = 0;
    if (n < N) {
        const int64_t o = ((int64_t) z*(C/TK) + tt)*N + n;
        const float m = mt[o];
        const float thr = Mq[(int64_t) (z/nsplit)*N + n] - dlog2;
        float s = 0.0f;
        for (int kb = 0; kb < TK/32; kb++) {
            half * row = P + (((int64_t) (z*gridDim.y + nb)*(C/32) + tt*(TK/32) + kb)*BN + r)*32;
            for (int e = 0; e < 32; e++) {
                const float p = __half2float(row[e]);
                if (p > 0.0f) {
                    tot++;
                    if (log2f(p) + m >= thr) {
                        s += p;
                        k++;
                    } else {
                        row[e] = __float2half(0.0f);
                    }
                }
            }
        }
        lt[o] = s;
    }
    atomicAdd(kept, k);
    atomicAdd(total, tot);
}

// dst[d, head, t] = sum_split w*O / sum_split w*l, w = exp(m_split - max m): the split merge
static __global__ void fa_fold_finalize(
        const float * __restrict__ O, const float * __restrict__ m, const float * __restrict__ l,
        float * __restrict__ dst, const int DV, const int nt, const int N, const int gqa, const int nsplit,
        const int64_t dst_s1, const int64_t dst_s2) {
    const int t   = blockIdx.x;
    const int hq  = blockIdx.y;
    const int kvh = blockIdx.z;
    const int n   = hq*nt + t;
    float M = -INFINITY;
    for (int sp = 0; sp < nsplit; sp++) {
        M = fmaxf(M, m[(int64_t) (kvh*nsplit + sp)*N + n]);
    }
    float w[8];
    float L = 0.0f;
    for (int sp = 0; sp < nsplit; sp++) {
        const int64_t o = (int64_t) (kvh*nsplit + sp)*N + n;
        w[sp] = m[o] == -INFINITY ? 0.0f : exp2f(m[o] - M);
        L += w[sp]*l[o];
    }
    const float inv = L > 0.0f ? 1.0f/L : 0.0f;
    float * dh = dst + (int64_t) t*dst_s2 + (int64_t) (kvh*gqa + hq)*dst_s1;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        float v = 0.0f;
        for (int sp = 0; sp < nsplit; sp++) {
            v += w[sp]*O[((int64_t) (kvh*nsplit + sp)*N + n)*DV + d];
        }
        dh[d] = v*inv;
    }
}

} // namespace fa_fold

static int ggml_cuda_fa_fold_env(const char * name, int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

// fa_fold_qk2 / fa_fold_pv2 through their register-bank-fixed SASS (fattn-fold-sass.h, built from the
// fa_fold namespace by p100-handoff/tools/sass-gemm/facubin.py; bit-identical, only register names
// differ). Loaded once per device; false (run the compiled kernel) if unavailable or GGML_CUDA_FA_SASS=0.
static bool fa_fold_sass_launch(const int which, const dim3 grid, cudaStream_t stream, void ** args) {
    static const bool on = ggml_cuda_fa_fold_env("GGML_CUDA_FA_SASS", 1) != 0;
    static int        state[GGML_CUDA_MAX_DEVICES][2] = {};
    static CUfunction fn[GGML_CUDA_MAX_DEVICES][2]    = {};
    if (!on) {
        return false;
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int & st = state[dev][which];
    if (st == 0) {
        CUmodule mod;
        const void * img  = which == 0 ? (const void *) fattn_fold_sass_qk2 : (const void *) fattn_fold_sass_pv2;
        const char * name = which == 0 ? fattn_fold_sass_name_qk2 : fattn_fold_sass_name_pv2;
        // GGML_CUDA_FA_SASS_DIR (development): load qk2.cubin / pv2.cubin from a directory instead
        const char * dir = getenv("GGML_CUDA_FA_SASS_DIR");
        CUresult r;
        if (dir) {
            const std::string path = std::string(dir) + (which == 0 ? "/qk2.cubin" : "/pv2.cubin");
            r = cuModuleLoad(&mod, path.c_str());
        } else {
            r = cuModuleLoadData(&mod, img);
        }
        st = r == CUDA_SUCCESS && cuModuleGetFunction(&fn[dev][which], mod, name) == CUDA_SUCCESS ? 1 : -1;
        if (st < 0) {
            GGML_LOG_WARN("%s: bank-fixed fold attention SASS unavailable, using the compiled kernel\n", __func__);
        }
    }
    if (st < 0) {
        return false;
    }
    if (cuLaunchKernel(fn[dev][which], grid.x, grid.y, grid.z, 256, 1, 1, 0, stream, args, nullptr) != CUDA_SUCCESS) {
        GGML_ABORT("cuLaunchKernel failed for the fold attention SASS");
    }
    return true;
}

static bool ggml_cuda_fa_fold_usable(const ggml_tensor * dst) {
    static const int mode = ggml_cuda_fa_fold_env("GGML_CUDA_FA_FOLD", 1);
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    return mode != 0 && !ggml_cuda_fa_gemm_prec32() &&
        K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0 &&
        K->ne[0] == V->ne[0] && K->ne[0] % 128 == 0 && K->ne[0] <= 512 &&
        K->ne[2] == V->ne[2] && Q->ne[1]*(Q->ne[2]/K->ne[2]) <= 65535*fa_fold::BN;
}

// GGML_CUDA_FA_SPARSITY=1 (diagnostic only, slow: syncs after every chunk): how much of each
// query's softmax mass sits in each 128-key tile, from the kernel's own per-tile max and sum.
// Prints, per call, the fraction of (128-query x 128-key) blocks a block-sparse kernel would
// have to keep to cover 99 / 99.9 / 99.99 % of every query's mass. Changes no output.
struct fa_sparsity_acc {
    int64_t N = 0, T = 0, nhkv = 0;
    std::vector<float> lm;   // [kvh][n][tile] log2 mass
    void init(int64_t N_, int64_t T_, int64_t nhkv_) {
        N = N_; T = T_; nhkv = nhkv_;
        lm.assign((size_t) (nhkv*N*T), -INFINITY);
    }
    void add(const std::vector<float> & mt, const std::vector<float> & lt, int64_t nz, int64_t nsplit,
             int64_t ctiles, int64_t tile0_of_split_c /* c/TK */, int64_t tiles_per_split) {
        for (int64_t z = 0; z < nz; z++) {
            const int64_t kvh = z / nsplit, sp = z % nsplit;
            for (int64_t tt = 0; tt < ctiles; tt++) {
                const int64_t tg = sp*tiles_per_split + tile0_of_split_c + tt;
                if (tg >= T) continue;
                for (int64_t n = 0; n < N; n++) {
                    const float m = mt[((size_t) (z*ctiles + tt))*N + n];
                    const float l = lt[((size_t) (z*ctiles + tt))*N + n];
                    lm[((size_t) (kvh*N + n))*T + tg] = (m == -INFINITY || !(l > 0.0f)) ? -INFINITY : m + log2f(l);
                }
            }
        }
    }
    void report(int dev, int64_t nkv, int64_t nt) {
        static std::mutex mu;
        static int call = 0;
        const double taus[3] = {0.99, 0.999, 0.9999};
        double qfrac[3] = {0, 0, 0}, bfrac[3] = {0, 0, 0};
        int64_t nq = 0, nb = 0;
        std::vector<std::pair<float, int>> v(T);
        std::vector<uint8_t> need[3];
        for (int k = 0; k < 3; k++) need[k].assign((size_t) T, 0);
        for (int64_t kvh = 0; kvh < nhkv; kvh++) {
            for (int64_t n0 = 0; n0 < N; n0 += 128) {
                for (int k = 0; k < 3; k++) std::fill(need[k].begin(), need[k].end(), 0);
                int64_t valid_tiles = 0;
                for (int64_t n = n0; n < std::min(N, n0 + 128); n++) {
                    const float * row = &lm[((size_t) (kvh*N + n))*T];
                    float mx = -INFINITY;
                    for (int64_t t = 0; t < T; t++) mx = std::max(mx, row[t]);
                    if (mx == -INFINITY) continue;
                    double tot = 0; int64_t nv = 0;
                    for (int64_t t = 0; t < T; t++) {
                        const float w = row[t] == -INFINITY ? 0.0f : exp2f(row[t] - mx);
                        v[t] = {w, (int) t}; tot += w; nv += w > 0.0f;
                    }
                    valid_tiles = std::max(valid_tiles, nv);
                    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.first > b.first; });
                    for (int k = 0; k < 3; k++) {
                        double c = 0; int64_t i = 0;
                        while (i < T && c < taus[k]*tot) { c += v[i].first; need[k][v[i].second] = 1; i++; }
                        qfrac[k] += nv ? (double) i / nv : 0;
                    }
                    nq++;
                }
                if (valid_tiles == 0) continue;
                for (int k = 0; k < 3; k++) {
                    int64_t cnt = 0;
                    for (int64_t t = 0; t < T; t++) cnt += need[k][t];
                    bfrac[k] += (double) cnt / valid_tiles;
                }
                nb++;
            }
        }
        std::lock_guard<std::mutex> lock(mu);
        fprintf(stderr, "fa_sparsity dev %d call %d nkv %lld nt %lld | per-query tiles kept 99%%/99.9%%/99.99%%: %.3f %.3f %.3f"
                " | 128x128 blocks kept: %.3f %.3f %.3f\n", dev, call++, (long long) nkv, (long long) nt,
                qfrac[0]/std::max<int64_t>(1, nq), qfrac[1]/std::max<int64_t>(1, nq), qfrac[2]/std::max<int64_t>(1, nq),
                bfrac[0]/std::max<int64_t>(1, nb), bfrac[1]/std::max<int64_t>(1, nb), bfrac[2]/std::max<int64_t>(1, nb));
    }
};

static void ggml_cuda_flash_attn_ext_fold(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fa_fold;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t D    = K->ne[0];
    const int64_t DV   = V->ne[0];
    const int64_t nt   = Q->ne[1];
    const int64_t nh   = Q->ne[2];
    const int64_t nkv  = K->ne[1];
    const int64_t nhkv = K->ne[2];
    const int64_t ns   = Q->ne[3];
    const int64_t gqa  = nh / nhkv;
    const int64_t N    = nt*gqa;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();

    // 2048 keys per chunk: +2.2% prefill at 262k over 1024. OPTLOG 226 had cut it to 1024 for VRAM; the
    // compact causal mask (OPTLOG 248) gave that back: GPU0 min free at 262k, vision, -ub 2048: 656 MiB
    static const int chunk_env  = ggml_cuda_fa_fold_env("GGML_CUDA_FA_FOLD_CHUNK", 2048);
    static const int nsplit_env = ggml_cuda_fa_fold_env("GGML_CUDA_FA_FOLD_SPLIT", 2);
    const int64_t nsplit = std::max(1, std::min(8, nsplit_env));
    const int64_t Lsp = ((nkv + nsplit - 1)/nsplit + TK - 1)/TK*TK;   // keys per split
    const int64_t C = std::max<int64_t>(TK, (std::min<int64_t>(std::min(chunk_env, MAXTILE*TK), Lsp) + TK - 1)/TK*TK);
    const int64_t nz = nhkv*nsplit;   // virtual heads

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<half>  Qf16(pool, D*N*nhkv);
    ggml_cuda_pool_alloc<half>  K16(pool, D*C*nz);
    ggml_cuda_pool_alloc<half2> Vp(pool, DV*(C/2)*nz);
    ggml_cuda_pool_alloc<half>  P(pool, C*((N + BN - 1)/BN*BN)*nz);
    ggml_cuda_pool_alloc<float> mt(pool, (C/TK)*N*nz);
    ggml_cuda_pool_alloc<float> lt(pool, (C/TK)*N*nz);
    ggml_cuda_pool_alloc<float> O(pool, DV*N*nz);
    ggml_cuda_pool_alloc<float> O2(pool, DV*N*nz);
    ggml_cuda_pool_alloc<float> mA(pool, N*nz);
    ggml_cuda_pool_alloc<float> mB(pool, N*nz);
    ggml_cuda_pool_alloc<float> l_state(pool, N*nz);
    ggml_cuda_pool_alloc<float> mask_first(pool, nt);

    const int64_t dst_s1 = dst->nb[1]/sizeof(float);
    const int64_t dst_s2 = dst->nb[2]/sizeof(float);
    ggml_cuda_pool_alloc<half> mask_stair(pool);
    const fattn_gemm_mask_view mview = fattn_gemm_mask_get(mask, nt, nkv, mask_stair, stream);
    const int64_t s_mask = mview.s;
    // compact mask: every row is a prefix, so fully masked tiles may be skipped (GGML_CUDA_FA_PREFIX_SKIP=0: off)
    static const bool prefix_env = ggml_cuda_fa_fold_env("GGML_CUDA_FA_PREFIX_SKIP", 1) != 0;
    const bool prefix = prefix_env && mask->type == GGML_TYPE_I32;

    fattn_gemm_mask_first_nz<256><<<nt, 256, 0, stream>>>(mview.data, mask_first.ptr, s_mask, (int) nkv);
    CUDA_CHECK(cudaGetLastError());

    // GGML_CUDA_FA_DUMP=dir (diagnostic): for calls with >= GGML_CUDA_FA_DUMP_MIN_NKV keys (default
    // 250000), write Q (f32), K and V (raw q4_0 rows), the per-query first masked key and a meta line,
    // so attention approximations can be scored offline against the exact result.
    {
        static const char * dump_dir = getenv("GGML_CUDA_FA_DUMP");
        static const int64_t dump_min = ggml_cuda_fa_fold_env("GGML_CUDA_FA_DUMP_MIN_NKV", 250000);
        static std::mutex dump_mu;
        static int dump_call = 0;
        if (dump_dir && nkv >= dump_min && ns == 1) {
            int id;
            {
                std::lock_guard<std::mutex> lock(dump_mu);
                id = dump_call++;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            char fn[1024];
            auto wr = [&](const char * what, const void * h, size_t n) {
                snprintf(fn, sizeof(fn), "%s/c%03d_d%d_%s.bin", dump_dir, id, ctx.device, what);
                FILE * f = fopen(fn, "wb");
                GGML_ASSERT(f);
                fwrite(h, 1, n, f);
                fclose(f);
            };
            {   // Q: [D, nt, nh] f32, rows gathered contiguously
                std::vector<float> h((size_t) D*nt*nh);
                for (int64_t hh = 0; hh < nh; hh++) {
                    CUDA_CHECK(cudaMemcpy2D(h.data() + (size_t) hh*nt*D, D*sizeof(float),
                        (const char *) Q->data + hh*Q->nb[2], Q->nb[1], D*sizeof(float), nt, cudaMemcpyDeviceToHost));
                }
                wr("q", h.data(), h.size()*sizeof(float));
            }
            for (int kv = 0; kv < 2; kv++) {   // K, V: [nhkv][nkv][row bytes] raw q4_0
                const ggml_tensor * t = kv ? V : K;
                const size_t rb = ggml_row_size(t->type, t->ne[0]);
                std::vector<char> h((size_t) rb*nkv*nhkv);
                for (int64_t hh = 0; hh < nhkv; hh++) {
                    CUDA_CHECK(cudaMemcpy2D(h.data() + (size_t) hh*nkv*rb, rb, (const char *) t->data + hh*t->nb[2], t->nb[1],
                        rb, nkv, cudaMemcpyDeviceToHost));
                }
                wr(kv ? "v" : "k", h.data(), h.size());
            }
            {
                std::vector<float> h(nt);
                CUDA_CHECK(cudaMemcpy(h.data(), mask_first.ptr, nt*sizeof(float), cudaMemcpyDeviceToHost));
                wr("maskfirst", h.data(), h.size()*sizeof(float));
            }
            snprintf(fn, sizeof(fn), "%s/c%03d_d%d_meta.txt", dump_dir, id, ctx.device);
            FILE * f = fopen(fn, "w");
            GGML_ASSERT(f);
            fprintf(f, "D %lld DV %lld nt %lld nh %lld nhkv %lld nkv %lld scale %.9g\n", (long long) D, (long long) DV,
                (long long) nt, (long long) nh, (long long) nhkv, (long long) nkv, scale);
            fclose(f);
        }
    }

    static const bool sparsity = ggml_cuda_fa_fold_env("GGML_CUDA_FA_SPARSITY", 0) != 0;
    // GGML_CUDA_FA_ORACLE_DELTA=d (natural-log units, diagnostic): drop keys whose logit is more than d
    // below the query's max over all keys; softmax over the rest. An upper bound on what any fast
    // key selector can achieve at that sparsity. Costs an extra QK pass.
    static const float oracle_delta = [] { const char * e = getenv("GGML_CUDA_FA_ORACLE_DELTA"); return e ? (float) atof(e) : 0.0f; }();
    const bool  oracle = oracle_delta > 0.0f;
    const float oracle_dlog2 = oracle_delta*1.44269504088896341f;
    ggml_cuda_pool_alloc<float> Mq(pool, oracle ? N*nhkv : 1);
    ggml_cuda_pool_alloc<unsigned long long> ocnt(pool, 2);
    fa_sparsity_acc sacc;
    std::vector<float> h_mt, h_lt;

    for (int64_t s = 0; s < ns; ++s) {
        if (sparsity) {
            sacc.init(N, nsplit*(Lsp/TK), nhkv);
        }
        for (int64_t kvh = 0; kvh < nhkv; ++kvh) {
            fattn_gemm_q_to_f16<<<dim3(nt, gqa, 1), 256, 0, stream>>>(
                (const char *) Q->data + s*Q->nb[3], Qf16.ptr + kvh*N*D,
                D, nt, Q->nb[1], Q->nb[2], kvh*gqa, scale*0.25f);
            CUDA_CHECK(cudaGetLastError());
        }
        float * O_cur = O.ptr;
        float * O_nxt = O2.ptr;
        float * m_cur = mA.ptr;
        float * m_nxt = mB.ptr;
        CUDA_CHECK(cudaMemsetAsync(O_cur, 0, DV*N*nz*sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(l_state.ptr, 0, N*nz*sizeof(float), stream));
        fattn_gemm_fill<<<(N*nz + 255)/256, 256, 0, stream>>>(m_cur, -INFINITY, N*nz);
        CUDA_CHECK(cudaGetLastError());

        if (oracle) {
            fattn_gemm_fill<<<(N*nhkv + 255)/256, 256, 0, stream>>>(Mq.ptr, -INFINITY, N*nhkv);
            CUDA_CHECK(cudaMemsetAsync(ocnt.ptr, 0, 2*sizeof(unsigned long long), stream));
            for (int64_t c = 0; c < Lsp; c += C) {
                const fa_fold_split g = {(int) nsplit, (int) Lsp, (int) c, (int) nkv};
                const int nkv_c = (int) std::min(C, Lsp - c);
                const int ntile = (nkv_c + TK - 1)/TK;
                const int nb  = (int) (D/QK4_0);
                const int kpb = 256/nb;
                const int npairs = ntile*TK/2;
                fa_fold_dequant<<<dim3((npairs + kpb - 1)/kpb, nz, 1), kpb*nb, 0, stream>>>(
                    (const char *) K->data + s*K->nb[3], (const char *) V->data + s*V->nb[3],
                    K16.ptr, Vp.ptr, K->nb[1], K->nb[2], V->nb[1], V->nb[2], (int) D, g, (int) C);
                fa_fold_qk2<<<dim3(ntile, (N + BN - 1)/BN, nz), 256, 0, stream>>>(K16.ptr, Qf16.ptr, mview.data,
                    mask_first.ptr, s_mask, P.ptr, mt.ptr, lt.ptr, (int) D, (int) N, (int) nt, (int) C, g);
                fa_oracle_max<<<dim3((N + 127)/128, nhkv, 1), 128, 0, stream>>>(mt.ptr, Mq.ptr, (int) N, (int) C, ntile, (int) nsplit, (int) nz);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        for (int64_t c = 0; c < Lsp; c += C) {
            const fa_fold_split g = {(int) nsplit, (int) Lsp, (int) c, (int) nkv};
            // the longest stream this chunk (split 0 is always full-length)
            const int nkv_c = (int) std::min(C, Lsp - c);
            const int ntile = (nkv_c + TK - 1)/TK;
            {
                const int nb  = (int) (D/QK4_0);
                const int kpb = 256/nb;   // key pairs per block
                const int npairs = ntile*TK/2;
                fa_fold_dequant<<<dim3((npairs + kpb - 1)/kpb, nz, 1), kpb*nb, 0, stream>>>(
                    (const char *) K->data + s*K->nb[3], (const char *) V->data + s*V->nb[3],
                    K16.ptr, Vp.ptr, K->nb[1], K->nb[2], V->nb[1], V->nb[2], (int) D, g, (int) C);
                CUDA_CHECK(cudaGetLastError());
            }
            {
                const dim3 grid(ntile, (N + BN - 1)/BN, nz);
                const half * a_k = K16.ptr; const half * a_q = Qf16.ptr; const half * a_m = mview.data;
                const float * a_mf = mask_first.ptr; int64_t a_sm = s_mask; half * a_p = P.ptr;
                float * a_mt = mt.ptr; float * a_lt = lt.ptr; int a_D = (int) D, a_N = (int) N, a_nt = (int) nt, a_C = (int) C;
                fa_fold_split a_g = g; bool a_pf = prefix;
                void * args[] = {&a_k, &a_q, &a_m, &a_mf, &a_sm, &a_p, &a_mt, &a_lt, &a_D, &a_N, &a_nt, &a_C, &a_g, &a_pf};
                if (!fa_fold_sass_launch(0, grid, stream, args)) {
                    fa_fold_qk2<<<grid, 256, 0, stream>>>(a_k, a_q, a_m, a_mf, a_sm,
                        a_p, a_mt, a_lt, a_D, a_N, a_nt, a_C, a_g, a_pf);
                }
                CUDA_CHECK(cudaGetLastError());
            }
            if (oracle) {
                fa_oracle_prune<<<dim3(ntile, (N + BN - 1)/BN, nz), BN, 0, stream>>>(P.ptr, mt.ptr, lt.ptr, Mq.ptr, oracle_dlog2,
                    (int) N, (int) C, (int) nsplit, ocnt.ptr, ocnt.ptr + 1);
                CUDA_CHECK(cudaGetLastError());
            }
            if (sparsity) {
                const size_t cnt = (size_t) (C/TK)*N*nz;
                h_mt.resize(cnt); h_lt.resize(cnt);
                CUDA_CHECK(cudaMemcpyAsync(h_mt.data(), mt.ptr, cnt*sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(h_lt.data(), lt.ptr, cnt*sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                // mt/lt are laid out [z][C/TK][N]; only the first ntile tiles of this chunk are written
                std::vector<float> cm((size_t) ntile*N*nz), cl((size_t) ntile*N*nz);
                for (int64_t z = 0; z < nz; z++) {
                    for (int64_t tt = 0; tt < ntile; tt++) {
                        memcpy(&cm[((size_t) (z*ntile + tt))*N], &h_mt[((size_t) (z*(C/TK) + tt))*N], N*sizeof(float));
                        memcpy(&cl[((size_t) (z*ntile + tt))*N], &h_lt[((size_t) (z*(C/TK) + tt))*N], N*sizeof(float));
                    }
                }
                // a split whose keys end before this chunk's ntile tiles leaves stale values: mask by key range
                for (int64_t z = 0; z < nz; z++) {
                    const int64_t off = (z % nsplit)*Lsp + c;
                    for (int64_t tt = 0; tt < ntile; tt++) {
                        if (off + tt*TK >= nkv || c + tt*TK >= Lsp) {
                            for (int64_t n = 0; n < N; n++) cm[((size_t) (z*ntile + tt))*N + n] = -INFINITY;
                        }
                    }
                }
                sacc.add(cm, cl, nz, nsplit, ntile, c/TK, Lsp/TK);
            }
            {
                const dim3 grid(DV/BM, (N + BN - 1)/BN, nz);
                static const bool pv2 = ggml_cuda_fa_fold_env("GGML_CUDA_FA_PV2", 1) != 0;
                const half2 * a_v = Vp.ptr; const half * a_p = P.ptr; const float * a_mt = mt.ptr; const float * a_lt = lt.ptr;
                const float * a_mi = m_cur; float * a_mo = m_nxt; float * a_l = l_state.ptr;
                const float * a_oi = O_cur; float * a_oo = O_nxt; int a_DV = (int) DV, a_N = (int) N, a_C = (int) C, a_nti = ntile;
                const float * a_mf = mask_first.ptr; int a_nt = (int) nt; fa_fold_split a_g = g; bool a_pf = prefix;
                void * args[] = {&a_v, &a_p, &a_mt, &a_lt, &a_mi, &a_mo, &a_l, &a_oi, &a_oo, &a_DV, &a_N, &a_C, &a_nti, &a_mf, &a_nt, &a_g, &a_pf};
                if (!pv2 || !fa_fold_sass_launch(1, grid, stream, args)) {
                    (pv2 ? fa_fold_pv2 : fa_fold_pv)<<<grid, 256, 0, stream>>>(a_v, a_p, a_mt, a_lt, a_mi, a_mo, a_l,
                        a_oi, a_oo, a_DV, a_N, a_C, a_nti, a_mf, a_nt, a_g, a_pf);
                }
                CUDA_CHECK(cudaGetLastError());
            }
            std::swap(O_cur, O_nxt);
            std::swap(m_cur, m_nxt);
        }

        fa_fold_finalize<<<dim3(nt, gqa, nhkv), 256, 0, stream>>>(
            O_cur, m_cur, l_state.ptr, (float *) dst->data + s*(dst->nb[3]/sizeof(float)),
            (int) DV, (int) nt, (int) N, (int) gqa, (int) nsplit, dst_s1, dst_s2);
        CUDA_CHECK(cudaGetLastError());
        if (sparsity) {
            sacc.report(ctx.device, nkv, nt);
        }
        if (oracle) {
            unsigned long long h[2];
            CUDA_CHECK(cudaMemcpyAsync(h, ocnt.ptr, sizeof(h), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, "fa_oracle dev %d delta %.2f nkv %lld nt %lld: kept %.4f of %llu nonzero logits\n", ctx.device,
                oracle_delta, (long long) nkv, (long long) nt, h[1] ? (double) h[0]/h[1] : 0.0, h[1]);
        }
    }
}

bool ggml_cuda_flash_attn_ext_gemm_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!mask || sinks) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (K->ne[0] != V->ne[0]) {
        return false;
    }
    // Only worth it once attention dominates; short contexts keep the tile kernel.
    static const int64_t min_kv = [] { const char * e = getenv("GGML_CUDA_FA_GEMM_MINKV"); return e ? (int64_t) atoll(e) : (int64_t) 4096; }();
    if (Q->ne[1] < 128 || K->ne[1] < min_kv) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    // The softmax kernel takes a single flat mask pointer: it applies mask->nb[1] per query
    // row but nothing per head or per sequence, while the sequence loop below advances Q, K,
    // V and dst by their nb[3]. Upstream's tile kernel offsets the mask by
    // nb33*(sequence % ne33) (fattn-tile.cuh:860); this path has no equivalent, so a mask
    // that is not broadcast across heads and sequences would silently be read from
    // sequence 0 for every sequence. Decline those shapes and let the tile kernel take them.
    if (mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    // F16 needs no conversion at all (cuBLAS takes an arbitrary lda, so we point it straight
    // at the cache). Anything else must have a strided dequantizer; note F16 itself is NOT in
    // ggml_get_to_fp16_nc_cuda's switch, so check the type before the function pointer.
    for (const ggml_tensor * t : {K, V}) {
        if (t->type != GGML_TYPE_F16 && ggml_get_to_fp16_nc_cuda(t->type) == nullptr) {
            return false;
        }
        if (t->nb[1] % sizeof(half) != 0) {
            return false;
        }
        // The strided dequantizers are handed s01 = nb[1]/type_size, i.e. they assume each KV
        // position is one contiguous run of blocks. Upstream asserts exactly this before the
        // same call (fattn-common.cuh:1036, GGML_ASSERT(K->nb[0] == ts)); a permuted view
        // would otherwise be read with the wrong stride and silently produce plausible-looking
        // attention. Decline instead, so such a tensor falls back to the tile kernel.
        if (t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
    }
    // PREC=32 reads V as fp32: in place for F32, otherwise it needs a strided fp32 dequantizer.
    // Decline rather than reach the GGML_ASSERT in the kernel.
    if (ggml_cuda_fa_gemm_prec32() && V->type != GGML_TYPE_F32 && ggml_get_to_fp32_nc_cuda(V->type) == nullptr) {
        return false;
    }
    return true;
}

void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_cuda_fa_fold_usable(dst)) {
        ggml_cuda_flash_attn_ext_fold(ctx, dst);
        return;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t D   = K->ne[0];
    const int64_t DV  = V->ne[0];
    const int64_t nt  = Q->ne[1];
    const int64_t nh  = Q->ne[2];
    const int64_t nkv = K->ne[1];
    const int64_t nhkv = K->ne[2];
    const int64_t ns  = Q->ne[3];
    const int64_t gqa = nh / nhkv;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    cudaStream_t stream = ctx.stream();
    cublasHandle_t cublas = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // Chunk size is bounded by the score-matrix scratch: S is nkv_c*nt*gqa floats.
    // 2048 keeps that near 100 MB at nt=2048, gqa=6, versus 512 MiB for the tile
    // path's whole-cache f16 conversion.
    const int64_t chunk = 2048;

    const bool prec32 = ggml_cuda_fa_gemm_prec32();

    ggml_cuda_pool & pool = ctx.pool();
    // ORDER MATTERS: the VMM pool requires frees in exact reverse order of allocations, and
    // destructors run in reverse order of DECLARATION. So every buffer is allocated at the
    // point it is declared -- the unused one of each fp16/fp32 pair stays null, and null is
    // skipped on free.
    ggml_cuda_pool_alloc<half>  Qf16(pool, D*nt*gqa);
    ggml_cuda_pool_alloc<half>  Kf16(pool, D*chunk);
    ggml_cuda_pool_alloc<half>  Vf16(pool);
    ggml_cuda_pool_alloc<float> Vf32(pool);
    if (prec32) { Vf32.alloc(DV*chunk); } else { Vf16.alloc(DV*chunk); }
    // scores, and the probabilities computed from them -- two buffers, never one (see the kernel)
    ggml_cuda_pool_alloc<half>  S(pool);
    ggml_cuda_pool_alloc<float> S32(pool);
    if (prec32) { S32.alloc(chunk*nt*gqa); } else { S.alloc(chunk*nt*gqa); }
    ggml_cuda_pool_alloc<half>  P(pool);
    ggml_cuda_pool_alloc<float> P32(pool);
    if (prec32) { P32.alloc(chunk*nt*gqa); } else { P.alloc(chunk*nt*gqa); }
    // the running output, accumulated out of place: O and O2 swap roles every chunk
    ggml_cuda_pool_alloc<float> O(pool, DV*nt*gqa);
    ggml_cuda_pool_alloc<float> O2(pool, DV*nt*gqa);
    // destination of the PV GEMM for one chunk; folded into the running output by fattn_gemm_accum_O.
    ggml_cuda_pool_alloc<half>  Otmp(pool);
    ggml_cuda_pool_alloc<float> Otmp32(pool);
    if (prec32) { Otmp32.alloc(DV*nt*gqa); } else { Otmp.alloc(DV*nt*gqa); }
    ggml_cuda_pool_alloc<float> m_state(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> l_state(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> corr(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> mask_first(pool, nt);

    // F16 is used in place, with cuBLAS's lda doing the striding -- no copy, no scratch.
    const bool K_is_f16 = K->type == GGML_TYPE_F16;
    const bool V_is_f16 = V->type == GGML_TYPE_F16;
    const to_fp16_nc_cuda_t to_fp16_K = K_is_f16 ? nullptr : ggml_get_to_fp16_nc_cuda(K->type);
    const to_fp16_nc_cuda_t to_fp16_V = V_is_f16 ? nullptr : ggml_get_to_fp16_nc_cuda(V->type);
    // PREC=32 reads V as fp32: an F32 cache in place (the fp32 analogue of F16 above), anything
    // else through the strided fp32 dequantizer. Note the two converter tables differ -- the
    // fp16 one covers F32 but not F16, the fp32 one F16 but not F32 -- so F32 must be in place.
    const bool V_is_f32 = V->type == GGML_TYPE_F32;
    const to_fp32_nc_cuda_t to_fp32_V = (prec32 && !V_is_f32) ? ggml_get_to_fp32_nc_cuda(V->type) : nullptr;
    GGML_ASSERT(!prec32 || V_is_f32 || to_fp32_V);
    GGML_ASSERT(K_is_f16 || to_fp16_K);
    GGML_ASSERT(prec32 || V_is_f16 || to_fp16_V);

    const int64_t dst_s1 = dst->nb[1]/sizeof(float);
    const int64_t dst_s2 = dst->nb[2]/sizeof(float);
    ggml_cuda_pool_alloc<half> mask_stair(pool);
    const fattn_gemm_mask_view mview = fattn_gemm_mask_get(mask, nt, nkv, mask_stair, stream);
    const int64_t s_mask = mview.s;

    // Where each mask row stops being zero; the mask is shared by every sequence and head.
    {
        dim3 grid(nt, 1, 1);
        fattn_gemm_mask_first_nz<256><<<grid, 256, 0, stream>>>(
            mview.data, mask_first.ptr, s_mask, (int) nkv);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int64_t s = 0; s < ns; ++s) {
        for (int64_t kvh = 0; kvh < nhkv; ++kvh) {
            const int64_t head0 = kvh*gqa;

            // Q -> f16, contiguous per head
            {
                dim3 grid(nt, gqa, 1);
                fattn_gemm_q_to_f16<<<grid, 256, 0, stream>>>(
                    (const char *) Q->data + s*Q->nb[3], Qf16.ptr,
                    D, nt, Q->nb[1], Q->nb[2], head0, scale*0.25f);
                CUDA_CHECK(cudaGetLastError());
            }

            float * O_cur = O.ptr;
            float * O_nxt = O2.ptr;
            CUDA_CHECK(cudaMemsetAsync(O_cur, 0, DV*nt*gqa*sizeof(float), stream));
            CUDA_CHECK(cudaMemsetAsync(l_state.ptr, 0, nt*gqa*sizeof(float), stream));
            // m = -inf, via a kernel: an H2D copy here would need a stream sync per head
            // group, i.e. 32 pipeline drains per batch.
            {
                const int64_t n = nt*gqa;
                fattn_gemm_fill<<<(n + 255)/256, 256, 0, stream>>>(m_state.ptr, -INFINITY, n);
                CUDA_CHECK(cudaGetLastError());
            }

            for (int64_t c = 0; c < nkv; c += chunk) {
                const int64_t nkv_c = std::min(chunk, nkv - c);

                // Dequantize just this chunk of K and V (or, for f16, use the cache in place).
                // This is what keeps the scratch bounded: the tile path converts all of K and V
                // on every call, which is 512 MiB per GPU at 262144 context.
                const char * Kp = (const char *) K->data + s*K->nb[3] + kvh*K->nb[2] + c*K->nb[1];
                const char * Vp = (const char *) V->data + s*V->nb[3] + kvh*V->nb[2] + c*V->nb[1];
                const half * Kmat;
                int64_t ldK;
                if (K_is_f16) {
                    Kmat = (const half *) Kp;
                    ldK  = K->nb[1]/sizeof(half);
                } else {
                    to_fp16_K(Kp, Kf16.ptr, D, nkv_c, 1, 1, K->nb[1]/ggml_type_size(K->type), 0, 0, stream);
                    Kmat = Kf16.ptr;
                    ldK  = D;
                }
                const half  * Vmat   = nullptr;
                const float * Vmat32 = nullptr;
                int64_t ldV;
                if (prec32) {
                    if (V_is_f32) {
                        Vmat32 = (const float *) Vp;
                        ldV    = V->nb[1]/sizeof(float);
                    } else {
                        // exact for q4_0: d*(q-8) needs 15 significant bits, float has 24
                        to_fp32_V(Vp, Vf32.ptr, DV, nkv_c, 1, 1, V->nb[1]/ggml_type_size(V->type), 0, 0, stream);
                        Vmat32 = Vf32.ptr;
                        ldV    = DV;
                    }
                } else if (V_is_f16) {
                    Vmat = (const half *) Vp;
                    ldV  = V->nb[1]/sizeof(half);
                } else {
                    to_fp16_V(Vp, Vf16.ptr, DV, nkv_c, 1, 1, V->nb[1]/ggml_type_size(V->type), 0, 0, stream);
                    Vmat = Vf16.ptr;
                    ldV  = DV;
                }

                // S = K^T Q  -> [nkv_c x nt] per head, column-major.
                //
                // Precision history (2026-09-12 audit). The original justification here was
                // FALSE: the tile kernel does NOT keep KQ in half -- upstream
                // fattn-tile.cuh:604 declares `float KQ_acc[...]`, an fp32 accumulator; what
                // it keeps in half are the products and Q_tmp. So COMPUTE_16F replaces an fp32
                // accumulation of k=256 terms with an fp16 one, measured at ~9.7e-3 mean
                // relative error per logit against upstream's ~2.4e-3 (4x worse).
                //
                // Fixed since, both free:
                //   - Q is pre-scaled by scale*0.25 in fattn_gemm_q_to_f16, restoring the 64x
                //     of fp16 overflow headroom upstream buys at fattn-tile.cuh:932-937; the
                //     softmax multiplies back by 4. This had to go in the conversion, not into
                //     cuBLAS `alpha`, because alpha is applied AFTER the accumulation and
                //     cannot stop a partial sum from overflowing. Before the fix the fp16
                //     accumulator went non-finite MORE often than the true dot product
                //     overflowed (37 vs 24 per 200k at element RMS 32), and an inf logit
                //     reaches expf(inf - inf) = NaN.
                //   - FATTN_KQ_MAX_OFFSET is added to the running max, as in tile, vec and
                //     mma, restoring 8x of headroom for the f16 probabilities and Otmp.
                //
                // The accumulator's precision itself is a measured decision, not an oversight:
                // COMPUTE_32F runs 6.2 vs 14.65 TFLOPS on this shape, and the 4x per-logit
                // error does not reach perplexity (see the precision note at the top).
                // GGML_CUDA_FA_GEMM_PREC=32 selects the fp32 accumulation.
                //
                // alpha/beta must match the COMPUTE type, not the data type -- with
                // COMPUTE_16F cuBLAS reads these as half*, with COMPUTE_32F as float*.
                // Mismatching them reinterprets the bits and silently degenerates attention
                // into a uniform average of V.
                if (prec32) {
                    // fp16 K and Q in, fp32 S out, fp32 compute: every half*half product is
                    // exact in fp32 and the k=256 sum accumulates in fp32.
                    const float alpha = 1.0f;
                    const float beta  = 0.0f;
                    CUBLAS_CHECK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nkv_c, nt*gqa, D,
                        &alpha,
                        Kmat,     CUDA_R_16F, ldK,
                        Qf16.ptr, CUDA_R_16F, D,
                        &beta,
                        S32.ptr,  CUDA_R_32F, nkv_c,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                } else {
                    const half alpha = __float2half(1.0f);
                    const half beta  = __float2half(0.0f);
                    CUBLAS_CHECK(cublasGemmEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nkv_c, nt*gqa, D,
                        &alpha,
                        Kmat,     CUDA_R_16F, ldK,
                        Qf16.ptr, CUDA_R_16F, D,
                        &beta,
                        S.ptr,    CUDA_R_16F, nkv_c,
                        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                }

                {
                    dim3 grid(nt, gqa, 1);
                    if (prec32) {
                        fattn_gemm_softmax<256, float><<<grid, 256, 0, stream>>>(
                            S32.ptr, P32.ptr, mview.data, mask_first.ptr,
                            m_state.ptr, l_state.ptr, corr.ptr, nkv_c, c, nt, s_mask, nkv_c*nt);
                    } else {
                        fattn_gemm_softmax<256, half><<<grid, 256, 0, stream>>>(
                            S.ptr, P.ptr, mview.data, mask_first.ptr,
                            m_state.ptr, l_state.ptr, corr.ptr, nkv_c, c, nt, s_mask, nkv_c*nt);
                    }
                    CUDA_CHECK(cudaGetLastError());
                }

                // Otmp = V P, then O = O*corr + Otmp.
                //
                // f16 compute by default: COMPUTE_32F with f16 inputs runs 6.4-6.7 TFLOPS on
                // Pascal against 11.0-14.6 for COMPUTE_16F, and PV is ~half of attention's
                // flops. The f16 sum here spans one chunk; cross-chunk accumulation is f32 in
                // fattn_gemm_accum_O. Upstream's Pascal tile kernel accumulates VKQ in half2
                // over the ENTIRE cache, so this is the more conservative of the two.
                // alpha/beta must match the COMPUTE type, not the data type.
                if (prec32) {
                    const float alpha = 1.0f;
                    const float beta  = 0.0f;
                    CUBLAS_CHECK(cublasGemmEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        DV, nt*gqa, nkv_c,
                        &alpha,
                        Vmat32,     CUDA_R_32F, ldV,
                        P32.ptr,    CUDA_R_32F, nkv_c,
                        &beta,
                        Otmp32.ptr, CUDA_R_32F, DV,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                } else {
                    // ALGO4, not DEFAULT. cuBLAS's fp16 GEMM kernels on Pascal come in two
                    // accuracy families: ALGO4-6 accumulate in blocks, ALGO1-3 in long fp16
                    // chains, and DEFAULT switches to the long chains from nt*gqa ~ 6000 on --
                    // i.e. every prefill ubatch of 1024 tokens or more. Against an fp64 product of
                    // the same fp16 inputs, at k=2048: NMSE 2.9e-5 for DEFAULT, 2.8e-6 for
                    // ALGO4-6, with the same split at every nt from 128 to 2048 (OPTLOG attempt
                    // 152). This is the dominant error of the fp16 path. ALGO4 is the most even
                    // of the three on speed: ~8% over DEFAULT at nt=2048, level at nt <= 1024,
                    // where ALGO6 is up to 1.5x slower. Any failure (an unsupported algorithm on
                    // another GPU or cuBLAS) falls back to DEFAULT.
                    const half alpha = __float2half(1.0f);
                    const half beta  = __float2half(0.0f);
                    for (const cublasGemmAlgo_t algo : {CUBLAS_GEMM_ALGO4, CUBLAS_GEMM_DEFAULT}) {
                        const cublasStatus_t st = cublasGemmEx(
                            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                            DV, nt*gqa, nkv_c,
                            &alpha,
                            Vmat,     CUDA_R_16F, ldV,
                            P.ptr,    CUDA_R_16F, nkv_c,
                            &beta,
                            Otmp.ptr, CUDA_R_16F, DV,
                            CUBLAS_COMPUTE_16F, algo);
                        if (st == CUBLAS_STATUS_SUCCESS) {
                            break;
                        }
                        if (algo == CUBLAS_GEMM_DEFAULT) {
                            CUBLAS_CHECK(st);
                        }
                    }
                }

                {
                    dim3 grid(nt, gqa, 1);
                    if (prec32) {
                        fattn_gemm_accum_O<float><<<grid, 256, 0, stream>>>(O_cur, O_nxt, Otmp32.ptr, corr.ptr, DV, nt);
                    } else {
                        fattn_gemm_accum_O<half><<<grid, 256, 0, stream>>>(O_cur, O_nxt, Otmp.ptr, corr.ptr, DV, nt);
                    }
                    CUDA_CHECK(cudaGetLastError());
                    std::swap(O_cur, O_nxt);
                }
            }

            {
                dim3 grid(nt, gqa, 1);
                fattn_gemm_finalize<<<grid, 256, 0, stream>>>(
                    O_cur, l_state.ptr, (float *) dst->data + s*(dst->nb[3]/sizeof(float)),
                    DV, nt, nh, head0, dst_s1, dst_s2);
                CUDA_CHECK(cudaGetLastError());
            }
        }
    }
}

// On by default for pre-Volta, where it is both faster and uses far less memory than the
// tile path. GGML_CUDA_FA_GEMM=0 falls back to upstream behaviour.
// NOTE: ggml_cuda_flash_attn_ext_get_alloc_size() must gate on exactly this same predicate --
// if the two disagree the kernel writes past the allocation.
bool ggml_cuda_fa_gemm_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FA_GEMM");
        return !s || (s[0] != '0');
    }();
    return enabled;
}
