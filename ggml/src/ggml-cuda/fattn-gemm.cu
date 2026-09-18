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
    if (Q->ne[1] < 128 || K->ne[1] < 4096) {
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
    const int64_t s_mask = mask->nb[1]/sizeof(half);

    // Where each mask row stops being zero; the mask is shared by every sequence and head.
    {
        dim3 grid(nt, 1, 1);
        fattn_gemm_mask_first_nz<256><<<grid, 256, 0, stream>>>(
            (const half *) mask->data, mask_first.ptr, s_mask, (int) nkv);
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
                            S32.ptr, P32.ptr, (const half *) mask->data, mask_first.ptr,
                            m_state.ptr, l_state.ptr, corr.ptr, nkv_c, c, nt, s_mask, nkv_c*nt);
                    } else {
                        fattn_gemm_softmax<256, half><<<grid, 256, 0, stream>>>(
                            S.ptr, P.ptr, (const half *) mask->data, mask_first.ptr,
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
