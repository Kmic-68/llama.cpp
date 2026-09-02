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
//       S = K^T Q                      (GEMM, f16 accumulate -- matches the tile kernel,
//                                       which also keeps KQ in half)
//       m_new = max(m, rowmax(S+mask))
//       corr  = exp(m - m_new);  P = exp(S + mask - m_new)
//       l     = l*corr + rowsum(P)
//       Otmp  = V P                    (GEMM, f16 accumulate over one chunk)
//       O     = O*corr + Otmp          (fused rescale + f32 accumulate across chunks)
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
#include "fattn-gemm.cuh"
#include "convert.cuh"

#include <cublas_v2.h>

// One block per (query token, head). Applies mask+scale, advances the running softmax
// statistics, writes P in f16, and reports the rescale factor for O.
template <int block_size>
static __global__ void fattn_gemm_softmax(
        const half   * __restrict__ S,        // [nkv_c x nt] per head, column-major (f16:
                                              // halves what is the dominant traffic term here)
        const half   * __restrict__ mask,     // [nkv_pad x nt], contiguous, may be null
        half         * __restrict__ P,        // out, same layout as S
        float        * __restrict__ m_state,  // [nt x nh]
        float        * __restrict__ l_state,  // [nt x nh]
        float        * __restrict__ corr_out, // [nt x nh]
        const float scale,
        const int nkv_c,      // keys in this chunk
        const int nkv_off,    // offset of this chunk within the full KV
        const int nt,
        const int64_t s_mask, // mask row stride in elements
        const int64_t s_head) // per-head stride of S/P in elements
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    const half  * Sh = S + h*s_head + (int64_t) t*nkv_c;
    half        * Ph = P + h*s_head + (int64_t) t*nkv_c;
    const half  * mh = mask ? mask + (int64_t) t*s_mask + nkv_off : nullptr;

    __shared__ float red[block_size/WARP_SIZE];

    // pass 1: row max of (scale*S + mask)
    float vmax = -FLT_MAX/2.0f;
    for (int j = tid; j < nkv_c; j += block_size) {
        float v = scale*__half2float(Sh[j]);
        if (mh) {
            v += __half2float(mh[j]);
        }
        vmax = fmaxf(vmax, v);
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
        float v = scale*__half2float(Sh[j]);
        if (mh) {
            v += __half2float(mh[j]);
        }
        const float p = (v <= -FLT_MAX/4.0f || m_new <= -FLT_MAX/4.0f) ? 0.0f : expf(v - m_new);
        Ph[j] = __float2half(p);
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
// chunk at nt=2048, gqa=6); this reads O+Otmp and writes O once (38 MB).
static __global__ void fattn_gemm_accum_O(
        float * __restrict__ O, const half * __restrict__ Otmp,
        const float * __restrict__ corr,
        const int DV, const int nt) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float c = corr[h*nt + t];
    const int64_t off = ((int64_t) h*nt + t)*DV;
    float      * Oh = O    + off;
    const half * Th = Otmp + off;
    for (int d = threadIdx.x; d < DV; d += blockDim.x) {
        Oh[d] = Oh[d]*c + __half2float(Th[d]);
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
static __global__ void fattn_gemm_q_to_f16(
        const char * __restrict__ Q, half * __restrict__ Qf16,
        const int D, const int nt, const int64_t nbq1, const int64_t nbq2,
        const int head0) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const float * q = (const float *) (Q + (int64_t) t*nbq1 + (int64_t)(head0 + h)*nbq2);
    half * o = Qf16 + ((int64_t) h*nt + t)*D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        o[d] = __float2half(q[d]);
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

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<half>  Qf16(pool, D*nt*gqa);
    ggml_cuda_pool_alloc<half>  Kf16(pool, D*chunk);
    ggml_cuda_pool_alloc<half>  Vf16(pool, DV*chunk);
    // P aliases S: in the second pass each thread reads S[j] then writes P[j] at the same
    // index, and every first-pass load is fenced by the reduction's __syncthreads, so one
    // buffer serves both. Worth 50 MB at ub=2048, which lowers the context length at which
    // this path's fixed scratch undercuts the tile path's context-proportional staging.
    ggml_cuda_pool_alloc<half>  S(pool, chunk*nt*gqa);
    half * const P_ptr = S.ptr;
    ggml_cuda_pool_alloc<float> O(pool, DV*nt*gqa);
    // f16 destination of the PV GEMM for one chunk; folded into O by fattn_gemm_accum_O.
    ggml_cuda_pool_alloc<half>  Otmp(pool, DV*nt*gqa);
    ggml_cuda_pool_alloc<float> m_state(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> l_state(pool, nt*gqa);
    ggml_cuda_pool_alloc<float> corr(pool, nt*gqa);

    // F16 is used in place, with cuBLAS's lda doing the striding -- no copy, no scratch.
    const bool K_is_f16 = K->type == GGML_TYPE_F16;
    const bool V_is_f16 = V->type == GGML_TYPE_F16;
    const to_fp16_nc_cuda_t to_fp16_K = K_is_f16 ? nullptr : ggml_get_to_fp16_nc_cuda(K->type);
    const to_fp16_nc_cuda_t to_fp16_V = V_is_f16 ? nullptr : ggml_get_to_fp16_nc_cuda(V->type);
    GGML_ASSERT((K_is_f16 || to_fp16_K) && (V_is_f16 || to_fp16_V));

    const int64_t dst_s1 = dst->nb[1]/sizeof(float);
    const int64_t dst_s2 = dst->nb[2]/sizeof(float);

    for (int64_t s = 0; s < ns; ++s) {
        for (int64_t kvh = 0; kvh < nhkv; ++kvh) {
            const int64_t head0 = kvh*gqa;

            // Q -> f16, contiguous per head
            {
                dim3 grid(nt, gqa, 1);
                fattn_gemm_q_to_f16<<<grid, 256, 0, stream>>>(
                    (const char *) Q->data + s*Q->nb[3], Qf16.ptr,
                    D, nt, Q->nb[1], Q->nb[2], head0);
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaMemsetAsync(O.ptr, 0, DV*nt*gqa*sizeof(float), stream));
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
                const half * Vmat;
                int64_t ldK, ldV;
                if (K_is_f16) {
                    Kmat = (const half *) Kp;
                    ldK  = K->nb[1]/sizeof(half);
                } else {
                    to_fp16_K(Kp, Kf16.ptr, D, nkv_c, 1, 1, K->nb[1]/ggml_type_size(K->type), 0, 0, stream);
                    Kmat = Kf16.ptr;
                    ldK  = D;
                }
                if (V_is_f16) {
                    Vmat = (const half *) Vp;
                    ldV  = V->nb[1]/sizeof(half);
                } else {
                    to_fp16_V(Vp, Vf16.ptr, DV, nkv_c, 1, 1, V->nb[1]/ggml_type_size(V->type), 0, 0, stream);
                    Vmat = Vf16.ptr;
                    ldV  = DV;
                }

                // S = K^T Q  -> [nkv_c x nt] per head, column-major.
                // f16 accumulate: the tile kernel also keeps KQ in half, so this matches
                // the existing numerical behaviour rather than degrading it.
                {
                    // f16 compute here, unlike PV: QK^T sums only k=D=256 terms and the tile
                    // kernel likewise keeps KQ in half, so this matches existing precision
                    // while running at the 2:1 fp16 rate (measured 14.65 vs 6.2 TFLOPS for
                    // COMPUTE_32F at this shape).
                    // alpha/beta must match the COMPUTE type, not the data type -- with
                    // COMPUTE_16F cuBLAS reads these as half*, with COMPUTE_32F as float*.
                    // Mismatching them reinterprets the bits and silently degenerates
                    // attention into a uniform average of V.
                    const half alpha = __float2half(1.0f);
                    const half beta  = __float2half(0.0f);
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        nkv_c, nt, D,
                        &alpha,
                        Kmat, CUDA_R_16F, ldK, 0,
                        Qf16.ptr, CUDA_R_16F, D, D*nt,
                        &beta,
                        S.ptr,    CUDA_R_16F, nkv_c, nkv_c*nt,
                        gqa, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                }

                {
                    dim3 grid(nt, gqa, 1);
                    fattn_gemm_softmax<256><<<grid, 256, 0, stream>>>(
                        S.ptr, mask ? (const half *) mask->data : nullptr, P_ptr,
                        m_state.ptr, l_state.ptr, corr.ptr,
                        scale, nkv_c, c, nt,
                        mask ? mask->nb[1]/sizeof(half) : 0,
                        nkv_c*nt);
                    CUDA_CHECK(cudaGetLastError());
                }

                // Otmp = V P, then O = O*corr + Otmp.
                //
                // f16 compute: COMPUTE_32F with f16 inputs runs 6.4-6.7 TFLOPS on Pascal
                // against 11.0-14.6 for COMPUTE_16F, and PV is ~half of attention's flops.
                // The f16 sum here spans one chunk; cross-chunk accumulation is f32 in
                // fattn_gemm_accum_O. Upstream's Pascal tile kernel accumulates VKQ in
                // half2 over the ENTIRE cache and passes the same tests, so this is the
                // more conservative of the two.
                // alpha/beta must match the COMPUTE type, not the data type.
                {
                    const half alpha = __float2half(1.0f);
                    const half beta  = __float2half(0.0f);
                    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        DV, nt, nkv_c,
                        &alpha,
                        Vmat, CUDA_R_16F, ldV, 0,
                        P_ptr,    CUDA_R_16F, nkv_c, nkv_c*nt,
                        &beta,
                        Otmp.ptr, CUDA_R_16F, DV, DV*nt,
                        gqa, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
                }

                {
                    dim3 grid(nt, gqa, 1);
                    fattn_gemm_accum_O<<<grid, 256, 0, stream>>>(O.ptr, Otmp.ptr, corr.ptr, DV, nt);
                    CUDA_CHECK(cudaGetLastError());
                }
            }

            {
                dim3 grid(nt, gqa, 1);
                fattn_gemm_finalize<<<grid, 256, 0, stream>>>(
                    O.ptr, l_state.ptr, (float *) dst->data + s*(dst->nb[3]/sizeof(float)),
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
