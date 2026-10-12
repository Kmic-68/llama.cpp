#include "p60-moe-gemm.cuh"
#include "mmid.cuh"

// Expert matmuls (MUL_MAT_ID) for prompt processing on sm_60 with fp16 math. sm_60 has no dp4a, so MMQ's int8 dot
// products are emulated (~2.5 TFLOPS on these shapes); here weights are dequantized to fp16 in shared memory and
// multiplied with HFMA2 (~7 TFLOPS).
//   1. mm_ids_helper: (token, slot) pairs sorted by expert, their src1 / dst rows, per-expert bounds
//   2. the src1 rows of the pairs, gathered in that order, converted to fp16
//   3. one block turns the expert bounds into a list of tiles (expert, first pair, pairs) - no host sync
//   4. a block computes BM rows x BN pairs of one expert: per 64-wide K step the weight tile is dequantized into
//      shared memory as k-pairs (half2 of k, k + 1), so one 128-bit load gives 4 rows or 4 pairs; HFMA2 over the
//      step, then the half2 accumulators are flushed to fp32.

namespace {

constexpr int BM = 64, BN = 64, BK = 64, KP = BK / 2, NTH = 256;

struct moe_job { int expert, p0, np; };

__device__ __forceinline__ void p60_scale_min_k4(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// src1 rows of the sorted pairs -> fp16 [pair][K]
__global__ void moe_gather_f16(const char * __restrict__ src1, int64_t nb11, const int32_t * __restrict__ ids_src1, int K,
        half2 * __restrict__ xs) {
    const int p = blockIdx.x;
    const float2 * x = (const float2 *) (src1 + (int64_t) ids_src1[p] * nb11);
    half2 * o = xs + (int64_t) p * (K / 2);
    for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
        const float2 v = x[i];
        o[i] = __floats2half2_rn(v.x, v.y);
    }
}

// expert bounds -> tiles of up to BN pairs; one block of n_experts (<= 1024) threads
__global__ void moe_jobs(const int32_t * __restrict__ bounds, int n_experts, moe_job * __restrict__ jobs, int * __restrict__ njobs) {
    __shared__ int cnt[1024];
    const int e = threadIdx.x;
    const int n = e < n_experts ? bounds[e + 1] - bounds[e] : 0;
    const int nt = (n + BN - 1) / BN;
    cnt[e] = nt;
    __syncthreads();
    // inclusive scan (Hillis-Steele)
    for (int o = 1; o < blockDim.x; o <<= 1) {
        const int v = e >= o ? cnt[e - o] : 0;
        __syncthreads();
        cnt[e] += v;
        __syncthreads();
    }
    const int first = cnt[e] - nt;
    for (int t = 0; t < nt; t++) {
        jobs[first + t] = { e, bounds[e] + t * BN, min(BN, n - t * BN) };
    }
    if (e == blockDim.x - 1) {
        *njobs = cnt[e];
    }
}

// TYPE: GGML_TYPE_Q4_K (144 B / 256), GGML_TYPE_Q5_K (176 B / 256) or GGML_TYPE_Q6_K (210 B / 256)
template <int TYPE>
__global__ void __launch_bounds__(NTH, 2) moe_gemm_f16(const uint8_t * __restrict__ W, int64_t row_bytes, int64_t exp_bytes, int K,
        const half2 * __restrict__ X, const moe_job * __restrict__ jobs, const int * __restrict__ njobs,
        const int32_t * __restrict__ ids_dst, char * __restrict__ dst, int64_t nb1) {
    __shared__ __align__(16) half2 AP[KP][BM];   // weights: AP[kp][row]
    __shared__ __align__(16) half2 BP[KP][BN];   // activations: BP[kp][pair]
    if ((int) blockIdx.x >= *njobs) {
        return;
    }
    const moe_job jb = jobs[blockIdx.x];
    const int m0 = blockIdx.y * BM;
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const uint8_t * Wexp = W + (int64_t) jb.expert * exp_bytes;
    constexpr int SBB = TYPE == GGML_TYPE_Q4_K ? 144 : TYPE == GGML_TYPE_Q5_K ? 176 : 210, QS = TYPE == GGML_TYPE_Q4_K ? 16 : 48;

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    const int lr = tid >> 2, lp = tid & 3;   // weight loader: row, 8 bytes of qs (16 weights)
    const int bpair = tid >> 2, bq = tid & 3; // activation loader: pair, 8 k-pairs

    for (int k0 = 0; k0 < K; k0 += BK) {
        if constexpr (TYPE == GGML_TYPE_Q6_K) {
            // 210 B superblocks (2 B aligned): step j covers y[64 j ..]: half n = j / 2, upper nibbles / qh bits 4-7
            // for odd j; first 32 weights from ql[64 n + l], the next 32 from ql[64 n + 32 + l], l = 8 lp + e
            const int sb = k0 / 256, j = (k0 % 256) / 64, n = j >> 1, odd = j & 1;
            const uint8_t * blk = Wexp + (int64_t) (m0 + lr) * row_bytes + sb * SBB;
            const unsigned short * ql0 = (const unsigned short *) (blk + 64 * n + 8 * lp);
            const unsigned short * ql1 = (const unsigned short *) (blk + 64 * n + 32 + 8 * lp);
            const unsigned short * qhp = (const unsigned short *) (blk + 128 + 32 * n + 8 * lp);
            uint8_t a0[8], a1[8], hh[8];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned short u0 = ql0[e], u1 = ql1[e], uh = qhp[e];
                a0[2 * e] = u0 & 0xFF; a0[2 * e + 1] = u0 >> 8;
                a1[2 * e] = u1 & 0xFF; a1[2 * e + 1] = u1 >> 8;
                hh[2 * e] = uh & 0xFF; hh[2 * e + 1] = uh >> 8;
            }
            const int is = lp >> 1;
            const int8_t * sc = (const int8_t *) (blk + 192 + 8 * n + 4 * odd);
            const float d = __half2float(*(const half *) (blk + 208));
            const float dl = d * sc[is], dh = d * sc[is + 2];
            const int sh = 4 * odd;
            float lo[8], hi[8];
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int q0 = ((a0[e] >> sh) & 0xF) | (((hh[e] >> sh) & 3) << 4);
                const int q1 = ((a1[e] >> sh) & 0xF) | (((hh[e] >> (sh + 2)) & 3) << 4);
                lo[e] = dl * (q0 - 32);
                hi[e] = dh * (q1 - 32);
            }
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
                AP[(8 * lp + e) / 2][lr]      = __floats2half2_rn(lo[e], lo[e + 1]);
                AP[(32 + 8 * lp + e) / 2][lr] = __floats2half2_rn(hi[e], hi[e + 1]);
            }
        } else {
            const int sb = k0 / 256, j = (k0 % 256) / 64;
            const uint8_t * blk = Wexp + (int64_t) (m0 + lr) * row_bytes + sb * SBB;
            const float d = __half2float(*(const half *) blk), dmin = __half2float(*(const half *) (blk + 2));
            int s0, mn0, s1, mn1;
            p60_scale_min_k4(2 * j, blk + 4, s0, mn0);
            p60_scale_min_k4(2 * j + 1, blk + 4, s1, mn1);
            const float d0 = d * s0, c0 = dmin * mn0, d1 = d * s1, c1 = dmin * mn1;
            const uint2 q8 = *(const uint2 *) (blk + QS + 32 * j + 8 * lp);
            const uint8_t * qb = (const uint8_t *) &q8;
            unsigned hb0 = 0, hb1 = 0;
            if constexpr (TYPE == GGML_TYPE_Q5_K) {
                const uint2 h8 = *(const uint2 *) (blk + 16 + 8 * lp);
                const uint8_t * hbb = (const uint8_t *) &h8;
#pragma unroll
                for (int e = 0; e < 8; e++) {
                    hb0 |= ((hbb[e] >> (2 * j)) & 1u) << e;
                    hb1 |= ((hbb[e] >> (2 * j + 1)) & 1u) << e;
                }
            }
            float lo[8], hi[8];
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int ql = qb[e] & 0xF, qh = qb[e] >> 4;
                if constexpr (TYPE == GGML_TYPE_Q5_K) { ql |= ((hb0 >> e) & 1) << 4; qh |= ((hb1 >> e) & 1) << 4; }
                lo[e] = d0 * ql - c0;
                hi[e] = d1 * qh - c1;
            }
#pragma unroll
            for (int e = 0; e < 8; e += 2) {
                AP[(8 * lp + e) / 2][lr]      = __floats2half2_rn(lo[e], lo[e + 1]);
                AP[(32 + 8 * lp + e) / 2][lr] = __floats2half2_rn(hi[e], hi[e + 1]);
            }
        }
        {
            if (bpair < jb.np) {
                const uint4 * src = (const uint4 *) (X + (int64_t) (jb.p0 + bpair) * (K / 2) + k0 / 2 + 8 * bq);
                const uint4 v = src[0], v2 = src[1];
                const half2 * h = (const half2 *) &v, * h2 = (const half2 *) &v2;
#pragma unroll
                for (int e = 0; e < 4; e++) { BP[8 * bq + e][bpair] = h[e]; BP[8 * bq + 4 + e][bpair] = h2[e]; }
            } else {
#pragma unroll
                for (int e = 0; e < 8; e++) BP[8 * bq + e][bpair] = __float2half2_rn(0.f);
            }
        }
        __syncthreads();
        half2 h[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) h[i][j] = __float2half2_rn(0.f);
#pragma unroll 8
        for (int kp = 0; kp < KP; kp++) {
            const uint4 a = *(const uint4 *) &AP[kp][4 * ty];
            const uint4 b = *(const uint4 *) &BP[kp][4 * tx];
            const half2 * ah = (const half2 *) &a, * bh = (const half2 *) &b;
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) h[i][j] = __hfma2(ah[i], bh[j], h[i][j]);
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) { const float2 f = __half22float2(h[i][j]); acc[i][j] += f.x + f.y; }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int p = 4 * tx + j;
        if (p < jb.np) {
            float * o = (float *) (dst + (int64_t) ids_dst[jb.p0 + p] * nb1) + m0 + 4 * ty;
            *(float4 *) o = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
        }
    }
}

} // namespace

bool ggml_cuda_p60_moe_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    static const bool enabled = getenv("GGML_CUDA_P60_MEGA") != nullptr && atoi(getenv("GGML_CUDA_P60_MEGA")) != 0 &&
        (getenv("GGML_CUDA_P60_MOE_GEMM") == nullptr || atoi(getenv("GGML_CUDA_P60_MOE_GEMM")) != 0);
    if (!enabled || ggml_cuda_info().devices[ctx.device].cc != 600) {
        return false;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], n_experts = src0->ne[2];
    const int64_t n_used = ids->ne[0], n_tokens = ids->ne[1];
    if ((src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            ids->type != GGML_TYPE_I32 || K % 256 != 0 || M % BM != 0 || n_experts > 1024 || src0->ne[3] != 1 ||
            src1->ne[0] != K || src1->ne[3] != 1 || dst->ne[3] != 1 || src1->nb[0] != sizeof(float) || src1->nb[1] % 8 != 0 ||
            (src1->ne[1] != 1 && src1->ne[1] != n_used) || src1->ne[2] != n_tokens || src1->nb[2] % src1->nb[1] != 0 ||
            dst->ne[0] != M || dst->ne[1] != n_used || dst->ne[2] != n_tokens || dst->nb[0] != sizeof(float) ||
            dst->nb[2] != dst->nb[1] * n_used || dst->nb[1] % 16 != 0 || ((uintptr_t) dst->data) % 16 != 0 ||
            src0->nb[1] != (size_t) K / 256 * (src0->type == GGML_TYPE_Q4_K ? 144 : src0->type == GGML_TYPE_Q5_K ? 176 : 210) ||
            src0->nb[2] != src0->nb[1] * M ||
            ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const int64_t n_pairs = n_tokens * n_used;

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_pairs);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_pairs);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_experts + 1);
    const int si1  = ids->nb[1] / sizeof(int32_t);
    const int sis1 = src1->nb[2] / src1->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
        n_experts, n_tokens, n_used, src1->ne[1], si1, sis1, /*write_inverse =*/ false, stream);

    ggml_cuda_pool_alloc<half2> xs(ctx.pool(), n_pairs * K / 2);
    moe_gather_f16<<<n_pairs, 256, 0, stream>>>((const char *) src1->data, src1->nb[1], ids_src1.get(), (int) K, xs.get());

    const int max_jobs = (int) ((n_pairs + BN - 1) / BN + n_experts);
    ggml_cuda_pool_alloc<moe_job> jobs(ctx.pool(), max_jobs);
    ggml_cuda_pool_alloc<int> njobs(ctx.pool(), 1);
    int nthr = 32;
    while (nthr < n_experts) {
        nthr *= 2;
    }
    moe_jobs<<<1, nthr, 0, stream>>>(bounds.get(), (int) n_experts, jobs.get(), njobs.get());

    const dim3 grid(max_jobs, M / BM);
    if (src0->type == GGML_TYPE_Q4_K) {
        moe_gemm_f16<GGML_TYPE_Q4_K><<<grid, NTH, 0, stream>>>((const uint8_t *) src0->data, src0->nb[1], src0->nb[2], (int) K,
            xs.get(), jobs.get(), njobs.get(), ids_dst.get(), (char *) dst->data, dst->nb[1]);
    } else if (src0->type == GGML_TYPE_Q5_K) {
        moe_gemm_f16<GGML_TYPE_Q5_K><<<grid, NTH, 0, stream>>>((const uint8_t *) src0->data, src0->nb[1], src0->nb[2], (int) K,
            xs.get(), jobs.get(), njobs.get(), ids_dst.get(), (char *) dst->data, dst->nb[1]);
    } else {
        moe_gemm_f16<GGML_TYPE_Q6_K><<<grid, NTH, 0, stream>>>((const uint8_t *) src0->data, src0->nb[1], src0->nb[2], (int) K,
            xs.get(), jobs.get(), njobs.get(), ids_dst.get(), (char *) dst->data, dst->nb[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
