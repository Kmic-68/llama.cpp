// Prefill GEMM for Pascal (P100): fp16 products, fp32 accumulation, at close to fp16 speed.
//
// cuBLAS COMPUTE_16F (what the f16 path used) accumulates each output over the whole of K in fp16:
// NMSE ~1.2e-5 against an fp64 product even with the blocked ALGO6, about half of the f16 path's
// model-level distance from fp32 (OPTLOG attempts 153, 221).
// COMPUTE_32F fixes it but runs at the fp32 FMA rate (-40% prefill).
//
// Here each thread keeps an 8x8 output tile as half2 accumulators (lanes = even/odd k) and every
// GGML_CUDA_GEMM_FOLD_K2 (default 128) k2 steps, i.e. 256 values of K, adds the two lanes (one
// HADD2) and moves the sum into fp32 with an integer half->float conversion: for a half in the high
// lane of a 32-bit word, an arithmetic shift right by 3 and a mask give the fp32 bit pattern of
// value * 2^-112 (the rebias is undone once, in the epilogue). 4 instructions per output per fold,
// where F2F would cost 4x more issue slots. The fp16 chains no longer grow with K: NMSE 1.4e-6
// against fp64 on a real 8704x5120 q6_K slice, cuBLAS COMPUTE_16F 1.2e-5 (OPTLOG attempt 221).
//
// What is left is the f16 rounding of the inputs, which any f16-multiply path has: KLD against an
// all-fp32 run 0.00125, cuBLAS f16 0.00152, all-fp32 with a different summation order 0.0006-0.001.
//
// Activations are prescaled per column by a power of 2 (exact) so their max lands in [8,16): the
// fp16 partial sums stay far from overflow and from the subnormal range.
//
// Outputs are rounded to f16 by default (GGML_CUDA_GEMM_FOLD=2): that keeps the tensor-parallel
// exchange compressed (f16-exact partials) and measured no less accurate than fp32 outputs
// (GGML_CUDA_GEMM_FOLD=1). Matmuls under GGML_CUDA_GEMM_FOLD_MINROWS (1024) rows run in fp32
// cuBLAS instead, since a 128-row tile leaves most SMs idle there. GGML_CUDA_GEMM_FOLD=0 falls back
// to cuBLAS COMPUTE_16F.

#include "gemm-fold.cuh"
#include "mmvq.cuh"

#include <unordered_map>
#include <vector>
#include <cuda.h>
#include "convert.cuh"
#include "unary.cuh"
#include "gemm-fold-u2-sass.h"

#include <algorithm>

namespace {

constexpr int BM  = 128;
constexpr int BN  = 128;
constexpr int BK2 = 16;     // k2 steps per smem tile = 32 values of K

__global__ void gemm_fold_prescale(const float * __restrict__ X, const int64_t s1, half * __restrict__ X16,
                                   float * __restrict__ cs, const int K, const int xexp) {
    const int n = blockIdx.x;
    const float * x = X + n*s1;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        m = fmaxf(m, fabsf(x[k]));
    }
    __shared__ float sm[32];
    m = warp_reduce_max(m);
    if ((threadIdx.x & 31) == 0) {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < blockDim.x/32 ? sm[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
        if (threadIdx.x == 0) {
            sm[0] = m;
        }
    }
    __syncthreads();
    m = sm[0];
    const int   e = xexp < 0 ? 0 : m > 0.0f && isfinite(m) ? ilogbf(m) - xexp : 0;
    const float s = ldexpf(1.0f, -e);
    half * y = X16 + (int64_t) n*K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        y[k] = __float2half(x[k]*s);
    }
    if (threadIdx.x == 0) {
        cs[n] = ldexpf(1.0f, e) * 0x1p112f;
    }
}

// Same outputs as gemm_fold_prescale, one read of X: the row is held in registers (float4 x NV per
// thread) between the max and the scaling. Needs K % 4 == 0, 16-byte aligned rows, K <= 1024*NV.
// GLU: X is the gate and U the up input of a split SwiGLU; x = silu(gate)*up, the expression
// unary_gated_op_kernel<op_silu> evaluates (so the GLU node's output is never materialized).
static __device__ __forceinline__ float4 gemm_fold_swiglu4(const float4 g, const float4 u) {
    return make_float4(ggml_cuda_op_silu_single(g.x) * u.x, ggml_cuda_op_silu_single(g.y) * u.y,
                       ggml_cuda_op_silu_single(g.z) * u.z, ggml_cuda_op_silu_single(g.w) * u.w);
}

template <int NV, bool GLU = false>
__global__ void __launch_bounds__(256) gemm_fold_prescale_v(const float * __restrict__ X, const int64_t s1,
        half * __restrict__ X16, float * __restrict__ cs, const int K, const int xexp,
        const float * __restrict__ U = nullptr, const int64_t su = 0) {
    const int n = blockIdx.x;
    const float4 * x = (const float4 *) (X + n*s1);
    const float4 * u = (const float4 *) (U + n*su);
    const int K4 = K/4;
    float4 r[NV];
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int k = threadIdx.x + 256*i;
        r[i] = k < K4 ? x[k] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (GLU && k < K4) {
            r[i] = gemm_fold_swiglu4(r[i], u[k]);
        }
        m = fmaxf(m, fmaxf(fmaxf(fabsf(r[i].x), fabsf(r[i].y)), fmaxf(fabsf(r[i].z), fabsf(r[i].w))));
    }
    __shared__ float sm[32];
    m = warp_reduce_max(m);
    if ((threadIdx.x & 31) == 0) {
        sm[threadIdx.x >> 5] = m;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < 8 ? sm[threadIdx.x] : 0.0f;
        m = warp_reduce_max(m);
        if (threadIdx.x == 0) {
            sm[0] = m;
        }
    }
    __syncthreads();
    m = sm[0];
    const int   e = xexp < 0 ? 0 : m > 0.0f && isfinite(m) ? ilogbf(m) - xexp : 0;
    const float s = ldexpf(1.0f, -e);
    half2 * y = (half2 *) (X16 + (int64_t) n*K);
#pragma unroll
    for (int i = 0; i < NV; i++) {
        const int k = threadIdx.x + 256*i;
        if (k < K4) {
            y[2*k + 0] = __halves2half2(__float2half(r[i].x*s), __float2half(r[i].y*s));
            y[2*k + 1] = __halves2half2(__float2half(r[i].z*s), __float2half(r[i].w*s));
        }
    }
    if (threadIdx.x == 0) {
        cs[n] = ldexpf(1.0f, e) * 0x1p112f;
    }
}

static __device__ __forceinline__ float gemm_fold_h(const half2 p) {
    const half2    s = __hadd2(p, __lowhigh2highlow(p));  // high lane = lo + hi
    const int32_t  v = ((int32_t) *(const uint32_t *) &s) >> 3;
    return __int_as_float(v & 0x8FFFE000);                 // = (lo + hi) * 2^-112; subnormal halves flush
}

template <int fold_k2, bool out16>
__global__ void __launch_bounds__(256, 1) gemm_fold_kernel(
        const half * __restrict__ W, const half * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int M, const int N, const int K, const int64_t sy, half * __restrict__ R) {
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BN];

    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BN;

    float acc[8][8];
    half2 h[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
            h[i][j]   = make_half2(0.0f, 0.0f);
        }
    }

    uint4 ra[2];
    uint4 rb[2];
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            ra[i] = m0 + r < M ? *(const uint4 *) (W + (int64_t) (m0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
            rb[i] = n0 + r < N ? *(const uint4 *) (X + (int64_t) (n0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            // XOR swizzle: the 4 threads of a row write 4 different k2 rows of the tile, which
            // without it land in the same bank (4-way conflict); reads stay 16-byte vectors
            const int rs = r ^ (c << 3);
            As[buf][c*4 + 0][rs] = ra[i].x; As[buf][c*4 + 1][rs] = ra[i].y;
            As[buf][c*4 + 2][rs] = ra[i].z; As[buf][c*4 + 3][rs] = ra[i].w;
            Bs[buf][c*4 + 0][rs] = rb[i].x; Bs[buf][c*4 + 1][rs] = rb[i].y;
            Bs[buf][c*4 + 2][rs] = rb[i].z; Bs[buf][c*4 + 3][rs] = rb[i].w;
        }
    };

    const int nt = K / (2*BK2);
    gload(0);
    sstore(0);
    __syncthreads();

    for (int it = 0; it < nt; it++) {
        const int buf = it & 1;
        if (it + 1 < nt) {
            gload((it + 1)*2*BK2);
        }
        // the first product of a chain starts it (HMUL2) instead of zeroing 64 registers
        const bool restart = (it*BK2) % fold_k2 == 0;
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int   sw = (k2 >> 2) << 3;
            const uint4 a0 = *(const uint4 *) &As[buf][k2][(ty*4) ^ sw];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][(64 + ty*4) ^ sw];
            const uint4 b0 = *(const uint4 *) &Bs[buf][k2][(tx*4) ^ sw];
            const uint4 b1 = *(const uint4 *) &Bs[buf][k2][(64 + tx*4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = k2 == 0 && restart ? __hmul2(ai, bj) : __hfma2(ai, bj, h[i][j]);
                }
            }
        }
        if (((it + 1)*BK2) % fold_k2 == 0 || it + 1 == nt) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] += gemm_fold_h(h[i][j]);
                }
            }
        }
        if (it + 1 < nt) {
            sstore(buf ^ 1);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = n0 + (j < 4 ? tx*4 + j : 64 + tx*4 + j - 4);
        if (n >= N) {
            continue;
        }
        const float s = cs[n];
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih*64 + ty*4;
            if (m >= M) {
                continue;
            }
            float4 v = make_float4(acc[ih*4 + 0][j]*s, acc[ih*4 + 1][j]*s, acc[ih*4 + 2][j]*s, acc[ih*4 + 3][j]*s);
            if (out16) {
                v.x = __half2float(__float2half(v.x)); v.y = __half2float(__float2half(v.y));
                v.z = __half2float(__float2half(v.z)); v.w = __half2float(__float2half(v.w));
            }
            *(float4 *) (Y + n*sy + m) = v;
            if (out16 && R != nullptr) {
                // the tensor-parallel peer's landing buffer, over P2P: the same f16 values the
                // compressed peer copy would send (v is f16-exact here)
                half2 r[2] = {__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w)};
                *(uint2 *) (R + n*sy + m) = *(const uint2 *) r;
            }
        }
    }
}

// The same products, chains and folds as gemm_fold_kernel<128, true>, with cheaper bookkeeping:
// load/store offsets computed once, the tile loop unrolled by two (compile-time buffer index), the
// fold as HADD2.F32 with the 2^-112 moved into the column scale (a power of two, so every rounding
// is unchanged), and blocks grouped by RASTER weight row-blocks for L2 reuse. Needs K % 64 == 0.
// Harness 8704x5120 N=2048: 14.57 -> 13.68 ms, 0 of 17.8M outputs differ.
// The u2 kernel's arithmetic for small batches: a BM x BNN tile (BNN = 32 or 64) instead of 128 columns,
// so a 9..64-token batch stops paying for a 128-wide tile of padding. Thread (tx, ty) keeps the same 8 rows
// and TN = BNN/16 adjacent columns; every output is the same chain of products, folds (HADD2 to fp32) and
// rounding as in gemm_fold_kernel_u2, only fewer columns per block. (gemm_fold_kernel's integer fold
// keeps the sums at 2^-112 scale, where outputs near zero land in fp32 subnormals: not the same values.)
template <int BNN>
__global__ void __launch_bounds__(256, 1) gemm_fold_kernel_narrow(
        const half * __restrict__ W, const half * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int M, const int N, const int K, const int64_t sy) {
    static_assert(BNN == 32 || BNN == 64, "BNN");
    constexpr int TN = BNN/16;
    constexpr int fold_k2 = 128;
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BNN];

    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const int m0 = blockIdx.x*BM;
    const int n0 = blockIdx.y*BNN;

    float acc[8][TN];
    half2 h[8][TN];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
            h[i][j]   = make_half2(0.0f, 0.0f);
        }
    }

    // B: BNN rows x 4 uint4 = BNN*4 loads (all threads for 64, the first half for 32)
    const bool bt = t < BNN*4;
    uint4 ra[2];
    uint4 rb;
    auto gload = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            ra[i] = m0 + r < M ? *(const uint4 *) (W + (int64_t) (m0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
        }
        const int r = t >> 2;
        const int c = t & 3;
        rb = bt && n0 + r < N ? *(const uint4 *) (X + (int64_t) (n0 + r)*K + k0 + c*8) : make_uint4(0, 0, 0, 0);
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int l = t + 256*i;
            const int r = l >> 2;
            const int c = l & 3;
            const int rs = r ^ (c << 3);
            As[buf][c*4 + 0][rs] = ra[i].x; As[buf][c*4 + 1][rs] = ra[i].y;
            As[buf][c*4 + 2][rs] = ra[i].z; As[buf][c*4 + 3][rs] = ra[i].w;
        }
        if (bt) {
            const int r = t >> 2;
            const int c = t & 3;
            const int rs = r ^ (c << 3);
            Bs[buf][c*4 + 0][rs] = rb.x; Bs[buf][c*4 + 1][rs] = rb.y;
            Bs[buf][c*4 + 2][rs] = rb.z; Bs[buf][c*4 + 3][rs] = rb.w;
        }
    };

    const int nt = K / (2*BK2);
    gload(0);
    sstore(0);
    __syncthreads();

    for (int it = 0; it < nt; it++) {
        const int buf = it & 1;
        if (it + 1 < nt) {
            gload((it + 1)*2*BK2);
        }
        const bool restart = (it*BK2) % fold_k2 == 0;
#pragma unroll
        for (int k2 = 0; k2 < BK2; k2++) {
            const int   sw = (k2 >> 2) << 3;
            const uint4 a0 = *(const uint4 *) &As[buf][k2][(ty*4) ^ sw];
            const uint4 a1 = *(const uint4 *) &As[buf][k2][(64 + ty*4) ^ sw];
            const uint32_t a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            uint32_t b[TN];
            if constexpr (TN == 4) {
                const uint4 b0 = *(const uint4 *) &Bs[buf][k2][(tx*4) ^ sw];
                b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            } else {
                const uint2 b0 = *(const uint2 *) &Bs[buf][k2][(tx*2) ^ sw];
                b[0] = b0.x; b[1] = b0.y;
            }
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = k2 == 0 && restart ? __hmul2(bj, ai) : __hfma2(bj, ai, h[i][j]);
                }
            }
        }
        if (((it + 1)*BK2) % fold_k2 == 0 || it + 1 == nt) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += __half2float(__hadd(__low2half(h[i][j]), __high2half(h[i][j])));
                }
            }
        }
        if (it + 1 < nt) {
            sstore(buf ^ 1);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < TN; j++) {
        const int n = n0 + tx*TN + j;
        if (n >= N) {
            continue;
        }
        const float s = cs[n] * 0x1p-112f;
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih*64 + ty*4;
            if (m >= M) {
                continue;
            }
            float4 v = make_float4(acc[ih*4 + 0][j]*s, acc[ih*4 + 1][j]*s, acc[ih*4 + 2][j]*s, acc[ih*4 + 3][j]*s);
            v.x = __half2float(__float2half(v.x)); v.y = __half2float(__float2half(v.y));
            v.z = __half2float(__float2half(v.z)); v.w = __half2float(__float2half(v.w));
            *(float4 *) (Y + n*sy + m) = v;
        }
    }
}

constexpr int GEMM_FOLD_RASTER = 4;

// PAIR: two weights sharing X in one launch (gemm_fold_pair). Rows [0, Ms) come from W into Y, rows
// [Ms, M) from W2 into Y2; Ms is a multiple of BM, so each block sits wholly on one side and every
// output is computed exactly as in two separate launches.
template <bool PAIR>
__global__ void __launch_bounds__(256, 1) gemm_fold_kernel_u2(
        const half * __restrict__ W, const half * __restrict__ X, const float * __restrict__ cs,
        float * __restrict__ Y, const int M, const int N, const int K, const int64_t sy,
        const half * __restrict__ W2, float * __restrict__ Y2, const int Ms, const int64_t sy2) {
    __shared__ __align__(16) uint32_t As[2][BK2][BM];
    __shared__ __align__(16) uint32_t Bs[2][BK2][BN];

    const int t   = threadIdx.x;
    const int tx  = t & 15;
    const int ty  = t >> 4;
    const int gm  = gridDim.x;
    const int gn  = gridDim.y;
    const int lin = blockIdx.x + blockIdx.y*gm;
    const int grp = lin / (GEMM_FOLD_RASTER*gn);
    const int gsz = min(GEMM_FOLD_RASTER, gm - grp*GEMM_FOLD_RASTER);
    const int m0g = (grp*GEMM_FOLD_RASTER + (lin % (GEMM_FOLD_RASTER*gn)) % gsz)*BM;
    const int n0  = ((lin % (GEMM_FOLD_RASTER*gn)) / gsz)*BN;
    const bool second = PAIR && m0g >= Ms;
    const half * Wc  = second ? W2 : W;
    float *      Yc  = second ? Y2 : Y;
    const int    m0  = second ? m0g - Ms : m0g;
    const int    Mc  = !PAIR ? M : second ? M - Ms : Ms;
    const int64_t syc = second ? sy2 : sy;

    float acc[8][8];
    half2 h[8][8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
            h[i][j]   = make_half2(0.0f, 0.0f);
        }
    }

    const uint4 * pa[2];
    const uint4 * pb[2];
    bool va[2], vb[2];
    int  so[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int l = t + 256*i, r = l >> 2, c = l & 3;
        va[i] = m0 + r < Mc;
        vb[i] = n0 + r < N;
        pa[i] = (const uint4 *) (Wc + (int64_t) (va[i] ? m0 + r : 0)*K + c*8);
        pb[i] = (const uint4 *) (X + (int64_t) (vb[i] ? n0 + r : 0)*K + c*8);
        so[i] = (c*4)*BM + (r ^ (c << 3));   // same XOR swizzle as gemm_fold_kernel
    }
    int ao[4], bo[4];
#pragma unroll
    for (int g = 0; g < 4; g++) {
        ao[g] = (ty*4) ^ (g << 3);
        bo[g] = (tx*4) ^ (g << 3);
    }

    uint4 ra[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    uint4 rb[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
    auto gload = [&](const int it) {   // tile it = K offset it*32 halves = it*4 uint4
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (va[i]) ra[i] = pa[i][it*4];
            if (vb[i]) rb[i] = pb[i][it*4];
        }
    };
    auto sstore = [&](const int buf) {
#pragma unroll
        for (int i = 0; i < 2; i++) {
            uint32_t * a = &As[buf][0][0] + so[i];
            uint32_t * b = &Bs[buf][0][0] + so[i];
            a[0] = ra[i].x; a[BM] = ra[i].y; a[2*BM] = ra[i].z; a[3*BM] = ra[i].w;
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
            // j outer and b in the first slot: the same chains (bit-identical), but ptxas then leaves ~210
            // instead of ~380 same-bank source pairs after bankfix (OPTLOG 267)
#pragma unroll
            for (int j = 0; j < 8; j++) {
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const half2 ai = *(const half2 *) &a[i];
                    const half2 bj = *(const half2 *) &b[j];
                    h[i][j] = k2 == 0 && restart ? __hmul2(bj, ai) : __hfma2(bj, ai, h[i][j]);
                }
            }
        }
    };
    auto fold = [&]() {
#pragma unroll
        for (int i = 0; i < 8; i++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[i][j] += __half2float(__hadd(__low2half(h[i][j]), __high2half(h[i][j])));
            }
        }
    };

    constexpr int TPF = 128/BK2;       // tiles per fold
    const int nt = K / (2*BK2);        // even: K % 64 == 0
    gload(0);
    sstore(0);
    __syncthreads();
    for (int it = 0; it < nt; it += 2) {
        gload(it + 1);
        tile(0, it % TPF == 0);
        sstore(1);
        __syncthreads();
        if (it + 2 < nt) {
            gload(it + 2);
        }
        tile(1, false);
        if ((it + 2) % TPF == 0 || it + 2 >= nt) {
            fold();
        }
        if (it + 2 < nt) {
            sstore(0);
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int n = n0 + (j < 4 ? tx*4 + j : 64 + tx*4 + j - 4);
        if (n >= N) {
            continue;
        }
        const float s = cs[n] * 0x1p-112f;
#pragma unroll
        for (int ih = 0; ih < 2; ih++) {
            const int m = m0 + ih*64 + ty*4;
            if (m >= Mc) {
                continue;
            }
            float4 v = make_float4(acc[ih*4 + 0][j]*s, acc[ih*4 + 1][j]*s, acc[ih*4 + 2][j]*s, acc[ih*4 + 3][j]*s);
            v.x = __half2float(__float2half(v.x)); v.y = __half2float(__float2half(v.y));
            v.z = __half2float(__float2half(v.z)); v.w = __half2float(__float2half(v.w));
            *(float4 *) (Yc + n*syc + m) = v;
        }
    }
}

} // namespace

static int ggml_cuda_gemm_fold_env(const char * name, int def) {
    const char * s = getenv(name);
    return s ? atoi(s) : def;
}

// gemm_fold_kernel_u2 through its register-bank-fixed SASS (gemm-fold-u2-sass.h, built from this file's
// kernel by p100-handoff/tools/sass-gemm/u2cubin.py): ptxas leaves ~40% of the HFMA2s with two
// operands in one register bank; renaming registers removes most of those stalls without changing any
// instruction. Loaded once per device; if it fails to load (or GGML_CUDA_GEMM_FOLD_SASS=0), the
// compiled kernel runs instead.
static void gemm_fold_launch_u2(const bool pair, const dim3 grid, cudaStream_t stream,
        const half * W, const half * X, const float * cs, float * Y, int M, int N, int K, int64_t sy,
        const half * W2, float * Y2, int Ms, int64_t sy2) {
    static const bool sass_env = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_SASS", 1) != 0;
    static int        state[GGML_CUDA_MAX_DEVICES][2] = {};   // 0 not tried, 1 loaded, -1 unavailable
    static CUfunction fn[GGML_CUDA_MAX_DEVICES][2]    = {};
    if (sass_env) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        int & st = state[dev][pair];
        if (st == 0) {
            CUmodule mod;
            const void * img  = pair ? (const void *) gemm_fold_u2_sass_1 : (const void *) gemm_fold_u2_sass_0;
            const char * name = pair ? gemm_fold_u2_sass_name_1 : gemm_fold_u2_sass_name_0;
            st = cuModuleLoadData(&mod, img) == CUDA_SUCCESS && cuModuleGetFunction(&fn[dev][pair], mod, name) == CUDA_SUCCESS ? 1 : -1;
            if (st < 0) {
                GGML_LOG_WARN("%s: bank-fixed u2 SASS unavailable, using the compiled kernel\n", __func__);
            }
        }
        if (st > 0) {
            void * args[] = {&W, &X, &cs, &Y, &M, &N, &K, &sy, &W2, &Y2, &Ms, &sy2};
            if (cuLaunchKernel(fn[dev][pair], grid.x, grid.y, grid.z, 256, 1, 1, 0, stream, args, nullptr) != CUDA_SUCCESS) {
                GGML_ABORT("cuLaunchKernel failed for the u2 SASS");
            }
            return;
        }
    }
    if (pair) {
        gemm_fold_kernel_u2<true><<<grid, 256, 0, stream>>>(W, X, cs, Y, M, N, K, sy, W2, Y2, Ms, sy2);
    } else {
        gemm_fold_kernel_u2<false><<<grid, 256, 0, stream>>>(W, X, cs, Y, M, N, K, sy, W2, Y2, Ms, sy2);
    }
}

static int ggml_cuda_gemm_fold_mode() {
    // 0: off (cuBLAS), 1: on, fp32 outputs, 2: on, outputs rounded to f16 (keeps the f16 peer exchange)
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD", 2);
    return v;
}

static int ggml_cuda_gemm_fold_k2() {
    // half2 steps per fp16 chain (16, 32, 64, 128): 128 -> a fold every 256 values of K
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_K2", 128);
    return v;
}

static int ggml_cuda_gemm_fold_xexp() {
    // per-column prescale puts max|x| in [2^xexp, 2^(xexp+1)); -1 disables the prescale
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_XEXP", 3);
    return v;
}

static int ggml_cuda_gemm_fold_min_rows() {
    // below this many rows a 128-row tile leaves most SMs idle; those (small) matmuls run in fp32 cuBLAS
    static const int v = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_MINROWS", 1024);
    return v;
}

static bool ggml_cuda_gemm_fold_eligible(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                         const ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return ggml_cuda_gemm_fold_mode() != 0 && cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA && fast_fp16_available(cc) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        !ggml_cuda_mmvq_chunked_ok(cc, src0, src1, dst);
}

bool ggml_cuda_gemm_fold_wants_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   const ggml_tensor * dst) {
    return ggml_cuda_gemm_fold_eligible(ctx, src0, src1, dst) && src0->ne[1] < ggml_cuda_gemm_fold_min_rows();
}

// Weight prefetch (see gemm-fold.cuh): per device, which fold weights followed each exchanged matmul
// (learned on the first pass), and a buffer the exchange wait dequantizes them into.
struct gemm_fold_prefetch_state {
    static constexpr int NMAX = 6;
    const void * trigger = nullptr;      // src0 of the matmul whose exchange is pending
    const void * learning = nullptr;     // trigger whose followers are being recorded
    std::unordered_map<const void *, std::vector<const ggml_tensor *>> next;
    half *       buf     = nullptr;
    size_t       cap     = 0;            // halfs
    bool         broken  = false;
    const void * held[NMAX] = {};
    size_t       off[NMAX]  = {};
};
static gemm_fold_prefetch_state gemm_fold_pf[GGML_CUDA_MAX_DEVICES];

static bool gemm_fold_prefetch_on() {
    static const bool on = ggml_cuda_gemm_fold_env("GGML_CUDA_FOLD_PREFETCH", 1) != 0;
    return on;
}

static int gemm_fold_prefetch_max() {
    // fold weights dequantized ahead per exchange wait (all the ones up to the next exchange by default)
    static const int v = std::max(1, std::min(gemm_fold_prefetch_state::NMAX,
        ggml_cuda_gemm_fold_env("GGML_CUDA_FOLD_PREFETCH_N", gemm_fold_prefetch_state::NMAX)));
    return v;
}

void ggml_cuda_gemm_fold_prefetch(ggml_backend_cuda_context & ctx) {
    auto & st = gemm_fold_pf[ctx.device];
    for (auto & h : st.held) {
        h = nullptr;
    }
    if (!gemm_fold_prefetch_on() || st.broken || st.trigger == nullptr) {
        return;
    }
    auto it = st.next.find(st.trigger);
    if (it == st.next.end() || it->second.empty()) {
        return;
    }
    const auto & w = it->second;
    size_t need = 0;
    for (const ggml_tensor * t : w) {
        need += (size_t) ggml_nelements(t);
    }
    if (need > st.cap) {
        if (st.buf != nullptr) {
            CUDA_CHECK(cudaFree(st.buf));
            st.buf = nullptr;
            st.cap = 0;
        }
        // keep a safety margin: GPU0 also drives the desktop
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        if (free_b < need*sizeof(half) + ((size_t) 1 << 30) || cudaMalloc(&st.buf, need*sizeof(half)) != cudaSuccess) {
            (void) cudaGetLastError();
            st.buf = nullptr;
            st.broken = true;
            return;
        }
        st.cap = need;
    }
    size_t o = 0;
    for (size_t i = 0; i < w.size(); i++) {
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(w[i]->type);
        to_fp16(w[i]->data, st.buf + o, ggml_nelements(w[i]), ctx.stream());
        st.held[i] = w[i]->data;
        st.off[i]  = o;
        o += (size_t) ggml_nelements(w[i]);
    }
}

// the prefetched f16 copy of w, or nullptr
static const half * gemm_fold_prefetched(ggml_backend_cuda_context & ctx, const ggml_tensor * w) {
    auto & st = gemm_fold_pf[ctx.device];
    for (int i = 0; i < gemm_fold_prefetch_state::NMAX; i++) {
        if (st.held[i] != nullptr && st.held[i] == w->data) {
            st.held[i] = nullptr;
            return st.buf + st.off[i];
        }
    }
    return nullptr;
}

// record the fold weights that follow an exchange, in order, until the next exchange (first pass only)
static void gemm_fold_learn(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * w2) {
    auto & st = gemm_fold_pf[ctx.device];
    if (st.learning != nullptr) {
        auto & v = st.next[st.learning];
        for (const ggml_tensor * t : {w, w2}) {
            if (t != nullptr && (int) v.size() < gemm_fold_prefetch_max() && ggml_get_to_fp16_cuda(t->type) != nullptr &&
                    t->type != GGML_TYPE_F16) {
                v.push_back(t);
            }
        }
    }
}

// an exchanged matmul: it ends the list being learned and starts its own (unless already known)
static void gemm_fold_trigger(ggml_backend_cuda_context & ctx, const void * w) {
    auto & st = gemm_fold_pf[ctx.device];
    st.trigger  = w;
    st.learning = st.next.count(w) ? nullptr : w;
    if (st.learning != nullptr) {
        st.next[w].clear();
    }
}

// Pairing (see gemm-fold.cuh): the graph loop names the next node; fold_try runs it too when it is a
// fold matmul on the same activations.
static thread_local ggml_tensor * gemm_fold_partner      = nullptr;
static thread_local bool          gemm_fold_partner_done = false;

void ggml_cuda_gemm_fold_set_partner(ggml_tensor * next) {
    static const bool on = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_PAIR", 1) != 0;
    gemm_fold_partner      = on ? next : nullptr;
    gemm_fold_partner_done = false;
}

bool ggml_cuda_gemm_fold_take_partner_done() {
    const bool d = gemm_fold_partner_done;
    gemm_fold_partner      = nullptr;
    gemm_fold_partner_done = false;
    return d;
}

static bool gemm_fold_shape_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                               const ggml_tensor * dst) {
    if (!ggml_cuda_gemm_fold_eligible(ctx, src0, src1, dst) || src0->ne[1] < ggml_cuda_gemm_fold_min_rows()) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        K % 32 != 0 || M % 4 != 0 || src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(dst) || M > INT_MAX || N > 65535*BN) {
        return false;
    }
    return src0->type == GGML_TYPE_F16 || ggml_get_to_fp16_cuda(src0->type) != nullptr;
}

// SwiGLU handoff (see gemm-fold.cuh): the GLU node the graph loop skipped, consumed by the next fold.
static thread_local const ggml_tensor * gemm_fold_glu = nullptr;

static bool gemm_fold_glu_input_ok(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->nb[1] % (4*sizeof(float)) == 0 &&
           ((uintptr_t) t->data) % 16 == 0 && t->ne[2] == 1 && t->ne[3] == 1;
}

bool ggml_cuda_gemm_fold_glu_ok(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, const ggml_tensor * mm) {
    static const bool on = ggml_cuda_gemm_fold_env("GGML_CUDA_FOLD_GLU", 1) != 0;
    static const bool pv = ggml_cuda_gemm_fold_env("GGML_CUDA_FOLD_PRESCALE_V", 1) != 0;
    const ggml_tensor * src0 = mm->src[0];
    const int64_t K = src0->ne[0];
    return on && pv && mm->src[1] == glu && gemm_fold_shape_ok(ctx, src0, glu, mm) &&
           K % 4 == 0 && K <= 1024*9 && glu->src[0]->ne[0] == K && glu->src[1]->ne[0] == K &&
           glu->src[0]->ne[1] == glu->ne[1] && glu->src[1]->ne[1] == glu->ne[1] &&
           gemm_fold_glu_input_ok(glu->src[0]) && gemm_fold_glu_input_ok(glu->src[1]);
}

void ggml_cuda_gemm_fold_set_glu(const ggml_tensor * glu) {
    gemm_fold_glu = glu;
}

bool ggml_cuda_gemm_fold_glu_pending() {
    return gemm_fold_glu != nullptr;
}

bool ggml_cuda_gemm_fold_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             ggml_tensor * dst) {
    const int mode = ggml_cuda_gemm_fold_mode();
    if (!gemm_fold_shape_ok(ctx, src0, src1, dst)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const to_fp16_cuda_t to_fp16 = src0->type == GGML_TYPE_F16 ? nullptr : ggml_get_to_fp16_cuda(src0->type);

    cudaStream_t stream = ctx.stream();

    const half * W16 = (const half *) src0->data;
    ggml_cuda_pool_alloc<half> w_alloc(ctx.pool());
    const half * w_pf = to_fp16 ? gemm_fold_prefetched(ctx, src0) : nullptr;
    if (w_pf != nullptr) {
        W16 = w_pf;
    } else if (to_fp16) {
        w_alloc.alloc(ggml_nelements(src0));
        to_fp16(src0->data, w_alloc.get(), ggml_nelements(src0), stream);
        W16 = w_alloc.get();
    }

    ggml_cuda_pool_alloc<half>  x_alloc(ctx.pool(), N*K);
    ggml_cuda_pool_alloc<float> s_alloc(ctx.pool(), N);
    {
        // one read of X (same outputs); GGML_CUDA_FOLD_PRESCALE_V=0: the two-pass kernel
        static const bool pv = ggml_cuda_gemm_fold_env("GGML_CUDA_FOLD_PRESCALE_V", 1) != 0;
        const float * xs = (const float *) src1->data;
        const int64_t s1 = src1->nb[1]/sizeof(float);
        const bool vec = pv && K % 4 == 0 && s1 % 4 == 0 && ((uintptr_t) xs) % 16 == 0;
        if (gemm_fold_glu != nullptr && src1 == gemm_fold_glu) {
            // skipped SwiGLU: gate and up straight from their tensors (checked by ggml_cuda_gemm_fold_glu_ok)
            const ggml_tensor * g = src1->src[0];
            const ggml_tensor * u = src1->src[1];
            const float * gd = (const float *) g->data;
            const float * ud = (const float *) u->data;
            const int64_t sg = g->nb[1]/sizeof(float);
            const int64_t su = u->nb[1]/sizeof(float);
            const int     xe = ggml_cuda_gemm_fold_xexp();
            if (K <= 1024*4) {
                gemm_fold_prescale_v<4, true><<<N, 256, 0, stream>>>(gd, sg, x_alloc.get(), s_alloc.get(), K, xe, ud, su);
            } else if (K <= 1024*6) {
                gemm_fold_prescale_v<6, true><<<N, 256, 0, stream>>>(gd, sg, x_alloc.get(), s_alloc.get(), K, xe, ud, su);
            } else {
                gemm_fold_prescale_v<9, true><<<N, 256, 0, stream>>>(gd, sg, x_alloc.get(), s_alloc.get(), K, xe, ud, su);
            }
            gemm_fold_glu = nullptr;
        } else if (vec && K <= 1024*4) {
            gemm_fold_prescale_v<4><<<N, 256, 0, stream>>>(xs, s1, x_alloc.get(), s_alloc.get(), K, ggml_cuda_gemm_fold_xexp());
        } else if (vec && K <= 1024*6) {
            gemm_fold_prescale_v<6><<<N, 256, 0, stream>>>(xs, s1, x_alloc.get(), s_alloc.get(), K, ggml_cuda_gemm_fold_xexp());
        } else if (vec && K <= 1024*9) {
            gemm_fold_prescale_v<9><<<N, 256, 0, stream>>>(xs, s1, x_alloc.get(), s_alloc.get(), K, ggml_cuda_gemm_fold_xexp());
        } else {
            gemm_fold_prescale<<<N, 256, 0, stream>>>(xs, s1, x_alloc.get(), s_alloc.get(), K, ggml_cuda_gemm_fold_xexp());
        }
    }

    const dim3 grid((M + BM - 1)/BM, (N + BN - 1)/BN);
    const int64_t sy = dst->nb[1]/sizeof(float);
    const half  * X16 = x_alloc.get();
    const float * cs  = s_alloc.get();
    float       * Y   = (float *) dst->data;
    const bool    o16 = mode == 2;

    // exchange source: token chunks with an event after each (see xchg in common.cuh)
    static const int xchg_chunks = std::max(1, std::min(8, ggml_cuda_gemm_fold_env("GGML_CUDA_XCHG_CHUNKS", 4)));
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capturing));
    // direct mode (GGML_CUDA_XCHG_DIRECT=1, off): also write the f16 outputs straight into the peer's landing
    // buffer over P2P. Measured 253 t/s against 411 for the chunked copies: the epilogue's scattered 8-byte
    // stores make poor PCIe transactions across the two CPU root ports (OPTLOG 237).
    static const bool xchg_direct = ggml_cuda_gemm_fold_env("GGML_CUDA_XCHG_DIRECT", 0) != 0;
    ggml_backend_cuda_context * peer = ctx.xchg_peer;
    // GGML_CUDA_GEMM_FOLD_U2=0 falls back to gemm_fold_kernel (bit-identical, slower)
    static const bool u2_env = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_U2", 1) != 0;
    const bool u2 = u2_env && o16 && K % 64 == 0 && ggml_cuda_gemm_fold_k2() == 128;

    // small batches: a 32/64-column tile, outputs identical to the 128-column kernels
    // (GGML_CUDA_GEMM_FOLD_NARROW=0: always 128 columns)
    static const int narrow = ggml_cuda_gemm_fold_env("GGML_CUDA_GEMM_FOLD_NARROW", 64);
    if (u2 && N <= narrow) {
        gemm_fold_partner = nullptr;
        gemm_fold_learn(ctx, src0, nullptr);
        if (N <= 32) {
            gemm_fold_kernel_narrow<32><<<dim3(grid.x, (N + 31)/32), 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy);
        } else {
            gemm_fold_kernel_narrow<64><<<dim3(grid.x, (N + 63)/64), 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy);
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    // gate/up style pair: one prescale (above) and one launch over both weights' rows
    ggml_tensor * pt = gemm_fold_partner;
    gemm_fold_partner = nullptr;
    // pairing holds both f16 weights at once (+ one weight of pool); only with room to spare (262k runs
    // leave GPU0 ~0.5 GB, and GPU0 also drives the desktop)
    static int pair_room[GGML_CUDA_MAX_DEVICES] = {0};   // 0 unknown, 1 yes, -1 no
    if (pt != nullptr && pair_room[ctx.device] == 0) {
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        pair_room[ctx.device] = free_b > ((size_t) 3 << 29) ? 1 : -1;
    }
    if (pt != nullptr && (pair_room[ctx.device] < 0 || gemm_fold_pf[ctx.device].broken)) {
        pt = nullptr;
    }
    if (pt != nullptr && u2 && !ctx.xchg_want && M % BM == 0 && pt->op == GGML_OP_MUL_MAT && pt->src[1] == src1 &&
            pt->type == GGML_TYPE_F32 && pt->src[0]->type == src0->type && pt->src[0]->ne[0] == K &&
            pt->src[0]->ne[1] >= ggml_cuda_gemm_fold_min_rows() && pt->src[0]->ne[1] % 4 == 0 &&
            pt->src[0]->ne[1] + M <= INT_MAX && pt->ne[1] == N && pt->op_params[0] == dst->op_params[0] &&
            ggml_cuda_gemm_fold_eligible(ctx, pt->src[0], src1, pt) && ggml_is_contiguous(pt->src[0]) &&
            ggml_is_contiguous(pt) && pt->src[0]->ne[2] == 1 && pt->src[0]->ne[3] == 1) {
        const int64_t M2 = pt->src[0]->ne[1];
        const half * W2 = (const half *) pt->src[0]->data;
        ggml_cuda_pool_alloc<half> w2_alloc(ctx.pool());
        gemm_fold_learn(ctx, src0, pt->src[0]);
        const half * w2_pf = to_fp16 ? gemm_fold_prefetched(ctx, pt->src[0]) : nullptr;
        if (w2_pf != nullptr) {
            W2 = w2_pf;
        } else if (to_fp16) {
            w2_alloc.alloc(ggml_nelements(pt->src[0]));
            to_fp16(pt->src[0]->data, w2_alloc.get(), ggml_nelements(pt->src[0]), stream);
            W2 = w2_alloc.get();
        }
        const dim3 gp((M + M2 + BM - 1)/BM, (N + BN - 1)/BN);
        gemm_fold_launch_u2(true, gp, stream, W16, X16, cs, Y, (int) (M + M2), (int) N, (int) K, sy,
            W2, (float *) pt->data, (int) M, (int64_t) (pt->nb[1]/sizeof(float)));
        CUDA_CHECK(cudaGetLastError());
        gemm_fold_partner_done = true;
        return true;
    }
    gemm_fold_learn(ctx, src0, nullptr);
    if (ctx.xchg_want && xchg_direct && peer != nullptr && ctx.peer_f16_ok == 1 && N >= 512 && o16 &&
            ggml_cuda_gemm_fold_k2() == 128 && capturing == cudaStreamCaptureStatusNone && ggml_is_contiguous(dst)) {
        auto & xc = ctx.xchg;
        half * R = (half *) peer->peer_stage_get(ggml_backend_cuda_context::PEER_STAGE_IN, (size_t) M*N*sizeof(half));
        if (peer->peer_stage_free == nullptr) {
            ggml_cuda_set_device(peer->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&peer->peer_stage_free, cudaEventDisableTiming));
        }
        ggml_cuda_set_device(ctx.device);
        if (xc.ev[0] == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&xc.ev[0], cudaEventDisableTiming));
        }
        // the peer must have consumed its previous delivery before this kernel overwrites it
        CUDA_CHECK(cudaStreamWaitEvent(stream, peer->peer_stage_free, 0));
        gemm_fold_kernel<128, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, R);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(xc.ev[0], stream));
        xc.n = 1;
        xc.rows = M;
        xc.col[0] = 0;
        xc.col[1] = N;
        xc.direct = true;
        xc.data = dst->data;
        return true;
    }

    if (ctx.xchg_want && xchg_chunks > 1 && N >= 512 && o16 && ggml_cuda_gemm_fold_k2() == 128 &&
            capturing == cudaStreamCaptureStatusNone && ggml_is_contiguous(dst)) {
        // The exchange's f16 narrowing kernels run on the copy stream. At default priority they wait
        // behind compute-stream work (the next GEMM chunk, prefetch dequants) for free SMs; high
        // priority lets the block scheduler dispatch them first. Scheduling only.
        static const bool xchg_prio = ggml_cuda_gemm_fold_env("GGML_CUDA_XCHG_PRIO", 1) != 0;
        if (xchg_prio && ctx.copy_stream == nullptr) {
            int lo = 0, hi = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.copy_stream, cudaStreamNonBlocking, hi));
        }
        gemm_fold_trigger(ctx, src0->data);
        const int64_t nb   = (N + BN - 1)/BN;
        const int     nch  = (int) std::min<int64_t>(xchg_chunks, nb);
        auto & xc = ctx.xchg;
        xc.n = nch;
        xc.rows = M;
        for (int c = 0; c <= nch; c++) {
            xc.col[c] = std::min<int64_t>(N, (nb*c/nch)*BN);
        }
        for (int c = 0; c < nch; c++) {
            if (xc.ev[c] == nullptr) {
                CUDA_CHECK(cudaEventCreateWithFlags(&xc.ev[c], cudaEventDisableTiming));
            }
            const int64_t n0 = xc.col[c], nc = xc.col[c + 1] - n0;
            const dim3 gc((M + BM - 1)/BM, (nc + BN - 1)/BN);
            if (u2) {
                gemm_fold_launch_u2(false, gc, stream, W16, X16 + n0*K, cs + n0, Y + n0*sy, (int) M, (int) nc, (int) K, sy, nullptr, nullptr, 0, 0);
            } else {
                gemm_fold_kernel<128, true><<<gc, 256, 0, stream>>>(W16, X16 + n0*K, cs + n0, Y + n0*sy, M, (int) nc, K, sy, nullptr);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(xc.ev[c], stream));
        }
        xc.direct = false;
        xc.data = dst->data;
        return true;
    }

    if (u2) {
        gemm_fold_launch_u2(false, grid, stream, W16, X16, cs, Y, (int) M, (int) N, (int) K, sy, nullptr, nullptr, 0, 0);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    switch (ggml_cuda_gemm_fold_k2()) {
        case 16: o16 ? gemm_fold_kernel<16, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                     : gemm_fold_kernel<16, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
        default:  o16 ? gemm_fold_kernel<128, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                      : gemm_fold_kernel<128, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
        case 64: o16 ? gemm_fold_kernel<64, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                     : gemm_fold_kernel<64, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
        case 32: o16 ? gemm_fold_kernel<32, true><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr)
                     : gemm_fold_kernel<32, false><<<grid, 256, 0, stream>>>(W16, X16, cs, Y, M, N, K, sy, nullptr); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
