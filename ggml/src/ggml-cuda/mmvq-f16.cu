#include "mmvq-f16.cuh"
#include "unary.cuh"
#include "mmvq.cuh"

#include <unordered_map>

// q6_K x f32 matvec for 2..16 columns on Pascal, in fp16 with short chains folded into fp32.
// 2..5 columns are one speculative sequence's verify; 6..16 are several sequences verified together
// (parallel server slots). Wider than MMVQ_F16_MAXK, the columns run as two launches.
//
// The integer path (mmvq.cu) is ALU-bound here: sm_60 has no DP4A, and its emulation costs ~2.2
// instructions per multiply-add. This path converts each weight pair to fp16 once (PRMT, HSUB2,
// HMUL2) and shares it across the columns, then does one HFMA2 per two multiply-adds.
//
// Numerics, per column c and window w of 1024 values (4 q6_K blocks):
//   x    -> half, prescaled by a power of 2 so |x| < 1 (the scaling is exact; one rounding of x
//           to 11 bits, against the integer path's rounding of x to 8 bits in q8_1)
//   w    = (d*1024) * sc * (q - 32) in half: one rounding of the weight
//   each lane runs a fused fp16 chain of 16 HFMA2 over its window, and the two lanes are added and
//   folded into an fp32 accumulator once per window.
// Against a double-precision reference on this model's real weights: NMSE ~4.6e-7, where the
// integer path's q8_1 activations give ~1.3e-4 (OPTLOG 203, 206).
//
// Layout constraints: each row must hold an even number of q6_K blocks, so that every row and every
// 4-block window starts 4-byte aligned and block b's 2-byte phase (b & 1) is known at compile time.

#define MMVQ_F16_NW  2   // warps per block
#define MMVQ_F16_RPW 4   // rows per warp
#define MMVQ_F16_NBF 4   // q6_K blocks per window (fold)
static constexpr int MMVQ_F16_WIN = MMVQ_F16_NBF*256;
#define MMVQ_F16_MAXK 12 // widest single launch: wider, the 4-row tile runs out of registers

// Columns are independent (each one's arithmetic never touches another's), so splitting a batch into
// launches over column ranges gives the same results.
template <typename F>
static void mmvq_f16_columns(const int64_t ncols, F && f) {
    const int64_t n0 = ncols <= MMVQ_F16_MAXK ? ncols : (ncols + 1)/2;
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        f(c0, std::min(n0, ncols - c0));
    }
}

template <typename F>
static void mmvq_f16_nc(const int64_t nc, F && f) {
    switch (nc) {
        case 2:  f(std::integral_constant<int, 2>{});  break;
        case 3:  f(std::integral_constant<int, 3>{});  break;
        case 4:  f(std::integral_constant<int, 4>{});  break;
        case 5:  f(std::integral_constant<int, 5>{});  break;
        case 6:  f(std::integral_constant<int, 6>{});  break;
        case 7:  f(std::integral_constant<int, 7>{});  break;
        case 8:  f(std::integral_constant<int, 8>{});  break;
        case 9:  f(std::integral_constant<int, 9>{});  break;
        case 10: f(std::integral_constant<int, 10>{}); break;
        case 11: f(std::integral_constant<int, 11>{}); break;
        case 12: f(std::integral_constant<int, 12>{}); break;
        default: GGML_ABORT("mmvq_f16: %d columns", (int) nc);
    }
}

// one warp per (window, column)
static __global__ void mmvq_f16_prep(const float * __restrict__ X, const int64_t sx, __half * __restrict__ XS,
                                     float * __restrict__ S, const int K) {
    const int w = blockIdx.x*blockDim.y + threadIdx.y, c = blockIdx.y, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    if (w >= nw) {
        return;
    }
    const float * x = X + c*sx + (int64_t) w*MMVQ_F16_WIN;
    constexpr int PER = MMVQ_F16_WIN/32;
    const int len = min(MMVQ_F16_WIN, K - w*MMVQ_F16_WIN);
    float v[PER];
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (j*128 < len) {
            f = __ldg((const float4 *) x + j*32 + threadIdx.x);
        }
        v[4*j] = f.x; v[4*j + 1] = f.y; v[4*j + 2] = f.z; v[4*j + 3] = f.w;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(f.x), fabsf(f.y)), fmaxf(fabsf(f.z), fabsf(f.w))));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    }
    int e = 0;
    if (m > 0.0f) {
        frexpf(m, &e);
    }
    const float inv = ldexpf(1.0f, -e);
    __half * xs = XS + (int64_t) c*K + (int64_t) w*MMVQ_F16_WIN;
#pragma unroll
    for (int j = 0; j < PER/4; ++j) {
        if (j*128 >= len) {
            break;
        }
        const __half2 a = __floats2half2_rn(v[4*j]*inv, v[4*j + 1]*inv), b = __floats2half2_rn(v[4*j + 2]*inv, v[4*j + 3]*inv);
        uint2 u;
        u.x = *(const uint32_t *) &a;
        u.y = *(const uint32_t *) &b;
        ((uint2 *) xs)[j*32 + threadIdx.x] = u;
    }
    if (threadIdx.x == 0) {
        S[c*nw + w] = ldexpf(1.0f, e - 10); // also undoes the 1024 on the weights
    }
}

// 32-bit field at byte offset OFF of block B, relative to a 4-aligned window base
template <int B, int OFF>
static __device__ __forceinline__ uint32_t mmvq_f16_fld(const uint32_t * wb, const int lane_words) {
    constexpr int A = 210*B + OFF;
    if constexpr ((A & 3) == 0) {
        return wb[A/4 + lane_words];
    } else {
        return __byte_perm(wb[A/4 + lane_words], wb[A/4 + 1 + lane_words], 0x5432);
    }
}

template <int NC, int RPW, int NBLK, int B = 0>
static __device__ __forceinline__ void mmvq_f16_blocks(
        const uint32_t * const * wb, const uint2 * const * scs, const __half * const * xw, __half2 (&t)[NC][RPW],
        const int P, const int ql_w, const int qh_w, const int vh_shift, const int sp) {
    if constexpr (B < NBLK) {
        const __half2 k1056 = __float2half2_rn(1056.0f);
        __half2 xa[NC][2], xb[NC][2];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const __half * xs = xw[c] + B*256;
            const uint2 a = __ldg((const uint2 *) (xs + P)), bb = __ldg((const uint2 *) (xs + P + 64));
            xa[c][0] = *(const __half2 *) &a.x;  xa[c][1] = *(const __half2 *) &a.y;
            xb[c][0] = *(const __half2 *) &bb.x; xb[c][1] = *(const __half2 *) &bb.y;
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint32_t vl = mmvq_f16_fld<B, 0>(wb[i], ql_w);
            const uint32_t vh = mmvq_f16_fld<B, 128>(wb[i], qh_w) >> vh_shift;
            // scales so and so+4, precomputed per window by mmvq_f16_scales (the same half ops)
            const uint2 sAB = scs[i][B*8 + sp];
            const __half2 sA = *(const __half2 *) &sAB.x, sB = *(const __half2 *) &sAB.y;
            const uint32_t qa = (vl & 0x0F0F0F0F) | ((vh << 4) & 0x30303030);
            const uint32_t qb = ((vl >> 4) & 0x0F0F0F0F) | (vh & 0x30303030);
            // half(1024 + q) per byte, minus 1056, times the scale
            const uint32_t h0 = __byte_perm(qa, 0x64646464u, 0x5140), h1 = __byte_perm(qa, 0x64646464u, 0x5342);
            const uint32_t h2 = __byte_perm(qb, 0x64646464u, 0x5140), h3 = __byte_perm(qb, 0x64646464u, 0x5342);
            const __half2 w0 = __hmul2(__hsub2(*(const __half2 *) &h0, k1056), sA);
            const __half2 w1 = __hmul2(__hsub2(*(const __half2 *) &h1, k1056), sA);
            const __half2 w2 = __hmul2(__hsub2(*(const __half2 *) &h2, k1056), sB);
            const __half2 w3 = __hmul2(__hsub2(*(const __half2 *) &h3, k1056), sB);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                __half2 u = B == 0 ? __hmul2(w0, xa[c][0]) : __hfma2(w0, xa[c][0], t[c][i]);
                u = __hfma2(w1, xa[c][1], u);
                u = __hfma2(w2, xb[c][0], u);
                t[c][i] = __hfma2(w3, xb[c][1], u);
            }
        }
        mmvq_f16_blocks<NC, RPW, NBLK, B + 1>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp);
    }
}

// A block's 16 scales, as the (sA, sB) half2 pairs its lanes use: lane l of the warp computes pair
// p = l % 8 (scales j and j + 4, j = p < 4 ? p : p + 4) of block l / 8 of the window, once, instead
// of every lane recomputing its pair for every block (~12 instructions per lane, block and row).
// The half arithmetic is exactly the per-lane version's: sc2 = (half(1152 + sc) - 1152) * (d*1024).
template <int RPW>
static __device__ __forceinline__ void mmvq_f16_scales(const uint32_t * const * wb, uint2 (*scst)[MMVQ_F16_NBF*8],
                                                       const int nblk, const int lane) {
    const int B = lane >> 3, p = lane & 7, j = p < 4 ? p : p + 4;
    if (B >= nblk) {
        return;
    }
    const __half2 k1152 = __float2half2_rn(1152.0f), k1024 = __float2half2_rn(1024.0f);
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const uint8_t * bb = (const uint8_t *) wb[i] + 210*B;
        const uint32_t sab = (uint32_t) bb[192 + j] | ((uint32_t) bb[196 + j] << 8);
        const uint32_t sh  = __byte_perm(sab ^ 0x8080u, 0x64646464u, 0x5140); // half(1024 + 128 + sc)
        const uint32_t d16 = *(const uint16_t *) (bb + 208);
        const uint32_t dd  = d16 | (d16 << 16);
        const __half2 d2  = __hmul2(*(const __half2 *) &dd, k1024);
        const __half2 sc2 = __hmul2(__hsub2(*(const __half2 *) &sh, k1152), d2);
        const __half2 sA  = __low2half2(sc2), sB = __high2half2(sc2);
        scst[i][B*8 + p] = make_uint2(*(const uint32_t *) &sA, *(const uint32_t *) &sB);
    }
}

// The gated delta net's gate and beta in one launch (GATE): rows [0, rows1) are alpha's (W, Y),
// giving softplus(alpha + b) * m; the rest are beta's (W2, Y2), giving sigmoid(beta). The epilogues
// are the unfused ADD, SOFTPLUS, MUL and SIGMOID kernels' operations, so results are bit-identical.
struct mmvq_f16_gate {
    const uint8_t * W2;
    float *         Y2;
    const float *   b;
    const float *   m;
    int             rows1;
};

static __device__ __forceinline__ float mmvq_f16_softplus(const float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x)); // op_softplus (unary.cu)
}

// GLU: the FFN's gate and up matvecs and the SWIGLU after them in one launch. Each warp takes RPW/2
// output rows: its first RPW/2 row slots walk the gate matrix (W), the rest the same rows of up
// (gate.W2), and the epilogue writes silu(gate) * up with the unfused SWIGLU kernel's expression.
// Every row's arithmetic is the unfused kernel's, so the result is bit-identical.
template <int NC, int RPW, int NWT, bool KS, int MINB = 12/NWT, bool GATE = false, bool GLU = false>
__launch_bounds__(NWT*WARP_SIZE, MINB) // 12/NWT: 168 registers, no spills: 6 blocks per SM (OPTLOG 210)
static __global__ void mmvq_f16_q6_K(const uint8_t * __restrict__ W, const int64_t row_bytes, const __half * __restrict__ XS,
                                     const float * __restrict__ S, float * __restrict__ Y, const int64_t sy,
                                     int rows, const int K, const mmvq_f16_gate gate = {}) {
    static_assert(!GATE || (KS && RPW == 1), "GATE: split-K, one row per block");
    static_assert(!GLU || (!KS && !GATE && RPW % 2 == 0), "GLU: row-parallel, gate/up row pairs");
    constexpr int ORW = GLU ? RPW/2 : RPW; // output rows per warp
    constexpr int RPB = NWT*RPW, WB = MMVQ_F16_NBF*210, NU = (WB + 15 + 15)/16;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + MMVQ_F16_NBF - 1)/MMVQ_F16_NBF;
    // KS (split K): the block's warps share its RPW rows and take every NWT-th window each
    int row0 = KS ? blockIdx.x*RPW : GLU ? (blockIdx.x*NWT + wid)*ORW : blockIdx.x*RPB + wid*RPW;
    bool second = false;
    if constexpr (GATE) {
        if (row0 >= gate.rows1) {
            second = true;
            row0 -= gate.rows1;
            rows -= gate.rows1;
            W = gate.W2;
            Y = gate.Y2;
        } else {
            rows = gate.rows1;
        }
    }

    __shared__ uint4 wst[NWT][RPW][NU];

    // this lane's 8 values of a block: P..P+3 (scale so) and P+64..P+67 (scale so+4)
    const int iqs = lane, P = 128*(iqs/16) + 4*(iqs % 16);
    const int ql_w = iqs, qh_w = 8*(iqs/16) + iqs % 8, vh_shift = 2*((iqs % 16)/8), so = 8*(iqs/16) + (iqs % 16)/4;
    const int sp = so < 8 ? so : so - 4; // this lane's scale pair (so, so + 4), see mmvq_f16_scales
    __shared__ uint2 scst[NWT][RPW][MMVQ_F16_NBF*8];
    const uint2 * scs[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        scs[i] = scst[wid][i];
    }

    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const char * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        if constexpr (GLU) {
            rp[i] = (const char *) (i < ORW ? W : gate.W2) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
        } else {
            rp[i] = (const char *) W + (int64_t) min(row0 + i, rows - 1)*row_bytes + (KS ? (int64_t) wid*WB : 0);
        }
    }
    const __half * xw[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xw[c] = XS + (int64_t) c*K + (KS ? (int64_t) wid*MMVQ_F16_WIN : 0);
    }

    for (int win = KS ? wid : 0; win < nw; win += KS ? NWT : 1) {
        const int nblk = min(MMVQ_F16_NBF, nb - win*MMVQ_F16_NBF);
        __syncwarp();
        const uint32_t * wb[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const char * g = rp[i];
            rp[i] += KS ? (int64_t) NWT*WB : WB;
            const int m = (int) ((uintptr_t) g & 15); // a multiple of 4
            wb[i] = (const uint32_t *) wst[wid][i] + (m >> 2);
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + nblk*210 + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        mmvq_f16_scales<RPW>(wb, scst[wid], nblk, lane);
        __syncwarp();
        __half2 t[NC][RPW];
        switch (nblk) {
            case 4: mmvq_f16_blocks<NC, RPW, 4>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break;
            default: mmvq_f16_blocks<NC, RPW, 2>(wb, scs, xw, t, P, ql_w, qh_w, vh_shift, sp); break; // nb % 4 == 2
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            xw[c] += KS ? NWT*MMVQ_F16_WIN : MMVQ_F16_WIN;
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
    if constexpr (KS) {
        // per-warp partials, then warp 0 adds them in a fixed order
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                if constexpr (GATE) {
                    const int r = row0 + i;
                    v = second ? 1.0f / (1.0f + expf(-v)) : mmvq_f16_softplus(v + gate.b[r]) * gate.m[r];
                }
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW]; // op_swiglu (unary.cu)
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Every other weight type: one kernel, one small unpack per type.
//
// The q6_K kernel above is scheduled around q6_K's 210-byte block. This one takes any type through a
// per-type unpack of 8 consecutive weights of a block (a "chunk"), written from ggml's reference
// dequantization (ggml-quants.c), and shares the rest: the prescaled fp16 activation and its cache,
// the per-window fp16 chains folded into fp32, the column split. Numerics per weight:
//   q    -> half(1024 + q) by byte permute (exact), minus the bias (exact)
//   w    = (q - bias) * half(d*1024*sc) [+ half(-dmin*1024*m)]: one or two roundings, as on q6_K
// Lane l takes chunks l, l + 32, l + 64, l + 96 of each 1024-value window: 16 HFMA2 per window and
// column, as on q6_K, then the fold into fp32 with the window's activation scale.

static __device__ __forceinline__ uint32_t mmvq_f16_ld2(const uint8_t * p) { // 2-byte aligned
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}
static __device__ __forceinline__ uint32_t mmvq_f16_ld4(const uint8_t * p) { // 4-byte aligned
    return *(const uint32_t *) p;
}
// 4 bits (bit k of b) -> bit 4 of byte k
static __device__ __forceinline__ uint32_t mmvq_f16_spread4(const uint32_t b) {
    return ((b & 1) << 4) | ((b & 2) << 11) | ((b & 4) << 18) | ((b & 8) << 25);
}
// 4 nibble indices (one per byte) -> kvalues_iq4nl[i] + 128 per byte
static __device__ __forceinline__ uint32_t mmvq_f16_iq4(const uint32_t v) {
    const uint32_t x = v & 0x07070707, y = x | (x >> 4);
    const uint32_t sel = (y & 0xFF) | ((y >> 8) & 0xFF00);
    const uint32_t lo = __byte_perm(0x3f2d1801u, 0x766a5d4fu, sel), hi = __byte_perm(0xa6998d81u, 0xf1d9c5b5u, sel);
    const uint32_t m = ((v >> 3) & 0x01010101) * 0xFF;
    return (lo & ~m) | (hi & m);
}
// 8 bytes at any byte address (2 aligned words + funnel shifts), e.g. an mxfp4 block's qs
static __device__ __forceinline__ void mmvq_f16_ld8u(const uint8_t * p, uint32_t & a0, uint32_t & a1) {
    const uint32_t * q = (const uint32_t *) ((uintptr_t) p & ~(uintptr_t) 3);
    const uint32_t sh = ((uintptr_t) p & 3) * 8;
    const uint32_t w0 = q[0], w1 = q[1], w2 = q[2];
    a0 = __funnelshift_r(w0, w1, sh);
    a1 = __funnelshift_r(w1, w2, sh);
}
// 8 bytes (q per byte, element order) -> 8 weights (q - bias)*s, or (q - 1024 + 1024 - bias)*s + mn
template <bool MIN>
static __device__ __forceinline__ void mmvq_f16_w8(const uint32_t v0, const uint32_t v1, const float bias, const float s, const float mn,
                                                   __half2 (&w)[4]) {
    const __half2 kb = __float2half2_rn(1024.0f + bias), s2 = __float2half2_rn(s);
    const uint32_t h[4] = { __byte_perm(v0, 0x64646464u, 0x5140), __byte_perm(v0, 0x64646464u, 0x5342),
                            __byte_perm(v1, 0x64646464u, 0x5140), __byte_perm(v1, 0x64646464u, 0x5342) };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const __half2 q = __hsub2(*(const __half2 *) &h[k], kb);
        if constexpr (MIN) {
            w[k] = __hfma2(q, s2, __float2half2_rn(mn));
        } else {
            w[k] = __hmul2(q, s2);
        }
    }
}
static __device__ __forceinline__ void mmvq_f16_sm_k4(const int j, const uint8_t * q, int & d, int & m) { // get_scale_min_k4
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}
static __device__ __forceinline__ float mmvq_f16_h(const uint8_t * p) {
    return __half2float(*(const __half *) p);
}

template <ggml_type T> struct mmvq_f16_t;

template <> struct mmvq_f16_t<GGML_TYPE_Q4_0> {
    static constexpr int QK = 32, BS = 18;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 2 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        mmvq_f16_w8<false>((mmvq_f16_ld2(qs) >> sh) & 0x0F0F0F0F, (mmvq_f16_ld2(qs + 4) >> sh) & 0x0F0F0F0F, 8.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q4_1> {
    static constexpr int QK = 32, BS = 20;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 4 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        mmvq_f16_w8<true>((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F, (mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F, 0.0f,
                          mmvq_f16_h(b)*1024.0f, mmvq_f16_h(b + 2)*1024.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q5_0> {
    static constexpr int QK = 32, BS = 22;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 6 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        const uint32_t hb = mmvq_f16_ld2(b + 2) >> (8*sub); // element e's bit 4 is qh bit e
        mmvq_f16_w8<false>(((mmvq_f16_ld2(qs) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb),
                           ((mmvq_f16_ld2(qs + 4) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb >> 4), 16.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q5_1> {
    static constexpr int QK = 32, BS = 24;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 8 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        const uint32_t hb = mmvq_f16_ld4(b + 4) >> (8*sub);
        mmvq_f16_w8<true>(((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb),
                          ((mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb >> 4), 0.0f,
                          mmvq_f16_h(b)*1024.0f, mmvq_f16_h(b + 2)*1024.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q8_0> {
    static constexpr int QK = 32, BS = 34;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 2 + 8*sub;
        mmvq_f16_w8<false>(mmvq_f16_ld2(qs) ^ 0x80808080, mmvq_f16_ld2(qs + 4) ^ 0x80808080, 128.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_IQ4_NL> {
    static constexpr int QK = 32, BS = 18;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const uint8_t * qs = b + 2 + 8*(sub & 1);
        const int sh = 4*(sub >> 1);
        mmvq_f16_w8<false>(mmvq_f16_iq4((mmvq_f16_ld2(qs) >> sh) & 0x0F0F0F0F), mmvq_f16_iq4((mmvq_f16_ld2(qs + 4) >> sh) & 0x0F0F0F0F),
                           128.0f, mmvq_f16_h(b)*1024.0f, 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_IQ4_XS> {
    static constexpr int QK = 256, BS = 136;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int ib = sub >> 2, t = sub & 3;
        const uint8_t * qs = b + 8 + 16*ib + 8*(t & 1);
        const int sh = 4*(t >> 1);
        const int ls = ((b[4 + ib/2] >> 4*(ib % 2)) & 0xF) | (((*(const uint16_t *) (b + 2) >> 2*ib) & 3) << 4);
        mmvq_f16_w8<false>(mmvq_f16_iq4((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F), mmvq_f16_iq4((mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F),
                           128.0f, mmvq_f16_h(b)*1024.0f*(float) (ls - 32), 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q2_K> {
    static constexpr int QK = 256, BS = 84;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int n = sub >> 4, j = (sub >> 2) & 3, l0 = 8*(sub & 3);
        const uint8_t * qs = b + 16 + 32*n + l0;
        const int sc = b[8*n + 2*j + (l0 >= 16)];
        mmvq_f16_w8<true>((mmvq_f16_ld4(qs) >> 2*j) & 0x03030303, (mmvq_f16_ld4(qs + 4) >> 2*j) & 0x03030303, 0.0f,
                          mmvq_f16_h(b + 80)*1024.0f*(float) (sc & 0xF), -mmvq_f16_h(b + 82)*1024.0f*(float) (sc >> 4), w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q3_K> {
    static constexpr int QK = 256, BS = 110;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int n = sub >> 4, j = (sub >> 2) & 3, l0 = 8*(sub & 3);
        const uint8_t * qs = b + 32 + 32*n + l0, * hm = b + l0;
        const int hs = 4*n + j;
        const uint32_t v0 = ((mmvq_f16_ld2(qs)     >> 2*j) & 0x03030303) | (((mmvq_f16_ld2(hm)     >> hs) & 0x01010101) << 2);
        const uint32_t v1 = ((mmvq_f16_ld2(qs + 4) >> 2*j) & 0x03030303) | (((mmvq_f16_ld2(hm + 4) >> hs) & 0x01010101) << 2);
        const int is = 8*n + 2*j + (l0 >= 16);
        const uint8_t * sc = b + 96;
        const int s6 = (is < 8 ? sc[is] & 0xF : sc[is - 8] >> 4) | (((sc[8 + is % 4] >> 2*(is/4)) & 3) << 4);
        mmvq_f16_w8<false>(v0, v1, 4.0f, mmvq_f16_h(b + 108)*1024.0f*(float) (s6 - 32), 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q4_K> {
    static constexpr int QK = 256, BS = 144;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int j = sub >> 3, t = sub & 7, is = 2*j + (t >> 2), sh = 4*(t >> 2);
        const uint8_t * qs = b + 16 + 32*j + 8*(t & 3);
        int sc, m;
        mmvq_f16_sm_k4(is, b + 4, sc, m);
        mmvq_f16_w8<true>((mmvq_f16_ld4(qs) >> sh) & 0x0F0F0F0F, (mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F, 0.0f,
                          mmvq_f16_h(b)*1024.0f*(float) sc, -mmvq_f16_h(b + 2)*1024.0f*(float) m, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_Q5_K> {
    static constexpr int QK = 256, BS = 176;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int j = sub >> 3, t = sub & 7, is = 2*j + (t >> 2), sh = 4*(t >> 2);
        const uint8_t * qs = b + 48 + 32*j + 8*(t & 3), * qh = b + 16 + 8*(t & 3);
        int sc, m;
        mmvq_f16_sm_k4(is, b + 4, sc, m);
        const uint32_t v0 = ((mmvq_f16_ld4(qs)     >> sh) & 0x0F0F0F0F) | (((mmvq_f16_ld4(qh)     >> is) & 0x01010101) << 4);
        const uint32_t v1 = ((mmvq_f16_ld4(qs + 4) >> sh) & 0x0F0F0F0F) | (((mmvq_f16_ld4(qh + 4) >> is) & 0x01010101) << 4);
        mmvq_f16_w8<true>(v0, v1, 0.0f, mmvq_f16_h(b)*1024.0f*(float) sc, -mmvq_f16_h(b + 2)*1024.0f*(float) m, w);
    }
};

// iq3_xxs / iq3_s / iq1_s / iq1_m (agent iq31). Grid values are positive bytes (iq3) or nibbles 0..2 (iq1).
// iq3 signs: the 4 sign bits of a grid word -> byte masks 0xFF; w = (g ^ m) + (m & 1 per byte) is exactly -g.
static __device__ __forceinline__ void mmvq_f16_sgn4(const uint32_t bits, uint32_t & m, uint32_t & one) {
    const uint32_t x = ((bits & 0xF) * 0x01010101u) & 0x08040201u;
    one = ((x + 0x7F7F7F7Fu) >> 7) & 0x01010101u;
    m   = (one << 8) - one;
}
static __device__ __forceinline__ uint32_t mmvq_f16_sgnw(const uint32_t g, const uint32_t bits) {
    uint32_t m, one;
    mmvq_f16_sgn4(bits, m, one);
    return ((g ^ m) + one) ^ 0x80808080u;   // signed byte + 128
}
template <> struct mmvq_f16_t<GGML_TYPE_IQ3_XXS> {
    static constexpr int QK = 256, BS = 98;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int ib = sub >> 2, l = sub & 3;
        const uint8_t * qs = b + 2 + 8*ib + 2*l;
        const uint32_t aux = mmvq_f16_ld2(b + 66 + 4*ib);
        const uint32_t sg = ksigns_iq2xs[(aux >> 7*l) & 127];
        mmvq_f16_w8<false>(mmvq_f16_sgnw(iq3xxs_grid[qs[0]], sg), mmvq_f16_sgnw(iq3xxs_grid[qs[1]], sg >> 4), 128.0f,
                           mmvq_f16_h(b)*1024.0f*0.25f*(float) (2*(aux >> 28) + 1), 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_IQ3_S> {
    static constexpr int QK = 256, BS = 110;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int ib = sub >> 2, l = sub & 3;
        const uint8_t * qs = b + 2 + 8*ib + 2*l;
        const uint32_t qh = b[66 + ib], sg = b[74 + 4*ib + l];
        const int ls = (b[106 + (ib >> 1)] >> 4*(ib & 1)) & 0xF;
        mmvq_f16_w8<false>(mmvq_f16_sgnw(iq3s_grid[qs[0] | ((qh << (8 - 2*l)) & 0x100)], sg),
                           mmvq_f16_sgnw(iq3s_grid[qs[1] | ((qh << (7 - 2*l)) & 0x100)], sg >> 4), 128.0f,
                           mmvq_f16_h(b)*1024.0f*(float) (2*ls + 1), 0.0f, w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_IQ1_S> {
    static constexpr int QK = 256, BS = 50;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int ib = sub >> 2, l = sub & 3;
        const uint32_t qh = *(const uint16_t *) (b + 34 + 2*ib);
        const uint32_t grid = iq1s_grid_gpu[b[2 + 4*ib + l] | (((qh >> 3*l) & 7) << 8)];
        const float s = mmvq_f16_h(b)*1024.0f*(float) (2*((qh >> 12) & 7) + 1);
        mmvq_f16_w8<true>(grid & 0x0F0F0F0F, (grid >> 4) & 0x0F0F0F0F, 0.0f, s, s*((qh & 0x8000) ? -1.125f : -0.875f), w);
    }
};
template <> struct mmvq_f16_t<GGML_TYPE_IQ1_M> {
    static constexpr int QK = 256, BS = 56;
    static __device__ __forceinline__ void w8(const uint8_t * b, const int sub, __half2 (&w)[4]) {
        const int ib = sub >> 2, l = sub & 3;
        const uint16_t * sc = (const uint16_t *) (b + 48);
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const uint32_t qhl = b[32 + 2*ib + (l >> 1)] >> 4*(l & 1);
        const uint32_t grid = iq1s_grid_gpu[b[4*ib + l] | ((qhl & 7) << 8)];
        const float s = __half2float(scale.f16)*1024.0f*(float) (2*((sc[ib >> 1] >> (6*(ib & 1) + 3*(l >> 1))) & 7) + 1);
        mmvq_f16_w8<true>(grid & 0x0F0F0F0F, (grid >> 4) & 0x0F0F0F0F, 0.0f, s, s*((qhl & 8) ? -1.125f : -0.875f), w);
    }
};

// RPW row slots per warp, NWT warps per block. GLU: slots [0, RPW/2) walk W (gate), the rest the
// same rows of W2 (up), and the epilogue writes silu(gate) * up as the SWIGLU kernel does.
// Each warp stages its rows' window bytes in shared memory with coalesced 16-byte loads (as the q6_K
// kernel does); the copy keeps each byte's address mod 16, so the types' 2- and 4-byte fields stay
// aligned. Then the unpacks read from shared memory.
template <ggml_type T, int NC, int RPW, int NWT, bool GLU>
__launch_bounds__(NWT*WARP_SIZE)
static __global__ void mmvq_f16_gen(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                    const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                    const int64_t sy, const int rows, const int K) {
    using tr = mmvq_f16_t<T>;
    constexpr int ORW = GLU ? RPW/2 : RPW, SUBS = tr::QK/8;
    constexpr int WB = (MMVQ_F16_WIN/tr::QK)*tr::BS, NU = (WB + 15 + 15)/16;
    __shared__ uint4 wst[NWT][RPW][NU];
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int row0 = (blockIdx.x*NWT + wid)*ORW;
    if (row0 >= rows) {
        return;
    }
    const int nchunk = K/8, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    for (int win = 0; win < nw; ++win) {
        const int wbytes = min((int64_t) WB, row_bytes - (int64_t) win*WB);
        const uint8_t * wb[RPW];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint8_t * g = rp[i] + (int64_t) win*WB;
            const int m = (int) ((uintptr_t) g & 15);
            wb[i] = (const uint8_t *) wst[wid][i] + m;
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + wbytes + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int kw = jj*32 + lane, k = win*128 + kw;
            if (k < nchunk) {
                const int blk = kw / SUBS, sub = kw % SUBS;
                __half2 x[NC][4];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const uint4 u = __ldg((const uint4 *) (XS + (int64_t) c*K + 8*k));
                    x[c][0] = *(const __half2 *) &u.x; x[c][1] = *(const __half2 *) &u.y;
                    x[c][2] = *(const __half2 *) &u.z; x[c][3] = *(const __half2 *) &u.w;
                }
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    __half2 w[4];
                    tr::w8(wb[i] + blk*tr::BS, sub, w);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 u = __hfma2(w[0], x[c][0], t[c][i]);
                        u = __hfma2(w[1], x[c][1], u);
                        u = __hfma2(w[2], x[c][2], u);
                        t[c][i] = __hfma2(w[3], x[c][3], u);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// q4_K, hand-scheduled (the generic kernel above re-derives the scales for every 8-weight chunk).
//
// A q4_K block is 144 bytes: [d, dmin | 12 bytes of 6-bit scales and mins | 128 bytes of nibbles],
// 8 sub-blocks of 32 values; byte l of qs group j (32 bytes) holds element 64j+l in its low nibble
// (sub-block 2j) and element 64j+32+l in its high nibble (sub-block 2j+1). Both the 16-byte header
// and every 16-byte slice of qs are 16-byte aligned, so lanes read them straight from global memory.
// Lane l of a warp owns, in each window of 4 blocks, block l/8 and the 16 bytes u = l%8 of its qs: 16
// low-nibble values (sub-block 2j) and the same 16 byte positions' high nibbles (sub-block 2j+1),
// j = u/2. So each lane decodes 2 scales and 2 mins per row and window, once, and runs a chain of
// 16 HFMA2 per column (the numerics of the generic path: w = (q - bias)*half(d*1024*sc) + half(-dmin*1024*m)).
// High nibbles are not shifted down: masked in place they unpack to half(1024 + 16q), and the 1/16 is
// folded into the scale exactly.
template <int NC, int RPW, int NWT, bool KS, int MINB, bool GLU>
__launch_bounds__(NWT*WARP_SIZE, MINB)
static __global__ void mmvq_f16_q4_K_k(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                       const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                       const int64_t sy, const int rows, const int K) {
    static_assert(!GLU || (!KS && RPW % 2 == 0), "GLU: row-parallel, gate/up row pairs");
    constexpr int ORW = GLU ? RPW/2 : RPW, RPB = NWT*RPW;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + 3)/4;
    // KS (split K): the block's warps share its rows and take every NWT-th window each
    const int row0 = KS ? blockIdx.x*ORW : GLU ? (blockIdx.x*NWT + wid)*ORW : blockIdx.x*RPB + wid*RPW;
    if (!KS && row0 >= rows) {
        return;
    }
    const int B = lane >> 3, u = lane & 7, j = u >> 1;
    const bool hi = j >= 2;
    const uint32_t sel = (j & 1) ? 0x4432u : 0x4410u;
    const uint32_t mA = hi ? 0x0F0F0F0Fu : 0x3F3F3F3Fu, hmk = hi ? 0x30303030u : 0u;
    const int xoff = 64*j + 16*(u & 1), qoff = 16 + 16*u;
    const __half2 k1024 = __float2half2_rn(1024.0f);

    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    for (int win = KS ? wid : 0; win < nw; win += KS ? NWT : 1) {
        const bool valid = 4*win + B < nb;
        const int bb = min(4*win + B, nb - 1);
        uint4 H[RPW], Q[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint8_t * b = rp[i] + (int64_t) bb*144;
            H[i] = __ldg((const uint4 *) b);
            Q[i] = __ldg((const uint4 *) (b + qoff));
        }
        __half2 sc[2][RPW], mn[2][RPW]; // [group][row]
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint32_t a = __byte_perm(H[i].y, 0, sel), M = __byte_perm(H[i].z, 0, sel), C = __byte_perm(H[i].w, 0, sel);
            const uint32_t sc16 = ((hi ? C : a) & mA) | ((a >> 2) & hmk);
            const uint32_t m16  = ((hi ? C >> 4 : M) & mA) | ((M >> 2) & hmk);
            const uint32_t dd = __byte_perm(H[i].x, 0, 0x1010), dm = __byte_perm(H[i].x, 0, 0x3232);
            const uint32_t hs = __byte_perm(sc16, 0x64646464u, 0x5140), hm = __byte_perm(m16, 0x64646464u, 0x5140);
            const __half2 d2 = __hmul2(*(const __half2 *) &dd, k1024), dm2 = __hneg2(__hmul2(*(const __half2 *) &dm, k1024));
            const __half2 s2 = __hmul2(__hsub2(*(const __half2 *) &hs, k1024), d2);
            const __half2 m2 = __hmul2(__hsub2(*(const __half2 *) &hm, k1024), dm2);
            sc[0][i] = __low2half2(s2);
            sc[1][i] = __hmul2(__high2half2(s2), __float2half2_rn(0.0625f));
            mn[0][i] = __low2half2(m2);
            mn[1][i] = __high2half2(m2);
        }
        __half2 t[NC][RPW];
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            __half2 x[NC][8];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#ifdef Q4_NOX
                const uint4 xa = make_uint4(lane*c+1, lane^c, g+lane, c), xb = make_uint4(lane+3, lane*g, lane^5, c+g);
#elif defined(Q4_XT)
                const uint4 * xp = (const uint4 *) (XS + (int64_t) c*K + bb*256 + 8*u + 128*g);
                const uint4 xa = __ldg(xp), xb = __ldg(xp + 8);
#else
                const uint4 * xp = (const uint4 *) (XS + (int64_t) c*K + bb*256 + xoff + 32*g);
                const uint4 xa = __ldg(xp), xb = __ldg(xp + 1);
#endif
                x[c][0] = *(const __half2 *) &xa.x; x[c][1] = *(const __half2 *) &xa.y;
                x[c][2] = *(const __half2 *) &xa.z; x[c][3] = *(const __half2 *) &xa.w;
                x[c][4] = *(const __half2 *) &xb.x; x[c][5] = *(const __half2 *) &xb.y;
                x[c][6] = *(const __half2 *) &xb.z; x[c][7] = *(const __half2 *) &xb.w;
            }
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const uint32_t qm = g == 0 ? 0x0F0F0F0Fu : 0xF0F0F0F0u;
                const uint32_t qv[4] = { Q[i].x & qm, Q[i].y & qm, Q[i].z & qm, Q[i].w & qm };
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t h0 = __byte_perm(qv[k], 0x64646464u, 0x5140), h1 = __byte_perm(qv[k], 0x64646464u, 0x5342);
                    const __half2 w0 = __hfma2(__hsub2(*(const __half2 *) &h0, k1024), sc[g][i], mn[g][i]);
                    const __half2 w1 = __hfma2(__hsub2(*(const __half2 *) &h1, k1024), sc[g][i], mn[g][i]);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 v = (g == 0 && k == 0) ? __hmul2(w0, x[c][0]) : __hfma2(w0, x[c][2*k], t[c][i]);
                        t[c][i] = __hfma2(w1, x[c][2*k + 1], v);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = valid ? __ldg(S + c*nw + win) : 0.0f;
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 v = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(v), acc[c][i]);
            }
        }
    }
    if constexpr (KS) {
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

template <typename F>
static void mmvq_f16_q4_K_nc(const int64_t nc, F && f) {
    switch (nc) {
        case 1:  f(std::integral_constant<int, 1>{});  break;
        case 2:  f(std::integral_constant<int, 2>{});  break;
        case 3:  f(std::integral_constant<int, 3>{});  break;
        case 4:  f(std::integral_constant<int, 4>{});  break;
        case 5:  f(std::integral_constant<int, 5>{});  break;
        case 6:  f(std::integral_constant<int, 6>{});  break;
        case 7:  f(std::integral_constant<int, 7>{});  break;
        case 8:  f(std::integral_constant<int, 8>{});  break;
        case 9:  f(std::integral_constant<int, 9>{});  break;
        case 10: f(std::integral_constant<int, 10>{}); break;
        case 11: f(std::integral_constant<int, 11>{}); break;
        case 12: f(std::integral_constant<int, 12>{}); break;
        default: GGML_ABORT("mmvq_f16_q4_K: %d columns", (int) nc);
    }
}

static void mmvq_f16_q4_K_launch(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                                 float * Y, const int64_t sy, const int64_t rows, const int64_t K, const int64_t ncols,
                                 const bool glu, cudaStream_t stream) {
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t nc) {
        mmvq_f16_q4_K_nc(nc, [&](auto ncc) {
            constexpr int NC = decltype(ncc)::value;
#ifdef Q4_SW
            constexpr int RPW = Q4_RPW, NWT = Q4_NWT, MINB = Q4_MINB;
            constexpr bool KS = Q4_KS;
#else
            constexpr int RPW = 4, NWT = 2, MINB = 6;
            constexpr bool KS = false;
#endif
            if (glu) {
                constexpr int RG = RPW < 2 ? 2 : RPW;
                const int g = (int) ((rows + NWT*(RG/2) - 1)/(NWT*(RG/2)));
                mmvq_f16_q4_K_k<NC, RG, NWT, false, MINB, true><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            } else {
                const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
                mmvq_f16_q4_K_k<NC, RPW, NWT, KS, MINB, false><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, nullptr, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            }
        });
    });
    CUDA_CHECK(cudaGetLastError());
}

#ifndef K23_ABL
#define K23_ABL 0
#endif
// ===================================================================================================
// q2_K and q3_K: dedicated kernel (k23)
// Lane l of a window (4 blocks = 1024 values) owns, in block B = l/8, half n = (l/4)&1 and quarter
// q4 = l&3, the 8 bytes qs[32n + 8q4 ..] and so, for the 4 two-bit planes j = 0..3, 4 chunks of 8
// weights: values 128n + 32j + 8q4 + [0, 8) of the block. Its scales are decoded once per window and
// row, as half2 pairs, not once per chunk. Weights: w = q*s [+ mn] in fp16 as in mmvq_f16_w8 (the same
// roundings); each lane's chain is 16 HFMA2 per window and column, folded into fp32 with the window's
// activation scale as before.
static __device__ __forceinline__ __half2 k23_h2(const uint32_t v) { return *(const __half2 *) &v; }
static __device__ __forceinline__ uint32_t k23_u(const __half2 v) { return *(const uint32_t *) &v; }
static __device__ __forceinline__ __half2 k23_bclo(const __half2 v) { return k23_h2(__byte_perm(k23_u(v), 0, 0x1010)); }
static __device__ __forceinline__ __half2 k23_bchi(const __half2 v) { return k23_h2(__byte_perm(k23_u(v), 0, 0x3232)); }

template <ggml_type T> struct mmvq_f16_k23t;

// q2_K, 84-byte blocks: every block 4-byte aligned
template <> struct mmvq_f16_k23t<GGML_TYPE_Q2_K> {
    static constexpr int BS = 84, NR = 5; // NR: registers per row (words)
    struct row_t { uint32_t qa, qb; __half2 s[4], m[4]; };
    template <typename LD>
    static __device__ __forceinline__ void load(row_t & r, const uint8_t * pb, const int n, const int q4, LD && ld) {
        const int h = q4 >> 1;
        const uint32_t sel = 0x4040u | ((uint32_t) (2 + h) << 8) | (uint32_t) h;
        r.qa = ld(pb + 16 + 32*n + 8*q4); r.qb = ld(pb + 20 + 32*n + 8*q4);
        const uint32_t sw0 = ld(pb + 8*n), sw1 = ld(pb + 8*n + 4), dd = ld(pb + 80);
        const __half2 k = __float2half2_rn(1024.0f);
        const __half2 d2 = __hmul2(k23_bclo(k23_h2(dd)), k), m2 = __hmul2(k23_bchi(k23_h2(dd)), __float2half2_rn(-1024.0f));
#if K23_ABL == 1
        for (int p = 0; p < 2; ++p) { r.s[2*p] = r.s[2*p+1] = d2; r.m[2*p] = r.m[2*p+1] = m2; (void)sw0; (void)sw1; (void)sel; }
        if (0)
#endif
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint32_t t = __byte_perm(p ? sw1 : sw0, 0, sel);
            const __half2 lo = __hmul2(__hsub2(k23_h2((t & 0x000F000Fu) | 0x64006400u), k), d2);
            const __half2 hi = __hmul2(__hsub2(k23_h2(((t >> 4) & 0x000F000Fu) | 0x64006400u), k), m2);
            r.s[2*p] = k23_bclo(lo); r.s[2*p + 1] = k23_bchi(lo);
            r.m[2*p] = k23_bclo(hi); r.m[2*p + 1] = k23_bchi(hi);
        }
    }
    static __device__ __forceinline__ void w8(const row_t & r, const int j, __half2 (&w)[4]) {
        const __half2 k = __float2half2_rn(1024.0f);
        const uint32_t v0 = (r.qa >> 2*j) & 0x03030303u, v1 = (r.qb >> 2*j) & 0x03030303u;
        const uint32_t h[4] = { __byte_perm(v0, 0x64646464u, 0x5140), __byte_perm(v0, 0x64646464u, 0x5342),
                                __byte_perm(v1, 0x64646464u, 0x5140), __byte_perm(v1, 0x64646464u, 0x5342) };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#if K23_ABL == 2
            w[i] = __hfma2(k23_h2(h[i]), r.s[j], r.m[j]);
#else
            w[i] = __hfma2(__hsub2(k23_h2(h[i]), k), r.s[j], r.m[j]);
#endif
        }
    }
};

// q3_K, 110-byte blocks: a block starts 4-byte aligned or 2 bytes off; fields are read as aligned
// words and funnel-shifted by the block's phase.
template <> struct mmvq_f16_k23t<GGML_TYPE_Q3_K> {
    static constexpr int BS = 110;
    struct row_t { uint32_t qa, qb, ha, hb; __half2 s[4]; };
    template <typename LD>
    static __device__ __forceinline__ void load(row_t & r, const uint8_t * pb, const int n, const int q4, LD && ld) {
        const int h = q4 >> 1;
        const uint32_t sel = 0x4040u | ((uint32_t) (2 + h) << 8) | (uint32_t) h;
        const uint32_t ph = ((uint32_t) (uintptr_t) pb & 2u)*8u;
        const uint8_t * a = (const uint8_t *) ((uintptr_t) pb & ~(uintptr_t) 3); // aligned block base - 0/2
        auto rd = [&](const int off) { return ld(a + off); };
        // 3 words qs/hm each need 3 raw words
        uint32_t q[3], m[3], sc[4];
#pragma unroll
        for (int i = 0; i < 3; ++i) { q[i] = rd(32 + 32*n + 8*q4 + 4*i); m[i] = rd(8*q4 + 4*i); }
#pragma unroll
        for (int i = 0; i < 4; ++i) { sc[i] = rd(96 + 4*i); }
        const uint32_t dw = rd(108);
        r.qa = __funnelshift_r(q[0], q[1], ph); r.qb = __funnelshift_r(q[1], q[2], ph);
        const uint32_t ma = __funnelshift_r(m[0], m[1], ph), mb = __funnelshift_r(m[1], m[2], ph);
        r.ha = (ma >> 4*n) << 2; r.hb = (mb >> 4*n) << 2;
        const uint32_t s0 = __funnelshift_r(sc[0], sc[1], ph), s1 = __funnelshift_r(sc[1], sc[2], ph), s2 = __funnelshift_r(sc[2], sc[3], ph);
        const __half2 k = __float2half2_rn(1024.0f);
        const __half2 d2 = __hmul2(k23_bclo(k23_h2((dw >> ph) & 0xFFFFu)), k);
        const __half2 k1056 = __float2half2_rn(1056.0f);
        const uint32_t t2 = __byte_perm(s2, 0, sel);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const uint32_t t = __byte_perm(p ? s1 : s0, 0, sel);
            const uint32_t v = ((t >> 4*n) & 0x000F000Fu) | (((t2 >> (4*n + 2*p)) & 0x00030003u) << 4) | 0x64006400u;
            const __half2 sv = __hmul2(__hsub2(k23_h2(v), k1056), d2);
            r.s[2*p] = k23_bclo(sv); r.s[2*p + 1] = k23_bchi(sv);
        }
    }
    static __device__ __forceinline__ void w8(const row_t & r, const int j, __half2 (&w)[4]) {
        const __half2 kb = __float2half2_rn(1028.0f);
        const uint32_t v0 = ((r.qa >> 2*j) & 0x03030303u) | ((r.ha >> j) & 0x04040404u);
        const uint32_t v1 = ((r.qb >> 2*j) & 0x03030303u) | ((r.hb >> j) & 0x04040404u);
        const uint32_t h[4] = { __byte_perm(v0, 0x64646464u, 0x5140), __byte_perm(v0, 0x64646464u, 0x5342),
                                __byte_perm(v1, 0x64646464u, 0x5140), __byte_perm(v1, 0x64646464u, 0x5342) };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            w[i] = __hmul2(__hsub2(k23_h2(h[i]), kb), r.s[j]);
        }
    }
};

template <ggml_type T, int NC, int RPW, int NWT, bool GLU, int STG, int MINB>
__launch_bounds__(NWT*WARP_SIZE, MINB)
static __global__ void mmvq_f16_k23(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                    const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                    const int64_t sy, const int rows, const int K) {
    using tr = mmvq_f16_k23t<T>;
    constexpr int ORW = GLU ? RPW/2 : RPW, BS = tr::BS;
    constexpr int WB = 4*BS, NU = (WB + 15 + 15)/16;
    __shared__ uint4 wst[STG ? NWT : 1][STG ? RPW : 1][STG ? NU : 1];
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int row0 = (blockIdx.x*NWT + wid)*ORW;
    if (row0 >= rows) {
        return;
    }
    const int nb = K/256, nw = (nb + 3)/4;
    const int B = lane >> 3, n = (lane >> 2) & 1, q4 = lane & 3;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const __half * xl = XS + (int64_t) B*256 + 128*n + 8*q4;
    for (int win = 0; win < nw; ++win) {
        const int nblk = min(4, nb - 4*win);
        const uint8_t * wb[RPW];
        if constexpr (STG) {
            __syncwarp();
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const uint8_t * g = rp[i] + (int64_t) win*WB;
                const int m = (int) ((uintptr_t) g & 15);
                wb[i] = (const uint8_t *) wst[wid][i] + m;
                const uint4 * g16 = (const uint4 *) (g - m);
                const int nu = (m + nblk*BS + 15)/16;
#pragma unroll
                for (int r = 0; r < (NU + 31)/32; ++r) {
                    const int k = r*32 + lane;
                    if (k < nu) {
                        wst[wid][i][k] = __ldg(g16 + k);
                    }
                }
            }
            __syncwarp();
        } else {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                wb[i] = rp[i] + (int64_t) win*WB;
            }
        }
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
        if (B < nblk) {
            typename tr::row_t r[RPW];
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                if constexpr (STG) {
                    tr::load(r[i], wb[i] + B*BS, n, q4, [](const uint8_t * p) { return *(const uint32_t *) p; });
                } else {
                    tr::load(r[i], wb[i] + B*BS, n, q4, [](const uint8_t * p) { return __ldg((const uint32_t *) p); });
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                __half2 x[NC][4];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
#if K23_ABL == 3
                    const uint4 u = make_uint4(c, j, win, 1);
#else
                    const uint4 u = __ldg((const uint4 *) (xl + (int64_t) c*K + win*MMVQ_F16_WIN + 32*j));
#endif
                    x[c][0] = *(const __half2 *) &u.x; x[c][1] = *(const __half2 *) &u.y;
                    x[c][2] = *(const __half2 *) &u.z; x[c][3] = *(const __half2 *) &u.w;
                }
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    __half2 w[4];
                    tr::w8(r[i], j, w);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 u = __hfma2(w[0], x[c][0], t[c][i]);
                        u = __hfma2(w[1], x[c][1], u);
                        u = __hfma2(w[2], x[c][2], u);
                        t[c][i] = __hfma2(w[3], x[c][3], u);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

// runtime config for sweeps (the final file hard-codes the winners)
static int g_k23_var = 0;
template <ggml_type T, int NC, int RPW, int NWT, int STG, int MINB>
static void mmvq_f16_k23_go(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                            float * Y, const int64_t sy, const int64_t rows, const int64_t K, const bool glu, cudaStream_t stream) {
    if (glu) {
        if constexpr (RPW % 2 == 0) {
            const int g = (int) ((rows + NWT*RPW/2 - 1)/(NWT*RPW/2));
            mmvq_f16_k23<T, NC, RPW, NWT, true, STG, MINB><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(W, W2, row_bytes, xs, sc, Y, sy, (int) rows, (int) K);
        }
    } else {
        const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
        mmvq_f16_k23<T, NC, RPW, NWT, false, STG, MINB><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(W, nullptr, row_bytes, xs, sc, Y, sy, (int) rows, (int) K);
    }
}

template <ggml_type T, int NC>
static void mmvq_f16_k23_var(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                             float * Y, const int64_t sy, const int64_t rows, const int64_t K, const bool glu, cudaStream_t stream) {
#define K23V(id, RPW, NWT, STG, MINB) case id: mmvq_f16_k23_go<T, NC, RPW, NWT, STG, MINB>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, glu, stream); break;
    switch (g_k23_var) {
        K23V(0, 4, 2, 1, 6)
        K23V(1, 4, 2, 0, 6)
        K23V(2, 2, 2, 1, 8)
        K23V(3, 2, 2, 0, 8)
        K23V(4, 4, 1, 0, 8)
        K23V(5, 2, 1, 0, 12)
        K23V(6, 4, 1, 1, 8)
        K23V(7, 8, 1, 0, 4)
        K23V(8, 1, 2, 0, 12)
        default: GGML_ABORT("k23 var");
    }
#undef K23V
}
template <ggml_type T>
static void mmvq_f16_k23_launch(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                                float * Y, const int64_t sy, const int64_t rows, const int64_t K, const int64_t ncols, const bool glu, cudaStream_t stream) {
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t nc) {
        const uint8_t * W_ = W; (void) W_;
        auto f = [&](auto ncc) {
            constexpr int NC = decltype(ncc)::value;
            mmvq_f16_k23_var<T, NC>(W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, rows, K, glu, stream);
        };
        if (nc == 1) { f(std::integral_constant<int, 1>{}); } else { mmvq_f16_nc(nc, f); }
    });
}

// ---------------------------------------------------------------------------------------------------
// q4_0 and q8_0 (32-value blocks of 18 and 34 bytes), multi-column, hand-scheduled.
// Two lanes per block: lane h takes 16 values of the block (q8_0: bytes 16h..16h+15; q4_0: bytes 8h..8h+7,
// both nibbles, i.e. values 8h..8h+7 and 16+8h..16+8h+7), so a window of 32 blocks is two passes of 16
// blocks and each lane runs one chain of 16 HFMA2 per window and column, as on q6_K. Row windows are staged
// in shared memory with coalesced 16-byte loads (rows are 16-byte aligned: K % 256 == 0); the block's
// 2-byte-misaligned qs words are realigned with one funnel shift each. Numerics per weight are the
// generic kernel's: (half(1024 + q) - (1024 + bias)) * half(d * 1024), chains of one window folded in fp32.
template <bool Q8, int NC, int RPW, int NWT, bool GLU, int MINB>
__launch_bounds__(NWT*WARP_SIZE, MINB)
static __global__ void mmvq_f16_32(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                   const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                   const int64_t sy, const int rows, const int K) {
    constexpr int ORW = GLU ? RPW/2 : RPW, BS = Q8 ? 34 : 18, WB = 32*BS, NU = WB/16, NUP = NU + 1;
    __shared__ uint4 wst[NWT][RPW][NUP];
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int row0 = (blockIdx.x*NWT + wid)*ORW;
    if (row0 >= rows) {
        return;
    }
    const int nb = K/32, nw = (nb + 31)/32;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const int h = lane & 1, b0 = lane >> 1;
    const __half * xl[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xl[c] = XS + (int64_t) c*K + b0*32 + (Q8 ? 16*h : 8*h);
    }
    const __half2 kb = __float2half2_rn(Q8 ? 1152.0f : 1032.0f), k1024 = __float2half2_rn(1024.0f);
    for (int win = 0; win < nw; ++win) {
        const int nblk = min(32, nb - win*32), nu = nblk*BS/16;
        __syncwarp();
        {
            uint4 ld[RPW][(NU + 31)/32];
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const uint4 * g16 = (const uint4 *) (rp[i] + (int64_t) win*WB);
#pragma unroll
                for (int r = 0; r < (NU + 31)/32; ++r) {
                    const int k = r*32 + lane;
                    if (k < nu) {
                        ld[i][r] = __ldg(g16 + k);
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
#pragma unroll
                for (int r = 0; r < (NU + 31)/32; ++r) {
                    const int k = r*32 + lane;
                    if (k < nu) {
                        wst[wid][i][k] = ld[i][r];
                    }
                }
            }
        }
        __syncwarp();
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const int blk = jj*16 + b0;
            if (blk < nblk) {
                __half2 x[NC][8];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const uint4 u0 = __ldg((const uint4 *) (xl[c] + win*1024 + jj*512));
                    const uint4 u1 = __ldg((const uint4 *) (xl[c] + win*1024 + jj*512 + 8 + (Q8 ? 0 : 8)));
                    x[c][0] = *(const __half2 *) &u0.x; x[c][1] = *(const __half2 *) &u0.y;
                    x[c][2] = *(const __half2 *) &u0.z; x[c][3] = *(const __half2 *) &u0.w;
                    x[c][4] = *(const __half2 *) &u1.x; x[c][5] = *(const __half2 *) &u1.y;
                    x[c][6] = *(const __half2 *) &u1.z; x[c][7] = *(const __half2 *) &u1.w;
                }
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    const uint32_t * sw = (const uint32_t *) wst[wid][i];
                    const int qoff = BS*blk + 2 + (Q8 ? 16*h : 8*h), wi = qoff >> 2, sh = (qoff & 2) << 3;
                    const uint32_t d16 = *(const uint16_t *) ((const uint8_t *) sw + BS*blk);
                    const __half2 s2 = __hmul2(__half2half2(__ushort_as_half((unsigned short) d16)), k1024);
                    uint32_t v[4];
                    if constexpr (Q8) {
                        const uint32_t w0 = sw[wi], w1 = sw[wi + 1], w2 = sw[wi + 2], w3 = sw[wi + 3], w4 = sw[wi + 4];
                        v[0] = __funnelshift_r(w0, w1, sh) ^ 0x80808080u; v[1] = __funnelshift_r(w1, w2, sh) ^ 0x80808080u;
                        v[2] = __funnelshift_r(w2, w3, sh) ^ 0x80808080u; v[3] = __funnelshift_r(w3, w4, sh) ^ 0x80808080u;
                    } else {
                        const uint32_t w0 = sw[wi], w1 = sw[wi + 1], w2 = sw[wi + 2];
                        const uint32_t a0 = __funnelshift_r(w0, w1, sh), a1 = __funnelshift_r(w1, w2, sh);
                        v[0] = a0 & 0x0F0F0F0Fu; v[1] = a1 & 0x0F0F0F0Fu;
                        v[2] = (a0 >> 4) & 0x0F0F0F0Fu; v[3] = (a1 >> 4) & 0x0F0F0F0Fu;
                    }
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t p0 = __byte_perm(v[k], 0x64646464u, 0x5140), p1 = __byte_perm(v[k], 0x64646464u, 0x5342);
                        const __half2 w0 = __hmul2(__hsub2(*(const __half2 *) &p0, kb), s2);
                        const __half2 w1 = __hmul2(__hsub2(*(const __half2 *) &p1, kb), s2);
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            t[c][i] = __hfma2(w0, x[c][2*k], t[c][i]);
                            t[c][i] = __hfma2(w1, x[c][2*k + 1], t[c][i]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

#ifndef K32_RPW
#define K32_RPW(Q8, NC) (NC <= 4 ? 4 : 2)
#endif
#ifndef K32_NWT
#define K32_NWT(Q8, NC) 2
#endif
#ifndef K32_MINB
#define K32_MINB(Q8, NC) 4
#endif
#ifndef K32_SPLIT
#define K32_SPLIT(n) ((n) <= 8 ? (n) : ((n) + 1)/2)
#endif
static void mmvq_f16_32_launch(const bool q8, const uint8_t * W, const uint8_t * W2, const int64_t row_bytes,
                               const __half * xs, const float * sc, float * Y, const int64_t sy, const int64_t rows,
                               const int64_t K, const int64_t ncols, const bool glu, const int nw, cudaStream_t stream) {
    const int64_t n0 = K32_SPLIT(ncols);
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        const int64_t n = std::min(n0, ncols - c0);
        auto go = [&](auto q8t, auto ncc) {
            constexpr bool Q8 = decltype(q8t)::value;
            constexpr int NC = decltype(ncc)::value;
            auto run = [&](auto rpwt, auto nwtt) {
                constexpr int RPW = decltype(rpwt)::value, NWT = decltype(nwtt)::value, MINB = 12/NWT;
                const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
                if constexpr (RPW <= 4) { if (glu) {
                    mmvq_f16_32<Q8, NC, 2*RPW, NWT, true, MINB><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                        W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
                    return; } }
                {
                    mmvq_f16_32<Q8, NC, RPW, NWT, false, MINB><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                        W, nullptr, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
                }
            };
            constexpr int cfg = 0;
            int rpw = cfg ? cfg/10 : K32_RPW(Q8, NC); if (glu && rpw > 4) { rpw = 4; }
            const int nwt = cfg ? cfg%10 : K32_NWT(Q8, NC);
            using I1 = std::integral_constant<int, 1>; using I2 = std::integral_constant<int, 2>;
            using I4 = std::integral_constant<int, 4>; using I8 = std::integral_constant<int, 8>;
            if (rpw == 2) { if (nwt == 1) run(I2{}, I1{}); else if (nwt == 2) run(I2{}, I2{}); else run(I2{}, I4{}); }
            else if (rpw == 4) { if (nwt == 1) run(I4{}, I1{}); else if (nwt == 2) run(I4{}, I2{}); else run(I4{}, I4{}); }
            else { if (nwt == 1) run(I8{}, I1{}); else if (nwt == 2) run(I8{}, I2{}); else run(I8{}, I4{}); }
        };
        auto gq = [&](auto ncc) { if (q8) go(std::true_type{}, ncc); else go(std::false_type{}, ncc); };
        switch (n) {
            case 1: gq(std::integral_constant<int, 1>{}); break;
            case 2: gq(std::integral_constant<int, 2>{}); break;
            case 3: gq(std::integral_constant<int, 3>{}); break;
            case 4: gq(std::integral_constant<int, 4>{}); break;
            case 5: gq(std::integral_constant<int, 5>{}); break;
            case 6: gq(std::integral_constant<int, 6>{}); break;
            case 7: gq(std::integral_constant<int, 7>{}); break;
            case 8: gq(std::integral_constant<int, 8>{}); break;
            default: GGML_ABORT("mmvq_f16_32: %d columns", (int) n);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------
// q4_1, q5_0, q5_1 (32-value blocks of 20, 22 and 24 bytes): the same lane-per-chunk schedule as the
// generic kernel, but the weights come straight from global memory (no shared staging, no syncs),
// the block's scale and min are converted once per chunk in half arithmetic (the same exact values as
// the generic unpack), RPW/NWT/min-blocks are chosen per column count, rows < 256 split K over the
// block's warps, and up to 12 columns go through one launch (the weights are read once).
// Numerics per weight are the generic unpack's: half(1024 + q) - bias, times half(d*1024), plus half(m*1024).

static __device__ __forceinline__ uint32_t mmvq_s32_ld4(const uint8_t * p) { return __ldg((const uint32_t *) p); }
static __device__ __forceinline__ uint32_t mmvq_s32_ld2(const uint8_t * p) { // 2-byte aligned
    return (uint32_t) __ldg((const uint16_t *) p) | ((uint32_t) __ldg((const uint16_t *) (p + 2)) << 16);
}
static __device__ __forceinline__ __half2 mmvq_s32_dup(const uint32_t v16) { // half2(h, h) * 1024
    return __hmul2(__half2half2(__ushort_as_half((unsigned short) v16)), __float2half2_rn(1024.0f));
}

template <bool MIN>
static __device__ __forceinline__ void mmvq_s32_w8(const uint32_t v0, const uint32_t v1, const __half2 kb, const __half2 s2, const __half2 mn2,
                                                   __half2 (&w)[4]) {
    const uint32_t h[4] = { __byte_perm(v0, 0x64646464u, 0x5140), __byte_perm(v0, 0x64646464u, 0x5342),
                            __byte_perm(v1, 0x64646464u, 0x5140), __byte_perm(v1, 0x64646464u, 0x5342) };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const __half2 q = __hsub2(*(const __half2 *) &h[k], kb);
        w[k] = MIN ? __hfma2(q, s2, mn2) : __hmul2(q, s2);
    }
}

template <ggml_type T> struct mmvq_s32;
// ld: raw words of this lane's chunk (8 values); w8: unpack them to 8 half weights
template <> struct mmvq_s32<GGML_TYPE_Q4_1> {
    static constexpr int BS = 20;
    static __device__ __forceinline__ void ld(const uint8_t * b, const int sub, uint32_t (&r)[4]) {
        const uint8_t * qs = b + 4 + 8*(sub & 1);
        r[0] = mmvq_s32_ld4(b); r[1] = mmvq_s32_ld4(qs); r[2] = mmvq_s32_ld4(qs + 4);
    }
    static __device__ __forceinline__ void w8(const uint32_t (&r)[4], const int sub, __half2 (&w)[4]) {
        const int sh = 4*(sub >> 1);
        mmvq_s32_w8<true>((r[1] >> sh) & 0x0F0F0F0F, (r[2] >> sh) & 0x0F0F0F0F, __float2half2_rn(1024.0f),
                          mmvq_s32_dup(r[0]), mmvq_s32_dup(r[0] >> 16), w);
    }
};
template <> struct mmvq_s32<GGML_TYPE_Q5_1> {
    static constexpr int BS = 24;
    static __device__ __forceinline__ void ld(const uint8_t * b, const int sub, uint32_t (&r)[4]) {
        const uint8_t * qs = b + 8 + 8*(sub & 1);
        r[0] = mmvq_s32_ld4(b); r[1] = mmvq_s32_ld4(b + 4); r[2] = mmvq_s32_ld4(qs); r[3] = mmvq_s32_ld4(qs + 4);
    }
    static __device__ __forceinline__ void w8(const uint32_t (&r)[4], const int sub, __half2 (&w)[4]) {
        const int sh = 4*(sub >> 1);
        const uint32_t hb = r[1] >> (8*sub);
        mmvq_s32_w8<true>(((r[2] >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb), ((r[3] >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb >> 4),
                          __float2half2_rn(1024.0f), mmvq_s32_dup(r[0]), mmvq_s32_dup(r[0] >> 16), w);
    }
};
template <> struct mmvq_s32<GGML_TYPE_Q5_0> {
    static constexpr int BS = 22;
    static __device__ __forceinline__ void ld(const uint8_t * b, const int sub, uint32_t (&r)[4]) {
        const uint8_t * qs = b + 6 + 8*(sub & 1);
        r[0] = __ldg((const uint16_t *) b); r[1] = mmvq_s32_ld2(b + 2); r[2] = mmvq_s32_ld2(qs); r[3] = mmvq_s32_ld2(qs + 4);
    }
    static __device__ __forceinline__ void w8(const uint32_t (&r)[4], const int sub, __half2 (&w)[4]) {
        const int sh = 4*(sub >> 1);
        const uint32_t hb = r[1] >> (8*sub);
        mmvq_s32_w8<false>(((r[2] >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb), ((r[3] >> sh) & 0x0F0F0F0F) | mmvq_f16_spread4(hb >> 4),
                           __float2half2_rn(1040.0f), mmvq_s32_dup(r[0]), __float2half2_rn(0.0f), w);
    }
};

// RPW rows per warp; NWT warps per block. KS: the block's warps share its RPW rows and take every NWT-th
// window each. GLU: row slots [0, RPW/2) walk W (gate), the rest W2 (up); epilogue silu(gate)*up.
template <ggml_type T, int NC, int RPW, int NWT, bool KS, int MINB, bool GLU, int PF = 0>
__launch_bounds__(NWT*WARP_SIZE, MINB)
static __global__ void mmvq_f16_s32k(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                     const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                     const int64_t sy, const int rows, const int K) {
    static_assert(!GLU || (!KS && RPW % 2 == 0), "GLU: row-parallel, gate/up row pairs");
    using tr = mmvq_s32<T>;
    constexpr int ORW = GLU ? RPW/2 : RPW;
    constexpr int WB = (MMVQ_F16_WIN/32)*tr::BS;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int row0 = KS ? blockIdx.x*RPW : (blockIdx.x*NWT + wid)*ORW;
    if (!KS && row0 >= rows) {
        return;
    }
    const int nchunk = K/8, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN, sub = lane & 3;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes + (lane >> 2)*tr::BS;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    uint32_t rw[PF ? 2 : 1][4][RPW][4];
    auto ldwin = [&](const int win, uint32_t (&r)[4][RPW][4]) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            if (win*128 + jj*32 + lane < nchunk) {
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    tr::ld(rp[i] + (int64_t) win*WB + jj*8*tr::BS, sub, r[jj][i]);
                }
            }
        }
    };
    constexpr int WSTEP = KS ? NWT : 1;
    if (PF && (KS ? wid : 0) < nw) {
        ldwin(KS ? wid : 0, rw[0]);
    }
    int par = 0;
    for (int win = KS ? wid : 0; win < nw; win += WSTEP) {
        if constexpr (PF) {
            if (win + WSTEP < nw) {
                ldwin(win + WSTEP, rw[par ^ 1]);
            }
        } else {
            ldwin(win, rw[0]);
        }
        const uint32_t (&cr)[4][RPW][4] = rw[PF ? par : 0];
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const int k = win*128 + jj*32 + lane;
            if (k < nchunk) {
                __half2 x[NC][4];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
#ifdef H_NOX
                    const uint4 u = make_uint4(c, c+1, k, 3);
#else
                    const uint4 u = __ldg((const uint4 *) (XS + (int64_t) c*K + 8*k));
#endif
                    x[c][0] = *(const __half2 *) &u.x; x[c][1] = *(const __half2 *) &u.y;
                    x[c][2] = *(const __half2 *) &u.z; x[c][3] = *(const __half2 *) &u.w;
                }
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    __half2 w[4];
#ifdef H_NOW
                    w[0] = *(const __half2 *) &cr[jj][i][0]; w[1] = *(const __half2 *) &cr[jj][i][1]; w[2] = *(const __half2 *) &cr[jj][i][2]; w[3] = *(const __half2 *) &cr[jj][i][0];
#else
                    tr::w8(cr[jj][i], sub, w);
#endif
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 u = __hfma2(w[0], x[c][0], t[c][i]);
                        u = __hfma2(w[1], x[c][1], u);
                        u = __hfma2(w[2], x[c][2], u);
                        t[c][i] = __hfma2(w[3], x[c][3], u);
                    }
                }
            }
        }
        par ^= 1;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
    if constexpr (KS) {
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

// geometry per type family / columns / rows (H_* macros override for sweeps)
#ifndef H_PF
#define H_PF 0
#endif
#ifndef H_RPW
#define H_RPW 0
#endif
#ifndef H_NWT
#define H_NWT 0
#endif
#ifndef H_MINB
#define H_MINB 0
#endif
#ifndef H_KS
#define H_KS -1
#endif

template <typename F>
static void mmvq_s32_nc(const int64_t nc, F && f) {
    if (nc == 1) { f(std::integral_constant<int, 1>{}); } else { mmvq_f16_nc(nc, f); }
}

template <ggml_type T>
static void mmvq_f16_s32_launch(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                                float * Y, const int64_t sy, const int64_t rows, const int64_t K, const int64_t ncols, const bool glu,
                                cudaStream_t stream) {
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t n) {
        mmvq_s32_nc(n, [&](auto ncc) {
            constexpr int NC = decltype(ncc)::value;
            auto go = [&](auto rpw, auto nwt, auto ks, auto minb, auto gl) {
                constexpr int RPW = decltype(rpw)::value, NWT = decltype(nwt)::value, MINB = decltype(minb)::value;
                constexpr bool KS = decltype(ks)::value, GL = decltype(gl)::value;
                const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*(GL ? RPW/2 : RPW) - 1)/(NWT*(GL ? RPW/2 : RPW)));
                mmvq_f16_s32k<T, NC, RPW, NWT, KS, MINB, GL, H_PF><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            };
            if (glu) {
                go(std::integral_constant<int, H_RPW ? H_RPW : 4>{}, std::integral_constant<int, H_NWT ? H_NWT : 1>{}, std::false_type{},
                   std::integral_constant<int, H_MINB ? H_MINB : 8>{}, std::true_type{});
            } else {
                go(std::integral_constant<int, H_RPW ? H_RPW : 4>{}, std::integral_constant<int, H_NWT ? H_NWT : 1>{}, std::false_type{},
                   std::integral_constant<int, H_MINB ? H_MINB : 8>{}, std::false_type{});
            }
        });
    });
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// iq2_xxs / iq2_xs / iq2_s (agent iq2): a dedicated kernel. A lane takes one whole 32-value
// sub-block of the window (a window is 4 blocks = 32 sub-blocks = 32 lanes), so the scale and the
// index words load once per 4 grid entries, and the sub-block's 16 HFMA2 per column run on the
// *unscaled* signed grid values (+-8/25/43, exact in fp16, |sum| <= 1376); the scale (d times the
// sub-block scale, 1024 folded in as in mmvq_f16_w8) multiplies the sub-block's sum once. Every
// chain still ends at the window.
template <int N, bool K7>
struct mmvq_f16_sgn_tab {
    alignas(16) uint32_t v[N][4];
    constexpr mmvq_f16_sgn_tab() : v() {
        for (int i = 0; i < N; ++i) {
            unsigned s = i;
            if (K7) { // 7-bit ksigns index; the 8th sign is the parity of the other seven
                unsigned p = 0;
                for (int b = 0; b < 7; ++b) { p ^= (s >> b) & 1; }
                s |= p << 7;
            }
            for (int k = 0; k < 4; ++k) {
                v[i][k] = (((s >> (2*k)) & 1) ? 0x8000u : 0u) | (((s >> (2*k + 1)) & 1) ? 0x80000000u : 0u);
            }
        }
    }
};
static __device__ const mmvq_f16_sgn_tab<128, true>  mmvq_f16_sgn7;
static __device__ const mmvq_f16_sgn_tab<256, false> mmvq_f16_sgn8;

static constexpr __host__ __device__ bool mmvq_f16_is_iq2(const ggml_type t) {
    return t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ2_S;
}
template <ggml_type T> struct mmvq_f16_iq2_t;
template <> struct mmvq_f16_iq2_t<GGML_TYPE_IQ2_XXS> { static constexpr int BS = 66; };
template <> struct mmvq_f16_iq2_t<GGML_TYPE_IQ2_XS>  { static constexpr int BS = 74; };
template <> struct mmvq_f16_iq2_t<GGML_TYPE_IQ2_S>   { static constexpr int BS = 82; };

// the sub-block's 4 grid entries (grid word pairs and sign masks) and its two scale factors
// (0.5 + ls)/4 for chunks 0-1 and 2-3 (iq2_xxs has one scale for all four)
template <ggml_type T>
static __device__ __forceinline__ void mmvq_f16_iq2_sub(const uint8_t * b, const int ib, uint2 (&g)[4], uint4 (&m)[4], float & f0, float & f1) {
    if constexpr (T == GGML_TYPE_IQ2_XXS) {
        const uint8_t * p = b + 2 + 8*ib;
        const uint32_t aux = mmvq_f16_ld2(p), aux32 = mmvq_f16_ld2(p + 4);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            g[l] = __ldg(((const uint2 *) iq2xxs_grid) + ((aux >> 8*l) & 0xFF));
            m[l] = __ldg((const uint4 *) &mmvq_f16_sgn7.v[(aux32 >> 7*l) & 127][0]);
        }
        f0 = f1 = (0.5f + (float) (aux32 >> 28))*0.25f;
    } else if constexpr (T == GGML_TYPE_IQ2_XS) {
        const uint8_t * p = b + 2 + 8*ib;
        const uint32_t q0 = mmvq_f16_ld2(p), q1 = mmvq_f16_ld2(p + 4), sc = b[66 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q = ((l < 2 ? q0 : q1) >> 16*(l & 1)) & 0xFFFF;
            g[l] = __ldg(((const uint2 *) iq2xs_grid) + (q & 511));
            m[l] = __ldg((const uint4 *) &mmvq_f16_sgn7.v[q >> 9][0]);
        }
        f0 = (0.5f + (float) (sc & 0xF))*0.25f;
        f1 = (0.5f + (float) (sc >> 4))*0.25f;
    } else {
        const uint32_t qs = mmvq_f16_ld2(b + 2 + 4*ib), sg = mmvq_f16_ld2(b + 34 + 4*ib), qh = b[66 + ib], sc = b[74 + ib];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            g[l] = __ldg(((const uint2 *) iq2s_grid) + (((qs >> 8*l) & 0xFF) | ((qh << (8 - 2*l)) & 0x300)));
            m[l] = __ldg((const uint4 *) &mmvq_f16_sgn8.v[(sg >> 8*l) & 0xFF][0]);
        }
        f0 = (0.5f + (float) (sc & 0xF))*0.25f;
        f1 = (0.5f + (float) (sc >> 4))*0.25f;
    }
}
// 8 grid bytes -> 4 half2 of +-g (exact): half(1024 + g) - 1024, then the sign bit
static __device__ __forceinline__ void mmvq_f16_iq2_w8(const uint2 g, const uint4 m, __half2 (&w)[4]) {
    const __half2 k = __float2half2_rn(1024.0f);
    const uint32_t h[4] = { __byte_perm(g.x, 0x64646464u, 0x5140), __byte_perm(g.x, 0x64646464u, 0x5342),
                            __byte_perm(g.y, 0x64646464u, 0x5140), __byte_perm(g.y, 0x64646464u, 0x5342) };
    const uint32_t mm[4] = { m.x, m.y, m.z, m.w };
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const __half2 v = __hsub2(*(const __half2 *) &h[j], k);
        const uint32_t r = *(const uint32_t *) &v ^ mm[j];
        w[j] = *(const __half2 *) &r;
    }
}

template <ggml_type T, int NC, int RPW, int NWT, bool GLU>
__launch_bounds__(NWT*WARP_SIZE)
static __global__ void mmvq_f16_iq2(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                    const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                    const int64_t sy, const int rows, const int K) {
    constexpr int BS = mmvq_f16_iq2_t<T>::BS;
    constexpr int ORW = GLU ? RPW/2 : RPW;
    constexpr int WB = (MMVQ_F16_WIN/256)*BS, NU = (WB + 15 + 15)/16;
    __shared__ uint4 wst[NWT][RPW][NU];
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int row0 = (blockIdx.x*NWT + wid)*ORW;
    if (row0 >= rows) {
        return;
    }
    const int nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const int blk = lane >> 3, ib = lane & 7;
    for (int win = 0; win < nw; ++win) {
        const int wbytes = min((int64_t) WB, row_bytes - (int64_t) win*WB);
        const uint8_t * wb[RPW];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint8_t * g = rp[i] + (int64_t) win*WB;
            const int m = (int) ((uintptr_t) g & 15);
            wb[i] = (const uint8_t *) wst[wid][i] + m;
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + wbytes + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
        const int k0 = win*128 + 4*lane; // first chunk of this lane's sub-block
        if (k0 < K/8) {
            __half2 x[NC][4][4];
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const uint4 u = __ldg((const uint4 *) (XS + (int64_t) c*K + 8*(k0 + l)));
                    x[c][l][0] = *(const __half2 *) &u.x; x[c][l][1] = *(const __half2 *) &u.y;
                    x[c][l][2] = *(const __half2 *) &u.z; x[c][l][3] = *(const __half2 *) &u.w;
                }
            }
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                uint2 g[4]; uint4 m[4]; float f0, f1;
                const float d = mmvq_f16_h(wb[i] + blk*BS)*1024.0f;
                mmvq_f16_iq2_sub<T>(wb[i] + blk*BS, ib, g, m, f0, f1);
                const __half2 s0 = __float2half2_rn(d*f0), s1 = __float2half2_rn(d*f1);
                __half2 u[NC][2];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    u[c][0] = u[c][1] = __float2half2_rn(0.0f);
                }
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    __half2 w[4];
                    mmvq_f16_iq2_w8(g[l], m[l], w);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 v = __hfma2(w[0], x[c][l][0], u[c][l >> 1]);
                        v = __hfma2(w[1], x[c][l][1], v);
                        v = __hfma2(w[2], x[c][l][2], v);
                        u[c][l >> 1] = __hfma2(w[3], x[c][l][3], v);
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    t[c][i] = __hfma2(u[c][0], s0, t[c][i]);
                    t[c][i] = __hfma2(u[c][1], s1, t[c][i]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}


// ---- mxfp4 (agent mx): its own kernel. A block is 17 bytes: e8m0 scale e, then 16 bytes of nibbles
// (byte k: low nibble = value k, high nibble = value k+16). A lane owns a unit of 8 bytes = 16 values
// (low nibbles = values 8j..8j+7, high nibbles = values 16+8j..), so both halves are plain 16-byte
// activation loads. The e2m1 magnitude is looked up with one PRMT per 4 nibbles straight into the high
// byte of an exact fp16 (kvalues 0,1,2,3,4,6,8,12 = 0x00,3C,40,42,44,46,48,4A), the sign comes from a
// PRMT in replicate-msb mode, and the weights stay exact small integers: the block scale (a power of
// two) multiplies the unit's dot product once instead of every weight (exact, so same result).
// Scale 2^(e-127) * 0.5 (kvalues are doubled) * 1024 is clamped to 2^-24..2^7 so it is a finite fp16.
static __device__ __forceinline__ uint32_t mmvq_f16_prmt(const uint32_t a, const uint32_t b, const uint32_t s) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
    return r;
}
// 4 bytes of nibbles -> weights as fp16 (kvalues_mxfp4, exact): lo[0..1] = values (0,1) (2,3) of the low
// nibbles, hi[0..1] = the same of the high nibbles
static __device__ __forceinline__ void mmvq_f16_mx_w4(const uint32_t v, uint32_t (&lo)[2], uint32_t (&hi)[2]) {
    const uint32_t vm = v & 0x77777777, vs = v << 4;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const uint32_t mag = mmvq_f16_prmt(0x42403C00u, 0x4A484644u, r ? vm >> 16 : vm);
        const uint32_t sg  = mmvq_f16_prmt(vs, v, r ? 0xFBEA : 0xD9C8);
        const uint32_t m   = mag | (sg & 0x80808080u);
        hi[r] = m & 0xFF00FF00u;
        lo[r] = mmvq_f16_prmt(m, 0, 0x2404);
    }
}
#ifndef MX_ACC2
#define MX_ACC2 0
#endif
template <int NC, int RPW, int NWT, bool GLU>
__launch_bounds__(NWT*WARP_SIZE)
static __global__ void mmvq_f16_mx(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                   const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                   const int64_t sy, const int rows, const int K) {
    constexpr int ORW = GLU ? RPW/2 : RPW;
    constexpr int WB = (MMVQ_F16_WIN/32)*17, NU = (WB + 15 + 15)/16;
    __shared__ uint4 wst[NWT][RPW][NU];
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int row0 = (blockIdx.x*NWT + wid)*ORW;
    if (row0 >= rows) {
        return;
    }
    const int nunit = K/16, nw = (K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    for (int win = 0; win < nw; ++win) {
        const int wbytes = min((int64_t) WB, row_bytes - (int64_t) win*WB);
        const uint8_t * wb[RPW];
        __syncwarp();
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            const uint8_t * g = rp[i] + (int64_t) win*WB;
            const int m = (int) ((uintptr_t) g & 15);
            wb[i] = (const uint8_t *) wst[wid][i] + m;
            const uint4 * g16 = (const uint4 *) (g - m);
            const int nu = (m + wbytes + 15)/16;
#pragma unroll
            for (int r = 0; r < (NU + 31)/32; ++r) {
                const int k = r*32 + lane;
                if (k < nu) {
                    wst[wid][i][k] = __ldg(g16 + k);
                }
            }
        }
        __syncwarp();
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const int u = jj*32 + lane, gu = win*64 + u;
            if (gu < nunit) {
                const int blk = u >> 1, j = u & 1;
                __half2 xa[NC][4], xc[NC][4];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const __half * xp = XS + (int64_t) c*K + (int64_t) win*MMVQ_F16_WIN + 32*blk + 8*j;
                    const uint4 ua = __ldg((const uint4 *) xp), uc = __ldg((const uint4 *) (xp + 16));
                    xa[c][0] = *(const __half2 *) &ua.x; xa[c][1] = *(const __half2 *) &ua.y;
                    xa[c][2] = *(const __half2 *) &ua.z; xa[c][3] = *(const __half2 *) &ua.w;
                    xc[c][0] = *(const __half2 *) &uc.x; xc[c][1] = *(const __half2 *) &uc.y;
                    xc[c][2] = *(const __half2 *) &uc.z; xc[c][3] = *(const __half2 *) &uc.w;
                }
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    const uint8_t * b = wb[i] + 17*blk;
                    uint32_t a0, a1;
                    mmvq_f16_ld8u(b + 1 + 8*j, a0, a1);
                    const int e = min(max((int) b[0], 94), 125);
                    const __half2 s2 = __float2half2_rn(__uint_as_float((uint32_t) (e + 9) << 23));
                    uint32_t lo0[2], hi0[2], lo1[2], hi1[2];
                    mmvq_f16_mx_w4(a0, lo0, hi0);
                    mmvq_f16_mx_w4(a1, lo1, hi1);
                    const __half2 wa[4] = { *(const __half2 *) &lo0[0], *(const __half2 *) &lo0[1], *(const __half2 *) &lo1[0], *(const __half2 *) &lo1[1] };
                    const __half2 wc[4] = { *(const __half2 *) &hi0[0], *(const __half2 *) &hi0[1], *(const __half2 *) &hi1[0], *(const __half2 *) &hi1[1] };
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 ua = __hmul2(wa[0], xa[c][0]);
                        __half2 uc = __hmul2(wc[0], xc[c][0]);
#pragma unroll
                        for (int q = 1; q < 4; ++q) {
                            ua = __hfma2(wa[q], xa[c][q], ua);
                            uc = __hfma2(wc[q], xc[c][q], uc);
                        }
                        t[c][i] = __hfma2(ua, s2, t[c][i]);
                        t[c][i] = __hfma2(uc, s2, t[c][i]);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}
#ifndef MX_RPW_LO
#define MX_RPW_LO 4
#endif
#ifndef MX_RPW_HI
#define MX_RPW_HI 2
#endif
#ifndef MX_NWT
#define MX_NWT 2
#endif
// ---- end mxfp4

static bool mmvq_f16_gen_type(const ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

// Whether the fp16 path takes ncols (>= 2) columns of type t, else the integer mmvq does. From the
// harness against test-backend-ops' integer timings at the model's per-GPU shapes (OPTLOG 282). One
// column always stays integer (solo decode; the fp16 kernels lose on small matrices there). Above
// MMVQ_MAX_BATCH_SIZE the integer side is MMQ, 3-5x slower, so every type takes fp16 there.
static bool mmvq_f16_gen_takes(const ggml_type t, const int64_t ncols) {
    // GGML_CUDA_MMVQ_F16_GEN_MIN: one threshold for every type (measurement)
    static const int env = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_GEN_MIN"); return s ? atoi(s) : 0; }();
    if (env > 0) {
        return ncols >= env;
    }
    if (ncols > MMVQ_MAX_BATCH_SIZE) {
        return true;
    }
    switch (t) {
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_MXFP4:
            return ncols >= 2;
        case GGML_TYPE_Q3_K:
            return ncols >= 3;
        case GGML_TYPE_Q8_0: // its integer path is the faster one at 6..8
            return ncols >= 3 && ncols <= 5;
        case GGML_TYPE_Q4_0: // its integer path is the faster one at 4..8
            return ncols >= 2 && ncols <= 3;
        default:
            return false;
    }
}

template <typename F>
static void mmvq_f16_gen_dispatch(const ggml_type t, F && f) {
    switch (t) {
        case GGML_TYPE_Q4_0:   f(std::integral_constant<ggml_type, GGML_TYPE_Q4_0>{});   break;
        case GGML_TYPE_Q4_1:   f(std::integral_constant<ggml_type, GGML_TYPE_Q4_1>{});   break;
        case GGML_TYPE_Q5_0:   f(std::integral_constant<ggml_type, GGML_TYPE_Q5_0>{});   break;
        case GGML_TYPE_Q5_1:   f(std::integral_constant<ggml_type, GGML_TYPE_Q5_1>{});   break;
        case GGML_TYPE_Q8_0:   f(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});   break;
        case GGML_TYPE_IQ4_NL: f(std::integral_constant<ggml_type, GGML_TYPE_IQ4_NL>{}); break;
        case GGML_TYPE_IQ4_XS: f(std::integral_constant<ggml_type, GGML_TYPE_IQ4_XS>{}); break;
        case GGML_TYPE_Q2_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q2_K>{});   break;
        case GGML_TYPE_Q3_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q3_K>{});   break;
        case GGML_TYPE_Q4_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q4_K>{});   break;
        case GGML_TYPE_Q5_K:   f(std::integral_constant<ggml_type, GGML_TYPE_Q5_K>{});   break;
        case GGML_TYPE_IQ3_XXS: f(std::integral_constant<ggml_type, GGML_TYPE_IQ3_XXS>{}); break;
        case GGML_TYPE_IQ3_S:   f(std::integral_constant<ggml_type, GGML_TYPE_IQ3_S>{});   break;
        case GGML_TYPE_IQ1_S:   f(std::integral_constant<ggml_type, GGML_TYPE_IQ1_S>{});   break;
        case GGML_TYPE_IQ1_M:   f(std::integral_constant<ggml_type, GGML_TYPE_IQ1_M>{});   break;
        default: GGML_ABORT("mmvq_f16_gen: type %s", ggml_type_name(t));
    }
}

// columns per launch: up to 8 (wider, split in two)
template <typename F>
static void mmvq_f16_gen_columns(const int64_t ncols, F && f) {
    const int64_t n0 = ncols <= 8 ? ncols : (ncols + 1)/2;
    for (int64_t c0 = 0; c0 < ncols; c0 += n0) {
        const int64_t n = std::min(n0, ncols - c0);
        switch (n) {
            case 1: f(c0, std::integral_constant<int, 1>{}); break;
            case 2: f(c0, std::integral_constant<int, 2>{}); break;
            case 3: f(c0, std::integral_constant<int, 3>{}); break;
            case 4: f(c0, std::integral_constant<int, 4>{}); break;
            case 5: f(c0, std::integral_constant<int, 5>{}); break;
            case 6: f(c0, std::integral_constant<int, 6>{}); break;
            case 7: f(c0, std::integral_constant<int, 7>{}); break;
            case 8: f(c0, std::integral_constant<int, 8>{}); break;
            default: GGML_ABORT("mmvq_f16_gen: %d columns", (int) n);
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// q5_K: a hand-scheduled kernel (the generic one decodes scales and unpacks bits per 8-value chunk).
//
// q5_K's 176-byte block is a multiple of 16, so every block is 16-byte aligned and the lanes load
// their bytes directly (no shared staging). A warp covers a 1024-value window = 4 blocks, 8 lanes per
// block; lane g = lane % 8 of a block owns 16 bytes of qs (bytes 16g.., sub-block pair j = g/2: 16
// low-nibble values of sub-block 2j and the 16 high-nibble values of sub-block 2j+1) and 16 bytes of
// qh: 32 values per row and window = 16 HFMA2 per column, as on q6_K. Each lane decodes the scale and
// min of its own two sub-blocks. Numerics per weight are the generic kernel's:
//   w = (half(1024 + q) - 1024) * half(d*1024*sc) + half(-dmin*1024*m), q = nibble | qh bit << 4
// then a chain of 16 HFMA2 per lane and window, folded into fp32 with the window's scale.
#ifndef K5_DEC
#define K5_DEC 0
#endif
template <int NC, int RPW, int NWT, bool KS, int MINB, bool GLU>
__launch_bounds__(NWT*WARP_SIZE, MINB)
static __global__ void mmvq_f16_q5_K(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                     const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                     const int64_t sy, const int rows, const int K) {
    static_assert(!GLU || (!KS && RPW % 2 == 0), "GLU: row-parallel, gate/up row pairs");
    constexpr int ORW = GLU ? RPW/2 : RPW, RPB = NWT*RPW, WB = 4*176;
    // 7+ columns on the 4-row tile: fold each half-window's 8-HFMA2 chain into fp32 right away instead
    // of holding NC*RPW half2 chains over the window (8704x5120 at 10 columns 234 -> 205 us, NMSE
    // 1.14e-6 -> 8.4e-7); fewer columns or rows keep the window-long chains (faster there)
    constexpr bool FOLD = NC >= 7 && RPW == 4;
    const int lane = threadIdx.x, wid = threadIdx.y;
    const int nb = K/256, nw = (nb + 3)/4;
    const int row0 = KS ? blockIdx.x*RPW : GLU ? (blockIdx.x*NWT + wid)*ORW : blockIdx.x*RPB + wid*RPW;
    const int B = lane >> 3, g = lane & 7, j = g >> 1, h = g & 1;
    const int rl = (4 - 2*j) & 31, rh = (3 - 2*j) & 31; // rotate qh's bit 2j / 2j+1 to bit 4 of each byte
    const bool hiS = j >= 2;                             // sub-blocks 4..7: the packed scale layout
    const int sh0 = 16*(j & 1);
    const char * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        const uint8_t * base = GLU && i >= ORW ? W2 : W;
        rp[i] = (const char *) base + (int64_t) min(row0 + (GLU ? i % ORW : i), rows - 1)*row_bytes + (KS ? (int64_t) wid*WB : 0) + B*176;
    }
    const __half * xw[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        xw[c] = XS + (int64_t) c*K + (KS ? (int64_t) wid*MMVQ_F16_WIN : 0) + B*256 + 64*j + 16*h;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    const __half2 k1024 = __float2half2_rn(1024.0f), kdm = __halves2half2(__float2half(1024.0f), __float2half(-1024.0f));

    for (int win = KS ? wid : 0; win < nw; win += KS ? NWT : 1) {
        const bool act = win*4 + B < nb;
        __half2 t[FOLD ? 1 : NC][RPW];
#pragma unroll
        for (int c = 0; c < (FOLD ? 1 : NC); ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
        if (act) {
            uint4 qs[RPW], qh[RPW];
            __half2 sL[RPW], mL[RPW], sH[RPW], mH[RPW];
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                qs[i] = __ldg((const uint4 *) (rp[i] + 48 + 16*g));
                qh[i] = __ldg((const uint4 *) (rp[i] + 16 + 16*h));
                const uint4 hd = __ldg((const uint4 *) rp[i]);
#if K5_DEC == 0
                const float d = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xFFFF)));
                const float dm = __half2float(__ushort_as_half((unsigned short) (hd.x >> 16)));
                {
                    const uint32_t x0 = hd.y >> sh0, x1 = hd.z >> sh0, x2 = hd.w >> sh0;
                    const uint32_t sc = hiS ? ((x2 & 15) | ((x0 >> 2) & 0x30)) : (x0 & 63);
                    const uint32_t m  = hiS ? (((x2 >> 4) & 15) | ((x1 >> 2) & 0x30)) : (x1 & 63);
                    sL[i] = __float2half2_rn(d*1024.0f*(float) sc);
                    mL[i] = __float2half2_rn(-dm*1024.0f*(float) m);
                }
                {
                    const uint32_t x0 = hd.y >> (sh0 + 8), x1 = hd.z >> (sh0 + 8), x2 = hd.w >> (sh0 + 8);
                    const uint32_t sc = hiS ? ((x2 & 15) | ((x0 >> 2) & 0x30)) : (x0 & 63);
                    const uint32_t m  = hiS ? (((x2 >> 4) & 15) | ((x1 >> 2) & 0x30)) : (x1 & 63);
                    sH[i] = __float2half2_rn(d*1024.0f*(float) sc);
                    mH[i] = __float2half2_rn(-dm*1024.0f*(float) m);
                }
#else
                // scale and min of sub-blocks 2j (lo) and 2j+1 (hi) as exact half integers, times the
                // half(d*1024) / half(-dmin*1024): the product d*1024*sc is exact in fp32 and so is the
                // half product, so the one rounding to half is the same as the generic kernel's
                const uint32_t x0 = hd.y >> sh0, x1 = hd.z >> sh0, x2 = hd.w >> sh0;
                const uint32_t pks = hiS ? ((x2 & 0x0F0Fu) | ((x0 >> 2) & 0x3030u)) : (x0 & 0x3F3Fu);
                const uint32_t pkm = hiS ? (((x2 >> 4) & 0x0F0Fu) | ((x1 >> 2) & 0x3030u)) : (x1 & 0x3F3Fu);
                const uint32_t word = pks | (pkm << 16);
                const uint32_t dd32 = hd.x;
                const __half2 dd = __hmul2(*(const __half2 *) &dd32, kdm);
                const __half2 dB = __low2half2(dd), nB = __high2half2(dd);
                const uint32_t c0 = __byte_perm(word, 0x64646464u, 0x4040), c1 = __byte_perm(word, 0x64646464u, 0x4141);
                const uint32_t c2 = __byte_perm(word, 0x64646464u, 0x4242), c3 = __byte_perm(word, 0x64646464u, 0x4343);
                sL[i] = __hmul2(__hsub2(*(const __half2 *) &c0, k1024), dB);
                sH[i] = __hmul2(__hsub2(*(const __half2 *) &c1, k1024), dB);
                mL[i] = __hmul2(__hsub2(*(const __half2 *) &c2, k1024), nB);
                mH[i] = __hmul2(__hsub2(*(const __half2 *) &c3, k1024), nB);
#endif
            }
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                __half2 wl[RPW][4], wh[RPW][4];
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const uint32_t q  = m == 0 ? (e == 0 ? qs[i].x : qs[i].y) : (e == 0 ? qs[i].z : qs[i].w);
                        const uint32_t hq = m == 0 ? (e == 0 ? qh[i].x : qh[i].y) : (e == 0 ? qh[i].z : qh[i].w);
                        const uint32_t lo = (q & 0x0F0F0F0Fu) | (__funnelshift_l(hq, hq, rl) & 0x10101010u);
                        const uint32_t hi = ((q >> 4) & 0x0F0F0F0Fu) | (__funnelshift_l(hq, hq, rh) & 0x10101010u);
                        const uint32_t a0 = __byte_perm(lo, 0x64646464u, 0x5140), a1 = __byte_perm(lo, 0x64646464u, 0x5342);
                        const uint32_t b0 = __byte_perm(hi, 0x64646464u, 0x5140), b1 = __byte_perm(hi, 0x64646464u, 0x5342);
                        wl[i][2*e]   = __hfma2(__hsub2(*(const __half2 *) &a0, k1024), sL[i], mL[i]);
                        wl[i][2*e+1] = __hfma2(__hsub2(*(const __half2 *) &a1, k1024), sL[i], mL[i]);
                        wh[i][2*e]   = __hfma2(__hsub2(*(const __half2 *) &b0, k1024), sH[i], mH[i]);
                        wh[i][2*e+1] = __hfma2(__hsub2(*(const __half2 *) &b1, k1024), sH[i], mH[i]);
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const uint4 xl = __ldg((const uint4 *) (xw[c] + 8*m)), xh = __ldg((const uint4 *) (xw[c] + 32 + 8*m));
                    const __half2 xlv[4] = { *(const __half2 *) &xl.x, *(const __half2 *) &xl.y, *(const __half2 *) &xl.z, *(const __half2 *) &xl.w };
                    const __half2 xhv[4] = { *(const __half2 *) &xh.x, *(const __half2 *) &xh.y, *(const __half2 *) &xh.z, *(const __half2 *) &xh.w };
                    const float sw = FOLD ? __ldg(S + c*nw + win) : 0.0f;
#pragma unroll
                    for (int i = 0; i < RPW; ++i) {
                        __half2 u = FOLD ? __float2half2_rn(0.0f) : t[FOLD ? 0 : c][i];
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            u = __hfma2(wl[i][k], xlv[k], u);
                        }
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            u = __hfma2(wh[i][k], xhv[k], u);
                        }
                        if constexpr (FOLD) {
                            const __half2 v = __hadd2(u, __lowhigh2highlow(u));
                            acc[c][i] = fmaf(sw, __low2float(v), acc[c][i]);
                        } else {
                            t[c][i] = u;
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            rp[i] += KS ? (int64_t) NWT*WB : WB;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            xw[c] += KS ? NWT*MMVQ_F16_WIN : MMVQ_F16_WIN;
            if constexpr (!FOLD) {
                const float s = __ldg(S + c*nw + win);
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                    acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
                }
            }
        }
    }
    if constexpr (KS) {
        __shared__ float red[NWT][NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                float v = acc[c][i];
#pragma unroll
                for (int o = 16; o; o >>= 1) {
                    v += __shfl_xor_sync(0xFFFFFFFF, v, o);
                }
                if (lane == 0) {
                    red[wid][c][i] = v;
                }
            }
        }
        __syncthreads();
        if (wid == 0 && lane < NC*RPW) {
            const int c = lane / RPW, i = lane % RPW;
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < NWT; ++w) {
                v += red[w][c][i];
            }
            if (row0 + i < rows) {
                Y[c*sy + row0 + i] = v;
            }
        }
        return;
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

struct mmvq_f16_q5_K_cfg { int rpw, nwt, ks, minb; };

// launch geometry per column count and row count
static mmvq_f16_q5_K_cfg mmvq_f16_q5_K_pick(const int nc, const int64_t rows, const bool glu) {
#ifdef K5_FORCE
    if (glu) { return { K5_G_RPW, K5_G_NWT, 0, K5_G_MINB }; }
    return { K5_F_RPW, K5_F_NWT, K5_F_KS, K5_F_MINB };
#else
    // from the k5s sweep (hx harness, us): the 2-warp tile spills past 8 columns (8704x5120 at 10:
    // 314 -> 239 with 4x1x8); small row counts want fewer rows per warp, and a K-split below 256 rows
    if (rows < 256) {
        return glu ? mmvq_f16_q5_K_cfg{ 2, 1, 0, 16 } : mmvq_f16_q5_K_cfg{ 1, 4, 1, 3 };
    }
    if (rows < 1024) {
        return nc <= 5 ? mmvq_f16_q5_K_cfg{ 2, 2, 0, 8 } : mmvq_f16_q5_K_cfg{ 2, 1, 0, 16 };
    }
    return nc <= 2 ? mmvq_f16_q5_K_cfg{ 4, 2, 0, 6 } : mmvq_f16_q5_K_cfg{ 4, 1, 0, 8 };
#endif
}

template <int NC, int RPW, int NWT, int KS, int MINB, bool GLU>
static bool mmvq_f16_q5_K_try(const mmvq_f16_q5_K_cfg c, const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs,
                              const float * sc, float * Y, const int64_t sy, const int64_t rows, const int K, cudaStream_t stream) {
    if (c.rpw != RPW || c.nwt != NWT || c.ks != KS || c.minb != MINB) {
        return false;
    }
    constexpr int ORW = GLU ? RPW/2 : RPW;
    const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*ORW - 1)/(NWT*ORW));
    mmvq_f16_q5_K<NC, RPW, NWT, KS != 0, MINB, GLU><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(W, W2, row_bytes, xs, sc, Y, sy, (int) rows, K);
    return true;
}

template <int NC, bool GLU>
static void mmvq_f16_q5_K_run(const mmvq_f16_q5_K_cfg c, const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs,
                              const float * sc, float * Y, const int64_t sy, const int64_t rows, const int K, cudaStream_t stream) {
#define K5T(R, N, S, M) mmvq_f16_q5_K_try<NC, R, N, S, M, GLU>(c, W, W2, row_bytes, xs, sc, Y, sy, rows, K, stream)
#ifdef K5_FORCE
    if constexpr (GLU) { K5T(K5_G_RPW, K5_G_NWT, 0, K5_G_MINB); } else { K5T(K5_F_RPW, K5_F_NWT, K5_F_KS, K5_F_MINB); }
#else
    if constexpr (!GLU) {
        if (K5T(1, 4, 1, 3)) { return; }
    }
    K5T(4, 2, 0, 6) || K5T(4, 1, 0, 8) || K5T(2, 2, 0, 8) || K5T(2, 1, 0, 16);
#endif
#undef K5T
}

// every column count, and the launch loop over column ranges
static bool mmvq_f16_q5_K_launch(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                                 float * Y, const int64_t sy, const int64_t rows, const int64_t K, const int64_t ncols, const bool glu,
                                 cudaStream_t stream) {
    if (((uintptr_t) W & 15) || (glu && ((uintptr_t) W2 & 15)) || (row_bytes & 15) || ((uintptr_t) xs & 15)) {
        return false;
    }
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t n) {
        const mmvq_f16_q5_K_cfg c = mmvq_f16_q5_K_pick((int) n, rows, glu);
        const __half * x = xs + c0*K;
        const float * s = sc + c0*nw;
        float * y = Y + c0*sy;
#define K5N(NCV) case NCV: if (glu) { mmvq_f16_q5_K_run<NCV, true>(c, W, W2, row_bytes, x, s, y, sy, rows, (int) K, stream); } \
                           else { mmvq_f16_q5_K_run<NCV, false>(c, W, W2, row_bytes, x, s, y, sy, rows, (int) K, stream); } break;
        switch (n) {
            K5N(1) K5N(2) K5N(3) K5N(4) K5N(5) K5N(6) K5N(7) K5N(8) K5N(9) K5N(10) K5N(11) K5N(12)
            default: GGML_ABORT("mmvq_f16_q5_K: %d columns", (int) n);
        }
#undef K5N
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}


// ---------------------------------------------------------------------------------------------
// IQ4_NL / IQ4_XS: dedicated kernel. A lane owns 16 weights of one 32-weight (sub-)block: 8 qs bytes,
// i.e. the elements 8h..8h+7 (low nibbles) and 16+8h..16+8h+7 (high nibbles). The 16-entry table is
// looked up with byte_perm straight from the packed nibbles (no unpacking): the low class (index < 8)
// by the sign-replicate mode (bit 3 of the selector nibble) giving zero for the other class, the high
// class from a flipped selector, plus 0x80 for the high class, so one lookup gives tab + 128 per
// byte in element-pair order. The weights are half(1024 + byte) - 1152 = the exact table values and
// the block scale is applied once to each lane's 16-term fp16 partial sum. A window is 32 blocks.
// PTX prmt keeps the selector nibble's bit 3 (sign replicate); __byte_perm masks it off
static __device__ __forceinline__ void mmvq_f16_iq4w(const uint32_t v, __half2 (&w)[4]) {
    const uint32_t vf = v ^ 0x88888888u, vh = v >> 16, vfh = vf >> 16;
    const uint32_t oa = mmvq_f16_prmt(0x3f2d1801u, 0x766a5d4fu, v)  | mmvq_f16_prmt(0x26190d01u, 0x71594535u, vf)  | (mmvq_f16_prmt(0x01010101u, 0x01010101u, vf)  << 7);
    const uint32_t ob = mmvq_f16_prmt(0x3f2d1801u, 0x766a5d4fu, vh) | mmvq_f16_prmt(0x26190d01u, 0x71594535u, vfh) | (mmvq_f16_prmt(0x01010101u, 0x01010101u, vfh) << 7);
    const __half2 kb = __float2half2_rn(1152.0f);
    const uint32_t h0 = __byte_perm(oa, 0x64646464u, 0x4240), h1 = __byte_perm(oa, 0x64646464u, 0x4341);
    const uint32_t h2 = __byte_perm(ob, 0x64646464u, 0x4240), h3 = __byte_perm(ob, 0x64646464u, 0x4341);
    w[0] = __hsub2(*(const __half2 *) &h0, kb); w[1] = __hsub2(*(const __half2 *) &h1, kb);
    w[2] = __hsub2(*(const __half2 *) &h2, kb); w[3] = __hsub2(*(const __half2 *) &h3, kb);
}

template <ggml_type T> struct mmvq_f16_iq4t;
template <> struct mmvq_f16_iq4t<GGML_TYPE_IQ4_NL> {
    // unit u of a row: qs (8 bytes, half h) and the scale d*1024
    static __device__ __forceinline__ void load(const uint8_t * rp, const int u, const int h, uint2 & q, float & s) {
        const uint8_t * b = rp + 18*(int64_t) u;
        const uint8_t * a0 = b + 2 + 8*h;
        const uintptr_t ad = (uintptr_t) a0;
        const uint32_t * a = (const uint32_t *) (ad & ~(uintptr_t) 3);
        const int sh = (int) (ad & 2)*8;
        const uint32_t w0 = __ldg(a), w1 = __ldg(a + 1), w2 = sh ? __ldg(a + 2) : 0u;
        q.x = __funnelshift_r(w0, w1, sh); q.y = __funnelshift_r(w1, w2, sh);
        s = __half2float(*(const __half *) b)*1024.0f;
    }
};
template <> struct mmvq_f16_iq4t<GGML_TYPE_IQ4_XS> {
    static __device__ __forceinline__ void load(const uint8_t * rp, const int u, const int h, uint2 & q, float & s) {
        const int gb = u >> 3, sb = u & 7;
        const uint8_t * b = rp + 136*(int64_t) gb;
        const uint2 hd = __ldg((const uint2 *) b);
        q = __ldg((const uint2 *) (b + 8 + 16*sb + 8*h));
        const int ls = (int) (((hd.y >> (4*sb)) & 0xF) | (((hd.x >> (16 + 2*sb)) & 3) << 4));
        s = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xFFFF)))*(1024.0f*(float) (ls - 32));
    }
};

template <ggml_type T, int NC, int RPW, int NWT, bool GLU>
__launch_bounds__(NWT*WARP_SIZE)
static __global__ void mmvq_f16_iq4k(const uint8_t * __restrict__ W, const uint8_t * __restrict__ W2, const int64_t row_bytes,
                                     const __half * __restrict__ XS, const float * __restrict__ S, float * __restrict__ Y,
                                     const int64_t sy, const int rows, const int K) {
    using tr = mmvq_f16_iq4t<T>;
    constexpr int ORW = GLU ? RPW/2 : RPW;
    const int lane = threadIdx.x, wid = threadIdx.y, h = lane & 1, ul = lane >> 1;
    const int row0 = (blockIdx.x*NWT + wid)*ORW;
    if (row0 >= rows) {
        return;
    }
    const int nunits = K/32, nw = (nunits + 31)/32;
    const uint8_t * rp[RPW];
#pragma unroll
    for (int i = 0; i < RPW; ++i) {
        rp[i] = (GLU && i >= ORW ? W2 : W) + (int64_t) min(row0 + i % ORW, rows - 1)*row_bytes;
    }
    float acc[NC][RPW];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            acc[c][i] = 0.0f;
        }
    }
    for (int win = 0; win < nw; ++win) {
        uint2 q[2][RPW];
        float sc[2][RPW];
#pragma unroll
        for (int st = 0; st < 2; ++st) {
            const int u = win*32 + st*16 + ul;
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                if (u < nunits) {
                    tr::load(rp[i], u, h, q[st][i], sc[st][i]);
                } else {
                    q[st][i] = make_uint2(0, 0); sc[st][i] = 0.0f;
                }
            }
        }
        __half2 t[NC][RPW];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                t[c][i] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int st = 0; st < 2; ++st) {
            const int u = win*32 + st*16 + ul;
            if (u < nunits) {
                __half2 xa[NC][4], xb[NC][4];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const __half * xp = XS + (int64_t) c*K + (int64_t) u*32 + 8*h;
                    const uint4 a = __ldg((const uint4 *) xp), b = __ldg((const uint4 *) (xp + 16));
                    xa[c][0] = *(const __half2 *) &a.x; xa[c][1] = *(const __half2 *) &a.y;
                    xa[c][2] = *(const __half2 *) &a.z; xa[c][3] = *(const __half2 *) &a.w;
                    xb[c][0] = *(const __half2 *) &b.x; xb[c][1] = *(const __half2 *) &b.y;
                    xb[c][2] = *(const __half2 *) &b.z; xb[c][3] = *(const __half2 *) &b.w;
                }
#pragma unroll
                for (int i = 0; i < RPW; ++i) {
                    __half2 w0[4], w1[4];
                    mmvq_f16_iq4w(q[st][i].x, w0);
                    mmvq_f16_iq4w(q[st][i].y, w1);
                    const __half2 s2 = __float2half2_rn(sc[st][i]);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        __half2 p = __hmul2(w0[0], xa[c][0]);
                        p = __hfma2(w0[2], xa[c][1], p);
                        p = __hfma2(w0[1], xb[c][0], p);
                        p = __hfma2(w0[3], xb[c][1], p);
                        p = __hfma2(w1[0], xa[c][2], p);
                        p = __hfma2(w1[2], xa[c][3], p);
                        p = __hfma2(w1[1], xb[c][2], p);
                        p = __hfma2(w1[3], xb[c][3], p);
                        t[c][i] = __hfma2(s2, p, t[c][i]);
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float s = __ldg(S + c*nw + win);
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                const __half2 u = __hadd2(t[c][i], __lowhigh2highlow(t[c][i]));
                acc[c][i] = fmaf(s, __low2float(u), acc[c][i]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        float v[RPW];
#pragma unroll
        for (int i = 0; i < RPW; ++i) {
            v[i] = acc[c][i];
#pragma unroll
            for (int o = 16; o; o >>= 1) {
                v[i] += __shfl_xor_sync(0xFFFFFFFF, v[i], o);
            }
        }
#pragma unroll
        for (int i = 0; i < ORW; ++i) {
            if (lane == i && row0 + i < rows) {
                if constexpr (GLU) {
                    Y[c*sy + row0 + i] = ggml_cuda_op_silu_single(v[i]) * v[i + ORW];
                } else {
                    Y[c*sy + row0 + i] = v[i];
                }
            }
        }
    }
}

#ifndef IQ4_RPW_LO
#define IQ4_RPW_LO 4
#endif
#ifndef IQ4_RPW_HI
#define IQ4_RPW_HI 2
#endif
#ifndef IQ4_NWT_LO
#define IQ4_NWT_LO 2
#endif
#ifndef IQ4_NWT_HI
#define IQ4_NWT_HI 2
#endif

template <ggml_type T>
static void mmvq_f16_iq4_launch(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                                float * Y, const int64_t sy, const int64_t rows, const int64_t K, const int64_t ncols, const bool glu,
                                cudaStream_t stream) {
    const int nw = (int) (((K/32) + 31)/32);
    mmvq_f16_gen_columns(ncols, [&](const int64_t c0, auto ncc) {
        constexpr int NC = decltype(ncc)::value;
        constexpr int RPW = NC <= 4 ? IQ4_RPW_LO : IQ4_RPW_HI, NWT = NC <= 4 ? IQ4_NWT_LO : IQ4_NWT_HI;
        if (glu) {
            const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
            mmvq_f16_iq4k<T, NC, 2*RPW, NWT, true><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
        } else {
            const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
            mmvq_f16_iq4k<T, NC, RPW, NWT, false><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                W, nullptr, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
        }
    });
}

template <ggml_type T>
static void mmvq_f16_iq2_launch(const uint8_t * W, const uint8_t * W2, const int64_t row_bytes, const __half * xs, const float * sc,
                                float * Y, const int64_t sy, const int64_t rows, const int64_t K, const int64_t ncols,
                                const bool glu, cudaStream_t stream) {
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_gen_columns(ncols, [&](const int64_t c0, auto ncc) {
        constexpr int NC = decltype(ncc)::value;
        constexpr int RPW = NC <= 4 ? 4 : 2, NWT = 2;
        const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
        if (glu) {
            mmvq_f16_iq2<T, NC, 2*RPW, NWT, true><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
        } else {
            mmvq_f16_iq2<T, NC, RPW, NWT, false><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                W, nullptr, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
        }
    });
}

// GLU: W2 = up (same shape as W)
static void mmvq_f16_gen_launch(const ggml_type type, const uint8_t * W, const uint8_t * W2, const int64_t row_bytes,
                                const __half * xs, const float * sc, float * Y, const int64_t sy, const int64_t rows,
                                const int64_t K, const int64_t ncols, const bool glu, cudaStream_t stream) {
    // The dedicated kernels lose to the generic one on small matrices at 5+ columns (512 rows), and
    // the iq4 kernel at 8 columns per launch (15..16 columns); those keep the generic kernel.
    const bool small_wide = rows < 1024 && ncols >= 5;
    if (type == GGML_TYPE_Q5_K && mmvq_f16_q5_K_launch(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream)) {
        return;
    }
    if (type == GGML_TYPE_Q4_K && !small_wide) {
        mmvq_f16_q4_K_launch(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream);
        return;
    }
    if (type == GGML_TYPE_IQ4_NL && ncols < 15) {
        mmvq_f16_iq4_launch<GGML_TYPE_IQ4_NL>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (type == GGML_TYPE_IQ4_XS && ncols < 15) {
        mmvq_f16_iq4_launch<GGML_TYPE_IQ4_XS>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if ((type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0) && K % 256 == 0 && row_bytes % 16 == 0 &&
        ((uintptr_t) W & 15) == 0 && (W2 == nullptr || ((uintptr_t) W2 & 15) == 0)) {
        const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
        mmvq_f16_32_launch(type == GGML_TYPE_Q8_0, W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, nw, stream);
        return;
    }
    // q4_1 always; q5_1 from 8 columns; q5_0 keeps the generic kernel (it measured faster there)
    if (type == GGML_TYPE_Q4_1) {
        mmvq_f16_s32_launch<GGML_TYPE_Q4_1>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream);
        return;
    }
    if (type == GGML_TYPE_Q5_1 && ncols >= 8) {
        mmvq_f16_s32_launch<GGML_TYPE_Q5_1>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream);
        return;
    }
    if (mmvq_f16_is_iq2(type)) {
        switch (type) {
            case GGML_TYPE_IQ2_XXS: mmvq_f16_iq2_launch<GGML_TYPE_IQ2_XXS>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream); break;
            case GGML_TYPE_IQ2_XS:  mmvq_f16_iq2_launch<GGML_TYPE_IQ2_XS>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream); break;
            default:                mmvq_f16_iq2_launch<GGML_TYPE_IQ2_S>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    if (type == GGML_TYPE_Q2_K && !small_wide) { mmvq_f16_k23_launch<GGML_TYPE_Q2_K>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream); CUDA_CHECK(cudaGetLastError()); return; }
    if (type == GGML_TYPE_Q3_K && !small_wide) { mmvq_f16_k23_launch<GGML_TYPE_Q3_K>(W, W2, row_bytes, xs, sc, Y, sy, rows, K, ncols, glu, stream); CUDA_CHECK(cudaGetLastError()); return; }
    if (type == GGML_TYPE_MXFP4) { // own kernel (mmvq_f16_mx)
        mmvq_f16_gen_columns(ncols, [&](const int64_t c0, auto ncc) {
            constexpr int NC = decltype(ncc)::value;
            constexpr int RPW = NC <= 4 ? MX_RPW_LO : MX_RPW_HI, NWT = MX_NWT;
            const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
            if (glu) {
                mmvq_f16_mx<NC, 2*RPW, NWT, true><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            } else {
                mmvq_f16_mx<NC, RPW, NWT, false><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, nullptr, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            }
        });
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    mmvq_f16_gen_dispatch(type, [&](auto tt) {
        constexpr ggml_type T = decltype(tt)::value;
        mmvq_f16_gen_columns(ncols, [&](const int64_t c0, auto ncc) {
            constexpr int NC = decltype(ncc)::value;
            constexpr int RPW = NC <= 4 ? 4 : 2, NWT = 2;
            if (glu) {
                const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
                mmvq_f16_gen<T, NC, 2*RPW, NWT, true><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, W2, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            } else {
                const int g = (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
                mmvq_f16_gen<T, NC, RPW, NWT, false><<<g, dim3(WARP_SIZE, NWT), 0, stream>>>(
                    W, nullptr, row_bytes, xs + c0*K, sc + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
            }
        });
    });
    CUDA_CHECK(cudaGetLastError());
}

// The prescaled fp16 activation, reused across consecutive matmuls that read the same src1 (gate
// and up, the q/k/v projections), like the integer path's q8_1 cache. Keyed on the src1 node, whose
// contents are fixed within one graph evaluation; cleared at the start of each one. One persistent
// buffer per CUDA context (contexts are per device).
struct mmvq_f16_cache {
    const ggml_tensor * src1 = nullptr;
    const void *        data = nullptr;
    int64_t             ncols = 0, K = 0;
    void *              buf  = nullptr;
    size_t              cap  = 0;
    int                 device = -1; // the buffer's device: a freed context's address can be reused by one on another GPU
};
static std::unordered_map<const ggml_backend_cuda_context *, mmvq_f16_cache> mmvq_f16_caches;

void ggml_cuda_mmvq_f16_invalidate(ggml_backend_cuda_context & ctx) {
    auto it = mmvq_f16_caches.find(&ctx);
    if (it != mmvq_f16_caches.end()) {
        it->second.src1 = nullptr;
        it->second.data = nullptr;
    }
}

static bool mmvq_f16_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int64_t ncols);
static void mmvq_f16_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, int64_t ncols, int64_t K,
                             __half ** xs_ptr, float ** sc_ptr);

bool ggml_cuda_mmvq_f16_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * mm_b,
                                 const ggml_tensor * b, const ggml_tensor * m, ggml_tensor * gate_out, ggml_tensor * beta_out,
                                 const int64_t ncols) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FUSE_GDN_GATE");
        return !s || atoi(s) != 0;
    }();
    const ggml_tensor * src1 = mm_a->src[1];
    const int64_t K = mm_a->src[0]->ne[0], ra = mm_a->src[0]->ne[1], rb = mm_b->src[0]->ne[1];
    auto vec_ok = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == ra && ggml_nelements(t) == ra && ggml_is_contiguous(t);
    };
    // outputs: ra (rb) floats per column, contiguous
    auto out_ok = [&](const ggml_tensor * t, int64_t r) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == r*mm_a->ne[1];
    };
    if (!enabled || mm_a->src[0]->type != GGML_TYPE_Q6_K || mm_b->src[0]->type != GGML_TYPE_Q6_K ||
            mm_b->src[1] != src1 || mm_b->src[0]->ne[0] != K || ra + rb >= 256 || ra != rb ||
            !mmvq_f16_shape_ok(mm_a->src[0], src1, mm_a, ncols) || !mmvq_f16_shape_ok(mm_b->src[0], src1, mm_b, ncols) ||
            !vec_ok(b) || !vec_ok(m) || !out_ok(gate_out, ra) || !out_ok(beta_out, rb)) {
        return false;
    }
    __half * xs_ptr;
    float  * sc_ptr;
    mmvq_f16_prepare(ctx, src1, ncols, K, &xs_ptr, &sc_ptr);
    mmvq_f16_gate g;
    g.W2 = (const uint8_t *) mm_b->src[0]->data;
    g.Y2 = (float *) beta_out->data;
    g.b  = (const float *) b->data;
    g.m  = (const float *) m->data;
    g.rows1 = (int) ra;
    const uint8_t * W = (const uint8_t *) mm_a->src[0]->data;
    float * Y = (float *) gate_out->data;
    const int g_blocks = (int) (ra + rb);
    const dim3 bdk(WARP_SIZE, 4);
    cudaStream_t stream = ctx.stream();
    const int64_t rbytes = mm_a->src[0]->nb[1];
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t n) {
        mmvq_f16_gate gc = g;
        gc.Y2 = g.Y2 + c0*rb;
        mmvq_f16_nc(n, [&](auto nc) {
            constexpr int NC = decltype(nc)::value;
            mmvq_f16_q6_K<NC, 1, 4, true, 3, true><<<g_blocks, bdk, 0, stream>>>(W, rbytes, xs_ptr + c0*K, sc_ptr + c0*nw, Y + c0*ra, ra, (int) (ra + rb), (int) K, gc);
        });
    });
    CUDA_CHECK(cudaGetLastError());
    if (ncols < mm_a->ne[1]) {
        // the GDN gate and beta that feed the recurrent state run on the whole padded width with
        // no n_active check of their own; zero the columns the kernel above left unwritten so a
        // padded verify does not carry stale compute-buffer bytes into the recurrent snapshot
        CUDA_CHECK(cudaMemsetAsync(Y + ncols*ra, 0, (size_t) (mm_a->ne[1] - ncols) * ra * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(g.Y2 + ncols*rb, 0, (size_t) (mm_b->ne[1] - ncols) * rb * sizeof(float), stream));
    }
    return true;
}

bool ggml_cuda_mmvq_f16_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_gate, const ggml_tensor * mm_up,
                            ggml_tensor * dst, const int64_t ncols) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FUSE_FFN_GLU");
        return !s || atoi(s) != 0;
    }();
    const ggml_tensor * src1 = mm_gate->src[1];
    const ggml_tensor * wg = mm_gate->src[0], * wu = mm_up->src[0];
    const int64_t K = wg->ne[0], rows = wg->ne[1];
    // only the big row-parallel shapes (the unfused kernel's 2 warps x 4 rows band)
    if (!enabled || wu->type != wg->type || mm_up->src[1] != src1 || wu->ne[0] != K || wu->ne[1] != rows || wu->nb[1] != wg->nb[1] || rows < 3072 ||
            !mmvq_f16_shape_ok(wg, src1, mm_gate, ncols) || !mmvq_f16_shape_ok(wu, src1, mm_up, ncols) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || dst->ne[0] != rows || ggml_nrows(dst) != mm_gate->ne[1]) {
        return false;
    }
    __half * xs_ptr;
    float  * sc_ptr;
    mmvq_f16_prepare(ctx, src1, ncols, K, &xs_ptr, &sc_ptr);
    mmvq_f16_gate g;
    g.W2 = (const uint8_t *) wu->data;
    const uint8_t * W = (const uint8_t *) wg->data;
    float * Y = (float *) dst->data;
    const int64_t sy = dst->nb[1]/sizeof(float);
    constexpr int ORW = MMVQ_F16_RPW/2;
    cudaStream_t stream = ctx.stream();
    const int64_t full = mm_gate->ne[1];
    if (wg->type != GGML_TYPE_Q6_K) {
        mmvq_f16_gen_launch(wg->type, W, g.W2, wg->nb[1], xs_ptr, sc_ptr, Y, sy, rows, K, ncols, true, stream);
        if (ncols < full) {
            CUDA_CHECK(cudaMemsetAsync(Y + ncols*sy, 0, (size_t) (full - ncols) * sy * sizeof(float), stream));
        }
        return true;
    }
    const int64_t rb = wg->nb[1];
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    mmvq_f16_columns(ncols, [&](const int64_t c0, const int64_t n) {
        mmvq_f16_nc(n, [&](auto nc) {
            constexpr int NC = decltype(nc)::value;
            // more than 5 columns: one warp per block, 8 blocks per SM (the 2-warp tile's 168 registers spill)
            constexpr int NWT = NC <= 5 ? MMVQ_F16_NW : 1, MINB = NC <= 5 ? 6 : 8;
            const int nblk = (int) ((rows + NWT*ORW - 1)/(NWT*ORW));
            mmvq_f16_q6_K<NC, MMVQ_F16_RPW, NWT, false, MINB, false, true><<<nblk, dim3(WARP_SIZE, NWT), 0, stream>>>(
                W, rb, xs_ptr + c0*K, sc_ptr + c0*nw, Y + c0*sy, sy, (int) rows, (int) K, g);
        });
    });
    CUDA_CHECK(cudaGetLastError());
    if (ncols < full) {
        // this FFN output does not feed GDN, but a short verify's unwritten columns still hold
        // stale compute-buffer bytes; zero them for the same reason as ggml_cuda_mmvq_f16_gdn_gate
        CUDA_CHECK(cudaMemsetAsync(Y + ncols*sy, 0, (size_t) (full - ncols) * sy * sizeof(float), stream));
    }
    return true;
}

static bool mmvq_f16_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int64_t ncols) {
    // GGML_CUDA_MMVQ_F16=0 turns the fp16 path off
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_MMVQ_F16");
        return !s || atoi(s) != 0;
    }();
    // GGML_CUDA_MMVQ_F16_MINROWS: the smallest row count that takes this path
    static const int64_t min_rows = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_MINROWS"); return s ? (int64_t) atoll(s) : (int64_t) 16; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t K = src0->ne[0], rows = src0->ne[1];
    // q6_K: its own kernel (K must hold an even number of blocks); the other types: the generic kernel
    const bool q6 = src0->type == GGML_TYPE_Q6_K;
    if (!q6 && mmvq_f16_gen_type(src0->type) && !mmvq_f16_gen_takes(src0->type, ncols)) {
        return false;
    }
    return !(!enabled || cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc) || !(q6 || mmvq_f16_gen_type(src0->type)) ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ncols < (q6 ? 2 : 1) || ncols > MMVQ_F16_MAX_COLS ||
            K % (q6 ? 512 : 256) != 0 || rows < min_rows || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[1] != ggml_row_size(src0->type, K) || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            (src1->nb[1] % 16) != 0 || ((uintptr_t) src1->data % 16) != 0 || ((uintptr_t) src0->data % 4) != 0);
}

static void mmvq_f16_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const int64_t ncols, const int64_t K,
                             __half ** xs_out, float ** sc_out) {
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    cudaStream_t stream = ctx.stream();

    mmvq_f16_cache & cache = mmvq_f16_caches[&ctx];
    if (cache.device != ctx.device) {
        // a stale entry from a destroyed context on another device: its buffer is not usable here
        if (cache.buf) {
            ggml_cuda_set_device(cache.device);
            CUDA_CHECK(cudaFree(cache.buf));
            ggml_cuda_set_device(ctx.device);
        }
        cache = mmvq_f16_cache();
        cache.device = ctx.device;
    }
    const size_t xs_bytes = (size_t) ncols*K*sizeof(__half);
    const size_t need     = xs_bytes + (size_t) ncols*nw*sizeof(float);
    const bool hit = cache.src1 == src1 && cache.data == src1->data && cache.ncols == ncols && cache.K == K;
    if (!hit) {
        if (cache.cap < need) {
            if (cache.buf) {
                CUDA_CHECK(cudaFree(cache.buf)); // synchronizes; only when the buffer grows
            }
            CUDA_CHECK(cudaMalloc(&cache.buf, need));
            cache.cap = need;
        }
        cache.src1 = src1; cache.data = src1->data; cache.ncols = ncols; cache.K = K;
    }
    __half * xs_ptr = (__half *) cache.buf;
    float  * sc_ptr = (float *) ((char *) cache.buf + xs_bytes);
    if (!hit) {
        const dim3 pb(WARP_SIZE, 4), pg((nw + 3)/4, ncols);
        mmvq_f16_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, src1->nb[1]/sizeof(float), xs_ptr, sc_ptr, (int) K);
    }
    *xs_out = xs_ptr;
    *sc_out = sc_ptr;
}

bool ggml_cuda_mmvq_f16_takes(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int64_t ncols) {
    return mmvq_f16_shape_ok(src0, src1, dst, ncols);
}

bool ggml_cuda_mmvq_f16_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            ggml_tensor * dst, const int64_t ncols) {
    const int64_t K = src0->ne[0], rows = src0->ne[1];
    // Where it measured faster than the integer path (OPTLOG 206), test-backend-ops at 5 columns, rows x K:
    // 8704x5120 154 -> 133 us, 6144x5120 115 -> 97, 5120x5120 98 -> 89, 3072x5120 67 -> 61,
    // 5120x8704 161 -> 146, 5120x3072 61 -> 57. K must hold an even number of q6_K blocks.
    if (!mmvq_f16_shape_ok(src0, src1, dst, ncols)) {
        return false;
    }
    const int nw = (int) ((K + MMVQ_F16_WIN - 1)/MMVQ_F16_WIN);
    cudaStream_t stream = ctx.stream();
    __half * xs_ptr;
    float  * sc_ptr;
    mmvq_f16_prepare(ctx, src1, ncols, K, &xs_ptr, &sc_ptr);
    const int64_t sy = dst->nb[1]/sizeof(float);
    const uint8_t * W = (const uint8_t *) src0->data;
    float * Y = (float *) dst->data;
    if (src0->type != GGML_TYPE_Q6_K) {
        mmvq_f16_gen_launch(src0->type, W, nullptr, src0->nb[1], xs_ptr, sc_ptr, Y, sy, rows, K, ncols, false, stream);
        return true;
    }
    // big matrices: 2 warps x 4 rows per block; mid-size: 1 row per warp; small (< 256 rows): the 4
    // warps of a block split K over one row, so there are enough blocks and warps to cover the GPU
    int64_t c0 = 0, nc = ncols; // the column range of the current launch
    auto launch = [&](auto rpw, auto nwt, auto ks, auto minb) {
        constexpr int  RPW  = decltype(rpw)::value;
        constexpr int  NWT  = decltype(nwt)::value;
        constexpr bool KS   = decltype(ks)::value;
        constexpr int  MINB = decltype(minb)::value;
        const dim3 bdk(WARP_SIZE, NWT);
        const int g = KS ? (int) ((rows + RPW - 1)/RPW) : (int) ((rows + NWT*RPW - 1)/(NWT*RPW));
        mmvq_f16_nc(nc, [&](auto ncc) {
            constexpr int NC = decltype(ncc)::value;
            mmvq_f16_q6_K<NC, RPW, NWT, KS, MINB><<<g, bdk, 0, stream>>>(W, src0->nb[1], xs_ptr + c0*K, sc_ptr + c0*nw, Y + c0*sy, sy, (int) rows, (int) K);
        });
    };
    using I = std::integral_constant<int, 1>;
    mmvq_f16_columns(ncols, [&](const int64_t c0_, const int64_t n_) {
    c0 = c0_;
    nc = n_;
    if (nc > 5) {
        // several sequences' verify: 6..12 columns per launch. The 2-warp tiles' 168 registers spill
        // here; one warp per block with up to 255 registers does not (8704x5120 at 10 columns: 213 us
        // against 367 for the 2-warp tile, which spills, and ~300 for 2-row tiles).
        if (rows >= 3072) {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, I{}, std::false_type{}, std::integral_constant<int, 8>{});
        } else if (rows >= 256) {
            launch(std::integral_constant<int, 2>{}, I{}, std::false_type{}, std::integral_constant<int, 12>{});
        } else {
            launch(I{}, std::integral_constant<int, 4>{}, std::true_type{}, std::integral_constant<int, 3>{});
        }
        return;
    }
    if (rows >= 3072) {
        // 6 or 7 blocks per SM (168 or 128 registers): take 7 when it needs fewer waves over the
        // SMs (3072 rows: 2 -> 1 wave, 49 vs 57 us at 5 columns; 6144: 3 -> 2; 8704: 4 -> 3).
        const int64_t nblk = (rows + MMVQ_F16_NW*MMVQ_F16_RPW - 1)/(MMVQ_F16_NW*MMVQ_F16_RPW);
        const int64_t nsm  = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const int64_t w6 = (nblk + 6*nsm - 1)/(6*nsm), w7 = (nblk + 7*nsm - 1)/(7*nsm);
        if (w7 < w6) {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 7>{});
        } else {
            launch(std::integral_constant<int, MMVQ_F16_RPW>{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 6>{});
        }
    } else if (rows >= 256) {
        // one warp of 2 rows per block: at 512 rows x 5120 and 5 columns (wk/wv under -sm tensor)
        // 13.3 us against 17.7 for 2 warps of 1 row (test-backend-ops, prep included). The rows'
        // arithmetic doesn't depend on the grouping, so the results are identical.
        // GGML_CUDA_MMVQ_F16_MID=0 restores 2 warps x 1 row.
        static const bool mid2 = [] { const char * s = getenv("GGML_CUDA_MMVQ_F16_MID"); return !s || atoi(s) != 0; }();
        if (mid2) {
            launch(std::integral_constant<int, 2>{}, I{}, std::false_type{}, std::integral_constant<int, 16>{});
        } else {
            launch(I{}, std::integral_constant<int, MMVQ_F16_NW>{}, std::false_type{}, std::integral_constant<int, 12/MMVQ_F16_NW>{});
        }
    } else {
        launch(I{}, std::integral_constant<int, 4>{}, std::true_type{}, std::integral_constant<int, 3>{});
    }
    });
    CUDA_CHECK(cudaGetLastError());
    return true;
}
