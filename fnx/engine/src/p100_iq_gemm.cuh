// p100_iq_gemm.cuh - GP100, the prompt path's expert products: IQ / Q2_0 weights decoded straight into shared memory, FP16 activations, FP32 sums.
// Include after ggml-common.h (blocks, grids) and cuda_fp16.h.
//
// What sets the shape (mikrotests on the card, 2026-11-02): HFMA2 alone reaches 7.9 T MAC/s, but shared memory delivers 3.4 TB/s, global
// reads 1.3-2 TB/s and a card's DRAM 525 GB/s, so a kernel that re-reads its operands stops at 2 T MAC/s; a register block of TL tokens x RL
// rows gets TL RL / (2 (TL + RL)) MAC per byte.  A block holds 128 lanes, each TL x RL pairs.  The weights go to shared memory through tables
// of FP16 values, so a weight row costs its 3 bits on DRAM instead of 16 bits read and written.  A block's tile keeps its 128 lanes busy
// whatever the expert's row count: few rows take more weight rows per block and shorter slices (mode).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace strata::kernels::p100f::iqg {

constexpr int kThreads = 128, kTL = 6, kRL = 8, kFold = 8, kMaxExperts = 16;

struct Tabs {
    uint2* g3xxs;    // 256: 4 half
    uint2* g3s;      // 512: 4 half
    uint4* g2s;      // 1024: 8 half
    uint4* sign8;    // 256: the sign masks of 8 halves from a sign byte
    uint4* sign7;    // 128: the same from IQ3_XXS's 7-bit field
    uint32_t* nib;   // 256: half2(kvalues[low nibble], kvalues[high nibble])
    uint32_t* nib4;  // 256: half2(low nibble - 8, high nibble - 8), Q4_0
    uint2* q20;      // 256: 4 half (code - 1)
    float* s12;      // 16: 1 + 2 s
    float* s6;       // 64: s - 32
    uint4* g2xxs;    // 256: 8 half (IQ2_XXS)
    uint4* g1p;      // 2048: 8 half, IQ1_M's grid value + 0.125
    uint4* g1m;      // 2048: the same - 0.125
    uint32_t* fp4;   // 256: half2(e2m1[low nibble], e2m1[high nibble]) doubled, NVFP4 (edge0_patch.py)
};

struct Args {
    Tabs t;
    const uint8_t* W;
    const uint4* X;
    float* Y;
    size_t expert_bytes, row_bytes;
    int ldx_u, ldy, N, K, n;
    int first[kMaxExperts + 1], r0[kMaxExperts], rows[kMaxExperts];
    int tiles[kMaxExperts], mode[kMaxExperts];
};

__device__ __forceinline__ half2 as_half2(uint32_t u) { half2 h; std::memcpy(&h, &u, 4); return h; }
__device__ __forceinline__ uint32_t as_u32(half2 h) { uint32_t u; std::memcpy(&u, &h, 4); return u; }
__device__ __forceinline__ uint32_t hmul(uint32_t a, half2 d) { return as_u32(__hmul2(as_half2(a), d)); }
__device__ __forceinline__ half2 dup_f(float f) { const half h = __float2half_rn(f); return __halves2half2(h, h); }
__device__ __forceinline__ half2 dup_u16(uint32_t u) { return as_half2((u & 0xffffu) | (u << 16)); }
__device__ __forceinline__ float h2f(uint16_t u) { half h; std::memcpy(&h, &u, 2); return __half2float(h); }
__device__ __forceinline__ uint32_t b2h(uint32_t b) {
    const half h = __float2half(float(b));
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return u;
}

__global__ void __launch_bounds__(256) tabs_kernel(Tabs t) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    auto mask8 = [](uint32_t s) {
        uint4 m;
        m.x = ((s & 1u) << 15) | ((s & 2u) << 30);
        m.y = ((s & 4u) << 13) | ((s & 8u) << 28);
        m.z = ((s & 16u) << 11) | ((s & 32u) << 26);
        m.w = ((s & 64u) << 9) | ((s & 128u) << 24);
        return m;
    };
    if (i < 256) {
        const uint32_t g = iq3xxs_grid[i];
        t.g3xxs[i] = make_uint2(b2h(g & 255) | (b2h((g >> 8) & 255) << 16), b2h((g >> 16) & 255) | (b2h(g >> 24) << 16));
        t.sign8[i] = mask8(uint32_t(i));
        t.nib[i] = as_u32(__floats2half2_rn(float(kvalues_iq4nl[i & 15]), float(kvalues_iq4nl[i >> 4])));
        t.nib4[i] = as_u32(__floats2half2_rn(float(int(i & 15) - 8), float(int(i >> 4) - 8)));
        {   // edge0_patch.py: NVFP4's e2m1 values, doubled as ggml's kvalues_fp4
            const float fv[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};
            t.fp4[i] = as_u32(__floats2half2_rn(fv[i & 15], fv[i >> 4]));
        }
        const half2 q0 = __floats2half2_rn(float(int(i & 3) - 1), float(int((i >> 2) & 3) - 1));
        const half2 q1 = __floats2half2_rn(float(int((i >> 4) & 3) - 1), float(int((i >> 6) & 3) - 1));
        t.q20[i] = make_uint2(as_u32(q0), as_u32(q1));
    }
    if (i < 128) t.sign7[i] = mask8(ksigns_iq2xs[i]);
    if (i < 512) {
        const uint32_t g = iq3s_grid[i];
        t.g3s[i] = make_uint2(b2h(g & 255) | (b2h((g >> 8) & 255) << 16), b2h((g >> 16) & 255) | (b2h(g >> 24) << 16));
    }
    if (i < 1024) {
        const uint2 g = reinterpret_cast<const uint2*>(iq2s_grid)[i];
        t.g2s[i] = make_uint4(b2h(g.x & 255) | (b2h((g.x >> 8) & 255) << 16), b2h((g.x >> 16) & 255) | (b2h(g.x >> 24) << 16),
                              b2h(g.y & 255) | (b2h((g.y >> 8) & 255) << 16), b2h((g.y >> 16) & 255) | (b2h(g.y >> 24) << 16));
    }
    if (i < 256) {
        const uint2 g = reinterpret_cast<const uint2*>(iq2xxs_grid)[i];
        t.g2xxs[i] = make_uint4(b2h(g.x & 255) | (b2h((g.x >> 8) & 255) << 16), b2h((g.x >> 16) & 255) | (b2h(g.x >> 24) << 16),
                                b2h(g.y & 255) | (b2h((g.y >> 8) & 255) << 16), b2h((g.y >> 16) & 255) | (b2h(g.y >> 24) << 16));
    }
    if (i < 2048) {   // iq1s_grid_gpu: value j + 1 in the low nibble of byte j, value j + 4 + 1 in its high nibble (ggml convert.cu)
        const uint32_t g = iq1s_grid_gpu[i];
        float v[8];
        for (int j = 0; j < 4; ++j) {
            v[j] = float(int((g >> (8 * j)) & 15) - 1);
            v[j + 4] = float(int((g >> (8 * j + 4)) & 15) - 1);
        }
        t.g1p[i] = make_uint4(as_u32(__floats2half2_rn(v[0] + 0.125f, v[1] + 0.125f)), as_u32(__floats2half2_rn(v[2] + 0.125f, v[3] + 0.125f)),
                              as_u32(__floats2half2_rn(v[4] + 0.125f, v[5] + 0.125f)), as_u32(__floats2half2_rn(v[6] + 0.125f, v[7] + 0.125f)));
        t.g1m[i] = make_uint4(as_u32(__floats2half2_rn(v[0] - 0.125f, v[1] - 0.125f)), as_u32(__floats2half2_rn(v[2] - 0.125f, v[3] - 0.125f)),
                              as_u32(__floats2half2_rn(v[4] - 0.125f, v[5] - 0.125f)), as_u32(__floats2half2_rn(v[6] - 0.125f, v[7] - 0.125f)));
    }
    if (i < 16) t.s12[i] = 1.0f + 2.0f * float(i);
    if (i < 64) t.s6[i] = float(i - 32);
}

// ---- 32 consecutive values (run 32 * run of a row), as 4 groups of 8 values in 4 half2 words each (the dq_* formulas of iq_kernels.cu).
// A lane takes 4 runs a slice: first all their block bytes (one round trip to memory, all in flight), then each run's tables.
#define P100_U16(p, off) (uint32_t(*reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(p) + (off))))

// NVFP4 (edge0_patch.py): a block of 64 is d[4] (UE4M3, a scale a 16 values) and qs[32] (sub-block s: bytes 8 s .. 8 s + 7, value j in
// the low nibble of byte j, j + 8 in its high one); run r of a row = half r & 1 of block r >> 1: sub-blocks A = 2 h, B = 2 h + 1.
// The scale's value is ggml's ue4m3 (with its 1/2, the codes being doubled) divided by NVFP4_PRE = 16 (the experts' tails take it back).
__device__ __forceinline__ float nvfp4_scale(uint32_t x) {
    x &= 255u;
    if (x == 0u || x == 0x7Fu) return 0.f;
    const int e = int(x >> 3) & 15, m = int(x & 7u);
    return e == 0 ? ldexpf(float(m), -14) : ldexpf(1.0f + float(m) * 0.125f, e - 12);
}
template <typename P8>
__device__ __forceinline__ void nvfp4_words(const P8* base, const int h, uint32_t* w) {
    const int q = 4 + 16 * h;
    w[0] = P100_U16(base, q) | (P100_U16(base, q + 2) << 16);
    w[1] = P100_U16(base, q + 4) | (P100_U16(base, q + 6) << 16);
    w[2] = P100_U16(base, q + 8) | (P100_U16(base, q + 10) << 16);
    w[3] = P100_U16(base, q + 12) | (P100_U16(base, q + 14) << 16);
    w[4] = P100_U16(base, 2 * h);
}
template <int TY> struct RunWords { static constexpr int n = 6; };
template <> struct RunWords<20> { static constexpr int n = 5; };
template <> struct RunWords<40> { static constexpr int n = 5; };   // NVFP4 (edge0_patch.py): codes of sub-blocks A, B; their scales
template <> struct RunWords<2> { static constexpr int n = 5; };
template <> struct RunWords<42> { static constexpr int n = 3; };
template <> struct RunWords<22> { static constexpr int n = 3; };
template <> struct RunWords<16> { static constexpr int n = 3; };   // IQ2_XXS: the codes, the signs and scale word, d
template <> struct RunWords<29> { static constexpr int n = 3; };   // IQ1_M: the codes, qh | the run's two scales << 16, d

// ---- q4x_patch.py: the Q4 XL expert types. A run's words: [0..7] its 32 values one a byte (v < 1024), [8] d | dmin << 16 (K-quants),
// d | m << 16 (Q5_1) or d (Q8_0), [9] the K-quant sub-block's scale | min << 8 (ggml get_scale_min_k4)
template <> struct RunWords<12> { static constexpr int n = 10; };   // Q4_K
template <> struct RunWords<13> { static constexpr int n = 10; };   // Q5_K
template <> struct RunWords<7> { static constexpr int n = 9; };     // Q5_1
template <> struct RunWords<8> { static constexpr int n = 9; };     // Q8_0
template <int TY> inline constexpr bool kQX = TY == 12 || TY == 13 || TY == 7 || TY == 8;
#define P100_U32(p, off) (P100_U16(p, off) | (P100_U16(p, (off) + 2) << 16))
// bit b of the low nibble of x to bit 4 of byte b
__device__ __forceinline__ uint32_t kq_spread4(uint32_t x) { return ((x & 1u) << 4) | ((x & 2u) << 11) | ((x & 4u) << 18) | ((x & 8u) << 25); }
// run ib of the block (K-quants: its sub-block ib; Q5_1, Q8_0: the block itself) at x
template <int TY>
__device__ __forceinline__ void kq_words(const uint8_t* x, const int ib, uint32_t (&w)[RunWords<TY>::n]) {
    if constexpr (TY == 12 || TY == 13) {
        constexpr int QS = TY == 12 ? 16 : 48;
        const int c = ib >> 1, sh = 4 * (ib & 1);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t v = (P100_U32(x, QS + 32 * c + 4 * k) >> sh) & 0x0F0F0F0Fu;
            if constexpr (TY == 13) v |= ((P100_U32(x, 16 + 4 * k) >> ib) & 0x01010101u) << 4;
            w[k] = v;
        }
        w[8] = P100_U32(x, 0);
        const uint8_t* q = x + 4;
        uint32_t s, m;
        if (ib < 4) s = q[ib] & 63, m = q[ib + 4] & 63;
        else s = (q[ib + 4] & 15) | ((q[ib - 4] >> 6) << 4), m = (q[ib + 4] >> 4) | ((q[ib] >> 6) << 4);
        w[9] = s | (m << 8);
    } else if constexpr (TY == 7) {
        const uint32_t qh = P100_U32(x, 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = P100_U32(x, 8 + 4 * k);
            w[k] = (q & 0x0F0F0F0Fu) | kq_spread4((qh >> (4 * k)) & 15u);
            w[k + 4] = ((q >> 4) & 0x0F0F0F0Fu) | kq_spread4((qh >> (16 + 4 * k)) & 15u);
        }
        w[8] = P100_U32(x, 0);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = P100_U32(x, 2 + 4 * k);
        w[8] = P100_U16(x, 0);
    }
}
// bytes b0, b1 of w (selector 0x4140: bytes 0 and 1, 0x4342: bytes 2 and 3) as the f16 pair (v0, v1), exact for v < 1024
__device__ __forceinline__ half2 kq_h2(uint32_t w, uint32_t sel) {
    return __hsub2(as_half2(__byte_perm(w, 0u, sel) | 0x64006400u), as_half2(0x64006400u));
}
template <int TY>
__device__ __forceinline__ void kq_finish(const uint32_t (&w)[RunWords<TY>::n], uint4* __restrict__ out) {
    if constexpr (TY == 12 || TY == 13) {   // d sc q - dmin m
        const half2 a = dup_f(h2f(uint16_t(w[8])) * float(w[9] & 255u)), b = dup_f(-h2f(uint16_t(w[8] >> 16)) * float(w[9] >> 8));
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[j] = make_uint4(as_u32(__hfma2(kq_h2(w[2 * j], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j], 0x4342u), a, b)),
                                as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4342u), a, b)));
    } else if constexpr (TY == 7) {   // d q + m
        const half2 a = dup_u16(w[8] & 0xffffu), b = dup_u16(w[8] >> 16);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[j] = make_uint4(as_u32(__hfma2(kq_h2(w[2 * j], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j], 0x4342u), a, b)),
                                as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4342u), a, b)));
    } else {   // d q: an int8 plus 128 as above, then 1152 off
        const half2 a = dup_u16(w[8]), o = as_half2(0x64806480u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t u0 = w[2 * j] ^ 0x80808080u, u1 = w[2 * j + 1] ^ 0x80808080u;
            const half2 q0 = __hsub2(as_half2(__byte_perm(u0, 0u, 0x4140u) | 0x64006400u), o), q1 = __hsub2(as_half2(__byte_perm(u0, 0u, 0x4342u) | 0x64006400u), o);
            const half2 q2 = __hsub2(as_half2(__byte_perm(u1, 0u, 0x4140u) | 0x64006400u), o), q3 = __hsub2(as_half2(__byte_perm(u1, 0u, 0x4342u) | 0x64006400u), o);
            out[j] = make_uint4(as_u32(__hmul2(q0, a)), as_u32(__hmul2(q1, a)), as_u32(__hmul2(q2, a)), as_u32(__hmul2(q3, a)));
        }
    }
}

template <int TY>
__device__ __forceinline__ void run_load(const uint8_t* __restrict__ row, const int run, uint32_t (&w)[RunWords<TY>::n]) {
    if constexpr (TY == 12 || TY == 13) {   // q4x_patch.py
        kq_words<TY>(row + (run >> 3) * (TY == 12 ? 144 : 176), run & 7, w);
    } else if constexpr (TY == 7 || TY == 8) {
        kq_words<TY>(row + run * (TY == 7 ? 24 : 34), 0, w);
    } else if constexpr (TY == 18) {   // IQ3_XXS
        const block_iq3_xxs* x = reinterpret_cast<const block_iq3_xxs*>(row) + (run >> 3);
        const int ib = run & 7;
        w[0] = P100_U16(x->qs, 8 * ib); w[1] = P100_U16(x->qs, 8 * ib + 2); w[2] = P100_U16(x->qs, 8 * ib + 4); w[3] = P100_U16(x->qs, 8 * ib + 6);
        const uint32_t a0 = P100_U16(x->qs, QK_K / 4 + 4 * ib), a1 = P100_U16(x->qs, QK_K / 4 + 4 * ib + 2);
        w[4] = a0 | (a1 << 16);
        w[5] = P100_U16(&x->d, 0);
    } else if constexpr (TY == 21) {   // IQ3_S
        const block_iq3_s* x = reinterpret_cast<const block_iq3_s*>(row) + (run >> 3);
        const int ib = run & 7;
        w[0] = P100_U16(x->qs, 8 * ib); w[1] = P100_U16(x->qs, 8 * ib + 2); w[2] = P100_U16(x->qs, 8 * ib + 4); w[3] = P100_U16(x->qs, 8 * ib + 6);
        const uint32_t s0 = P100_U16(x->signs, 4 * ib), s1 = P100_U16(x->signs, 4 * ib + 2);
        w[4] = s0 | (s1 << 16);
        const uint32_t qh = x->qh[ib], sc = (uint32_t(x->scales[ib / 2]) >> (4 * (ib & 1))) & 15, du = P100_U16(&x->d, 0);
        w[5] = qh | (sc << 8) | (du << 16);
    } else if constexpr (TY == 22) {   // IQ2_S
        const block_iq2_s* x = reinterpret_cast<const block_iq2_s*>(row) + (run >> 3);
        const int ib = run & 7;
        const uint32_t i0 = P100_U16(x->qs, 4 * ib), i1 = P100_U16(x->qs, 4 * ib + 2);
        const uint32_t s0 = P100_U16(x->qs, QK_K / 8 + 4 * ib), s1 = P100_U16(x->qs, QK_K / 8 + 4 * ib + 2);
        const uint32_t qh = x->qh[ib], sc = x->scales[ib], du = P100_U16(&x->d, 0);
        w[0] = i0 | (i1 << 16);
        w[1] = s0 | (s1 << 16);
        w[2] = qh | (sc << 8) | (du << 16);
    } else if constexpr (TY == 23) {   // IQ4_XS
        const block_iq4_xs* x = reinterpret_cast<const block_iq4_xs*>(row) + (run >> 3);
        const int ib = run & 7;
        const uint2 v0 = __ldg(reinterpret_cast<const uint2*>(x->qs + 16 * ib)), v1 = __ldg(reinterpret_cast<const uint2*>(x->qs + 16 * ib + 8));
        w[0] = v0.x; w[1] = v0.y; w[2] = v1.x; w[3] = v1.y;
        const uint32_t sl = x->scales_l[ib / 2], sh = x->scales_h, du = P100_U16(&x->d, 0);
        w[4] = ((sl >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        w[5] = du;
    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py)
        nvfp4_words(reinterpret_cast<const uint8_t*>(row) + 36 * (run >> 1), run & 1, w);
    } else if constexpr (TY == 20) {   // IQ4_NL
        const block_iq4_nl* x = reinterpret_cast<const block_iq4_nl*>(row) + run;
        w[0] = P100_U16(x->qs, 0) | (P100_U16(x->qs, 2) << 16);
        w[1] = P100_U16(x->qs, 4) | (P100_U16(x->qs, 6) << 16);
        w[2] = P100_U16(x->qs, 8) | (P100_U16(x->qs, 10) << 16);
        w[3] = P100_U16(x->qs, 12) | (P100_U16(x->qs, 14) << 16);
        w[4] = P100_U16(&x->d, 0);
    } else {   // Q2_0: blocks of 64, 2 bits a value
        const block_q2_0* x = reinterpret_cast<const block_q2_0*>(row) + (run >> 1);
        const int part = (run & 1) * 8;
        w[0] = P100_U16(x->qs, part) | (P100_U16(x->qs, part + 2) << 16);
        w[1] = P100_U16(x->qs, part + 4) | (P100_U16(x->qs, part + 6) << 16);
        w[2] = P100_U16(&x->d, 0);
    }
}

// 16 nibbles of 4 words: a word's 2 pairs of bytes, each pair's nibbles at shift sh of each byte
__device__ __forceinline__ uint32_t nib_pair(const Tabs& t, const uint32_t v, const half2 dd) {
    return hmul(__ldg(t.nib + ((v & 15) | ((v >> 4) & 0xf0))), dd);
}

template <int TY>
__device__ __forceinline__ void run_finish(const Tabs& t, const uint32_t (&w)[RunWords<TY>::n], uint4* __restrict__ out) {
    if constexpr (kQX<TY>) {
        kq_finish<TY>(w, out);   // q4x_patch.py
    } else if constexpr (TY == 18) {
        const uint32_t aux = w[4];
        const half2 dd = dup_f(h2f(uint16_t(w[5])) * float(1 + 2 * (aux >> 28)) * 0.25f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 sg = __ldg(t.sign7 + ((aux >> (7 * j)) & 127));
            const uint2 g1 = __ldg(t.g3xxs + (w[j] & 255)), g2 = __ldg(t.g3xxs + (w[j] >> 8));
            out[j] = make_uint4(hmul(g1.x ^ sg.x, dd), hmul(g1.y ^ sg.y, dd), hmul(g2.x ^ sg.z, dd), hmul(g2.y ^ sg.w, dd));
        }
    } else if constexpr (TY == 21) {
        const uint32_t qh = w[5] & 255, sc = (w[5] >> 8) & 255;
        const half2 dd = dup_f(h2f(uint16_t(w[5] >> 16)) * float(1 + 2 * sc));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t i1 = (w[j] & 255) | ((qh << (8 - 2 * j)) & 256), i2 = (w[j] >> 8) | ((qh << (7 - 2 * j)) & 256);
            const uint4 sg = __ldg(t.sign8 + ((w[4] >> (8 * j)) & 255));
            const uint2 g1 = __ldg(t.g3s + i1), g2 = __ldg(t.g3s + i2);
            out[j] = make_uint4(hmul(g1.x ^ sg.x, dd), hmul(g1.y ^ sg.y, dd), hmul(g2.x ^ sg.z, dd), hmul(g2.y ^ sg.w, dd));
        }
    } else if constexpr (TY == 22) {
        const uint32_t qh = w[2] & 255, sc = (w[2] >> 8) & 255;
        const float df = h2f(uint16_t(w[2] >> 16)) * 0.125f;
        const half2 d0 = dup_f(df * float(1 + 2 * (sc & 15))), d1 = dup_f(df * float(1 + 2 * (sc >> 4)));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t idx = ((w[0] >> (8 * j)) & 255) | ((qh << (8 - 2 * j)) & 0x300u);
            const uint4 sg = __ldg(t.sign8 + ((w[1] >> (8 * j)) & 255));
            const uint4 g = __ldg(t.g2s + idx);
            const half2 dd = j < 2 ? d0 : d1;
            out[j] = make_uint4(hmul(g.x ^ sg.x, dd), hmul(g.y ^ sg.y, dd), hmul(g.z ^ sg.z, dd), hmul(g.w ^ sg.w, dd));
        }
#define FP4P(v) hmul(__ldg(t.fp4 + (((v) & 15) | (((v) >> 4) & 0xf0))), dd)
    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py): out[0] values 0-7 (A's low nibbles), out[1] 8-15 (A's high), out[2], out[3] B's
        const half2 da = dup_f(nvfp4_scale(w[4])), db = dup_f(nvfp4_scale(w[4] >> 8));
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t a = w[2 * g], b = w[2 * g + 1];
            const half2 dd = g ? db : da;
            out[2 * g] = make_uint4(FP4P(a & 0xffff), FP4P(a >> 16), FP4P(b & 0xffff), FP4P(b >> 16));
            out[2 * g + 1] = make_uint4(FP4P((a >> 4) & 0x0fff), FP4P(a >> 20), FP4P((b >> 4) & 0x0fff), FP4P(b >> 20));
        }
#undef FP4P
    } else if constexpr (TY == 23 || TY == 20) {
        half2 dd;
        if constexpr (TY == 23) dd = dup_f(h2f(uint16_t(w[5])) * float(int(w[4]) - 32));
        else dd = dup_u16(w[4]);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t a = w[2 * g], b = w[2 * g + 1];
            out[g] = make_uint4(nib_pair(t, a & 0xffff, dd), nib_pair(t, a >> 16, dd), nib_pair(t, b & 0xffff, dd), nib_pair(t, b >> 16, dd));
            out[g + 2] = make_uint4(nib_pair(t, (a >> 4) & 0x0fff, dd), nib_pair(t, a >> 20, dd), nib_pair(t, (b >> 4) & 0x0fff, dd), nib_pair(t, b >> 20, dd));
        }
    } else {
        const half2 dd = dup_u16(w[2]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = (w[j >> 1] >> (16 * (j & 1))) & 0xffff;
            const uint2 g1 = __ldg(t.q20 + (q & 255)), g2 = __ldg(t.q20 + (q >> 8));
            out[j] = make_uint4(hmul(g1.x, dd), hmul(g1.y, dd), hmul(g2.x, dd), hmul(g2.y, dd));
        }
    }
}

__device__ __forceinline__ void fold_one(half2& h, float& a) {
    const float2 f = __half22float2(h);
    a += f.x + f.y;
    h = as_half2(0u);
}

template <int TY>
__launch_bounds__(kThreads)
__global__ void gemm_kernel(Args a) {
    extern __shared__ uint4 sm[];
    constexpr int TL = kTL, RL = kRL, NT = kThreads, WU = 2048;
    uint4* const sw = sm;            // [rows of the tile][KU]: always 2048 uint4
    uint4* const sx = sm + WU;       // [tokens of the tile][KU]
    const int tid = int(threadIdx.x), b = int(blockIdx.x);
    int e = 0;
    while (e + 1 < a.n && b >= a.first[e + 1]) ++e;
    const int m = a.mode[e], rel = b - a.first[e];
    const int RBL = 4 + m, KU = 16 >> m, TB = (8 >> m) * TL, NB = 128 << m;
    const int tt = rel % a.tiles[e], rr = rel / a.tiles[e];
    const int n00 = rr * NB, t00 = a.r0[e] + tt * TB, tend = a.r0[e] + a.rows[e], N = a.N;
    const int ni = tid & ((1 << RBL) - 1), ti = tid >> RBL;
    const int n0 = n00 + ni * RL, t0 = t00 + ti * TL;
    const bool act = t0 < tend && n0 < N;
    const uint8_t* const we = a.W + std::size_t(e) * a.expert_bytes;
    const uint4* const X = a.X;
    const int ldx_u = a.ldx_u;
    const std::size_t row_bytes = a.row_bytes;
    const Tabs t = a.t;
    half2 h[TL][RL];
    float s[TL][RL];
#pragma unroll
    for (int i = 0; i < TL; ++i)
#pragma unroll
        for (int r = 0; r < RL; ++r) { h[i][r] = as_half2(0u); s[i][r] = 0.0f; }
    const int nu = a.K / 8;
    for (int u0 = 0; u0 < nu; u0 += KU) {
        // the weights: a lane decodes 4 runs of 32 values (one per KU / 4 groups of a row's slice), 16 groups a lane
        const int RU = KU >> 2;
        {
            uint32_t rw[4][RunWords<TY>::n];
            int ro[4], qo[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int un = tid + j * NT;
                ro[j] = un / RU;
                qo[j] = un % RU;
                run_load<TY>(we + std::size_t(min(n00 + ro[j], N - 1)) * row_bytes, (u0 >> 2) + qo[j], rw[j]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (n00 + ro[j] < N) run_finish<TY>(t, rw[j], sw + ro[j] * KU + 4 * qo[j]);
        }
        const int xu = TB * KU;
        for (int j0 = tid; j0 < xu; j0 += 4 * NT) {
            uint4 v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int i = j0 + q * NT, r = i / KU, c = i % KU;
                if (i < xu) v[q] = __ldg(X + std::size_t(min(t00 + r, tend - 1)) * ldx_u + u0 + c);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q)
                if (j0 + q * NT < xu) sx[j0 + q * NT] = v[q];
        }
        __syncthreads();
        if (act) {
            // every lane walks the slice from its own position: a sum does not care, and the lanes' reads stay on different shared-memory
            // banks (all at one position they queue on one bank, half the speed)
            const int wb = ni * RL * KU, xb = ti * TL * KU;
            int cc = tid & (KU - 1);
            for (int c = 0; c < KU; ++c) {
                uint4 wr[RL], xr[TL];
#pragma unroll
                for (int r = 0; r < RL; ++r) wr[r] = sw[wb + r * KU + cc];
#pragma unroll
                for (int i = 0; i < TL; ++i) xr[i] = sx[xb + i * KU + cc];
                cc = (cc + 1) & (KU - 1);
#define P100_MACP(F) \
                _Pragma("unroll") \
                for (int i = 0; i < TL; ++i) \
                    _Pragma("unroll") \
                    for (int r = 0; r < RL; ++r) h[i][r] = __hfma2(as_half2(wr[r].F), as_half2(xr[i].F), h[i][r]);
                P100_MACP(x) P100_MACP(y) P100_MACP(z) P100_MACP(w)
#undef P100_MACP
                if (((c + 1) & (kFold - 1)) == 0 && c + 1 < KU) {
#pragma unroll
                    for (int i = 0; i < TL; ++i)
#pragma unroll
                        for (int r = 0; r < RL; ++r) fold_one(h[i][r], s[i][r]);
                }
            }
#pragma unroll
            for (int i = 0; i < TL; ++i)
#pragma unroll
                for (int r = 0; r < RL; ++r) fold_one(h[i][r], s[i][r]);
        }
        __syncthreads();
    }
    if (act) {
#pragma unroll
        for (int i = 0; i < TL; ++i)
            if (t0 + i < tend) {
                float* yr = a.Y + std::size_t(t0 + i) * a.ldy + n0;
#pragma unroll
                for (int r = 0; r < RL; ++r) yr[r] = s[i][r];
            }
    }
}

// ---- gemm4: a lane owns a row.  A warp reading 2-byte fields of 32 different rows fetches a cache line for each: 47 times the traffic of the
// weights, which on the card's L2 costs as much as the products.  So a lane first reads its row's next IQ block with aligned 16-byte loads into
// shared memory (the bytes in flight, one round trip) and decodes from there.
template <int TY> inline constexpr int kSU = TY == 12 ? 10 : TY == 13 ? 12 : TY == 7 ? 7 : TY == 8 ? 10 : TY == 23 ? 10 : TY == 18 || TY == 21 ? 8 : TY == 22 ? 7 : TY == 20 || TY == 2 || TY == 40 ? 6 : TY == 16 || TY == 29 ? 5 : 4;
template <int TY> inline constexpr int kRunsGr = TY == 20 || TY == 2 || TY == 42 || TY == 7 || TY == 8 || TY == 40 ? 4 : 8;
template <int TY> inline constexpr int kGrBytes = TY == 12 ? 144 : TY == 13 ? 176 : TY == 7 ? 96 : TY == 8 ? 136 : TY == 18 ? 98 : TY == 21 ? 110 : TY == 22 ? 82 : TY == 23 ? 136 : TY == 20 || TY == 2 || TY == 40 ? 72 : TY == 16 ? 66 : TY == 29 ? 56 : 36;

template <int TY>
__device__ __forceinline__ void run_load_s(const uint8_t* p, const int run, uint32_t (&w)[RunWords<TY>::n]) {
    if constexpr (TY == 12 || TY == 13) {   // q4x_patch.py: a group is one super-block
        kq_words<TY>(p, run & 7, w);
    } else if constexpr (TY == 7 || TY == 8) {   // four blocks
        kq_words<TY>(p + (run & 3) * (TY == 7 ? 24 : 34), 0, w);
    } else if constexpr (TY == 18) {
        const block_iq3_xxs* x = reinterpret_cast<const block_iq3_xxs*>(p);
        const int ib = run & 7;
        w[0] = P100_U16(x->qs, 8 * ib); w[1] = P100_U16(x->qs, 8 * ib + 2); w[2] = P100_U16(x->qs, 8 * ib + 4); w[3] = P100_U16(x->qs, 8 * ib + 6);
        const uint32_t a0 = P100_U16(x->qs, QK_K / 4 + 4 * ib), a1 = P100_U16(x->qs, QK_K / 4 + 4 * ib + 2);
        w[4] = a0 | (a1 << 16);
        w[5] = P100_U16(&x->d, 0);
    } else if constexpr (TY == 21) {
        const block_iq3_s* x = reinterpret_cast<const block_iq3_s*>(p);
        const int ib = run & 7;
        w[0] = P100_U16(x->qs, 8 * ib); w[1] = P100_U16(x->qs, 8 * ib + 2); w[2] = P100_U16(x->qs, 8 * ib + 4); w[3] = P100_U16(x->qs, 8 * ib + 6);
        const uint32_t s0 = P100_U16(x->signs, 4 * ib), s1 = P100_U16(x->signs, 4 * ib + 2);
        w[4] = s0 | (s1 << 16);
        const uint32_t qh = x->qh[ib], sc = (uint32_t(x->scales[ib / 2]) >> (4 * (ib & 1))) & 15, du = P100_U16(&x->d, 0);
        w[5] = qh | (sc << 8) | (du << 16);
    } else if constexpr (TY == 22) {
        const block_iq2_s* x = reinterpret_cast<const block_iq2_s*>(p);
        const int ib = run & 7;
        const uint32_t i0 = P100_U16(x->qs, 4 * ib), i1 = P100_U16(x->qs, 4 * ib + 2);
        const uint32_t s0 = P100_U16(x->qs, QK_K / 8 + 4 * ib), s1 = P100_U16(x->qs, QK_K / 8 + 4 * ib + 2);
        const uint32_t qh = x->qh[ib], sc = x->scales[ib], du = P100_U16(&x->d, 0);
        w[0] = i0 | (i1 << 16);
        w[1] = s0 | (s1 << 16);
        w[2] = qh | (sc << 8) | (du << 16);
    } else if constexpr (TY == 16) {   // IQ2_XXS: d, qs[32] (a run: 4 code bytes, then its signs and scale word)
        const block_iq2_xxs* x = reinterpret_cast<const block_iq2_xxs*>(p);
        const int ib = run & 7;
        w[0] = P100_U16(x->qs, 8 * ib) | (P100_U16(x->qs, 8 * ib + 2) << 16);
        w[1] = P100_U16(x->qs, 8 * ib + 4) | (P100_U16(x->qs, 8 * ib + 6) << 16);
        w[2] = P100_U16(&x->d, 0);
    } else if constexpr (TY == 29) {   // IQ1_M: qs[32], qh[16], scales[8] (the block's f16 scale in the top nibbles of the four words)
        const block_iq1_m* x = reinterpret_cast<const block_iq1_m*>(p);
        const int ib = run & 7;
        const uint32_t s0 = P100_U16(x->scales, 0), s1 = P100_U16(x->scales, 2), s2 = P100_U16(x->scales, 4), s3 = P100_U16(x->scales, 6);
        const uint32_t du = (s0 >> 12) | ((s1 >> 8) & 0x00f0u) | ((s2 >> 4) & 0x0f00u) | (s3 & 0xf000u);
        const uint32_t sc = (P100_U16(x->scales, 2 * (ib / 2)) >> (6 * (ib & 1))) & 63;
        w[0] = P100_U16(x->qs, 4 * ib) | (P100_U16(x->qs, 4 * ib + 2) << 16);
        w[1] = P100_U16(x->qh, 2 * ib) | (sc << 16);
        w[2] = du;
    } else if constexpr (TY == 23) {
        const block_iq4_xs* x = reinterpret_cast<const block_iq4_xs*>(p);
        const int ib = run & 7;
        w[0] = P100_U16(x->qs, 16 * ib) | (P100_U16(x->qs, 16 * ib + 2) << 16);
        w[1] = P100_U16(x->qs, 16 * ib + 4) | (P100_U16(x->qs, 16 * ib + 6) << 16);
        w[2] = P100_U16(x->qs, 16 * ib + 8) | (P100_U16(x->qs, 16 * ib + 10) << 16);
        w[3] = P100_U16(x->qs, 16 * ib + 12) | (P100_U16(x->qs, 16 * ib + 14) << 16);
        const uint32_t sl = x->scales_l[ib / 2], sh = x->scales_h, du = P100_U16(&x->d, 0);
        w[4] = ((sl >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
        w[5] = du;
    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py): a group is two blocks, four runs
        nvfp4_words(p + 36 * ((run & 3) >> 1), run & 1, w);
    } else if constexpr (TY == 20 || TY == 2) {
        const block_iq4_nl* x = reinterpret_cast<const block_iq4_nl*>(p) + (run & 3);
        w[0] = P100_U16(x->qs, 0) | (P100_U16(x->qs, 2) << 16);
        w[1] = P100_U16(x->qs, 4) | (P100_U16(x->qs, 6) << 16);
        w[2] = P100_U16(x->qs, 8) | (P100_U16(x->qs, 10) << 16);
        w[3] = P100_U16(x->qs, 12) | (P100_U16(x->qs, 14) << 16);
        w[4] = P100_U16(&x->d, 0);
    } else {
        const block_q2_0* x = reinterpret_cast<const block_q2_0*>(p) + ((run & 3) >> 1);
        const int part = (run & 1) * 8;
        w[0] = P100_U16(x->qs, part) | (P100_U16(x->qs, part + 2) << 16);
        w[1] = P100_U16(x->qs, part + 4) | (P100_U16(x->qs, part + 6) << 16);
        w[2] = P100_U16(&x->d, 0);
    }
}

// the tables in shared memory: a lookup in global memory is a round trip for each of a run's 12 lookups, the larger part of a small expert's time
template <int TY> struct TabU { static constexpr int n = 64; };
template <> struct TabU<18> { static constexpr int n = 256; };
template <> struct TabU<21> { static constexpr int n = 512; };
template <> struct TabU<22> { static constexpr int n = 256; };
template <> struct TabU<42> { static constexpr int n = 128; };
template <> struct TabU<16> { static constexpr int n = 384; };   // IQ2_XXS: its grid (256), the 7-bit signs (128)
template <> struct TabU<29> { static constexpr int n = 8; };     // IQ1_M: its 64 KB of grids stay in global memory
// q4x_patch.py: no tables, and none of their room either: Q8_0's 256-lane job kernels (down's mode 5) need exactly Pascal's 48 KiB
template <> struct TabU<12> { static constexpr int n = 0; };
template <> struct TabU<13> { static constexpr int n = 0; };
template <> struct TabU<7> { static constexpr int n = 0; };
template <> struct TabU<8> { static constexpr int n = 0; };

template <int TY>
__device__ __forceinline__ void stage_tabs(const Tabs& t, uint4* tab, const int tid, const int nt) {
    if constexpr (kQX<TY>) {   // q4x_patch.py: no tables
    } else if constexpr (TY == 18) {
        for (int i = tid; i < 128; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.g3xxs) + i);
        for (int i = tid; i < 128; i += nt) tab[128 + i] = __ldg(t.sign7 + i);
    } else if constexpr (TY == 21) {
        for (int i = tid; i < 256; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.g3s) + i);
        for (int i = tid; i < 256; i += nt) tab[256 + i] = __ldg(t.sign8 + i);
    } else if constexpr (TY == 22) {
        for (int i = tid; i < 256; i += nt) tab[i] = __ldg(t.sign8 + i);
    } else if constexpr (TY == 16) {
        for (int i = tid; i < 256; i += nt) tab[i] = __ldg(t.g2xxs + i);
        for (int i = tid; i < 128; i += nt) tab[256 + i] = __ldg(t.sign7 + i);
    } else if constexpr (TY == 29) {
    } else if constexpr (TY == 2) {
        for (int i = tid; i < 64; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.nib4) + i);
    } else if constexpr (TY == 40) {   // edge0_patch.py
        for (int i = tid; i < 64; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.fp4) + i);
    } else if constexpr (TY == 23 || TY == 20) {
        for (int i = tid; i < 64; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.nib) + i);
    } else {
        for (int i = tid; i < 128; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.q20) + i);
    }
}

__device__ __forceinline__ uint32_t nib_pair_s(const uint32_t* nib, const uint32_t v, const half2 dd) {
    return hmul(nib[(v & 15) | ((v >> 4) & 0xf0)], dd);
}
// run_finish with the tables at hand (IQ2_S's 16 KB grid stays in global memory, its 4 lookups in flight together)
template <int TY>
__device__ __forceinline__ void finish_s(const uint4* tab, const Tabs& t, const uint32_t (&w)[RunWords<TY>::n], uint4* out) {
    if constexpr (kQX<TY>) {
        kq_finish<TY>(w, out);   // q4x_patch.py
    } else if constexpr (TY == 18) {
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint4* sign = tab + 128;
        const uint32_t aux = w[4];
        const half2 dd = dup_f(h2f(uint16_t(w[5])) * float(1 + 2 * (aux >> 28)) * 0.25f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 sg = sign[(aux >> (7 * j)) & 127];
            const uint2 g1 = grid[w[j] & 255], g2 = grid[w[j] >> 8];
            out[j] = make_uint4(hmul(g1.x ^ sg.x, dd), hmul(g1.y ^ sg.y, dd), hmul(g2.x ^ sg.z, dd), hmul(g2.y ^ sg.w, dd));
        }
    } else if constexpr (TY == 21) {
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint4* sign = tab + 256;
        const uint32_t qh = w[5] & 255, sc = (w[5] >> 8) & 255;
        const half2 dd = dup_f(h2f(uint16_t(w[5] >> 16)) * float(1 + 2 * sc));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t i1 = (w[j] & 255) | ((qh << (8 - 2 * j)) & 256), i2 = (w[j] >> 8) | ((qh << (7 - 2 * j)) & 256);
            const uint4 sg = sign[(w[4] >> (8 * j)) & 255];
            const uint2 g1 = grid[i1], g2 = grid[i2];
            out[j] = make_uint4(hmul(g1.x ^ sg.x, dd), hmul(g1.y ^ sg.y, dd), hmul(g2.x ^ sg.z, dd), hmul(g2.y ^ sg.w, dd));
        }
    } else if constexpr (TY == 22) {
        const uint32_t qh = w[2] & 255, sc = (w[2] >> 8) & 255;
        const float df = h2f(uint16_t(w[2] >> 16)) * 0.125f;
        const half2 d0 = dup_f(df * float(1 + 2 * (sc & 15))), d1 = dup_f(df * float(1 + 2 * (sc >> 4)));
        uint4 gv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) gv[j] = __ldg(t.g2s + (((w[0] >> (8 * j)) & 255) | ((qh << (8 - 2 * j)) & 0x300u)));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 sg = tab[(w[1] >> (8 * j)) & 255];
            const uint4 g = gv[j];
            const half2 dd = j < 2 ? d0 : d1;
            out[j] = make_uint4(hmul(g.x ^ sg.x, dd), hmul(g.y ^ sg.y, dd), hmul(g.z ^ sg.z, dd), hmul(g.w ^ sg.w, dd));
        }
    } else if constexpr (TY == 16) {   // IQ2_XXS: d (0.5 + s) / 4 = d (1 + 2 s) / 8
        const uint4* grid = tab;
        const uint4* sign = tab + 256;
        const half2 dd = dup_f(h2f(uint16_t(w[2])) * float(1 + 2 * (w[1] >> 28)) * 0.125f);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 sg = sign[(w[1] >> (7 * j)) & 127];
            const uint4 g = grid[(w[0] >> (8 * j)) & 255];
            out[j] = make_uint4(hmul(g.x ^ sg.x, dd), hmul(g.y ^ sg.y, dd), hmul(g.z ^ sg.z, dd), hmul(g.w ^ sg.w, dd));
        }
    } else if constexpr (TY == 29) {   // IQ1_M: d (1 + 2 s) (grid value +- 0.125), a scale each 16 values
        const uint32_t sc = w[1] >> 16;
        const float df = h2f(uint16_t(w[2]));
        const half2 d0 = dup_f(df * float(1 + 2 * (sc & 7))), d1 = dup_f(df * float(1 + 2 * ((sc >> 3) & 7)));
        uint4 gv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t qh = (w[1] >> (8 * (j >> 1))) & 255;
            const uint32_t idx = ((w[0] >> (8 * j)) & 255) | ((qh << ((j & 1) ? 4 : 8)) & 0x700u);
            gv[j] = __ldg(((qh & ((j & 1) ? 0x80u : 0x08u)) ? t.g1m : t.g1p) + idx);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const half2 dd = j < 2 ? d0 : d1;
            out[j] = make_uint4(hmul(gv[j].x, dd), hmul(gv[j].y, dd), hmul(gv[j].z, dd), hmul(gv[j].w, dd));
        }
#define FP4P(v) nib_pair_s(reinterpret_cast<const uint32_t*>(tab), (v), dd)
    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py): out[0] values 0-7 (A's low nibbles), out[1] 8-15 (A's high), out[2], out[3] B's
        const half2 da = dup_f(nvfp4_scale(w[4])), db = dup_f(nvfp4_scale(w[4] >> 8));
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t a = w[2 * g], b = w[2 * g + 1];
            const half2 dd = g ? db : da;
            out[2 * g] = make_uint4(FP4P(a & 0xffff), FP4P(a >> 16), FP4P(b & 0xffff), FP4P(b >> 16));
            out[2 * g + 1] = make_uint4(FP4P((a >> 4) & 0x0fff), FP4P(a >> 20), FP4P((b >> 4) & 0x0fff), FP4P(b >> 20));
        }
#undef FP4P
    } else if constexpr (TY == 23 || TY == 20 || TY == 2) {
        const uint32_t* nib = reinterpret_cast<const uint32_t*>(tab);
        half2 dd;
        if constexpr (TY == 23) dd = dup_f(h2f(uint16_t(w[5])) * float(int(w[4]) - 32));
        else dd = dup_u16(w[4]);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t a = w[2 * g], b = w[2 * g + 1];
            out[g] = make_uint4(nib_pair_s(nib, a & 0xffff, dd), nib_pair_s(nib, a >> 16, dd), nib_pair_s(nib, b & 0xffff, dd), nib_pair_s(nib, b >> 16, dd));
            out[g + 2] = make_uint4(nib_pair_s(nib, (a >> 4) & 0x0fff, dd), nib_pair_s(nib, a >> 20, dd), nib_pair_s(nib, (b >> 4) & 0x0fff, dd), nib_pair_s(nib, b >> 20, dd));
        }
    } else {
        const uint2* q20 = reinterpret_cast<const uint2*>(tab);
        const half2 dd = dup_u16(w[2]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q = (w[j >> 1] >> (16 * (j & 1))) & 0xffff;
            const uint2 g1 = q20[q & 255], g2 = q20[q >> 8];
            out[j] = make_uint4(hmul(g1.x, dd), hmul(g1.y, dd), hmul(g2.x, dd), hmul(g2.y, dd));
        }
    }
}

constexpr int k4NB = 128, k4KU = 8, k4TB = 48;

template <int TY>
__launch_bounds__(kThreads)
__global__ void gemm4_kernel(Args a) {
    extern __shared__ uint4 sm[];
    constexpr int TL = kTL, RL = kRL, NT = kThreads, KU = k4KU, TB = k4TB, SU = kSU<TY>;
    uint4* const sw = sm;                 // [k4NB rows][KU]
    uint4* const sx = sm + k4NB * KU;     // [TB tokens][KU]
    uint4* const sg = sx + TB * KU;       // [k4NB rows][SU]: a row's IQ block
    uint4* const tab = sg + k4NB * SU;    // the tables
    const int tid = int(threadIdx.x), b = int(blockIdx.x);
    int e = 0;
    while (e + 1 < a.n && b >= a.first[e + 1]) ++e;
    const int rel = b - a.first[e];
    const int tt = rel % a.tiles[e], rr = rel / a.tiles[e];
    const int n00 = rr * k4NB, t00 = a.r0[e] + tt * TB, tend = a.r0[e] + a.rows[e], N = a.N;
    const int ni = tid & 15, ti = tid >> 4;
    const int n0 = n00 + ni * RL, t0 = t00 + ti * TL;
    const bool act = t0 < tend && n0 < N;
    const int my_row = n00 + tid;
    const uint8_t* const we = a.W + std::size_t(e) * a.expert_bytes;
    const uint4* const X = a.X;
    const int ldx_u = a.ldx_u;
    const std::size_t row_bytes = a.row_bytes;
    const Tabs t = a.t;
    half2 h[TL][RL];
    float s[TL][RL];
#pragma unroll
    for (int i = 0; i < TL; ++i)
#pragma unroll
        for (int r = 0; r < RL; ++r) { h[i][r] = as_half2(0u); s[i][r] = 0.0f; }
    stage_tabs<TY>(t, tab, tid, NT);
    __syncthreads();
    const int nu = a.K / 8;
    int off = 0;
    for (int u0 = 0; u0 < nu; u0 += KU) {
        const int run0 = u0 >> 2;
        if ((run0 & (kRunsGr<TY> - 1)) == 0 && my_row < N) {
            const uintptr_t ad = reinterpret_cast<uintptr_t>(we + std::size_t(my_row) * row_bytes + std::size_t(run0 / kRunsGr<TY>) * kGrBytes<TY>);
            const uint4* al = reinterpret_cast<const uint4*>(ad & ~uintptr_t(15));
            off = int(ad & 15);
            uint4 v[SU];
#pragma unroll
            for (int q = 0; q < SU; ++q) v[q] = __ldg(al + q);
#pragma unroll
            for (int q = 0; q < SU; ++q) sg[tid * SU + q] = v[q];
        }
        if (my_row < N) {
            const uint8_t* gp = reinterpret_cast<const uint8_t*>(sg + tid * SU) + off;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t rw[RunWords<TY>::n];
                run_load_s<TY>(gp, run0 + j, rw);
                uint4 o[4];
                finish_s<TY>(tab, t, rw, o);
#pragma unroll
                for (int q = 0; q < 4; ++q) sw[tid * KU + 4 * j + q] = o[q];
            }
        }
        for (int j0 = tid; j0 < TB * KU; j0 += 4 * NT) {
            uint4 v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int i = j0 + q * NT, r = i / KU, c = i % KU;
                if (i < TB * KU && t00 + r < tend) v[q] = __ldg(X + std::size_t(t00 + r) * ldx_u + u0 + c);
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int i = j0 + q * NT, r = i / KU;
                if (i < TB * KU && t00 + r < tend) sx[i] = v[q];
            }
        }
        __syncthreads();
        if (act) {
            const int wb = ni * RL * KU, xb = ti * TL * KU;
            int cc = tid & (KU - 1);
            for (int c = 0; c < KU; ++c) {
                uint4 wr[RL], xr[TL];
#pragma unroll
                for (int r = 0; r < RL; ++r) wr[r] = sw[wb + r * KU + cc];
#pragma unroll
                for (int i = 0; i < TL; ++i) xr[i] = sx[xb + i * KU + cc];
                cc = (cc + 1) & (KU - 1);
#define P100_MACP(F) \
                _Pragma("unroll") \
                for (int i = 0; i < TL; ++i) \
                    _Pragma("unroll") \
                    for (int r = 0; r < RL; ++r) h[i][r] = __hfma2(as_half2(wr[r].F), as_half2(xr[i].F), h[i][r]);
                P100_MACP(x) P100_MACP(y) P100_MACP(z) P100_MACP(w)
#undef P100_MACP
            }
#pragma unroll
            for (int i = 0; i < TL; ++i)
#pragma unroll
                for (int r = 0; r < RL; ++r) fold_one(h[i][r], s[i][r]);
        }
        __syncthreads();
    }
    if (act) {
#pragma unroll
        for (int i = 0; i < TL; ++i)
            if (t0 + i < tend) {
                float* yr = a.Y + std::size_t(t0 + i) * a.ldy + n0;
#pragma unroll
                for (int r = 0; r < RL; ++r) yr[r] = s[i][r];
            }
    }
}

inline bool type_ok(int ty) { return ty == 18 || ty == 21 || ty == 22 || ty == 23 || ty == 20 || ty == 42; }

inline const Tabs& tabs(cudaStream_t st) {
    static Tabs per[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    Tabs& t = per[dev & 15];
    if (t.g3xxs == nullptr) {
        cudaMalloc(&t.g3xxs, 256 * 8); cudaMalloc(&t.g3s, 512 * 8); cudaMalloc(&t.g2s, 1024 * 16); cudaMalloc(&t.sign8, 256 * 16);
        cudaMalloc(&t.sign7, 128 * 16); cudaMalloc(&t.nib, 256 * 4); cudaMalloc(&t.nib4, 256 * 4); cudaMalloc(&t.q20, 256 * 8); cudaMalloc(&t.s12, 16 * 4); cudaMalloc(&t.s6, 64 * 4);
        cudaMalloc(&t.g2xxs, 256 * 16); cudaMalloc(&t.g1p, 2048 * 16); cudaMalloc(&t.g1m, 2048 * 16); cudaMalloc(&t.fp4, 256 * 4);
        tabs_kernel<<<8, 256, 0, st>>>(t);
    }
    return t;
}

// Rows [bounds[e], bounds[e + 1]) of X and Y belong to expert e; W holds n_experts matrices of n_out rows of K values.
inline bool launch(int ty, const void* W, std::size_t expert_bytes, std::size_t row_bytes, int n_out, int K, const void* X, int ldx,
                   const int32_t* bounds, int n_experts, float* Y, int ldy, void* stream) {
    if (!type_ok(ty) || n_experts <= 0 || n_experts > kMaxExperts || n_out % kRL != 0 || K % 128 != 0 || ldx % 8 != 0) return false;
    cudaStream_t st = (cudaStream_t) stream;
    Args a{};
    a.t = tabs(st);
    a.W = (const uint8_t*) W;
    a.X = (const uint4*) X;
    a.Y = Y;
    a.expert_bytes = expert_bytes;
    a.row_bytes = row_bytes;
    a.ldx_u = ldx / 8;
    a.ldy = ldy;
    a.N = n_out;
    a.K = K;
    a.n = n_experts;
    int blocks = 0;
    for (int e = 0; e < n_experts; ++e) {
        const int T = bounds[e + 1] - bounds[e];
        a.r0[e] = bounds[e];
        a.rows[e] = T;
        a.first[e] = blocks;
        const int m = T <= 2 * kTL ? 2 : T <= 4 * kTL ? 1 : 0;
        const int TB = (8 >> m) * kTL, NB = 128 << m;
        a.mode[e] = m;
        a.tiles[e] = std::max(1, (T + TB - 1) / TB);
        if (T > 0) blocks += a.tiles[e] * ((n_out + NB - 1) / NB);
    }
    a.first[n_experts] = blocks;
    // an expert without rows gets no block: `first` of the next equals its own, and the kernel's search skips it
    if (blocks == 0) return true;
    constexpr std::size_t smem = 48 * 1024;
    switch (ty) {
        case 18: gemm_kernel<18><<<blocks, kThreads, smem, st>>>(a); break;
        case 21: gemm_kernel<21><<<blocks, kThreads, smem, st>>>(a); break;
        case 22: gemm_kernel<22><<<blocks, kThreads, smem, st>>>(a); break;
        case 23: gemm_kernel<23><<<blocks, kThreads, smem, st>>>(a); break;
        case 20: gemm_kernel<20><<<blocks, kThreads, smem, st>>>(a); break;
        default: gemm_kernel<42><<<blocks, kThreads, smem, st>>>(a); break;
    }
    return cudaGetLastError() == cudaSuccess;
}

inline bool launch4(int ty, const void* W, std::size_t expert_bytes, std::size_t row_bytes, int n_out, int K, const void* X, int ldx,
                    const int32_t* bounds, int n_experts, float* Y, int ldy, void* stream) {
    if (!type_ok(ty) || n_experts <= 0 || n_experts > kMaxExperts || n_out % kRL != 0 || K % 128 != 0 || ldx % 8 != 0) return false;
    cudaStream_t st = (cudaStream_t) stream;
    Args a{};
    a.t = tabs(st);
    a.W = (const uint8_t*) W;
    a.X = (const uint4*) X;
    a.Y = Y;
    a.expert_bytes = expert_bytes;
    a.row_bytes = row_bytes;
    a.ldx_u = ldx / 8;
    a.ldy = ldy;
    a.N = n_out;
    a.K = K;
    a.n = n_experts;
    int blocks = 0;
    for (int e = 0; e < n_experts; ++e) {
        const int T = bounds[e + 1] - bounds[e];
        a.r0[e] = bounds[e];
        a.rows[e] = T;
        a.first[e] = blocks;
        a.tiles[e] = std::max(1, (T + k4TB - 1) / k4TB);
        if (T > 0) blocks += a.tiles[e] * ((n_out + k4NB - 1) / k4NB);
    }
    a.first[n_experts] = blocks;
    if (blocks == 0) return true;
    constexpr std::size_t base = (std::size_t(k4NB) * k4KU + std::size_t(k4TB) * k4KU) * 16;
    switch (ty) {
        case 18: gemm4_kernel<18><<<blocks, kThreads, base + (k4NB * kSU<18> + TabU<18>::n) * 16, st>>>(a); break;
        case 21: gemm4_kernel<21><<<blocks, kThreads, base + (k4NB * kSU<21> + TabU<21>::n) * 16, st>>>(a); break;
        case 22: gemm4_kernel<22><<<blocks, kThreads, base + (k4NB * kSU<22> + TabU<22>::n) * 16, st>>>(a); break;
        case 23: gemm4_kernel<23><<<blocks, kThreads, base + (k4NB * kSU<23> + TabU<23>::n) * 16, st>>>(a); break;
        case 20: gemm4_kernel<20><<<blocks, kThreads, base + (k4NB * kSU<20> + TabU<20>::n) * 16, st>>>(a); break;
        default: gemm4_kernel<42><<<blocks, kThreads, base + (k4NB * kSU<42> + TabU<42>::n) * 16, st>>>(a); break;
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels::p100f::iqg
