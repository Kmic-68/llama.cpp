// p100_moe_rows.cuh: a decode window's routed experts on GP100: few tokens per expert, so a lane owns a row and nothing waits in shared memory.
// gemm4 (p100_iq_gemm.cuh) is built for a prompt chunk: its 48 x 128 tile fills shared memory, so an SM holds one block and a block decodes its weights
// whatever the tokens: 290 us a wave. Here shared memory holds only the tables, so an SM holds several blocks and the time goes with the work.
// A group of an IQ row is loaded once with 16-byte loads and shifted into place, so its fields are registers at known positions.
#pragma once

namespace fnx {
namespace {

using namespace strata::kernels::p100f::iqg;

template <int TY> inline constexpr int kGrW = (kGrBytes<TY> + 3) / 4;
#ifndef FNX_ROW_G
#define FNX_ROW_G 2
#endif
template <int TY> inline constexpr int kRowG = TY == 20 || TY == 2 || TY == 7 || TY == 8 || TY == 40 ? 1 : FNX_ROW_G;       // lanes per row: a gate|up row is long, a down row short
inline constexpr int kRowT = 256;

template <int P, int NW>
__device__ __forceinline__ uint32_t g16(const uint32_t (&W)[NW]) {
    static_assert((P & 1) == 0 && (P >> 2) < NW, "a 16-bit field inside the group");
    if constexpr ((P & 2) != 0) return W[P >> 2] >> 16;
    else return W[P >> 2] & 0xffffu;
}
template <int P, int NW>
__device__ __forceinline__ uint32_t g8(const uint32_t (&W)[NW]) {
    static_assert((P >> 2) < NW, "a byte inside the group");
    return (W[P >> 2] >> ((P & 3) * 8)) & 255u;
}

// run IB of the group: the words of run_load_s, from the normalized group
template <int TY, int IB>
__device__ __forceinline__ void run_words(const uint32_t (&W)[kGrW<TY>], uint32_t (&w)[RunWords<TY>::n]) {
    if constexpr (TY == 12 || TY == 13) {   // q4x_patch.py, Q4_K / Q5_K: d, dmin, scales[12], (Q5_K: qh[32]), qs[128]; run IB = sub-block IB
        constexpr int C = IB >> 1, SH = 4 * (IB & 1), QS = TY == 12 ? 16 : 48;
#define P100_KQW(k) \
        w[k] = ((g16<QS + 32 * C + 4 * k>(W) | (g16<QS + 32 * C + 4 * k + 2>(W) << 16)) >> SH) & 0x0F0F0F0Fu; \
        if constexpr (TY == 13) w[k] |= (((g16<16 + 4 * k>(W) | (g16<18 + 4 * k>(W) << 16)) >> IB) & 0x01010101u) << 4;
        P100_KQW(0) P100_KQW(1) P100_KQW(2) P100_KQW(3) P100_KQW(4) P100_KQW(5) P100_KQW(6) P100_KQW(7)
#undef P100_KQW
        w[8] = g16<0>(W) | (g16<2>(W) << 16);
        if constexpr (IB < 4) w[9] = (g8<4 + IB>(W) & 63u) | ((g8<8 + IB>(W) & 63u) << 8);
        else w[9] = ((g8<8 + IB>(W) & 15u) | ((g8<IB>(W) >> 6) << 4)) | (((g8<8 + IB>(W) >> 4) | ((g8<4 + IB>(W) >> 6) << 4)) << 8);
    } else if constexpr (TY == 7) {   // Q5_1: blocks of d, m, qh, qs[16]
        constexpr int B = 24 * IB;
        const uint32_t qh = g16<B + 4>(W) | (g16<B + 6>(W) << 16);
#define P100_K51(k) { \
            const uint32_t q = g16<B + 8 + 4 * k>(W) | (g16<B + 10 + 4 * k>(W) << 16); \
            w[k] = (q & 0x0F0F0F0Fu) | kq_spread4((qh >> (4 * k)) & 15u); \
            w[k + 4] = ((q >> 4) & 0x0F0F0F0Fu) | kq_spread4((qh >> (16 + 4 * k)) & 15u); }
        P100_K51(0) P100_K51(1) P100_K51(2) P100_K51(3)
#undef P100_K51
        w[8] = g16<B>(W) | (g16<B + 2>(W) << 16);
    } else if constexpr (TY == 8) {   // Q8_0: blocks of d, qs[32]
        constexpr int B = 34 * IB;
#define P100_K80(k) w[k] = g16<B + 2 + 4 * k>(W) | (g16<B + 4 + 4 * k>(W) << 16);
        P100_K80(0) P100_K80(1) P100_K80(2) P100_K80(3) P100_K80(4) P100_K80(5) P100_K80(6) P100_K80(7)
#undef P100_K80
        w[8] = g16<B>(W);
    } else if constexpr (TY == 18) {   // IQ3_XXS: d, qs[96] (codes, then the signs and scale word of each run)
        w[0] = g16<2 + 8 * IB>(W); w[1] = g16<4 + 8 * IB>(W); w[2] = g16<6 + 8 * IB>(W); w[3] = g16<8 + 8 * IB>(W);
        w[4] = g16<66 + 4 * IB>(W) | (g16<68 + 4 * IB>(W) << 16);
        w[5] = g16<0>(W);
    } else if constexpr (TY == 21) {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
        w[0] = g16<2 + 8 * IB>(W); w[1] = g16<4 + 8 * IB>(W); w[2] = g16<6 + 8 * IB>(W); w[3] = g16<8 + 8 * IB>(W);
        w[4] = g16<74 + 4 * IB>(W) | (g16<76 + 4 * IB>(W) << 16);
        const uint32_t sc = (g8<106 + IB / 2>(W) >> (4 * (IB & 1))) & 15u;
        w[5] = g8<66 + IB>(W) | (sc << 8) | (g16<0>(W) << 16);
    } else if constexpr (TY == 22) {   // IQ2_S: d, qs[64] (codes, then signs), qh[8], scales[8]
        w[0] = g16<2 + 4 * IB>(W) | (g16<4 + 4 * IB>(W) << 16);
        w[1] = g16<34 + 4 * IB>(W) | (g16<36 + 4 * IB>(W) << 16);
        w[2] = g8<66 + IB>(W) | (g8<74 + IB>(W) << 8) | (g16<0>(W) << 16);
    } else if constexpr (TY == 16) {   // IQ2_XXS: d, qs[64] (a run: 4 code bytes, then its signs and scale word)
        w[0] = g16<2 + 8 * IB>(W) | (g16<4 + 8 * IB>(W) << 16);
        w[1] = g16<6 + 8 * IB>(W) | (g16<8 + 8 * IB>(W) << 16);
        w[2] = g16<0>(W);
    } else if constexpr (TY == 29) {   // IQ1_M: qs[32], qh[16], scales[8]
        w[0] = g16<4 * IB>(W) | (g16<4 * IB + 2>(W) << 16);
        w[1] = g16<32 + 2 * IB>(W) | (((g16<48 + 2 * (IB / 2)>(W) >> (6 * (IB & 1))) & 63u) << 16);
        w[2] = (g16<48>(W) >> 12) | ((g16<50>(W) >> 8) & 0x00f0u) | ((g16<52>(W) >> 4) & 0x0f00u) | (g16<54>(W) & 0xf000u);
    } else if constexpr (TY == 23) {   // IQ4_XS: d, scales_h, scales_l[4], qs[128]
        w[0] = g16<8 + 16 * IB>(W) | (g16<10 + 16 * IB>(W) << 16);
        w[1] = g16<12 + 16 * IB>(W) | (g16<14 + 16 * IB>(W) << 16);
        w[2] = g16<16 + 16 * IB>(W) | (g16<18 + 16 * IB>(W) << 16);
        w[3] = g16<20 + 16 * IB>(W) | (g16<22 + 16 * IB>(W) << 16);
        w[4] = ((g8<4 + IB / 2>(W) >> (4 * (IB & 1))) & 15u) | (((g16<2>(W) >> (2 * IB)) & 3u) << 4);
        w[5] = g16<0>(W);
    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py): block IB >> 1, half IB & 1 (p100_iq_gemm.cuh's nvfp4_words)
        constexpr int B = 36 * (IB >> 1), Q = B + 4 + 16 * (IB & 1);
        w[0] = g16<Q>(W) | (g16<Q + 2>(W) << 16);
        w[1] = g16<Q + 4>(W) | (g16<Q + 6>(W) << 16);
        w[2] = g16<Q + 8>(W) | (g16<Q + 10>(W) << 16);
        w[3] = g16<Q + 12>(W) | (g16<Q + 14>(W) << 16);
        w[4] = g16<B + 2 * (IB & 1)>(W);
    } else {   // IQ4_NL, Q4_0: blocks of d, qs[16]
        constexpr int B = 18 * IB;
        w[0] = g16<B + 2>(W) | (g16<B + 4>(W) << 16);
        w[1] = g16<B + 6>(W) | (g16<B + 8>(W) << 16);
        w[2] = g16<B + 10>(W) | (g16<B + 12>(W) << 16);
        w[3] = g16<B + 14>(W) | (g16<B + 16>(W) << 16);
        w[4] = g16<B + 0>(W);
    }
}

// the group at p, at most `avail` bytes of it in its row, as words from its first byte (p is even)
template <int TY>
__device__ __forceinline__ void group_load(const uint8_t* p, const int avail, uint32_t (&W)[kGrW<TY>], const bool cg) {
    constexpr int SU = kSU<TY>, NV = SU * 4;
    const uintptr_t ad = reinterpret_cast<uintptr_t>(p);
    const uint4* al = reinterpret_cast<const uint4*>(ad & ~uintptr_t(15));
    const int off = int(ad & 15), bytes = off + min(kGrBytes<TY>, avail);
    uint32_t V[NV];
#pragma unroll
    for (int q = 0; q < SU; ++q) {
        const uint4 v = q * 16 < bytes ? (cg ? __ldcg(al + q) : __ldg(al + q)) : make_uint4(0u, 0u, 0u, 0u);
        V[4 * q] = v.x; V[4 * q + 1] = v.y; V[4 * q + 2] = v.z; V[4 * q + 3] = v.w;
    }
    const bool s8 = (off & 8) != 0, s4 = (off & 4) != 0, s2 = (off & 2) != 0;
#pragma unroll
    for (int i = 0; i < NV - 2; ++i) V[i] = s8 ? V[i + 2] : V[i];
#pragma unroll
    for (int i = 0; i < NV - 1; ++i) V[i] = s4 ? V[i + 1] : V[i];
#pragma unroll
    for (int i = 0; i < kGrW<TY>; ++i) W[i] = s2 ? (V[i] >> 16) | (V[i + 1] << 16) : V[i];
}

struct RowArgs {
    Tabs t;
    const char* wa;
    size_t sa, row_bytes;
    const char* const* ep;
    const Job* jobs;
    const int* njobs;
    const uint4* X;
    float* Y;
    int ldx_u, N, K;
    int phase;   // 0 every job; 1 the jobs not marked (w >= 0); 2 the marked ones (w = ~expert)
    int ldcg;    // FNX_ROWS_LDCG: the weights read past L1 (ld.global.cg)
    // split 2 (perexp_patch.py), phase 2 only: the copier's words for this part, the list's number, the list (cache_list), the layer
    const unsigned* wdone;
    const unsigned* wwant;
    const int* wlist;
    int wlayer;
    unsigned* werr;
};

// one run of the row: its 32 weights, then each token's sum. On the IQ gate|up types the products stay on the grid's small integers and the
// run's scale goes once on each sum: 16 multiplies a run fewer, and an FP16 sum of 16 products keeps its bits as well as the scaled ones
template <int TY, int TT, int IB>
__device__ __forceinline__ void row_run(const uint32_t (&W)[kGrW<TY>], const uint4* tab, const Tabs& t, const int run, const uint4* const (&xp)[TT], const int n,
                                        float (&s)[TT]) {
    uint32_t rw[RunWords<TY>::n];
    run_words<TY, IB>(W, rw);
    if constexpr (TY == 18 || TY == 21 || TY == 22 || TY == 16 || TY == 29) {
        uint4 o[4];
        float d0, d1;
        if constexpr (TY == 18) {
            const uint2* grid = reinterpret_cast<const uint2*>(tab);
            const uint4* sign = tab + 128;
            const uint32_t aux = rw[4];
            d0 = d1 = h2f(uint16_t(rw[5])) * float(1 + 2 * (aux >> 28)) * 0.25f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 sg = sign[(aux >> (7 * j)) & 127];
                const uint2 g1 = grid[rw[j] & 255], g2 = grid[rw[j] >> 8];
                o[j] = make_uint4(g1.x ^ sg.x, g1.y ^ sg.y, g2.x ^ sg.z, g2.y ^ sg.w);
            }
        } else if constexpr (TY == 21) {
            const uint2* grid = reinterpret_cast<const uint2*>(tab);
            const uint4* sign = tab + 256;
            const uint32_t qh = rw[5] & 255, sc = (rw[5] >> 8) & 255;
            d0 = d1 = h2f(uint16_t(rw[5] >> 16)) * float(1 + 2 * sc);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t i1 = (rw[j] & 255) | ((qh << (8 - 2 * j)) & 256), i2 = (rw[j] >> 8) | ((qh << (7 - 2 * j)) & 256);
                const uint4 sg = sign[(rw[4] >> (8 * j)) & 255];
                const uint2 g1 = grid[i1], g2 = grid[i2];
                o[j] = make_uint4(g1.x ^ sg.x, g1.y ^ sg.y, g2.x ^ sg.z, g2.y ^ sg.w);
            }
        } else if constexpr (TY == 16) {   // IQ2_XXS: the grid rows and the 7-bit signs staged, one scale a run
            const uint4* grid = tab;
            const uint4* sign = tab + 256;
            d0 = d1 = h2f(uint16_t(rw[2])) * float(1 + 2 * (rw[1] >> 28)) * 0.125f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 sg = sign[(rw[1] >> (7 * j)) & 127];
                const uint4 g = grid[(rw[0] >> (8 * j)) & 255];
                o[j] = make_uint4(g.x ^ sg.x, g.y ^ sg.y, g.z ^ sg.z, g.w ^ sg.w);
            }
        } else if constexpr (TY == 29) {   // IQ1_M: grid value +- 0.125 from global memory, a scale each 16 values
            const uint32_t sc = rw[1] >> 16;
            const float df = h2f(uint16_t(rw[2]));
            d0 = df * float(1 + 2 * (sc & 7));
            d1 = df * float(1 + 2 * ((sc >> 3) & 7));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t qh = (rw[1] >> (8 * (j >> 1))) & 255;
                const uint32_t idx = ((rw[0] >> (8 * j)) & 255) | ((qh << ((j & 1) ? 4 : 8)) & 0x700u);
                o[j] = __ldg(((qh & ((j & 1) ? 0x80u : 0x08u)) ? t.g1m : t.g1p) + idx);
            }
        } else {
            const uint32_t qh = rw[2] & 255, sc = (rw[2] >> 8) & 255;
            const float df = h2f(uint16_t(rw[2] >> 16)) * 0.125f;
            d0 = df * float(1 + 2 * (sc & 15));
            d1 = df * float(1 + 2 * (sc >> 4));
            uint4 gv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) gv[j] = __ldg(t.g2s + (((rw[0] >> (8 * j)) & 255) | ((qh << (8 - 2 * j)) & 0x300u)));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint4 sg = tab[(rw[1] >> (8 * j)) & 255];
                o[j] = make_uint4(gv[j].x ^ sg.x, gv[j].y ^ sg.y, gv[j].z ^ sg.z, gv[j].w ^ sg.w);
            }
        }
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k >= n) continue;
            half2 h = as_half2(0u);
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const uint4 x = __ldg(xp[k] + 4 * run + q);
                h = __hfma2(as_half2(o[q].x), as_half2(x.x), h);
                h = __hfma2(as_half2(o[q].y), as_half2(x.y), h);
                h = __hfma2(as_half2(o[q].z), as_half2(x.z), h);
                h = __hfma2(as_half2(o[q].w), as_half2(x.w), h);
            }
            float2 f = __half22float2(h);
            float a = f.x + f.y;
            if constexpr (TY == 22 || TY == 29) {
                a *= d0;
                h = as_half2(0u);
            }
#pragma unroll
            for (int q = 2; q < 4; ++q) {
                const uint4 x = __ldg(xp[k] + 4 * run + q);
                h = __hfma2(as_half2(o[q].x), as_half2(x.x), h);
                h = __hfma2(as_half2(o[q].y), as_half2(x.y), h);
                h = __hfma2(as_half2(o[q].z), as_half2(x.z), h);
                h = __hfma2(as_half2(o[q].w), as_half2(x.w), h);
            }
            f = __half22float2(h);
            if constexpr (TY == 22 || TY == 29) s[k] += a + d1 * (f.x + f.y);
            else s[k] += d0 * (f.x + f.y);
        }
    } else {
        uint4 o[4];
        finish_s<TY>(tab, t, rw, o);
#pragma unroll
        for (int k = 0; k < TT; ++k) {
            if (k >= n) continue;
            half2 h = as_half2(0u);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint4 x = __ldg(xp[k] + 4 * run + q);
                h = __hfma2(as_half2(o[q].x), as_half2(x.x), h);
                h = __hfma2(as_half2(o[q].y), as_half2(x.y), h);
                h = __hfma2(as_half2(o[q].z), as_half2(x.z), h);
                h = __hfma2(as_half2(o[q].w), as_half2(x.w), h);
            }
            fold_one(h, s[k]);
        }
    }
}

template <int TY, int TT, int BT, int G>
__global__ void __launch_bounds__(BT) k_moe_rows(RowArgs a) {
    constexpr int RUNS = kRunsGr<TY>, GB = kGrBytes<TY>;
    extern __shared__ uint4 tab[];
    const int tid = int(threadIdx.x), ji = int(blockIdx.y);
    if (ji >= *a.njobs) return;
    const Job* const J = a.jobs + ji;
    int we_i = __ldg(&J->w);
    if (a.phase == 1 && we_i < 0) return;
    if (a.phase == 2) {
        if (we_i >= 0) return;
        we_i = ~we_i;
        if (a.wdone && tid == 0) {   // split 2: this expert's part, by its place in the list (the stage_tabs barrier holds the block)
            const int n = min(a.wlist[0], MAXT * TOPK), g = a.wlayer * NEXP + we_i;
            int k = 0;
            while (k < n && a.wlist[1 + k] != g) ++k;
            const unsigned want = *(volatile const unsigned*) a.wwant;
            if (k < n && want) {
                const long long t0 = clock64();
                while ((int) (*(volatile const unsigned*) &a.wdone[k] - want) < 0)
                    if (clock64() - t0 > 4000000000LL) {
                        atomicOr(a.werr, 1u);
                        break;
                    }
                __threadfence();
            }
        }
    }
    const Tabs t = a.t;
    stage_tabs<TY>(t, tab, tid, BT);
    __syncthreads();
    const int n = min(TT, __ldg(&J->n));
    const int row = int(blockIdx.x) * (BT / G) + tid / G, part = tid % G, N = a.N;
    float s[TT];
#pragma unroll
    for (int k = 0; k < TT; ++k) s[k] = 0.0f;
    if (row < N) {
        const uint4* xp[TT];
#pragma unroll
        for (int k = 0; k < TT; ++k) xp[k] = a.X + (k < n ? size_t(__ldg(&J->y[k])) * a.ldx_u : size_t(0));
        const char* base = a.ep ? a.ep[we_i] : a.wa + size_t(we_i) * a.sa;
        const uint8_t* rowp = reinterpret_cast<const uint8_t*>(base) + size_t(row) * a.row_bytes;
        const int nruns = a.K / 32, ngr = (nruns + RUNS - 1) / RUNS;
        const int g0 = part * ngr / G, g1 = (part + 1) * ngr / G;
        for (int g = g0; g < g1; ++g) {
            uint32_t W[kGrW<TY>];
            const int off_b = g * GB;
            group_load<TY>(rowp + off_b, int(a.row_bytes) - off_b, W, a.ldcg != 0);
            const int r0 = g * RUNS;
            if constexpr (RUNS == 8) {
                row_run<TY, TT, 0>(W, tab, t, r0, xp, n, s);
                row_run<TY, TT, 1>(W, tab, t, r0 + 1, xp, n, s);
                row_run<TY, TT, 2>(W, tab, t, r0 + 2, xp, n, s);
                row_run<TY, TT, 3>(W, tab, t, r0 + 3, xp, n, s);
                row_run<TY, TT, 4>(W, tab, t, r0 + 4, xp, n, s);
                row_run<TY, TT, 5>(W, tab, t, r0 + 5, xp, n, s);
                row_run<TY, TT, 6>(W, tab, t, r0 + 6, xp, n, s);
                row_run<TY, TT, 7>(W, tab, t, r0 + 7, xp, n, s);
            } else {
                row_run<TY, TT, 0>(W, tab, t, r0, xp, n, s);
                row_run<TY, TT, 1>(W, tab, t, r0 + 1, xp, n, s);
                if (r0 + 2 < nruns) row_run<TY, TT, 2>(W, tab, t, r0 + 2, xp, n, s);
                if (r0 + 3 < nruns) row_run<TY, TT, 3>(W, tab, t, r0 + 3, xp, n, s);
            }
        }
    }
    if constexpr (G > 1) {
#pragma unroll
        for (int o = 1; o < G; o <<= 1)
#pragma unroll
            for (int k = 0; k < TT; ++k) s[k] += __shfl_xor_sync(0xffffffffu, s[k], o);
    }
    if (row < N && part == 0) {
#pragma unroll
        for (int k = 0; k < TT; ++k)
            if (k < n) a.Y[size_t(__ldg(&J->out[k])) * N + row] = s[k];
    }
}

template <int TY, int BT, int G = kRowG<TY>>
void rows_launch_bt(const RowArgs& a, int T, int njobs_cap, cudaStream_t s) {
    constexpr size_t smem = size_t(TabU<TY>::n) * 16;
    const dim3 grid((a.N + BT / G - 1) / (BT / G), njobs_cap);
    if (T <= 1) k_moe_rows<TY, 1, BT, G><<<grid, BT, smem, s>>>(a);
    else if (T == 2) k_moe_rows<TY, 2, BT, G><<<grid, BT, smem, s>>>(a);
    else if (T == 3) k_moe_rows<TY, 3, BT, G><<<grid, BT, smem, s>>>(a);
    else if (T == 4) k_moe_rows<TY, 4, BT, G><<<grid, BT, smem, s>>>(a);
    else if (T <= 6) k_moe_rows<TY, 6, BT, G><<<grid, BT, smem, s>>>(a);
    else if (T <= 8) k_moe_rows<TY, 8, BT, G><<<grid, BT, smem, s>>>(a);
    else if (T <= 12) k_moe_rows<TY, 12, BT, G><<<grid, BT, smem, s>>>(a);
    else k_moe_rows<TY, 16, BT, G><<<grid, BT, smem, s>>>(a);
}

// FNX_ROWS_BT (phases 0 and 1) and FNX_ROWS_BT2 (phase 2, the experts copied in after the wait): threads a block, 256 (the
// default), 128 or 64. A smaller block takes fewer rows and computes each the same way; phase 2 launches the whole grid and only
// the copied experts' blocks work (5 a gate|up and 10 a down expert of 256 threads), too few for 56 SMs
inline int rows_bt(const char* name) {
    const char* e = std::getenv(name);
    const int v = e ? std::atoi(e) : 256;
    if (v != 64 && v != 128 && v != 256) throw std::runtime_error(std::string(name) + ": 64, 128 or 256 threads a block");
    return v;
}
// FNX_ROWS_LDCG=1: the weights past L1, which keeps IQ2_S's grid (read from global memory) and the window's rows
inline int rows_ldcg() {
    static const int v = [] {
        const char* e = std::getenv("FNX_ROWS_LDCG");
        return e && std::atoi(e) != 0 ? 1 : 0;
    }();
    return v;
}

template <int TY>
void rows_launch(const RowArgs& a, int T, int njobs_cap, cudaStream_t s) {
    static const int bt1 = rows_bt("FNX_ROWS_BT"), bt2 = rows_bt("FNX_ROWS_BT2");
    // FNX_NOC_FILE g2 (IDEJE 34): phase 2's rows with 4 or 8 lanes, so one row's weights are not one lane's (the sums add in another order)
    // 9: by the window (mikro/noc_test.cu): gate|up 8 lanes up to T 4, else 4; down 4 lanes up to T 4, else as phase 1
    if (const int g2 = noc_mode().g2; a.phase == 2 && g2) {
        const bool down = kRowG<TY> == 1;
        const int g = g2 != 9 ? g2 : down ? (T <= 4 ? 4 : 0) : (T <= 4 ? 8 : 4);
        if (g > kRowG<TY>) {
            if (g == 8) rows_launch_bt<TY, 128, 8>(a, T, njobs_cap, s);
            else rows_launch_bt<TY, 128, 4>(a, T, njobs_cap, s);
            return;
        }
    }
    const int bt = a.phase == 2 ? bt2 : bt1;
    if (bt == 64) rows_launch_bt<TY, 64>(a, T, njobs_cap, s);
    else if (bt == 128) rows_launch_bt<TY, 128>(a, T, njobs_cap, s);
    else rows_launch_bt<TY, 256>(a, T, njobs_cap, s);
}

}  // namespace

bool p100_rows_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_P100_ROWS");
        return e ? std::atoi(e) != 0 : FNX_ARCH < 70;
    }();
    return on;
}

namespace {
__global__ void __launch_bounds__(256) k_f32_f16_rows(const float* __restrict__ x, size_t n, __half* __restrict__ out) {
    const size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(x[i]);
}
}  // namespace
void p100_f32_f16(const float* x, size_t n, uint16_t* out, void* st) {
    k_f32_f16_rows<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) st>>>(x, n, reinterpret_cast<__half*>(out));
}

// Y[job.out[k] * n_out + row] = sum over the row of W[job.w][row][c] * x[job.y[k] * n_in + c]; x: FP16 rows, Y: floats
void p100_moe_rows_phase(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, int T, const uint16_t* x, float* out, int phase, void* st) {
    p100_moe_rows_phase_w(A, sa, jobs, njobs, njobs_cap, T, x, out, phase, st, nullptr, nullptr, nullptr, 0, nullptr);
}
void p100_moe_rows_phase_w(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, int T, const uint16_t* x, float* out, int phase, void* st,
                           const unsigned* done, const unsigned* want, const int* list, int layer, unsigned* err) {
    cudaStream_t s = (cudaStream_t) st;
    if (A.n_in % 32 || A.n_in / 8 % 4) throw std::runtime_error("p100 moe rows: columns not in units of 32");
    RowArgs a{};
    a.t = tabs(s);
    a.wa = (const char*) A.p;
    a.sa = sa;
    a.row_bytes = A.row_bytes;
    a.ep = A.ep;
    a.jobs = jobs;
    a.njobs = njobs;
    a.X = reinterpret_cast<const uint4*>(x);
    a.Y = out;
    a.ldx_u = A.n_in / 8;
    a.N = A.n_out;
    a.K = A.n_in;
    a.phase = phase;
    a.ldcg = rows_ldcg();
    a.wdone = phase == 2 ? done : nullptr, a.wwant = want, a.wlist = list, a.wlayer = layer, a.werr = err;
    switch (A.type) {
        case T_IQ3_XXS: rows_launch<18>(a, T, njobs_cap, s); break;
        case T_IQ3_S: rows_launch<21>(a, T, njobs_cap, s); break;
        case T_IQ2_S: rows_launch<22>(a, T, njobs_cap, s); break;
        case T_IQ4_XS: rows_launch<23>(a, T, njobs_cap, s); break;
        case T_IQ4_NL: rows_launch<20>(a, T, njobs_cap, s); break;
        case T_NVFP4: rows_launch<40>(a, T, njobs_cap, s); break;   // edge0_patch.py
        case T_Q4_0: rows_launch<2>(a, T, njobs_cap, s); break;
        case T_IQ2_XXS: rows_launch<16>(a, T, njobs_cap, s); break;
        case T_IQ1_M: rows_launch<29>(a, T, njobs_cap, s); break;
        case T_Q4_K: rows_launch<12>(a, T, njobs_cap, s); break;   // q4x_patch.py
        case T_Q5_K: rows_launch<13>(a, T, njobs_cap, s); break;
        case T_Q5_1: rows_launch<7>(a, T, njobs_cap, s); break;
        case T_Q8_0: rows_launch<8>(a, T, njobs_cap, s); break;
        default: throw std::runtime_error("p100 moe rows: expert type " + std::to_string(A.type) + " not handled");
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100 moe rows: ") + cudaGetErrorString(e));
}
void p100_moe_rows(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, int T, const uint16_t* x, float* out, void* st) {
    p100_moe_rows_phase(A, sa, jobs, njobs, njobs_cap, T, x, out, 0, st);
}

bool p100_overlap_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_P100_OVERLAP");
        return e ? std::atoi(e) != 0 : FNX_ARCH < 70;
    }();
    return on;
}

namespace {
// the jobs of the experts the plan listed get w = ~w in both lists, one warp
__global__ void __launch_bounds__(32) k_jobs_split(Job* gu, Job* dn, const int* njobs, int layer, const int* list) {
    const int k = list[0];
    if (k == 0) return;
    const int nj = *njobs, base = layer * NEXP;
    for (int j = threadIdx.x; j < nj; j += 32) {
        const int g = base + gu[j].w;
        for (int i = 1; i <= k; ++i)
            if (list[i] == g) {
                gu[j].w = ~gu[j].w;
                dn[j].w = ~dn[j].w;
                break;
            }
    }
}
// k_cache_copy (kernels.cu) on few blocks: beside the row kernels, and a card's PCIe takes no more anyway
__global__ void k_cache_copy_side(ExpertCache ec, int p, const int* list, size_t gu_bytes, size_t dn_bytes) {
    const int n = list[0];
    const size_t ug = gu_bytes / 16, per = ug + dn_bytes / 16;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < (size_t) n * per; i += (size_t) gridDim.x * blockDim.x) {
        const int k = (int) (i / per);
        const size_t o = i % per;
        const int g = list[1 + k];
        char* at = ec.pool[p] + (size_t) (ec.slot_of[g] - ec.s0[p]) * ec.slot_bytes[p];
        if (o < ug) reinterpret_cast<uint4*>(at)[o] = reinterpret_cast<const uint4*>(ec.host_gu[g])[o];
        else reinterpret_cast<uint4*>(at + ec.dn_off[p])[o - ug] = reinterpret_cast<const uint4*>(ec.host_dn[g])[o - ug];
    }
}
// the last `count` slots of pool p lent to the prompt path (their experts back to their host copies, ref 3) or given back empty (ref 0: the first misses take them; FNX_PF_REFILL gives them back full instead, Engine::pf_refill_start)
__global__ void __launch_bounds__(256) k_tail_slots(ExpertCache ec, int p, int count, bool lend) {
    const int s1 = ec.s0[p] + ec.n[p];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const int s = s1 - 1 - i;
        if (lend && ec.fill_list) ec.fill_list[s] = ec.owner[s];   // FNX_PF_REFILL: the expert it held, to take again after the read
        if (const int o = ec.owner[s]; o >= 0) {
            ec.slot_of[o] = -1;
            ec.ep_gu[o] = ec.host_gu[o];
            ec.ep_dn[o] = ec.host_dn[o];
        }
        ec.owner[s] = -1;
        ec.ref[s] = lend ? 3 : 0;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) ec.lent[EC_POOLS + p] = lend ? count : 0;
}
// FNX_PREFETCH: layer `layer`'s guessed experts (the first *want of each row's top 10, k_probe_guess) not in a slot take one, as
// cache_plan gives a miss one: CLOCK over the pool's slots not lent, the old expert back to its host copy. ref 1, so a later guess of the
// same plan passes it once and a wrong guess goes at the hand's next turn. list: how many took a slot, then their ids (k_dma_publish2)
__device__ __forceinline__ void pf_plan(const int* __restrict__ guess, int T, int layer, const ExpertCache& ec, const int* __restrict__ want, int* list) {
    __shared__ unsigned seen[NEXP / 32];
    const int lane = threadIdx.x, w = min(*want, TOPK), base = layer * NEXP, p = ec.pool_of[layer], lent = ec.lent[p];
    if (w <= 0 || (ec.arena_full && (p != ec.arena_pool || lent == 0))) {
        if (lane == 0) list[0] = 0;
        return;
    }
    const int s0 = ec.s0[p] + lent, n = ec.n[p] - lent - ec.lent[EC_POOLS + p];
    for (int i = lane; i < NEXP / 32; i += 32) seen[i] = 0;
    __syncwarp();
    for (int i = lane; i < T * TOPK; i += 32)
        if (i % TOPK < w)
            if (const int e = guess[i]; e >= 0 && ec.slot_of[base + e] < 0) atomicOr(&seen[e >> 5], 1u << (e & 31));
    __syncwarp();
    int hand = ec.hand[p] - s0, k = 0;
    if (hand < 0 || hand >= n) hand = 0;
    for (int wd = 0; wd < NEXP / 32; ++wd)
        for (unsigned bits = seen[wd]; bits; bits &= bits - 1) {
            const int g = base + wd * 32 + __ffs(bits) - 1;
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
                char* at = ec.pool[p] + (size_t) (victim - ec.s0[p]) * ec.slot_bytes[p];
                ec.owner[victim] = g;
                ec.slot_of[g] = victim;
                ec.ref[victim] = 1;
                ec.ep_gu[g] = at;
                ec.ep_dn[g] = at + ec.dn_off[p];
                list[1 + k] = g;
            }
            ++k;
            __syncwarp();
        }
    if (lane == 0) {
        ec.hand[p] = s0 + hand;
        list[0] = k;
    }
}
__global__ void __launch_bounds__(32) k_pf_plan(const int* __restrict__ guess, int T, int layer, ExpertCache ec, const int* __restrict__ want, int* __restrict__ list) {
    pf_plan(guess, T, layer, ec, want, list);
}
__device__ __forceinline__ float silu_over(float x) { return x / (1.0f + __expf(-x)); }
// swiglu of every pair once the listed experts' gate|up rows are there; nothing when no expert was listed
__global__ void k_swiglu_after(const float* __restrict__ gu, int ffs, int P, __half* __restrict__ h, const int* __restrict__ list) {
    if (list[0] == 0) return;
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) P * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t r = e / ffs, i = e % ffs;
        const float g = gu[r * 2 * ffs + i];
        h[e] = __float2half(silu_over(g) * gu[r * 2 * ffs + ffs + i]);
    }
}
}  // namespace
void p100_tail_slots(const ExpertCache& ec, int p, int count, bool lend, void* st) {
    k_tail_slots<<<32, 256, 0, (cudaStream_t) st>>>(ec, p, count, lend);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100_tail_slots: ") + cudaGetErrorString(e));
}
void p100_pf_plan(const int* guess, int T, int layer, const ExpertCache& ec, const int* want, int* list, void* st) {
    k_pf_plan<<<1, 32, 0, (cudaStream_t) st>>>(guess, T, layer, ec, want, list);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100_pf_plan: ") + cudaGetErrorString(e));
}
void p100_jobs_split(Job* gu, Job* dn, const int* njobs, int layer, const int* list, void* st) {
    k_jobs_split<<<1, 32, 0, (cudaStream_t) st>>>(gu, dn, njobs, layer, list);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100_jobs_split: ") + cudaGetErrorString(e));
}
void p100_cache_copy(int layer, size_t gu_stride, size_t dn_stride, const ExpertCache& ec, const int* list, void* st) {
    const int p = ec.pool_of[layer];
    if (gu_stride != ec.dn_off[p] || ec.dn_off[p] + dn_stride != ec.slot_bytes[p])
        throw std::runtime_error("p100_cache_copy: the layer's experts are not its pool's size");
    static const int blocks = [] { const char* e = std::getenv("FNX_COPY_BLOCKS"); return e ? std::atoi(e) : 40; }();
    static const int threads = [] { const char* e = std::getenv("FNX_COPY_THREADS"); return e ? std::atoi(e) : 256; }();
    k_cache_copy_side<<<blocks, threads, 0, (cudaStream_t) st>>>(ec, p, list, gu_stride, dn_stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100_cache_copy: ") + cudaGetErrorString(e));
}
void p100_swiglu_after(const float* gu, int ffs, int P, uint16_t* h16, const int* list, void* st) {
    k_swiglu_after<<<(unsigned) std::min<size_t>(4096, ((size_t) P * ffs + 255) / 256), 256, 0, (cudaStream_t) st>>>(gu, ffs, P, reinterpret_cast<__half*>(h16), list);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100_swiglu_after: ") + cudaGetErrorString(e));
}

}  // namespace fnx
