// Dense Q8_0 matrix x up to MAXT q8_1 rows, from the split layout (int8 rows, f16 scale per 32) (included by
// kernels.cu). Each lane takes 16 columns per step (one 16-byte load per row), each warp Q8S_R rows, so a lane's
// activation values are loaded once per step for all its rows. Fewer rows per warp for small matrices: more blocks.
namespace {

constexpr int Q8S_THREADS = 256, Q8S_SEGS = 3;

// A kernel of T rows of window and R rows a warp: its form of the int8 dot product and its loops, chosen per kernel on GP100
// (idea 63, mikro/kane/q8s_izbira.cu; every choice gives the same bits). The xmad form is about twice as fast alone and holds
// more registers: it pays where the kernel stays on its step of blocks an SM (64 / 85 / 128 registers at 256 threads).
//   FNX_Q8S_X1, _X2, _X4: bit T set: the kernel of T rows with 1, 2, 4 rows a warp takes xmad;
//   FNX_Q8S_C1, _C2, _C4: bit T set: it takes the compiler's own form (looked at first);
//   FNX_Q8S_MANY2: bit T set: 2 rows a warp take the loop of the wide windows (weights held, 4 activation rows at a time)
//                  at T <= 8 too;
//   FNX_Q8S_TC8: bit T set: 4 rows a warp take the activations 8 rows at a time;
//   FNX_Q8S_R4_T: matrices of 8192 rows and more take 4 rows a warp from this many rows of window on (2 below);
//   FNX_Q8S_TWO_T: matrices of 2048 rows and more: from this many rows of window on two launches, 8 rows and the rest
//                  (kernels.cu's q8s_run).
#ifndef FNX_Q8S_X1
#define FNX_Q8S_X1 0x1c86cu
#endif
#ifndef FNX_Q8S_X2
#define FNX_Q8S_X2 0xfed8u
#endif
#ifndef FNX_Q8S_X4
#define FNX_Q8S_X4 0x7fe0u
#endif
#ifndef FNX_Q8S_C1
#define FNX_Q8S_C1 0x82u
#endif
#ifndef FNX_Q8S_C2
#define FNX_Q8S_C2 0u
#endif
#ifndef FNX_Q8S_C4
#define FNX_Q8S_C4 0u
#endif
#ifndef FNX_Q8S_MANY2
#define FNX_Q8S_MANY2 0x118u
#endif
#ifndef FNX_Q8S_TC8
#define FNX_Q8S_TC8 0x6200u
#endif
#ifndef FNX_Q8S_R4_T
#define FNX_Q8S_R4_T 5
#endif
#ifndef FNX_Q8S_TWO_T
#define FNX_Q8S_TWO_T 15
#endif
constexpr int q8s_x(int T, int R) {
    if (((R == 1 ? FNX_Q8S_C1 : R == 2 ? FNX_Q8S_C2 : FNX_Q8S_C4) >> T) & 1u) return 2;
    return (((R == 1 ? FNX_Q8S_X1 : R == 2 ? FNX_Q8S_X2 : FNX_Q8S_X4) >> T) & 1u) ? 1 : 0;
}
// activation rows at a time against the step's held weights; 0: all T rows' activations held, the weights row by row.
// q8s_tc0: as it was before the choice (the kernels that are not k_q8s keep it)
constexpr int q8s_tc0(int T, int R) { return ((T > 8 && R >= 2) || (R >= 4 && T > 4)) ? 4 : 0; }
constexpr int q8s_tc(int T, int R) {
    if (R == 4 && ((FNX_Q8S_TC8 >> T) & 1u)) return 8;
    if (R == 2 && ((FNX_Q8S_MANY2 >> T) & 1u)) return 4;
    return q8s_tc0(T, R);
}

// up to Q8S_SEGS matrices over the same activations in one launch, their rows one after another: segment s holds rows
// [end[s - 1], end[s]) of the launch, its own weights and output (row r of the segment at out[s][t * stride[s] + r])
struct Q8Segs {
    const int8_t* q[Q8S_SEGS];
    const uint16_t* d[Q8S_SEGS];
    float* out[Q8S_SEGS];
    int end[Q8S_SEGS];
    int stride[Q8S_SEGS];
};
__device__ __forceinline__ int q8s_seg(const Q8Segs& g, int row) { return row < g.end[0] ? 0 : row < g.end[1] ? 1 : 2; }
// a segment's field by selects: indexing the parameter arrays by a run-time s puts them in local memory (q8s 60% slower)
template <typename F> __device__ __forceinline__ F q8s_pick(int s, F a, F b, F c) { return s == 0 ? a : s == 1 ? b : c; }

// SEG: the rows' segments looked up (q8s_gemv_segs); else one matrix, g's first segment, its row pointers made in the
// loops as before (held per row in registers they cost the 16-row path ~30-80%)
// block bx of the launch (a kernel of its own, or part of a launch shared with other work: k_q8s_ab)
// X, TC: the dot product's form and the loop (above); k_q8s takes its kernel's, the other users vmad and the loop as before
template <int T, int Q8S_R, bool SPLIT, bool SEG, int X = 0, int TC = q8s_tc0(T, Q8S_R)>
__device__ __forceinline__ void q8s_block(const Q8Segs& g, int rows, int cols, const block_q8_1* __restrict__ y,
                                          const int8_t* __restrict__ tq, const float* __restrict__ td, int bx) {
    constexpr int Q8S_ROWS = Q8S_THREADS / 32 * Q8S_R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = bx * Q8S_ROWS + warp * Q8S_R, nb = cols / 32;
    // each of the warp's rows: its segment's weights at the row's place in it
    const int8_t* qr[Q8S_R];
    const uint16_t* dr[Q8S_R];
    if constexpr (SEG) {
#pragma unroll
        for (int r = 0; r < Q8S_R; ++r) {
            const int row = min(row0 + r, rows - 1), s = q8s_seg(g, row), lr = row - q8s_pick(s, 0, g.end[0], g.end[1]);
            qr[r] = q8s_pick(s, g.q[0], g.q[1], g.q[2]) + (size_t) lr * cols;
            dr[r] = q8s_pick(s, g.d[0], g.d[1], g.d[2]) + (size_t) lr * nb;
        }
    }
    auto wq = [&](int r) { return SEG ? qr[r] : g.q[0] + (size_t) min(row0 + r, rows - 1) * cols; };
    auto wd = [&](int r) { return SEG ? dr[r] : g.d[0] + (size_t) min(row0 + r, rows - 1) * nb; };
    float acc[Q8S_R][T];
#pragma unroll
    for (int r = 0; r < Q8S_R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.f;
    if constexpr (TC > 0) {
        // the step's weights for the warp's rows held, the activations TC rows at a time against them: all T rows'
        // held took up to ~160 registers (16 rows, 4 a warp: one block an SM, 12% occupancy). kbench, 16 rows: the
        // head 722 -> 495 us, qkvz 60 -> 48; at 5-8 rows it pays only with 4 rows a warp, and not at all with 1. Each
        // sum takes the same steps in the same order
#pragma unroll 1
        for (int c = lane * 16; c < cols; c += 512) {
            const int b = c >> 5, half = (c >> 4) & 1;
            int4 w[Q8S_R];
            float dw[Q8S_R];
#pragma unroll
            for (int r = 0; r < Q8S_R; ++r) {
                w[r] = __ldg(reinterpret_cast<const int4*>(wq(r) + c));
                dw[r] = __half2float(__ushort_as_half(__ldg(wd(r) + b)));
            }
#pragma unroll
            for (int t0 = 0; t0 < T; t0 += TC) {
                int a[TC][4];
                float da[TC];
#pragma unroll
                for (int tt = 0; tt < TC; ++tt) {
                    const int t = min(t0 + tt, T - 1);
                    if constexpr (SPLIT) {
                        const int4 v = *reinterpret_cast<const int4*>(tq + (size_t) t * cols + c);
                        a[tt][0] = v.x;
                        a[tt][1] = v.y;
                        a[tt][2] = v.z;
                        a[tt][3] = v.w;
                        da[tt] = td[(size_t) t * nb + b];
                    } else {
                        const block_q8_1* yb = y + (size_t) t * nb + b;
                        const int* qs = reinterpret_cast<const int*>(yb->qs) + half * 4;
#pragma unroll
                        for (int k = 0; k < 4; ++k) a[tt][k] = qs[k];
                        da[tt] = __low2float(yb->ds);
                    }
                }
#pragma unroll
                for (int r = 0; r < Q8S_R; ++r) {
                    if (row0 + r >= rows) break;
#pragma unroll
                    for (int tt = 0; tt < TC; ++tt) {
                        if (t0 + tt >= T) break;
                        int s = dp4a_x<X>(w[r].x, a[tt][0], 0);
                        s = dp4a_x<X>(w[r].y, a[tt][1], s);
                        s = dp4a_x<X>(w[r].z, a[tt][2], s);
                        s = dp4a_x<X>(w[r].w, a[tt][3], s);
                        acc[r][t0 + tt] = fmaf(dw[r] * da[tt], (float) s, acc[r][t0 + tt]);
                    }
                }
            }
        }
    } else {
#pragma unroll 2
    for (int c = lane * 16; c < cols; c += 512) {
        const int b = c >> 5, half = (c >> 4) & 1;
        int a[T][4];
        float da[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if constexpr (SPLIT) {
                // the twin: 16 values in one load, the block's scale as a float
                const int4 v = *reinterpret_cast<const int4*>(tq + (size_t) t * cols + c);
                a[t][0] = v.x;
                a[t][1] = v.y;
                a[t][2] = v.z;
                a[t][3] = v.w;
                da[t] = td[(size_t) t * nb + b];
            } else {
                const block_q8_1* yb = y + (size_t) t * nb + b;
                const int* qs = reinterpret_cast<const int*>(yb->qs) + half * 4;
#pragma unroll
                for (int k = 0; k < 4; ++k) a[t][k] = qs[k];
                da[t] = __low2float(yb->ds);
            }
        }
#pragma unroll
        for (int r = 0; r < Q8S_R; ++r) {
            if (row0 + r >= rows) break;
            const int4 w = __ldg(reinterpret_cast<const int4*>(wq(r) + c));
            const float dw = __half2float(__ushort_as_half(__ldg(wd(r) + b)));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                int s = dp4a_x<X>(w.x, a[t][0], 0);
                s = dp4a_x<X>(w.y, a[t][1], s);
                s = dp4a_x<X>(w.z, a[t][2], s);
                s = dp4a_x<X>(w.w, a[t][3], s);
                acc[r][t] = fmaf(dw * da[t], (float) s, acc[r][t]);
            }
        }
    }
    }
#pragma unroll
    for (int r = 0; r < Q8S_R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && row0 + r < rows) {
                const int row = row0 + r, s = SEG ? q8s_seg(g, row) : 0;
                q8s_pick(s, g.out[0], g.out[1], g.out[2])[(size_t) t * q8s_pick(s, g.stride[0], g.stride[1], g.stride[2]) + row -
                                                          q8s_pick(s, 0, g.end[0], g.end[1])] = v;
            }
        }
}

template <int T, int Q8S_R, bool SPLIT, bool SEG, int X = q8s_x(T, Q8S_R), int TC = q8s_tc(T, Q8S_R)>
__global__ void __launch_bounds__(Q8S_THREADS) k_q8s(const Q8Segs g, int rows, int cols, const block_q8_1* __restrict__ y,
                                                    const int8_t* __restrict__ tq, const float* __restrict__ td) {
    q8s_block<T, Q8S_R, SPLIT, SEG, X, TC>(g, rows, cols, y, tq, td, blockIdx.x);
}

// The shared expert inside the routed experts' launches (no second stream): its gate|up as extra blocks of the router's
// launch, its swiglu and down as an extra row of blocks of the routed down's. Both matrices have fewer than 4096 rows:
// q8s with one row a warp, as q8s_gemv runs them alone. Its FF slice is SH_MAXF at most (model.hpp).
struct ShexpUp {               // gate|up: rows 2 * ffs over the hc read's q8_1 rows y (and its split twin tq / td)
    Q8Segs w;
    int rows = 0, cols = 0, blocks = 0;   // blocks: the launch's last `blocks` blocks (0: none)
    const block_q8_1* y = nullptr;
    const int8_t* tq = nullptr;
    const float* td = nullptr;
};
struct ShexpDown {             // swiglu of gu ([T][2 * ffs]: gate, then up), as int8 per 32, then down (rows N) into w's out
    Q8Segs w;
    int rows = 0, ffs = 0;     // rows 0: none
    const float* gu = nullptr;
};
template <int T> __device__ __forceinline__ void shexp_up_block(const ShexpUp& s, int bx) {
    if (s.tq) q8s_block<T, 1, true, false>(s.w, s.rows, s.cols, s.y, s.tq, s.td, bx);
    else q8s_block<T, 1, false, false>(s.w, s.rows, s.cols, s.y, nullptr, nullptr, bx);
}
// this block's share (q8s blocks bx0, bx0 + nbx, ...) of the down rows, after the swiglu made in shared memory as
// k_swiglu_stacked makes its split twin
template <int T> __device__ void shexp_down_blocks(const ShexpDown& s, int bx0, int nbx) {
    __shared__ __align__(16) int8_t tq[T * SH_MAXF];
    __shared__ float td[T * (SH_MAXF / 32)];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nb = s.ffs / 32;
    for (int it = warp; it < T * nb; it += blockDim.x / 32) {
        const int t = it / nb, b = it % nb;
        const float* g = s.gu + (size_t) t * 2 * s.ffs;
        const float xi = silu_(g[b * 32 + lane]) * g[s.ffs + b * 32 + lane];
        const float amax = wmax(fabsf(xi)), d = amax / 127.0f;
        tq[t * s.ffs + b * 32 + lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi / d);
        if (lane == 0) td[t * nb + b] = __half2float(__float2half(d));
    }
    __syncthreads();
    const int nqb = (s.rows + Q8S_THREADS / 32 - 1) / (Q8S_THREADS / 32);
    for (int bx = bx0; bx < nqb; bx += nbx) q8s_block<T, 1, true, false>(s.w, s.rows, s.ffs, nullptr, tq, td, bx);
}

// FNX_DENSE_H (tests): q8s's products in half precision for wide windows (split38: 1.3-1.4x q8s's best at 6 to 16 rows on 8192 and
// 32768 x 2560, 1.2-1.4x on 2560 x 3072). Each step's 16 weights of a row decoded once for the window's rows (under exponent 0x64:
// 1024 + code + 128, minus 1152, exact), each row's 16 activation codes from the int8 twin once for the warp's rows (under 0x48:
// 8 + (code + 128) / 128, minus 9: code / 128, exact), 16 products a lane in half2 (|sum| <= 1008: no overflow), then fp32 with the
// scales (the twin's times 128). Lossy at the 1e-4 level: against the exact sums 6.09e-3 relative, q8s's 6.08e-3 (the activations'
// int8 rounding dominates)
__device__ __forceinline__ half2 h8_h2(uint32_t x) { return *reinterpret_cast<half2*>(&x); }
__device__ __forceinline__ uint32_t h8_prmt(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
// four weight codes as two exact half2 (bytes 0, 1 and 2, 3)
__device__ __forceinline__ void h8_w(uint32_t x, half2& lo, half2& hi) {
    const uint32_t u = x ^ 0x80808080u;
    lo = __hsub2(h8_h2(h8_prmt(u, 0x64646464u, 0x5140u)), h8_h2(0x64806480u));
    hi = __hsub2(h8_h2(h8_prmt(u, 0x64646464u, 0x5342u)), h8_h2(0x64806480u));
}
// four activation codes as two exact half2 of code / 128
__device__ __forceinline__ void h8_a(uint32_t x, half2& lo, half2& hi) {
    const uint32_t u = x ^ 0x80808080u;
    lo = __hsub2(h8_h2(h8_prmt(u, 0x48484848u, 0x5140u)), h8_h2(0x48804880u));
    hi = __hsub2(h8_h2(h8_prmt(u, 0x48484848u, 0x5342u)), h8_h2(0x48804880u));
}
// out[t * stride + row]; R rows a warp, MINB blocks an SM asked of the register allocation
template <int T, int R, int MINB>
__global__ void __launch_bounds__(Q8S_THREADS, MINB) k_h8(const int8_t* __restrict__ wq, const uint16_t* __restrict__ wd, int rows,
                                                         int cols, const int8_t* __restrict__ tq, const float* __restrict__ td,
                                                         float* __restrict__ out, int stride) {
    constexpr int ROWS_B = Q8S_THREADS / 32 * R;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row0 = blockIdx.x * ROWS_B + warp * R, nb = cols / 32;
    float acc[R][T];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.f;
#pragma unroll 1
    for (int c = lane * 16; c < cols; c += 512) {
        const int b = c >> 5;
        half2 w[R][8];
        float dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = min(row0 + r, rows - 1);
            const uint4 q = __ldg(reinterpret_cast<const uint4*>(wq + (size_t) row * cols + c));
            dw[r] = __half2float(__ushort_as_half(__ldg(wd + (size_t) row * nb + b)));
            h8_w(q.x, w[r][0], w[r][1]);
            h8_w(q.y, w[r][2], w[r][3]);
            h8_w(q.z, w[r][4], w[r][5]);
            h8_w(q.w, w[r][6], w[r][7]);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            half2 x[8];
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(tq + (size_t) t * cols + c));
            h8_a(v.x, x[0], x[1]);
            h8_a(v.y, x[2], x[3]);
            h8_a(v.z, x[4], x[5]);
            h8_a(v.w, x[6], x[7]);
            const float sx = __ldg(td + (size_t) t * nb + b) * 128.f;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                half2 h = __hmul2(w[r][0], x[0]);
#pragma unroll
                for (int k = 1; k < 8; ++k) h = __hfma2(w[r][k], x[k], h);
                const float2 f = __half22float2(h);
                acc[r][t] = fmaf(dw[r] * sx, f.x + f.y, acc[r][t]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0 && row0 + r < rows) out[(size_t) t * stride + row0 + r] = v;
        }
}

}  // namespace
