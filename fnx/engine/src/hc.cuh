// The hyper-connection read (build_hc_mix) in one launch of H_NS * H_RG = 56 blocks' work, with two grid barriers
// (included by kernels.cu). The Q8_0 mixes run on int8 activations with dp4a, as llama.cpp's CUDA path quantizes a
// Q8_0 matmul's input to q8_1.
//   down:   block (slice s of 352 or 384 columns, row group): R' = R + bo * 2 sigmoid(inj / HC) for the slice, xs = R' * w_norm,
//           its square sums, xs as int8 per 32; partial dots of the slice for the group's rows (the down rows, then the
//           four inject rows in bf16 on xs itself). Then the block issues its up weights' loads.
//   reduce: (after the first barrier) one warp per 32 rows of lo and token: rs per stream from the slices' square sums,
//           lo = silu(rs-weighted sum of the slices' partials / HC) as int8 per 32; the inject rows likewise.
//   up:     (after the second barrier) block b = 32 columns: every stream's up row of each column . lo, then
//           mixed = mean_c xn_c * sigmoid(.), R <- R' when a write was folded in, mixed as q8_1 when asked.
// The barriers need every block of the grid resident at once: the grid is one block per SM, up to 56 (hc_prepare), and a
// block of a card with fewer SMs takes the down parts of 56 / SMs of them in turn; a block takes the up shares
// vb = blockIdx.x + k * grid in turn. Nothing else may run on the card during a read (the engine's
// second-stream branches fork after it). A barrier that waits too long sets bit 4 of err.
namespace {

// 56 equal rectangles, one per SM of a P100: H_NS slices (H_SL: the widest) x H_RG row groups; group 1 has the inject rows too, a warp of its own
constexpr int H_SL = 384, H_NS = 28, H_SPS = H_NS / HC, H_R0 = 160;
constexpr int H_ROWS = LR + HC, H_RG = 2;
constexpr int H_TPRD = 4, H_DT = ((H_ROWS - H_R0) * H_TPRD + 31) / 32 * 32, H_BLOCKS = H_NS * H_RG;
static_assert(H_NS % HC == 0 && H_SL <= H_DT && H_R0 * H_TPRD % 32 == 0, "slices inside a stream, a thread per column, the inject rows from a warp's start");
// slice s starts at column unit s * 320 / 28 (a unit: q8's 32 columns), so every stream starts a slice and the slices have 11 or 12 units
__host__ __device__ __forceinline__ constexpr int hc_col0(int s) { return s * (D / 32) / H_NS * 32; }
// up: shares of 32 columns (mixed as q8_1 needs its blocks whole), walked over the grid, H_TPRU lanes per column; lane b < LR / 32 takes lo's 32-value block b
constexpr int H_UPC = 32, H_TPRU = 16, H_UB = LR / 32, H_UPS = N / H_UPC;
static_assert(H_UPC * H_TPRU <= H_DT && H_UB <= H_TPRU, "the up lanes");
// the grid barrier's counter runs modulo hc_bar_mod(g), a multiple of the grid size g, so every barrier's g arrivals
// stay in one generation [k * g, (k + 1) * g) across the wrap
__host__ __device__ constexpr unsigned hc_bar_mod(unsigned g) { return 0xffffffffu / g * g; }

struct HcArgs {
    float* R;
    const float* bo;
    const float* inj_prev;
    const float* w_norm;
    Q8S w_down;
    Q8S w_up;
    const uint16_t* w_inj;
    float eps;
    float* mixed;
    float* inject;
    float* part;   // [H_NS][T][H_ROWS]
    float* ssp;    // [T][H_NS]: the slices' square sums
    int8_t* loq;   // [T][LR]: lo as int8 per 32
    float* lod;    // [T][LR / 32]: their scales
    unsigned* bar;       // the grid barrier's counter (modulo hc_bar_mod(grid))
    unsigned* err;       // bit 4: a barrier timed out
    block_q8_1* q8;      // mixed as q8_1 too, or null
    int8_t* tq = nullptr;   // and its split twin, or null
    float* td = nullptr;
    int up2 = 0;            // FNX_HC_UP2 (idea 62): 1 the next share's up weights after the share's dots, 2 a later share's to L2 early
};

__device__ __forceinline__ float rprime(const HcArgs& a, int t, int c, int i) {
    float r = a.R[(size_t) t * D + c * N + i];
    if (a.bo) r = fmaf(a.bo[(size_t) t * N + i], 2.0f * sigm(a.inj_prev[t * HC + c] / (float) HC), r);
    return r;
}

// the hc read's int8 dots in the xmad form from FNX_HC_X_T rows of window on (idea 63, mikro/kane/hc_dp4a_test.cu on GP100: the
// kernel holds 80 registers either way; under that many rows xmad's longer code costs more than its rows gain)
#ifndef FNX_HC_X_T
#define FNX_HC_X_T 7
#endif
#ifndef FNX_HC_LOW_F
#define FNX_HC_LOW_F 0
#endif
constexpr int hc_x(int T) { return T >= FNX_HC_X_T ? 1 : FNX_HC_LOW_F; }
// 8 int8 weights (uint2) . 8 int8 activations (two ints)
template <int X> __device__ __forceinline__ int dot8i(const uint2 w, const int2 x) {
    return dp4a_x<X>((int) w.y, x.y, dp4a_x<X>((int) w.x, x.x, 0));
}

// FNX_HC_F16: two of four int8 weights (bytes already xor 0x80, so b + 128) as f16, exactly: a byte permute puts each into the
// mantissa of 1024 (0x64..), and 1024 + 128 comes off. sel 0x4140: bytes 0 and 1, 0x4342: bytes 2 and 3
__device__ __forceinline__ __half2 i8_h2(unsigned x, unsigned sel) {
    const unsigned v = __byte_perm(x, 0x64646464u, sel), m = 0x64806480u;
    return __hsub2(*reinterpret_cast<const __half2*>(&v), *reinterpret_cast<const __half2*>(&m));
}
// 8 int8 weights (uint2) . 8 f16 activations of at most 1 (uint4): four HFMA2, each half summing 4 products under 128 * 4
__device__ __forceinline__ float dot8h(const uint2 w, const uint4 x) {
    const unsigned a = w.x ^ 0x80808080u, b = w.y ^ 0x80808080u;
    __half2 s = __hmul2(i8_h2(a, 0x4140), *reinterpret_cast<const __half2*>(&x.x));
    s = __hfma2(i8_h2(a, 0x4342), *reinterpret_cast<const __half2*>(&x.y), s);
    s = __hfma2(i8_h2(b, 0x4140), *reinterpret_cast<const __half2*>(&x.z), s);
    s = __hfma2(i8_h2(b, 0x4342), *reinterpret_cast<const __half2*>(&x.w), s);
    return __low2float(s) + __high2float(s);
}

// one warp quantizes one 32-value block: lane i's value v -> q[i], the block's scale -> *d (lane 0)
__device__ __forceinline__ void q8_block(float v, int8_t* q, float* d) {
    const float amax = wmax(fabsf(v));
    const float sc = amax / 127.0f;
    q[threadIdx.x & 31] = amax == 0.0f ? 0 : (int8_t) roundf(v / sc);
    if ((threadIdx.x & 31) == 0) *d = sc;
}
// FNX_HC_F16: the same block for the f16 dots: lane i's value v -> h[i] = v / amax, the block's amax -> *d (lane 0)
__device__ __forceinline__ void h_block(float v, __half* h, float* d) {
    const float amax = wmax(fabsf(v));
    h[threadIdx.x & 31] = __float2half_rn(amax == 0.0f ? 0.0f : v / amax);
    if ((threadIdx.x & 31) == 0) *d = amax;
}

// every block of the grid gets here before any goes on (the counter grows by the grid size g per barrier, modulo
// hc_bar_mod(g)). The block's last thread arrives and waits until the counter has left its barrier's generation: it is
// then 1 to 2 * g - 1 past the generation's start, as no block can pass the next barrier before this one does. A fence
// waits for the thread's own loads too, and that thread loads no up weights
__device__ __forceinline__ void grid_sync(unsigned* bar, unsigned* err, unsigned g) {
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
        __threadfence();
        const unsigned mod = hc_bar_mod(g);
        const unsigned old = atomicInc(bar, mod - 1u), base = old - old % g;
        const long long t0 = clock64();
        for (;;) {
            const unsigned cur = *(volatile unsigned*) bar;
            if ((cur >= base ? cur - base : cur + (mod - base)) >= g) break;
            if (clock64() - t0 > 2000000000LL) {
                atomicOr(err, 4u);
                break;
            }
            __nanosleep(64);
        }
        __threadfence();
    }
    __syncthreads();
}

// rs for token k, stream c: 1 / sqrt(mean of R'^2 over the stream + eps), from the slices' square sums
__device__ __forceinline__ float rs_of(const HcArgs& a, int k, int c) {
    float v = 0.f;
    for (int ss = 0; ss < H_SPS; ++ss) v += __ldcg(a.ssp + (size_t) k * H_NS + c * H_SPS + ss);
    return rsqrtf(v / (float) N + a.eps);
}

// the up part's arrays (after the second barrier) in the down part's xs (done before the first): 16 rows fit 48 KB
template <int T, bool H16> struct HcUp {
    typename std::conditional<H16, __half, int8_t>::type lq[T][LR];   // FNX_HC_F16: f16 (scaled per 32 by ld)
    float ld[T][LR / 32];
    float rs[T][HC];
    float ms[T][H_UPC];
    float gs[T][HC][H_UPC];
};

// ONE: one grid barrier instead of two: after it every block makes the whole of lo itself (the same sums in the same
// order, so the same bits) from the slices' partials in L2, and block 0 the inject rows; ~T * 77 KB of L2 reads a block,
// so for small windows only
// WALK: the grid is short of H_BLOCKS and each block takes the shares vb = blockIdx.x + k * gridDim.x in turn; else
// vb = blockIdx.x alone, without the loops (the walk costs a register over the 79 of 80 SMs' form, and a stack slot at
// T = 15)
// H16 (FNX_HC_F16): the dots' activations in f16, scaled per 32 to at most 1, and the weights made f16 (dot8h)
template <int T, bool ONE, bool WALK, bool H16>
__global__ void __launch_bounds__(H_DT, 1) k_hc(HcArgs a) {
    using Lq = typename std::conditional<H16, __half, int8_t>::type;
    static_assert(sizeof(HcUp<T, H16>) <= sizeof(float) * T * H_SL && LR % 16 == 0, "the up part's arrays inside xs, aligned");
    __shared__ __align__(16) float xs[T][H_SL];
    __shared__ __align__(16) Lq xq[T][H_SL];
    __shared__ float xd[T][H_SL / 32];
    __shared__ float red[T][H_DT / 32];
    HcUp<T, H16>& up = *reinterpret_cast<HcUp<T, H16>*>(&xs[0][0]);
    auto& lq = up.lq;
    auto& ld = up.ld;
    auto& rs = up.rs;
    auto& ms = up.ms;
    auto& gs = up.gs;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const unsigned grid = WALK ? gridDim.x : (unsigned) H_BLOCKS;
    // ---------------- down, for each of the block's shares vb
    int vb = blockIdx.x;
    do {
    if (vb != blockIdx.x) __syncthreads();   // the last share's dots are done with xs / xq / xd
    const int s = vb / H_RG, grp = vb % H_RG;
    const int col0 = hc_col0(s), width = hc_col0(s + 1) - col0, c = col0 / N, i0 = col0 - c * N;
    // thread t < H_SL takes column t of every row, the rows' loads side by side (a row at a time, each cost a memory
    // latency: T of them on the group-0 blocks, which the others then waited for at the first barrier)
    static_assert(H_SL <= H_DT, "a thread per column");
    {
        float r[T];
#pragma unroll
        for (int k = 0; k < T; ++k) r[k] = 0.f;
        if (t < width) {
            const float wn = a.w_norm[col0 + t];
#pragma unroll
            for (int k = 0; k < T; ++k) r[k] = rprime(a, k, c, i0 + t);
#pragma unroll
            for (int k = 0; k < T; ++k) xs[k][t] = r[k] * wn;
        } else if (t < H_SL) {   // a narrower slice: zeros, so the dots' last unit adds nothing
#pragma unroll
            for (int k = 0; k < T; ++k) xs[k][t] = 0.f;
        }
        if (grp == 0) {
#pragma unroll
            for (int k = 0; k < T; ++k) {
                const float v = wsum(r[k] * r[k]);
                if (lane == 0) red[k][warp] = v;
            }
        }
    }
    __syncthreads();
    if (grp == 0 && t < T) {
        float v = 0.f;
        for (int w = 0; w < H_DT / 32; ++w) v += red[t][w];
        a.ssp[(size_t) t * H_NS + s] = v;
    }
    for (int q = warp; q < T * (H_SL / 32); q += H_DT / 32) {
        const int k = q / (H_SL / 32), blk = q % (H_SL / 32);
        if constexpr (H16) h_block(xs[k][blk * 32 + lane], &xq[k][blk * 32], &xd[k][blk]);
        else q8_block(xs[k][blk * 32 + lane], &xq[k][blk * 32], &xd[k][blk]);
    }
    __syncthreads();
    {
        constexpr int NQ = H_SL / 8 / H_TPRD;
        const int j = t % H_TPRD, row = grp * H_R0 + t / H_TPRD;
        const int nrows = a.w_inj ? H_ROWS : LR;
        const bool live = row < (grp ? nrows : H_R0);   // a lane past the group's rows has nothing to do
        const int rr = live ? row : 0;
        // the thread's weights to registers, then one pass per row t (rolled: unrolled over T the kernel is tens of KB
        // of code, fetched cold at every launch)
        if (!live) {
        } else if (rr < LR) {
            const uint2* w2 = reinterpret_cast<const uint2*>(a.w_down.q + (size_t) rr * D + col0);
            const uint16_t* wd = a.w_down.d + ((size_t) rr * D + col0) / 32;
            uint2 u[NQ];
            float sc[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                u[q] = __ldg(w2 + q * H_TPRD + j);
                sc[q] = f16(__ldg(wd + (q * H_TPRD + j) / 4));
            }
#pragma unroll 1
            for (int k = 0; k < T; ++k) {
                float acc = 0.f;
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const int ch = q * H_TPRD + j;
                    if constexpr (H16) acc = fmaf(sc[q] * xd[k][ch / 4], dot8h(u[q], *reinterpret_cast<const uint4*>(&xq[k][ch * 8])), acc);
                    else acc = fmaf(sc[q] * xd[k][ch / 4], (float) dot8i<hc_x(T)>(u[q], *reinterpret_cast<const int2*>(&xq[k][ch * 8])), acc);
                }
                for (int o = 1; o < H_TPRD; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (live && j == 0) a.part[((size_t) s * T + k) * H_ROWS + row] = acc;
            }
        } else {
            const uint4* w4 = reinterpret_cast<const uint4*>(a.w_inj + (size_t) (rr - LR) * D + col0);
            uint4 u[NQ];
#pragma unroll
            for (int q = 0; q < NQ; ++q) u[q] = __ldg(w4 + q * H_TPRD + j);
#pragma unroll 1
            for (int k = 0; k < T; ++k) {
                float acc = 0.f;
#pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    const int ch = q * H_TPRD + j;
                    const uint16_t* hh = reinterpret_cast<const uint16_t*>(&u[q]);
                    const float4 x0 = *reinterpret_cast<const float4*>(&xs[k][ch * 8]);
                    const float4 x1 = *reinterpret_cast<const float4*>(&xs[k][ch * 8 + 4]);
                    acc += bf(hh[0]) * x0.x + bf(hh[1]) * x0.y + bf(hh[2]) * x0.z + bf(hh[3]) * x0.w + bf(hh[4]) * x1.x +
                           bf(hh[5]) * x1.y + bf(hh[6]) * x1.z + bf(hh[7]) * x1.w;
                }
                for (int o = 1; o < H_TPRD; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (live && j == 0) a.part[((size_t) s * T + k) * H_ROWS + row] = acc;
            }
        }
    }
    } while (WALK && (vb += grid) < H_BLOCKS);
    // the up weights of share vb's 32 columns: lane uj < H_UB of column udd takes block uj of each stream's row. The
    // block's first share's are in flight through the reduction (nothing reads them before the barrier, so nothing
    // waits for them); a later share's load after the last share's up part
    const int uj = t % H_TPRU, udd = t / H_TPRU;
    const bool ulive = t < H_UPC * H_TPRU && uj < H_UB;
    static_assert(H_UPC * H_TPRU < H_DT, "the block's last thread has no up lane (it runs the grid barrier)");
    uint4 wq[HC][2] = {};
    uint16_t wsh[HC] = {};
    const auto load_up = [&](int vb) {
        if (ulive)
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                const size_t urow = (size_t) cc * N + vb * H_UPC + udd;
                const uint4* wp = reinterpret_cast<const uint4*>(a.w_up.q + urow * LR + uj * 32);
                wq[cc][0] = __ldg(wp);
                wq[cc][1] = __ldg(wp + 1);
                wsh[cc] = __ldg(a.w_up.d + urow * (LR / 32) + uj);
            }
    };
    load_up(blockIdx.x);
    if ((a.up2 & 2) && ulive)
        for (int v2 = (int) blockIdx.x + (int) grid; v2 < H_UPS; v2 += (int) grid)
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                const size_t urow = (size_t) cc * N + v2 * H_UPC + udd;
                const int8_t* wp = a.w_up.q + urow * LR + uj * 32;
                asm volatile("prefetch.global.L2 [%0];" :: "l"(wp));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(wp + 16));
                if (uj == 0) asm volatile("prefetch.global.L2 [%0];" :: "l"(a.w_up.d + urow * (LR / 32)));
            }
    grid_sync(a.bar, a.err, grid);
    if constexpr (ONE) {
        // ---------------- reduce, in every block: rs, then lo per (token k, lo block b) a warp into lq / ld
        if (t < T * HC) rs[t / HC][t % HC] = rs_of(a, t / HC, t % HC);
        __syncthreads();
        for (int it = warp; it < T * (LR / 32); it += H_DT / 32) {
            const int k = it / (LR / 32), r = (it % (LR / 32)) * 32 + lane;
            float pv[H_NS];
#pragma unroll
            for (int q = 0; q < H_NS; ++q) pv[q] = __ldcg(a.part + ((size_t) q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = fmaf(rs[k][cc], p, sum);
            }
            if constexpr (H16) h_block(silu_(sum / (float) HC), &lq[k][r & ~31], &ld[k][r / 32]);
            else q8_block(silu_(sum / (float) HC), &lq[k][r & ~31], &ld[k][r / 32]);
        }
        if (blockIdx.x == 0 && a.w_inj && warp < T && lane < HC) {
            const int k = warp, r = LR + lane;
            float pv[H_NS];
#pragma unroll
            for (int q = 0; q < H_NS; ++q) pv[q] = __ldcg(a.part + ((size_t) q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = fmaf(rs[k][cc], p, sum);
            }
            a.inject[k * HC + lane] = sum;
        }
        __syncthreads();
    } else {
    // ---------------- reduce: item (token k, lo block b) per warp, then each token's inject rows
    {
        constexpr int ITEMS = T * (LR / 32) + T;
        for (int it = blockIdx.x * (H_DT / 32) + warp; it < ITEMS; it += grid * (H_DT / 32)) {
            const bool inj_item = it >= T * (LR / 32);
            const int k = inj_item ? it - T * (LR / 32) : it / (LR / 32);
            // every lane runs the same path (q8_block shuffles across the warp); only an inject item's first HC lanes
            // have a row
            const bool active = !inj_item || (a.w_inj && lane < HC);
            const int r = !active ? 0 : inj_item ? LR + lane : (it % (LR / 32)) * 32 + lane;
            float pv[H_NS];
#pragma unroll
            for (int q = 0; q < H_NS; ++q) pv[q] = __ldcg(a.part + ((size_t) q * T + k) * H_ROWS + r);
            float sum = 0.f;
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                float p = 0.f;
#pragma unroll
                for (int ss = 0; ss < H_SPS; ++ss) p += pv[cc * H_SPS + ss];
                sum = fmaf(rs_of(a, k, cc), p, sum);
            }
            if (inj_item) {
                if (active) a.inject[k * HC + lane] = sum;
            } else if constexpr (H16) h_block(silu_(sum / (float) HC), reinterpret_cast<__half*>(a.loq) + k * LR + (r & ~31), a.lod + k * (LR / 32) + r / 32);
            else q8_block(silu_(sum / (float) HC), a.loq + k * LR + (r & ~31), a.lod + k * (LR / 32) + r / 32);
        }
    }
    grid_sync(a.bar, a.err, grid);
    // ---------------- up
    if (t < T * HC) rs[t / HC][t % HC] = rs_of(a, t / HC, t % HC);
    for (int e = t; e < T * LR * (int) sizeof(Lq) / 4; e += H_DT) reinterpret_cast<int*>(&lq[0][0])[e] = __ldcg(reinterpret_cast<const int*>(a.loq) + e);
    for (int e = t; e < T * (LR / 32); e += H_DT) ld[e / (LR / 32)][e % (LR / 32)] = __ldcg(a.lod + e);
    __syncthreads();
    }
    // ---------------- up, share vb = 32 columns b
    vb = blockIdx.x;
    do {
    if (vb != blockIdx.x) {
        __syncthreads();   // the last share is done with gs / ms
        if (!(a.up2 & 1)) load_up(vb);   // FNX_HC_UP2 1: loaded after the last share's dots
    }
    float wsc[HC];
#pragma unroll
    for (int cc = 0; cc < HC; ++cc) wsc[cc] = ulive ? f16(wsh[cc]) : 0.f;
    // one pass per row k, rolled (see the down part): the lane's block of lo against its block of each stream's row
#pragma unroll 1
    for (int k = 0; k < T; ++k) {
        const float dl = ld[k][ulive ? uj : 0];
        float acc[HC];
        if constexpr (H16) {
            const uint4* xp = reinterpret_cast<const uint4*>(&lq[k][(ulive ? uj : 0) * 32]);
            const uint4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
#pragma unroll
            for (int cc = 0; cc < HC; ++cc) {
                const uint4 w0 = wq[cc][0], w1 = wq[cc][1];
                const float s = dot8h(make_uint2(w0.x, w0.y), x0) + dot8h(make_uint2(w0.z, w0.w), x1) + dot8h(make_uint2(w1.x, w1.y), x2) +
                                dot8h(make_uint2(w1.z, w1.w), x3);
                acc[cc] = wsc[cc] * dl * s;
            }
        } else {
        const int4* xp = reinterpret_cast<const int4*>(&lq[k][(ulive ? uj : 0) * 32]);
        const int4 x0 = xp[0], x1 = xp[1];
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) {
            const uint4 w0 = wq[cc][0], w1 = wq[cc][1];
            int sacc = dp4a_x<hc_x(T)>((int) w0.x, x0.x, 0);
            sacc = dp4a_x<hc_x(T)>((int) w0.y, x0.y, sacc);
            sacc = dp4a_x<hc_x(T)>((int) w0.z, x0.z, sacc);
            sacc = dp4a_x<hc_x(T)>((int) w0.w, x0.w, sacc);
            sacc = dp4a_x<hc_x(T)>((int) w1.x, x1.x, sacc);
            sacc = dp4a_x<hc_x(T)>((int) w1.y, x1.y, sacc);
            sacc = dp4a_x<hc_x(T)>((int) w1.z, x1.z, sacc);
            sacc = dp4a_x<hc_x(T)>((int) w1.w, x1.w, sacc);
            acc[cc] = wsc[cc] * dl * (float) sacc;
        }
        }
#pragma unroll
        for (int cc = 0; cc < HC; ++cc) {
#pragma unroll
            for (int o = 1; o < H_TPRU; o <<= 1) acc[cc] += __shfl_xor_sync(0xffffffffu, acc[cc], o);
            if (t < H_UPC * H_TPRU && uj == 0) gs[k][cc][udd] = acc[cc];   // the sigmoid below, once per element on
        }                                                                      // every lane (here a warp issued it for 2)
    }
    if ((a.up2 & 1) && vb + (int) grid < H_UPS) load_up(vb + (int) grid);   // the share's own scales are in wsc already
    __syncthreads();
    // item (k, c, col): this block's 32 columns of stream c, row k, coalesced along the columns
    const int b = vb;
    for (int e = t; e < T * HC * H_UPC; e += H_DT) {
        const int col = e % H_UPC, cc = (e / H_UPC) % HC, k = e / (H_UPC * HC), dc = b * H_UPC + col;
        const float rv = rprime(a, k, cc, dc);
        if (a.bo) a.R[(size_t) k * D + cc * N + dc] = rv;
        gs[k][cc][col] = sigm(gs[k][cc][col]) * (rv * a.w_norm[cc * N + dc] * rs[k][cc]);
    }
    __syncthreads();
    for (int e = t; e < T * H_UPC; e += H_DT) {
        const int col = e % H_UPC, k = e / H_UPC;
        const float sum = (gs[k][0][col] + gs[k][1][col] + gs[k][2][col] + gs[k][3][col]) / (float) HC;
        a.mixed[(size_t) k * N + b * H_UPC + col] = sum;
        ms[k][col] = sum;
    }
    if (!a.q8) continue;
    __syncthreads();
    static_assert(H_UPC == 32, "one q8_1 block per token per up block");
    if (warp < T) {
        const float xi = ms[warp][lane];
        const float amax = wmax(fabsf(xi)), sm = wsum(xi);
        const float dq = amax / 127.0f;
        block_q8_1& yq = a.q8[(size_t) warp * (N / 32) + b];
        const int8_t qv = amax == 0.0f ? 0 : (int8_t) roundf(xi / dq);
        yq.qs[lane] = qv;
        if (lane == 0) yq.ds = make_half2(__float2half(dq), __float2half(sm));
        if (a.tq) {
            a.tq[(size_t) warp * N + b * 32 + lane] = qv;
            if (lane == 0) a.td[(size_t) warp * (N / 32) + b] = __half2float(__float2half(dq));
        }
    }
    } while ((vb += grid) < H_UPS);
}

// one: k_hc's one-barrier form (built for windows of up to 8 rows); grid: the card's, from hc_prepare
template <int T, bool H16> void hc_launch(const HcArgs& a, cudaStream_t st, bool one, int grid) {
    if constexpr (T <= 8) {
        if (one) {
            if (grid < H_BLOCKS) k_hc<T, true, true, H16><<<grid, H_DT, 0, st>>>(a);
            else k_hc<T, true, false, H16><<<grid, H_DT, 0, st>>>(a);
            return;
        }
    }
    if (grid < H_BLOCKS) k_hc<T, false, true, H16><<<grid, H_DT, 0, st>>>(a);
    else k_hc<T, false, false, H16><<<grid, H_DT, 0, st>>>(a);
}
template <int T> void hc_launch(const HcArgs& a, cudaStream_t st, bool one, int grid, bool h16) {
    if (h16) hc_launch<T, true>(a, st, one, grid);
    else hc_launch<T, false>(a, st, one, grid);
}

}  // namespace
