// QSA: the attention layers' block selection (included by kernels.cu), as llama.cpp's qwen4exp graph builds it.
// Blocks of QSA_POOL positions in order; a block's key is the mean of its members' raw indexer keys, RMS-normed and
// rotated to its first member's position, kept in f16 (the cache type). A row at position p sees the complete
// blocks [0, (p + 1) / 4) and always attends its tail, the last (p + 1) % 4 positions; it scores the visible blocks
// with its indexer query (4 heads, each RMS-normed and rotated to p): sum over heads of relu(q_h . k) / sqrt(128),
// and keeps the QSA_TOP highest (the lower block on a tie; llama.cpp's CUDA top-k takes tied blocks in atomic order,
// so any choice among exact ties is one it can make). With QSA_TOP or fewer visible blocks every block is kept: the
// attention is dense. Every card computes the same selection from the same inputs.
namespace {

constexpr int QSA_POOL = 4, QSA_TOP = QSA_TOP_K / QSA_POOL, QSA_DIM = QSA_KEY, QSA_ROT = NROT;
static_assert(QSA_DIM % 16 == 0 && QSA_ROT <= QSA_DIM && QSA_TOP_K % QSA_POOL == 0,
              "the score tiles take QSA_DIM in sixteens, the rotary must fit a key, and top_k must divide into whole pools");
constexpr int QSA_ROWS = QSA_HEADS * QSA_DIM + QSA_DIM;   // the projection's rows: q (QSA_HEADS heads), then k

constexpr int QSA_MAXSEL = QSA_TOP * QSA_POOL + QSA_POOL - 1;

__device__ __forceinline__ float bf16f(uint16_t h) { return __uint_as_float((unsigned) h << 16); }

// y[t][r] = W_r . x[t] for the QSA_ROWS rows (r < QSA_HEADS * QSA_DIM: wq row r, else wk row r - that many), bf16 weights,
// f32 activations; a warp per row
template <int T>
__global__ void __launch_bounds__(256) k_idx_proj(const uint16_t* __restrict__ wq, const uint16_t* __restrict__ wk,
                                                 const float* __restrict__ x, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, r = blockIdx.x * 8 + warp;
    if (r >= QSA_ROWS) return;
    const uint16_t* w = r < QSA_HEADS * QSA_DIM ? wq + (size_t) r * N : wk + (size_t) (r - QSA_HEADS * QSA_DIM) * N;
    float acc[T] = {};
#pragma unroll 4
    for (int c = lane * 2; c < N; c += 64) {
        const unsigned wv = *reinterpret_cast<const unsigned*>(w + c);
        const float w0 = bf16f((uint16_t) (wv & 0xffffu)), w1 = bf16f((uint16_t) (wv >> 16));
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float2 xv = *reinterpret_cast<const float2*>(x + (size_t) t * N + c);
            acc[t] = fmaf(w0, xv.x, fmaf(w1, xv.y, acc[t]));
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * QSA_ROWS + r] = v;
    }
}

// x (QSA_DIM values, one per thread): RMS norm with w, then NEOX rotary on the first QSA_ROT dims at pos
__device__ __forceinline__ float idx_norm_rope(float v, const float* w, int pos, float eps, const Rope& base, float* sh, float* buf) {
    const int i = threadIdx.x;
    const float ms = block_sum(v * v, sh) / (float) QSA_DIM;
    const float y = v * rsqrtf(ms + eps) * w[i];
    buf[i] = y;
    __syncthreads();
    float out = y;
    if (i < QSA_ROT) {
        const int k = i % (QSA_ROT / 2);
        const float theta = (float) rope_pos(base, pos, k) * rope_unit_freq(base, k, QSA_ROT);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        sn *= base.mscale; cs *= base.mscale;
        const float x0 = buf[k], x1 = buf[k + QSA_ROT / 2];
        out = i < QSA_ROT / 2 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    __syncthreads();
    return out;
}

// block (row t, part): part < 4: indexer query head `part` normed and rotated into q; part 4: the raw key into the ring
__global__ void __launch_bounds__(QSA_DIM) k_idx_prep(const float* __restrict__ y, const float* __restrict__ qn, __half* ring, int ring_rows,
                                                      float* __restrict__ q, const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.x, part = blockIdx.y, i = threadIdx.x, p = *pos0p + t;
    __shared__ float sh[32];
    __shared__ float buf[QSA_DIM];
    const float* yt = y + (size_t) t * QSA_ROWS;
    if (part < QSA_HEADS) {
        q[(size_t) t * QSA_HEADS * QSA_DIM + part * QSA_DIM + i] =
            idx_norm_rope(yt[part * QSA_DIM + i], qn, p, eps, base, sh, buf);
    } else {
        ring[(size_t) (p % ring_rows) * QSA_DIM + i] = __float2half_rn(yt[QSA_HEADS * QSA_DIM + i]);
    }
}

// a row whose position completes a block pools it: the mean of the members' raw keys (f16), normed, rotated to the
// first member, stored f16
__global__ void __launch_bounds__(QSA_DIM) k_idx_pool(const __half* ring, int ring_rows, const float* __restrict__ kn, __half* pooled, KvPages g,
                                                      const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.x, i = threadIdx.x, p = *pos0p + t;
    if ((p + 1) % QSA_POOL != 0) return;
    __shared__ float sh[32];
    __shared__ float buf[QSA_DIM];
    float v = 0.f;
#pragma unroll
    for (int m = 0; m < QSA_POOL; ++m) v += __half2float(ring[(size_t) ((p - (QSA_POOL - 1) + m) % ring_rows) * QSA_DIM + i]);
    v *= 1.0f / (float) QSA_POOL;
    const int j = (p + 1) / QSA_POOL - 1;
    pooled[kvp_block(g, j) + i] = __float2half_rn(idx_norm_rope(v, kn, p - (QSA_POOL - 1), eps, base, sh, buf));
}

// k_idx_prep and k_idx_pool in one launch: parts 0 .. QSA_HEADS as k_idx_prep's, part QSA_HEADS + 1 k_idx_pool's, which
// takes the window's own raw keys from y (rounded to f16 as the ring holds them) and the earlier ones from the ring
__global__ void __launch_bounds__(QSA_DIM) k_idx_prep_pool(const float* __restrict__ y, const float* __restrict__ qn,
                                                           const float* __restrict__ kn, __half* ring, int ring_rows, float* __restrict__ q,
                                                           __half* pooled, KvPages g, const int* pos0p, float eps, Rope base) {
    const int t = blockIdx.x, part = blockIdx.y, i = threadIdx.x, pos0 = *pos0p, p = pos0 + t;
    __shared__ float sh[32];
    __shared__ float buf[QSA_DIM];
    const float* yt = y + (size_t) t * QSA_ROWS;
    if (part < QSA_HEADS) {
        q[(size_t) t * QSA_HEADS * QSA_DIM + part * QSA_DIM + i] = idx_norm_rope(yt[part * QSA_DIM + i], qn, p, eps, base, sh, buf);
    } else if (part == QSA_HEADS) {
        ring[(size_t) (p % ring_rows) * QSA_DIM + i] = __float2half_rn(yt[QSA_HEADS * QSA_DIM + i]);
    } else {
        if ((p + 1) % QSA_POOL != 0) return;
        float v = 0.f;
#pragma unroll
        for (int m = 0; m < QSA_POOL; ++m) {
            const int pm = p - (QSA_POOL - 1) + m;
            v += __half2float(pm >= pos0 ? __float2half_rn(y[(size_t) (pm - pos0) * QSA_ROWS + QSA_HEADS * QSA_DIM + i])
                                         : ring[(size_t) (pm % ring_rows) * QSA_DIM + i]);
        }
        v *= 1.0f / (float) QSA_POOL;
        const int j = (p + 1) / QSA_POOL - 1;
        pooled[kvp_block(g, j) + i] = __float2half_rn(idx_norm_rope(v, kn, p - (QSA_POOL - 1), eps, base, sh, buf));
    }
}

// scores[t][j] for the visible blocks j of rows with more than QSA_TOP of them: a thread per block
__global__ void __launch_bounds__(256) k_idx_score(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                   float* __restrict__ scores, int max_pools, const int* pos0p, LaneRows lr) {
    const int t = blockIdx.y, p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL;
    if (nv <= QSA_TOP) return;
    g = row_pages(g, lr, t);
    __shared__ float qs[QSA_HEADS * QSA_DIM];
    for (int k = threadIdx.x; k < QSA_HEADS * QSA_DIM; k += blockDim.x) qs[k] = q[(size_t) t * QSA_HEADS * QSA_DIM + k];
    __syncthreads();
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nv) return;
    const uint4* kr = reinterpret_cast<const uint4*>(pooled + kvp_block(g, j));
    float d[QSA_HEADS] = {};
#pragma unroll 4
    for (int c = 0; c < QSA_DIM / 8; ++c) {
        const uint4 u = kr[c];
        const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 kv = __half22float2(h2[k]);
            const int dim = c * 8 + 2 * k;
#pragma unroll
            for (int h = 0; h < QSA_HEADS; ++h) d[h] = fmaf(qs[h * QSA_DIM + dim], kv.x, fmaf(qs[h * QSA_DIM + dim + 1], kv.y, d[h]));
        }
    }
    const float w = rsqrtf((float) QSA_DIM);
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < QSA_HEADS; ++h) s += fmaxf(d[h], 0.f) * w;
    scores[(size_t) t * max_pools + j] = s;
}

// k_idx_score in tiles (FNX_QSA_TILE=1, the P100 port). k_idx_score reads a key again for every row and a query value from shared
// memory for every multiply (~0.5 T MAC/s: at a 130k position a quarter and at 400k over a half of the prompt path's time, and a
// tenth and more of a decode window's). Here a thread scores R rows by P pools at once: a key is read once for R rows and a row's
// query values once for P pools. A (row, pool, head)'s sum runs over the dims in k_idx_score's order with its fmaf pairs and the
// heads' relus are summed as there, so the same scores bit for bit (mikro/khc/qsa_adapt_test.cu compares them).
// The best tile depends on the rows and the position, and a decode window's launches are captured in a graph per T, so the
// kernel picks the tile itself, from T and the window's position: every block of a launch picks the same one. The grid is a
// count of blocks from T and the context only; a tile of rows gets an equal share of them and its threads stride over the pools
// the tile's last row sees. One position for all rows (*pos0p + t): not for lanes (k_idx_score there).
// The rows a thread are as measured with the rows forced (qsa_adapt_test.cu's f columns; mikro/khc/pravilo.py scores a rule on
// them). A prompt's group (T > 16): 4 everywhere (8 are no faster even with the keys far past the L2). A decode window: 8 only
// where the keys are far past the L2 and 8 rows leave half the tiles of 4; else 4 from QSA_TILE_FAR pools; nearer whichever of
// 1, 2, 4 takes the least time by the waves of threads it needs (a wave: the threads the card runs at once; about 9, 19 and
// 35 us for 1, 2 and 4 rows a thread), the more rows on a tie.
constexpr int QSA_TILE_WAVE = 13824;   // threads
constexpr int QSA_TILE_FAR = 16384;    // pools
__host__ __device__ __forceinline__ int qsa_tile_rows(int T, int nv) {
    if (T <= 2) return 1;
    if (T <= 4) return T * nv <= 45000 ? 1 : 4;
    if (T > 16) return 4;
    if (nv >= ((((T - 1) >> 2) & 1) ? 24576 : 81920)) return 8;   // T 5..8 and 13..16: two tiles of 4 become one of 8
    if (nv >= QSA_TILE_FAR) return 4;
    const int h = (nv + 1) / 2;   // threads a tile of 2 pools a thread keeps busy
    const int c1 = ((T * nv + QSA_TILE_WAVE - 1) / QSA_TILE_WAVE) * 9;
    const int c2 = ((((T + 1) / 2) * h + QSA_TILE_WAVE - 1) / QSA_TILE_WAVE) * 19;
    const int c4 = ((((T + 3) / 4) * h + QSA_TILE_WAVE - 1) / QSA_TILE_WAVE) * 35;
    return c4 <= c2 && c4 <= c1 ? 4 : c2 <= c1 ? 2 : 1;
}

// the tile's R rows (from t0; `rows` of them real, the rest zeros in qs) against the pools j0, j0 + step, ... below nvm, P at a time
template <int R, int P>
__device__ __forceinline__ void idx_score_tile(const float* qs, const __half* __restrict__ pooled, const KvPages& g, float* __restrict__ scores,
                                               int max_pools, int t0, int rows, int p0, int nvm, int j_first, int step) {
    const float w = rsqrtf((float) QSA_DIM);
    for (int j0 = j_first; j0 < nvm; j0 += step) {
        const uint4* kr[P];
#pragma unroll
        for (int a = 0; a < P; ++a) kr[a] = reinterpret_cast<const uint4*>(pooled + kvp_block(g, min(j0 + a, nvm - 1)));
        float d[R][P][QSA_HEADS];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int a = 0; a < P; ++a)
#pragma unroll
                for (int h = 0; h < QSA_HEADS; ++h) d[r][a][h] = 0.f;
#pragma unroll 2
        for (int c = 0; c < QSA_DIM / 8; ++c) {
            float2 kv[P][4];
#pragma unroll
            for (int a = 0; a < P; ++a) {
                const uint4 u = kr[a][c];
                const __half2* h2 = reinterpret_cast<const __half2*>(&u);
#pragma unroll
                for (int k = 0; k < 4; ++k) kv[a][k] = __half22float2(h2[k]);
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int dim = c * 8 + 2 * k;
#pragma unroll
                for (int r = 0; r < R; ++r)
#pragma unroll
                    for (int h = 0; h < QSA_HEADS; ++h) {
                        const float qx = qs[(r * QSA_HEADS + h) * QSA_DIM + dim], qy = qs[(r * QSA_HEADS + h) * QSA_DIM + dim + 1];
#pragma unroll
                        for (int a = 0; a < P; ++a) d[r][a][h] = fmaf(qx, kv[a][k].x, fmaf(qy, kv[a][k].y, d[r][a][h]));
                    }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int nv = (p0 + r + 1) / QSA_POOL;   // the row's own pools, as k_idx_score's
            if (r >= rows || nv <= QSA_TOP) continue;
#pragma unroll
            for (int a = 0; a < P; ++a) {
                float s = 0.f;
#pragma unroll
                for (int h = 0; h < QSA_HEADS; ++h) s += fmaxf(d[r][a][h], 0.f) * w;
                if (j0 + a < nv) scores[(size_t) (t0 + r) * max_pools + j0 + a] = s;
            }
        }
    }
}

// RMAX: the most rows a thread (a power of two, 8 at most; no less than qsa_tile_rows gives for the launch's T: 1 for T <= 2,
// 4 for T <= 4, else 8; the instance sizes the shared queries and what the compiler has to keep in registers). The grid: no
// fewer blocks than the next power of two from T, so every tile has one (a tile's share is a power of two: shifts, no division,
// since a block with nothing to score should leave as fast as it can).
template <int RMAX>
__global__ void __launch_bounds__(256) k_idx_score_a(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                     float* __restrict__ scores, int max_pools, const int* pos0p, int T) {
    const int pos0 = *pos0p;
    int R = qsa_tile_rows(T, (pos0 + T) / QSA_POOL);
    if (R > RMAX) R = RMAX;
    const int lr = 31 - __clz(R), P = R > 1 ? 2 : 1, tiles = (T + R - 1) >> lr;
    const int lb = (31 - __clz((int) gridDim.x)) - (32 - __clz(tiles - 1));   // a tile's share of the blocks: 1 << lb
    const int ti = blockIdx.x >> lb, bi = blockIdx.x & ((1 << lb) - 1);
    if (ti >= tiles) return;
    const int t0 = ti * R, rows = min(R, T - t0), p0 = pos0 + t0;
    const int nvm = (p0 + rows) / QSA_POOL;   // the tile's last row sees the most pools
    if (nvm <= QSA_TOP || bi * (int) blockDim.x * P >= nvm) return;   // the whole block: no pool of the tile's to score
    __shared__ float qs[RMAX * QSA_HEADS * QSA_DIM];
    for (int k = threadIdx.x; k < R * QSA_HEADS * QSA_DIM; k += blockDim.x)
        qs[k] = k < rows * QSA_HEADS * QSA_DIM ? q[(size_t) t0 * QSA_HEADS * QSA_DIM + k] : 0.f;
    __syncthreads();
    const int j0 = (bi * (int) blockDim.x + (int) threadIdx.x) * P, step = ((int) blockDim.x << lb) * P;
    switch (R) {
        case 1: idx_score_tile<1, 1>(qs, pooled, g, scores, max_pools, t0, rows, p0, nvm, j0, step); break;
        case 2:
            if (RMAX >= 2) idx_score_tile<2, 2>(qs, pooled, g, scores, max_pools, t0, rows, p0, nvm, j0, step);
            break;
        case 4:
            if (RMAX >= 4) idx_score_tile<4, 2>(qs, pooled, g, scores, max_pools, t0, rows, p0, nvm, j0, step);
            break;
        default:
            if (RMAX >= 8) idx_score_tile<8, 2>(qs, pooled, g, scores, max_pools, t0, rows, p0, nvm, j0, step);
            break;
    }
}

#if FNX_ARCH >= 70
// the same for a prompt's rows on the tensor cores: a block takes 16 rows (their 4 heads: 64 rows of q, in f16 high and
// low halves) against 64 blocks' keys (f16 already), f32 accumulation; each key is read once per 16 rows instead of once
// per row. Columns past a row's visible blocks get scores too (the select reads only the visible ones)
constexpr int QSC_ROWS = 16, QSC_KEYS = 64, QSC_LD = QSA_DIM + 8, QSC_SLD = 16 + 4;
__global__ void __launch_bounds__(256) k_idx_score_tc(const float* __restrict__ q, const __half* __restrict__ pooled, KvPages g,
                                                      float* __restrict__ scores, int max_pools, const int* pos0p, int T) {
    using namespace nvcuda;
    const int t0 = blockIdx.y * QSC_ROWS, j0 = blockIdx.x * QSC_KEYS, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int tl = min(T, t0 + QSC_ROWS) - 1;   // the block's last row sees the most blocks
    const int nv_max = (*pos0p + tl + 1) / QSA_POOL;
    if (nv_max <= QSA_TOP || j0 >= nv_max) return;
    __shared__ __align__(32) __half qh[QSC_ROWS * QSA_HEADS][QSC_LD];
    __shared__ __align__(32) __half ql[QSC_ROWS * QSA_HEADS][QSC_LD];
    __shared__ __align__(32) float st[8][16][QSC_SLD];
    for (int e = tid; e < QSC_ROWS * QSA_HEADS * QSA_DIM; e += 256) {
        const int r = e / QSA_DIM, d = e % QSA_DIM;
        const float x = t0 + r / QSA_HEADS < T ? q[(size_t) t0 * QSA_HEADS * QSA_DIM + e] : 0.f;
        const __half h = __float2half_rn(x);
        qh[r][d] = h;
        ql[r][d] = __float2half_rn(x - __half2float(h));
    }
    __syncthreads();
    const float w = rsqrtf((float) QSA_DIM);
    // 16 output tiles (4 of rows x 4 of keys); warp w takes row tile w / 2, key tiles 2 (w % 2) and 2 (w % 2) + 1
    const int rt = warp / 2;
    for (int kk = 0; kk < 2; ++kk) {
        const int kt = (warp % 2) * 2 + kk, jt = j0 + kt * 16;
        if (jt >= nv_max) break;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs;
        wmma::fill_fragment(fs, 0.f);
#pragma unroll
        for (int d0 = 0; d0 < QSA_DIM; d0 += 16) {
            wmma::load_matrix_sync(fb, pooled + kvp_block(g, jt) + d0, QSA_DIM);   // 16 blocks never cross a page (48)
            wmma::load_matrix_sync(fa, &qh[rt * 16][d0], QSC_LD);
            wmma::mma_sync(fs, fa, fb, fs);
            wmma::load_matrix_sync(fa, &ql[rt * 16][d0], QSC_LD);
            wmma::mma_sync(fs, fa, fb, fs);
        }
        wmma::store_matrix_sync(&st[warp][0][0], fs, QSC_SLD, wmma::mem_row_major);
        __syncwarp();
        // the tile's 4 rows x 16 keys: the heads' relus summed in k_idx_score's order
        for (int e = lane; e < 4 * 16; e += 32) {
            const int r = e / 16, c = e % 16, t = t0 + rt * 4 + r, j = jt + c;
            if (t >= T || j >= max_pools) continue;
            float s = 0.f;
#pragma unroll
            for (int h = 0; h < QSA_HEADS; ++h) s += fmaxf(st[warp][r * QSA_HEADS + h][c], 0.f) * w;
            scores[(size_t) t * max_pools + j] = s;
        }
        __syncwarp();
    }
}
#endif  // FNX_ARCH >= 70

// row t's attended positions on this card (block j owned by card kv_card(j)), as local indices kv_local_block(j) * 4 + p % 4, ascending,
// into cells[t] (n_cells[t] of them), or n_cells[t] = -1 when every visible block is kept (dense). A block per row: a radix select of the QSA_TOP-th highest score (scores >= 0: their bits
// order as unsigned), then the blocks above it and the first ties by index, in index order, then the tail.
constexpr int QSA_SEL_THREADS = 1024;

__global__ void __launch_bounds__(QSA_SEL_THREADS) k_idx_select(const float* __restrict__ scores, int* __restrict__ cells,
                                                                int* __restrict__ n_cells, int max_pools, const int* pos0p,
                                                                int card, LaneRows lr) {
    const int t = blockIdx.x, tid = threadIdx.x, p = row_pos(lr, pos0p, t), nv = (p + 1) / QSA_POOL, n_tail = (p + 1) % QSA_POOL;
    if (nv <= QSA_TOP) {
        if (tid == 0) n_cells[t] = -1;
        return;
    }
    const unsigned* key = reinterpret_cast<const unsigned*>(scores + (size_t) t * max_pools);
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    unsigned prefix = 0, mask = 0, want = QSA_TOP;   // find the want-th largest among keys matching prefix on mask
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += QSA_SEL_THREADS) hist[b] = 0;
        __syncthreads();
        // a warp's keys of one bin counted with one atomic (scores share their high bits: per key, the first pass's
        // atomics all hit two or three bins and queue)
        for (int j0 = 0; j0 < nv; j0 += QSA_SEL_THREADS) {
            const int j = j0 + tid;
            const unsigned k = j < nv ? key[j] : 0u;
            const bool in = j < nv && (k & mask) == prefix;
            const unsigned bin = in ? (k >> shift) & 255u : 256u;
#if FNX_HIP
            const unsigned peers = (unsigned) __match_any_sync(0xffffffffull, bin);   // HIP wants a 64-bit mask (wave32: the high half is empty)
#elif FNX_ARCH >= 70
            const unsigned peers = __match_any_sync(0xffffffffu, bin);
#endif
#if FNX_HIP || FNX_ARCH >= 70
            if (in && (tid & 31) == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned) __popc(peers));
#else
            if (in) atomicAdd(&hist[bin], 1u);
#endif
        }
        __syncthreads();
        if (tid < 32) {   // the highest bin b > 0 where the count from 255 down reaches want, else 0; acc: the count above b
            const int lane = tid, hi = 255 - 8 * lane;   // lane l: bins hi .. hi - 7
            unsigned own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) own += hist[hi - i];
            unsigned before = own;   // inclusive scan over the lanes (higher bins first), then exclusive
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned v = __shfl_up_sync(0xffffffffu, before, o);
                if (lane >= o) before += v;
            }
            before -= own;
            const unsigned hit = __ballot_sync(0xffffffffu, before + own >= want);
            const int first = hit ? __ffs(hit) - 1 : 31;
            if (lane == first) {
                unsigned acc = before;
                int b = hi - 7;
                for (int i = 0; i < 8; ++i) {
                    const int bb = hi - i;
                    if (bb == 0 || acc + hist[bb] >= want) {
                        b = bb;
                        break;
                    }
                    acc += hist[bb];
                }
                prefix_s = prefix | ((unsigned) b << shift);
                want_s = want - acc;
            }
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    // prefix: the threshold key; `want` ties at it are kept (the first by index). Contiguous chunks per thread
    const unsigned thr = prefix;
    const int chunk = (nv + QSA_SEL_THREADS - 1) / QSA_SEL_THREADS, j0 = min(nv, tid * chunk), j1 = min(nv, j0 + chunk);
    int n_gt = 0, n_eq = 0;
    for (int j = j0; j < j1; ++j) {
        const unsigned k = key[j];
        n_gt += k > thr;
        n_eq += k == thr;
    }
    __shared__ int s_gt[QSA_SEL_THREADS], s_eq[QSA_SEL_THREADS];
    s_gt[tid] = n_gt;
    s_eq[tid] = n_eq;
    __syncthreads();
    // exclusive scans (Hillis-Steele over the block)
    for (int o = 1; o < QSA_SEL_THREADS; o <<= 1) {
        const int a = tid >= o ? s_gt[tid - o] : 0, e = tid >= o ? s_eq[tid - o] : 0;
        __syncthreads();
        s_gt[tid] += a;
        s_eq[tid] += e;
        __syncthreads();
    }
    // which blocks of the chunk are taken (ties: the first `want` by index), then how many of them this card owns
    const int eq_start = s_eq[tid] - n_eq;
    int own = 0;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            own += take && kv_card(j) == card;
        }
    }
    __syncthreads();
    s_gt[tid] = own;
    __syncthreads();
    for (int o = 1; o < QSA_SEL_THREADS; o <<= 1) {
        const int a = tid >= o ? s_gt[tid - o] : 0;
        __syncthreads();
        s_gt[tid] += a;
        __syncthreads();
    }
    int out = s_gt[tid] - own;   // owned selected blocks before this chunk
    int* row = cells + (size_t) t * QSA_MAXSEL;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            if (take && kv_card(j) == card) {
#pragma unroll
                for (int m = 0; m < QSA_POOL; ++m) row[out * QSA_POOL + m] = kv_local_block(j) * QSA_POOL + m;
                ++out;
            }
        }
    }
    if (tid == QSA_SEL_THREADS - 1) {   // the last chunk's thread: `out` is the card's selected-block count
        int n = out * QSA_POOL;
        const int jt = p / QSA_POOL;   // the tail's block (incomplete)
        if (kv_card(jt) == card)
            for (int m = 0; m < n_tail; ++m) row[n++] = kv_local_block(jt) * QSA_POOL + (p - n_tail + 1 + m) % QSA_POOL;
        n_cells[t] = n;
    }
}

// k_idx_select with a bound and candidates (FNX_QSA_SEL=1, the P100 port). k_idx_select passes over a row's scores seven times:
// four with an atomic a key into 256 counters and three in a chunk a thread (a line a lane, so a warp's step touches 32 lines);
// at a 130k position that is 5.6 us a row and at 390k 22 us (a tenth and more of the prompt path's time there), with over 70
// barriers a row. Here three light passes, a line a warp's step each: (1) a thread's largest score over its strided share; the
// bound L is the least over the warps of a warp's 16th largest such maximum, so 32 x 16 = 512 scores at least are no less than L
// and every block k_idx_select would take is among the scores no less than L (about 1,100 of them whatever the row's length:
// mikro/khc/sim_meja.py); (2) the count of those a warp, a warp's scores being a contiguous range; (3) their keys and indices
// into shared memory in index order. Then k_idx_select's own radix steps, ties and output on the candidates alone, with the
// prefix sums a warp by shuffles (two barriers a sum). The 512th largest of the candidates is the 512th largest of the row and
// the ties at it keep their index order, so the same cells and count as k_idx_select's (mikro/khc/qsa_sel_test.cu compares
// them). A row with more candidates than QSA_SELC_CAP (scores nearly all equal) goes through k_idx_select's passes as they were.
// ALL: a row of no more scores than that skips (1) and (2), every score a candidate. One position for all rows (*pos0p + t):
// not for lanes (k_idx_select there).
constexpr int QSA_SELC_CAP = 3072;                                                       // candidates a row at most
constexpr int QSA_SELC_PER = (QSA_SELC_CAP + QSA_SEL_THREADS - 1) / QSA_SEL_THREADS;   // candidates a thread at most

// the block's exclusive prefix sum of v in thread order, and the sum (32 warps; s: 32 shared ints, free again at return)
__device__ __forceinline__ int selc_scan(int v, int* s, int& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inc = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += u;
    }
    if (lane == 31) s[warp] = inc;
    __syncthreads();
    const int w = s[lane];   // the warps' sums, one a lane
    int winc = w;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, winc, o);
        if (lane >= o) winc += u;
    }
    const int before = __shfl_sync(0xffffffffu, winc - w, warp);
    total = __shfl_sync(0xffffffffu, winc, 31);
    __syncthreads();
    return before + inc - v;
}

// k_idx_select's passes over all the row's scores (its text; hist: 256, ps: 2, s_gt and s_eq: QSA_SEL_THREADS shared words)
static __device__ __noinline__ void selc_all(const unsigned* __restrict__ key, int* __restrict__ row, int* __restrict__ n_cells, int t, int p,
                                      int card, unsigned* hist, unsigned* ps, int* s_gt, int* s_eq) {
    const int tid = threadIdx.x, nv = (p + 1) / QSA_POOL, n_tail = (p + 1) % QSA_POOL;
    unsigned prefix = 0, mask = 0, want = QSA_TOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += QSA_SEL_THREADS) hist[b] = 0;
        __syncthreads();
        for (int j0 = 0; j0 < nv; j0 += QSA_SEL_THREADS) {
            const int j = j0 + tid;
            const unsigned k = j < nv ? key[j] : 0u;
            if (j < nv && (k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid < 32) {
            const int lane = tid, hi = 255 - 8 * lane;
            unsigned own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) own += hist[hi - i];
            unsigned before = own;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned v = __shfl_up_sync(0xffffffffu, before, o);
                if (lane >= o) before += v;
            }
            before -= own;
            const unsigned hit = __ballot_sync(0xffffffffu, before + own >= want);
            const int first = hit ? __ffs(hit) - 1 : 31;
            if (lane == first) {
                unsigned acc = before;
                int b = hi - 7;
                for (int i = 0; i < 8; ++i) {
                    const int bb = hi - i;
                    if (bb == 0 || acc + hist[bb] >= want) {
                        b = bb;
                        break;
                    }
                    acc += hist[bb];
                }
                ps[0] = prefix | ((unsigned) b << shift);
                ps[1] = want - acc;
            }
        }
        __syncthreads();
        prefix = ps[0];
        want = ps[1];
        mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned thr = prefix;
    const int chunk = (nv + QSA_SEL_THREADS - 1) / QSA_SEL_THREADS, j0 = min(nv, tid * chunk), j1 = min(nv, j0 + chunk);
    int n_eq = 0;
    for (int j = j0; j < j1; ++j) n_eq += key[j] == thr;
    s_eq[tid] = n_eq;
    __syncthreads();
    for (int o = 1; o < QSA_SEL_THREADS; o <<= 1) {
        const int e = tid >= o ? s_eq[tid - o] : 0;
        __syncthreads();
        s_eq[tid] += e;
        __syncthreads();
    }
    const int eq_start = s_eq[tid] - n_eq;
    int own = 0;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            own += take && kv_card(j) == card;
        }
    }
    s_gt[tid] = own;
    __syncthreads();
    for (int o = 1; o < QSA_SEL_THREADS; o <<= 1) {
        const int a = tid >= o ? s_gt[tid - o] : 0;
        __syncthreads();
        s_gt[tid] += a;
        __syncthreads();
    }
    int out = s_gt[tid] - own;
    {
        int eq_before = eq_start;
        for (int j = j0; j < j1; ++j) {
            const unsigned k = key[j];
            bool take = k > thr;
            if (k == thr) take = eq_before++ < (int) want;
            if (take && kv_card(j) == card) {
#pragma unroll
                for (int m = 0; m < QSA_POOL; ++m) row[out * QSA_POOL + m] = kv_local_block(j) * QSA_POOL + m;
                ++out;
            }
        }
    }
    if (tid == QSA_SEL_THREADS - 1) {
        int n = out * QSA_POOL;
        const int jt = p / QSA_POOL;
        if (kv_card(jt) == card)
            for (int m = 0; m < n_tail; ++m) row[n++] = kv_local_block(jt) * QSA_POOL + (p - n_tail + 1 + m) % QSA_POOL;
        n_cells[t] = n;
    }
}

template <int ALL>
__global__ void __launch_bounds__(QSA_SEL_THREADS, 2) k_idx_select_c(const float* __restrict__ scores, int* __restrict__ cells,
                                                                     int* __restrict__ n_cells, int max_pools, const int* pos0p, int card) {
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int p = *pos0p + t, nv = (p + 1) / QSA_POOL, n_tail = (p + 1) % QSA_POOL;
    if (nv <= QSA_TOP) {
        if (tid == 0) n_cells[t] = -1;
        return;
    }
    const unsigned* key = reinterpret_cast<const unsigned*>(scores + (size_t) t * max_pools);
    int* row = cells + (size_t) t * QSA_MAXSEL;
    __shared__ unsigned ck[QSA_SELC_CAP];   // the candidates' keys and indices, in index order
    __shared__ int cj[QSA_SELC_CAP];
    __shared__ unsigned hist[256];
    __shared__ unsigned sw[32], ps[2];
    __shared__ int sc[32];
    const unsigned FULL = 0xffffffffu;
    // a warp's scores in passes 2 and 3: wc of them from warp * wc, 32 a step
    const int wc = (((nv + 31) >> 5) + 31) & ~31, w0 = min(nv, warp * wc), w1 = min(nv, w0 + wc);
    unsigned L = 0;
    int woff = w0, nc = nv;
    if (nv > ALL) {
        unsigned m = 0;   // 1: the largest of the thread's scores (tid, tid + 1024, ...); none: 0, which bounds nothing wrongly
        for (int j = tid; j < nv; j += QSA_SEL_THREADS) m = max(m, key[j]);
        int rank = 0;   // the lanes with a larger maximum (ties: the lower lane first)
#pragma unroll
        for (int r = 1; r < 32; ++r) {
            const int o = (lane + r) & 31;
            const unsigned v = __shfl_sync(FULL, m, o);
            rank += v > m || (v == m && o < lane);
        }
        const unsigned hit = __ballot_sync(FULL, rank == 15);
        const unsigned wm = __shfl_sync(FULL, m, __ffs(hit) - 1);   // the warp's 16th largest
        if (lane == 0) sw[warp] = wm;
        __syncthreads();
        L = sw[lane];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) L = min(L, __shfl_xor_sync(FULL, L, o));
        int cnt = 0;   // 2: the warp's candidates
        for (int j0 = w0; j0 < w1; j0 += 32) {
            const int j = j0 + lane;
            const unsigned k = j < w1 ? key[j] : 0u;
            cnt += __popc(__ballot_sync(FULL, j < w1 && k >= L));
        }
        if (lane == 0) sc[warp] = cnt;
        __syncthreads();
        const int w = sc[lane];
        int winc = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(FULL, winc, o);
            if (lane >= o) winc += u;
        }
        woff = __shfl_sync(FULL, winc - w, warp);
        nc = __shfl_sync(FULL, winc, 31);
        __syncthreads();   // sc is free again
        if (nc > QSA_SELC_CAP) {
            selc_all(key, row, n_cells, t, p, card, hist, ps, cj, cj + QSA_SEL_THREADS);
            return;
        }
    }
    for (int j0 = w0; j0 < w1; j0 += 32) {   // 3: the candidates in index order
        const int j = j0 + lane;
        const unsigned k = j < w1 ? key[j] : 0u;
        const bool c = j < w1 && k >= L;
        const unsigned mk = __ballot_sync(FULL, c);
        if (c) {
            const int i = woff + __popc(mk & ((1u << lane) - 1u));
            ck[i] = k;
            cj[i] = j;
        }
        woff += __popc(mk);
    }
    __syncthreads();
    // a thread's candidates: per of them from tid * per, in registers from here on
    const int per = (nc + QSA_SEL_THREADS - 1) / QSA_SEL_THREADS, i0 = min(nc, tid * per), n = min(nc, i0 + per) - i0;
    unsigned k[QSA_SELC_PER];
    int jj[QSA_SELC_PER];
#pragma unroll
    for (int s = 0; s < QSA_SELC_PER; ++s) {
        k[s] = s < n ? ck[i0 + s] : 0u;
        jj[s] = s < n ? cj[i0 + s] : 0;
    }
    unsigned prefix = 0, mask = 0, want = QSA_TOP;   // k_idx_select's radix steps: the want-th largest among the keys matching prefix on mask
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
#pragma unroll
        for (int s = 0; s < QSA_SELC_PER; ++s) {
            const bool in = s < n && (k[s] & mask) == prefix;
            const unsigned bin = (k[s] >> shift) & 255u;
            if (shift == 24) {   // the candidates share their high bits: a warp's keys of one bin counted with one atomic
                unsigned todo = __ballot_sync(FULL, in);
                while (todo) {
                    const int lead = __ffs(todo) - 1;
                    const unsigned b = __shfl_sync(FULL, bin, lead);
                    const unsigned same = __ballot_sync(FULL, in && bin == b);
                    if (lane == lead) atomicAdd(&hist[b], (unsigned) __popc(same));
                    todo &= ~same;
                }
            } else if (in) {
                atomicAdd(&hist[bin], 1u);
            }
        }
        __syncthreads();
        if (tid < 32) {   // the highest bin b > 0 where the count from 255 down reaches want, else 0; acc: the count above b
            const int hi = 255 - 8 * lane;   // lane l: bins hi .. hi - 7
            unsigned own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) own += hist[hi - i];
            unsigned before = own;   // inclusive scan over the lanes (higher bins first), then exclusive
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned v = __shfl_up_sync(FULL, before, o);
                if (lane >= o) before += v;
            }
            before -= own;
            const unsigned hit = __ballot_sync(FULL, before + own >= want);
            const int first = hit ? __ffs(hit) - 1 : 31;
            if (lane == first) {
                unsigned acc = before;
                int b = hi - 7;
                for (int i = 0; i < 8; ++i) {
                    const int bb = hi - i;
                    if (bb == 0 || acc + hist[bb] >= want) {
                        b = bb;
                        break;
                    }
                    acc += hist[bb];
                }
                ps[0] = prefix | ((unsigned) b << shift);
                ps[1] = want - acc;
            }
        }
        __syncthreads();
        prefix = ps[0];
        want = ps[1];
        mask |= 255u << shift;
    }
    // prefix: the threshold key; `want` ties at it are kept (the first by index)
    const unsigned thr = prefix;
    int n_eq = 0;
#pragma unroll
    for (int s = 0; s < QSA_SELC_PER; ++s) n_eq += s < n && k[s] == thr;
    int total;
    int eq_before = selc_scan(n_eq, sc, total);
    int own = 0;
    unsigned mine = 0;   // the thread's candidates taken and on this card
#pragma unroll
    for (int s = 0; s < QSA_SELC_PER; ++s)
        if (s < n) {
            bool take = k[s] > thr;
            if (k[s] == thr) take = eq_before++ < (int) want;
            if (take && kv_card(jj[s]) == card) {
                mine |= 1u << s;
                ++own;
            }
        }
    int out = selc_scan(own, sc, total);   // owned selected blocks before this thread's; total: the card's selected-block count
#pragma unroll
    for (int s = 0; s < QSA_SELC_PER; ++s)
        if ((mine >> s) & 1u) {
#pragma unroll
            for (int m = 0; m < QSA_POOL; ++m) row[out * QSA_POOL + m] = kv_local_block(jj[s]) * QSA_POOL + m;
            ++out;
        }
    if (tid == 0) {
        int nn = total * QSA_POOL;
        const int jt = p / QSA_POOL;   // the tail's block (incomplete)
        if (kv_card(jt) == card)
            for (int m = 0; m < n_tail; ++m) row[nn++] = kv_local_block(jt) * QSA_POOL + (p - n_tail + 1 + m) % QSA_POOL;
        n_cells[t] = nn;
    }
}

}  // namespace
