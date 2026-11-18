// Attention over a prompt chunk (included by kernels.cu after the decode attention): one block per (row, KV head)
// walks the row's attended positions on this card, APC at a time, for the KV head's ATT_QPK query heads (rows of a
// 16-row tile) on the tensor cores: S = q k^T and O += p v as f16 x f16 with f32 accumulation, q and p rounded to f16
// as llama.cpp's CUDA attention takes them. Online softmax per head.
// The partial (acc, max, sum) of every head goes to part[owner card][row][head of ATT_OWN][HD + 2]; the merge across cards
// (k_pf_attn_merge, after the exchange) applies the gate.
namespace {

constexpr int APC = 32;                  // positions per tile
constexpr int AP_NPT = APC / 16;         // their 16-position tiles
constexpr int AP_THREADS = 256;
constexpr int AP_LD = HD + 8;            // halves per Q / K / V row in shared memory (a multiple of 8, for wmma)
constexpr int AP_LDF = HD + 4;           // floats per row of the f32 staging tile
constexpr int AP_PLD = APC + 8;          // halves per P row
constexpr int AP_W = HD + 2;             // floats per partial: acc, max, sum
constexpr size_t AP_SMEM = sizeof(__half) * (16 * AP_LD + APC * AP_LD + 16 * AP_PLD) + sizeof(float) * (16 * AP_LDF + 3 * 16);
static_assert(ATT_QPK <= 16, "a KV head's query heads in one 16-row tile");
static_assert(AP_THREADS / 32 * 2 * 16 == HD, "eight warps of two 16-column O tiles");
static_assert(AP_THREADS / 32 * APC <= AP_LDF && APC % 16 == 0, "the S partials of a tile side by side in st");

#if FNX_ARCH >= 70
__global__ void __launch_bounds__(AP_THREADS) k_pf_attn(const float* __restrict__ qg, const __half* __restrict__ kc,
                                                        const __half* __restrict__ vc, KvPages g, const int* __restrict__ pos0p,
                                                        const int* __restrict__ cells, const int* __restrict__ n_cells, int card,
                                                        int C, int r0, float* __restrict__ part) {
    using namespace nvcuda;
    extern __shared__ __align__(16) unsigned char ap_smem[];
    __half* qh = reinterpret_cast<__half*>(ap_smem);           // [16][AP_LD]
    __half* kv = qh + 16 * AP_LD;                               // [APC][AP_LD]: K, then V
    __half* ph = kv + APC * AP_LD;                              // [16][AP_PLD]
    float* st = reinterpret_cast<float*>(ph + 16 * AP_PLD);     // [16][AP_LDF]: S partials, then the P V tile
    float* sm = st + 16 * AP_LDF;                               // [16]: running max
    float* sz = sm + 16;                                        // [16]: running sum
    float* ssc = sz + 16;                                       // [16]: this tile's rescale
    const int t = r0 + blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x, warp = tid >> 5;
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(*pos0p + t, card);
    // q rows (heads kvh * QPK + j) as f16; rows past QPK are zero
    for (int e = tid; e < 16 * HD; e += AP_THREADS) {
        const int r = e / HD, d = e % HD;
        const float x = r < ATT_QPK ? qg[((size_t) t * NQH + kvh * ATT_QPK + r) * 2 * HD + d] : 0.f;
        qh[r * AP_LD + d] = __float2half_rn(x);
    }
    if (tid < 16) {
        sm[tid] = -INFINITY;
        sz[tid] = 0.f;
    }
    // this thread's O elements: row orow, columns ocol + 16 j
    const int orow = tid / 16, ocol = tid % 16;
    float o[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = 0.f;
    const float scale = rsqrtf((float) HD);
    // a tile's K or V rows, 8 halves a load, gathered into registers ahead of their use: V during S and the softmax,
    // the next tile's K during P V (each a load of the position's index, then of the row)
    constexpr int AP_GL = APC * HD / 8 / AP_THREADS;
    static_assert(APC * HD / 8 % AP_THREADS == 0, "a tile's loads spread evenly");
    auto gather = [&](const __half* base, int c, int nr, uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            r[i] = make_uint4(0, 0, 0, 0);
            if (pp < nr) {
                const int pos = list ? list[c + pp] : c + pp;
                if (g.q8) {   // 8 int8 values and their group's scale, as 8 halves
                    const size_t at = kvq_row(g, pos) + (size_t) kvh * HD + d8 * 8;
                    const uint2 q = *reinterpret_cast<const uint2*>(reinterpret_cast<const int8_t*>(base) + at);
                    const float d = __half2float(base[kvq_scale(g, pos) + ((size_t) kvh * HD + d8 * 8) / KV8_GROUP]);
                    const int8_t* qb = reinterpret_cast<const int8_t*>(&q);
                    __align__(16) __half h[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) h[k] = __float2half_rn(qb[k] * d);
                    r[i] = *reinterpret_cast<const uint4*>(h);
                } else {
                    r[i] = *reinterpret_cast<const uint4*>(base + kvp_row(g, pos) + (size_t) kvh * HD + d8 * 8);
                }
            }
        }
    };
    auto place = [&](const uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            *reinterpret_cast<uint4*>(kv + pp * AP_LD + d8 * 8) = r[i];
        }
    };
    uint4 kr[AP_GL], vr[AP_GL];
    if (ctx > 0) gather(kc, 0, min(APC, ctx), kr);
    for (int c0 = 0; c0 < ctx; c0 += APC) {
        const int np = min(APC, ctx - c0);
        __syncthreads();   // the last tile's V and P reads are done
        place(kr);
        gather(vc, c0, np, vr);
        __syncthreads();
        // S partials: warp w takes dimensions [32 w, 32 w + 32) (two k-steps) for each 16-position tile
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fs[AP_NPT];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) wmma::load_matrix_sync(fa[ks], qh + warp * 32 + ks * 16, AP_LD);
#pragma unroll
            for (int pt = 0; pt < AP_NPT; ++pt) {
                wmma::fill_fragment(fs[pt], 0.f);
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    wmma::load_matrix_sync(fb, kv + pt * 16 * AP_LD + warp * 32 + ks * 16, AP_LD);
                    wmma::mma_sync(fs[pt], fa[ks], fb, fs[pt]);
                }
                // the eight warps' partials side by side in st: warp w at columns [APC w, APC w + APC)
                wmma::store_matrix_sync(st + warp * APC + pt * 16, fs[pt], AP_LDF, wmma::mem_row_major);
            }
        }
        __syncthreads();
        // the scores: thread (row r, positions c + 16 u) sums the eight partials; the row's online-softmax step (16
        // lanes a row)
        {
            const int r = tid / 16, c = tid % 16;
            float sv[AP_NPT], mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                float sacc = 0.f;
#pragma unroll
                for (int w = 0; w < AP_THREADS / 32; ++w) sacc += st[r * AP_LDF + w * APC + u * 16 + c];
                sv[u] = c + 16 * u < np ? sacc * scale : -INFINITY;
                mx = fmaxf(mx, sv[u]);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off, 16));
            const float mo = sm[r], mn = fmaxf(mo, mx);
            float ps = 0.f;
#pragma unroll
            for (int u = 0; u < AP_NPT; ++u) {
                const float p = sv[u] == -INFINITY ? 0.f : __expf(sv[u] - mn);
                ps += p;
                ph[r * AP_PLD + u * 16 + c] = __float2half_rn(p);
            }
#pragma unroll
            for (int off = 8; off; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off, 16);
            __syncwarp();
            if (c == 0) {
                const float sc = mo == -INFINITY ? 0.f : __expf(mo - mn);
                ssc[r] = sc;
                sz[r] = sz[r] * sc + ps;
                sm[r] = mn;
            }
        }
        // V rows of the tile into the K buffer (its S reads are done: the barrier orders them)
        __syncthreads();
        place(vr);
        __syncthreads();
        if (c0 + APC < ctx) gather(kc, c0 + APC, min(APC, ctx - c0 - APC), kr);
        // P V: warp w takes O columns [32 w, 32 w + 32) (two 16-column tiles), the tile's positions in k-steps of 16
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fp[AP_NPT];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fv;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fo;
#pragma unroll
            for (int ks = 0; ks < AP_NPT; ++ks) wmma::load_matrix_sync(fp[ks], ph + ks * 16, AP_PLD);
#pragma unroll
            for (int ct = 0; ct < 2; ++ct) {
                const int d0 = warp * 32 + ct * 16;
                wmma::fill_fragment(fo, 0.f);
#pragma unroll
                for (int ks = 0; ks < AP_NPT; ++ks) {
                    wmma::load_matrix_sync(fv, kv + ks * 16 * AP_LD + d0, AP_LD);
                    wmma::mma_sync(fo, fp[ks], fv, fo);
                }
                wmma::store_matrix_sync(st + d0, fo, AP_LDF, wmma::mem_row_major);
            }
        }
        __syncthreads();
        const float sc = ssc[orow];
#pragma unroll
        for (int j = 0; j < 16; ++j) o[j] = fmaf(o[j], sc, st[orow * AP_LDF + ocol + 16 * j]);
    }
    __syncthreads();
    if (orow < ATT_QPK) {
        const int h = kvh * ATT_QPK + orow, k = h / ATT_OWN, hl = h % ATT_OWN;
        float* dst = part + (((size_t) k * C + t) * ATT_OWN + hl) * AP_W;
#pragma unroll
        for (int j = 0; j < 16; ++j) dst[ocol + 16 * j] = o[j];
        if (ocol == 0) {
            dst[HD] = sm[orow];
            dst[HD + 1] = sz[orow];
        }
    }
}

#endif  // FNX_ARCH >= 70

#if FNX_ARCH < 70
// GP100 has no tensor cores: the same tile, every sum in registers. A warp holds two query rows, a lane one row's 8 positions and 64 of the 256
// dimensions, so a tile's K and V stay in shared memory and nothing else does: an SM holds two blocks, not one. Lane l of a warp: row bit l >> 4,
// position group (l >> 2) & 3, dimension part l & 3. The parts' sums meet by a shuffle, then the groups' softmax by another.
constexpr size_t AP_SMEM60 = sizeof(__half) * (16 * AP_LD + APC * AP_LD);
__global__ void __launch_bounds__(AP_THREADS) k_pf_attn(const float* __restrict__ qg, const __half* __restrict__ kc,
                                                        const __half* __restrict__ vc, KvPages g, const int* __restrict__ pos0p,
                                                        const int* __restrict__ cells, const int* __restrict__ n_cells, int card,
                                                        int C, int r0, float* __restrict__ part) {
    extern __shared__ __align__(16) unsigned char ap_smem[];
    __half* qh = reinterpret_cast<__half*>(ap_smem);           // [16][AP_LD]
    __half* kv = qh + 16 * AP_LD;                               // [APC][AP_LD]: K, then V
    const int t = r0 + blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int* list = cells && n_cells[t] >= 0 ? cells + (size_t) t * QSA_MAXSEL : nullptr;
    const int ctx = list ? n_cells[t] : kv_owned(*pos0p + t, card);
    for (int e = tid; e < 16 * HD; e += AP_THREADS) {
        const int r = e / HD, d = e % HD;
        const float x = r < ATT_QPK ? qg[((size_t) t * NQH + kvh * ATT_QPK + r) * 2 * HD + d] : 0.f;
        qh[r * AP_LD + d] = __float2half_rn(x);
    }
    constexpr int AP_GL = APC * HD / 8 / AP_THREADS;
    static_assert(APC * HD / 8 % AP_THREADS == 0, "a tile's loads spread evenly");
    auto gather = [&](const __half* base, int c, int nr, uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            r[i] = make_uint4(0, 0, 0, 0);
            if (pp < nr) {
                const int pos = list ? list[c + pp] : c + pp;
                if (g.q8) {   // 8 int8 values and their group's scale, as 8 halves
                    const size_t at = kvq_row(g, pos) + (size_t) kvh * HD + d8 * 8;
                    const uint2 q = *reinterpret_cast<const uint2*>(reinterpret_cast<const int8_t*>(base) + at);
                    const float d = __half2float(base[kvq_scale(g, pos) + ((size_t) kvh * HD + d8 * 8) / KV8_GROUP]);
                    const int8_t* qb = reinterpret_cast<const int8_t*>(&q);
                    __align__(16) __half h[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) h[k] = __float2half_rn(qb[k] * d);
                    r[i] = *reinterpret_cast<const uint4*>(h);
                } else {
                    r[i] = *reinterpret_cast<const uint4*>(base + kvp_row(g, pos) + (size_t) kvh * HD + d8 * 8);
                }
            }
        }
    };
    auto place = [&](const uint4 (&r)[AP_GL]) {
#pragma unroll
        for (int i = 0; i < AP_GL; ++i) {
            const int e = tid + i * AP_THREADS, pp = e / (HD / 8), d8 = e % (HD / 8);
            *reinterpret_cast<uint4*>(kv + pp * AP_LD + d8 * 8) = r[i];
        }
    };
    __syncthreads();
    const int row = warp * 2 + ((lane >> 4) & 1), grp = (lane >> 2) & 3, dp = lane & 3;
    const bool live = warp * 2 < ATT_QPK;   // a warp's two rows: past the KV head's query heads nothing is computed
    uint4 qu[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) qu[u] = *reinterpret_cast<const uint4*>(qh + row * AP_LD + dp * 64 + u * 8);
    float o[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) o[i] = 0.f;
    float mo = -INFINITY, zv = 0.f;
    const float scale = rsqrtf((float) HD);
    uint4 kr[AP_GL], vr[AP_GL];
    if (ctx > 0) gather(kc, 0, min(APC, ctx), kr);
    for (int c0 = 0; c0 < ctx; c0 += APC) {
        const int np = min(APC, ctx - c0);
        __syncthreads();   // the last tile's V reads are done
        place(kr);
        gather(vc, c0, np, vr);
        __syncthreads();
        float pr[8];
        float sc = 0.f;
        if (live) {
            float s[8];
            half2 h[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[i] = 0.f;
                h[i] = __halves2half2(__float2half(0.f), __float2half(0.f));
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const uint4 q = qu[u];
                const half2 qx = *reinterpret_cast<const half2*>(&q.x), qy = *reinterpret_cast<const half2*>(&q.y);
                const half2 qz = *reinterpret_cast<const half2*>(&q.z), qw = *reinterpret_cast<const half2*>(&q.w);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint4 k = *reinterpret_cast<const uint4*>(kv + (grp * 8 + i) * AP_LD + dp * 64 + u * 8);
                    h[i] = __hfma2(qx, *reinterpret_cast<const half2*>(&k.x), h[i]);
                    h[i] = __hfma2(qy, *reinterpret_cast<const half2*>(&k.y), h[i]);
                    h[i] = __hfma2(qz, *reinterpret_cast<const half2*>(&k.z), h[i]);
                    h[i] = __hfma2(qw, *reinterpret_cast<const half2*>(&k.w), h[i]);
                }
                if ((u & 3) == 3) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float2 f = __half22float2(h[i]);
                        s[i] += f.x + f.y;
                        h[i] = __halves2half2(__float2half(0.f), __float2half(0.f));
                    }
                }
            }
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[i] += __shfl_xor_sync(0xffffffffu, s[i], 1);
                s[i] += __shfl_xor_sync(0xffffffffu, s[i], 2);
                s[i] = grp * 8 + i < np ? s[i] * scale : -INFINITY;
                mx = fmaxf(mx, s[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
            const float mn = fmaxf(mo, mx);
            float ps = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                pr[i] = s[i] == -INFINITY ? 0.f : __expf(s[i] - mn);
                ps += pr[i];
            }
            ps += __shfl_xor_sync(0xffffffffu, ps, 4);
            ps += __shfl_xor_sync(0xffffffffu, ps, 8);
            sc = mo == -INFINITY ? 0.f : __expf(mo - mn);
            zv = zv * sc + ps;
            mo = mn;
        }
        __syncthreads();   // the tile's S reads are done: its V rows into the K buffer
        place(vr);
        __syncthreads();
        if (c0 + APC < ctx) gather(kc, c0 + APC, min(APC, ctx - c0 - APC), kr);
        if (live) {
            half2 pd[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const __half ph = __float2half_rn(pr[i]);
                pd[i] = __halves2half2(ph, ph);
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const __half zero = __float2half(0.f);
                half2 ax = __halves2half2(zero, zero), ay = ax, az = ax, aw = ax;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint4 v = *reinterpret_cast<const uint4*>(kv + (grp * 8 + i) * AP_LD + dp * 64 + u * 8);
                    ax = __hfma2(pd[i], *reinterpret_cast<const half2*>(&v.x), ax);
                    ay = __hfma2(pd[i], *reinterpret_cast<const half2*>(&v.y), ay);
                    az = __hfma2(pd[i], *reinterpret_cast<const half2*>(&v.z), az);
                    aw = __hfma2(pd[i], *reinterpret_cast<const half2*>(&v.w), aw);
                }
                const float2 fx = __half22float2(ax), fy = __half22float2(ay), fz = __half22float2(az), fw = __half22float2(aw);
                o[u * 8] = fmaf(o[u * 8], sc, fx.x);
                o[u * 8 + 1] = fmaf(o[u * 8 + 1], sc, fx.y);
                o[u * 8 + 2] = fmaf(o[u * 8 + 2], sc, fy.x);
                o[u * 8 + 3] = fmaf(o[u * 8 + 3], sc, fy.y);
                o[u * 8 + 4] = fmaf(o[u * 8 + 4], sc, fz.x);
                o[u * 8 + 5] = fmaf(o[u * 8 + 5], sc, fz.y);
                o[u * 8 + 6] = fmaf(o[u * 8 + 6], sc, fw.x);
                o[u * 8 + 7] = fmaf(o[u * 8 + 7], sc, fw.y);
            }
        }
    }
    __syncthreads();
    if (live) {
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            o[i] += __shfl_xor_sync(0xffffffffu, o[i], 4);
            o[i] += __shfl_xor_sync(0xffffffffu, o[i], 8);
        }
        if (grp == 0 && row < ATT_QPK) {
            const int h = kvh * ATT_QPK + row, k = h / ATT_OWN, hl = h % ATT_OWN;
            float* dst = part + (((size_t) k * C + t) * ATT_OWN + hl) * AP_W;
#pragma unroll
            for (int i = 0; i < 64; ++i) dst[dp * 64 + i] = o[i];
            if (dp == 0) {
                dst[HD] = mo;
                dst[HD + 1] = zv;
            }
        }
    }
}
#endif  // FNX_ARCH < 70

// this card's heads: its own partial and the others' (from card g + 1 in slot 0, g + 2 in slot 1, ...), merged; the gate
__global__ void __launch_bounds__(HD) k_pf_attn_merge(const float* __restrict__ own, const float* __restrict__ recv,
                                                      const float* __restrict__ qg, float* __restrict__ out, int C, int card) {
    const int t = blockIdx.x, hl = blockIdx.y, i = threadIdx.x;
    const size_t at = ((size_t) t * ATT_OWN + hl) * AP_W;
    float m = own[at + HD], z = own[at + HD + 1], a = own[at + i];
#pragma unroll
    for (int k = 0; k < NCARD - 1; ++k) {
        const float* src = recv + (size_t) k * C * ATT_OWN * AP_W + at;
        att_merge(m, z, a, src[HD], src[HD + 1], src[i]);
    }
    const int hg = card * ATT_OWN + hl;
    out[((size_t) t * ATT_OWN + hl) * HD + i] = a / z * sigm(qg[((size_t) t * NQH + hg) * 2 * HD + HD + i]);
}

}  // namespace
