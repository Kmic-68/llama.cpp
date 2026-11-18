// p100_attn.cuh: a prompt chunk's attention on GP100, int8 K/V: Strata's score and value kernels for the P100 (src/kernels/cuda/qsa_pattn.cu, PR #3943) over the split
// engine's paged K/V and partials. A lane keeps 8 dimensions of its 12 query heads in registers, so a cell costs it one read of shared memory for 96 products; the
// tile kernel in attn_prompt.cuh read shared memory every 8. Score: block (chunk of 64 cells, KV head, row). Value: block (KV head, row), merging the chunks as it folds.
// The partial (acc, max, sum) of a head is the tile kernel's, so k_pf_attn_merge and the exchange stay as they are.
#pragma once
#include <cfloat>

namespace fnx {
namespace {

constexpr int PA_G = NQH / NKVH, PA_OWN = NQH / NCARD, PA_W = HD + 2;
constexpr int PA_CHUNK = 64, PA_THREADS = 256, PA_WARPS = PA_THREADS / 32, PA_NSC = HD / KV8_GROUP;
constexpr int PA_NCM = 33;   // chunks the value kernel keeps (the selection's 2051 cells)
static_assert(PA_G == 12 && HD == 256 && KV8_GROUP == 32, "laid out for 12 query heads a KV head, 256 dimensions, a scale per 32");
// floats per (row, KV head, chunk): e [G][CHUNK], the cells' K/V row and scale offsets (int64 each), m [G], l [G]
constexpr int PA_BLK = PA_G * PA_CHUNK + 4 * PA_CHUNK + 2 * PA_G;
constexpr int PA_BATCH = 128;   // rows a launch pair; the scratch holds their blocks

__device__ __forceinline__ float pa_warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 16); v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4); v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    return v;
}
__device__ __forceinline__ float pa_warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16)); v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4)); v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    return v;
}
__device__ __forceinline__ int pa_owned(int p, int card) {   // kv_owned (kernels.cu)
    const int b = p / 4;
    const int blocks = (b + NCARD - 1 - card) / NCARD;
    return blocks * 4 + (kv_card(b) == card ? p % 4 + 1 : 0);
}

// slot i of lane L holds head 6 ((i / 6) ^ b3) + ((i % 6 + 3 b2) % 6) (b_k: bit k of L): every lane keeps and sends the same slots, the reduce needs no selects
__device__ __forceinline__ int pa_head(int i, int lane) {
    const int b3 = (lane >> 3) & 1, b2 = (lane >> 2) & 1;
    return 6 * ((i / 6) ^ b3) + ((i % 6 + 3 * b2) % 6);
}
__device__ __forceinline__ void pa_dequant8(const int8_t* krow, float sc, int lane, float (&k8)[8]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(krow + lane * 8);
    const int8_t* cb = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; ++e) k8[e] = (float) cb[e] * sc;
}
__device__ __forceinline__ float pa_dot8(const float (&k8)[8], const float4& qa, const float4& qb) {
    return k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w + k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
}

__global__ void __launch_bounds__(PA_THREADS, 2) k_pa_score(const float* __restrict__ qg, const int8_t* __restrict__ kq, const uint16_t* __restrict__ ks, KvPages g,
                                                            const int* __restrict__ pos0p, const int* __restrict__ cells, const int* __restrict__ n_cells, int card,
                                                            int r0, int maxsel, float scale, float* __restrict__ blocks, int n_chunks) {
    const int z = blockIdx.z, row_t = r0 + z, chunk = blockIdx.x, kvh = blockIdx.y;
    float* const blk = blocks + (((size_t) z * NKVH + kvh) * n_chunks + chunk) * PA_BLK;
    long long* const blk_row = reinterpret_cast<long long*>(blk + PA_G * PA_CHUNK);
    long long* const blk_sc = blk_row + PA_CHUNK;
    float* const blk_m = blk + PA_G * PA_CHUNK + 4 * PA_CHUNK;
    float* const blk_l = blk_m + PA_G;
    __shared__ __align__(16) int8_t sk[PA_CHUNK][HD];
    __shared__ __align__(16) float sks[PA_CHUNK][PA_NSC];
    __shared__ float sp[PA_G][PA_CHUNK];
    __shared__ long long srow[PA_CHUNK];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nc = cells ? __ldg(n_cells + row_t) : -1;
    const int* const list = nc >= 0 ? cells + (size_t) row_t * maxsel : nullptr;
    const int n_ids = nc >= 0 ? nc : pa_owned(*pos0p + row_t, card);
    const int c0 = chunk * PA_CHUNK, n_here = min(PA_CHUNK, n_ids - c0);
    if (n_here <= 0) {
        if (t < PA_G) {
            blk_m[t] = -FLT_MAX;
            blk_l[t] = 0.0f;
        }
        return;
    }
    __shared__ long long sscale[PA_CHUNK];
    if (t < PA_CHUNK) {
        long long r = -1, s = 0;
        if (t < n_here) {
            const int pos = list ? __ldg(list + c0 + t) : c0 + t;
            r = (long long) kvq_row(g, pos) + (long long) kvh * HD;
            s = (long long) kvq_scale(g, pos) + (long long) kvh * HD / KV8_GROUP;
        }
        srow[t] = r;
        sscale[t] = s;
        blk_row[t] = r;
        blk_sc[t] = s;
    }
    // the KV head's query heads, in the permuted slots
    float4 qa[PA_G], qb[PA_G];
    const float* const qh = qg + ((size_t) row_t * NQH + kvh * PA_G) * 2 * HD;
#pragma unroll
    for (int i = 0; i < PA_G; ++i) {
        const float4* p4 = reinterpret_cast<const float4*>(qh + (size_t) pa_head(i, lane) * 2 * HD + lane * 8);
        qa[i] = __ldg(p4);
        qb[i] = __ldg(p4 + 1);
    }
    __syncthreads();
    {   // the chunk's K codes (16-byte loads, every one before any store) and scales; an absent cell reads the start of the pool
        uint4 kv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = t + PA_THREADS * j, c = idx >> 4, part = idx & 15;
            const long long row = srow[c];
            const long long rr = (c < n_here && row >= 0) ? row : 0;
            kv[j] = __ldg(reinterpret_cast<const uint4*>(kq + rr) + part);
        }
        float sc[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = t + PA_THREADS * j, c = idx >> 3, gi = idx & 7;
            const long long at = c < n_here ? sscale[c] : 0;
            sc[j] = __half2float(__ushort_as_half(__ldg(ks + at + gi)));
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) reinterpret_cast<uint4*>(&sk[0][0])[t + PA_THREADS * j] = kv[j];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = t + PA_THREADS * j;
            sks[idx >> 3][idx & 7] = sc[j];
        }
    }
    __syncthreads();
    {   // two cells at a time, the reduce over the lanes as a scatter of the heads
        const bool h16 = (lane & 16) != 0, h2 = (lane & 2) != 0, even = (lane & 1) == 0;
        const int hb = 6 * ((lane >> 3) & 1) + 3 * ((lane >> 2) & 1);
        const int h0 = hb + (h2 ? 2 : 0), h1 = hb + 1, gl = lane >> 2;
        for (int i = 0; i < PA_CHUNK / PA_WARPS; i += 2) {
            const int ca = warp + PA_WARPS * i, cb = warp + PA_WARPS * (i + 1);
            const bool va = ca < n_here, vb = cb < n_here;
            if (!va && !vb) {
                if (lane < PA_G) {
                    sp[lane][ca] = -FLT_MAX;
                    sp[lane][cb] = -FLT_MAX;
                }
                continue;
            }
            const int c1 = h16 ? cb : ca, c2 = h16 ? ca : cb;
            float r[PA_G];
            {
                float k8[8];
                pa_dequant8(sk[c2], sks[c2][gl], lane, k8);
                float p2[PA_G];
#pragma unroll
                for (int s = 0; s < PA_G; ++s) p2[s] = pa_dot8(k8, qa[s], qb[s]);
                pa_dequant8(sk[c1], sks[c1][gl], lane, k8);
#pragma unroll
                for (int s = 0; s < PA_G; ++s) r[s] = pa_dot8(k8, qa[s], qb[s]) + __shfl_xor_sync(0xffffffffu, p2[s], 16);
            }
            float r6[6];
#pragma unroll
            for (int s = 0; s < 6; ++s) r6[s] = r[s] + __shfl_xor_sync(0xffffffffu, r[s + 6], 8);
            float r3[3];
#pragma unroll
            for (int s = 0; s < 3; ++s) r3[s] = r6[s] + __shfl_xor_sync(0xffffffffu, r6[s + 3], 4);
            const float x0 = (h2 ? r3[2] : r3[0]) + __shfl_xor_sync(0xffffffffu, h2 ? r3[0] : r3[2], 2);
            const float x1 = r3[1] + __shfl_xor_sync(0xffffffffu, r3[1], 2);
            const float o0 = x0 + __shfl_xor_sync(0xffffffffu, x0, 1);
            const float o1 = x1 + __shfl_xor_sync(0xffffffffu, x1, 1);
            const bool vmine = h16 ? vb : va;
            if (even && vmine) {
                sp[h0][c1] = o0 * scale;
                if (!h2) sp[h1][c1] = o1 * scale;
            }
            if (lane < PA_G) {
                if (!va) sp[lane][ca] = -FLT_MAX;
                if (!vb) sp[lane][cb] = -FLT_MAX;
            }
        }
    }
    __syncthreads();
    // the chunk's softmax, a warp per head
    for (int h = warp; h < PA_G; h += PA_WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = pa_warp_max(fmaxf(a, b));
        const float ea = lane < n_here ? __expf(a - m) : 0.0f;
        const float eb = lane + 32 < n_here ? __expf(b - m) : 0.0f;
        blk[h * PA_CHUNK + lane] = ea;
        blk[h * PA_CHUNK + lane + 32] = eb;
        const float l = pa_warp_sum(ea + eb);
        if (lane == 0) {
            blk_m[h] = m;
            blk_l[h] = l;
        }
    }
}

// four dimensions a thread: a cell's four codes are one 32-bit load, its 12 weights three float4 reads for 48 fmaf; V loads run four cells ahead. The chunks merge as it folds
__global__ void __launch_bounds__(64) k_pa_value(const int8_t* __restrict__ vq, const uint16_t* __restrict__ vs, const int* __restrict__ pos0p, const int* __restrict__ cells,
                                                 const int* __restrict__ n_cells, int card, int r0, const float* __restrict__ blocks, int n_chunks, int C, float* __restrict__ part) {
    constexpr int NT = 64;
    const int kvh = blockIdx.x, z = blockIdx.y, row_t = r0 + z, t = threadIdx.x;
    const float* const blk0 = blocks + ((size_t) z * NKVH + kvh) * n_chunks * PA_BLK;
    __shared__ float sm_m[PA_NCM][PA_G], sm_w[PA_NCM][PA_G], sm_l[PA_NCM][PA_G];
    __shared__ float sL[PA_G], sM[PA_G];
    __shared__ __align__(16) float spt[PA_CHUNK][PA_G];
    __shared__ long long srow[PA_CHUNK], sscale[PA_CHUNK];
    const int nc = cells ? __ldg(n_cells + row_t) : -1;
    const int width = nc >= 0 ? nc : pa_owned(*pos0p + row_t, card);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / PA_CHUNK + 1) : 0;
    for (int i = t; i < n_act * PA_G; i += NT) {
        const int c = i / PA_G, h = i % PA_G;
        const float* const mp = blk0 + (size_t) c * PA_BLK + PA_G * PA_CHUNK + 4 * PA_CHUNK;
        sm_m[c][h] = __ldg(mp + h);
        sm_l[c][h] = __ldg(mp + PA_G + h);
    }
    __syncthreads();
    if (t < PA_G) {
        float M = -FLT_MAX;
        for (int c = 0; c < n_act; ++c) M = fmaxf(M, sm_m[c][t]);
        float L = 0.0f;
        for (int c = 0; c < n_act; ++c) {
            const float m = sm_m[c][t];
            const float w = __expf(m - M);
            sm_w[c][t] = w;
            const float L_next = fmaf(sm_l[c][t], w, L);
            L = m == -FLT_MAX ? L : L_next;
        }
        sL[t] = L;
        sM[t] = M;
    }
    const int d0 = 4 * t, gi = d0 / KV8_GROUP;
    float acc[4][PA_G];
#pragma unroll
    for (int h = 0; h < PA_G; ++h)
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d][h] = 0.0f;
    for (int c = 0; c < n_act; ++c) {
        const float* const blk = blk0 + (size_t) c * PA_BLK;
        const long long* const blk_row = reinterpret_cast<const long long*>(blk + PA_G * PA_CHUNK);
        const int n_here = min(PA_CHUNK, width - c * PA_CHUNK);
        __syncthreads();
        srow[t] = __ldg(blk_row + t);
        sscale[t] = __ldg(blk_row + PA_CHUNK + t);
#pragma unroll
        for (int j = 0; j < PA_G * PA_CHUNK / NT; ++j) {
            const int idx = t + NT * j, h = idx / PA_CHUNK, cell = idx % PA_CHUNK;
            spt[cell][h] = __ldg(blk + idx);
        }
        __syncthreads();
        float a[4][PA_G];
#pragma unroll
        for (int h = 0; h < PA_G; ++h)
#pragma unroll
            for (int d = 0; d < 4; ++d) a[d][h] = 0.0f;
        const auto fold4 = [&](int cl, uint32_t codes, float sc) {
            const float v0 = (float) (int8_t) (codes & 0xFF) * sc, v1 = (float) (int8_t) ((codes >> 8) & 0xFF) * sc;
            const float v2 = (float) (int8_t) ((codes >> 16) & 0xFF) * sc, v3 = (float) (int8_t) (codes >> 24) * sc;
            const float4* w4 = reinterpret_cast<const float4*>(spt[cl]);
            const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
            const float ww[PA_G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
            for (int h = 0; h < PA_G; ++h) {
                a[0][h] = fmaf(ww[h], v0, a[0][h]);
                a[1][h] = fmaf(ww[h], v1, a[1][h]);
                a[2][h] = fmaf(ww[h], v2, a[2][h]);
                a[3][h] = fmaf(ww[h], v3, a[3][h]);
            }
        };
        const auto vld = [&](int cl, uint32_t& codes, float& sc) {
            codes = __ldg(reinterpret_cast<const uint32_t*>(vq + srow[cl] + d0));
            sc = __half2float(__ushort_as_half(__ldg(vs + sscale[cl] + gi)));
        };
        int cl = 0;
        for (; cl + 4 <= n_here; cl += 4) {
            uint32_t k0, k1, k2, k3;
            float s0, s1, s2, s3;
            vld(cl, k0, s0); vld(cl + 1, k1, s1); vld(cl + 2, k2, s2); vld(cl + 3, k3, s3);
            fold4(cl, k0, s0);
            fold4(cl + 1, k1, s1);
            fold4(cl + 2, k2, s2);
            fold4(cl + 3, k3, s3);
        }
        for (; cl < n_here; ++cl) {
            uint32_t k0;
            float s0;
            vld(cl, k0, s0);
            fold4(cl, k0, s0);
        }
#pragma unroll
        for (int h = 0; h < PA_G; ++h) {
            const bool live = !(sm_m[c][h] == -FLT_MAX);
            const float w = sm_w[c][h];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const float acc_next = fmaf(a[d][h], w, acc[d][h]);
                acc[d][h] = live ? acc_next : acc[d][h];
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < PA_G; ++h) {
        const int hg = kvh * PA_G + h, k = hg / PA_OWN, hl = hg % PA_OWN;
        float* const dst = part + (((size_t) k * C + row_t) * PA_OWN + hl) * PA_W;
#pragma unroll
        for (int d = 0; d < 4; ++d) dst[d0 + d] = acc[d][h];
        if (t == 0) {
            dst[HD] = sM[h] == -FLT_MAX ? -INFINITY : sM[h];
            dst[HD + 1] = sL[h];
        }
    }
}

// a window's rows: block (chunk, KV head, row) folds its chunk's cells into unnormalized sums for the KV head's 12 query heads (64 threads, 4 dimensions each)
__global__ void __launch_bounds__(64) k_pa_value_chunk(const int8_t* __restrict__ vq, const uint16_t* __restrict__ vs, const int* __restrict__ pos0p, const int* __restrict__ cells,
                                                       const int* __restrict__ n_cells, int card, const float* __restrict__ blocks, int n_chunks, float* __restrict__ acc_out) {
    constexpr int NT = 64;
    static_assert(NT == PA_CHUNK, "a thread per cell when the chunk's rows are staged");
    const int c = blockIdx.x, kvh = blockIdx.y, row_t = blockIdx.z, t = threadIdx.x;
    const int nc = cells ? __ldg(n_cells + row_t) : -1;
    const int width = nc >= 0 ? nc : pa_owned(*pos0p + row_t, card);
    const int n_here = min(PA_CHUNK, width - c * PA_CHUNK);
    if (n_here <= 0) return;   // the merge stops before this chunk
    const float* const blk = blocks + (((size_t) row_t * NKVH + kvh) * n_chunks + c) * PA_BLK;
    const long long* const blk_row = reinterpret_cast<const long long*>(blk + PA_G * PA_CHUNK);
    __shared__ __align__(16) float spt[PA_CHUNK][PA_G];
    __shared__ long long srow[PA_CHUNK], sscale[PA_CHUNK];
    srow[t] = __ldg(blk_row + t);
    sscale[t] = __ldg(blk_row + PA_CHUNK + t);
#pragma unroll
    for (int j = 0; j < PA_G * PA_CHUNK / NT; ++j) {
        const int idx = t + NT * j, h = idx / PA_CHUNK, cell = idx % PA_CHUNK;
        spt[cell][h] = __ldg(blk + idx);
    }
    __syncthreads();
    const int d0 = 4 * t, gi = d0 / KV8_GROUP;
    float a[4][PA_G];
#pragma unroll
    for (int h = 0; h < PA_G; ++h)
#pragma unroll
        for (int dd = 0; dd < 4; ++dd) a[dd][h] = 0.0f;
    const auto fold4 = [&](int cl, uint32_t codes, float sc) {
        const float v0 = (float) (int8_t) (codes & 0xFF) * sc, v1 = (float) (int8_t) ((codes >> 8) & 0xFF) * sc;
        const float v2 = (float) (int8_t) ((codes >> 16) & 0xFF) * sc, v3 = (float) (int8_t) (codes >> 24) * sc;
        const float4* w4 = reinterpret_cast<const float4*>(spt[cl]);
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        const float ww[PA_G] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w};
#pragma unroll
        for (int h = 0; h < PA_G; ++h) {
            a[0][h] = fmaf(ww[h], v0, a[0][h]);
            a[1][h] = fmaf(ww[h], v1, a[1][h]);
            a[2][h] = fmaf(ww[h], v2, a[2][h]);
            a[3][h] = fmaf(ww[h], v3, a[3][h]);
        }
    };
    const auto vld = [&](int cl, uint32_t& codes, float& sc) {
        codes = __ldg(reinterpret_cast<const uint32_t*>(vq + srow[cl] + d0));
        sc = __half2float(__ushort_as_half(__ldg(vs + sscale[cl] + gi)));
    };
    int cl = 0;
    for (; cl + 4 <= n_here; cl += 4) {
        uint32_t k0, k1, k2, k3;
        float s0, s1, s2, s3;
        vld(cl, k0, s0); vld(cl + 1, k1, s1); vld(cl + 2, k2, s2); vld(cl + 3, k3, s3);
        fold4(cl, k0, s0);
        fold4(cl + 1, k1, s1);
        fold4(cl + 2, k2, s2);
        fold4(cl + 3, k3, s3);
    }
    for (; cl < n_here; ++cl) {
        uint32_t k0;
        float s0;
        vld(cl, k0, s0);
        fold4(cl, k0, s0);
    }
    float* const out = acc_out + (((size_t) row_t * NKVH + kvh) * n_chunks + c) * (PA_G * HD);
#pragma unroll
    for (int h = 0; h < PA_G; ++h) *reinterpret_cast<float4*>(out + (size_t) h * HD + d0) = make_float4(a[0][h], a[1][h], a[2][h], a[3][h]);
}

// block (query head, row): its chunks' sums weighed by exp(m_c - M), as attn()'s partial: part[(t * NQH + h) * (HD + 2)] = acc (HD), m, z
__global__ void __launch_bounds__(HD) k_pa_merge_chunks(const float* __restrict__ blocks, const float* __restrict__ acc_in, int n_chunks, const int* __restrict__ pos0p,
                                                        const int* __restrict__ cells, const int* __restrict__ n_cells, int card, float* __restrict__ part) {
    const int hg = blockIdx.x, row_t = blockIdx.y, i = threadIdx.x, kvh = hg / PA_G, h = hg % PA_G;
    const int nc = cells ? __ldg(n_cells + row_t) : -1;
    const int width = nc >= 0 ? nc : pa_owned(*pos0p + row_t, card);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / PA_CHUNK + 1) : 0;
    const float* const blk0 = blocks + ((size_t) row_t * NKVH + kvh) * n_chunks * PA_BLK;
    float M = -FLT_MAX;
    for (int c = 0; c < n_act; ++c) M = fmaxf(M, __ldg(blk0 + (size_t) c * PA_BLK + PA_G * PA_CHUNK + 4 * PA_CHUNK + h));
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_act; ++c) {
        const float* const mp = blk0 + (size_t) c * PA_BLK + PA_G * PA_CHUNK + 4 * PA_CHUNK;
        const float m = __ldg(mp + h);
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(__ldg(mp + PA_G + h), w, L);
        acc = fmaf(w, __ldg(acc_in + ((((size_t) row_t * NKVH + kvh) * n_chunks + c) * PA_G + h) * HD + i), acc);
    }
    float* const o = part + ((size_t) row_t * NQH + hg) * (HD + 2);
    o[i] = acc;
    if (i == 0) {
        o[HD] = M == -FLT_MAX ? -INFINITY : M;
        o[HD + 1] = L;
    }
}

}  // namespace

bool p100_attn_dec_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_P100_ATTN_DEC");
        return e ? std::atoi(e) != 0 : false;
    }();
    return on;
}
size_t p100_attn_dec_floats(int T) {
    const size_t n_chunks = (qsa_maxsel() + PA_CHUNK - 1) / PA_CHUNK;
    return (size_t) T * NKVH * n_chunks * (PA_BLK + PA_G * HD);
}

// a window's T rows: attn()'s partials for every head over this card's cells (scratch: p100_attn_dec_floats(T))
void p100_attn_window(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells, const int* n_cells, int card, int T,
                      float* part, float* scratch, void* st) {
    if (T == 0) return;
    if (!g.q8) throw std::runtime_error("p100 attention: int8 K/V only");
    cudaStream_t s = (cudaStream_t) st;
    const int n_chunks = (qsa_maxsel() + PA_CHUNK - 1) / PA_CHUNK;
    float* const blocks = scratch;
    float* const acc = scratch + (size_t) T * NKVH * n_chunks * PA_BLK;
    const float scale = 1.0f / sqrtf((float) HD);
    k_pa_score<<<dim3((unsigned) n_chunks, (unsigned) NKVH, (unsigned) T), PA_THREADS, 0, s>>>(qg, reinterpret_cast<const int8_t*>(kc), kc, g, pos0, cells, n_cells, card, 0,
                                                                                               qsa_maxsel(), scale, blocks, n_chunks);
    k_pa_value_chunk<<<dim3((unsigned) n_chunks, (unsigned) NKVH, (unsigned) T), 64, 0, s>>>(reinterpret_cast<const int8_t*>(vc), vc, pos0, cells, n_cells, card, blocks, n_chunks, acc);
    k_pa_merge_chunks<<<dim3((unsigned) NQH, (unsigned) T), HD, 0, s>>>(blocks, acc, n_chunks, pos0, cells, n_cells, card, part);
}

bool p100_attn_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_P100_ATTN");
        return e ? std::atoi(e) != 0 : FNX_ARCH < 70;
    }();
    return on;
}
size_t p100_attn_scratch_floats() { return (size_t) PA_BATCH * NKVH * ((qsa_maxsel() + PA_CHUNK - 1) / PA_CHUNK) * PA_BLK; }

// rows [r0, r0 + T) of a chunk of C: the partials as attn_prompt's
void p100_attn_prompt(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells, const int* n_cells, int card, int C, int r0, int T,
                      float* part, float* scratch, void* st) {
    if (T == 0) return;
    if (!g.q8) throw std::runtime_error("p100 attention: int8 K/V only");
    cudaStream_t s = (cudaStream_t) st;
    const int n_chunks = (qsa_maxsel() + PA_CHUNK - 1) / PA_CHUNK;
    if (n_chunks > PA_NCM) throw std::runtime_error("p100 attention: a selection of " + std::to_string(qsa_maxsel()) + " cells");
    const float scale = 1.0f / sqrtf((float) HD);
    for (int t0 = 0; t0 < T; t0 += PA_BATCH) {
        const int nb = std::min(PA_BATCH, T - t0);
        k_pa_score<<<dim3((unsigned) n_chunks, (unsigned) NKVH, (unsigned) nb), PA_THREADS, 0, s>>>(qg, reinterpret_cast<const int8_t*>(kc), kc, g, pos0, cells, n_cells, card, r0 + t0, qsa_maxsel(),
                                                                                                    scale, scratch, n_chunks);
        k_pa_value<<<dim3((unsigned) NKVH, (unsigned) nb), 64, 0, s>>>(reinterpret_cast<const int8_t*>(vc), vc, pos0, cells, n_cells, card, r0 + t0, scratch, n_chunks, C, part);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100 attention: ") + cudaGetErrorString(e));
}

}  // namespace fnx
