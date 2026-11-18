// The prompt path's routed experts on GP100: Strata's FP16 expert kernel (p100_iq_gemm.cuh) over the split engine's jobs.
// An expert's tokens come from its BigJob instead of a contiguous range, and its matrix from moe_expert (a VRAM slot or the mapped host copy).
#include "kernels.hpp"
#include "p100_over.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"


#include "p100_iq_gemm.cuh"

namespace fnx {
namespace {

using namespace strata::kernels::p100f::iqg;

__device__ __forceinline__ const char* moe_expert_(const char* wa, size_t sa, const char* const* ep, int e) {
    return ep ? ep[e] : wa + (size_t) e * sa;
}

struct JobArgs {
    Tabs t;
    const char* wa;
    size_t sa, row_bytes;
    const char* const* ep;
    const BigJob* jobs;
    const int* njobs;
    const uint4* X;
    float* Y;
    int ldx_u, N, K;
    int fit;   // k_job_v: FNX_PF_JOBV 3's lanes for a block of 256
};

// gemm4_kernel over one tile of one job's tokens: block (rows group, 2 job + tile)
template <int TY>
__launch_bounds__(kThreads)
__global__ void k_gemm4_job(JobArgs a) {
    extern __shared__ uint4 sm[];
    constexpr int TL = kTL, RL = kRL, NT = kThreads, KU = k4KU, TB = k4TB, SU = kSU<TY>;
    uint4* const sw = sm;                 // [k4NB rows][KU]
    uint4* const sx = sm + k4NB * KU;     // [TB tokens][KU]
    uint4* const sg = sx + TB * KU;       // [k4NB rows][SU]: a row's IQ block
    uint4* const tab = sg + k4NB * SU;    // the tables
    int* const sj = reinterpret_cast<int*>(tab + TabU<TY>::n);   // the job
    const int tid = int(threadIdx.x), ji = int(blockIdx.y) / 2, tile = int(blockIdx.y) & 1;
    if (ji >= *a.njobs) return;
    const int* const srcj = reinterpret_cast<const int*>(a.jobs + ji);
    if (tile * TB >= srcj[1]) return;
    for (int k = tid; k < int(sizeof(BigJob) / 4); k += NT) sj[k] = srcj[k];
    const int t00 = tile * TB, n00 = int(blockIdx.x) * k4NB, N = a.N;
    const int ni = tid & 15, ti = tid >> 4;
    const int n0 = n00 + ni * RL, t0 = t00 + ti * TL;
    const int my_row = n00 + tid;
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
    const BigJob& jb = *reinterpret_cast<const BigJob*>(sj);
    const int tend = jb.n;
    const bool act = t0 < tend && n0 < N;
    const uint8_t* const we = reinterpret_cast<const uint8_t*>(moe_expert_(a.wa, a.sa, a.ep, jb.w));
    const int nu = a.K / 8;
    int off = 0;
    for (int u0 = 0; u0 < nu; u0 += KU) {
        const int run0 = u0 >> 2;
        if ((run0 & (kRunsGr<TY> - 1)) == 0 && my_row < N) {
            const uintptr_t ad = reinterpret_cast<uintptr_t>(we + std::size_t(my_row) * row_bytes + std::size_t(run0 / kRunsGr<TY>) * kGrBytes<TY>);
            const uint4* al = reinterpret_cast<const uint4*>(ad & ~uintptr_t(15));
            const uintptr_t row_end = reinterpret_cast<uintptr_t>(we + std::size_t(my_row + 1) * row_bytes);   // the last row may end a card's allocation
            off = int(ad & 15);
            uint4 v[SU];
#pragma unroll
            for (int q = 0; q < SU; ++q) v[q] = reinterpret_cast<uintptr_t>(al + q) < row_end ? __ldg(al + q) : make_uint4(0, 0, 0, 0);
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
                if (i < TB * KU && t00 + r < tend) v[q] = __ldg(X + std::size_t(jb.y[t00 + r]) * ldx_u + u0 + c);
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
#define P100_MACJ(F) \
                _Pragma("unroll") \
                for (int i = 0; i < TL; ++i) \
                    _Pragma("unroll") \
                    for (int r = 0; r < RL; ++r) h[i][r] = __hfma2(as_half2(wr[r].F), as_half2(xr[i].F), h[i][r]);
                P100_MACJ(x) P100_MACJ(y) P100_MACJ(z) P100_MACJ(w)
#undef P100_MACJ
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
                float* yr = a.Y + std::size_t(jb.out[t0 + i]) * N + n0;
#pragma unroll
                for (int r = 0; r < RL; ++r) yr[r] = s[i][r];
            }
    }
}

template <int TY>
void job_launch(const JobArgs& a, int njobs_cap, cudaStream_t s) {
    constexpr std::size_t smem = (std::size_t(k4NB) * k4KU + std::size_t(k4TB) * k4KU + std::size_t(k4NB) * kSU<TY> + TabU<TY>::n) * 16 + sizeof(BigJob);
    static_assert(smem <= 48 * 1024, "p100 moe: shared memory past Pascal's limit");
    static_assert(PF_JOB <= 2 * k4TB, "p100 moe: a job fills at most two tiles");
    k_gemm4_job<TY><<<dim3((a.N + k4NB - 1) / k4NB, njobs_cap * 2), kThreads, smem, s>>>(a);
}

// ---- FNX_PF_JOBV (2026-11-10): k_gemm4_job's sums in k_gemm4_job's order, the lanes fitted to the tile.
//   1: a tile with few tokens takes lanes of fewer rows x tokens, so all lanes work and the products' time follows the tokens
//      (k_gemm4_job: 8 rows x 6 tokens a lane whatever the tile holds, the lanes past the tile's tokens idle);
//   2: also one block of 256 lanes for a whole job of 96 tokens where its shared memory fits, the weights decoded once
//      (two tiles of 48: twice). IQ4_XS does not fit and stays with 128 lanes.
//   3: 2, and a block of 256 with at most 36 tokens takes 1's lanes for them: its first 128 lanes do the products with fewer loads a
//      product, the other 128 only decode (2's lanes of half the rows x tokens for all 256 were slower from 6 to 36 tokens).
// The job is read where it lies (a lane's activation rows and outputs), not copied to shared memory.
// An accumulator's order of sums is k_gemm4_job's: the columns of 8 units rotated by its row group of 8's index & 7, a fold after
// every 8 units. mikro/k35/k35b_test.cu compares the outputs bit for bit.
template <int TY, int NT> inline constexpr int kJobVTab = TY == 21 && NT > k4NB ? 256 : TabU<TY>::n;   // IQ3_S in 256 lanes: its grid only
template <int TY, int NT> inline constexpr int kJobVSmem = (k4NB * k4KU + NT / k4NB * k4TB * k4KU + k4NB * kSU<TY> + kJobVTab<TY, NT>) * 16;

template <int TY, int NT>
__device__ __forceinline__ void jobv_tabs(const Tabs& t, uint4* tab, const int tid) {
    if constexpr (TY == 21 && NT > k4NB) {
        for (int i = tid; i < 256; i += NT) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.g3s) + i);
    } else {
        stage_tabs<TY>(t, tab, tid, NT);
    }
}

template <int TY, int NT>
__device__ __forceinline__ void jobv_finish(const uint4* tab, const Tabs& t, const uint32_t (&w)[RunWords<TY>::n], uint4* out) {
    if constexpr (TY == 21 && NT > k4NB) {   // finish_s<21> with the sign masks computed (tabs_kernel's mask8): their 4 KB do not fit beside 96 tokens
        const uint2* grid = reinterpret_cast<const uint2*>(tab);
        const uint32_t qh = w[5] & 255, sc = (w[5] >> 8) & 255;
        const half2 dd = dup_f(h2f(uint16_t(w[5] >> 16)) * float(1 + 2 * sc));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t i1 = (w[j] & 255) | ((qh << (8 - 2 * j)) & 256), i2 = (w[j] >> 8) | ((qh << (7 - 2 * j)) & 256);
            const uint32_t sb = w[4] >> (8 * j);
            const uint2 g1 = grid[i1], g2 = grid[i2];
            out[j] = make_uint4(hmul(g1.x ^ (((sb & 1u) << 15) | ((sb & 2u) << 30)), dd), hmul(g1.y ^ (((sb & 4u) << 13) | ((sb & 8u) << 28)), dd),
                                hmul(g2.x ^ (((sb & 16u) << 11) | ((sb & 32u) << 26)), dd), hmul(g2.y ^ (((sb & 64u) << 9) | ((sb & 128u) << 24)), dd));
        }
    } else {
        finish_s<TY>(tab, t, w, out);
    }
}

// one tile's products with lanes of RL rows x TL tokens: 128 / RL row groups by NT RL / 128 token groups
template <int TY, int NT, int RL, int TL, bool PF, int NBX = k4NB>
__device__ __forceinline__ void jobv_tile(const JobArgs& a, uint4* const sm, const BigJob* const jg, const int t00, const int tend) {
    constexpr int KU = k4KU, SU = kSU<TY>, NB = NBX, TBX = NBX == k4NB ? NT / NB * k4TB : 2 * k4TB;   // jobv5: 64 rows x 96 tokens
    constexpr int NRG = NB / RL, TG = NT / NRG, TBm = TG * TL, XL = (TBm * KU + NT - 1) / NT;
    uint4* const sw = sm;                 // [NB rows][KU]
    uint4* const sx = sm + NB * KU;       // [TBX tokens][KU]
    uint4* const sg = sx + TBX * KU;      // [NB rows][SU]: a row's IQ block
    const uint4* const tab = sg + NB * SU;
    const int tid = int(threadIdx.x), n00 = int(blockIdx.x) * NB, N = a.N;
    const int rg = tid % NRG, tg = tid / NRG;
    const int n0 = n00 + rg * RL, t0 = t00 + tg * TL;
    const int dl = tid & (NB - 1), dj = tid / NB, drow = n00 + dl;   // the row a lane decodes, and which of its two runs (NT = 256)
    const uint4* const X = a.X;
    const int ldx_u = a.ldx_u;
    const std::size_t row_bytes = a.row_bytes;
    const Tabs t = a.t;
    int xi[XL];   // the activation rows this lane brings
#pragma unroll
    for (int q = 0; q < XL; ++q) {
        const int i = tid + q * NT, r = t00 + i / KU;
        xi[q] = i < TBm * KU && r < tend ? __ldg(&jg->y[r]) : -1;
    }
    half2 h[TL][RL];
    float s[TL][RL];
#pragma unroll
    for (int i = 0; i < TL; ++i)
#pragma unroll
        for (int r = 0; r < RL; ++r) { h[i][r] = as_half2(0u); s[i][r] = 0.0f; }
    const bool act = t0 < tend && n0 < N;
    const uint8_t* const we = reinterpret_cast<const uint8_t*>(moe_expert_(a.wa, a.sa, a.ep, __ldg(&jg->w)));
    const int nu = a.K / 8;
    int off = 0;
    // PF (FNX_PF_JOBV 4, jobv4_patch.py): the group's bytes and the step's activation rows come from registers loaded a step (a group) ahead
    uint4 vx[PF ? XL : 1], vg[PF ? SU : 1];
    int offn = 0;
    auto g_load = [&](const int run) {
        if (drow < N) {
            const uintptr_t ad = reinterpret_cast<uintptr_t>(we + std::size_t(drow) * row_bytes + std::size_t(run / kRunsGr<TY>) * kGrBytes<TY>);
            offn = int(ad & 15);
            if (dj == 0) {
                const uint4* al = reinterpret_cast<const uint4*>(ad & ~uintptr_t(15));
                const uintptr_t row_end = reinterpret_cast<uintptr_t>(we + std::size_t(drow + 1) * row_bytes);
#pragma unroll
                for (int q = 0; q < (PF ? SU : 1); ++q) vg[q] = reinterpret_cast<uintptr_t>(al + q) < row_end ? __ldg(al + q) : make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto x_load = [&](const int u) {
#pragma unroll
        for (int q = 0; q < (PF ? XL : 1); ++q)
            if (xi[q] >= 0) vx[q] = __ldg(X + std::size_t(xi[q]) * ldx_u + u + (tid & (KU - 1)));
    };
    if constexpr (PF) {
        g_load(0);
        x_load(0);
    }
    for (int u0 = 0; u0 < nu; u0 += KU) {
        const int run0 = u0 >> 2;
        if (PF && (run0 & (kRunsGr<TY> - 1)) == 0) {
            if (drow < N) {
                off = offn;
                if (dj == 0) {
#pragma unroll
                    for (int q = 0; q < (PF ? SU : 1); ++q) sg[dl * SU + q] = vg[q];
                }
            }
            if ((run0 + kRunsGr<TY>) * 4 < nu) g_load(run0 + kRunsGr<TY>);   // in flight for the group's steps
            if (NT > NB) __syncthreads();
        } else if (!PF && (run0 & (kRunsGr<TY> - 1)) == 0) {
            if (drow < N) {
                const uintptr_t ad = reinterpret_cast<uintptr_t>(we + std::size_t(drow) * row_bytes + std::size_t(run0 / kRunsGr<TY>) * kGrBytes<TY>);
                off = int(ad & 15);
                if (dj == 0) {
                    const uint4* al = reinterpret_cast<const uint4*>(ad & ~uintptr_t(15));
                    const uintptr_t row_end = reinterpret_cast<uintptr_t>(we + std::size_t(drow + 1) * row_bytes);   // the last row may end a card's allocation
                    uint4 v[SU];
#pragma unroll
                    for (int q = 0; q < SU; ++q) v[q] = reinterpret_cast<uintptr_t>(al + q) < row_end ? __ldg(al + q) : make_uint4(0, 0, 0, 0);
#pragma unroll
                    for (int q = 0; q < SU; ++q) sg[dl * SU + q] = v[q];
                }
            }
            if (NT > NB) __syncthreads();   // the row's second lane decodes from the first's bytes
        }
        if (drow < N) {
            const uint8_t* gp = reinterpret_cast<const uint8_t*>(sg + dl * SU) + off;
            if constexpr (NT == NB) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    uint32_t rw[RunWords<TY>::n];
                    run_load_s<TY>(gp, run0 + j, rw);
                    uint4 o[4];
                    jobv_finish<TY, NT>(tab, t, rw, o);
#pragma unroll
                    for (int q = 0; q < 4; ++q) sw[dl * KU + 4 * j + q] = o[q];
                }
            } else if (dj < KU / 4) {   // jobv5: 64 rows, a step's two runs on a row's first two of four lanes
                uint32_t rw[RunWords<TY>::n];
                run_load_s<TY>(gp, run0 + dj, rw);
                uint4 o[4];
                jobv_finish<TY, NT>(tab, t, rw, o);
#pragma unroll
                for (int q = 0; q < 4; ++q) sw[dl * KU + 4 * dj + q] = o[q];
            }
        }
        if constexpr (PF) {
#pragma unroll
            for (int q = 0; q < XL; ++q)
                if (xi[q] >= 0) sx[tid + q * NT] = vx[q];
            if (u0 + KU < nu) x_load(u0 + KU);   // in flight across the products
        } else {
            uint4 v[XL];
#pragma unroll
            for (int q = 0; q < XL; ++q)
                if (xi[q] >= 0) v[q] = __ldg(X + std::size_t(xi[q]) * ldx_u + u0 + (tid & (KU - 1)));
#pragma unroll
            for (int q = 0; q < XL; ++q)
                if (xi[q] >= 0) sx[tid + q * NT] = v[q];
        }
        __syncthreads();
        if (act) {
            const int wb = rg * RL * KU, xb = tg * TL * KU;
            int cc = ((rg * RL) >> 3) & (KU - 1);   // k_gemm4_job's rotation: the row group of 8's index
            for (int c = 0; c < KU; ++c) {
                uint4 wr[RL], xr[TL];
#pragma unroll
                for (int r = 0; r < RL; ++r) wr[r] = sw[wb + r * KU + cc];
#pragma unroll
                for (int i = 0; i < TL; ++i) xr[i] = sx[xb + i * KU + cc];
                cc = (cc + 1) & (KU - 1);
#define P100_MACV(F) \
                _Pragma("unroll") \
                for (int i = 0; i < TL; ++i) \
                    _Pragma("unroll") \
                    for (int r = 0; r < RL; ++r) h[i][r] = __hfma2(as_half2(wr[r].F), as_half2(xr[i].F), h[i][r]);
                P100_MACV(x) P100_MACV(y) P100_MACV(z) P100_MACV(w)
#undef P100_MACV
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
                float* yr = a.Y + std::size_t(__ldg(&jg->out[t0 + i])) * N + n0;
#pragma unroll
                for (int r = 0; r < RL; ++r) yr[r] = s[i][r];
            }
    }
}

// block (rows group, tile of a job): NT = 128 two tiles of 48 tokens a job, NT = 256 one of 96
template <int TY, int NT, bool PF>
__launch_bounds__(NT)
__global__ void k_job_v(JobArgs a) {
    extern __shared__ uint4 sm[];
    constexpr int W = NT / k4NB, TBX = W * k4TB, TPJ = (PF_JOB + TBX - 1) / TBX;
    const int ji = int(blockIdx.y) / TPJ, tile = int(blockIdx.y) % TPJ;
    if (ji >= *a.njobs) return;
    const BigJob* const jg = a.jobs + ji;
    const int tend = __ldg(&jg->n), t00 = tile * TBX;
    if (t00 >= tend) return;
    jobv_tabs<TY, NT>(a.t, sm + k4NB * k4KU + TBX * k4KU + k4NB * kSU<TY>, int(threadIdx.x));
    __syncthreads();
    const int nt = tend - t00;
    const int w = a.fit && nt <= 36 ? 1 : W;   // the lanes for the tokens of one group of 128, or of all the block's
    if (nt > 36 * w) jobv_tile<TY, NT, 8, 6, PF>(a, sm, jg, t00, tend);
    else if (nt > 24 * w) jobv_tile<TY, NT, 4, 9, PF>(a, sm, jg, t00, tend);
    else if (nt > 18 * w) jobv_tile<TY, NT, 4, 6, PF>(a, sm, jg, t00, tend);
    else if (nt > 12 * w) jobv_tile<TY, NT, 2, 9, PF>(a, sm, jg, t00, tend);
    else if (nt > 9 * w) jobv_tile<TY, NT, 2, 6, PF>(a, sm, jg, t00, tend);
    else if (nt > 6 * w) jobv_tile<TY, NT, 1, 9, PF>(a, sm, jg, t00, tend);
    else if (nt > 3 * w) jobv_tile<TY, NT, 1, 6, PF>(a, sm, jg, t00, tend);
    else jobv_tile<TY, NT, 1, 3, PF>(a, sm, jg, t00, tend);
}

// FNX_PF_JOBV 5 (jobv5_patch.py): block (64 rows, a job of at most 96 tokens), two blocks an SM
template <int TY, int NT> inline constexpr int kJobV64Smem = (64 * k4KU + 2 * k4TB * k4KU + 64 * kSU<TY> + kJobVTab<TY, NT>) * 16;
template <int TY, int NT>
__launch_bounds__(NT, 2)
__global__ void k_job_v64(JobArgs a) {
    extern __shared__ uint4 sm[];
    const int ji = int(blockIdx.y);
    if (ji >= *a.njobs) return;
    const BigJob* const jg = a.jobs + ji;
    const int tend = __ldg(&jg->n);
    if (tend <= 0) return;
    jobv_tabs<TY, NT>(a.t, sm + 64 * k4KU + 2 * k4TB * k4KU + 64 * kSU<TY>, int(threadIdx.x));
    __syncthreads();
    const int nt = tend;
    if (nt > 72) jobv_tile<TY, NT, 4, 6, false, 64>(a, sm, jg, 0, tend);
    else if (nt > 48) jobv_tile<TY, NT, 2, 9, false, 64>(a, sm, jg, 0, tend);
    else if (nt > 36) jobv_tile<TY, NT, 2, 6, false, 64>(a, sm, jg, 0, tend);
    else if (nt > 24) jobv_tile<TY, NT, 1, 9, false, 64>(a, sm, jg, 0, tend);
    else if (nt > 12) jobv_tile<TY, NT, 1, 6, false, 64>(a, sm, jg, 0, tend);
    else jobv_tile<TY, NT, 1, 3, false, 64>(a, sm, jg, 0, tend);
}

// ---- FNX_PF_JOBV 7 (jobv7_patch.py, idea 57): mode 4's tiles, a lane's run words straight from the expert two steps ahead
// (run_load), the decoded weights in two buffers [2][rows][KU]: step st decodes step st + 1 into buffer (st + 1) & 1, then multiplies
// from buffer st & 1, then one barrier; the activations one buffer, stored after it, then a second barrier. Sums as k_gemm4_job's,
// in its order.
template <int TY> inline constexpr int kJobV7Smem = (2 * k4NB * k4KU + 2 * k4TB * k4KU + kJobVTab<TY, 256>) * 16;

template <int TY, int RL, int TL>
__device__ __forceinline__ void jobv7_tile(const JobArgs& a, uint4* const sm, const BigJob* const jg, const int t00, const int tend) {
    constexpr int NT = 256, KU = k4KU, NB = k4NB, TBX = 2 * k4TB;
    constexpr int NRG = NB / RL, TG = NT / NRG, TBm = TG * TL, XL = (TBm * KU + NT - 1) / NT;
    uint4* const sx = sm + 2 * NB * KU;   // [TBX tokens][KU]
    const uint4* const tab = sx + TBX * KU;
    const int tid = int(threadIdx.x), n00 = int(blockIdx.x) * NB, N = a.N;
    const int rg = tid % NRG, tg = tid / NRG;
    const int n0 = n00 + rg * RL, t0 = t00 + tg * TL;
    const int dl = tid & (NB - 1), dj = tid / NB, drow = n00 + dl;   // the row a lane decodes, and which of the step's two runs
    const uint4* const X = a.X;
    const int ldx_u = a.ldx_u;
    const Tabs t = a.t;
    int xi[XL];   // the activation rows this lane brings
#pragma unroll
    for (int q = 0; q < XL; ++q) {
        const int i = tid + q * NT, r = t00 + i / KU;
        xi[q] = i < TBm * KU && r < tend ? __ldg(&jg->y[r]) : -1;
    }
    half2 h[TL][RL];
    float s[TL][RL];
#pragma unroll
    for (int i = 0; i < TL; ++i)
#pragma unroll
        for (int r = 0; r < RL; ++r) { h[i][r] = as_half2(0u); s[i][r] = 0.0f; }
    const bool act = t0 < tend && n0 < N;
    const uint8_t* const we = reinterpret_cast<const uint8_t*>(moe_expert_(a.wa, a.sa, a.ep, __ldg(&jg->w)));
    const uint8_t* const wrow = we + std::size_t(min(drow, N - 1)) * a.row_bytes;   // a row past N decodes the last (its products unused)
    const int ns = a.K / 8 / KU;   // steps of KU units: step st holds runs 2 st and 2 st + 1
    uint32_t rw[RunWords<TY>::n], rn[RunWords<TY>::n];
    uint4 vx[XL];
    const auto x_load = [&](const int u) {
#pragma unroll
        for (int q = 0; q < XL; ++q)
            if (xi[q] >= 0) vx[q] = __ldg(X + std::size_t(xi[q]) * ldx_u + u + (tid & (KU - 1)));
    };
    const auto x_store = [&] {
#pragma unroll
        for (int q = 0; q < XL; ++q)
            if (xi[q] >= 0) sx[tid + q * NT] = vx[q];
    };
    const auto decode = [&](uint4* __restrict__ const buf) {
        uint4 o[4];
        jobv_finish<TY, NT>(tab, t, rw, o);
#pragma unroll
        for (int q = 0; q < 4; ++q) buf[dl * KU + 4 * dj + q] = o[q];
    };
    // step 0 decoded and stored; step 1's words and activations in flight
    run_load<TY>(wrow, dj, rw);
    x_load(0);
    decode(sm);
    x_store();
    if (ns > 1) {
        run_load<TY>(wrow, 2 + dj, rw);
        x_load(KU);
    }
    __syncthreads();
    for (int st = 0; st < ns; ++st) {
        const uint4* __restrict__ const cur = sm + (st & 1) * NB * KU;   // apart from the buffer the decode writes (as k35f measured it)
        if (st + 2 < ns) run_load<TY>(wrow, 2 * (st + 2) + dj, rn);   // in flight across the step
        decode(sm + ((st + 1) & 1) * NB * KU);   // the next step's weights (the last step's: a buffer nobody reads)
        if (act) {
            const int wb = rg * RL * KU, xb = tg * TL * KU;
            int cc = ((rg * RL) >> 3) & (KU - 1);   // k_gemm4_job's rotation: the row group of 8's index
#pragma unroll
            for (int c = 0; c < KU; ++c) {
                uint4 wr[RL], xr[TL];
#pragma unroll
                for (int r = 0; r < RL; ++r) wr[r] = cur[wb + r * KU + cc];
#pragma unroll
                for (int i = 0; i < TL; ++i) xr[i] = sx[xb + i * KU + cc];
                cc = (cc + 1) & (KU - 1);
#define P100_MACV(F) \
                _Pragma("unroll") \
                for (int i = 0; i < TL; ++i) \
                    _Pragma("unroll") \
                    for (int r = 0; r < RL; ++r) h[i][r] = __hfma2(as_half2(wr[r].F), as_half2(xr[i].F), h[i][r]);
                P100_MACV(x) P100_MACV(y) P100_MACV(z) P100_MACV(w)
#undef P100_MACV
            }
#pragma unroll
            for (int i = 0; i < TL; ++i)
#pragma unroll
                for (int r = 0; r < RL; ++r) fold_one(h[i][r], s[i][r]);
        }
        __syncthreads();   // the products are done with sx and with cur, the next step's weights are in the other buffer
        if (st + 1 < ns) x_store();
        if (st + 2 < ns) {
            x_load((st + 2) * KU);
#pragma unroll
            for (int q = 0; q < RunWords<TY>::n; ++q) rw[q] = rn[q];
        }
        __syncthreads();
    }
    if (act) {
#pragma unroll
        for (int i = 0; i < TL; ++i)
            if (t0 + i < tend) {
                float* yr = a.Y + std::size_t(__ldg(&jg->out[t0 + i])) * N + n0;
#pragma unroll
                for (int r = 0; r < RL; ++r) yr[r] = s[i][r];
            }
    }
}

// block (rows group, tile of a job of at most 96 tokens), 256 lanes, mode 4's lanes for the tokens
template <int TY>
__launch_bounds__(256)
__global__ void k_job_v7(JobArgs a) {
    extern __shared__ uint4 sm[];
    constexpr int TBX = 2 * k4TB, TPJ = (PF_JOB + TBX - 1) / TBX;
    const int ji = int(blockIdx.y) / TPJ, tile = int(blockIdx.y) % TPJ;
    if (ji >= *a.njobs) return;
    const BigJob* const jg = a.jobs + ji;
    const int tend = __ldg(&jg->n), t00 = tile * TBX;
    if (t00 >= tend) return;
    jobv_tabs<TY, 256>(a.t, sm + 2 * k4NB * k4KU + TBX * k4KU, int(threadIdx.x));
    __syncthreads();
    const int nt = tend - t00;
    const int w = a.fit && nt <= 36 ? 1 : 2;
    if (nt > 36 * w) jobv7_tile<TY, 8, 6>(a, sm, jg, t00, tend);
    else if (nt > 24 * w) jobv7_tile<TY, 4, 9>(a, sm, jg, t00, tend);
    else if (nt > 18 * w) jobv7_tile<TY, 4, 6>(a, sm, jg, t00, tend);
    else if (nt > 12 * w) jobv7_tile<TY, 2, 9>(a, sm, jg, t00, tend);
    else if (nt > 9 * w) jobv7_tile<TY, 2, 6>(a, sm, jg, t00, tend);
    else if (nt > 6 * w) jobv7_tile<TY, 1, 9>(a, sm, jg, t00, tend);
    else if (nt > 3 * w) jobv7_tile<TY, 1, 6>(a, sm, jg, t00, tend);
    else jobv7_tile<TY, 1, 3>(a, sm, jg, t00, tend);
}

template <int TY>
void jobv_launch(const JobArgs& a, int njobs_cap, int mode, cudaStream_t s) {
    static_assert(kJobVSmem<TY, 128> <= 48 * 1024, "p100 moe: shared memory past Pascal's limit");
    if (mode == 7) {   // jobv7: IQ2_S and where the two buffers do not fit, mode 4
        if constexpr (TY != 22 && kJobV7Smem<TY> <= 48 * 1024) {
            k_job_v7<TY><<<dim3(unsigned((a.N + k4NB - 1) / k4NB), njobs_cap * ((PF_JOB + 2 * k4TB - 1) / (2 * k4TB))), 256, kJobV7Smem<TY>, s>>>(a);
            return;
        }
        mode = 4;
    }
    const unsigned rows = unsigned((a.N + k4NB - 1) / k4NB);
    if constexpr (kJobVSmem<TY, 256> <= 48 * 1024) {
        if (mode == 5) {
            static_assert(PF_JOB <= 2 * k4TB, "jobv5: a job in one tile of 96");
            k_job_v64<TY, 256><<<dim3(unsigned((a.N + 63) / 64), njobs_cap), 256, kJobV64Smem<TY, 256>, s>>>(a);
            return;
        }
        if (mode >= 2) {
            if (mode >= 4) k_job_v<TY, 256, true><<<dim3(rows, njobs_cap * ((PF_JOB + 2 * k4TB - 1) / (2 * k4TB))), 256, kJobVSmem<TY, 256>, s>>>(a);
            else k_job_v<TY, 256, false><<<dim3(rows, njobs_cap * ((PF_JOB + 2 * k4TB - 1) / (2 * k4TB))), 256, kJobVSmem<TY, 256>, s>>>(a);
            return;
        }
    }
    if (mode >= 4) k_job_v<TY, 128, true><<<dim3(rows, njobs_cap * ((PF_JOB + k4TB - 1) / k4TB)), 128, kJobVSmem<TY, 128>, s>>>(a);
    else k_job_v<TY, 128, false><<<dim3(rows, njobs_cap * ((PF_JOB + k4TB - 1) / k4TB)), 128, kJobVSmem<TY, 128>, s>>>(a);
}

}  // namespace

// mode 0: k_gemm4_job; 1, 2, 3, 4: FNX_PF_JOBV's kernel (the same bits)
void p100_moe_mode(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* x, float* out, int mode, void* st) {
    cudaStream_t s = (cudaStream_t) st;
    if (A.n_in % 64) throw std::runtime_error("p100 moe: columns not in units of 64");
    JobArgs a{};
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
    if (mode == 6) mode = A.n_in <= 640 ? 5 : 4;   // jobv6: down (K 320 a card) on mode 5's tiles, gate|up on mode 4
    if (mode == 7) mode = A.n_in <= 640 ? 5 : 7;   // jobv7: down as 6, gate|up on k_job_v7 (IQ2_S on mode 4)
    a.fit = mode >= 3;   // jobv4: mode 4 is 3 with its loads ahead; jobv5: mode 5 its own kernel
#define P100_JOB(TY) (mode > 0 ? jobv_launch<TY>(a, njobs_cap, mode, s) : job_launch<TY>(a, njobs_cap, s))
    switch (A.type) {
        case T_IQ3_XXS: P100_JOB(18); break;
        case T_IQ3_S: P100_JOB(21); break;
        case T_IQ2_S: P100_JOB(22); break;
        case T_IQ4_XS: P100_JOB(23); break;
        case T_IQ4_NL: P100_JOB(20); break;
        case T_NVFP4: P100_JOB(40); break;   // edge0_patch.py
        case T_Q4_K: P100_JOB(12); break;   // q4x_patch.py
        case T_Q5_K: P100_JOB(13); break;
        case T_Q5_1: P100_JOB(7); break;
        case T_Q8_0: P100_JOB(8); break;
        case T_Q4_0: P100_JOB(2); break;
        case T_IQ2_XXS: job_launch<16>(a, njobs_cap, s); break;
        case T_IQ1_M: job_launch<29>(a, njobs_cap, s); break;
        default: throw std::runtime_error("p100 moe: expert type " + std::to_string(A.type) + " not handled");
    }
#undef P100_JOB
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("p100 moe: ") + cudaGetErrorString(e));
}

void p100_moe(const DW& A, size_t sa, const BigJob* jobs, const int* njobs, int njobs_cap, const uint16_t* x, float* out, void* st) {
    static const int mode = [] {
        const char* e = std::getenv("FNX_PF_JOBV");
        const int m = e ? std::atoi(e) : 0;
        if (m < 0 || m > 7) throw std::runtime_error("FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes), 4 (3, the next step's loads ahead), 5 (tiles of 64 rows, two blocks an SM), 6 (down as 5, gate|up as 4) or 7 (down as 5, gate|up with two weight buffers and the words from the expert, IQ2_S as 4)");
        if (m) std::fprintf(stderr, "fnx: FNX_PF_JOBV %d: the prompt path's expert kernel with fitted lanes\n", m);
        return m;
    }();
    p100_moe_mode(A, sa, jobs, njobs, njobs_cap, x, out, mode, st);
}

}  // namespace fnx

#include "p100_moe_rows.cuh"
#include "p100_attn.cuh"
#include "p100_dma.cuh"

namespace fnx {
namespace {
NocMode noc_from_env() {
    NocMode m;
    if (const char* e = std::getenv("FNX_ROWS_G2")) m.g2 = std::atoi(e);
    if (const char* e = std::getenv("FNX_DMA_SPLIT")) m.split = std::max(0, std::min(2, std::atoi(e)));
    if (const char* e = std::getenv("FNX_MEMO")) m.memo = std::atoi(e);
    if (m.g2 != 0 && m.g2 != 4 && m.g2 != 8 && m.g2 != 9) throw std::runtime_error("FNX_ROWS_G2: 0, 4 or 8 lanes a row, 9 by the window");
    return m;
}
NocMode& noc_store() {
    static NocMode m = noc_from_env();
    return m;
}
std::atomic<int> noc_gen_{0};
}  // namespace

const NocMode& noc_mode() { return noc_store(); }
int noc_gen() { return noc_gen_.load(std::memory_order_relaxed); }

// FNX_NOC_FILE: "g2 split memo" at most every 20 ms; true when a mode changed (the next windows are captured again)
bool noc_tick() {
    static const char* file = std::getenv("FNX_NOC_FILE");
    if (!file) return false;
    static std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds(20)) return false;
    last = now;
    NocMode m = noc_store();
    if (FILE* f = std::fopen(file, "r")) {
        int g2 = 0, split = 0, memo = 0;
        if (std::fscanf(f, "%d %d %d", &g2, &split, &memo) == 3) m.g2 = g2 == 4 || g2 == 8 || g2 == 9 ? g2 : 0, m.split = std::max(0, std::min(2, split)), m.memo = memo;
        std::fclose(f);
    }
    NocMode& cur = noc_store();
    if (m.g2 == cur.g2 && m.split == cur.split && m.memo == cur.memo) return false;
    cur = m;
    noc_gen_.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "fnx: FNX_NOC_FILE: g2 %d split %d memo %d\n", m.g2, m.split, m.memo);
    return true;
}
}  // namespace fnx

// ---- the routing memo (memo_patch.py)
namespace fnx {
namespace {
__global__ void k_memo_put_win(const int* __restrict__ topi, int T, const int* __restrict__ ctl, int layer, uint16_t* memo, int cap) {
    const int i = threadIdx.x;
    if (i >= T * TOPK) return;
    const int pos = ctl[0] + i / TOPK;
    memo[((size_t) (pos % cap) * NL + layer) * TOPK + i % TOPK] = (uint16_t) topi[i];
}
__global__ void k_memo_put_rows(const int* __restrict__ topi, int C, int pos0, int layer, uint16_t* memo, int cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * TOPK) return;
    memo[((size_t) ((pos0 + i / TOPK) % cap) * NL + layer) * TOPK + i % TOPK] = (uint16_t) topi[i];
}
// mg: [NL][MAXT][TOPK], -1 where a row has no memo; mg[NL * MAXT * TOPK]: 1 when any row has one
__global__ void k_memo_merge(const int* __restrict__ mg, int layer, int T, int* guess) {
    if (!mg[NL * MAXT * TOPK]) return;
    const int i = threadIdx.x, r = i / TOPK;
    if (r >= T) return;
    const int* row = mg + ((size_t) layer * MAXT + r) * TOPK;
    if (row[0] >= 0) guess[i] = row[i % TOPK];
}
// stat: [0] layer-windows, [1] experts missing at the plan, [2] guessed experts copied ahead (pf_n: this layer planned the next),
// [3] rows with a memo guess, [4] hits of their memo's first 6 among the row's 10 picks, [5] hits of the probe's first 6 (layers 1..)
__global__ void k_memo_stat(const int* __restrict__ topi, const int* __restrict__ mg, const int* __restrict__ probe, const int* __restrict__ cache_list,
                            int layer, int T, int pf_n, const int* __restrict__ pf_list, unsigned long long* stat) {
    const int r = threadIdx.x;
    if (r == 0) {
        atomicAdd(&stat[0], 1ull);
        if (cache_list) atomicAdd(&stat[1], (unsigned long long) cache_list[0]);
        if (pf_n && pf_list) atomicAdd(&stat[2], (unsigned long long) pf_list[0]);
    }
    if (r >= T || !mg[NL * MAXT * TOPK] || layer == 0) return;
    const int* row = mg + ((size_t) layer * MAXT + r) * TOPK;
    if (row[0] < 0) return;
    int hm = 0, hp = 0;
    for (int k = 0; k < 6; ++k)
        for (int j = 0; j < TOPK; ++j) {
            hm += row[k] == topi[r * TOPK + j];
            hp += probe[r * TOPK + k] == topi[r * TOPK + j];
        }
    atomicAdd(&stat[3], 1ull);
    atomicAdd(&stat[4], (unsigned long long) hm);
    atomicAdd(&stat[5], (unsigned long long) hp);
}
}  // namespace

void memo_put_win(const int* topi, int T, const int* ctl, int layer, uint16_t* memo, int cap, void* st) {
    k_memo_put_win<<<1, MAXT * TOPK, 0, (cudaStream_t) st>>>(topi, T, ctl, layer, memo, cap);
}
void memo_put_rows(const int* topi, int C, int pos0, int layer, uint16_t* memo, int cap, void* st) {
    if (C > 0) k_memo_put_rows<<<(C * TOPK + 255) / 256, 256, 0, (cudaStream_t) st>>>(topi, C, pos0, layer, memo, cap);
}
void memo_merge(const int* mg, int layer, int T, int* guess, void* st) {
    k_memo_merge<<<1, MAXT * TOPK, 0, (cudaStream_t) st>>>(mg, layer, T, guess);
}
void memo_stat(const int* topi, const int* mg, const int* probe, const int* cache_list, int layer, int T, int pf_n, const int* pf_list,
               unsigned long long* stat, void* st) {
    k_memo_stat<<<1, 32, 0, (cudaStream_t) st>>>(topi, mg, probe, cache_list, layer, T, pf_n, pf_list, stat);
}
}  // namespace fnx
