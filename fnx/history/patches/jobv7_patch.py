#!/usr/bin/env python3
# jobv7_patch.py SRC_DIR: idea 57 (2026-11-12). FNX_PF_JOBV 7: down as 5; gate|up on k_job_v7, except IQ2_S (mode 4). k_job_v7 is mode
# 4's tile (128 rows x 96 tokens, 256 lanes) with a lane's run words read straight from the expert two steps ahead (run_load: the 16 KB
# of the rows' IQ blocks in shared memory go) and the decoded weights in two buffers of 16 KB: a step decodes the next step's weights
# into the other buffer before its products, so one barrier waits for both. The same decoded values, the same products in the same
# order: the same bits (mikro/kane/i57/k35f_test). k35f_test (chunk of 4096 rows, card 0): gate|up IQ3_XXS 378.4 -> 355.5 ms, IQ3_S
# 218.3 -> 214.6, IQ4_XS 28.5 -> 21.2, IQ2_S 422.0 -> 451.9 (so IQ2_S stays on mode 4); the decode inside the products instead (in
# the middle, or decoded there and stored after) was 10 % slower: the products are bound by their instructions, not by the decode's
# latency. Each edit must match as counted.
import sys

src = sys.argv[1]


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1)
    open(p, "w").write(s)


V7 = r'''// ---- FNX_PF_JOBV 7 (jobv7_patch.py, idea 57): mode 4's tiles, a lane's run words straight from the expert two steps ahead
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

'''

edit("p100_moe.cu", [(
    "template <int TY>\n"
    "void jobv_launch(const JobArgs& a, int njobs_cap, int mode, cudaStream_t s) {\n"
    "    static_assert(kJobVSmem<TY, 128> <= 48 * 1024, \"p100 moe: shared memory past Pascal's limit\");\n",
    V7 +
    "template <int TY>\n"
    "void jobv_launch(const JobArgs& a, int njobs_cap, int mode, cudaStream_t s) {\n"
    "    static_assert(kJobVSmem<TY, 128> <= 48 * 1024, \"p100 moe: shared memory past Pascal's limit\");\n"
    "    if (mode == 7) {   // jobv7: IQ2_S and where the two buffers do not fit, mode 4\n"
    "        if constexpr (TY != 22 && kJobV7Smem<TY> <= 48 * 1024) {\n"
    "            k_job_v7<TY><<<dim3(unsigned((a.N + k4NB - 1) / k4NB), njobs_cap * ((PF_JOB + 2 * k4TB - 1) / (2 * k4TB))), 256, kJobV7Smem<TY>, s>>>(a);\n"
    "            return;\n"
    "        }\n"
    "        mode = 4;\n"
    "    }\n",
    1,
), (
    "    if (mode == 6) mode = A.n_in <= 640 ? 5 : 4;   // jobv6: down (K 320 a card) on mode 5's tiles, gate|up on mode 4\n",
    "    if (mode == 6) mode = A.n_in <= 640 ? 5 : 4;   // jobv6: down (K 320 a card) on mode 5's tiles, gate|up on mode 4\n"
    "    if (mode == 7) mode = A.n_in <= 640 ? 5 : 7;   // jobv7: down as 6, gate|up on k_job_v7 (IQ2_S on mode 4)\n",
    1,
), (
    "        if (m < 0 || m > 6) throw std::runtime_error(\"FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes), 4 (3, the next step's loads ahead), 5 (tiles of 64 rows, two blocks an SM) or 6 (down as 5, gate|up as 4)\");\n",
    "        if (m < 0 || m > 7) throw std::runtime_error(\"FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes), 4 (3, the next step's loads ahead), 5 (tiles of 64 rows, two blocks an SM), 6 (down as 5, gate|up as 4) or 7 (down as 5, gate|up with two weight buffers and the words from the expert, IQ2_S as 4)\");\n",
    1,
)])
print("jobv7_patch: p100_moe.cu")
