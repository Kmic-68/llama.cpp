#!/usr/bin/env python3
# jobv4_patch.py (2026-11-11): FNX_PF_JOBV=4 in p100_moe.cu (the dev tree): mode 3's kernel with the next step's activation rows and
# the next group's IQ bytes loaded into registers while the current step computes (their global loads in flight across the
# products instead of before them). The products, their order and the folds are mode 3's: the same bits. Saves the original as
# mikro/k35/p100_moe.cu.pred-jobv4 once.
import shutil, sys
from pathlib import Path
src = Path("/srv/build/strata-hkrati/upstream/src/split/src/p100_moe.cu")
bak = Path("/srv/build/strata-hkrati/mikro/k35/p100_moe.cu.pred-jobv4")
code = src.read_text(encoding="utf-8")
if "jobv4" in code:
    sys.exit("already patched")
if not bak.exists():
    shutil.copy2(src, bak)
def rep(old, new, n=1):
    global code
    assert code.count(old) == n, (code.count(old), old[:80])
    code = code.replace(old, new)

rep("""template <int TY, int NT, int RL, int TL>
__device__ __forceinline__ void jobv_tile(""", """template <int TY, int NT, int RL, int TL, bool PF>
__device__ __forceinline__ void jobv_tile(""")
# the loop: PF's form beside the original
old_loop_head = """    const uint8_t* const we = reinterpret_cast<const uint8_t*>(moe_expert_(a.wa, a.sa, a.ep, __ldg(&jg->w)));
    const int nu = a.K / 8;
    int off = 0;
    for (int u0 = 0; u0 < nu; u0 += KU) {
        const int run0 = u0 >> 2;
        if ((run0 & (kRunsGr<TY> - 1)) == 0) {
            if (drow < N) {"""
rep(old_loop_head, """    const uint8_t* const we = reinterpret_cast<const uint8_t*>(moe_expert_(a.wa, a.sa, a.ep, __ldg(&jg->w)));
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
            if (drow < N) {""")
old_x = """        {
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
            const int wb = rg * RL * KU, xb = tg * TL * KU;"""
rep(old_x, """        if constexpr (PF) {
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
            const int wb = rg * RL * KU, xb = tg * TL * KU;""")
# k_job_v and its dispatch
rep("""template <int TY, int NT>
__launch_bounds__(NT)
__global__ void k_job_v(JobArgs a) {""", """template <int TY, int NT, bool PF>
__launch_bounds__(NT)
__global__ void k_job_v(JobArgs a) {""")
for rl, tl in ((8, 6), (4, 9), (4, 6), (2, 9), (2, 6), (1, 9), (1, 6), (1, 3)):
    rep(f"jobv_tile<TY, NT, {rl}, {tl}>(a, sm, jg, t00, tend);", f"jobv_tile<TY, NT, {rl}, {tl}, PF>(a, sm, jg, t00, tend);")
rep("""        if (mode >= 2) {
            k_job_v<TY, 256><<<dim3(rows, njobs_cap * ((PF_JOB + 2 * k4TB - 1) / (2 * k4TB))), 256, kJobVSmem<TY, 256>, s>>>(a);
            return;
        }
    }
    k_job_v<TY, 128><<<dim3(rows, njobs_cap * ((PF_JOB + k4TB - 1) / k4TB)), 128, kJobVSmem<TY, 128>, s>>>(a);""",
"""        if (mode >= 2) {
            if (mode >= 4) k_job_v<TY, 256, true><<<dim3(rows, njobs_cap * ((PF_JOB + 2 * k4TB - 1) / (2 * k4TB))), 256, kJobVSmem<TY, 256>, s>>>(a);
            else k_job_v<TY, 256, false><<<dim3(rows, njobs_cap * ((PF_JOB + 2 * k4TB - 1) / (2 * k4TB))), 256, kJobVSmem<TY, 256>, s>>>(a);
            return;
        }
    }
    if (mode >= 4) k_job_v<TY, 128, true><<<dim3(rows, njobs_cap * ((PF_JOB + k4TB - 1) / k4TB)), 128, kJobVSmem<TY, 128>, s>>>(a);
    else k_job_v<TY, 128, false><<<dim3(rows, njobs_cap * ((PF_JOB + k4TB - 1) / k4TB)), 128, kJobVSmem<TY, 128>, s>>>(a);""")
rep("""        if (m < 0 || m > 3) throw std::runtime_error("FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job) or 3 (2, a short job on 1's lanes)");""",
    """        if (m < 0 || m > 4) throw std::runtime_error("FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes) or 4 (3, the next step's loads ahead)");""")
rep("""// mode 0: k_gemm4_job; 1, 2, 3: FNX_PF_JOBV's kernel (the same bits)""", """// mode 0: k_gemm4_job; 1, 2, 3, 4: FNX_PF_JOBV's kernel (the same bits)""")
rep("""    a.fit = mode >= 3;""", """    a.fit = mode >= 3;   // jobv4: mode 4 is 3 with its loads ahead""")
src.write_text(code, encoding="utf-8")
print("patched")
