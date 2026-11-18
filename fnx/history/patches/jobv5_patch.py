#!/usr/bin/env python3
# jobv5_patch.py (2026-11-11): FNX_PF_JOBV=5 in p100_moe.cu (the dev tree): tiles of 64 rows x 96 tokens (mode 3's are 128 x 96) in
# at most 32 KB of shared memory and 128 registers a lane, so two blocks share an SM and one decodes while the other multiplies
# (mikro/k35 k35d: without the decode gate|up takes 20 % less time, down 15 %). Every accumulator's sums are mode 3's (the same
# rotation by the row's group of 8, a fold every 8 units): the same bits. Lanes for 64 rows: (4,6) past 72 tokens, (2,9) past 48,
# (2,6) past 36, (1,9) past 24, (1,6) past 12, else (1,3). Saves the original as mikro/k35/p100_moe.cu.pred-jobv5 once.
import shutil, sys
from pathlib import Path
src = Path("/srv/build/strata-hkrati/upstream/src/split/src/p100_moe.cu")
bak = Path("/srv/build/strata-hkrati/mikro/k35/p100_moe.cu.pred-jobv5")
code = src.read_text(encoding="utf-8")
if "k_job_v64" in code:
    sys.exit("already patched")
if not bak.exists():
    shutil.copy2(src, bak)
def rep(old, new, n=1):
    global code
    assert code.count(old) == n, (code.count(old), old[:90])
    code = code.replace(old, new)

# jobv_tile: the tile's rows as a parameter (NBX: 128 as before, or 64 with 96 tokens)
rep("""template <int TY, int NT, int RL, int TL, bool PF>
__device__ __forceinline__ void jobv_tile(const JobArgs& a, uint4* const sm, const BigJob* const jg, const int t00, const int tend) {
    constexpr int KU = k4KU, SU = kSU<TY>, NB = k4NB, TBX = NT / NB * k4TB;""",
"""template <int TY, int NT, int RL, int TL, bool PF, int NBX = k4NB>
__device__ __forceinline__ void jobv_tile(const JobArgs& a, uint4* const sm, const BigJob* const jg, const int t00, const int tend) {
    constexpr int KU = k4KU, SU = kSU<TY>, NB = NBX, TBX = NBX == k4NB ? NT / NB * k4TB : 2 * k4TB;   // jobv5: 64 rows x 96 tokens""")
# the decode: with 64 rows a row has four lanes, its two runs a step on the first two
rep("""            } else {
                uint32_t rw[RunWords<TY>::n];
                run_load_s<TY>(gp, run0 + dj, rw);
                uint4 o[4];
                jobv_finish<TY, NT>(tab, t, rw, o);
#pragma unroll
                for (int q = 0; q < 4; ++q) sw[dl * KU + 4 * dj + q] = o[q];
            }""",
"""            } else if (dj < KU / 4) {   // jobv5: 64 rows, a step's two runs on a row's first two of four lanes
                uint32_t rw[RunWords<TY>::n];
                run_load_s<TY>(gp, run0 + dj, rw);
                uint4 o[4];
                jobv_finish<TY, NT>(tab, t, rw, o);
#pragma unroll
                for (int q = 0; q < 4; ++q) sw[dl * KU + 4 * dj + q] = o[q];
            }""")
# the 64-row kernel and its launch
anchor = """template <int TY>
void jobv_launch(const JobArgs& a, int njobs_cap, int mode, cudaStream_t s) {"""
rep(anchor, """// FNX_PF_JOBV 5 (jobv5_patch.py): block (64 rows, a job of at most 96 tokens), two blocks an SM
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

""" + anchor)
rep("""        if (mode >= 2) {
            if (mode >= 4)""", """        if (mode == 5) {
            static_assert(PF_JOB <= 2 * k4TB, "jobv5: a job in one tile of 96");
            k_job_v64<TY, 256><<<dim3(unsigned((a.N + 63) / 64), njobs_cap), 256, kJobV64Smem<TY, 256>, s>>>(a);
            return;
        }
        if (mode >= 2) {
            if (mode >= 4)""")
rep("""        if (m < 0 || m > 4) throw std::runtime_error("FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes) or 4 (3, the next step's loads ahead)");""",
    """        if (m < 0 || m > 5) throw std::runtime_error("FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes), 4 (3, the next step's loads ahead) or 5 (tiles of 64 rows, two blocks an SM)");""")
rep("""    a.fit = mode >= 3;   // jobv4: mode 4 is 3 with its loads ahead""", """    a.fit = mode >= 3;   // jobv4: mode 4 is 3 with its loads ahead; jobv5: mode 5 its own kernel""")
src.write_text(code, encoding="utf-8")
print("patched")
