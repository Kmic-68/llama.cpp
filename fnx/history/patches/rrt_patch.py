#!/usr/bin/env python3
# rrt_patch.py (2026-11-11): FNX_RR_TIMES=1 (the dev tree, a measurement): k_router_route stamps the GPU's globaltimer at its first
# block's start, at the tail's start (the last router block), after the top 10, after the jobs, after the cache plan and at the end;
# every 400 windows card 0's mean intervals over the launches since the last report go to stderr ("rr times"). Unset: the stamps
# are skipped (a flag load a block). Saves the originals in mikro/nvp/*.pred-rrt once.
import shutil, sys
from pathlib import Path
S = Path("/srv/build/strata-hkrati/upstream/src/split/src")
B = Path("/srv/build/strata-hkrati/mikro/nvp")
def patch(name, pairs):
    p = S / name
    c = p.read_text(encoding="utf-8")
    if "rr_tt" in c or "rr_times_dump" in c:
        sys.exit(name + ": already patched")
    b = B / (name + ".pred-rrt")
    if not b.exists():
        shutil.copy2(p, b)
    for old, new in pairs:
        assert c.count(old) == 1, (name, old[:70])
        c = c.replace(old, new, 1)
    p.write_text(c, encoding="utf-8")
patch("moe.cuh", [
    ("""template <int T>
__global__ void __launch_bounds__(RR_THREADS) k_router_route(""",
     """// FNX_RR_TIMES (rrt_patch.py): the launch's stamps [first start, tail, top 10, jobs, plan, end] in a ring of 4096 launches
__device__ unsigned long long rr_tt[4096][6];
__device__ unsigned rr_tn;
__device__ int rr_on;
__device__ __forceinline__ unsigned long long rr_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <int T>
__global__ void __launch_bounds__(RR_THREADS) k_router_route("""),
    ("""    const int nrr = (int) gridDim.x - su.blocks;              // the router's own blocks; the shared expert's after them
""", """    const int nrr = (int) gridDim.x - su.blocks;              // the router's own blocks; the shared expert's after them
    const bool rr_t = rr_on != 0;
    if (rr_t && threadIdx.x == 0) atomicMin(&rr_tt[rr_tn & 4095u][0], rr_now());
"""),
    ("""    if (!last) return;
""", """    if (!last) return;
    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][1] = rr_now();
"""),
    ("""    __syncthreads();
    // jobs: one per distinct expert, numbered in the order of each expert's first pick""", """    __syncthreads();
    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][2] = rr_now();
    // jobs: one per distinct expert, numbered in the order of each expert's first pick"""),
    ("""    if (!ec.n_slots) return;
""", """    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][3] = rr_now();
    if (!ec.n_slots) return;
"""),
    ("""    if (warp == 0) cache_plan(gu, nj, layer, ec, list);
""", """    if (warp == 0) cache_plan(gu, nj, layer, ec, list);
    if (rr_t && e == 0) rr_tt[rr_tn & 4095u][4] = rr_now();
"""),
    ("""    if (e < nj && k) {
        const int g = layer * NEXP + gu[e].w;
        for (int i = 1; i <= k; ++i)
            if (list[i] == g) {
                gu[e].w = ~gu[e].w;
                dn[e].w = ~dn[e].w;
                break;
            }
    }
}""", """    if (e < nj && k) {
        const int g = layer * NEXP + gu[e].w;
        for (int i = 1; i <= k; ++i)
            if (list[i] == g) {
                gu[e].w = ~gu[e].w;
                dn[e].w = ~dn[e].w;
                break;
            }
    }
    if (rr_t) {
        __syncthreads();
        if (e == 0) {
            const unsigned n = rr_tn & 4095u;
            rr_tt[n][5] = rr_now();
            rr_tt[(n + 1) & 4095u][0] = ~0ull;
            rr_tn = rr_tn + 1;
        }
    }
}"""),
])
patch("kernels.cu", [
    ("""#undef FNX_RR
    ck_launch("router_route");
}
""", """#undef FNX_RR
    ck_launch("router_route");
}

// FNX_RR_TIMES (rrt_patch.py): the first call turns the stamps on; each later one prints the mean intervals (us) of the launches since
// the last on card dev (the caller's current card afterwards: dev)
void rr_times_dump(int dev) {
    static std::vector<unsigned> last(16, 0u);
    static std::vector<int> on(16, 0);
    cudaSetDevice(dev);
    if (!on[(size_t) dev]) {
        const int one = 1;
        std::vector<unsigned long long> init((size_t) 4096 * 6, ~0ull);
        cudaMemcpyToSymbol(rr_tt, init.data(), init.size() * 8);
        cudaMemcpyToSymbol(rr_on, &one, sizeof one);
        cudaMemcpyFromSymbol(&last[(size_t) dev], rr_tn, sizeof(unsigned));
        on[(size_t) dev] = 1;
        return;
    }
    unsigned n = 0;
    std::vector<unsigned long long> t((size_t) 4096 * 6);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&n, rr_tn, sizeof n);
    cudaMemcpyFromSymbol(t.data(), rr_tt, t.size() * 8);
    double s[5] = {}, all = 0;
    long c = 0;
    for (unsigned i = last[(size_t) dev]; i != n && c < 4000; ++i) {
        const unsigned long long* r = &t[(size_t) (i & 4095u) * 6];
        if (r[0] == ~0ull || r[5] == ~0ull || r[5] < r[0]) continue;
        bool ok = true;
        for (int k = 1; k < 6; ++k) ok = ok && r[k] != ~0ull && r[k] >= r[k - 1];
        if (!ok) continue;
        for (int k = 0; k < 5; ++k) s[k] += (double) (r[k + 1] - r[k]) / 1000.0;
        all += (double) (r[5] - r[0]) / 1000.0;
        ++c;
    }
    last[(size_t) dev] = n;
    if (c)
        std::fprintf(stderr, "rr times (card %d, %ld launches, us): to the tail %.1f, top 10 %.1f, jobs %.1f, plan %.1f, publish %.1f; all %.1f\\n", dev, c,
                     s[0] / c, s[1] / c, s[2] / c, s[3] / c, s[4] / c, all / c);
}
"""),
])
patch("kernels.hpp", [
    ("""void top_tokens(float* logits, int n, int T, int v0,""", """void rr_times_dump(int dev);   // FNX_RR_TIMES (rrt_patch.py)
void top_tokens(float* logits, int n, int T, int v0,"""),
])
patch("engine_common.cu", [
    ("""    if (const char* e = window_error(err, false)) throw std::runtime_error(std::string("window: ") + e);
    std::vector<int> out(T);""", """    static const bool rr_times = env_flag("FNX_RR_TIMES");   // rrt_patch.py
    if (rr_times) {
        static long rr_w = 0;
        if (rr_w++ % 400 == 0) rr_times_dump(cb_[0]->dev);
    }
    if (const char* e = window_error(err, false)) throw std::runtime_error(std::string("window: ") + e);
    std::vector<int> out(T);"""),
])
print("patched")
