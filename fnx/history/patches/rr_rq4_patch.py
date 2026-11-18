#!/usr/bin/env python3
# rr_rq4_patch.py SRC_DIR: idea 60, second form (2026-11-12). The router's warps take 2 rows each below 14 rows of window (rr_rq), so
# its NEXP + 1 rows need 65 blocks of 256 on 56 SMs and the last block starts 23-31 us after the first (FNX_RR_TIMES); split71 found
# no gain in launching the shared expert's branch after it (FNX_RR_ORDER). FNX_RR_RQ4=1: 4 rows a warp at every window size, 33
# blocks in one wave. Every logit's sum takes the same steps in the same order (moe.cuh's note on rr_rq): the same bits. Each edit must
# match as counted.
import sys

src = sys.argv[1]


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1)
    open(p, "w").write(s)


edit("moe.cuh", [(
    "template <int T>\n"
    "__global__ void __launch_bounds__(RR_THREADS) k_router_route(",
    "template <int T, int RQ = rr_rq<T>()>   // RQ: FNX_RR_RQ4 (rr_rq4_patch.py) takes 4 at every T\n"
    "__global__ void __launch_bounds__(RR_THREADS) k_router_route(",
    1,
), (
    "    constexpr int RR_RQ = rr_rq<T>(), RR_ROWS = rr_rows<T>();\n",
    "    constexpr int RR_RQ = RQ, RR_ROWS = RR_THREADS / 32 / RR_WPR * RQ;\n",
    1,
)])
edit("kernels.cu", [(
    "#define FNX_RR(t) k_router_route<t><<<(NEXP + 1 + rr_rows<t>() - 1) / rr_rows<t>() + su.blocks, RR_THREADS, 0, s>>>(W, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su, tail, rr_fast)\n",
    "    static const bool rq4 = [] {   // rr_rq4_patch.py\n"
    "        const bool q = env_int(\"FNX_RR_RQ4\", 0) != 0;\n"
    "        if (q) std::fprintf(stderr, \"fnx: FNX_RR_RQ4: the router's warps take 4 rows at every window size\\n\");\n"
    "        return q;\n"
    "    }();\n"
    "#define FNX_RR(t) do { if (rq4 && rr_rq<t>() != 4) k_router_route<t, 4><<<(NEXP + 1 + RR_THREADS / 32 / RR_WPR * 4 - 1) / (RR_THREADS / 32 / RR_WPR * 4) + su.blocks, RR_THREADS, 0, s>>>(W, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su, tail, rr_fast); \\\n"
    "    else k_router_route<t><<<(NEXP + 1 + rr_rows<t>() - 1) / rr_rows<t>() + su.blocks, RR_THREADS, 0, s>>>(W, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su, tail, rr_fast); } while (0)\n",
    1,
)])
print("rr_rq4_patch: moe.cuh, kernels.cu")
