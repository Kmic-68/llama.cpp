#!/usr/bin/env python3
# rr_order_patch.py SRC_DIR: idea 60 (2026-11-12). In the FFN half the shared expert's branch (the second stream: its gate|up, swiglu,
# down and the next layer's guess) is launched before the router, so its blocks take the SMs first and the router's last of 65 blocks
# starts 23-31 us after its first (FNX_RR_TIMES); the router is on the critical path (router -> copies -> phase 2), the shared expert
# is not (it joins at the end). FNX_RR_ORDER: 0 as before; 1 the branch still forks before the router (beside it) but is launched
# after it, so the router's blocks are handed out first; 2 the branch forks after the router's end. The same kernels on the same
# data: the same sums. Each edit must match as counted.
import sys

src = sys.argv[1]


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1)
    open(p, "w").write(s)


edit("engine.cu", [(
    "    if (!sh_in) {\n"
    "        ck(cudaEventRecord(b.fork, b.st), \"fork\");\n"
    "        ck(cudaStreamWaitEvent(b.st2, b.fork, 0), \"fork wait\");\n"
    "        q8s_gemv(L.sh_gu.s, 2 * ffs, N, b.xq, b.sh_gu, 2 * ffs, T, b.st2);\n",
    "    // FNX_RR_ORDER (idea 60): 0 the branch launched before the router; 1 forked before it but launched after it (the router's\n"
    "    // blocks get the SMs first); 2 forked after the router's end\n"
    "    static const int rr_order = [] {\n"
    "        const int o = (int) env_int(\"FNX_RR_ORDER\", 0);\n"
    "        if (o < 0 || o > 2) throw std::runtime_error(\"FNX_RR_ORDER: 0, 1 or 2\");\n"
    "        if (o) std::fprintf(stderr, \"fnx: FNX_RR_ORDER %d: the shared expert's branch after the router%s\\n\", o, o == 1 ? \"'s launch\" : \"\");\n"
    "        return o;\n"
    "    }();\n"
    "    if (!sh_in && rr_order < 2) ck(cudaEventRecord(b.fork, b.st), \"fork\");\n"
    "    const auto shared_branch = [&] {\n"
    "        ck(cudaStreamWaitEvent(b.st2, b.fork, 0), \"fork wait\");\n"
    "        q8s_gemv(L.sh_gu.s, 2 * ffs, N, b.xq, b.sh_gu, 2 * ffs, T, b.st2);\n",
    1,
), (
    "        ck(cudaEventRecord(b.join, b.st2), \"join\");\n"
    "    }\n"
    "    const int lyr = (int) (&L - w.L.data());\n",
    "        ck(cudaEventRecord(b.join, b.st2), \"join\");\n"
    "    };\n"
    "    if (!sh_in && rr_order == 0) shared_branch();\n"
    "    const int lyr = (int) (&L - w.L.data());\n",
    1,
), (
    "                 lyr, b.cache_list, sh_in ? &L.sh_gu.s : nullptr, 2 * ffs, b.xq, b.sh_gu, &rt);\n",
    "                 lyr, b.cache_list, sh_in ? &L.sh_gu.s : nullptr, 2 * ffs, b.xq, b.sh_gu, &rt);\n"
    "    if (!sh_in && rr_order == 2) ck(cudaEventRecord(b.fork, b.st), \"fork\");\n"
    "    if (!sh_in && rr_order > 0) shared_branch();\n",
    1,
)])
print("rr_order_patch: engine.cu")
