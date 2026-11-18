#!/usr/bin/env python3
# nvp_patch.py: FNX_NVPROF_PASS=K in prefill.cu (the dev tree): under `nvprof --profile-from-start off` the trace holds the
# process's K-th prompt pass alone (1-based; the cards synchronized before and after). Unset or 0: nothing changes.
# Saves the original as mikro/nvp/prefill.cu.pred-nvp once.
import shutil, sys
from pathlib import Path
src = Path("/srv/build/strata-hkrati/upstream/src/split/src/prefill.cu")
bak = Path("/srv/build/strata-hkrati/mikro/nvp/prefill.cu.pred-nvp")
code = src.read_text(encoding="utf-8")
if "FNX_NVPROF_PASS" in code:
    sys.exit("already patched")
if not bak.exists():
    shutil.copy2(src, bak)
inc = "#include <cuda_runtime.h>\n"
assert code.count(inc) == 1
code = code.replace(inc, inc + "#include <cuda_profiler_api.h>\n", 1)
head = """    const auto t_pass = std::chrono::steady_clock::now();
"""
assert code.count(head) == 1
code = code.replace(head, head + """    // FNX_NVPROF_PASS=K: nvprof's trace (--profile-from-start off) of the process's K-th prompt pass alone
    static const int nvp_pass = (int) env_int("FNX_NVPROF_PASS", 0);
    static int nvp_no = 0;
    const bool nvp_on = nvp_pass > 0 && ++nvp_no == nvp_pass;
    if (nvp_on) {
        for (CardBuf* b : cb_) ckc(cudaSetDevice(b->dev), "set"), ckc(cudaDeviceSynchronize(), "nvprof sync");
        ckc(cudaProfilerStart(), "nvprof start");
    }
""", 1)
tail = """    static const bool timing = env_flag("FNX_PF_TIMING");
    if (timing) {
        std::fprintf(stderr, "prefill pass at %d"""
assert code.count(tail) == 1
code = code.replace(tail, r"""    if (nvp_on) {
        for (CardBuf* b : cb_) ckc(cudaSetDevice(b->dev), "set"), ckc(cudaDeviceSynchronize(), "nvprof sync");
        ckc(cudaProfilerStop(), "nvprof stop");
        std::fprintf(stderr, "nvprof: pass %d traced (%d rows)\n", nvp_no, rows);
    }
""" + tail, 1)
src.write_text(code, encoding="utf-8")
print("patched")
