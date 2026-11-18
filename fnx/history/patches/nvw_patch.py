#!/usr/bin/env python3
# nvw_patch.py: FNX_NVPROF_WIN=K in engine_common.cu (the dev tree): under `nvprof --profile-from-start off` the trace holds the
# process's decode windows K .. K + FNX_NVPROF_WINS - 1 (default 40; 1-based, graph launches of single requests). Unset: nothing
# changes. Saves the original as mikro/nvp/engine_common.cu.pred-nvw once.
import shutil, sys
from pathlib import Path
src = Path("/srv/build/strata-hkrati/upstream/src/split/src/engine_common.cu")
bak = Path("/srv/build/strata-hkrati/mikro/nvp/engine_common.cu.pred-nvw")
code = src.read_text(encoding="utf-8")
if "FNX_NVPROF_WIN" in code:
    sys.exit("already patched")
if not bak.exists():
    shutil.copy2(src, bak)
inc = "#include <cuda_runtime.h>\n"
assert code.count(inc) == 1
code = code.replace(inc, inc + "#include <cuda_profiler_api.h>\n", 1)
a = """        const auto l0 = std::chrono::steady_clock::now();
        on_cards([&](int c) {
            CardBuf& b = *cb_[c];
            upload(c, b, m_.cards[c]);
            ck(cudaGraphLaunch(b.win_graphs[T].exec, b.st), "graph launch");"""
assert code.count(a) == 1
code = code.replace(a, r"""        // FNX_NVPROF_WIN=K: nvprof's trace (--profile-from-start off) of the process's windows K .. K + FNX_NVPROF_WINS - 1
        static const long nvw0 = env_int("FNX_NVPROF_WIN", 0), nvwn = env_int("FNX_NVPROF_WINS", 40);
        static long nvw_no = 0;
        nvw_on = nvw0 > 0 && ++nvw_no >= nvw0 && nvw_no < nvw0 + nvwn;
        if (nvw_on && nvw_no == nvw0) ck(cudaProfilerStart(), "nvprof start");
        nvw_last = nvw_on && nvw_no == nvw0 + nvwn - 1;
""" + a, 1)
b = """    if (const char* e = window_error(err, false)) throw std::runtime_error(std::string("window: ") + e);
    std::vector<int> out(T);"""
assert code.count(b) == 1
code = code.replace(b, r"""    if (nvw_last) {
        ck(cudaProfilerStop(), "nvprof stop");
        std::fprintf(stderr, "nvprof: windows traced\n");
    }
""" + b, 1)
c = """    h_.ctl[0] = pos_;
    h_.ctl[1] = (int) take_tags(1);
    float* bv = h_.bv;"""
assert code.count(c) == 1
code = code.replace(c, "    bool nvw_on = false, nvw_last = false;   // FNX_NVPROF_WIN\n" + c, 1)
src.write_text(code, encoding="utf-8")
print("patched")
