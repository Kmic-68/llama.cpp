#!/usr/bin/env python3
# stage_patch.py (2026-11-11): FNX_PF_STAGE_ALL=1 in prefill.cu (the dev tree): a prompt pass's stage holds the largest layer's
# experts as if none were in a slot (FNX_PF_STAGE_MIB still caps it). The stage was sized by the slots at load, but while reading
# the prompt path's lent slots (FNX_PF_LEND) hold no experts either, so 16 of 47 layers missed more than fit and their kernels read
# the rest in place from pinned RAM over PCIe, at about twice the time (the trace of 2026-11-11, mikro/nvp). Unset: as before.
# Saves the original as mikro/nvp/prefill.cu.pred-stage once.
import shutil, sys
from pathlib import Path
src = Path("/srv/build/strata-hkrati/upstream/src/split/src/prefill.cu")
bak = Path("/srv/build/strata-hkrati/mikro/nvp/prefill.cu.pred-stage")
code = src.read_text(encoding="utf-8")
if "FNX_PF_STAGE_ALL" in code:
    sys.exit("already patched")
if not bak.exists():
    shutil.copy2(src, bak)
old = """    size_t most = 0;
    for (int l = 0; l < NL; ++l) {
        size_t n = 0;
        for (int e = 0; e < NEXP; ++e) n += m_.resident[(size_t) l][(size_t) e] ? 0 : 1;"""
assert code.count(old) == 1
code = code.replace(old, """    // FNX_PF_STAGE_ALL=1 (stage_patch.py): every expert counted, as the lent slots' are missing too while reading
    static const bool all = env_flag("FNX_PF_STAGE_ALL");
    size_t most = 0;
    for (int l = 0; l < NL; ++l) {
        size_t n = 0;
        if (all && l == NL - 1) continue;   // the last layer runs no experts while reading
        for (int e = 0; e < NEXP; ++e) n += !all && m_.resident[(size_t) l][(size_t) e] ? 0 : 1;""", 1)
src.write_text(code, encoding="utf-8")
print("patched")
