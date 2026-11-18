#!/usr/bin/env python3
# jobv6_patch.py (2026-11-11): FNX_PF_JOBV=6 in p100_moe.cu (the dev tree): down (K at most 640 a card) through mode 5's tiles of 64
# rows, gate|up through mode 4 (k35e: down 617.8 -> 531.1 ms a chunk of 4096 rows, gate|up 951 -> 957 with mode 5 but 917 with mode 4).
# The same bits as mode 3. Saves the original as mikro/k35/p100_moe.cu.pred-jobv6 once.
import shutil, sys
from pathlib import Path
src = Path("/srv/build/strata-hkrati/upstream/src/split/src/p100_moe.cu")
bak = Path("/srv/build/strata-hkrati/mikro/k35/p100_moe.cu.pred-jobv6")
code = src.read_text(encoding="utf-8")
if "jobv6" in code:
    sys.exit("already patched")
if not bak.exists():
    shutil.copy2(src, bak)
def rep(old, new):
    global code
    assert code.count(old) == 1, old[:80]
    code = code.replace(old, new)
rep("""    a.fit = mode >= 3;   // jobv4: mode 4 is 3 with its loads ahead; jobv5: mode 5 its own kernel""",
    """    if (mode == 6) mode = A.n_in <= 640 ? 5 : 4;   // jobv6: down (K 320 a card) on mode 5's tiles, gate|up on mode 4
    a.fit = mode >= 3;   // jobv4: mode 4 is 3 with its loads ahead; jobv5: mode 5 its own kernel""")
rep("""        if (m < 0 || m > 5) throw std::runtime_error("FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes), 4 (3, the next step's loads ahead) or 5 (tiles of 64 rows, two blocks an SM)");""",
    """        if (m < 0 || m > 6) throw std::runtime_error("FNX_PF_JOBV: 0 (k_gemm4_job), 1 (fitted lanes), 2 (also 256 lanes a job), 3 (2, a short job on 1's lanes), 4 (3, the next step's loads ahead), 5 (tiles of 64 rows, two blocks an SM) or 6 (down as 5, gate|up as 4)");""")
src.write_text(code, encoding="utf-8")
print("patched")
