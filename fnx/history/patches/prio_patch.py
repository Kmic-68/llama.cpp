#!/usr/bin/env python3
# prio_patch.py (2026-11-11): FNX_PRIO=1 in engine.cu (the dev tree): each card's own stream (b->st: the windows, the prompt
# passes) at the card's highest stream priority. FNX_RR_TIMES showed the router's last block starting 23-31 us after its first
# while the second stream's kernels (the shared expert, the next layer's guess) held the SMs; with the priority the waiting blocks of
# the own stream take free SMs first. Scheduling only: the same work and sums. Unset: as before. Saves the original once.
import shutil, sys
from pathlib import Path
p = Path("/srv/build/strata-hkrati/upstream/src/split/src/engine.cu")
b = Path("/srv/build/strata-hkrati/mikro/nvp/engine.cu.pred-prio")
c = p.read_text(encoding="utf-8")
if "FNX_PRIO" in c:
    sys.exit("already patched")
if not b.exists():
    shutil.copy2(p, b)
old = """        ck(cudaStreamCreateWithFlags(&b->st, cudaStreamNonBlocking), "stream");
        own_stream(b->st);"""
assert c.count(old) == 1
new = """        // FNX_PRIO=1 (prio_patch.py): the card's own stream at its highest priority, so the router's and the experts' blocks take
        // free SMs before the second stream's (the shared expert, the next layer's guess); scheduling only
        static const bool prio = env_flag("FNX_PRIO");
        int prio_least = 0, prio_most = 0;
        if (prio) ck(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_most), "stream priorities");
        ck(cudaStreamCreateWithPriority(&b->st, cudaStreamNonBlocking, prio ? prio_most : 0), "stream");
        own_stream(b->st);"""
c = c.replace(old, new, 1)
p.write_text(c, encoding="utf-8")
print("patched")
