#!/usr/bin/env python3
# sp2_patch.py SRC_DIR (2026-11-12, idea 58, after sp_patch.py): FNX_PF_SP only for a pass whose second chunk has FNX_PF_SP_MIN rows
# (default 3584) or more. A chunk's gather crosses the link while the next chunk's hc read runs; behind a short second chunk (a read of
# 4,096 + 20 rows) it waits instead: k58/pre*b, 4116 tokens: 793.6 -> 784.3 tok/s with FNX_PF_SP, 6000 (4096 + 1904): +0.7 %. Such a
# pass takes the path as before (exact either way). split74: reads of 4096 + 486 and 4096 + 3085 rows -1.2 and -0.5 %, longer +3.6 to +4.6 %:
# chunk 0's gather hides only behind chunk 1's hc read on half its rows, about as long from some 3,000 rows (3584: the 4096 + 3085 read too stays as before). Each edit must match as counted.
import sys

src = sys.argv[1]
p = src + "/prefill.cu"
s = open(p).read()
a = "        sp_pass = p0.sp && nk >= 2 && p0.f16acc && !draft8_sp && !p0.ring_ce.direct && !p0.rccl;\n"
b = ("        static const int sp_min = (int) env_int(\"FNX_PF_SP_MIN\", 3584);   // sp2_patch.py: a short second chunk hides no gather\n"
     "        sp_pass = p0.sp && nk >= 2 && (int) chunks[1].tokens.size() >= sp_min && p0.f16acc && !draft8_sp && !p0.ring_ce.direct && !p0.rccl;\n")
assert s.count(a) == 1, s.count(a)
open(p, "w").write(s.replace(a, b))
print("sp2_patch: prefill.cu")
