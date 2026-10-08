#!/usr/bin/env python3
"""G3: per-type matvec bandwidth at the model's per-GPU shapes, from GGML_CUDA_OP_PROFILE output.
Note: the op profiler syncs once per graph and prints only the top 45 entries per device."""
import re, sys
from collections import defaultdict
BPW = {"q8_0": 34 / 32, "q6_K": 210 / 256, "q5_K": 176 / 256, "q4_K": 144 / 256}
for path in sys.argv[1:]:
    dev = None
    agg = defaultdict(lambda: [0.0, 0, 0.0])  # type -> ms, calls, bytes
    shapes = defaultdict(lambda: [0.0, 0])
    for line in open(path):
        m = re.match(r"op profile, device (\d+)", line)
        if m: dev = int(m.group(1)); continue
        m = re.match(r"\s+([\d.]+) ms\s+[\d.]+%\s+x(\d+)\s+MUL_MAT (\w+) (\d+)x(\d+) n=(\d+)", line)
        if m and dev is not None:
            ms, calls, ty, k, mrows, n = float(m[1]), int(m[2]), m[3], int(m[4]), int(m[5]), int(m[6])
            if ty not in BPW: continue
            by = k * mrows * BPW[ty] * calls
            a = agg[(dev, ty, n)]; a[0] += ms; a[1] += calls; a[2] += by
            s = shapes[(ty, k, mrows, n)]; s[0] += ms; s[1] += calls
    print("==", path.split("/")[-1])
    for (dev, ty, n), (ms, calls, by) in sorted(agg.items()):
        print(f"  dev{dev} {ty:5s} n={n}: {calls:6d} calls, {1000*ms/calls:7.1f} us/call, {by/ms/1e6:6.1f} GB/s effective")
    for (ty, k, mr, n), (ms, calls) in sorted(shapes.items(), key=lambda kv: -kv[1][0])[:8]:
        print(f"    shape {ty} {k}x{mr} n={n}: {1000*ms/calls:.1f} us/call, {k*mr*BPW[ty]/(ms/calls)/1e6:.1f} GB/s")
