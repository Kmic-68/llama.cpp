#!/usr/bin/env python3
"""Round 2: do ubatches overlap across the GPUs under -sm layer? (nvprof GPU trace of llama-bench pp16384)
The trace holds the built-in warmup pass and the measured pass; they are split at the longest all-idle gap
and only the measured pass is analysed. Reports, for the measured pass: per-GPU busy share, and the share of
wall time with 0 / 1 / 2 / 3 GPUs busy at once, plus a per-GPU timeline of 'first ubatch start / last end'."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from analyze_i1 import load, devs_of, union

def main(path):
    gpu, _ = load(path)
    work = [z for z in gpu if not z["name"].startswith("[CUDA memcpy HtoD") and not z["name"].startswith("[CUDA memset")]
    allu = union([(z["s"], z["e"]) for z in work])
    # split at the longest gap in the middle 80% of the trace
    t0, t1 = allu[0][0], allu[-1][1]
    gaps = [(allu[k + 1][0] - allu[k][1], k) for k in range(len(allu) - 1)
            if t0 + 0.1 * (t1 - t0) < allu[k][1] < t0 + 0.9 * (t1 - t0)]
    g, k = max(gaps)
    w0, w1 = allu[k + 1][0], t1
    print(f"{path.split('/')[-1]}: split at a {g/1e3:.1f} ms all-idle gap; measured pass {(w1-w0)/1e6:.2f} s")
    devs = sorted({d for z in work for d in devs_of(z) if d is not None})
    per = {}
    for d in devs:
        per[d] = union([(max(z["s"], w0), min(z["e"], w1)) for z in work if d in devs_of(z) and z["e"] > w0 and z["s"] < w1])
        b = sum(e - s for s, e in per[d])
        print(f"  GPU{d}: busy {100*b/(w1-w0):.1f}% of the pass, first activity +{(per[d][0][0]-w0)/1e3:.0f} ms, last activity end +{(per[d][-1][1]-w0)/1e3:.0f} ms")
    ev = []
    for d in devs:
        for s, e in per[d]:
            ev.append((s, 1)); ev.append((e, -1))
    ev.sort()
    cnt = 0; last = w0; acc = [0.0] * (len(devs) + 1)
    for t, x in ev:
        acc[cnt] += t - last; last = t; cnt += x
    acc[cnt] += w1 - last
    W = w1 - w0
    print("  share of the pass with N GPUs busy at once: " + ", ".join(f"{n}: {100*a/W:.1f}%" for n, a in enumerate(acc)))
    busy1 = W - acc[0]
    print(f"  overlap: {100*sum(acc[2:])/busy1:.1f}% of the time any GPU is busy, two or more are busy (pure serial layer pipeline would be ~0%)")

if __name__ == "__main__":
    main(sys.argv[1])
