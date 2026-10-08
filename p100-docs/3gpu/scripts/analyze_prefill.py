#!/usr/bin/env python3
"""G2 prefill attribution from an nvprof GPU trace of llama-bench -p 2048 -d D (-r 1, no warmup).
Window = first to last exchange copy (depth fill + test, averaged over depth 0..D+2048).
3 GPUs (generic butterfly: 2->0 fold, 0<->1 swap, 0->2 copy-back), per exchange:
  ready_d   = when device d's partial is ready: start of 2->0 (d=2), start of 1->0 (d=1),
              end of dev0's last non-exchange kernel before the fold add (d=0)
  imbalance = max(ready) - mean(ready)   (time the average GPU waits for the slowest one)
  latency   = end of the last copy of the exchange - max(ready)
2 GPUs: per-device busy/idle and exchange op time only.
"""
import sys, re
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from analyze_decode import load, cat, devices_of
from collections import defaultdict

def union(iv):
    iv = sorted(iv); tot = 0.0; cs = ce = None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None: tot += ce - cs
    return tot

def pcat(z):
    n = z["name"]; c = cat(z)
    if c in ("xchg_copy", "xchg_add", "xchg_convert", "xchg_p2p"): return "xchg"
    if "fattn" in n or "flash_attn" in n or "softmax" in n: return "attn"
    if "gemm" in n.lower() or "fold" in n or "sgemm" in n or "hgemm" in n or "cutlass" in n: return "gemm"
    if c.startswith("matvec"): return "matvec"
    if c in ("htod", "dtoh", "other_copy"): return "hostcopy"
    return "other"

def main(path, ubatches):
    ev = load(path)
    cp = [z for z in ev if z["name"] == "[CUDA memcpy PtoP]"]
    if not cp:
        print("no PtoP copies in trace"); return
    w0, w1 = cp[0]["s"], cp[-1]["e"]
    devs = sorted({d for z in ev for d in devices_of(z) if d is not None})
    print(f"== {path.split('/')[-1]}: {len(devs)} GPUs, window {(w1-w0)/1e3:.1f} ms, {ubatches} ubatches -> {(w1-w0)/1e3/ubatches:.1f} ms/ubatch under nvprof; {len(cp)} peer copies")
    for d in devs:
        mine = [z for z in ev if d in devices_of(z) and z["e"] > w0 and z["s"] < w1]
        by = defaultdict(list)
        for z in mine: by[pcat(z)].append((max(z["s"], w0), min(z["e"], w1)))
        busy = union([iv for v in by.values() for iv in v])
        W = w1 - w0
        parts = ", ".join(f"{k} {100*union(v)/W:.1f}%" for k, v in sorted(by.items(), key=lambda kv: -union(kv[1])))
        print(f"  dev{d}: busy {100*busy/W:.1f}%, idle {100*(W-busy)/W:.1f}% | {parts}")
    if len(devs) != 3:
        return
    # group exchanges
    exch = []; cur = None
    for z in cp:
        if z["src"] == 2 and z["dst"] == 0:
            if cur: exch.append(cur)
            cur = {"f20": z, "rest": []}
        elif cur is not None:
            cur["rest"].append(z)
    if cur: exch.append(cur)
    k0 = sorted([z for z in ev if z["dev"] == 0 and pcat(z) not in ("xchg", "hostcopy") and not z["name"].startswith("[")], key=lambda z: z["e"])
    import bisect
    ends0 = [z["e"] for z in k0]
    imb = lat = 0.0; n = 0; first_ready_sum = 0.0
    for x in exch:
        c10 = [z for z in x["rest"] if z["src"] == 1 and z["dst"] == 0]
        if not c10: continue
        r2 = x["f20"]["s"]; r1 = c10[0]["s"]
        i = bisect.bisect_right(ends0, max(r1, r2)) - 1
        r0 = ends0[i] if i >= 0 else r2
        ready = [r0, r1, r2]
        done = max([z["e"] for z in x["rest"]] + [x["f20"]["e"]])
        imb += max(ready) - sum(ready) / 3; lat += done - max(ready); n += 1
    W = w1 - w0
    print(f"  exchanges {n}: imbalance wait {imb/1e3:.1f} ms ({100*imb/W:.1f}% of window), exchange latency after last-ready {lat/1e3:.1f} ms ({100*lat/W:.1f}%), "
          f"per exchange {imb/n:.0f} + {lat/n:.0f} us")
    print(f"  s_pp (imbalance + exchange latency) = {100*(imb+lat)/W:.1f}% of prefill time")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
