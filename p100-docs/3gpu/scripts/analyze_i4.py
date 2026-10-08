#!/usr/bin/env python3
"""Round 3: per-kernel matvec table. Shapes/types/names from GGML_CUDA_OP_PROFILE=2 (graphs off, event-timed
per node, 64 tokens, top 45 keys per device); graphs-on kernel time per quant type from the nvprof trace (last 32 of 36
tokens). Per-shape times are scaled per quant type so their sum matches the graphs-on nvprof matvec time.
Usage: analyze_i4.py <opprof.err> <nvprof.csv> [ceiling_GBps]"""
import re, sys, statistics
from collections import defaultdict
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from analyze_i1 import load, cat
BPW = {"q8_0": 8.5, "q6_K": 6.5625, "q5_K": 5.5, "q4_K": 4.5, "f16": 16, "f32": 32}
TYPENUM = {8: "q8_0", 14: "q6_K", 13: "q5_K", 12: "q4_K"}
NTOK = 64
def opprof(path):
    dev = None; rows = defaultdict(lambda: [0.0, 0]); other = defaultdict(float); ndev = 0
    for l in open(path, errors="replace"):
        m = re.match(r"op profile, device (\d+)", l)
        if m: dev = int(m.group(1)); ndev += 1; continue
        m = re.match(r"\s+([0-9.]+) ms\s+[0-9.]+%\s+x(\d+)\s+(.*)$", l)
        if not m or dev is None: continue
        ms, n, key = float(m.group(1)), int(m.group(2)), m.group(3).strip()
        mm = re.match(r"(fused:)?MUL_MAT (\S+) (\d+)x(\d+) n=(\d+)(?: \[(.*)\])?", key)
        if mm:
            k = (mm.group(2), int(mm.group(3)), int(mm.group(4)), (mm.group(6) or "").rstrip("-") or "(unnamed)", bool(mm.group(1)))
            rows[k][0] += ms; rows[k][1] += n
        else:
            other[key.split(" [")[0].split(" n=")[0]] += ms
    return rows, other, ndev
def main(op, trace, ceil=605.0):
    rows, other, ndev = opprof(op)
    gpu, _ = load(trace)
    marks = [z for z in gpu if z["name"].startswith("[CUDA memcpy DtoH") and z["dev"] == 0]
    marks = marks[-33:]; w0, w1 = marks[0]["e"], marks[-1]["e"]; T = 32
    nv = defaultdict(float); nvn = defaultdict(int); allmv = 0.0
    for z in gpu:
        if z["s"] < w0 or z["e"] > w1 or cat(z) != "matvec": continue
        m = re.search(r"ggml_type=\(ggml_type\)(\d+)|ggml_type=(\d+)", z["name"])
        t = TYPENUM.get(int(m.group(1) or m.group(2)), "other") if m else ("f16path" if "f16" in z["name"] else "other")
        nv[t] += z["d"]; nvn[t] += 1; allmv += z["d"]
    print(f"nvprof graphs-on, last {T} tokens: matvec kernel time per token, mean per GPU: {allmv/T/ndev/1e3:.2f} ms; by type: "
          + ", ".join(f"{t} {v/T/ndev/1e3:.2f} ms ({nvn[t]/T/ndev:.0f} kernels)" for t, v in sorted(nv.items(), key=lambda kv: -kv[1])))
    opsum = defaultdict(float)
    for (t, K, N, name, fused), (ms, n) in rows.items(): opsum[t] += ms
    print("op profile (graphs off), listed matvec keys, per token mean per GPU: " + ", ".join(f"{t} {v/NTOK/ndev:.2f} ms" for t, v in sorted(opsum.items(), key=lambda kv: -kv[1]))
          + f"; total {sum(opsum.values())/NTOK/ndev:.2f} ms")
    scale = {t: (nv[t] / T / 1e3) / (opsum[t] / NTOK) if opsum.get(t) and nv.get(t) else 1.0 for t in opsum}
    # a type whose listed keys cover well under the nvprof total is truncated (top-45 list), not slower: do not scale it
    scale = {t: (v if 0.9 <= v <= 1.1 else 1.0) for t, v in scale.items()}
    print("scale op-profile -> graphs-on nvprof, per type: " + ", ".join(f"{t} x{s:.3f}" for t, s in scale.items()))
    tot = sum(nv.values()) / T / ndev / 1e3
    out = []
    for (t, K, N, name, fused), (ms, n) in rows.items():
        per_call_us = ms * 1e3 / n * scale.get(t, 1.0)
        byts = K * N * BPW.get(t, 8) / 8
        gbps = byts / (per_call_us * 1e-6) / 1e9
        calls_tok = n / NTOK / ndev
        ms_tok = per_call_us * calls_tok / 1e3
        ideal = byts / (ceil * 1e9) * 1e3 * calls_tok
        out.append((ms_tok - ideal, t, K, N, name, calls_tok, byts / 2**20, per_call_us, gbps, ms_tok, 100 * ms_tok / tot))
    out.sort(reverse=True)
    print(f"\n| # | quant | K x N (per-GPU slice) | tensor (op name) | calls/token/GPU | MiB/call | us/call | GB/s | % of {ceil:.0f} | ms/token | share of matvec | ms lost to ceiling |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, (lost, t, K, N, name, c, mib, us, g, ms, sh) in enumerate(out, 1):
        print(f"| {i} | {t} | {K} x {N} | {name} | {c:.1f} | {mib:.2f} | {us:.1f} | {g:.0f} | {100*g/ceil:.0f}% | {ms:.3f} | {sh:.1f}% | {lost:.3f} |")
    print(f"\nlisted shapes: {sum(o[9] for o in out):.2f} of {tot:.2f} ms/token matvec ({100*sum(o[9] for o in out)/tot:.0f}%); lost to the ceiling in listed shapes: {sum(o[0] for o in out):.2f} ms/token")
    agg = defaultdict(lambda: [0.0, 0.0])
    for lost, t, K, N, name, c, mib, us, g, ms, sh in out:
        agg[(t, name)][0] += lost; agg[(t, name)][1] += ms
    print("\nby tensor class (quant, op name): ms lost / ms per token")
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"  {k[0]} {k[1]}: {v[0]:.3f} / {v[1]:.3f}")
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 605.0)
