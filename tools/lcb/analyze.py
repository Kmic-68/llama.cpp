#!/usr/bin/env python3
"""Score an LCB run: Wilson interval, difficulty breakdown, and what the
sample can and cannot exclude relative to the published 90.3."""
import json, sys, math

rows = json.load(open(sys.argv[1]))
rows = [r for r in rows if "ok" in r]
n = len(rows); k = sum(r["ok"] for r in rows)
p = k / n if n else 0.0

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    ph = k / n; d = 1 + z*z/n
    c = (ph + z*z/(2*n)) / d
    h = z*math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / d
    return (max(0.0, c-h), min(1.0, c+h))

lo, hi = wilson(k, n)
print(f"LiveCodeBench v6 (test6 slice)  pass@1 = {k}/{n} = {100*p:.1f}%")
print(f"  95% Wilson interval: {100*lo:.1f}% .. {100*hi:.1f}%   (width {100*(hi-lo):.1f} pts)")
print(f"  published Qwen3.8-27B LCB v6: 90.3%")
print(f"  -> published value is {'INSIDE' if lo <= 0.903 <= hi else 'OUTSIDE'} this interval")

print("\nby difficulty (test6 skews hard; full v6 is easier, so this is a lower bound):")
for d in ("easy", "medium", "hard"):
    s = [r for r in rows if r.get("difficulty") == d]
    if s:
        kk = sum(r["ok"] for r in s)
        l2, h2 = wilson(kk, len(s))
        print(f"  {d:<6} {kk:>2}/{len(s):<2} = {100*kk/len(s):5.1f}%   [{100*l2:.0f}..{100*h2:.0f}]")

print("\nby platform:")
for pl in ("atcoder", "leetcode"):
    s = [r for r in rows if r.get("platform") == pl]
    if s: print(f"  {pl:<9} {sum(r['ok'] for r in s):>2}/{len(s)}")

tr = [r for r in rows if r.get("finish") == "length"]
print(f"\ntruncated at token cap: {len(tr)}/{n}"
      + (f"  ({', '.join(r['id'] for r in tr[:8])})" if tr else ""))
toks = [r.get("tok", 0) for r in rows]
if toks:
    toks_s = sorted(toks)
    print(f"completion tokens: median {toks_s[len(toks_s)//2]}, max {max(toks)}, total {sum(toks):,}")

fails = [r for r in rows if not r["ok"]]
if fails:
    print("\nfailures:")
    for r in fails:
        print(f"  {r['id']:<16} {r.get('difficulty',''):<6} {r.get('passed','?')}/{r.get('total','?')} tests"
              f"  {r.get('err','')[:60]}")
