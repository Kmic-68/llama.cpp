#!/usr/bin/env python3
"""lgrcmp.py REF.lgr OTHER.lgr [...]: fnx_run's measures of OTHER against REF (top-1 agreement, KL(ref || other) over
the union of both top-64s with a missing token at that side's 64th logit, top-10 overlap)."""
import math
import struct
import sys


def load(path):
    with open(path, "rb") as f:
        b = f.read()
    assert b[:4] == b"LGR1", path
    k, np_ = struct.unpack_from("<Ii", b, 4)
    off = 12
    prompt = struct.unpack_from(f"<{np_}i", b, off)
    off += 4 * np_
    (ns,) = struct.unpack_from("<i", b, off)
    off += 4
    steps = []
    for _ in range(ns):
        tok, lse = struct.unpack_from("<if", b, off)
        off += 8
        pairs = struct.unpack_from(f"<{2 * k}i", b, off)
        vals = struct.unpack_from(f"<{2 * k}f", b, off)
        off += 8 * k
        ids = pairs[0::2]
        v = vals[1::2]
        steps.append((tok, lse, ids, v))
    return prompt, steps


def compare(r, o):
    pr, rs = r
    po, os_ = o
    assert pr == po, "different prompts"
    n = min(len(rs), len(os_))
    agree = run = 0
    open_ = True
    kls, ov = [], 0
    for i in range(n):
        ta, la, ia, va = rs[i]
        tb, lb, ib, vb = os_[i]
        assert ta == tb, f"step {i} took different tokens"
        if ia[0] == ib[0]:
            agree += 1
            run += open_
        else:
            open_ = False
        da, db = dict(zip(ia, va)), dict(zip(ib, vb))
        kl = 0.0
        for t in set(ia) | set(ib):
            p = math.exp(da.get(t, va[-1]) - la)
            q = math.exp(db.get(t, vb[-1]) - lb)
            if p > 0 and q > 0:
                kl += p * math.log(p / q)
        kls.append(max(kl, 0.0))
        ov += len(set(ia[:10]) & set(ib[:10]))
    s = sorted(kls)
    return dict(n=n, top1=100.0 * agree / n, run=run, kl_mean=sum(kls) / n, kl_median=s[n // 2],
                kl_p99=s[min(n - 1, int(0.99 * n))], kl_max=s[-1], top10=ov / n)


if __name__ == "__main__":
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        c = compare(ref, load(p))
        print(f"{p}: top-1 {c['top1']:.2f}% (first {c['run']} in a row), KL mean {c['kl_mean']:.3e} median "
              f"{c['kl_median']:.2e} p99 {c['kl_p99']:.2e} max {c['kl_max']:.2e}, top-10 {c['top10']:.2f} (n {c['n']})")
