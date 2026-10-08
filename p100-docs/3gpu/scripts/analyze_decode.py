#!/usr/bin/env python3
"""Decode attribution from nvprof GPU traces (round 1, G1).

Usage: analyze_decode.py <trace_n4.csv> <trace_n36.csv>
Counts come from differencing n36 - n4 (32 tokens). Timing comes from a steady-state window in the n36
trace: the last 32 tokens, delimited by counting exchange copies (exchanges/token from the differencing).
Per device, inside the window: busy = union of kernels and copies touching the device; categories by
kernel name; idle gaps classified by what ends them (an exchange op -> exchange wait, else other).
"""
import csv, sys, re
from collections import Counter, defaultdict

def load(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("==")]
    r = csv.DictReader(lines); next(r)
    ev = []
    for x in r:
        try:
            s = float(x["Start"]); d = float(x["Duration"])
        except ValueError:
            continue
        name = x["Name"]
        dev = re.search(r"\((\d+)\)", x["Device"] or "")
        dev = int(dev.group(1)) if dev else None
        src = re.search(r"\((\d+)\)", x.get("Src Dev") or ""); dst = re.search(r"\((\d+)\)", x.get("Dst Dev") or "")
        ev.append(dict(s=s, e=s + d, d=d, name=name, dev=dev, stream=x["Stream"],
                       src=int(src.group(1)) if src else None, dst=int(dst.group(1)) if dst else None))
    ev.sort(key=lambda z: z["s"])
    return ev

def cat(z):
    n = z["name"]
    if n == "[CUDA memcpy PtoP]": return "xchg_copy"
    if n.startswith("[CUDA memcpy HtoD"): return "htod"
    if n.startswith("[CUDA memcpy DtoH"): return "dtoh"
    if n.startswith("[CUDA m"): return "other_copy"
    if "k_bin_bcast" in n: return "xchg_add"
    if "k_ar_p2p" in n: return "xchg_p2p"
    if "convert" in n or "k_probe" in n or "k_add_f16" in n: return "xchg_convert"
    m = re.search(r"mul_mat_vec_q<ggml_type=(\d+)", n)
    if m: return "matvec_t%s" % m.group(1)
    if "mmvq_f16" in n: return "matvec_f16"
    if "flash_attn" in n or "fattn" in n: return "attn"
    if "quantize_q8_1" in n: return "quant_act"
    return "other_kernel"

def devices_of(z):
    if z["name"] == "[CUDA memcpy PtoP]":
        return {z["src"], z["dst"]}
    return {z["dev"]}

def counts(ev):
    c = Counter()
    for z in ev:
        for d in devices_of(z):
            c[(d, cat(z))] += 1
    return c

def main(p4, p36):
    e4, e36 = load(p4), load(p36)
    c4, c36 = counts(e4), counts(e36)
    ntok = 32
    devs = sorted({d for (d, _) in c36 if d is not None})
    # exchanges per token: copies landing on device 0 is a stable marker; use all PtoP copies
    ismark = lambda z: z["name"] == "[CUDA memcpy PtoP]" or ("k_ar_p2p" in z["name"] and z["dev"] == 0)
    if not any(z["name"] == "[CUDA memcpy PtoP]" for z in e36):
        print("no PtoP copies: 2-GPU one-kernel P2P AllReduce path; marker = k_ar_p2p kernels on dev0")
    px4 = sum(1 for z in e4 if ismark(z))
    px36 = sum(1 for z in e36 if ismark(z))
    copies_per_tok = (px36 - px4) / ntok
    print(f"exchange markers per token: {copies_per_tok:.1f}")
    print("per-token counts by device and category (n36 - n4)/32:")
    for d in devs:
        row = {k[1]: (c36[k] - c4.get(k, 0)) / ntok for k in c36 if k[0] == d}
        print(f"  dev{d}: " + ", ".join(f"{k}={v:.1f}" for k, v in sorted(row.items()) if abs(v) >= 0.5))
    # steady window = last ntok tokens of n36, delimited by PtoP copy index
    pt = [z for z in e36 if ismark(z)]
    k = int(round(copies_per_tok * ntok))
    w0, w1 = pt[-k]["s"], pt[-1]["e"]
    wall = (w1 - w0) / ntok
    print(f"\nsteady-state window: {ntok} tokens, {wall:.1f} us/token under nvprof")
    res = {}
    for d in devs:
        iv = sorted([(max(z["s"], w0), min(z["e"], w1), cat(z)) for z in e36
                     if d in devices_of(z) and z["e"] > w0 and z["s"] < w1])
        catt = defaultdict(float)
        for s, e, c in iv:
            catt[c] += e - s
        # union and gap classification
        busy = 0.0; gaps_x = 0.0; gaps_o = 0.0; cur_s, cur_e = None, None
        merged = []
        for s, e, c in iv:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            merged.append((cur_s, cur_e))
        busy = sum(e - s for s, e in merged)
        # gap before each merged block: exchange-wait if the block starts with an exchange op
        starts = {}
        for s, e, c in iv:
            starts.setdefault(s, c)
        prev_e = w0
        for s, e in merged:
            g = s - prev_e
            if g > 0:
                first = min((x for x in iv if x[0] == s), key=lambda x: x[0])[2]
                if first.startswith("xchg"):
                    gaps_x += g
                else:
                    gaps_o += g
            prev_e = e
        gaps_o += max(0.0, w1 - prev_e)
        per = lambda v: v / ntok
        xchg_ops = catt["xchg_copy"] + catt["xchg_add"] + catt["xchg_convert"] + catt["xchg_p2p"]
        res[d] = dict(wall=wall, busy=per(busy), xchg_ops=per(xchg_ops), xchg_wait=per(gaps_x), other_idle=per(gaps_o),
                      matvec=per(sum(v for c, v in catt.items() if c.startswith("matvec"))), attn=per(catt["attn"]),
                      htod=per(catt["htod"]), cats={c: per(v) for c, v in catt.items()})
        r = res[d]
        s_dec = (r["xchg_ops"] + r["xchg_wait"]) / wall
        print(f"dev{d}: busy {r['busy']:.0f} us, matvec {r['matvec']:.0f}, xchg ops {r['xchg_ops']:.0f}, "
              f"xchg-adjacent idle {r['xchg_wait']:.0f}, other idle {r['other_idle']:.0f}, htod {r['htod']:.0f} us/token; "
              f"s_dec = {100*s_dec:.1f}%")
        print("   categories us/token: " + ", ".join(f"{c}={v:.0f}" for c, v in sorted(r["cats"].items(), key=lambda kv: -kv[1])))
    sd = [(res[d]["xchg_ops"] + res[d]["xchg_wait"]) / wall for d in devs]
    print(f"\ns_dec mean {100*sum(sd)/len(sd):.1f}%, max {100*max(sd):.1f}% (nvprof-inflated wall {wall:.0f} us/token)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
