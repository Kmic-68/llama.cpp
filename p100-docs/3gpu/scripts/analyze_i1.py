#!/usr/bin/env python3
"""Round 2: per-token decode budget from an nvprof GPU+API trace of llama-bench -n 36 (3 GPUs).

Usage: analyze_i1.py <trace.csv> <real_ms_per_token> [<tl_server.err>]

Tokens: each decode token ends with a logits DtoH copy on every device; the device-0 DtoH marks the token
boundary (GPU side), and its cudaMemcpyAsync API call (same Correlation_ID) marks it on the host side.
The last 32 complete tokens are analysed.

GPU side (kernel and copy durations under nvprof are taken as real):
  per device and token: union-busy time; time by category; idle = window - busy, split into
  exchange-adjacent idle (the gap ends at an exchange op) and other idle.
Profiler correction (round 1 method B): the real token time comes from the unprofiled tg512 run; real idle =
  real token time - trace busy; it is split exchange/other in the trace's proportions.
Host side: API calls per token by name, their summed durations (profiled), per device by Correlation_ID
  (launch-type calls), non-launch calls attributed to the device of the next launch. Graph launches/captures.
Imbalance: per token, max - mean over devices of non-exchange GPU busy time.
"""
import csv, re, sys, statistics
from collections import Counter, defaultdict

def load(path):
    lines = open(path, errors="replace").read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('"Start"'))
    rd = csv.reader(lines[i:]); hdr = next(rd); next(rd)
    ix = {h: k for k, h in enumerate(hdr)}
    gpu, api = [], []
    for r in rd:
        if len(r) < len(hdr):
            continue
        try:
            s = float(r[ix["Start"]]); d = float(r[ix["Duration"]])
        except ValueError:
            continue
        name = r[ix["Name"]]; cid = r[ix["Correlation_ID"]]
        dev = r[ix["Device"]]
        if dev == "" and not name.startswith("["):
            api.append(dict(s=s, e=s + d, d=d, name=name, cid=cid)); continue
        m = re.search(r"\((\d+)\)", dev); sd = re.search(r"\((\d+)\)", r[ix["Src Dev"]] or ""); dd = re.search(r"\((\d+)\)", r[ix["Dst Dev"]] or "")
        size = r[ix["Size"]]
        gpu.append(dict(s=s, e=s + d, d=d, name=name, cid=cid, dev=int(m.group(1)) if m else None,
                        src=int(sd.group(1)) if sd else None, dst=int(dd.group(1)) if dd else None,
                        size=float(size) if size else 0.0, stream=r[ix["Stream"]]))
    gpu.sort(key=lambda z: z["s"]); api.sort(key=lambda z: z["s"])
    return gpu, api

def cat(z):
    n = z["name"]
    if n == "[CUDA memcpy PtoP]": return "xchg_copy"
    if "k_bin_bcast" in n and "op_add" in n: return "xchg_add"
    if n.startswith("[CUDA memcpy DtoH"): return "dtoh"
    if n.startswith("[CUDA memcpy HtoD"): return "htod"
    if n.startswith("[CUDA m"): return "other_copy"
    if "mul_mat_vec" in n or "mmvq" in n: return "matvec"
    if "quantize_q8_1" in n: return "quant_act"
    if "flash_attn" in n or "fattn" in n or "rope" in n or "set_rows" in n or "fwht" in n: return "attn_full"
    if "gated_delta_net" in n or "ssm_conv" in n or "concat_non_cont" in n or "softplus" in n: return "deltanet"
    if "rms_norm" in n: return "norm"
    if "get_rows" in n: return "get_rows"
    return "other_kernel"

def devs_of(z):
    if z["name"] == "[CUDA memcpy PtoP]":
        return {z["src"], z["dst"]}
    return {z["dev"]}

def union(iv):
    iv = sorted(iv); tot = 0.0; cs = ce = None; out = []
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None: out.append((cs, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None: out.append((cs, ce))
    return out

LAUNCH = ("cudaLaunchKernel", "cudaMemcpyAsync", "cudaMemcpyPeerAsync", "cudaMemcpy2DAsync", "cudaMemsetAsync", "cudaMemset",
          "cudaGraphLaunch", "cuLaunchKernel", "cudaLaunchCooperativeKernel")

def main(path, real_ms, tl=None):
    gpu, api = load(path)
    devs = sorted({z["dev"] for z in gpu if z["dev"] is not None})
    nd = len(devs)
    dtoh0 = [z for z in gpu if z["name"].startswith("[CUDA memcpy DtoH") and z["dev"] == devs[0]]
    # keep the logits copies: the most common size
    sz = Counter(round(z["size"], 3) for z in dtoh0).most_common(1)[0][0]
    marks = [z for z in dtoh0 if round(z["size"], 3) == sz]
    print(f"trace {path.split('/')[-1]}: {len(gpu)} GPU rows, {len(api)} API rows, {nd} devices, {len(marks)} token marks (logits DtoH {sz} MB on dev{devs[0]})")
    T = min(32, len(marks) - 1)
    marks = marks[-(T + 1):]
    apic = {a["cid"]: a for a in api}
    win = [(marks[k]["e"], marks[k + 1]["e"]) for k in range(T)]
    wall = statistics.mean(b - a for a, b in win)
    print(f"steady window: last {T} tokens, {wall / 1e3:.2f} ms/token under nvprof; real (unprofiled tg512) {real_ms:.2f} ms/token; inflation x{wall / 1e3 / real_ms:.2f}")

    # --- GPU side, per device
    w0, w1 = win[0][0], win[-1][1]
    sel = [z for z in gpu if z["e"] > w0 and z["s"] < w1]
    res = {}
    for d in devs:
        mine = [z for z in sel if d in devs_of(z)]
        ivs = [(max(z["s"], w0), min(z["e"], w1), cat(z)) for z in mine]
        catt = defaultdict(float)
        for s, e, c in ivs: catt[c] += e - s
        merged = union([(s, e) for s, e, _ in ivs])
        busy = sum(e - s for s, e in merged)
        starts = defaultdict(list)
        for s, e, c in ivs: starts[s].append(c)
        gx = go = 0.0; prev = w0
        for s, e in merged:
            g = s - prev
            if g > 0:
                if any(c.startswith("xchg") for c in starts.get(s, [])): gx += g
                else: go += g
            prev = e
        go += max(0.0, w1 - prev)
        comp = busy - sum(e - s for s, e in union([(s, e) for s, e, c in ivs if c.startswith("xchg")]))
        res[d] = dict(busy=busy / T, cats={k: v / T for k, v in catt.items()}, gx=gx / T, go=go / T, comp=comp / T)
    # exchanges per token and bytes
    px = [z for z in sel if z["name"] == "[CUDA memcpy PtoP]"]
    adds = [z for z in sel if cat(z) == "xchg_add"]
    sizes = Counter(round(z["size"] * 1024, 1) for z in px)
    print(f"\nexchanges: {len(px) / T:.1f} peer copies/token, {len(adds) / T:.1f} exchange ADDs/token -> {len(px) / T / 4:.1f} exchanges/token "
          f"(4 copies each, generic 3-GPU butterfly); copy sizes KB: {dict(sizes.most_common(4))}")

    print("\nper device, ms per token (trace kernel/copy time is real; idle corrected to the unprofiled token time):")
    hdr = ["dev", "busy", "matvec", "quant_act", "attn_full", "deltanet", "norm", "other_k", "xchg_ops", "trace_idle", "real_idle", "  xchg_wait", "  other_idle(launch/host)"]
    print(" | ".join(hdr))
    out = {}
    for d in devs:
        r = res[d]; c = r["cats"]
        tidle = r["gx"] + r["go"]
        ridle = max(0.0, real_ms * 1e3 - r["busy"])
        fx = r["gx"] / tidle if tidle > 0 else 0
        xops = c.get("xchg_copy", 0) + c.get("xchg_add", 0)
        other_k = sum(v for k, v in c.items() if k in ("other_kernel", "get_rows", "other_copy", "htod", "dtoh"))
        out[d] = dict(busy=r["busy"], matvec=c.get("matvec", 0), quant=c.get("quant_act", 0), attn=c.get("attn_full", 0), dn=c.get("deltanet", 0),
                      norm=c.get("norm", 0), other=other_k, xops=xops, ridle=ridle, xw=ridle * fx, oi=ridle * (1 - fx), comp=r["comp"])
        o = out[d]
        print(f"GPU{d} | {o['busy']/1e3:.2f} | {o['matvec']/1e3:.2f} | {o['quant']/1e3:.2f} | {o['attn']/1e3:.2f} | {o['dn']/1e3:.2f} | {o['norm']/1e3:.2f} | "
              f"{o['other']/1e3:.2f} | {o['xops']/1e3:.2f} | {tidle/1e3:.2f} | {ridle/1e3:.2f} | {o['xw']/1e3:.2f} | {o['oi']/1e3:.2f}")
    # imbalance of compute
    comps = [out[d]["comp"] for d in devs]
    imb = max(comps) - statistics.mean(comps)
    print(f"\nnon-exchange GPU busy per device (ms/token): {', '.join(f'GPU{d} {out[d]['comp']/1e3:.2f}' for d in devs)}; "
          f"imbalance (max - mean) {imb/1e3:.2f} ms/token = {100*imb/1e3/real_ms:.1f}% of the token")
    mean = lambda k: statistics.mean(out[d][k] for d in devs) / 1e3
    print("\nbudget (mean over GPUs, ms/token and % of the real token):")
    for k, lab in (("matvec", "matvec"), ("quant", "activation quantize (for matvec)"), ("attn", "full attention (FA, rope, KV write, FWHT)"),
                   ("dn", "delta-net (GDN, conv, gating)"), ("norm", "norms"), ("other", "other kernels/copies"), ("xops", "exchange ops (copies+adds)"),
                   ("xw", "exchange-adjacent idle"), ("oi", "other idle (launch gaps / host-bound)")):
        v = mean(k); print(f"  {lab:45s} {v:7.2f} ms  {100*v/real_ms:5.1f}%")

    # --- host side
    m_api = [apic.get(z["cid"]) for z in marks]
    if all(m_api):
        hwin = [(m_api[k]["s"], m_api[k + 1]["s"]) for k in range(T)]
        h0, h1 = hwin[0][0], hwin[-1][1]
        ha = [a for a in api if h0 <= a["s"] < h1]
        cnt = Counter(a["name"] for a in ha); dur = defaultdict(float)
        for a in ha: dur[a["name"]] += a["d"]
        print(f"\nhost API per token (profiled durations): {sum(cnt.values())/T:.0f} calls, {sum(dur.values())/T/1e3:.2f} ms summed")
        for k, v in cnt.most_common(14):
            print(f"  {k:42s} {v/T:7.1f} calls  {dur[k]/T/1e3:7.3f} ms")
        sync = dur.get("cudaStreamSynchronize", 0) / T
        print(f"  enqueue (all but cudaStreamSynchronize/cudaDeviceSynchronize/cudaEventSynchronize): "
              f"{(sum(dur.values()) - dur.get('cudaStreamSynchronize', 0) - dur.get('cudaDeviceSynchronize', 0) - dur.get('cudaEventSynchronize', 0))/T/1e3:.2f} ms")
        glaunch = cnt.get("cudaGraphLaunch", 0) / T
        caps = sum(cnt.get(k, 0) for k in ("cudaStreamBeginCapture", "cudaStreamEndCapture", "cudaGraphInstantiate", "cudaGraphInstantiateWithFlags", "cudaGraphExecUpdate", "cudaGraphExecUpdate_v2")) / T
        # graph launches per device
        gdev = Counter()
        gpucid = defaultdict(set)
        for z in sel:
            for d in devs_of(z):
                if d is not None: gpucid[z["cid"]].add(d)
        for a in ha:
            if a["name"] == "cudaGraphLaunch":
                for d in gpucid.get(a["cid"], {"?"}): gdev[d] += 1
        print(f"\nCUDA graphs: {glaunch:.2f} cudaGraphLaunch/token (per device: {dict((k, round(v/T, 2)) for k, v in gdev.items())}), "
              f"{caps:.2f} capture/instantiate/update calls/token")
        # per-device enqueue attribution
        per = defaultdict(float); pending = []
        for a in ha:
            if a["name"] in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"):
                continue
            pending.append(a)
            if a["name"] in LAUNCH:
                ds = gpucid.get(a["cid"])
                d = min(ds) if ds else "?"
                if a["name"] == "cudaMemcpyPeerAsync" and ds: d = min(ds)
                for p in pending: per[d] += p["d"]
                pending = []
        print("host enqueue per device (profiled ms/token, by Correlation_ID; non-launch calls go to the next launch's device): "
              + ", ".join(f"GPU{k} {v/T/1e3:.2f}" for k, v in sorted(per.items(), key=lambda kv: str(kv[0]))))
        # host span vs GPU: how much of each token the host spends before the last launch
        last_launch = []
        for a0, a1 in hwin:
            ls = [a for a in ha if a0 <= a["s"] < a1 and a["name"] in LAUNCH]
            if ls: last_launch.append(ls[-1]["e"] - a0)
        print(f"host: from token start to its last launch {statistics.mean(last_launch)/1e3:.2f} ms (profiled)")
    if tl:
        ev = []
        for l in open(tl, errors="replace"):
            m = re.match(r"TL (\S+) (\d+)", l)
            if m: ev.append((m.group(1), int(m.group(2))))
        enq, wait, tot = [], [], []
        last_c0 = None
        for k in range(len(ev) - 2):
            if ev[k][0] == "V0" and ev[k + 1][0] == "V1" and ev[k + 2][0] == "V2":
                enq.append(ev[k + 1][1] - ev[k][1]); wait.append(ev[k + 2][1] - ev[k + 1][1])
        cyc = [ev[k + 1][1] - ev[k][1] for k in range(len(ev) - 1) if False]
        v0 = [t for n, t in ev if n == "V0"]
        if enq:
            n = len(enq); sl = slice(max(1, n - 400), n)
            e, w = enq[sl], wait[sl]
            per_tok = [b - a for a, b in zip(v0[:-1], v0[1:])][sl] if len(v0) > 1 else []
            print(f"\nLLAMA_TL (server, unprofiled), last {len(e)} decode steps: llama_decode enqueue {statistics.median(e)/1e3:.2f} ms median, "
                  f"sync wait {statistics.median(w)/1e3:.2f} ms median" + (f", step-to-step {statistics.median(per_tok)/1e3:.2f} ms" if per_tok else ""))

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
