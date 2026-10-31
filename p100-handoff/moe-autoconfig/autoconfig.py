#!/usr/bin/env python3
"""autoconfig.py [--calib FILE] [--profile FILE] [--ratio R | --workload chat|agent|rag] [--ctx N] [--top K]

Picks the expert-cache server configuration (LRU slots per layer, -ub/-b, assembly prefetch, tail assembly) that minimizes the
time per generated token for a workload of R prompt tokens per generated token:

    t = R / prefill_tps + 1 / decode_tps

so a config that loses 1 t/s of decode but gains 100 t/s of prefill wins exactly when its saved prompt time is larger than the
lost decode time at that R. Everything comes from measurements on this machine (the calibration file): VRAM per card is
    base + slot_mib * S + act_mib * U + asm_mib * (1 + prefetch) * (1 - tail * S / 512),
decode time per token is a + b * misses(S) (misses from the model's routing trace, hitcurve.json), prefill time per token is
c + k * (NE - S) / U + d * U (separately with and without prefetch). Each is a least-squares fit of the calibration points; the
VRAM fit adds its largest under-prediction as a margin.

Resources are read live: free VRAM per card (nvidia-smi: what other processes use now) and MemAvailable (the host pool pins
(512 - S) experts per layer). A profile can raise what other processes are assumed to use (a desktop streamer that grows when
a client connects) and lower the per-card limit.
"""
import argparse, json, os, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))

def lstsq(rows, ys):
    # normal equations (small systems)
    n = len(rows[0])
    A = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(A[k][i]))
        A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        for k in range(i + 1, n):
            f = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x

def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [tuple(int(v) for v in l.split(",")) for l in out.strip().splitlines()]

def mem_available_mib():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"):
            return int(l.split()[1]) // 1024
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="calib.json written by calibrate.py (models/<model>/<host>/)")
    ap.add_argument("--profile", default=None, help="JSON: limit_mib [per card], assume_other_mib [per card], ram_headroom_mib")
    ap.add_argument("--ratio", type=float, default=None, help="prompt tokens processed per generated token")
    ap.add_argument("--workload", choices=["chat", "agent", "rag"], default="agent")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="print the chosen config as JSON only")
    a = ap.parse_args()

    cal = json.load(open(a.calib))
    hc = json.load(open(os.path.join(os.path.dirname(a.calib), cal["hitcurve"])))
    misses = {int(s): p["misses_per_token"] for s, p in hc["points"].items()}
    NE = cal.get("n_expert", 512)
    ratio = a.ratio if a.ratio is not None else {"chat": 3.0, "agent": 10.0, "rag": 30.0}[a.workload]
    ctx = a.ctx or cal["ctx"]
    prof = json.load(open(a.profile)) if a.profile else {}

    # VRAM fit (server usage per card)
    rows, ys = [], []
    for p in cal["vram_points"]:
        S, U, pf, tail = p["S"], p["U"], p["prefetch"], p["tail"]
        rows.append([1.0, U, (1 + pf) * (1 - tail * S / NE)]); ys.append(p["mib"] - cal["slot_mib"] * S)
    base, act, asm = lstsq(rows, ys)
    margin = max(0.0, max(y - (base + act * r[1] + asm * r[2]) for r, y in zip(rows, ys)))
    for p in cal.get("vram_lower_bounds", []):   # killed runs: usage was at least this
        S, U, pf, tail = p["S"], p["U"], p["prefetch"], p["tail"]
        pred = base + cal["slot_mib"] * S + act * U + asm * (1 + pf) * (1 - tail * S / NE)
        margin = max(margin, p["mib"] - pred)
    kv_full = cal["kv_mib_per_card_at_ctx"]
    base += kv_full * (ctx / cal["ctx"] - 1.0)

    # decode fit: ms per token = a + b * misses(S)
    da, db = lstsq([[1.0, misses[p["S"]]] for p in cal["decode_points"]], [1000.0 / p["tps"] for p in cal["decode_points"]])
    # prefill fits per prefetch mode: ms per token = c + k*(512-S)/U (+ d*U with enough points)
    pre = {}
    for pf in (0, 1):
        pts = [p for p in cal["prefill_points"] if p["prefetch"] == pf]
        use_d = len(pts) >= 4
        rows = [[1.0, (NE - p["S"]) / p["U"]] + ([p["U"] / 1000.0] if use_d else []) for p in pts]
        x = lstsq(rows, [1000.0 / p["tps"] for p in pts])
        pre[pf] = x + ([0.0] if not use_d else [])

    if subprocess.run(["sh", "-c", "ps -eo comm | grep -qx llama-server"]).returncode == 0:
        print("warning: a llama-server is running; its VRAM counts as used by other processes", file=sys.stderr)
    g = gpus()
    n_cards = cal["cards"]
    limits, others = [], []
    for i in range(n_cards):
        total, used = g[i]
        lim = prof.get("limit_mib", [total - cal.get("reserve_mib", 300)] * n_cards)[i]
        oth = max(used, prof.get("assume_other_mib", [0] * n_cards)[i])
        limits.append(lim); others.append(oth)
    room = min(l - o for l, o in zip(limits, others))
    ram = mem_available_mib() - prof.get("ram_headroom_mib", 4096)

    cands = []
    for U in cal["ubatch_choices"]:
        for pf in (0, 1):
            for tail in sorted({p["tail"] for p in cal["vram_points"] if p["prefetch"] == pf}):
                fixed = base + margin + act * U + asm * (1 + pf)
                # largest S on the hit curve that fits VRAM and RAM
                best = None
                for S in sorted(misses):
                    use = fixed + cal["slot_mib"] * S - asm * (1 + pf) * tail * S / NE
                    host = (NE - S) * cal["expert_mib_per_layer_host"] * cal["pooled_layers"]
                    if use <= room and host <= ram and S <= cal.get("max_slots", 400):
                        best = (S, use, host)
                if best is None:
                    continue
                S, use, host = best
                dec = 1000.0 / (da + db * misses[S])
                c, k, d = pre[pf]
                prf = 1000.0 / (c + k * (NE - S) / U + d * U / 1000.0)
                t = ratio / prf + 1.0 / dec
                cands.append(dict(S=S, U=U, prefetch=pf, tail=tail, vram_mib=round(use), host_mib=round(host),
                                  decode_tps=round(dec, 2), prefill_tps=round(prf, 1), sec_per_gen_token=t))
    if not cands:
        print("no configuration fits (room %d MiB per card, RAM %d MiB)" % (room, ram)); sys.exit(1)
    cands.sort(key=lambda c: c["sec_per_gen_token"])
    best = cands[0]
    best["slot_file"] = os.path.join(os.path.dirname(os.path.abspath(a.calib)), "slots-%d.txt" % best["S"])
    best["env"] = {"LLAMA_MOE_LRU_EXCLUSIVE": "1", "GGML_CUDA_LRU_DMA": "1", "GGML_CUDA_LRU_ASYNC": "1", "GGML_CUDA_LRU_CE": "1",
                   "CUDA_MODULE_LOADING": "EAGER", "LLAMA_MOE_LRU_PREFETCH": str(best["prefetch"]),
                   "LLAMA_MOE_LRU_TAIL": str(best["tail"]), "LLAMA_MOE_LRU_SLOTS": best["slot_file"],
                   "LLAMA_UB_KV_BUDGET": str(cal["kv_budget"]) if best["U"] > 512 else "0", "GGML_CUDA_LRU_STASH": "0"}
    best["args"] = ["-c", str(ctx), "--moe-lru-slots", str(best["S"]), "-ub", str(best["U"]), "-b", str(best["U"])]
    if a.json:
        print(json.dumps(best, indent=1)); return
    print(f"room {room} MiB per card (limit {limits}, other {others}, margin {margin:.0f}), RAM for the host pool {ram} MiB")
    print(f"workload: {ratio:g} prompt tokens per generated token; ctx {ctx}")
    print("| slots | -ub | prefetch | tail | VRAM/card MiB | prefill t/s | decode t/s | ms per generated token |")
    print("|---|---|---|---|---|---|---|---|")
    for c in cands[:a.top]:
        print(f"| {c['S']} | {c['U']} | {c['prefetch']} | {c['tail']} | {c['vram_mib']} | {c['prefill_tps']} | {c['decode_tps']} | "
              f"{1000 * c['sec_per_gen_token']:.1f} |")
    server = os.path.join(cal["bin"], "llama-server") if cal.get("bin") else "llama-server"
    model = ["-m", cal["model"]] if cal.get("model") else []
    print("\nchosen:", " ".join(f"{k}={v}" for k, v in best["env"].items()), server, " ".join(model + best["args"]))

if __name__ == "__main__":
    main()
