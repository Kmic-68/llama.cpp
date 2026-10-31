#!/usr/bin/env python3
"""calibrate.py --model GGUF --bin DIR [--out DIR] [--profile FILE] [--guard] [--ctx N] [--corpus FILE]

Measures what autoconfig.py needs for ANY MoE GGUF on THIS machine, and writes OUT/calib.json, OUT/hitcurve.json and
OUT/slots-S.txt (default OUT = autoconfig/models/<model>/<host>/):
  1. expert sizes per layer from the GGUF (bytes of ffn_*_exps / n_expert), the number of cards from nvidia-smi;
  2. probe 1 (decode-oriented, -ub 256, uniform slots) with LLAMA_MOE_TRACE: the routing trace, decode speed, VRAM;
  3. the misses-vs-slots curve and per-layer slot budgets from that trace (hitcurve.py);
  4. probes 2-4 (-ub 2048 / prefetch + tail at 2048 and 3072), slot counts chosen from the VRAM fit so far with a margin:
     VRAM (per card: peak minus what was used before the start), warm prefill (2nd of two prompts), decode.
Every server runs with the profile's limits in mind; --guard wraps it in guard.py + vram-floor.so (this repo's safety stack).
"""
import argparse, json, os, re, socket, statistics, struct, subprocess, sys, threading, time, urllib.request

here = os.path.dirname(os.path.abspath(__file__))
# --guard: a directory holding guard.py and tools/vram-floor.so (a VRAM/RAM watchdog; optional, not shipped)
moe = os.environ.get("MOE_GUARD_DIR", os.path.join(os.path.dirname(here), "moe"))
sys.path.insert(0, here)
from autoconfig import lstsq, gpus, mem_available_mib

# ---------------------------------------------------------------- GGUF: expert bytes per layer
BLK = {0: (1, 4), 1: (1, 2), 8: (32, 34), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 10: (256, 84), 11: (256, 110),
       12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 110),
       20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2), 39: (32, 17)}
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

def gguf_experts(first_shard):
    shards = [first_shard]
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first_shard)
    if m:
        shards = [f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]
    per_layer = {}; n_expert = None; ctx_train = None
    for fn in shards:
        f = open(fn, "rb")
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        def val(t):
            if t == 8: return f.read(rd("Q")).decode("utf8", "replace")
            if t == 9:
                et = rd("I"); n = rd("Q")
                if et in SZ: f.seek(SZ[et] * n, 1); return None
                return [val(et) for _ in range(n)]
            return struct.unpack("<" + {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}[t], f.read(SZ[t]))[0]
        f.read(4); rd("I"); nt = rd("Q"); nkv = rd("Q")
        for _ in range(nkv):
            k = f.read(rd("Q")).decode(); v = val(rd("I"))
            if k.endswith(".context_length"): ctx_train = v
        for _ in range(nt):
            name = f.read(rd("Q")).decode(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; t = rd("I"); rd("Q")
            mm = re.match(r"blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.weight", name)
            if not mm or t not in BLK:
                continue
            ne = 1
            for d in dims: ne *= d
            il = int(mm.group(1)); n_expert = dims[2]
            per_layer[il] = per_layer.get(il, 0) + ne // BLK[t][0] * BLK[t][1]
    return {il: b / n_expert / 2**20 for il, b in per_layer.items()}, n_expert, ctx_train

# ---------------------------------------------------------------- server control
class Server:
    def __init__(self, a, tag, S, U, pf, tail, slot_file=None, trace=None):
        self.log = os.path.join(a.out, f"probe-{tag}.log")
        env = dict(os.environ, LLAMA_MOE_LRU_EXCLUSIVE="1", GGML_CUDA_LRU_DMA="1", GGML_CUDA_LRU_ASYNC="1", GGML_CUDA_LRU_CE="1",
                   CUDA_MODULE_LOADING="EAGER", GGML_CUDA_P2P="1", LLAMA_MOE_LRU_PREFETCH=str(pf), LLAMA_MOE_LRU_TAIL=str(tail),
                   GGML_CUDA_LRU_STASH="0", LLAMA_UB_KV_BUDGET=str(a.kv_budget if U > 512 else 0), LD_LIBRARY_PATH=a.bin)
        if slot_file: env["LLAMA_MOE_LRU_SLOTS"] = slot_file
        if trace: env["LLAMA_MOE_TRACE"] = trace
        cmd = [os.path.join(a.bin, "llama-server"), "-m", a.model, "-c", str(a.ctx), "-ngl", "99", "--fit", "off", "-fa", "on",
               "-sm", "tensor", "--no-repack", "--n-cpu-moe", "999", "--moe-lru-slots", str(S), "-ub", str(U), "-b", str(U),
               "-np", "1", "--jinja", "-t", str(os.cpu_count() // 2), "--port", str(a.port), "--host", "127.0.0.1",
               "-ctk", "q4_0", "-ctv", "q4_0", "--temp", "1.0", "--top-k", "20", "--top-p", "0.95", "--min-p", "0.0"]
        if a.guard:
            cmd = [os.path.join(moe, "guard.py"), self.log, "--", "env", "LD_PRELOAD=" + os.path.join(moe, "tools", "vram-floor.so")] + cmd
            self.p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.p = subprocess.Popen(cmd, env=env, stdout=open(self.log, "w"), stderr=subprocess.STDOUT)
        self.base = [u for _, u in gpus()]
        self.peak = list(self.base); self.stop_flag = False
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        while not self.stop_flag:
            for i, (_, u) in enumerate(gpus()):
                self.peak[i] = max(self.peak[i], u)
            time.sleep(0.5)

    def usage(self):   # server's own peak per card
        g = self.log + ".guard"
        if A.guard and os.path.exists(g):
            # guard.py's baseline is taken after its GPU0 canary started, and it samples every 0.1 s
            txt = open(g).read()
            m = re.search(r"baseline GPU0 (\d+) GPU1 (\d+)", txt)
            if m:
                base = [int(m.group(1)), int(m.group(2))]
                peak = list(base)
                for l in txt.splitlines():
                    mm = re.match(r"[0-9.]+ g0 (\d+) g1 (\d+)", l)
                    if mm:
                        peak = [max(peak[0], int(mm.group(1))), max(peak[1], int(mm.group(2)))]
                return max(p - b for p, b in zip(peak, base))
        return max(p - b for p, b in zip(self.peak, self.base))

    def wait_ready(self, timeout=1800):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.p.poll() is not None:
                return False
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{A.port}/health", timeout=2)
                return True
            except Exception:
                time.sleep(3)
        return False

    def stop(self):
        self.stop_flag = True
        pid = self.p.pid
        if A.guard:   # guard.py does not forward signals: stop the server it started (its PID is in the guard log)
            m = re.findall(r"started pid (\d+)", open(self.log + ".guard").read()) if os.path.exists(self.log + ".guard") else []
            pid = int(m[-1]) if m else None
        if pid is not None:
            try:
                os.kill(pid, 15)
                for _ in range(30):
                    os.kill(pid, 0); time.sleep(1)
                os.kill(pid, 9)   # a server stuck in its exit cleanup
            except ProcessLookupError:
                pass
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill(); self.p.wait()
        time.sleep(3)

def chat(content, n, extra=None):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": n, "seed": 7, "temperature": 1.0, "top_k": 20,
            "top_p": 0.95, "min_p": 0.0, "cache_prompt": False}
    body.update(extra or {})
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{A.port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600))
    return r["timings"]

PROMPTS = ["Write a short story about a lighthouse keeper.", "Explain how a refrigerator works.",
           "List ten ideas for a weekend project and explain one.", "Describe the water cycle to a child.",
           "Write a Python function that merges two sorted lists, with comments.", "Summarize the causes of the French Revolution."]

def measure(srv, prefill_chars):
    dec = statistics.median(chat(p, 192)["predicted_per_second"] for p in PROMPTS)
    txt = open(A.corpus, errors="ignore").read()[:prefill_chars] + "\n\nSummarize the text above in one sentence."
    chat(txt, 1)                                   # cold: the first long prompt after a start is slow
    pre = chat(txt, 1)["prompt_per_second"]        # warm
    return dec, pre

def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--bin", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--profile", default=None)
    ap.add_argument("--guard", action="store_true"); ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--corpus", default=os.path.join(os.path.dirname(here), "ppl-orig.txt"))
    ap.add_argument("--port", type=int, default=8099); ap.add_argument("--kv-budget", type=int, default=134217728)
    ap.add_argument("--margin-mib", type=int, default=600, help="VRAM kept free per card when picking probe slot counts")
    A = ap.parse_args()
    sizes, n_expert, ctx_train = gguf_experts(A.model)
    A.ctx = A.ctx or ctx_train or 32768
    cards = len(gpus())
    name = re.sub(r"(-00001-of-\d{5})?\.gguf$", "", os.path.basename(A.model))
    A.out = A.out or os.path.join(here, "models", name, socket.gethostname())
    os.makedirs(A.out, exist_ok=True)
    prof = json.load(open(A.profile)) if A.profile else {}
    layers = sorted(sizes)
    slot_mib = sum(sizes.values()) / cards            # one slot in every layer, per card (-sm tensor splits experts)
    exp_host = sum(sizes.values()) / len(layers)      # mean expert MiB per layer in the host pool
    json.dump({str(k): v for k, v in sizes.items()}, open(os.path.join(A.out, "sizes.json"), "w"))
    print(f"{name}: {len(layers)} MoE layers, {n_expert} experts, {slot_mib:.1f} MiB per slot per card, ctx {A.ctx}, {cards} cards")

    g = gpus()
    limits = prof.get("limit_mib", [t - 300 for t, _ in g])
    others = [max(u, o) for (_, u), o in zip(g, prof.get("assume_other_mib", [0] * cards))]
    room = min(l - o for l, o in zip(limits, others)) - A.margin_mib
    ram = mem_available_mib() - prof.get("ram_headroom_mib", 4096)
    s_ram = max(n_expert // 16, int(n_expert - ram / (exp_host * len(layers))) + 1)   # fewest slots the host pool allows
    vram, dec_pts, pre_pts = [], [], []

    def run(tag, S, U, pf, tail, slot_file=None, trace=None):
        print(f"probe {tag}: slots {S}, -ub {U}, prefetch {pf}, tail {tail}", flush=True)
        srv = Server(A, tag, S, U, pf, tail, slot_file, trace)
        ok = srv.wait_ready()
        try:
            if not ok:
                print(f"  server did not come up (see {srv.log})"); return False
            d, p = measure(srv, 40000 if U >= 1024 else 12000)
        except Exception as e:
            print(f"  failed: {e} (see {srv.log})"); return False
        finally:
            srv.stop()
        use = srv.usage()
        print(f"  decode {d:.2f} t/s, warm prefill {p:.1f} t/s, VRAM {use} MiB per card")
        vram.append({"S": S, "U": U, "prefetch": pf, "tail": tail, "mib": use})
        if trace is None:   # the trace hook syncs on every MoE op: its speeds are not the server's
            dec_pts.append({"S": S, "tps": round(d, 2)}); pre_pts.append({"S": S, "U": U, "prefetch": pf, "tps": round(p, 1)})
        return True

    # probe 1: a modest uniform pool, decode shape, routing trace
    trace = os.path.join(A.out, "trace.bin")
    if os.path.exists(trace):
        os.rename(trace, trace + f".{int(time.time())}")
    S1 = 5 * ((max(s_ram, min(n_expert // 4, int(room * 0.5 / slot_mib))) + 4) // 5)
    if not run("1", S1, 256, 0, 0, trace=trace):
        sys.exit(1)
    # the misses curve and per-layer budgets
    subprocess.run([sys.executable, os.path.join(here, "hitcurve.py"), trace, os.path.join(A.out, "hitcurve.json"),
                    str(5 * ((max(8, n_expert // 16) + 4) // 5)), str(n_expert * 3 // 4), "5"], check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, SIZES=os.path.join(A.out, "sizes.json")))
    hc = sorted(int(s) for s in json.load(open(os.path.join(A.out, "hitcurve.json")))["points"])
    snap = lambda S: max([s for s in hc if s <= S] or [hc[0]])
    base1 = vram[0]["mib"] - slot_mib * S1                       # everything but the slots at -ub 256
    # probes 2-4: the largest S the measured base leaves room for, assuming the -ub / assembly overhead measured so far
    extra = {}
    for tag, U, pf, tail in (("2", 2048, 0, 0), ("3", 2048, 1, 1), ("4", 3072, 1, 1)):
        guess = extra.get((pf, tail), 1500 if pf else 800) * (1.0 if U == 2048 else 1.3)
        S = snap(int((room - base1 - guess) / slot_mib))
        if S < max(s_ram, hc[0]):
            print(f"probe {tag}: no room (would need S >= {s_ram})"); continue
        if run(tag, S, U, pf, tail, slot_file=os.path.join(A.out, f"slots-{S}.txt")):
            extra[(pf, tail)] = vram[-1]["mib"] - slot_mib * S - base1

    # probe 5: the decode shape (-ub 256) at the most slots the measured -ub 256 base allows, with the optimized budgets
    S5 = snap(int((room - base1) / slot_mib))
    run("5", S5, 256, 0, 0, slot_file=os.path.join(A.out, f"slots-{S5}.txt"))

    cal = {"_note": f"calibrate.py {time.strftime('%Y-%m-%d %H:%M')} on {socket.gethostname()}: {name}",
           "model": A.model, "bin": A.bin, "cards": cards, "ctx": A.ctx, "kv_mib_per_card_at_ctx": 0, "slot_mib": round(slot_mib, 2),
           "reserve_mib": 300, "expert_mib_per_layer_host": round(exp_host, 3), "pooled_layers": len(layers), "n_expert": n_expert,
           "kv_budget": A.kv_budget, "hitcurve": "hitcurve.json", "ubatch_choices": [256, 512, 1024, 1536, 2048, 2560, 3072],
           "vram_points": vram, "decode_points": dec_pts, "prefill_points": pre_pts}
    json.dump(cal, open(os.path.join(A.out, "calib.json"), "w"), indent=1)
    print("wrote", os.path.join(A.out, "calib.json"))

if __name__ == "__main__":
    main()
