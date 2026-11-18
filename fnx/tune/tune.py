#!/usr/bin/env python3
"""Measures the exact switches on your machine and writes the fastest profile it found: each candidate on a freshly loaded
engine, through Strata's server and a shortened bench/bench.py, against your own baseline.

    python3 tune/tune.py --data /data/flash-next                       # DATA/profile.json as the start, about 2-3 hours
    python3 tune/tune.py --data /data/flash-next --only prefetch,dma --screen
    python3 tune/tune.py --data /data/flash-next --memory-max 58G      # inside a memory cgroup, as a smaller machine would be
    python3 tune/tune.py --data /data/flash-next --dry-run             # the plan, nothing loaded

It first runs tune/check.py's look at the machine and prints its suggestion, then loads the baseline twice (the spread
between the two is this machine's noise), then tries the switches one group at a time. Each test has a threshold: the
larger of --min-gain (2 %; 4 % for fresh writing and writing after the long read, as the cards' clock follows recent load
and short tests differ by 4-5 %) and that test's noise. A change is kept when at least one test is faster by more than
its threshold, the mean of all tests by at least half --min-gain, no test is slower by more than its threshold, and its
first-token logits (FNX_DUMP_FIRST_LOGITS) are the baseline's bit for bit (every switch tried is one whose output effect
is "none" in docs/SWITCHES.md). Of a group's kept candidates the fastest wins, and the next group is measured against it
(one run, so a lucky one raises the bar: the search errs towards fewer changes). At the end the best profile runs once
more; only if that run is also faster than the baseline by the same rule is it written, else the starting profile is.

Groups (--only):
  prefetch  FNX_PREFETCH_WANT (guessed next-layer copies a row) and FNX_PREFETCH_MIN_T, and prefetch off; with --screen
            the WANT values are first compared in one load through FNX_PREFETCH_FILE (edits only, interleaved), and only
            the ones ahead there get a fresh load
  dma       FNX_DMA_SPLIT 0 / 1 / 2
  experts   FNX_EXPERT_RAM_GIB ±1 (how much of the experts stays in RAM only; less = more in VRAM, until the load refuses);
            FNX_EXPERT_DYN_GIB ±2 where the profile sets it, within the RAM there is
  ple       FNX_PLE_RAM 0 / 1 (the 26.8 GiB lookup table in RAM or read from the disk), only when the RAM allows it
  stage     FNX_PF_STAGE_MIB / FNX_PF_TAIL_MIB (the prompt path's staging buffers and lent slots); not in the default set,
            as ours are at the size of the largest layer already and smaller ones were slower on ours
  convcache --conversation-cache-mib, only when the baseline came close to swapping (it changes RAM, not speed)
FNX_ROUTE_PRIOR (lossy: other answers) is tried only with --allow-lossy, and the logits then differ by design.

Safety: it refuses to start beside another engine, with the cards in use or with something answering on --port; before
each load it checks the RAM available against what the profile holds; it stops the whole run when swap use grows by
more than --swap-mib (512) while an arm's tests run, or by more than --swap-load-mib (2048) during its load (a full
machine pages out other programs' idle memory while the engine pins its own; a profile that does not fit keeps
swapping while it works); it stops only its own server and the processes it started
(Strata runs the engine in a process group of its own, so they are tracked by PID), never by name. It writes progress
to WORK/progress.log, which ends with a line "END <result>" (ok, ok-no-gain, stopped-..., error-...); the file WORK/STOP
ends it after the running arm. Run it detached for a long run, e.g.
    setsid nohup python3 tune/tune.py --data DATA > /dev/null 2>&1 < /dev/null &
"""
import argparse
import copy
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RELEASE / "bench"))
import check  # noqa: E402
import bench  # noqa: E402

METRICS = {"fresh": ("fresh",), "edit": ("edit_py", "edit_md"), "long": ("long_read", "long_write"), "medium": ("medium",)}
MIN_GAIN = {"fresh": 4.0, "long_write": 4.0}   # clock-sensitive; the others take --min-gain
GROUPS = ("prefetch", "dma", "experts", "ple", "convcache")
# the only switches it changes: output effect "none" in docs/SWITCHES.md (FNX_ROUTE_PRIOR only with --allow-lossy)
EXACT = {"FNX_PREFETCH", "FNX_PREFETCH_WANT", "FNX_PREFETCH_MIN_T", "FNX_PREFETCH_FILE", "FNX_DMA_SPLIT", "FNX_EXPERT_DYN_GIB",
         "FNX_EXPERT_RAM_GIB", "FNX_PF_STAGE_MIB", "FNX_PF_TAIL_MIB", "FNX_PLE_RAM", "--conversation-cache-mib"}


class Stop(Exception):
    pass


class Tuner:
    def __init__(self, a):
        self.a = a
        self.work = Path(a.work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "dump").mkdir(exist_ok=True)
        (self.work / "logs").mkdir(exist_ok=True)
        self.prog = self.work / "progress.log"
        self.base = json.loads(Path(a.config).read_text())
        self.strata = Path(a.strata).resolve()
        self.tests = [t for t in a.tests.split(",") if t]
        self.srv = None
        self.procs = set()   # (pid, start time) of the server and everything under it, as seen so far
        self.results = []   # every arm: name, changes, metrics, ram, error
        self.noise = {}

    # ------------------------------------------------------------------ log
    def log(self, *x):
        line = time.strftime("%F %T ") + " ".join(str(v) for v in x)
        print(line, flush=True)
        with open(self.prog, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------ configs
    def config(self, name, changes, dump=True):
        c = copy.deepcopy(self.base)
        for k, v in changes.items():
            if k.startswith("--"):
                i = c["args"].index(k)
                c["args"][i + 1] = str(v)
            elif v is None:
                c["env"].pop(k, None)
            else:
                c["env"][k] = str(v)
        c["log"] = str(self.work / "logs" / f"{name}.log")
        c["env"].pop("FNX_DUMP_FIRST_LOGITS", None)
        if dump:
            d = self.work / "dump" / name
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("first.*"):
                f.unlink()
            c["env"]["FNX_DUMP_FIRST_LOGITS"] = str(d / "first")
        path = self.work / f"cfg-{name}.json"
        path.write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8")
        return c, path

    def env_of(self, changes):
        e = dict(self.base["env"])
        e.update({k: v for k, v in changes.items() if not k.startswith("--")})
        return {k: v for k, v in e.items() if v is not None}

    def holds_gib(self, changes):
        """what the engine holds once loaded with these changes (README's numbers), GiB"""
        e = self.env_of(changes)
        q4 = "q4_k_xl" in self.base.get("model_name", "").lower() or "Q4_K_XL" in " ".join(self.base["args"])
        base_e = self.base["env"]
        g = check.HOLDS["q4_k_xl" if q4 else ("iq3_s_64gb" if base_e.get("FNX_PLE_RAM", "1") == "0" else "iq3_s")]   # then the changes
        if e.get("FNX_PLE_RAM", "1") != base_e.get("FNX_PLE_RAM", "1"):
            g += 26.8 if e.get("FNX_PLE_RAM", "1") == "1" else -26.8
        if "FNX_EXPERT_DYN_GIB" in e and "FNX_EXPERT_DYN_GIB" in base_e:
            g += float(e["FNX_EXPERT_DYN_GIB"]) - float(base_e["FNX_EXPERT_DYN_GIB"])
        cc = self.arg("--conversation-cache-mib")
        if "--conversation-cache-mib" in changes and cc:
            g += (float(changes["--conversation-cache-mib"]) - float(cc)) / 1024
        return g

    def arg(self, k):
        a = self.base["args"]
        return a[a.index(k) + 1] if k in a else None

    # ------------------------------------------------------------------ processes
    def track(self):
        """the server and everything under it, by parent PID (Strata starts the engine in a process group of its own, so the
        server's group does not hold it); remembered with each process's start time, so a reused PID is never taken"""
        if not self.srv:
            return self.procs
        stats = {}
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                st = proc_stat(int(p.name))
                if st:
                    stats[int(p.name)] = st
        tree, grew = {self.srv.pid}, True
        while grew:
            add = {pid for pid, st in stats.items() if st[1] in tree and pid not in tree}
            tree |= add
            grew = bool(add)
        self.procs |= {(pid, stats[pid][2]) for pid in tree if pid in stats}
        return self.procs

    def alive(self):
        out = []
        for pid, start in self.procs:
            st = proc_stat(pid)
            if st and st[2] == start and st[0] != "Z":   # a zombie is gone already (the server's is reaped below)
                out.append(pid)
        return out

    def signal_all(self, sig):
        self.track()
        if self.srv:
            try:
                os.killpg(self.srv.pid, sig)
            except ProcessLookupError:
                pass
        for pid in self.alive():
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def stop_server(self):
        if not self.srv:
            return
        self.signal_all(signal.SIGTERM)
        end = time.time() + 120
        while time.time() < end and self.alive():
            self.srv.poll()   # reaps the server
            time.sleep(1)
        left = self.alive()
        if left:
            self.log(f"still running after 120 s, killed: {left}")
            self.signal_all(signal.SIGKILL)
        try:
            self.srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self.srv, self.procs = None, set()
        end = time.time() + 90   # the driver frees the cards' memory a moment after the engine is gone
        while time.time() < end and cards_used() > 1024:
            time.sleep(3)
        if cards_used() > 1024:
            self.log(f"the cards still hold {cards_used():.0f} MiB after the engine stopped")

    def start_server(self, name, cfg_path):
        py = self.strata / ".venv" / "bin" / "python"
        cmd = [str(py), "serve/server.py", "--engine", "strata", "--config", str(cfg_path), "--port", str(self.a.port), "--host", "127.0.0.1"]
        if self.a.memory_max:
            cmd = ["systemd-run", "--user", "--scope", "-q", f"--unit=p100-tune-{name}-{int(time.time())}",
                   "-p", f"MemoryMax={self.a.memory_max}", "-p", "MemorySwapMax=0", "--"] + cmd
        out = open(self.work / f"server-{name}.log", "w")
        self.srv = subprocess.Popen(cmd, cwd=self.strata, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)

    def wait_loaded(self, minutes=25):
        url = f"http://127.0.0.1:{self.a.port}/health"
        end, t0 = time.time() + minutes * 60, time.time()
        while time.time() < end:
            self.track()   # the engine appears under the server during the load
            if self.srv.poll() is not None:
                raise RuntimeError(f"the server ended (code {self.srv.returncode}) before loading")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if json.load(r).get("loaded"):
                        return time.time() - t0
            except Exception:
                pass
            if (self.work / "STOP").exists():
                raise Stop("STOP")
            time.sleep(5)
        raise RuntimeError(f"not loaded in {minutes} minutes")

    # ------------------------------------------------------------------ one arm
    def arm(self, name, changes, screen=None):
        """load, run the tests (or the screen), stop; returns the result dict (metrics, or error)"""
        if (self.work / "STOP").exists():
            raise Stop("STOP")
        bad = [k for k in changes if k not in EXACT and not (self.a.allow_lossy and k == "FNX_ROUTE_PRIOR")]
        if bad:
            raise RuntimeError(f"not an exact switch: {bad}")   # a guard against a plan that grew past docs/SWITCHES.md
        need = self.holds_gib(changes)
        mi = check.meminfo()
        avail = mi["MemAvailable"] / check.GIB
        res = {"name": name, "changes": changes, "need_gib": round(need, 1)}
        self.log(f"=== arm {name}: {fmt(changes) or 'the starting profile'}; needs about {need:.0f} GiB, {avail:.1f} available")
        limit = parse_size(self.a.memory_max) / check.GIB if self.a.memory_max else None
        if avail < need - 5 or (limit and limit < need - 5):
            res["error"] = f"not enough RAM ({avail:.1f} GiB available{f', cgroup {limit:.0f}' if limit else ''}, needs about {need:.0f})"
            self.log(f"arm {name}: skipped, {res['error']}")
            self.results.append(res)
            return res
        _, path = self.config(name, changes, dump=screen is None)
        watch = Watch(mi["SwapTotal"] - mi["SwapFree"], self.a.swap_load_mib, self.a.swap_mib, self)
        t0 = time.time()
        try:
            self.start_server(name, path)
            watch.start()
            res["load_s"] = round(self.wait_loaded())
            watch.loaded()
            self.log(f"arm {name}: loaded in {res['load_s']} s, {check.meminfo()['MemAvailable'] / check.GIB:.1f} GiB available")
            b = bench.Bench(f"http://127.0.0.1:{self.a.port}", "strata", self.strata, 3600)
            if screen:
                res["screen"] = screen(b)
            else:
                raw = {}
                for t in self.tests:
                    if watch.tripped:
                        break
                    raw[t] = getattr(b, t)()
                res["raw"] = raw
                res["metrics"] = metrics(raw)
                self.log(f"arm {name}: {fmt(res['metrics'])}")
        except Stop:
            raise
        except Exception as e:   # a load the engine refuses, a request that fails: logged, the next arm runs
            res["error"] = str(e)[:300]
            self.log(f"arm {name}: FAILED {res['error']}")
            tail = (self.work / "logs" / f"{name}.log")
            if tail.exists():
                for line in tail.read_text(errors="replace").splitlines()[-4:]:
                    self.log(f"  engine: {line[:200]}")
        finally:
            watch.stop()
            self.stop_server()
        res["min_avail_gib"] = round(watch.min_avail, 1)
        res["swap_grew_mib"] = {"load": round(watch.grew_load), "tests": round(watch.grew_tests)}
        res["wall_s"] = round(time.time() - t0)
        if watch.tripped:
            res["error"] = watch.tripped
        self.results.append(res)
        self.save()
        if watch.tripped:
            self.log(f"arm {name}: {res['error']}; stopping the run (a profile that swaps is no candidate)")
            raise Stop("swap")
        return res

    def save(self):
        (self.work / "results.json").write_text(json.dumps({"noise": self.noise, "arms": self.results}, indent=1, default=str) + "\n")

    # ------------------------------------------------------------------ the plan
    def plan(self, groups, suggestion):
        e = self.base["env"]
        out = []   # (group, [changes, ...])
        if "prefetch" in groups and e.get("FNX_PREFETCH") == "1":
            want, mint = int(e.get("FNX_PREFETCH_WANT", 10)), int(e.get("FNX_PREFETCH_MIN_T", 1))
            wants = [w for w in (2, 3, 4, 6, 9) if w != want]
            sug = (suggestion or {}).get("FNX_PREFETCH_WANT")
            if sug and int(sug) in wants:   # check.py's suggestion first
                wants.remove(int(sug))
                wants.insert(0, int(sug))
            out.append(("prefetch", [{"FNX_PREFETCH_WANT": w} for w in wants] + [{"FNX_PREFETCH": 0}]))
            out.append(("prefetch_min_t", [{"FNX_PREFETCH_MIN_T": t} for t in (2, 4) if t != mint]))
        if "dma" in groups and e.get("FNX_P100_DMA") == "1":
            cur = e.get("FNX_DMA_SPLIT", "0")
            out.append(("dma", [{"FNX_DMA_SPLIT": v} for v in ("0", "1", "2") if v != cur]))
        if "experts" in groups:
            r = float(e.get("FNX_EXPERT_RAM_GIB", 3))
            out.append(("expert_ram", [{"FNX_EXPERT_RAM_GIB": f"{r - 1:.1f}"}, {"FNX_EXPERT_RAM_GIB": f"{r + 1:.1f}"}]))
            if "FNX_EXPERT_DYN_GIB" in e:
                d = int(float(e["FNX_EXPERT_DYN_GIB"]))
                out.append(("expert_dyn", [{"FNX_EXPERT_DYN_GIB": d + 2}, {"FNX_EXPERT_DYN_GIB": max(1, d - 2)}]))
        if "ple" in groups:
            out.append(("ple", [{"FNX_PLE_RAM": "0" if e.get("FNX_PLE_RAM", "1") == "1" else "1"}]))
        if "stage" in groups:
            s, t = int(e.get("FNX_PF_STAGE_MIB", 384)), int(e.get("FNX_PF_TAIL_MIB", 2450))
            out.append(("stage", [{"FNX_PF_STAGE_MIB": s // 2, "FNX_PF_TAIL_MIB": t - s}, {"FNX_PF_TAIL_MIB": t + 512}]))
        if self.a.allow_lossy:
            out.append(("lossy", [{"FNX_ROUTE_PRIOR": "0.15"}]))
        return out

    # ------------------------------------------------------------------ screening FNX_PREFETCH_WANT in one load
    def screen_prefetch(self, best_changes, values):
        f = self.work / "prefetch-want.txt"
        cur = int(self.env_of(best_changes).get("FNX_PREFETCH_WANT", 10))
        order = [cur] + [v for v in values if v != cur]
        f.write_text(f"{cur}\n")

        def run(b):
            self.log("screen: warming up (4 fresh prompts)")
            for p in bench.FRESH[:4]:
                b.chat([{"role": "user", "content": p}], 300)
            score = {v: [] for v in order}
            for rnd, seq in enumerate((order, order[::-1])):   # forth and back, so the cache's learning favours none
                for v in seq:
                    f.write_text(f"{v}\n")
                    time.sleep(0.1)
                    e = b.edit()
                    score[v].append((e["py"] + e["md"]) / 2)
                    self.log(f"screen: FNX_PREFETCH_WANT={v} round {rnd + 1}: edits {e['py']} / {e['md']}")
            return {v: round(sum(s) / len(s), 2) for v, s in score.items()}
        res = self.arm("screen", dict(best_changes, FNX_PREFETCH_FILE=str(f)), screen=run)
        sc = res.get("screen")
        if not sc:
            return values
        ahead = sorted((v for v in values if v != cur and sc[v] > sc[cur] * (1 + self.a.min_gain / 100)), key=lambda v: -sc[v])
        self.log(f"screen: edits by value {sc}; ahead of {cur}: {ahead[:2] or 'none'}")
        return ahead[:2]

    # ------------------------------------------------------------------ deciding
    def better(self, new, old):
        """(accepted, reason): faster on the mean of the tests by more than the noise, and no test slower by more than its noise"""
        ups, worst, geo = [], None, []
        for k, v in new.items():
            o = old.get(k)
            if not o or not v:
                continue
            r = v / o
            thr = self.threshold(k)
            geo.append(math.log(r))
            if r < 1 - thr and (worst is None or r < worst[1]):
                worst = (k, r)
            if r > 1 + thr:
                ups.append((k, r))
        if not geo:
            return False, "nothing to compare"
        g = math.exp(sum(geo) / len(geo))
        if worst:
            return False, f"{worst[0]} {100 * (worst[1] - 1):+.1f} %"
        if not ups or g < 1 + self.a.min_gain / 100 / 2:
            return False, f"in the noise (mean {100 * (g - 1):+.1f} %)"
        return True, ", ".join(f"{k} {100 * (r - 1):+.1f} %" for k, r in ups) + f" (mean {100 * (g - 1):+.1f} %)"

    def threshold(self, k):
        return max(MIN_GAIN.get(k, self.a.min_gain), self.noise.get(k, 0)) / 100

    # ------------------------------------------------------------------ the run
    def run(self):
        a = self.a
        groups = [g for g in a.only.split(",") if g] if a.only else list(GROUPS)
        self.log(f"tune.py: {a.config}, tests {','.join(self.tests)}, groups {','.join(groups)}, work {self.work}")
        info = check.inspect(argparse.Namespace(data=a.data, models=None, seconds=3, no_copy=a.no_copy, force_copy=False, no_disk=False),
                             log=lambda *x: self.log(" ", *x))
        (self.work / "check.md").write_text(info["report"] + "\n")
        for line in info["report"].split("### Verdict", 1)[-1].strip().splitlines():
            self.log(line)
        if "FNX_EXPERT_MASK" in self.base["env"]:
            self.log("the starting profile routes through an expert mask (--coder, lossy against the full model): kept as it is; "
                     "every comparison is against the same mask")
        suggestion = info["profile"].get("env")
        plan = self.plan(groups, suggestion)
        self.log("plan: " + "; ".join(f"{g}: " + " | ".join(fmt(c) for c in cs) for g, cs in plan))
        n = 2 + sum(len(cs) for _, cs in plan) + 1
        self.log(f"about {n} loads of 8-12 minutes each: {n * 10 // 60} h {n * 10 % 60} min (fewer with --screen)")
        if a.dry_run:
            return "dry-run"
        busy = cards_used()
        if busy > 1024:
            raise SystemExit(f"the cards are in use ({busy:.0f} MiB): stop the other engine first")
        if check.sh(["pgrep", "-x", "strata-split-2"]).strip():
            raise SystemExit("another engine (strata-split-2) runs: stop it first")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", a.port)) == 0:   # its /health would be taken for our engine's
                raise SystemExit(f"something answers on port {a.port} already: --port another")
        if a.memory_max and subprocess.run(["systemd-run", "--user", "--scope", "-q", "true"], capture_output=True).returncode:
            raise SystemExit("--memory-max needs systemd-run --user (a user session of systemd)")

        # the baseline, twice: the spread is this machine's noise
        b1 = self.arm("base1", {})
        b2 = self.arm("base2", {}) if a.base_runs > 1 else None
        if "metrics" not in b1 or (b2 and "metrics" not in b2):
            return "error-baseline"
        base = dict(b1["metrics"])
        if b2:
            b2["logits"] = self.same_logits("base1", "base2")
            for k in base:
                x, y = b1["metrics"].get(k), b2["metrics"].get(k)
                if x and y:
                    self.noise[k] = round(100 * abs(x - y) / ((x + y) / 2), 2)
                    base[k] = (x + y) / 2
            self.log("noise between the two baselines (%): " + fmt(self.noise))
        best, best_m = {}, base
        for group, cands in plan:
            if group == "prefetch" and a.screen and "FNX_PREFETCH" in self.base["env"]:
                values = [c["FNX_PREFETCH_WANT"] for c in cands if "FNX_PREFETCH_WANT" in c]
                kept = self.screen_prefetch(best, values)
                cands = [c for c in cands if c.get("FNX_PREFETCH_WANT") in kept] + [c for c in cands if "FNX_PREFETCH" in c]
            if group == "convcache":
                continue
            if group == "prefetch_min_t" and self.env_of(best).get("FNX_PREFETCH") == "0":
                continue   # prefetch is off: its window size means nothing
            winner = None
            for i, c in enumerate(cands):
                name = f"{group}{i + 1}"
                r = self.arm(name, dict(best, **c))
                if "metrics" not in r:
                    continue
                ok, why = self.better(r["metrics"], best_m)
                r["logits"] = self.same_logits("base1", name)
                if ok and group != "lossy" and r["logits"] is False:
                    ok, why = False, "its logits differ from the baseline's (not exact)"
                r["verdict"] = ("kept " if ok else "not kept: ") + why
                self.log(f"{name}: {r['verdict']}")
                if ok and (winner is None or geo(r["metrics"], best_m) > geo(winner[1], best_m)):
                    winner = (c, r["metrics"])
            if winner:
                best = dict(best, **winner[0])
                best_m = winner[1]
                self.log(f"{group}: now {fmt(best)}")
        if "convcache" in groups:
            low = min((r.get("min_avail_gib", 99) for r in self.results if "metrics" in r), default=99)
            cc = self.arg("--conversation-cache-mib")
            if low < 4 and cc and int(cc) > 1000:
                best["--conversation-cache-mib"] = int(cc) // 2
                self.log(f"convcache: at the lowest {low} GiB available, so --conversation-cache-mib {int(cc) // 2} (RAM, not speed)")

        # the result, once more: faster than the baseline by the same rule, and the baseline's logits
        final = self.arm("final", best) if best else None
        same = self.same_logits("base1", "final") if final and "metrics" in final else None
        if final:
            final["logits"] = same
        held = False
        if final and "metrics" in final:
            held, why = self.better(final["metrics"], base)
            final["verdict"] = f"{'faster' if held else 'not faster'} than the baseline: {why}"
            self.log(f"final: {final['verdict']}")
        chosen = best if held and (same is not False or "FNX_ROUTE_PRIOR" in best) else {}
        self.write(chosen, best, base, final, same)
        if same is False and "FNX_ROUTE_PRIOR" not in best:
            return "error-logits"
        return "ok" if chosen or not best else "ok-no-gain"

    def same_logits(self, ref, name):
        """True (all bit for bit), False (some differ), None (nothing to compare); logs the count and the largest KL"""
        d = self.work / "dump"
        files = sorted(d.joinpath(ref).glob("first.*"), key=lambda p: int(p.suffix[1:]))
        same = diff = missing = 0
        worst = 0.0
        for f in files:
            g = d / name / f.name
            if not g.exists():
                missing += 1
                continue
            x, y = f.read_bytes(), g.read_bytes()
            if x == y:
                same += 1
                continue
            diff += 1
            worst = max(worst, kl(x, y))
        if not files:
            return None
        self.log(f"logits {name} vs {ref}: {same} of {len(files)} bit for bit, {diff} different (largest KL {worst:.3g}), {missing} missing")
        if diff:
            return False
        return True if same else None

    def write(self, best, tried, base, final, same):
        cfg, _ = self.config("tuned", best, dump=False)
        cfg["log"] = self.base.get("log")
        out = Path(self.a.out)
        out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
        L = ["## tune/tune.py", "", f"Start: `{self.a.config}`; tests: {', '.join(self.tests)}; "
             f"noise between two baselines: {fmt(self.noise) or 'not measured'} (%).", "",
             "| Arm | Changes | " + " | ".join(metric_names(self.tests)) + " | Load s | RAM low GiB | Logits | Verdict |",
             "|---|---|" + "---:|" * len(metric_names(self.tests)) + "---:|---:|---|---|"]
        for r in self.results:
            if "screen" in r:
                L.append(f"| screen | FNX_PREFETCH_FILE | edits by value: {fmt(r['screen'])} |" + " |" * (len(metric_names(self.tests)) + 3))
                continue
            m = r.get("metrics", {})
            L.append(f"| {r['name']} | {fmt(r['changes']) or '-'} | " + " | ".join(f"{m[k]:.1f}" if m.get(k) else "-" for k in metric_names(self.tests))
                     + f" | {r.get('load_s', '-')} | {r.get('min_avail_gib', '-')} | {LOGITS.get(r.get('logits', 'x'), '-')}"
                     + f" | {r.get('error') or r.get('verdict', '')} |")
        L.append("")
        if tried and not best:
            L.append(f"**Best in the search:** {fmt(tried)}; the final run did not confirm it (or its logits differed).")
        L.append(f"**Written:** {fmt(best) or 'the starting profile'}, to `{out}`.")
        if final and "metrics" in final:
            L.append("**Against the baseline:** " + ", ".join(f"{k} {100 * (v / base[k] - 1):+.1f} %" for k, v in final["metrics"].items() if base.get(k)))
        L.append("**First-token logits of the written profile against the baseline:** "
                 + {True: "the same bit for bit", False: "DIFFERENT", None: "not compared (no final run)"}[same])
        arms = [r for r in self.results if "logits" in r]
        differ = [r["name"] for r in arms if r["logits"] is False]
        L.append(f"**Every arm against base1:** {sum(r['logits'] is True for r in arms)} of {len(arms)} the same bit for bit"
                 + (f"; different: {', '.join(differ)}" if differ else "") + ".")
        (self.work / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        for line in L:
            self.log(line)


class Watch(threading.Thread):
    """every 5 s: the lowest RAM available, and swap growth over the arm's start (during the load) or over the level at the
    end of the load (during the tests); past its limit it stops the arm's server"""

    def __init__(self, swap0, limit_load_mib, limit_tests_mib, tuner):
        super().__init__(daemon=True)
        self.swap0, self.limit_load, self.limit_tests, self.t = swap0, limit_load_mib, limit_tests_mib, tuner
        self.swap_loaded = None
        self.min_avail, self.grew_load, self.grew_tests, self.tripped, self.done = 1e9, 0.0, 0.0, "", threading.Event()

    def loaded(self):
        mi = check.meminfo()
        self.swap_loaded = mi["SwapTotal"] - mi["SwapFree"]

    def run(self):
        while not self.done.wait(5):
            mi = check.meminfo()
            used = mi["SwapTotal"] - mi["SwapFree"]
            self.min_avail = min(self.min_avail, mi["MemAvailable"] / check.GIB)
            if self.swap_loaded is None:
                self.grew_load = max(self.grew_load, (used - self.swap0) / check.MIB)
                over = self.grew_load > self.limit_load and f"swap grew by {self.grew_load:.0f} MiB during the load"
            else:
                self.grew_tests = max(self.grew_tests, (used - self.swap_loaded) / check.MIB)
                over = self.grew_tests > self.limit_tests and f"swap grew by {self.grew_tests:.0f} MiB during the tests"
            if over and not self.tripped:
                self.tripped = over
                self.t.log(f"{over}: stopping the arm")
                self.t.signal_all(signal.SIGTERM)

    def stop(self):
        self.done.set()


LOGITS = {True: "same", False: "DIFFERENT", None: "-"}


def metrics(raw):
    m = {}
    if "fresh" in raw:
        m["fresh"] = raw["fresh"].get("tok_s")
    if "edit" in raw:
        m["edit_py"], m["edit_md"] = raw["edit"].get("py"), raw["edit"].get("md")
    if "long" in raw:
        m["long_read"], m["long_write"] = raw["long"].get("read_tok_s"), raw["long"].get("write_tok_s")
    if "medium" in raw:
        v = [x["tok_s"] for x in raw["medium"].values() if x.get("tok_s")]
        m["medium"] = round(sum(v) / len(v), 1) if v else None
    if "image" in raw:
        m["image_write"] = raw["image"].get("write_tok_s")
    return {k: v for k, v in m.items() if v}


def metric_names(tests):
    return [k for t in tests for k in METRICS.get(t, ())]


def geo(new, old):
    rs = [math.log(v / old[k]) for k, v in new.items() if old.get(k) and v]
    return sum(rs) / len(rs) if rs else 0


def kl(x, y):
    a, b = struct.unpack(f"{len(x) // 4}f", x), struct.unpack(f"{len(y) // 4}f", y)
    if len(a) != len(b):
        return float("inf")
    ma, mb = max(a), max(b)
    za, zb = sum(math.exp(v - ma) for v in a), sum(math.exp(v - mb) for v in b)
    la, lb = math.log(za), math.log(zb)
    return sum(math.exp(v - ma) / za * ((v - ma - la) - (w - mb - lb)) for v, w in zip(a, b))


def proc_stat(pid):
    """(state, parent PID, start time) of a process, or None"""
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return f[0], int(f[1]), int(f[19])
    except (OSError, ValueError, IndexError):
        return None


def cards_used():
    out = check.sh(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"])
    return max((float(l.rsplit(",", 1)[1]) for l in out.splitlines() if "P100" in l), default=0.0)


def parse_size(s):
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}.get(s[-1:], 1)
    return float(s[:-1] if s[-1:] in "KMGT" else s) * mult


def fmt(d):
    return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the folder setup.sh filled")
    ap.add_argument("--config", help="the starting profile (default DATA/profile.json)")
    ap.add_argument("--strata", help="Strata v0.2.9 with its .venv (default DATA/strata-v0.2.9)")
    ap.add_argument("--work", help="where configs, logs and results go (default DATA/tune-<date>)")
    ap.add_argument("--out", help="the tuned profile (default DATA/profile.tuned.json)")
    ap.add_argument("--port", type=int, default=18097)
    ap.add_argument("--tests", default="fresh,edit,long", help="bench.py tests per arm (default fresh,edit,long: about 4 minutes)")
    ap.add_argument("--only", help=f"comma list of groups ({', '.join(GROUPS)}, stage); default all but stage")
    ap.add_argument("--screen", action="store_true", help="compare FNX_PREFETCH_WANT values in one load first")
    ap.add_argument("--min-gain", type=float, default=2.0, help="percent a change must win by (fresh writing: at least 4)")
    ap.add_argument("--base-runs", type=int, default=2, choices=(1, 2), help="baseline loads (2: measures the noise)")
    ap.add_argument("--memory-max", help="run every arm in a memory cgroup of this size without swap, e.g. 58G")
    ap.add_argument("--swap-mib", type=float, default=512, help="stop when swap use grows by this much while an arm's tests run")
    ap.add_argument("--swap-load-mib", type=float, default=2048, help="stop when swap use grows by this much during a load")
    ap.add_argument("--allow-lossy", action="store_true", help="also try FNX_ROUTE_PRIOR=0.15 (other answers)")
    ap.add_argument("--no-copy", action="store_true", help="check.py without its copy test")
    ap.add_argument("--dry-run", action="store_true", help="print the check and the plan; load nothing")
    a = ap.parse_args()
    a.config = a.config or str(Path(a.data) / "profile.json")
    a.strata = a.strata or str(Path(a.data) / "strata-v0.2.9")
    a.work = a.work or str(Path(a.data) / time.strftime("tune-%Y%m%d-%H%M"))
    a.out = a.out or str(Path(a.data) / "profile.tuned.json")
    if not Path(a.config).exists():
        sys.exit(f"no {a.config}: run setup.sh first, or give --config")
    if not a.dry_run and not (Path(a.strata) / "serve" / "server.py").exists():
        sys.exit(f"no Strata server in {a.strata} (--strata)")
    if not a.allow_lossy and float(json.loads(Path(a.config).read_text())["env"].get("FNX_ROUTE_PRIOR", "0")) > 0:
        sys.exit("the starting profile has FNX_ROUTE_PRIOR > 0 (lossy): logits would not compare; start from FNX_ROUTE_PRIOR=0")
    t = Tuner(a)
    result = "error"
    def stop(*_):
        raise Stop("signal")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)
    try:
        result = t.run()
    except Stop as e:
        result = f"stopped-{e}"
    except KeyboardInterrupt:
        result = "stopped-ctrl-c"
    except SystemExit as e:
        t.log(str(e))
        result = "error-start"
    except Exception as e:
        t.log(f"error: {e!r}")
        result = "error"
    finally:
        t.stop_server()
        t.save()
        t.log(f"END {result}")


if __name__ == "__main__":
    main()
