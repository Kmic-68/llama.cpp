#!/usr/bin/env python3
"""Looks at a machine with two Tesla P100 before (or instead of) running the engine: the cards and their PCIe paths, RAM,
NUMA, the IOMMU and the disk with the model, measures host-to-card copies, names the bottlenecks and suggests a profile.

    python3 tune/check.py --data /data/flash-next                   # the folder setup.sh fills (its profile.json and models/)
    python3 tune/check.py --models /mnt/models --out report.md      # also writes the report for a GitHub issue
    python3 tune/check.py --data /data/flash-next --write-profile /data/flash-next/profile.check.json

It takes about a minute and never loads the model. It needs Python 3.10, nvidia-smi and the driver's libcuda.so.1 (no
CUDA toolkit). The copy test takes about 1 GiB of each card for a few seconds, so it is skipped on a card that is in use
(more than 1 GiB taken), for example by a running engine; --force-copy runs it anyway. Run it as the user that runs the
engine; nothing here needs root.

What it measures, and what we got on our machine (README, "The machine"):
  host to card, pinned memory, 64 MiB copies: 12.3 GB/s per card (1 MiB copies 10.8), the same from either CPU's memory and
  with both cards at once (the README's 11.9 and, with the IOMMU translating before we set iommu=pt, 7.6-8 GB/s are from an
  earlier test of ours)
  card to host: the cards trade their sums through host memory while reading a prompt
  the PCIe link while copying (an idle P100 drops to Gen1, so the idle link says little)
  the disk: 16 MiB reads in a row (the load reads 48 GB) and random 128 KiB reads (a token's row of the 26.8 GiB lookup
  table, read from the disk by the 64 GB mode and the UD-Q4_K_XL profiles)

Rules marked [measured] rest on a measurement on our machine; [unmeasured] ones are what the design implies (we have one
machine). tune/tune.py measures the switches on yours.
"""
import argparse
import ctypes
import json
import mmap
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
GIB = 1 << 30
MIB = 1 << 20

# the model files (fill_profile.py's MODELS): the first is read at the load, the lookup table (per_layer_token_embd) is the last
MODEL_FILES = {
    "iq3_s": [f"Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{i}-of-00002.gguf" for i in (1, 2)],
    "q4_k_xl": [f"Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf" for i in (1, 2, 3, 4)],
}
# what the engine holds once loaded, GiB (README, "Requirements", "64 GB of RAM", "Unsloth's UD-Q4_K_XL"; tune.py uses them too)
HOLDS = {"iq3_s": 85, "iq3_s_64gb": 46.5, "q4_k_xl": 73}
OURS = {"h2d": 12.3, "h2d_iommu": 7.8, "power_w": 185, "sm_mhz": 1328, "disk_seq": 0.5}
GEN_GTS = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}   # per lane and direction, after line coding


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def rd(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def meminfo():
    out = {}
    for line in rd("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        out[k] = int(v.split()[0]) * 1024 if v.split() else 0
    return out


# ------------------------------------------------------------------ cards and their PCIe paths
GPU_FIELDS = ("index,name,pci.bus_id,compute_cap,memory.total,memory.used,pcie.link.gen.current,pcie.link.gen.max,"
              "pcie.link.width.current,pcie.link.width.max,power.limit,power.default_limit,power.max_limit,clocks.sm,"
              "clocks.max.sm,temperature.gpu,driver_version")


def cards():
    out = []
    for line in sh(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"]).splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) != len(GPU_FIELDS.split(",")):
            continue
        c = dict(zip(GPU_FIELDS.split(","), v))
        bus = c["pci.bus_id"].lower()   # 00000000:37:00.0 -> 0000:37:00.0
        c["bdf"] = bus[-12:] if len(bus) > 12 else bus
        for k in ("memory.total", "memory.used", "power.limit", "power.default_limit", "power.max_limit", "clocks.sm", "clocks.max.sm"):
            try:
                c[k] = float(c[k])
            except ValueError:
                c[k] = None
        c["path"] = pci_path(c["bdf"])
        dev = f"/sys/bus/pci/devices/{c['bdf']}"
        c["numa_node"] = int(rd(f"{dev}/numa_node", "-1") or -1)
        c["local_cpus"] = rd(f"{dev}/local_cpulist")
        c["iommu_type"] = rd(f"{dev}/iommu_group/type") if os.path.exists(f"{dev}/iommu_group") else "none"
        c["bar1_gib"] = bar1_gib(f"{dev}/resource")
        out.append(c)
    return out


def bar1_gib(path):
    """BAR1's size from sysfs (the engine checks it the same way, card_check.hpp); 0: left without an address"""
    lines = rd(path).splitlines()
    if len(lines) < 2:
        return None
    try:
        start, end, _ = (int(x, 16) for x in lines[1].split()[:3])
    except ValueError:
        return None
    return 0 if start == 0 else round((end - start + 1) / GIB, 2)


def link(dev):
    def gen(s):
        return GEN_GTS.get(s.split()[0], None) if s else None
    w = lambda s: int(s) if s.isdigit() else None   # noqa: E731
    return {"bdf": os.path.basename(dev), "gen": gen(rd(f"{dev}/current_link_speed")), "width": w(rd(f"{dev}/current_link_width")),
            "max_gen": gen(rd(f"{dev}/max_link_speed")), "max_width": w(rd(f"{dev}/max_link_width"))}


def pci_path(bdf):
    """the card and every bridge above it up to the root port, each with its link (sysfs)"""
    path, d = [], os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    while re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", os.path.basename(d)):
        path.append(link(d))
        d = os.path.dirname(d)
    return path


def shared_bridge(a, b):
    """the bridges both cards' paths go through (a PCIe switch with one uplink): their BDFs"""
    pa = {x["bdf"] for x in a["path"][1:]}
    return [x["bdf"] for x in b["path"][1:] if x["bdf"] in pa]


def narrowest(c, live=None):
    """the slowest link on the card's path; live: each link as seen while copying, by BDF (else as read idle)"""
    best = None
    for x in c["path"]:
        g, w = x["gen"], x["width"]
        if live and x["bdf"] in live:
            g, w = live[x["bdf"]].get("gen", g), live[x["bdf"]].get("width", w)
        if g is None or w is None:
            continue
        gbs = LANE_GBS.get(g, 0) * w
        if best is None or gbs < best[0]:
            best = (gbs, g, w, x["bdf"])
    return best


# ------------------------------------------------------------------ host-to-card copies through the driver API (libcuda.so.1)
class Cuda:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.call("cuInit", 0)

    def call(self, name, *args):
        r = getattr(self.lib, name)(*args)
        if r != 0:
            s = ctypes.c_char_p()
            self.lib.cuGetErrorString(r, ctypes.byref(s))
            raise RuntimeError(f"{name}: {s.value.decode() if s.value else r}")

    def context(self, bdf):
        dev, ctx = ctypes.c_int(), ctypes.c_void_p()
        self.call("cuDeviceGetByPCIBusId", ctypes.byref(dev), bdf.encode())
        self.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
        return dev, ctx


class Copier:
    """one card: a pinned host buffer allocated on the CPUs it is pinned to, a device buffer, timed synchronous copies"""

    def __init__(self, cu, card, size, cpus):
        self.cu, self.card, self.size = cu, card, size
        self.dev, self.ctx = cu.context(card["bdf"])
        self.host, self.devp = ctypes.c_void_p(), ctypes.c_uint64()
        self.cpus = cpus
        old = os.sched_getaffinity(0)
        try:
            if cpus:
                os.sched_setaffinity(0, cpus)   # the pinned pages come from this thread's NUMA node (first touch below)
            self.cu.call("cuCtxSetCurrent", self.ctx)
            self.cu.call("cuMemHostAlloc", ctypes.byref(self.host), ctypes.c_size_t(size), 0)
            ctypes.memset(self.host, 1, size)
            self.cu.call("cuMemAlloc_v2", ctypes.byref(self.devp), ctypes.c_size_t(size))
        finally:
            os.sched_setaffinity(0, old)

    def run(self, seconds, chunk, to_card=True, result=None):
        old = os.sched_getaffinity(0)
        try:
            if self.cpus:
                os.sched_setaffinity(0, self.cpus)
            return self._run(seconds, chunk, to_card, result)
        finally:
            os.sched_setaffinity(0, old)

    def _run(self, seconds, chunk, to_card, result):
        self.cu.call("cuCtxSetCurrent", self.ctx)
        n_per = self.size // chunk
        f = self.cu.lib.cuMemcpyHtoD_v2 if to_card else self.cu.lib.cuMemcpyDtoH_v2
        moved, t0 = 0, time.perf_counter()
        for _ in range(2):   # warm up
            self._copy(f, to_card, 0, chunk)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for i in range(n_per):
                self._copy(f, to_card, i * chunk, chunk)
            moved += n_per * chunk
        gbs = moved / (time.perf_counter() - t0) / 1e9
        if result is not None:
            result.append(gbs)
        return gbs

    def _copy(self, f, to_card, off, n):
        if to_card:
            r = f(ctypes.c_uint64(self.devp.value + off), ctypes.c_void_p(self.host.value + off), ctypes.c_size_t(n))
        else:
            r = f(ctypes.c_void_p(self.host.value + off), ctypes.c_uint64(self.devp.value + off), ctypes.c_size_t(n))
        if r != 0:
            raise RuntimeError(f"copy failed ({r})")

    def close(self):
        try:
            self.cu.call("cuCtxSetCurrent", self.ctx)
            self.cu.lib.cuMemFree_v2(self.devp)
            self.cu.lib.cuMemFreeHost(self.host)
            self.cu.lib.cuDevicePrimaryCtxRelease_v2(self.dev)
        except Exception:
            pass


def node_cpus(node):
    return parse_cpus(rd(f"/sys/devices/system/node/node{node}/cpulist")) if node >= 0 else set()


def parse_cpus(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part.strip().isdigit():
            out.add(int(part))
    return out


def copy_tests(cs, seconds, log):
    """per card: to the card from its own node's memory, from the other node's, back to the host; then both cards at once,
    with the link sampled while copying"""
    cu = Cuda()
    nodes = sorted({int(p.name[4:]) for p in Path("/sys/devices/system/node").glob("node[0-9]*")})
    res = {}
    copiers = []
    allowed = os.sched_getaffinity(0)
    try:
        for c in cs:
            own = c["numa_node"] if c["numa_node"] >= 0 else (nodes[0] if nodes else -1)
            cpus = node_cpus(own) & allowed
            log(f"  card {c['index']} ({c['bdf']}): copies from node {own}'s memory")
            cp = Copier(cu, c, 256 * MIB, cpus)
            copiers.append(cp)
            r = {"node": own}
            live = {}
            sampler = threading.Thread(target=sample_links, args=(c, seconds, live))
            sampler.start()
            r["h2d"] = round(cp.run(seconds, 64 * MIB), 2)
            sampler.join()
            r["path_live"] = live
            live = live.get(c["bdf"], {})
            r["link_live"] = live
            r["h2d_1mib"] = round(cp.run(seconds / 2, MIB), 2)
            r["d2h"] = round(cp.run(seconds / 2, 64 * MIB, to_card=False), 2)
            other = [n for n in nodes if n != own]
            if other:
                ocpus = node_cpus(other[0]) & allowed
                if ocpus:
                    far = Copier(cu, c, 256 * MIB, ocpus)
                    try:
                        r["h2d_other_node"] = round(far.run(seconds / 2, 64 * MIB), 2)
                        r["other_node"] = other[0]
                    finally:
                        far.close()
            log(f"    to the card {r['h2d']} GB/s (1 MiB copies {r['h2d_1mib']}), back {r['d2h']} GB/s"
                + (f", from node {r['other_node']}'s memory {r['h2d_other_node']}" if "h2d_other_node" in r else "")
                + f"; link while copying: Gen{live.get('gen')} x{live.get('width')}")
            res[c["index"]] = r
        if len(copiers) == 2:
            out = []
            ts = [threading.Thread(target=cp.run, args=(seconds, 64 * MIB, True, out)) for cp in copiers]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            res["both"] = [round(x, 2) for x in out]
            log(f"  both cards at once: {' + '.join(f'{x:.2f}' for x in out)} GB/s")
    finally:
        for cp in copiers:
            cp.close()
    return res


def sample_links(c, seconds, out):
    """every link on the card's path while it copies (an idle link may train down, a bridge's too): the highest generation
    and width each showed after the first 0.3 s, by BDF"""
    devs = [os.path.realpath(f"/sys/bus/pci/devices/{x['bdf']}") for x in c["path"]]
    end = time.time() + seconds
    time.sleep(min(0.3, seconds / 4))
    while time.time() < end - 0.1:
        for d in devs:
            x, o = link(d), out.setdefault(os.path.basename(d), {})
            for k in ("gen", "width"):
                if x[k] and x[k] > o.get(k, 0):
                    o[k] = x[k]
        time.sleep(0.1)


# ------------------------------------------------------------------ the disk with the model
def find_models(a):
    """(folder, model, [paths]) from --models, or from DATA/profile.json's --native, or DATA/models"""
    folders = []
    if a.models:
        folders.append(Path(a.models))
    if a.data:
        cfg = Path(a.data) / "profile.json"
        try:
            c = json.loads(cfg.read_text())
            folders.append(Path(c["args"][c["args"].index("--native") + 1]).parent)
        except (OSError, ValueError, KeyError):
            pass
        folders.append(Path(a.data) / "models")
    for f in folders:
        for model, names in MODEL_FILES.items():
            paths = [f / n for n in names]
            if all(p.exists() for p in paths):
                return f, model, paths
    return (folders[0] if folders else None), None, []


def block_device(path):
    try:
        st = os.stat(path)
    except OSError:
        return {}
    sysdev = os.path.realpath(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
    name = os.path.basename(sysdev)
    q = sysdev + "/queue" if os.path.isdir(sysdev + "/queue") else os.path.dirname(sysdev) + "/queue"
    slaves = sorted(os.listdir(sysdev + "/slaves")) if os.path.isdir(sysdev + "/slaves") else []
    dm = rd(sysdev + "/dm/name")
    return {"device": name + (f" ({dm})" if dm else ""), "under": slaves, "rotational": rd(q + "/rotational") == "1",
            "fs": sh(["findmnt", "-n", "-o", "FSTYPE", "-T", str(path)]).strip()}


def disk_tests(paths, seconds, log):
    """16 MiB reads in a row from the middle of the first file, random 128 KiB reads from the lookup table's file; O_DIRECT,
    so the page cache does not answer (falls back to plain reads where the file system refuses it, and says so)"""
    res = {}
    buf = mmap.mmap(-1, 16 * MIB)   # page-aligned, as O_DIRECT needs

    def opener(p):
        try:
            return os.open(p, os.O_RDONLY | os.O_DIRECT), True
        except OSError:
            return os.open(p, os.O_RDONLY), False
    def pread(state, view, off):
        """state = [fd, direct, path]; a file system that takes O_DIRECT at open but refuses the read: plain reads"""
        try:
            return os.preadv(state[0], [view], off)
        except OSError:
            if not state[1]:
                raise
            os.close(state[0])
            state[0], state[1] = os.open(state[2], os.O_RDONLY), False
            return os.preadv(state[0], [view], off)
    first, table = max(paths[:-1], key=os.path.getsize), paths[-1]   # the largest of the others (UD-Q4_K_XL's first part is 11 MB)
    # in a row: 4 readers taking the next 16 MiB block in turn (the load reads through the page cache with readahead,
    # which keeps several requests in flight; one O_DIRECT reader gave half of that on our RAID)
    size = os.path.getsize(first)
    nxt, lock, moved, direct, t0 = [(size // 4) & ~(16 * MIB - 1)], threading.Lock(), [0], [True], time.perf_counter()

    def reader():
        st, b = [*opener(first), first], mmap.mmap(-1, 16 * MIB)
        try:
            while time.perf_counter() - t0 < seconds:
                with lock:
                    off = nxt[0]
                    nxt[0] += 16 * MIB
                if off + 16 * MIB > size:
                    break
                n = pread(st, b, off)
                with lock:
                    moved[0] += n
        finally:
            direct[0] = direct[0] and st[1]
            os.close(st[0])
    ts = [threading.Thread(target=reader) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    res["seq_gbs"] = round(moved[0] / (time.perf_counter() - t0) / 1e9, 3)
    res["seq_direct"] = direct[0]
    st = [*opener(table), table]
    try:
        size = os.fstat(st[0]).st_size
        rnd = random.Random()   # new places each run: a RAID controller's cache answered a repeated run
        small = memoryview(buf)[:128 * 1024]
        n_ops, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            pread(st, small, rnd.randrange(0, (size - 128 * 1024) // 4096) * 4096)
            n_ops += 1
        dt = time.perf_counter() - t0
        res["rand_ops"] = round(n_ops / dt)
        res["rand_gbs"] = round(n_ops * 128 * 1024 / dt / 1e9, 3)
        res["rand_direct"] = st[1]
    finally:
        os.close(st[0])
    log(f"  in a row (4 readers) {res['seq_gbs']} GB/s, random 128 KiB {res['rand_ops']} reads/s ({res['rand_gbs']} GB/s)"
        + ("" if res["seq_direct"] and res["rand_direct"] else "; without O_DIRECT, the page cache may have answered"))
    return res


# ------------------------------------------------------------------ the host
def host():
    mi = meminfo()
    cpu = {}
    for line in rd("/proc/cpuinfo").splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k == "model name":
            cpu.setdefault("model", v)
        elif k == "physical id":
            cpu.setdefault("sockets", set()).add(v)
        elif k == "vendor_id":
            cpu.setdefault("vendor", v)
        elif k == "cpu family":
            cpu.setdefault("family", int(v))
        elif k == "model" and v.isdigit():
            cpu.setdefault("model_no", int(v))
        elif k == "flags":
            cpu.setdefault("flags", set(v.split()))
    nodes = {}
    for p in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        m = re.search(r"MemTotal:\s+(\d+) kB", rd(p / "meminfo"))
        nodes[int(p.name[4:])] = {"mem_gib": round(int(m.group(1)) / 1048576, 1) if m else None, "cpus": rd(p / "cpulist")}
    cmdline = rd("/proc/cmdline")
    thp = re.search(r"\[(\w+)\]", rd("/sys/kernel/mm/transparent_hugepage/enabled"))
    return {
        "kernel": platform.release(), "os": os_name(), "cpu": cpu.get("model"), "vendor": cpu.get("vendor"),
        "sockets": len(cpu.get("sockets", ())) or None, "threads": os.cpu_count(), "arch": platform.machine(),
        "family": cpu.get("family"), "model_no": cpu.get("model_no"),
        "isa": sorted(f for f in ("avx", "avx2", "f16c", "fma", "avx512f") if f in cpu.get("flags", ())),
        "mem_gib": round(mi.get("MemTotal", 0) / GIB, 1), "avail_gib": round(mi.get("MemAvailable", 0) / GIB, 1),
        "swap_gib": round(mi.get("SwapTotal", 0) / GIB, 1), "swap_used_gib": round((mi.get("SwapTotal", 0) - mi.get("SwapFree", 0)) / GIB, 1),
        "numa": nodes, "cmdline_iommu": [t for t in cmdline.split() if "iommu" in t],
        "iommu_units": len(list(Path("/sys/class/iommu").glob("*"))) if os.path.isdir("/sys/class/iommu") else 0,
        "thp": thp.group(1) if thp else None,
        "hypervisor": "hypervisor" in cpu.get("flags", ()),
        "virt": sh(["systemd-detect-virt"]).strip() or None,
        "governor": rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or None,
        "cpufreq": rd("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver") or None,
        "io_pressure": float((re.search(r"some avg10=([\d.]+)", rd("/proc/pressure/io")) or [0, 0])[1]),
        "cuda_driver": (re.search(r"CUDA Version:\s*([\d.]+)", sh(["nvidia-smi"])) or [None, None])[1],
        "engines": [p for p in sh(["pgrep", "-x", "strata-split-2"]).split()],
    }


def num(x):
    return "?" if x is None else f"{x:.0f}"


def os_name():
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)', rd("/etc/os-release"), re.M)
    return m.group(1) if m else platform.system()


# ------------------------------------------------------------------ verdict
def judge(h, cs, copies, disk, model, data_free):
    """the profile to use and the findings: a list of (level, what, advice, basis); level is stop (does not run), slow
    (costs speed), warn (something breaks or needs a change) or note"""
    found = []

    def add(level, what, advice, basis):
        found.append({"level": level, "what": what, "advice": advice, "basis": basis})
    p100 = [c for c in cs if "P100" in c["name"] and c["compute_cap"] == "6.0" and (c["memory.total"] or 0) >= 16000]
    if len(p100) != 2:
        add("stop", f"{len(p100)} Tesla P100 16 GB found (cards: {', '.join(c['name'] for c in cs) or 'none'})",
            "the engine is built for exactly two P100 16 GB (cc 6.0); on other cards use Strata", "the engine refuses others")
    drv = cs[0]["driver_version"] if cs else ""
    if drv and drv.split(".")[0].isdigit() and int(drv.split(".")[0]) > 580:
        add("note", f"driver {drv}", "NVIDIA's 580 branch is the last one for Pascal; ours is 580.178.04", "")

    # the processor and the board
    if h.get("arch") and h["arch"] != "x86_64":
        add("stop", f"a {h['arch']} processor", "the release is built and tested for x86-64 Linux only", "README, Requirements")
    if h.get("isa") is not None and "avx2" not in h["isa"] and h.get("vendor") in ("GenuineIntel", "AuthenticAMD"):
        add("warn", f"the processor has no AVX2 ({h['cpu']})",
            "the engine and Strata's Python server need none (build.sh sets no -march; nvcc compiles the host code for plain "
            "x86-64), but Strata's picture encoder is built for an AVX2 baseline by default (STRATA_PORTABLE=ON in "
            "tools/vision/CMakeLists.txt) and stops with an illegal instruction here: use setup.sh --no-pictures, or build "
            "strata-vision with -DSTRATA_PORTABLE=OFF (for this processor)", "[unmeasured] on such a processor; read from the build files")
    snb_gen = [((copies or {}).get(c["index"]) or {}).get("link_live", {}).get("gen") for c in cs]
    if h.get("vendor") == "GenuineIntel" and h.get("family") == 6 and h.get("model_no") == 45 and not all(g and g >= 3 for g in snb_gen):
        add("slow", "a Sandy Bridge-EP processor (Xeon E5 v1 / Core i7 39xx): its PCIe 3.0 was never validated, and NVIDIA's "
            "driver runs the cards at Gen2 on it by default (about 6 GB/s instead of 12)",
            "the driver's module option NVreg_EnablePCIeGen3=1 (in /etc/modprobe.d, then rebuild the initramfs and reboot) "
            "turns Gen3 on; check the link while copying with this script. A Xeon E5 v2 (Ivy Bridge-EP) runs Gen3 on the "
            "same boards", "[unmeasured] here; NVIDIA's driver README")
    if h.get("hypervisor"):
        add("slow", f"a virtual machine ({h.get('virt') or 'hypervisor flag set'})",
            "with the cards passed through, their DMA goes through the host's IOMMU, which the guest cannot set to "
            "passthrough: copies run slower (this script measures them). Run on the host, or set iommu=pt there",
            "[unmeasured] in a VM; [measured] translation on ours: 7.6-8 GB/s instead of 11.9")
    if h.get("threads") and h["threads"] < 8:
        add("slow", f"{h['threads']} CPU threads",
            "the engine keeps one host thread per card spinning for work, Strata's server and the prompt path's 12 "
            "embedding threads (FNX_PF_EMBED_THREADS) need more; ours has 16", "[unmeasured] below 16")
    if h.get("governor") == "powersave" and h.get("cpufreq") and "pstate" not in h["cpufreq"]:
        add("slow", f"CPU frequency governor powersave ({h['cpufreq']}): the cores stay at their lowest clock",
            "set ondemand, schedutil or performance; each token waits on the host threads between kernel launches", "[unmeasured]")
    for c in cs:
        if c.get("bar1_gib") == 0:
            add("stop", f"card {c['index']}: BAR1 has no address", "turn on Above 4G Decoding (64-bit MMIO) in the BIOS: "
                "a P100 needs a 16 GiB BAR1", "the engine refuses such a card (card_check.hpp)")
    mem = h["mem_gib"]
    if mem >= 88:
        prof = {"ram": 96, "model": "iq3_s", "setup": "bash setup.sh DATA",
                "why": f"{mem} GiB of RAM: the default IQ3_S profile (512k) fits; it is the faster file"}
        if mem < 120:
            prof["q4"] = "UD-Q4_K_XL (--model q4_k_xl) fits too, tightly (10 GiB left at the lowest on ours)"
        else:
            prof["q4"] = "UD-Q4_K_XL (--model q4_k_xl) fits with room; its lookup table may stay in the page cache"
    elif mem >= 60:
        prof = {"ram": 64, "model": "iq3_s", "setup": "bash setup.sh DATA --ram 64",
                "why": f"{mem} GiB of RAM: the 64 GB mode (262,144 positions, the lookup table read from the disk), IQ3_S only"}
    else:
        prof = {"ram": 0, "model": None, "setup": None, "why": f"{mem} GiB of RAM: too little; the 64 GB mode needs about 60 GiB"}
        add("stop", f"{mem} GiB of RAM", "64 GB at least (the load's 55.6 GiB must fit beside the system)", "[measured] in a 58 GiB cgroup")
    key = "iq3_s_64gb" if prof["ram"] == 64 else (model or prof["model"] or "iq3_s")
    holds = HOLDS.get(key, HOLDS["iq3_s"])
    if prof["ram"] and h["avail_gib"] < holds - 5 and not h["engines"]:   # run.sh's rule; the load's extra comes partly from the page cache
        add("slow", f"{h['avail_gib']} GiB of RAM available now, the {key} profile holds about {holds:.0f} GiB once loaded",
            "stop what else runs before the load, or the load swaps or fails", "[measured] on ours")
    if h["swap_used_gib"] > 1:
        add("note", f"{h['swap_used_gib']} GiB of swap in use", "a load that swaps stalls; tune.py stops on new swap", "")
    if h["engines"]:
        add("note", f"an engine runs (pid {', '.join(h['engines'])})", "RAM and card numbers above include it; the copy test skips busy cards", "")

    # the links
    pf = None
    for c in p100:
        r = (copies or {}).get(c["index"], {})
        live = r.get("link_live") or {}
        own = c["path"][0] if c["path"] else {}
        g, w = live.get("gen", own.get("gen")), live.get("width", own.get("width"))
        mg, mw = own.get("max_gen"), own.get("max_width")
        tag = f"card {c['index']} ({c['bdf']})"
        nar = narrowest(c, r.get("path_live"))
        if live and g and w and mg and mw and (g < mg or w < mw):
            add("slow", f"{tag}: link Gen{g} x{w} while copying, the card can do Gen{mg} x{mw}",
                "a slot wired narrower, a riser, or the slot's speed set lower in the BIOS; reseat or move the card", "[measured] ours run Gen3 x16")
        elif not live and mw and w and w < mw:
            add("note", f"{tag}: link x{w} idle (max x{mw})", "the copy test (cards free) tells whether it holds under load", "")
        if nar and nar[3] != c["bdf"]:
            add("slow" if r.get("path_live") else "note",
                f"{tag}: the bridge {nar[3]} above it runs Gen{nar[1]} x{nar[2]}{'' if r.get('path_live') else ' idle'}, narrower than the card",
                "the card's traffic shares or squeezes through that link" + ("" if r.get("path_live") else "; the copy test tells whether it holds under load"),
                "[unmeasured]")
        if c["iommu_type"] not in ("identity", "none", ""):
            add("slow", f"{tag}: the IOMMU translates its DMA (group type {c['iommu_type']})",
                "add iommu=pt to the kernel command line (or turn VT-d/AMD-Vi off)", "[measured] ours: 7.6-8 GB/s instead of 11.9 (an earlier test)")
        h2d = r.get("h2d")
        if h2d:
            if h2d < 0.85 * OURS["h2d"]:
                pf = min(pf or 9, 3 if h2d < 8 else 4)
                add("slow", f"{tag}: {h2d} GB/s to the card (ours {OURS['h2d']})",
                    "experts that are not in VRAM cross this link for every decode window and prompt chunk: expect reading to slow down "
                    "about in proportion, writing less; fewer guessed next-layer copies (FNX_PREFETCH_WANT 2-3, FNX_PREFETCH_MIN_T 2) "
                    "leave the link to the window's own misses", "[unmeasured] rule, from UD-Q4_K_XL on ours where PCIe was the limit: 6 a row → 2 gave edits +15-20 %")
            far = r.get("h2d_other_node")
            if far and far < 0.85 * h2d:
                add("note", f"{tag}: {far} GB/s from the other CPU's memory against {h2d} from its own",
                    "the engine pins each card's host thread to the card's own CPU; FNX_NUMA_EXPERTS=1 places its RAM copies there too (same bits)",
                    "[measured] ours: the same from either node, and FNX_NUMA_EXPERTS=1 in the noise")
            if r.get("d2h") and r["d2h"] < 0.7 * h2d:
                add("note", f"{tag}: {r['d2h']} GB/s back to the host",
                    "the cards trade their sums through host memory while reading a prompt", "[unmeasured]")
        if c["power.limit"] and c["power.limit"] < OURS["power_w"] - 5:
            add("slow", f"{tag}: power limit {c['power.limit']:.0f} W", f"ours ran at {OURS['power_w']} W; raise it with nvidia-smi -pl",
                "[unmeasured] below 185 W")
        elif c["power.limit"] and c["power.limit"] >= 240:
            add("note", f"{tag}: power limit {c['power.limit']:.0f} W", "long reads were at most 1-1.5 % faster at 250 W than at 185 W on ours",
                "[measured]")
    both = (copies or {}).get("both")
    singles = [(copies or {}).get(c["index"], {}).get("h2d") for c in p100]
    if both and all(singles) and sum(both) < 0.8 * sum(singles):
        pf = min(pf or 9, 3)
        add("slow", f"both cards at once: {' + '.join(map(str, both))} GB/s against {' and '.join(map(str, singles))} alone",
            "the two cards share a link (a PCIe switch or one root port); the engine copies to both at once. Put them on slots "
            "of different root ports (or CPUs); else treat it like a narrow link", "[unmeasured]")
    if len(p100) == 2:
        sb = shared_bridge(*p100)
        if sb:
            add("note", f"both cards hang from the bridge {sb[0]}", "see the copy test of both cards at once", "")
    for n, v in h["numa"].items():
        if v["mem_gib"] is not None and v["mem_gib"] < 1 and any(c["numa_node"] == n for c in p100):
            add("slow", f"NUMA node {n} carries a card but has no memory", "fill that CPU's memory channels too", "[unmeasured]")
    if copies is None:
        add("note", "copies not measured", "run with the cards free (or --force-copy) for the most telling number", "")

    # the disk
    if disk:
        if disk.get("seq_gbs") is not None and disk["seq_gbs"] < 0.3:
            add("slow", f"the disk reads {disk['seq_gbs']} GB/s in a row",
                f"the load reads about 48 GB: about {48 / max(disk['seq_gbs'], 0.01) / 60:.0f} minutes for that part; an SSD helps",
                f"[measured] ours {OURS['disk_seq']} GB/s, about 100 s of the load")
        disk_table = prof["ram"] == 64 or model == "q4_k_xl"
        if disk_table and disk.get("rand_ops") is not None and disk["rand_ops"] < 2000:
            add("slow", f"the disk does {disk['rand_ops']} random 128 KiB reads a second",
                "this profile reads the lookup table from the disk: after a load each new token's row is a random read until the "
                "page cache holds it; an SSD or NVMe makes the first long reads after a load faster",
                "[measured] ours (disks behind a RAID controller) with UD-Q4_K_XL: the first 38k read after a load 765 tok/s, later ones about 900")
        elif not disk_table and disk.get("rand_ops") is not None and disk["rand_ops"] < 200:
            add("note", f"the disk does {disk['rand_ops']} random 128 KiB reads a second (spinning disks, or behind a RAID controller)",
                "only the load reads it with this profile; the 64 GB mode and UD-Q4_K_XL would read the lookup table from it", "")
    if data_free is not None and model is None and data_free < 95:
        add("note", f"{data_free:.0f} GB free where the data folder is and no model files found",
            "setup.sh needs about 95 GB (UD-Q4_K_XL 125)", "")
    if disk and h.get("io_pressure", 0) > 5:
        add("note", f"other work waited for I/O {h['io_pressure']:.0f} % of the time when the disk was measured",
            "the disk numbers may be low; run again when the machine is quiet", "")
    if h["thp"] and h["thp"] != "madvise":
        add("note", f"transparent huge pages: {h['thp']}", "ours runs madvise; others not measured", "")
    if pf:
        prof["env"] = {"FNX_PREFETCH_WANT": str(min(pf, 6)), "FNX_PREFETCH_MIN_T": "2"}
        prof["env_basis"] = "[unmeasured] a narrow link: start tune.py from fewer guessed next-layer copies"
    return prof, found


# ------------------------------------------------------------------ report
def report(h, cs, copies, disk, dinfo, folder, model, prof, found):
    L = ["## Machine report (tune/check.py)", "", "| | |", "|---|---|",
         f"| OS, kernel | {h['os']}, {h['kernel']} |",
         f"| CPU | {h['cpu']} x {h['sockets']} ({h['threads']} threads; {' '.join(h.get('isa') or []) or 'no AVX'}), NUMA nodes: " +
         ", ".join(f"{n}: {v['mem_gib']} GiB, CPUs {v['cpus']}" for n, v in h["numa"].items()) + " |",
         f"| RAM | {h['mem_gib']} GiB, {h['avail_gib']} available now; swap {h['swap_gib']} GiB ({h['swap_used_gib']} used) |",
         f"| IOMMU | {h['iommu_units']} units; kernel: {' '.join(h['cmdline_iommu']) or '(nothing set)'}; THP {h['thp']} |",
         f"| Driver | {cs[0]['driver_version'] if cs else '?'} (CUDA {h['cuda_driver']}) |"]
    for c in cs:
        path = " ← ".join(f"{x['bdf']} Gen{x['gen']}x{x['width']} (max Gen{x['max_gen']}x{x['max_width']})" for x in c["path"])
        L.append(f"| Card {c['index']} | {c['name']} cc {c['compute_cap']}, {num(c['memory.total'])} MiB ({num(c['memory.used'])} used), "
                 f"{c['power.limit']} W (default {c['power.default_limit']}), SM {c['clocks.sm']}/{c['clocks.max.sm']} MHz, "
                 f"{c['temperature.gpu']} °C, NUMA node {c['numa_node']}, IOMMU group {c['iommu_type']}, BAR1 {c.get('bar1_gib')} GiB |")
        L.append(f"| Card {c['index']} path | {path} |")
    if copies:
        for c in cs:
            r = copies.get(c["index"])
            if r:
                live = r.get("link_live") or {}
                L.append(f"| Copies, card {c['index']} | to the card {r['h2d']} GB/s (1 MiB copies {r['h2d_1mib']}), back {r['d2h']}"
                         + (f", from node {r['other_node']}: {r['h2d_other_node']}" if "h2d_other_node" in r else "")
                         + f"; link while copying Gen{live.get('gen')} x{live.get('width')} |")
        if copies.get("both"):
            L.append(f"| Copies, both at once | {' + '.join(map(str, copies['both']))} GB/s |")
    else:
        L.append("| Copies | not measured |")
    if disk:
        L.append(f"| Disk | {dinfo.get('device')} {dinfo.get('fs')}{' (rotational)' if dinfo.get('rotational') else ''}: "
                 f"{disk['seq_gbs']} GB/s in a row, {disk['rand_ops']} random 128 KiB reads/s"
                 f"{'' if disk['seq_direct'] and disk['rand_direct'] else ' (no O_DIRECT)'} |")
    L.append(f"| Model files | {model or 'not found'}{f' in {folder}' if folder else ''} |")
    L += ["", f"Ours for comparison: {OURS['h2d']} GB/s to each card (Gen3 x16, iommu=pt), 185 W, 96 GB, disk {OURS['disk_seq']} GB/s.", "",
          "### Verdict", "", f"- **Profile:** {prof['why']}." + (f" Set up with `{prof['setup']}`." if prof["setup"] else "")]
    if prof.get("q4"):
        L.append(f"- {prof['q4']}.")
    if prof.get("env"):
        L.append(f"- **Start tuning from:** {' '.join(f'{k}={v}' for k, v in prof['env'].items())} ({prof['env_basis']}).")
    order = {"stop": 0, "slow": 1, "warn": 2, "note": 3}
    for f in sorted(found, key=lambda x: order[x["level"]]):
        L.append(f"- **{f['level']}**: {f['what']}. {f['advice'][:1].upper()}{f['advice'][1:]}." + (f" _{f['basis']}_" if f["basis"] else ""))
    if not any(f["level"] in ("stop", "slow") for f in found):
        L.append("- No bottleneck found against our machine.")
    return "\n".join(L)


def inspect(a, log=print):
    log("cards")
    cs = cards()
    if not cs:
        log("  nvidia-smi shows no card")
    for c in cs:
        log(f"  {c['index']}: {c['name']} cc {c['compute_cap']} {c['bdf']}, {num(c['memory.used'])} MiB used, NUMA {c['numa_node']}, "
            f"IOMMU {c['iommu_type']}, link idle Gen{c['path'][0]['gen'] if c['path'] else '?'} x{c['path'][0]['width'] if c['path'] else '?'}")
    log("host")
    h = host()
    log(f"  {h['cpu']}, {h['mem_gib']} GiB RAM ({h['avail_gib']} available), IOMMU units {h['iommu_units']}, kernel {' '.join(h['cmdline_iommu'])}")
    copies = None
    p100 = [c for c in cs if "P100" in c["name"]]
    busy = [c for c in p100 if (c["memory.used"] or 0) > 1024]
    if a.no_copy:
        log("copies: skipped (--no-copy)")
    elif busy and not a.force_copy:
        log(f"copies: skipped, card(s) {', '.join(c['index'] for c in busy)} in use (--force-copy runs anyway)")
    elif p100:
        log("copies (pinned host memory, driver API)")
        try:
            copies = copy_tests(p100[:2], a.seconds, log)
        except (OSError, RuntimeError, AttributeError) as e:   # AttributeError: a libcuda.so.1 without one of the calls
            log(f"  failed: {e}")
    folder, model, paths = find_models(a)
    disk, dinfo = None, {}
    if paths and not a.no_disk:
        dinfo = block_device(paths[0])
        log(f"disk ({model} in {folder}, {dinfo.get('device')})")
        try:
            disk = disk_tests(paths, a.seconds, log)
            disk["rotational"] = dinfo.get("rotational")
        except OSError as e:
            log(f"  failed: {e}")
    elif not paths:
        log("disk: no model files found (--data or --models)")
    data_free = shutil.disk_usage(a.data).free / 1e9 if a.data and os.path.isdir(a.data) else None
    prof, found = judge(h, cs, copies, disk, model, data_free)
    text = report(h, cs, copies, disk, dinfo, folder, model, prof, found)
    return {"host": h, "cards": cs, "copies": copies, "disk": disk, "disk_device": dinfo, "model": model,
            "profile": prof, "findings": found, "report": text}


def write_profile(base, prof, out):
    c = json.loads(Path(base).read_text())
    for k, v in (prof.get("env") or {}).items():
        c["env"][k] = v
    Path(out).write_text(json.dumps(c, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", help="the folder setup.sh fills (profile.json, models/)")
    ap.add_argument("--models", help="the folder with the model files, if not DATA/models")
    ap.add_argument("--seconds", type=float, default=3, help="per copy and disk test (default 3)")
    ap.add_argument("--no-copy", action="store_true", help="skip the copy test")
    ap.add_argument("--force-copy", action="store_true", help="run the copy test on cards that are in use")
    ap.add_argument("--no-disk", action="store_true", help="skip the disk test")
    ap.add_argument("--out", help="write the report (Markdown) here")
    ap.add_argument("--json", help="write everything measured as JSON here")
    ap.add_argument("--write-profile", metavar="OUT", help="DATA/profile.json with the suggested switches, written to OUT")
    a = ap.parse_args()
    if sys.platform != "linux":
        sys.exit("check.py runs on the Linux machine with the cards")
    r = inspect(a)
    print()
    print(r["report"])
    if a.out:
        Path(a.out).write_text(r["report"] + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")
    if a.json:
        Path(a.json).write_text(json.dumps({k: v for k, v in r.items() if k != "report"}, indent=1, default=str) + "\n", encoding="utf-8")
    if a.write_profile:
        if not a.data or not (Path(a.data) / "profile.json").exists():
            sys.exit("--write-profile needs --data with a profile.json")
        write_profile(Path(a.data) / "profile.json", r["profile"], a.write_profile)
        print(f"wrote {a.write_profile} ({' '.join(f'{k}={v}' for k, v in (r['profile'].get('env') or {}).items()) or 'no changes'})")


if __name__ == "__main__":
    main()
