#!/usr/bin/env python3
"""The speed tests we ran on the engine, against a running Strata server (OpenAI-compatible, serve/server.py).

    python3 bench/bench.py --strata-dir /data/flash-next/strata-v0.2.9 --url http://127.0.0.1:8080 --wait --out results.json
    python3 bench/bench.py --strata-dir /data/flash-next/strata-v0.2.9 --tests fresh,edit,long,medium,image ...
    python3 bench/bench.py --strata-dir /data/flash-next/strata-v0.2.9 --dry-run      # build the inputs, print their sizes, no server

Tests (the names in brackets are those of our measurement logs):
  fresh   [sveze]     12 new prompts, 300 tokens each, temperature 0, reasoning "low": all tokens over all decode time
  edit    [urejanje]  the model repeats a file with a small change (lookup drafts pay here), 3 runs each, mean of runs 2-3:
                      py = Strata's tools/collect_profile_traces.py, "rename args to argumenti";
                      md = Strata's docs/SECOND_GPU.md (first 5,000 characters), "replace peer with sosed"
  long    [dolgo]     one cold read of 120,000 characters (~36k tokens), then up to 200 tokens: read and write speed
  medium  [srednje]   a running conversation gets new messages of 6k ... 72k characters: read speed of each
  image   [slika]     a generated 96x96 two-colour PNG: time to the first answer, then writing after it (needs pictures)

The texts come from a Strata checkout at v0.2.9 (the server's own tree is fine): docs/*.md and src/**/*.cpp in path
order for the reads, and the two files above for the edits. The checkout must be at v0.2.9 with those paths unchanged,
so every run reads the same text (--any-commit skips the check). Some prompts are in Slovenian, as in our runs: the
engine's draft vocabulary and our numbers were tuned with them.

Run it right after the engine has loaded, with nothing else using it, and once only: the expert cache adapts to what it
has seen, so a second run on the same engine reads and writes faster than a fresh one.
"""
import argparse
import base64
import json
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

STRATA_V029 = "c62d5e3a15b40ea51c96aae9e8d36928965f53ab"   # Strata v0.2.9 (github.com/Niko1221/Strata)
INPUT_PATHS = ("docs", "src", "tools/collect_profile_traces.py")

# Our numbers (2 x Tesla P100 16 GB PCIe x16 Gen3, 2 x Xeon Bronze 3106, 96 GB DDR4; profile/ with FNX_ROUTE_PRIOR=0): this
# benchmark, each on a freshly loaded engine, one run each (README, "Our numbers"), on this folder's sources built with jobs of
# 96 rows (build.sh's default; e56c3f60): UD-Q4_K_XL on 16 Nov 2026, IQ3_S and Edge0-145B on 17 Nov, all before 18 Nov's switch
# (FNX_MTP_NORM_STREAMS) and UD-Q4_K_XL settings, with which this benchmark has not been run (README, "Unsloth's UD-Q4_K_XL").
# Keys: --ours, or the model the server reports (auto: q4_k_xl is the 512k profile; edge0 Edge0-145B's).
REFERENCE = {
    "iq3_s": {   # the IQ3_S profile (512k); that morning's first run with pictures on this file (our rounds: first answer 2.6-2.7 s)
        "fresh": 66.45, "edit_py": 110.65, "edit_md": 115.8, "long_read": 1012.1, "long_write": 61.5,
        "medium": {6000: 729, 12000: 854, 18000: 900, 24000: 842, 36000: 1000, 48000: 927, 72000: 961},
        "image_first_s": 5.14, "image_write": 52.9,
    },
    "iq3_s_64gb": {   # the IQ3_S profile with --ram 64 (256k), in a 58 GiB memory cgroup, the lookup table cold
        "fresh": 63.77, "edit_py": 93.6, "edit_md": 99.2, "long_read": 806.5, "long_write": 57.4,
        "medium": {6000: 628, 12000: 741, 18000: 795, 24000: 736, 36000: 898, 48000: 825, 72000: 862},
    },
    "q4_k_xl": {   # the UD-Q4_K_XL profile (512k), a 96-row build, the lookup table cold at the start (README, "Unsloth's UD-Q4_K_XL")
        "fresh": 50.92, "edit_py": 62.45, "edit_md": 63.7, "long_read": 794.6, "long_write": 49.2,
        "medium": {6000: 537, 12000: 716, 18000: 767, 24000: 722, 36000: 877, 48000: 779, 72000: 840},
        "image_first_s": 6.45, "image_write": 46.8,   # the first run of that window; the three runs after it: 2.99-3.02 s
    },
    "q4_k_xl_256k": {   # its 256k profile, run right after the 512k one: the lookup table mostly in the page cache
        "fresh": 50.94, "edit_py": 63.35, "edit_md": 64.95, "long_read": 929.0, "long_write": 48.5,
        "medium": {6000: 605, 12000: 808, 18000: 856, 24000: 807, 36000: 969, 48000: 869, 72000: 930},
        "image_first_s": 2.99, "image_write": 47.0,
    },
    "edge0": {   # Edge0-145B (512k), 17 Nov 2026, a 96-row build (e56c3f60, our Edge0 production engine since then; README, "Edge0-145B")
        "fresh": 55.0, "edit_py": 96.55, "edit_md": 98.9, "long_read": 691.0, "long_write": 57.1,
        "medium": {6000: 535, 12000: 612, 18000: 637, 24000: 607, 36000: 686, 48000: 647, 72000: 669},
        "image_first_s": 5.08, "image_write": 43.3,
    },
}
UNITS = {"image_first_s": "s"}

FRESH = [
    "Razloži v približno 200 besedah, kako deluje hranilnik toplote v hiši in kako ga dimenzioniramo.",
    "Napiši funkcijo v Pythonu, ki prebere CSV z datumi in vrne mesečne vsote po kategorijah. Dodaj primer.",
    "Explain in about 200 words how a B-tree index speeds up database lookups and when it does not help.",
    "Write a small Dart class for a retrying HTTP client with exponential backoff and a cancel method.",
    "Opiši v približno 200 besedah zgodovino železnice na Slovenskem in njen pomen za gospodarstvo.",
    "Napiši bash skripto, ki v mapi poišče datoteke, večje od 1 GB, in jih razvrsti po datumu spremembe.",
    "Write a SQL schema for a small library with loans and fines, then a query listing overdue books per member.",
    "Razloži v približno 200 besedah razliko med TCP in UDP ter navedi, kdaj uporabimo katerega.",
    "Napiši v JavaScriptu funkcijo debounce in throttle z testi, brez zunanjih knjižnic.",
    "Pojasni v približno 200 besedah, kako pravilno obrezujemo jablane in zakaj je čas pomemben.",
    "Write a C function that parses a key=value config file into a linked list, with careful error handling.",
    "Popravi in razloži: def povprecje(xs): return sum(xs) / len(xs) ko je seznam prazen, in dodaj teste pytest.",
]
SYSTEM_SHORT = "Si natančen pomočnik. Odgovarjaš kratko, v slovenščini."


class Bench:
    def __init__(self, url, model, strata_dir, timeout):
        self.url, self.model, self.root, self.timeout = url.rstrip("/"), model, Path(strata_dir), timeout
        self._files = None

    # ------------------------------------------------------------------ inputs
    def corpus(self, chars, skip=0):
        """docs/*.md then src/**/*.cpp in path order, each under a "=== path ===" line; chars characters after skip."""
        if self._files is None:
            key = lambda p: p.parts   # noqa: E731  (the order sorted() gives Paths, made explicit)
            self._files = sorted(self.root.glob("docs/*.md"), key=key) + sorted(self.root.glob("src/**/*.cpp"), key=key)
        parts, total = [], 0
        for f in self._files:
            t = f.read_text(encoding="utf-8", errors="replace")
            parts.append(f"\n\n=== {f.relative_to(self.root).as_posix()} ===\n{t}")
            total += len(parts[-1])
            if total >= chars + skip:
                break
        text = "".join(parts)[skip:skip + chars]
        if len(text) < chars:
            sys.exit(f"the corpus in {self.root} has fewer than {chars + skip} characters")
        return text

    def edit_tasks(self):
        src = (self.root / "tools/collect_profile_traces.py").read_text(encoding="utf-8", errors="replace")
        doc = (self.root / "docs/SECOND_GPU.md").read_text(encoding="utf-8", errors="replace")[:5000]
        return (("py", "Tu je Python datoteka:\n\n```python\n" + src + "\n```\n\nPreimenuj spremenljivko `args` v `argumenti` "
                 "in vrni celotno datoteko, sicer nespremenjeno. Vrni samo kodo.", 1600),
                ("md", "Tu je dokument:\n\n" + doc + "\n\nBesedo `peer` povsod zamenjaj z besedo `sosed` in vrni celoten dokument, "
                 "sicer nespremenjen. Vrni samo dokument.", 1600))

    # ------------------------------------------------------------------ requests
    def http(self, method, path, body=None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.load(r)

    def chat(self, messages, max_tokens, effort="low", sampling=None):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "reasoning_effort": effort, "stream": False,
                **(sampling if sampling is not None else {"temperature": 0.0})}
        t0 = time.time()
        out = self.http("POST", "/v1/chat/completions", body)
        return out, time.time() - t0

    def wait_loaded(self, minutes):
        end = time.time() + minutes * 60
        while time.time() < end:
            try:
                if self.http("GET", "/health", timeout=5).get("loaded"):
                    return
            except Exception:
                pass
            time.sleep(5)
        sys.exit(f"the server at {self.url} did not report loaded within {minutes} minutes")

    # ------------------------------------------------------------------ tests
    def fresh(self):
        gen, ms, each = 0, 0.0, []
        for i, prompt in enumerate(FRESH):
            out, _ = self.chat([{"role": "user", "content": prompt}], 300)
            t = out.get("timings") or {}
            gen, ms = gen + (t.get("predicted_n") or 0), ms + (t.get("predicted_ms") or 0.0)
            each.append(round(t.get("predicted_per_second") or 0, 1))
            log(f"  fresh {i}: {each[-1]} tok/s")
        return {"tok_s": round(gen / ms * 1000, 2) if ms else None, "each": each}

    def edit(self):
        res = {}
        for name, prompt, n in self.edit_tasks():
            vals = []
            for r in range(3):
                out, _ = self.chat([{"role": "user", "content": prompt}], n, effort="none")
                t = out.get("timings") or {}
                vals.append(t.get("predicted_per_second") or 0)
                log(f"  edit {name} {r + 1}: {vals[-1]:.2f} tok/s ({t.get('predicted_n')} tok)")
            res[name] = round(sum(vals[1:]) / 2, 2)
            res[name + "_runs"] = [round(v, 2) for v in vals]
            res[name + "_tokens"] = t.get("predicted_n")
        return res

    def long(self):
        out, wall = self.chat([{"role": "user", "content": self.corpus(120_000) +
                                "\n\nV petih alinejah v slovenščini povzemi, kaj je Strata."}], 200)
        t = out.get("timings") or {}
        log(f"  long: {t.get('prompt_n')} tok @ {t.get('prompt_per_second')} tok/s, writes {t.get('predicted_per_second')}")
        return {"prompt_n": t.get("prompt_n"), "read_tok_s": t.get("prompt_per_second"), "write_tok_s": t.get("predicted_per_second"),
                "wall_s": round(wall, 1)}

    def medium(self):
        skip = 1_900_000
        msgs = [{"role": "system", "content": SYSTEM_SHORT},
                {"role": "user", "content": self.corpus(6_000, skip) + "\n\nOdgovori samo z: V redu."}]
        skip += 6_000
        out, _ = self.chat(msgs, 8, effort="none")
        msgs.append({"role": "assistant", "content": out["choices"][0]["message"].get("content") or ""})
        res = {}
        for chars in (6_000, 12_000, 18_000, 24_000, 36_000, 48_000, 72_000):
            msgs.append({"role": "user", "content": self.corpus(chars, skip) + "\n\nOdgovori samo z: V redu."})
            skip += chars
            out, wall = self.chat(msgs, 8, effort="none")
            msgs.append({"role": "assistant", "content": out["choices"][0]["message"].get("content") or ""})
            t = out.get("timings") or {}
            n, pm = t.get("prompt_n") or 0, t.get("prompt_ms") or 0.0
            res[str(chars)] = {"read": n, "ms": round(pm), "tok_s": round(n / max(pm, 1) * 1000)}
            log(f"  medium {chars} chars: read {n} tokens in {pm:.0f} ms, {res[str(chars)]['tok_s']} tok/s, {wall:.2f} s in all")
        return res

    def image(self):
        url = "data:image/png;base64," + two_colour_png()
        msgs = [{"role": "user", "content": [{"type": "text", "text": "Kateri dve barvi sta na tej sliki? Odgovori kratko."},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
        out, wall = self.chat(msgs, 60, effort="none", sampling={"temperature": 0.0, "presence_penalty": 0})
        answer = out["choices"][0]["message"].get("content") or ""
        log(f"  image: {answer[:80]!r}, {wall:.1f} s")
        msgs += [{"role": "assistant", "content": answer},
                 {"role": "user", "content": "Opiši v 80 besedah, kje bi tako sliko lahko uporabili."}]
        out2, _ = self.chat(msgs, 160, effort="none",
                            sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "seed": 93})
        t2 = out2.get("timings") or {}
        log(f"  image follow-up: writes {t2.get('predicted_per_second') or 0:.1f} tok/s ({t2.get('predicted_n')} tok)")
        return {"answer": answer, "first_s": round(wall, 2), "write_tok_s": t2.get("predicted_per_second")}


def two_colour_png():
    """96 x 96: the left half red, the right half blue."""
    w, h = 96, 96
    rows = b"".join(b"\x00" + (b"\xd0\x20\x20" * (w // 2)) + (b"\x20\x40\xd0" * (w // 2)) for _ in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows))
           + chunk(b"IEND", b""))
    return base64.b64encode(png).decode()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def check_checkout(root):
    if not (root / "docs/SECOND_GPU.md").is_file() or not (root / "tools/collect_profile_traces.py").is_file():
        sys.exit(f"{root} is not a Strata checkout (docs/SECOND_GPU.md, tools/collect_profile_traces.py)")
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", *INPUT_PATHS],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"cannot check {root} with git ({e}); --any-commit skips the check")
    if head != STRATA_V029:
        sys.exit(f"{root} is at {head}, not Strata v0.2.9 ({STRATA_V029}): the texts would differ (--any-commit runs anyway)")
    if dirty:
        sys.exit(f"{root} has changes under {', '.join(INPUT_PATHS)}:\n{dirty}")


def summary(results, ours):
    ref = REFERENCE[ours]
    lines = ["", f"{'test':<16}{'yours':>10}{'ours':>10}  unit   (ours: {ours})"]

    def row(name, value, key):
        mine = ref.get(key)
        lines.append(f"{name:<16}{'-' if value is None else f'{value:g}':>10}{'-' if mine is None else f'{mine:g}':>10}  "
                     f"{UNITS.get(key, 'tok/s')}")
    r = results
    if "fresh" in r:
        row("fresh", r["fresh"].get("tok_s"), "fresh")
    if "edit" in r:
        row("edit py", r["edit"].get("py"), "edit_py")
        row("edit md", r["edit"].get("md"), "edit_md")
    if "long" in r:
        row("long read", r["long"].get("read_tok_s"), "long_read")
        row("long, write", r["long"].get("write_tok_s"), "long_write")
    if "medium" in r:
        for chars, mine in ref["medium"].items():
            v = r["medium"].get(str(chars), {})
            lines.append(f"{'medium ' + str(chars // 1000) + 'k ch':<16}{v.get('tok_s', '-'):>10}{mine:>10}  tok/s  "
                         f"({v.get('read', '?')} tokens)")
    if "image" in r:
        row("image, first", r["image"].get("first_s"), "image_first_s")
        row("image, write", r["image"].get("write_tok_s"), "image_write")
    return "\n".join(lines)


def which_reference(b):
    """auto: q4_k_xl or edge0 when the server's model name says so, else iq3_s (the 64 GB mode is chosen with --ours)."""
    try:
        ids = [m.get("id", "") for m in b.http("GET", "/v1/models", timeout=10).get("data", [])]
    except Exception:
        ids = []
    if any("edge0" in i.lower() for i in ids):
        return "edge0"
    return "q4_k_xl" if any("q4_k_xl" in i.lower() for i in ids) else "iq3_s"


def main():
    ap = argparse.ArgumentParser(description="Speed tests of the P100 Flash-Next engine through Strata's server.")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--strata-dir", required=True, help="a Strata checkout at v0.2.9 (the texts are read from it)")
    ap.add_argument("--tests", default="fresh,edit,long,medium", help="comma list of fresh, edit, long, medium, image")
    ap.add_argument("--model", default="strata",
                    help="the model name sent with each request (our runs sent 'strata' and the server served the profile's model)")
    ap.add_argument("--out", help="write all numbers as JSON here")
    ap.add_argument("--wait", type=float, nargs="?", const=20, default=0, help="wait up to N minutes (default 20) for /health loaded")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per request")
    ap.add_argument("--any-commit", action="store_true", help="skip the check that the checkout is v0.2.9")
    ap.add_argument("--dry-run", action="store_true", help="build the inputs and print their sizes; no server")
    ap.add_argument("--ours", choices=["auto", *REFERENCE], default="auto",
                    help="which of our runs to print beside yours (auto: by the model the server reports)")
    a = ap.parse_args()

    tests = [t.strip() for t in a.tests.split(",") if t.strip()]
    bad = [t for t in tests if t not in ("fresh", "edit", "long", "medium", "image")]
    if bad:
        sys.exit(f"unknown tests: {', '.join(bad)}")
    root = Path(a.strata_dir).expanduser().resolve()
    if not a.any_commit:
        check_checkout(root)
    b = Bench(a.url, a.model, root, a.timeout)
    if a.dry_run:
        long_text = b.corpus(120_000)
        print(f"long: {len(long_text)} characters, starts {long_text[:60]!r}")
        for name, prompt, _ in b.edit_tasks():
            print(f"edit {name}: prompt of {len(prompt)} characters")
        skip = 1_906_000
        for chars in (6_000, 12_000, 18_000, 24_000, 36_000, 48_000, 72_000):
            print(f"medium {chars}: {len(b.corpus(chars, skip))} characters from offset {skip}")
            skip += chars
        print(f"image: {len(two_colour_png())} base64 characters")
        return
    if a.wait:
        b.wait_loaded(a.wait)
    ours = which_reference(b) if a.ours == "auto" else a.ours
    results = {"url": a.url, "started": time.strftime("%F %T"), "tests": tests, "ours": ours}
    for t in tests:
        log(f"=== {t}")
        results[t] = getattr(b, t)()
    results["finished"] = time.strftime("%F %T")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary(results, ours))


if __name__ == "__main__":
    main()
