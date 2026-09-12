#!/usr/bin/env python3
"""LiveCodeBench v6 pass@1 against an llama-server OpenAI endpoint.
Official LCB prompt format, greedy. Judged in bwrap: no network, all-tests-must-pass."""
import json, re, base64, zlib, pickle, random, subprocess, time, argparse, urllib.request, os, sys, threading
from concurrent.futures import ThreadPoolExecutor

AP = argparse.ArgumentParser()
AP.add_argument("--port", type=int, default=8080)
AP.add_argument("--data", required=True)
AP.add_argument("--out", required=True)
AP.add_argument("--n", type=int, default=24)
AP.add_argument("--seed", type=int, default=1234)
AP.add_argument("--max-tokens", type=int, default=16384)
AP.add_argument("--tests", type=int, default=40)
AP.add_argument("--workers", type=int, default=1)
AP.add_argument("--resume", action="store_true")
A = AP.parse_args()

ROWS = [json.loads(l) for l in open(A.data)]
random.Random(A.seed).shuffle(ROWS)
ROWS = ROWS[:A.n]

# --resume: keep finished problems, re-run only what is missing
EXIST = {}
if A.resume and os.path.exists(A.out):
    try:
        for r in json.load(open(A.out)):
            if "ok" in r:
                EXIST[r["id"]] = r
    except Exception:
        EXIST = {}
    ROWS = [r for r in ROWS if r["question_id"] not in EXIST]
    print(f"resume: {len(EXIST)} done, {len(ROWS)} to go", flush=True)

def tests_of(r):
    pub = json.loads(r["public_test_cases"])
    raw = base64.b64decode(r["private_test_cases"])
    try:
        prv = json.loads(zlib.decompress(raw).decode())
    except Exception:
        prv = json.loads(pickle.loads(zlib.decompress(raw)))
    return (pub + prv)[:A.tests]

SYS = ("You are an expert Python programmer. You will be given a question (problem specification) "
       "and will generate a correct Python program that matches the specification and passes all tests.")

def prompt(r):
    q = "### Question:\n" + r["question_content"] + "\n\n"
    if r["starter_code"].strip():
        q += ("### Format: You will use the following starter code to write the solution to the "
              "problem and enclose your code within delimiters.\n```python\n" + r["starter_code"] + "\n```\n\n")
    else:
        q += ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout "
              "(do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n"
              "```python\n# YOUR CODE HERE\n```\n\n")
    return q + "### Answer: (use the provided format with backticks)\n"

def generate(r):
    body = json.dumps({
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": prompt(r)}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": A.max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{A.port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as f:
        j = json.load(f)
    c = j["choices"][0]
    return c["message"]["content"], j.get("usage", {}).get("completion_tokens", 0), c.get("finish_reason", "")

def extract(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    b = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return b[-1] if b else text

HERE = os.path.dirname(os.path.abspath(__file__))
def judge(code, r, work, tag=""):
    mode = "func" if r["starter_code"].strip() else "stdin"
    m = re.search(r"def\s+(\w+)\s*\(\s*self", r["starter_code"] or "")
    payload = {"code": code, "tests": tests_of(r), "mode": mode,
               "method": m.group(1) if m else ""}
    pj = os.path.join(work, f"payload{tag}.json")
    json.dump(payload, open(pj, "w"))
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--tmpfs", "/tmp", "--unshare-net", "--unshare-pid", "--die-with-parent",
           "--ro-bind", pj, "/tmp/payload.json",
           "--ro-bind", os.path.join(HERE, "driver.py"), "/tmp/driver.py",
           "--chdir", "/tmp", "python3", "/tmp/driver.py"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return json.loads(p.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": 0, "err": "HARD_TIMEOUT"}
    except Exception as e:
        return {"passed": 0, "total": 0, "err": f"JUDGE {type(e).__name__} {str(p.stderr)[-160:] if 'p' in dir() else e}"}

work = os.path.dirname(os.path.abspath(A.out)); os.makedirs(work, exist_ok=True)
results, lock, t0, done = {}, threading.Lock(), time.time(), [0]

def one(i_r):
    i, r = i_r
    ts = time.time()
    try:
        text, ntok, fin = generate(r)
    except Exception as e:
        return i, {"id": r["question_id"], "ok": False, "err": f"GEN {e}", "tok": 0,
                   "difficulty": r["difficulty"], "platform": r["platform"]}
    code = extract(text)
    # each worker needs its own payload file
    v = judge(code, r, work, tag=str(i))
    ok = v["total"] > 0 and v["passed"] == v["total"]
    row = {"id": r["question_id"], "platform": r["platform"], "difficulty": r["difficulty"],
           "ok": bool(ok), "passed": v["passed"], "total": v["total"], "err": v["err"],
           "tok": ntok, "finish": fin, "code": code}
    with lock:
        results[i] = row
        done[0] += 1
        d = done[0]
        npass = sum(x["ok"] for x in results.values())
        el = time.time() - t0
        print(f"[{d:3d}/{len(ROWS)}] {r['question_id']:<16} {r['difficulty']:<6} "
              f"{'PASS' if ok else 'FAIL'} {v['passed']:>2}/{v['total']:<2} {ntok:6d}tok "
              f"{time.time()-ts:6.1f}s | {npass}/{d}={100*npass/d:5.1f}% "
              f"eta {el/d*(len(ROWS)-d)/60:6.1f}m {'TRUNC' if fin=='length' else ''} {v['err'][:40]}",
              flush=True)
        json.dump(list(EXIST.values()) + [results[k] for k in sorted(results)], open(A.out, "w"), indent=1)
    return i, row

with ThreadPoolExecutor(max_workers=A.workers) as ex:
    list(ex.map(one, list(enumerate(ROWS))))

rows = list(EXIST.values()) + [results[k] for k in sorted(results)]
npass = sum(r["ok"] for r in rows)
trunc = sum(r.get("finish") == "length" for r in rows)
n = len(rows)
print(f"\nPASS@1 = {npass}/{n} = {100*npass/n:.1f}%   truncated {trunc}   wall {(time.time()-t0)/60:.1f} min")
for d in ("easy", "medium", "hard"):
    s = [x for x in rows if x.get("difficulty") == d]
    if s: print(f"  {d:<6} {sum(x['ok'] for x in s)}/{len(s)}")
