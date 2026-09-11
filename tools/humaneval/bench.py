#!/usr/bin/env python3
"""Minimal HumanEval pass@1 harness against an llama-server OpenAI endpoint.
Greedy (temperature 0). Generated code runs under bwrap: no network, tmpfs /tmp."""
import json, re, subprocess, sys, time, argparse, urllib.request, os

AP = argparse.ArgumentParser()
AP.add_argument("--port", type=int, default=8080)
AP.add_argument("--out", required=True)
AP.add_argument("--limit", type=int, default=0)
AP.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "HumanEval.jsonl"))
A = AP.parse_args()

PROBS = [json.loads(l) for l in open(A.data)]
if A.limit:
    PROBS = PROBS[:A.limit]

SYS = "You are an expert Python programmer."
def user_msg(p):
    return ("Complete the following Python function. Return the complete function, "
            "including its signature and any needed imports, in a single ```python code block. "
            "Do not include tests or explanation.\n\n```python\n" + p["prompt"] + "```")

def generate(p):
    body = json.dumps({
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": user_msg(p)}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 2048, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{A.port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    c = j["choices"][0]
    return c["message"]["content"], j.get("usage", {}).get("completion_tokens", 0), c.get("finish_reason","")

def extract(text, p):
    # drop reasoning traces if the template emits them
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S) if "</think>" in text else text
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    code = blocks[0] if blocks else text
    if f"def {p['entry_point']}" in code:
        return code
    # model returned only a body -> graft onto the original signature
    return p["prompt"] + code

BWRAP = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
         "--tmpfs", "/tmp", "--unshare-net", "--unshare-pid", "--die-with-parent",
         "--chdir", "/tmp"]

def check(code, p, workdir):
    prog = code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\nprint('__PASS__')\n"
    path = os.path.join(workdir, "prog.py")
    open(path, "w").write(prog)
    try:
        r = subprocess.run(BWRAP + ["python3", path], capture_output=True, text=True, timeout=20)
        return ("__PASS__" in r.stdout), (r.stderr.strip().splitlines() or [""])[-1][:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"

work = os.path.dirname(os.path.abspath(A.out))
os.makedirs(work, exist_ok=True)
rows, npass, t0, toks = [], 0, time.time(), 0
for i, p in enumerate(PROBS):
    ts = time.time()
    try:
        text, ntok, fin = generate(p)
    except Exception as e:
        rows.append({"task_id": p["task_id"], "passed": False, "err": f"GEN {e}", "tok": 0}); continue
    code = extract(text, p)
    ok, err = check(code, p, work)
    npass += ok; toks += ntok
    rows.append({"task_id": p["task_id"], "passed": ok, "err": "" if ok else err,
                 "tok": ntok, "finish": fin, "code": code})
    el = time.time() - t0
    print(f"[{i+1:3d}/{len(PROBS)}] {p['task_id']:<14} {'PASS' if ok else 'FAIL'} "
          f"{ntok:5d}tok {time.time()-ts:5.1f}s  running {npass}/{i+1}={100*npass/(i+1):5.1f}%  "
          f"eta {el/(i+1)*(len(PROBS)-i-1)/60:5.1f}m", flush=True)

json.dump(rows, open(A.out, "w"), indent=1)
n = len(rows)
print(f"\nPASS@1 = {npass}/{n} = {100*npass/n:.2f}%   mean {toks/max(n,1):.0f} completion tok   "
      f"wall {(time.time()-t0)/60:.1f} min")
