#!/usr/bin/env python3
"""Re-run the judge over stored completions. No GPU, no regeneration:
fixes results that a harness bug scored wrong."""
import json, sys, re, base64, zlib, pickle, subprocess, os

data, resf = sys.argv[1], sys.argv[2]
ROWS = {json.loads(l)["question_id"]: json.loads(l) for l in open(data)}
res = json.load(open(resf))
HERE = os.path.dirname(os.path.abspath(__file__))
work = os.path.dirname(os.path.abspath(resf))

def tests_of(r, k=40):
    pub = json.loads(r["public_test_cases"]); raw = base64.b64decode(r["private_test_cases"])
    try: prv = json.loads(zlib.decompress(raw).decode())
    except Exception: prv = json.loads(pickle.loads(zlib.decompress(raw)))
    return (pub + prv)[:k]

changed = 0
for row in res:
    if "code" not in row or not row["code"].strip():
        continue
    r = ROWS[row["id"]]
    mode = "func" if r["starter_code"].strip() else "stdin"
    m = re.search(r"def\s+(\w+)\s*\(\s*self", r["starter_code"] or "")
    pj = os.path.join(work, "rejudge.json")
    json.dump({"code": row["code"], "tests": tests_of(r), "mode": mode,
               "method": m.group(1) if m else ""}, open(pj, "w"))
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
           "--unshare-net", "--unshare-pid", "--die-with-parent",
           "--ro-bind", pj, "/tmp/payload.json",
           "--ro-bind", os.path.join(HERE, "driver.py"), "/tmp/driver.py",
           "--chdir", "/tmp", "python3", "/tmp/driver.py"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
        v = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        v = {"passed": 0, "total": 0, "err": f"REJUDGE {type(e).__name__}"}
    ok = v["total"] > 0 and v["passed"] == v["total"]
    if ok != row["ok"]:
        changed += 1
        print(f"  {row['id']:<16} {row['ok']} -> {ok}   ({v['passed']}/{v['total']}) {row.get('err','')[:50]}")
    row.update(ok=bool(ok), passed=v["passed"], total=v["total"], err=v["err"])

json.dump(res, open(resf, "w"), indent=1)
print(f"rejudged {len(res)} rows, {changed} changed")
