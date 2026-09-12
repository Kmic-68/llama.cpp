#!/usr/bin/env python3
"""Runs inside bwrap. Reads /payload.json = {code, tests, mode, method}.
Prints one JSON line: {"passed": n, "total": n, "err": "..."}"""
import json, sys, io, signal, contextlib

P = json.load(open("/tmp/payload.json"))
code, tests, mode = P["code"], P["tests"], P["mode"]

class TO(Exception): pass
def _alarm(s, f): raise TO()
signal.signal(signal.SIGALRM, _alarm)

def norm(s):
    return "\n".join(l.rstrip() for l in s.strip().splitlines())

def run_stdin(t):
    g = {"__name__": "__main__"}
    out = io.StringIO()
    # TextIOWrapper over BytesIO so sys.stdin.buffer works: solutions routinely
    # read via sys.stdin.buffer.read(), which a bare StringIO cannot serve.
    sys.stdin = io.TextIOWrapper(io.BytesIO(t["input"].encode()), encoding="utf-8")
    with contextlib.redirect_stdout(out):
        try:
            exec(compile(code, "sol.py", "exec"), g)
        except SystemExit:
            pass
    return norm(out.getvalue()) == norm(t["output"])

_fg = None
def run_func(t):
    global _fg
    if _fg is None:
        _fg = {"__name__": "__lcb__"}
        exec(compile(code, "sol.py", "exec"), _fg)
    args = [json.loads(l) for l in t["input"].strip().splitlines()]
    exp = json.loads(t["output"])
    fn = getattr(_fg["Solution"](), P["method"])
    got = fn(*args)
    if isinstance(got, tuple):
        got = list(got)
    return json.loads(json.dumps(got)) == exp

npass, err = 0, ""
for t in tests:
    signal.alarm(8)
    try:
        ok = run_stdin(t) if mode == "stdin" else run_func(t)
    except TO:
        err = err or "TIMEOUT"; ok = False
    except BaseException as e:
        err = err or f"{type(e).__name__}: {e}"[:160]; ok = False
    finally:
        signal.alarm(0)
    if not ok:
        break
    npass += 1

print(json.dumps({"passed": npass, "total": len(tests), "err": err}))
