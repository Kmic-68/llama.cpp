#!/usr/bin/env python3
"""Append the 2026-09-15/16 update to VERIFICATION.md and insert the session block at the top of
RESUME-HERE.md (both under p100-handoff/). Run from the repo root. Refuses on placeholders or on a
second run."""
import sys

VER = "p100-handoff/VERIFICATION.md"
VER_NEW = "p100-handoff/wip-2026-09-14/VERIFICATION-0915-draft.md"
RES = "p100-handoff/RESUME-HERE.md"
RES_NEW = "p100-handoff/wip-2026-09-14/RESUME-HERE-0915-draft.md"
TITLE = "# Resume point — two silent data races closed, prefill matmuls 10x more accurate and faster\n"

for f in (VER_NEW, RES_NEW):
    body = open(f).read()
    if "@@" in body:
        sys.exit("placeholders left in " + f + ":\n" + "\n".join(l for l in body.splitlines() if "@@" in l))

ver, ver_new = open(VER).read(), open(VER_NEW).read()
marker = "### Update 2026-09-15"
if marker in ver:
    sys.exit("VERIFICATION.md already has the update")
if not ver.endswith("\n"):
    ver += "\n"
open(VER, "w").write(ver + ver_new)
print("appended:", VER)

res, res_new = open(RES).read(), open(RES_NEW).read()
if "SESSION 2026-09-13/15" in res:
    sys.exit("RESUME-HERE.md already has the session block")
anchor = "All work is committed. Tree is clean apart from your own `CLAUDE.md` edit and untracked\n`ppl.txt`.\n"
if anchor not in res:
    sys.exit("could not find the insertion anchor in " + RES)
# the new block opens with its own "all work is committed" line, so drop the old one
res = res.replace(anchor, res_new.rstrip() + "\n", 1)
first_nl = res.index("\n") + 1
res = TITLE + res[first_nl:]
open(RES, "w").write(res)
print("inserted:", RES)
