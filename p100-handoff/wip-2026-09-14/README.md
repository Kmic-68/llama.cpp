Working material for OPTLOG attempts 151-153 (2026-09-12 to 09-16). The prose drafts that lived
here have been merged and deleted; they are now:

- `OPTLOG.md`, attempts 151, 152 and 153
- `p100-handoff/VERIFICATION.md`, the 2026-09-15/16 update
- `p100-handoff/RESUME-HERE.md`, the session block at the top
- `p100-handoff/release-sync/docs/AUDIT-2026-09-12.md`, "Found after the audit"

What is left is what the prose refers to:

- `scripts/` — the run queues and the three scripts that assembled those documents
  (`assemble_optlog.sh`, `apply_audit.py`, `apply_handoff.py`). They will refuse to run twice.
- `harness/` — standalone C/C++ probes: cuBLAS algorithm sweeps (`matmul2.c`, `pvalgo.c`,
  `sgemm_time.c`), and `stress.cpp`, which runs one ggml op repeatedly on identical input and
  compares every launch bit for bit (the in-place kernel stress in §6).
- `logs/` — the raw output behind the tables, `logs/0916/` being the virtual-device work of §8c.
- `code/` — the two temporary patches that reproduce the race experiments, and the peer-copy fix
  variant that was measured and rejected. See `code/README.md`.

The binaries these were run against are not kept; rebuild from the commit named in each patch
header, into `build-opt`, and copy `build-opt/bin` to a snapshot directory before switching away.
