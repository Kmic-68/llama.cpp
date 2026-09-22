# Engineering record

Supporting material behind `../p100-docs/` and `../OPTLOG.md`. The current state and next steps
are in `../HANDOFF.md`.

| path | what it is |
|---|---|
| `ppl-orig.txt` | **the perplexity gate corpus.** The 2.6209 ± 0.0199 band belongs to this file. `../tools/gate.sh` uses it |
| `CORPUS.md` | why `../ppl.txt` reads 2.7566 on every build, and how the corpora differ |
| `VERIFICATION.md` | the numerical audit of the kernel changes: what's bit-exact, and what's merely equal |
| `tools/` | measurement scripts (`MEASURE.md`), the server sweep, and the bit-exactness proofs |
| `bench/` | MTP and small-batch benchmark scripts |
| `release-sync/` | `refresh-build.sh` and `sync.sh`, which build the release bundle on `/mnt/fast` |
| `lcb*.json`, `lcb-report.html` | LiveCodeBench quality runs. `lcb_greedy_BROKEN.json` is a known-bad run kept for comparison |
| `wip-2026-09-13/`, `wip-2026-09-14/` | raw logs and harnesses from the race hunts (OPTLOG attempts 151-153) |

Older session diffs and handoff notes were removed in the 2026-09-22 cleanup. They're in git
history before that commit.
