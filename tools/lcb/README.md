# LiveCodeBench v6 harness (P100 A/B + absolute check)

Why this and not HumanEval: HumanEval (2021, 164 problems) is in every training
corpus and saturated — useful only as a "nothing is broken" detector. Qwen3.8-27B
publishes **LiveCodeBench v6 = 90.3**, so that is the comparable number.

Data: `test6.jsonl` from `livecodebench/code_generation_lite` (175 problems,
contests 2025-01-04 .. 2025-04-06; 112 AtCoder stdin + 63 LeetCode functional;
43 easy / 52 medium / 80 hard). The HF dataset viewer refuses this repo (loading
script), and Python 3.14 has no `datasets`/`pyarrow` wheels here, so fetch the
raw file:

    curl -sL -o lcb_v6.jsonl \
      https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl

Run (server must already be up — see ../humaneval/serve.sh, but use -c 24576
--parallel 1: reasoning traces on hard problems exceed 14k tokens):

    python3 tools/lcb/bench.py --port 8080 --data lcb_v6.jsonl \
      --out lcb_head.json --n 40 --seed 1234 --max-tokens 22000

## Cost, measured
~10 min/problem: hard problems generate 10-20k reasoning tokens at ~24 t/s
(long-context rate, not the 31 t/s of tg256). 40 problems ~= 6-7 h per arm.
Budget accordingly; this does not fit in a 2-hour window.

## Traps
- **bwrap bind targets must land in the tmpfs.** `--ro-bind / /` makes the root
  read-only, so bwrap cannot create a mount point at `/payload.json`. Bind to
  `/tmp/payload.json` *after* `--tmpfs /tmp`. Same bug in a different dress cost
  the HumanEval run: `--tmpfs /tmp` masks any file written to a scratch dir that
  lives under /tmp.
- **Validate the judge without the model.** Feed it a hand-written correct
  solution and a deliberately wrong one; expect 40/40 and 0/40. Both harness
  bugs above showed up as 0% pass rates that looked like model failures.
- `test6.jsonl` is the newest v6 slice and skews hard (46% hard), so a score
  here is a lower bound relative to the published full-v6 90.3.
- Judge is all-or-nothing over 40 tests/problem, matching LCB.
