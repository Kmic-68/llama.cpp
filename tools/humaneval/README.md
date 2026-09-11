# HumanEval pass@1 harness (paired stock-vs-HEAD)

Purpose: answer "did the optimization work damage coding ability" by running the
SAME 164 problems on two builds and comparing per-problem outcomes. The paired
comparison is the point -- it cancels the abliteration, harness and quantization
offsets that make comparison against a published score meaningless.

## Status: set up, NOT yet run (paused 2026-09-11)

Done: dataset verified (164 problems), harness written, llama-server on HEAD
loads in ~6 s and answers /health. Smoke test was interrupted before running.

## Run it

    # dataset (not committed, ~214 KB)
    curl -sL https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
      | gunzip > tools/humaneval/HumanEval.jsonl

    # 1. smoke test 3 problems first -- confirms code extraction and token budget
    ./tools/humaneval/serve.sh head 8080 &        # wait for {"status":"ok"} on /health
    python3 tools/humaneval/bench.py --port 8080 --out /tmp/smoke.json --limit 3

    # 2. full run, each arm ~40 min at 31 t/s (ESTIMATE, not yet measured)
    python3 tools/humaneval/bench.py --port 8080 --out head.json
    # then kill the server, `serve.sh stock 8080`, and repeat into stock.json

## Notes / traps
- serve.sh routes through each build's own libs (the RUNPATH trap: see
  tools/runbuild.sh). `head` = build-opt; `stock`/`prefix` = /mnt/fast snapshots.
- Generated code runs under bwrap: --unshare-net, tmpfs /tmp, 20 s timeout.
  Never run these completions unsandboxed.
- temperature 0. The chat template emits reasoning; bench.py strips <think>...</think>.
- /mnt/fast was mounted READ-ONLY (fuseblk ro) on 2026-09-11, so scratch output
  must go to / or the scratchpad. Snapshots there are still readable.
- Compare with McNemar on discordant pairs, not by eyeballing the two percentages.
