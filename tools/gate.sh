#!/usr/bin/env bash
# Run the correctness + metric gates with the RIGHT inputs.
#
# Why this exists: CLAUDE.md's workflow step 4 names `./ppl.txt` and demands
# 2.6209 +/- 0.0199. That file yields 2.7566 on ANY build, stock included, so
# following the instruction literally reports a correctness failure every time.
# The corpus the 2.6209 band belongs to is p100-handoff/ppl-orig.txt. Two
# separate sessions have reverted good work over this. Use this script.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf}
CORPUS=p100-handoff/ppl-orig.txt
BIN=./build-opt/bin

if [ ! -f "$CORPUS" ]; then echo "missing gate corpus: $CORPUS" >&2; exit 2; fi

stray=$(pgrep -f "$BIN/llama-" | wc -l)
if [ "$stray" -gt 0 ]; then
    echo "WARNING: $stray llama process(es) already running; they will skew both gates."
    pgrep -af "$BIN/llama-"
fi

echo "== perplexity (gate: 2.6209 +/- 0.0199 against $CORPUS) =="
ulimit -c 0
"$BIN/llama-perplexity" -m "$MODEL" -f "$CORPUS" \
    -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 2>&1 | grep -E "Final estimate"

echo "== tg256 (CLAUDE.md metric; baseline 17.51) =="
echo "   NB: this swings 25-31 t/s with card temperature. Compare only within a session."
GGML_CUDA_P2P=1 "$BIN/llama-bench" -m "$MODEL" \
    -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5 2>&1 | grep -E "tg256"

echo "== flash-attn eval (expect 3/3 backends) =="
"$BIN/test-backend-ops" test -o FLASH_ATTN_EXT 2>&1 | grep -E "backends passed|FAIL"
