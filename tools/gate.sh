#!/usr/bin/env bash
# Run the correctness + metric gates with the RIGHT inputs.
#
# Why this exists: the 2.6209 +/- 0.0199 perplexity band belongs to one corpus,
# p100-handoff/ppl-orig.txt. ./ppl.txt is a different file and reads 2.7566 on
# ANY build, stock included, and two sessions reverted good work after gating
# on it by hand. Keeping the corpus next to the number, in a script, stops that.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf}
# the repo keeps the corpus in p100-handoff/; the release bundle ships it beside this script
if   [ -f p100-handoff/ppl-orig.txt ];           then CORPUS=p100-handoff/ppl-orig.txt
else CORPUS=tools/perplexity-gate-corpus.txt; fi
# repo layout uses build-opt/bin; the release package ships binaries in build/
if   [ -x ./build-opt/bin/llama-perplexity ]; then BIN=./build-opt/bin
elif [ -x ./build/llama-perplexity ];        then BIN=./build
else echo "no llama-perplexity found in ./build-opt/bin or ./build" >&2; exit 2; fi
# The binaries' RUNPATH points at the tree they were built in; without this, a copied build
# (the release bundle included) silently runs build-opt's libggml-cuda instead of its own.
export LD_LIBRARY_PATH="$(cd "$BIN" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ ! -f "$CORPUS" ]; then echo "missing gate corpus: $CORPUS" >&2; exit 2; fi

stray=$(pgrep -f "$BIN/llama-" | wc -l)
if [ "$stray" -gt 0 ]; then
    echo "WARNING: $stray llama process(es) already running; they will skew both gates."
    pgrep -af "$BIN/llama-"
fi

# The metric runs FIRST, on cold cards. This is not cosmetic: the same build measures
# 30.75 +/- 0.19 cold and 24.9 +/- 2.6 straight after a perplexity run. A hot-card reading
# looks exactly like a 20% regression.
echo "== tg256 (upstream at the fork point: 17.51) =="
nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader | sed 's/^/   GPU temp before: /'
ulimit -c 0
GGML_CUDA_P2P=1 "$BIN/llama-bench" -m "$MODEL" \
    -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5 2>&1 | grep -E "tg256"

echo "== perplexity (gate: 2.6209 +/- 0.0199 against $CORPUS) =="
ulimit -c 0
"$BIN/llama-perplexity" -m "$MODEL" -f "$CORPUS" \
    -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 2>&1 | grep -E "Final estimate"

# NOTE: this is FLASH_ATTN_EXT ONLY, roughly 1% of the op suite. Its "3/3 backends" has
# been quoted in handoffs as if it meant the whole suite passed. It does not. The full
# suite (~25 min, ~16k tests) caught an intermittent CUDA1 failure that this never would.
# Run `./tools/gate.sh --full` before claiming a build is clean.
if [ "${1:-}" = "--full" ]; then
    echo "== FULL op suite (~16k tests, ~25 min) =="
    "$BIN/test-backend-ops" test 2>&1 | tee /tmp/gate-ops-full.log | grep -E "backends passed|FAIL"
    echo "   (full log: /tmp/gate-ops-full.log)"
else
    echo "== flash-attn eval ONLY (-o FLASH_ATTN_EXT; NOT the full suite) =="
    "$BIN/test-backend-ops" test -o FLASH_ATTN_EXT 2>&1 | grep -E "backends passed|FAIL"
    echo "   run './tools/gate.sh --full' for the whole suite"
fi
