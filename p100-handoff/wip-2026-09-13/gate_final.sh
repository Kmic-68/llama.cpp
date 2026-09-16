#!/usr/bin/env bash
# The CLAUDE.md gates on the final build: tg256 on cool cards, perplexity against ppl-orig.txt
# (band 2.6209 +/- 0.0199), FLASH_ATTN_EXT eval, then the full op suite (log kept in scratch, not /tmp).
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad/mine
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
echo "library $(sha256sum build-opt/bin/libggml-cuda.so | cut -c1-16) ($(date +%T))"
echo "stray llama processes: $(pgrep -f 'build-opt/bin/llama-' | wc -l)"
until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le 50 ]; do sleep 10; done
echo "== tg256 (baseline 17.51) temps $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | paste -sd/)"
ulimit -c 0
GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5 2>&1 | grep -E "tg256"
echo "== perplexity gate (2.6209 +/- 0.0199, ppl-orig.txt) ($(date +%T))"
./build-opt/bin/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 2>&1 | grep -E "Final estimate"
echo "== FLASH_ATTN_EXT eval ($(date +%T))"
./build-opt/bin/test-backend-ops test -o FLASH_ATTN_EXT 2>&1 | grep -E "backends passed|FAIL" | head -5
echo "== full op suite ($(date +%T))"
./build-opt/bin/test-backend-ops test > $S/gate-ops-full.log 2>&1; echo "   exit=$?"
grep -E "backends passed|FAIL" $S/gate-ops-full.log | head -10
echo "== done ($(date +%T))"
