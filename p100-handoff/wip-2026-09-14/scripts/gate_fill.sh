#!/usr/bin/env bash
# Full gate on the build that adds GGML_OP_FILL for disabled slices (snap-fill).
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=$S/snap-fill/bin
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
mkdir -p $S/gatefill
ulimit -c 0
cool() { local lim=${1:-46}; until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le $lim ]; do sleep 10; done; }
echo "=== gate_fill $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16) ($(date +%T))"
echo "=== (1) tg256 on cool cards, -r 5 (baseline 17.51) ($(date +%T))"
cool 45
GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$B $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5 2>&1 | grep -E "tg256|error"
echo "=== (2) does a 2-device run ever hit the zero-slice path? ($(date +%T))"
LD_LIBRARY_PATH=$B $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -sm tensor -ngl 99 -c 4096 \
    -ctk q4_0 -ctv q4_0 --chunks 2 -v > $S/gatefill/v2dev_q6.log 2>&1
echo "   zero-slice lines: $(grep -c 'zero-sized slice' $S/gatefill/v2dev_q6.log)  ($(date +%T))"
echo "=== (3) perplexity gate, ppl-orig.txt -c 4096 (2.6209 +/- 0.0199) ($(date +%T))"
LD_LIBRARY_PATH=$B $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -sm tensor -ngl 99 -c 4096 \
    -ctk q4_0 -ctv q4_0 2>&1 | tee $S/gatefill/ppl.log | grep -E "Final estimate|error"
echo "=== (4) FLASH_ATTN_EXT eval ($(date +%T))"
LD_LIBRARY_PATH=$B $B/test-backend-ops test -o FLASH_ATTN_EXT > $S/gatefill/fa.log 2>&1; echo "   exit=$?"
grep -E 'tests passed|backends passed|FAIL' $S/gatefill/fa.log | head -4
echo "=== (5) full op suite ($(date +%T))"
LD_LIBRARY_PATH=$B $B/test-backend-ops test > $S/gatefill/full.log 2>&1; echo "   exit=$?"
grep -E 'tests passed|backends passed|FAIL' $S/gatefill/full.log | head -6
echo "=== gate_fill done ($(date +%T))"
