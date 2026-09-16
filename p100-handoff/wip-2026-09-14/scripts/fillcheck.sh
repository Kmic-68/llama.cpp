#!/usr/bin/env bash
# Does zero-filling the disabled slice (GGML_OP_FILL instead of SCALE by 0) remove the nan at
# 3 and 4 virtual devices? The first run is verbose, to see whether a zero-sized slice occurs at all.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=$S/snap-fill/bin
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
ulimit -c 0
mkdir -p $S/fc
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
f() { local name=$1 dev=$2 verbose=$3
  local extra=(); [ "$verbose" = "v" ] && extra=(-v)
  env GGML_CUDA_DEVICES=$dev GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$B \
      $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 "${extra[@]}" > $S/fc/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/fc/$name.log)]  $(grep -c 'zero-sized slice' $S/fc/$name.log) zero-slice lines  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/fc/$name.log) errors ($(date +%T))"; }
echo "=== (F) with GGML_OP_FILL ($(date +%T))"
f v3dev 3 v
for i in 1 2 3 4; do f f3dev_$i 3; done
f v4dev 4 v
for i in 1 2 3;   do f f4dev_$i 4; done
for i in 1 2;     do f f2dev_$i 2; done
echo "=== (F) done ($(date +%T))"
