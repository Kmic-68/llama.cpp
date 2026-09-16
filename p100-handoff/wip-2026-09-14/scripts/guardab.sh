#!/usr/bin/env bash
# One binary, one line toggled: does the same-GPU copy guard itself make 3 virtual devices nan?
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
I=$S/snap-inject2/bin
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
ulimit -c 0
mkdir -p $S/gab
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
g() { local name=$1; shift
  env "$@" GGML_CUDA_DEVICES=3 GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$I \
      $I/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/gab/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/gab/$name.log)]  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/gab/$name.log) errors ($(date +%T))"; }
echo "=== (G) same-GPU guard on/off, 3 virtual devices, no delay ($(date +%T))"
for i in 1 2 3 4; do
  g on_$i  GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1
  g off_$i GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=0
done
echo "=== (G) done ($(date +%T))"
