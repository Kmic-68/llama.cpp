#!/usr/bin/env bash
# The same-physical-device copy branch, isolated: two virtual devices on ONE card, so every
# tensor-parallel exchange takes that branch and none takes the peer branch. Q4_0 fits on one
# P100 at -ub 512. fp32 matmuls => every exchange is uncompressed, the vulnerable path.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
I=$S/snap-inject2/bin
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
ulimit -c 0
mkdir -p $S/samedev
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
s() { local name=$1; shift
  env "$@" CUDA_VISIBLE_DEVICES=1 GGML_CUDA_DEVICES=2 GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$I \
      $I/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/samedev/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/samedev/$name.log)]  $(grep 'ADD graphs delayed' $S/samedev/$name.log | tail -1)  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/samedev/$name.log) errors ($(date +%T))"; }
echo "=== (S) 2 virtual devices on ONE card, $(basename $M), -ub 512, fp32 matmuls, 2 chunks ($(date +%T))"
s s1_nodelay_guard   GGML_CUDA_SAMEDEV_WAIT_DST=1
s s2_nodelay_noguard GGML_CUDA_SAMEDEV_WAIT_DST=0
s s3_delay_guard     GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
s s4_delay_noguard   GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
s s5_delay_noguard_b GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
s s6_delay_guard_b   GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
echo "=== (S) done ($(date +%T))"
