#!/usr/bin/env bash
# The same-physical-device copy branch, demonstrated at THREE virtual devices on two P100s.
# Round-robin puts virtual 0 and 2 on GPU0 and virtual 1 on GPU1, so the non-power-of-2 fold
# (push_data 2->0) and the copy-back (0->2) are same-GPU copies while 0<->1 is a peer copy.
# Delaying virtual device 0's all-reduce ADD lets device 2's copy into device 0's reduction
# buffer overtake the reader. Four virtual devices are not usable as an instrument: they hit an
# independent nan (see probe_virt.sh).
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
I=$S/snap-inject2/bin
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
ulimit -c 0
mkdir -p $S/sd3
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
t() { local name=$1; shift
  env "$@" GGML_CUDA_TEMP_DELAY_N=2500 GGML_CUDA_TEMP_DELAY_REPS=4 GGML_CUDA_DEVICES=3 GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$I \
      $I/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/sd3/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/sd3/$name.log)]  $(grep 'ADD graphs delayed' $S/sd3/$name.log | tail -1)  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/sd3/$name.log) errors ($(date +%T))"; }
echo "=== (S3) 3 virtual devices on 2 GPUs, Q4_0, -ub 512, fp32 matmuls, 2 chunks ($(date +%T))"
t t1_nodelay_guard   GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1
t t2_nodelay_noguard GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=0
t t3_delay0_guard    GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=0
t t4_delay0_noguard  GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=0


t t7_delay0_noguard_b GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=0
t t8_delay0_guard_b  GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=0
echo "=== (S3) done ($(date +%T))"
