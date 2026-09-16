#!/usr/bin/env bash
# (I) The same-physical-device copy path, retried on the Q4_0 model so four virtual devices fit.
# GGML_CUDA_DEVICES=4 puts two virtual devices on each GPU, so each butterfly step has one same-GPU
# exchange (0<->2, 1<->3) and one peer exchange (0<->1, 2<->3). Delay virtual device 2's all-reduce
# ADD and see which guard matters.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
M=${VIRT_MODEL:-/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf}
I=$S/snap-inject2/bin
UB=${VIRT_UB:-2048}
ulimit -c 0
mkdir -p $S/inject
until grep -q '=== queue_final3 done' $S/queue_final3.log 2>/dev/null || ! ps -o pid= -p ${QPID:-1} >/dev/null 2>&1; do sleep 20; done
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
echo "=== (I2) 4 virtual devices on 2 GPUs, $(basename $M), -ub $UB, fp32 matmuls, P2P, 2 chunks ($(date +%T))"
v() { local name=$1; shift
  env "$@" GGML_CUDA_DEVICES=4 GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$I \
      $I/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub $UB --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/inject/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/inject/$name.log)]  $(grep 'ADD graphs delayed' $S/inject/$name.log | tail -1)  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/inject/$name.log) errors ($(date +%T))"; }
v v2_nodelay_both     GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1
v v2_delay_both       GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=2
v v2_delay_nosamedev  GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=2
v v2_delay_nopeer     GGML_CUDA_PEER_WAIT_DST=0 GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=2
v v2_delay_neither    GGML_CUDA_PEER_WAIT_DST=0 GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=2
echo "=== (I2) done ($(date +%T))"
