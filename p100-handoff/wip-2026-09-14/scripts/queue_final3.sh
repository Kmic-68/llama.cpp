#!/usr/bin/env bash
# The true final build (adds the same-GPU virtual-device copy guard): gate, op suites, the
# virtual-device race test, and the long in-place kernel stress.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
B=$S/snap-final3/bin
I=$S/snap-inject2/bin
F=$S/final
mkdir -p $S/inject
waitmodel() { until [ -r $M ]; do sleep 5; done; }
cool() { local lim=${1:-48} t0=$(date +%s); until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le $lim ]; do sleep 10; done
         echo "   (cooled ${lim}C in $(( $(date +%s) - t0 ))s)"; }
ulimit -c 0
echo "=== final3 $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16), inject2 $(sha256sum $I/libggml-cuda.so.0.21.0 | cut -c1-16) ($(date +%T))"

echo "=== (J1) tg256 on cool cards, -r 5 (baseline 17.51) ($(date +%T))"
waitmodel; cool 45
GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$B $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5 2>&1 | grep -E "tg256|error"
echo "=== (J2) perplexity gate, ppl-orig.txt -c 4096 (2.6209 +/- 0.0199) ($(date +%T))"
waitmodel
LD_LIBRARY_PATH=$B $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 2>&1 | grep -E "Final estimate|error"

echo "=== (K) what the precise modes cost, final3: fp32 matmuls and fp32 GEMM attention ($(date +%T))"
k() { local tag=$1; shift; waitmodel; cool 46
  env "$@" GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$B $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 "${BENCH[@]}" 2>&1 \
    | grep -oE "(pp[0-9]+( @ d[0-9]+)?|tg[0-9]+) +\| +[0-9.]+ ± [0-9.]+" | sed "s/^/   $tag: /"; }
BENCH=(-p 512,2048 -n 0 -b 2048 -ub 2048 -r 2)
k "default (fp16 matmul, fp16 attention)"
k "fp32 matmuls"                          GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
k "fp32 attention"                        GGML_CUDA_FA_GEMM_PREC=32
k "fp32 matmuls + fp32 attention"         GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM_PREC=32
BENCH=(-p 0 -n 256 -r 3)
k "decode, default"
k "decode, fp32 matmuls + fp32 attention" GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM_PREC=32

echo "=== (I) 4 virtual devices on 2 GPUs: the same-GPU copy path, fp32 matmuls, P2P, 2 chunks ($(date +%T))"
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
v() { local name=$1; shift; waitmodel
  env "$@" GGML_CUDA_DEVICES=4 GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$I \
      $I/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub 2048 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/inject/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/inject/$name.log)]  $(grep 'ADD graphs delayed' $S/inject/$name.log | tail -1)  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/inject/$name.log) errors ($(date +%T))"; }
v virt_nodelay_both     GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1
v virt_delay_both       GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=2
v virt_delay_nosamedev  GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=2
v virt_delay_nopeer     GGML_CUDA_PEER_WAIT_DST=0 GGML_CUDA_SAMEDEV_WAIT_DST=1 GGML_CUDA_TEMP_DELAY_ADD_DEV=2
v virt_delay_neither    GGML_CUDA_PEER_WAIT_DST=0 GGML_CUDA_SAMEDEV_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=2

echo "=== (H) in-place upstream kernels, 2000 launches each, GPU1, in parallel with the op suites ($(date +%T))"
( for c in softmax_ip rmsnorm_ip norm_ip; do
    CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$B timeout 1800 $S/stress/stress $c 2000 2>&1 | grep -E '^case|^  iter' | tail -12
  done; echo "=== (H) done ($(date +%T))" ) > $S/stress2000.log 2>&1 &
STRESS_PID=$!

echo "=== (J3) FLASH_ATTN_EXT eval and full op suite on final3 ($(date +%T))"
LD_LIBRARY_PATH=$B $B/test-backend-ops test -o FLASH_ATTN_EXT > $F/fa3.log 2>&1; echo "   FA exit=$? ($(date +%T))"; grep -E 'tests passed|backends passed|FAIL' $F/fa3.log | head -6
LD_LIBRARY_PATH=$B $B/test-backend-ops test > $F/full3.log 2>&1; echo "   full exit=$? ($(date +%T))"; grep -E 'tests passed|backends passed|FAIL' $F/full3.log | head -8

wait $STRESS_PID 2>/dev/null
cat $S/stress2000.log
echo "=== queue_final3 done ($(date +%T))"
