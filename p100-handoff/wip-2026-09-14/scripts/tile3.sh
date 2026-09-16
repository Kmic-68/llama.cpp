#!/usr/bin/env bash
# Does the 3-virtual-device nan follow the GEMM attention path? Same build, tile kernel instead.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=$S/snap-fill/bin
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
ulimit -c 0
mkdir -p $S/tl
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
t() { local name=$1; shift
  env "$@" GGML_CUDA_DEVICES=3 GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$B \
      $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/tl/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/tl/$name.log)]  $(grep -ciE 'cuda error|out of memory' $S/tl/$name.log) errors ($(date +%T))"; }
echo "=== (T) 3 virtual devices, tile attention vs GEMM attention, fill build ($(date +%T))"
for i in 1 2 3 4 5; do t tile_$i GGML_CUDA_FA_GEMM=0; done
for i in 1 2 3;     do t gemm_$i; done
echo "=== (T) done ($(date +%T))"
bash $S/gate_fill.sh
