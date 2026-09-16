#!/usr/bin/env bash
# Is GGML_CUDA_DEVICES=4 nan because of a race, or is virtual-device emulation broken on its own?
# Shipping build, no knobs, no injected delay. 2 chunks each.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=$S/snap-final3/bin
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
MTP=/mnt/fast/models/Qwen3.8-27B-MTP-ONLY-Q6_K.gguf
ulimit -c 0
mkdir -p $S/probe
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
p() { local name=$1 mdl=$2; shift 2
  env "$@" GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$B \
      $B/llama-perplexity -m $mdl -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/probe/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/probe/$name.log)]  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/probe/$name.log) errors  $(grep -oE 'emulating [0-9]+ virtual device\(s\) on [0-9]+' $S/probe/$name.log | head -1) ($(date +%T))"; }
echo "=== (P) virtual-device probes, shipping build ($(date +%T))"
p p0_2dev_ctl   $M                          GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
p p2_3dev       $M   GGML_CUDA_DEVICES=3    GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
p p3_4dev_fp16  $M   GGML_CUDA_DEVICES=4
p p4_4dev_tile  $M   GGML_CUDA_DEVICES=4    GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM=0
p p5_mtp_2dev   $MTP GGML_CUDA_DEVICES=2    GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
p p6_mtp_1card  $MTP GGML_CUDA_DEVICES=2    GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 CUDA_VISIBLE_DEVICES=1
echo "=== (P) done ($(date +%T))"
