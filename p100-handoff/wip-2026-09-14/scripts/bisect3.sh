#!/usr/bin/env bash
# Which build started reading nan at 3 virtual devices? Old release = clean 3/3, final3 = 2/3 nan.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
ulimit -c 0
mkdir -p $S/bi
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
b() { local name=$1 bin=$2; shift 2
  env "$@" GGML_CUDA_DEVICES=3 GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$bin \
      $bin/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/bi/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/bi/$name.log)]  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/bi/$name.log) errors ($(date +%T))"; }
echo "=== (B) which build, 3 virtual devices, fp32 matmuls ($(date +%T))"
for i in 1 2 3; do b vc2_$i    $S/snap-vc2/bin;    done
for i in 1 2 3; do b final2_$i $S/snap-final2/bin; done
echo "=== (B) done ($(date +%T))"
