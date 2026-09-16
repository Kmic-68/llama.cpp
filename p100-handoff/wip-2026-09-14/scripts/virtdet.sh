#!/usr/bin/env bash
# Is GGML_CUDA_DEVICES (>2 virtual devices) deterministic at all, and is that new?
# Same command repeated; two physical devices as the control; the 2026-09-06 release binaries
# (none of this session's fixes) as the "was it always like this" arm. No delay injection, no knobs.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=$S/snap-final3/bin
R=/mnt/fast/p100-llamacpp-release/build
M=/mnt/fast/models/Qwen3.8-27B-Q4_0.gguf
ulimit -c 0
mkdir -p $S/vd
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
r() { local name=$1 bin=$2; shift 2
  env "$@" GGML_CUDA_P2P=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$bin \
      $bin/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 2048 -ub 512 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/vd/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/vd/$name.log)]  $(grep -ciE 'cuda error|out of memory|GGML_ASSERT' $S/vd/$name.log) errors ($(date +%T))"; }
echo "=== (D) repeated identical runs ($(date +%T))"
for i in 1 2 3; do r new_2dev_$i  $B; done
for i in 1 2 3; do r new_3dev_$i  $B GGML_CUDA_DEVICES=3; done
for i in 1 2 3; do r old_3dev_$i  $R GGML_CUDA_DEVICES=3; done
for i in 1 2;   do r old_2dev_$i  $R; done
echo "=== (D) done ($(date +%T))"
