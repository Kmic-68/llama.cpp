#!/bin/bash
cd /home/kaden/llama-opt
R=$(GGML_CUDA_P2P=1 timeout 900 ./build-opt/bin/llama-speculative-simple \
  -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf --spec-type draft-mtp \
  --spec-draft-n-max $1 --spec-draft-p-min $2 -ngld 99 -p "$3" \
  -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 2>&1)
S=$(echo "$R" | grep -oE "decoded +[0-9]+ tokens in +[0-9.]+ seconds, speed: +[0-9.]+ t/s" | grep -oE "[0-9.]+ t/s")
A=$(echo "$R" | grep -oE "accept +=  *[0-9.]+%")
echo "  [$4] n-max=$1 p-min=$2 -> ${S}  ${A}"
