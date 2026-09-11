#!/usr/bin/env bash
# serve.sh <head|stock|prefix> <port>
set -uo pipefail
L=$1; P=$2
if [ "$L" = "head" ]; then BIN=/home/kaden/llama-opt/build-opt/bin; LD=$BIN
else BIN=/mnt/fast/p100-scratch/build-$L/bin; LD=$BIN; fi
exec env GGML_CUDA_P2P=1 LD_LIBRARY_PATH="$LD" "$BIN/llama-server" \
  -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
  -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -c 8192 \
  --host 127.0.0.1 --port "$P" --no-warmup
