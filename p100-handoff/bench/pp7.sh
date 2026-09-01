#!/bin/bash
cd /home/kaden/llama-opt
L=/tmp/pp_$1.txt
GGML_CUDA_P2P=1 timeout 900 nvprof --print-gpu-summary --log-file $L ./build-opt/bin/llama-bench \
  -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
  -p 512 -n 0 -b 7 -ub 7 -r 2 2>&1 | grep -oE "pp512 *\| *[0-9.]* ± *[0-9.]*"
grep -oE "[0-9.]+%[^v]*void mul_mat_vec_q<ggml_type=14, int=7" $L | head -1
