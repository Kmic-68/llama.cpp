#!/bin/bash
cd /home/kaden/llama-opt
echo -n "  REG(n=4): "; cuobjdump -res-usage build-opt/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o 2>/dev/null | grep -A1 "_Z13mul_mat_vec_qIL9ggml_type14ELi4E" | grep -o "REG:[0-9]* STACK:[0-9]*" | tr '\n' ' '
GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 512 -n 0 -b 4 -ub 4 -r 2 2>&1 | grep -oE "pp512 *\| *[0-9.]* ± *[0-9.]*"
