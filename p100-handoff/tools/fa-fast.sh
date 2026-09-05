#!/bin/bash
# fast FA preset: only kv=262144 q4_0 shapes. ~7s per shape instead of ~7min for the suite.
cd /home/kaden/llama-opt
for nb in 1 4 6; do
  v=$(CUDA_VISIBLE_DEVICES=1 ./build-opt/bin/test-backend-ops perf -o FLASH_ATTN_EXT \
        -p "kv=262144,nb=${nb},.*type_K=q4_0" 2>&1 \
      | sed 's/\x1b\[[0-9;]*m//g' | grep -ao "[0-9.]\+ us/run" | head -1 | grep -o "[0-9.]\+")
  printf "nb=%s:%s " "$nb" "${v:-FAIL}"
done
echo
