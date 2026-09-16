#!/usr/bin/env bash
# Final build, no model needed: FLASH_ATTN_EXT eval + full op suite, then a longer race stress of the in-place upstream kernels.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
F=$S/final; B=$S/snap-final/bin
ulimit -c 0
export LD_LIBRARY_PATH=$B
echo "=== (F6) final build $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16): FLASH_ATTN_EXT eval and full op suite ($(date +%T))"
$B/test-backend-ops test -o FLASH_ATTN_EXT > $F/fa.log 2>&1; echo "   FA exit=$? ($(date +%T))"; grep -E 'tests passed|backends passed|FAIL' $F/fa.log | head -6
$B/test-backend-ops test > $F/full.log 2>&1; echo "   full exit=$? ($(date +%T))"; grep -E 'tests passed|backends passed|FAIL' $F/full.log | head -8
echo "=== (F6) done ($(date +%T))"
echo "=== (R4b) in-place upstream kernels, 2000 iterations each, GPU1 ($(date +%T))"
for c in softmax_ip rmsnorm_ip norm_ip; do
  CUDA_VISIBLE_DEVICES=1 timeout 1800 $S/stress/stress $c 2000 2>&1 | grep -E '^case|^  iter' | tail -12
done
echo "=== queue_f6 done ($(date +%T))"
