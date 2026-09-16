#!/usr/bin/env bash
# No-model GPU work while /mnt/fast is unmounted: PV algorithm timing and the upstream kernel race stress.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=$S/snap-peerfix2/bin
ulimit -c 0
echo "=== (R3) PV algorithm timing, round-robin, GPU1 ($(date +%T))"
CUDA_VISIBLE_DEVICES=1 $S/pv/pvalgo
echo "=== (R4) upstream kernel race stress, 100 iterations each, GPU1 ($(date +%T))"
for c in softmax_oop softmax_ip groupnorm_oop rmsnorm_ip norm_ip; do
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$B timeout 900 $S/stress/stress $c 100 2>&1 | grep -vE '^(ggml_cuda_init|  Device|load_backend)' | tail -12
done
echo "=== queue_nomodel done ($(date +%T))"
