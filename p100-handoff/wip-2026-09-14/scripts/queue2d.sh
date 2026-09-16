#!/usr/bin/env bash
# Second GPU queue (rewritten): the hash hook now prints to stderr (llama's log callback drops ggml INFO,
# so the first node-hash runs captured nothing). Determinism on the OOP build, in-place vs OOP
# bit-identity, OOP speed, FA eval.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
A=$S/snap-ipnh/bin; O=$S/snap-oopnh/bin
waitmodel() { until [ -r $M ]; do sleep 5; done; }
cool() { local lim=${1:-48} t0=$(date +%s); until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le $lim ]; do sleep 10; done
         echo "   (cooled ${lim}C in $(( $(date +%s) - t0 ))s: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | paste -sd/))"; }
ulimit -c 0
echo "=== (P) peer-copy ordering fix: event A/B (fp32 matmuls, no P2P) and decode speed ($(date +%T))"
bash $S/nan/nanhash.sh
echo "=== (6) speed: pp2048 @ d16384, in-place vs OOP, 2 rounds ($(date +%T))"
for v in inplace oop oop inplace; do
  waitmodel; cool 48
  if [ $v = inplace ]; then L=$A; else L=$O; fi
  LD_LIBRARY_PATH=$L GGML_CUDA_P2P=1 $L/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -b 2048 -p 2048 -n 0 -d 16384 -r 2 2>&1 \
    | grep -oE "pp2048 @ d[0-9]+ +\| +[0-9.]+ ± [0-9.]+|error.*" | sed "s/^/   $v: /"
done
echo "=== (7) FLASH_ATTN_EXT eval on the OOP build ($(date +%T))"
LD_LIBRARY_PATH=$O $O/test-backend-ops test -o FLASH_ATTN_EXT > $S/oop/fa.log 2>&1; echo "   exit=$?"
grep -E 'tests passed|backends passed|FAIL' $S/oop/fa.log | head -6
echo "=== queue2 done ($(date +%T))"
