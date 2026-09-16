#!/usr/bin/env bash
# After queue2d: the non-serialising peer-fix variant, decode speed of all three, PV timing, stress.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
B=$S/snap-peerfix2/bin
waitmodel() { until [ -r $M ]; do sleep 5; done; }
cool() { local lim=${1:-48}; until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le $lim ]; do sleep 10; done; }
ulimit -c 0
until grep -q '=== queue2 done' $S/queue2.log 2>/dev/null; do sleep 30; done
export LD_LIBRARY_PATH=$B
echo "=== (R1) peer fix, variant 2 (wait on the destination's existing work marker): event runs ($(date +%T)), lib $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16)"
for r in 1 2 3 4 5 6; do
  waitmodel
  GGML_CUDA_PEER_WAIT_DST=2 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub 2048 --chunks 15 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/nan/p_w2_r$r.log 2>&1
  echo "   wait_dst=2 run $r: $(grep -cE '^ +[0-9]+ +nan' $S/nan/p_w2_r$r.log) nan rows, last: $(grep -E '^ +[0-9]+ +' $S/nan/p_w2_r$r.log | tail -1) ($(date +%T))"
done
python3 - "$S" <<'PY'
import re, sys, collections, glob, os
S = sys.argv[1]
def rows(f):
    return [m.group(3) for m in (re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", l) for l in open(f)) if m]
runs = {os.path.basename(f)[2:-4]: rows(f) for f in sorted(glob.glob(f"{S}/nan/p_w[012]_r*.log"))}
n = min(len(v) for v in runs.values())
maj = [collections.Counter(v[i] for v in runs.values()).most_common(1)[0][0] for i in range(n)]
for w in ("0", "1", "2"):
    ks = sorted(k for k in runs if k.startswith(w + "_"))
    firsts = [next((i + 1 for i in range(n) if runs[k][i] != maj[i]), None) for k in ks]
    print(f"   wait_dst={w}: {sum(f is not None for f in firsts)} of {len(ks)} runs with an event; first divergent chunk {firsts}")
PY
echo "=== (R2) decode speed, production flags (P2P): off / record+wait / existing marker, interleaved ($(date +%T))"
for w in 0 1 2 2 1 0; do
  waitmodel; cool 46
  GGML_CUDA_PEER_WAIT_DST=$w GGML_CUDA_P2P=1 $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 3 2>&1 | grep -oE "tg256 +\| +[0-9.]+ ± [0-9.]+" | sed "s/^/   wait_dst=$w: /"
done
for w in 0 1 2 2 1 0; do
  waitmodel; cool 46
  GGML_CUDA_PEER_WAIT_DST=$w GGML_CUDA_P2P=1 timeout 900 $B/llama-speculative-simple -m $M --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 \
      -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      > $S/nan/mtp2_w$w.txt 2> $S/nan/mtp2_w$w.log
  echo "   wait_dst=$w MTP: $(grep -oE 'speed: +[0-9.]+ t/s' $S/nan/mtp2_w$w.log | tail -1)  $(grep -oE 'accept +=  *[0-9.]+%' $S/nan/mtp2_w$w.log | tail -1)  text $(md5sum < $S/nan/mtp2_w$w.txt | cut -c1-8)"
done
unset LD_LIBRARY_PATH
echo "=== (R3) PV algorithm timing, round-robin, GPU1 ($(date +%T))"
cool 45; CUDA_VISIBLE_DEVICES=1 $S/pv/pvalgo
echo "=== (R4) upstream kernel race stress, 100 iterations each, GPU1 ($(date +%T))"
for c in softmax_oop softmax_ip groupnorm_oop rmsnorm_ip norm_ip; do
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$B timeout 900 $S/stress/stress $c 100 2>&1 | grep -E '^case|^  iter' | tail -8
done
echo "=== queue_rest done ($(date +%T))"
