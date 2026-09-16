#!/usr/bin/env bash
# (Replaces the hashed hunt: any host synchronize between graph computes orders the destination's
# ADD before the next copy, which is exactly what the suspected race needs to be absent.)
# Suspect: an uncompressed peer copy overwrites the meta backend's reused reduction buffer before
# the destination's ADD of the previous exchange has read it. A/B on identical input, no hooks,
# alternating: fix off (GGML_CUDA_PEER_WAIT_DST=0) vs on, 6 runs each, fp32 matmuls (uncompressed
# exchanges), no P2P -- the configuration that showed events in 3 of 6 runs. Events are counted as
# chunks whose value differs from the per-chunk majority over all 12 runs.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
B=$S/snap-peerfix/bin
export LD_LIBRARY_PATH=$B
ulimit -c 0
echo "   library $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16)"
for r in 1 2 3 4 5 6; do
  for w in 0 1; do
    until [ -r $M ]; do sleep 5; done
    GGML_CUDA_PEER_WAIT_DST=$w GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub 2048 --chunks 15 \
        -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/nan/p_w${w}_r$r.log 2>&1
    echo "   wait_dst=$w run $r: $(grep -cE '^ +[0-9]+ +nan' $S/nan/p_w${w}_r$r.log) nan rows, last: $(grep -E '^ +[0-9]+ +' $S/nan/p_w${w}_r$r.log | tail -1) ($(date +%T))"
  done
done
python3 - "$S" <<'PY'
import re, sys, collections
S = sys.argv[1]
def rows(f):
    out = []
    for line in open(f):
        m = re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", line)
        if m: out.append(m.group(3))
    return out
runs = {(w, r): rows(f"{S}/nan/p_w{w}_r{r}.log") for w in (0, 1) for r in range(1, 7)}
n = min(len(v) for v in runs.values())
# an event makes this chunk's cumulative value, and every later one, differ; count first divergences
maj = [collections.Counter(v[i] for v in runs.values()).most_common(1)[0][0] for i in range(n)]
for w in (0, 1):
    ev = []
    for r in range(1, 7):
        v = runs[(w, r)]
        first = next((i + 1 for i in range(n) if v[i] != maj[i]), None)
        ev.append(first)
    print(f"   wait_dst={w}: runs with an event {sum(e is not None for e in ev)} of 6; first divergent chunk per run {ev}")
PY
echo "   -- decode speed, production flags (P2P), fix off vs on, interleaved"
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le 48 ]; do sleep 10; done; }
for w in 0 1 1 0; do
  cool
  GGML_CUDA_PEER_WAIT_DST=$w GGML_CUDA_P2P=1 $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 3 2>&1 | grep -oE "tg256 +\| +[0-9.]+ ± [0-9.]+" | sed "s/^/   wait_dst=$w: /"
done
for w in 0 1 1 0; do
  cool
  GGML_CUDA_PEER_WAIT_DST=$w GGML_CUDA_P2P=1 timeout 900 $B/llama-speculative-simple -m $M --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 \
      -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      > $S/nan/mtp_w${w}.txt 2> $S/nan/mtp_w${w}.log
  echo "   wait_dst=$w MTP: $(grep -oE 'speed: +[0-9.]+ t/s' $S/nan/mtp_w${w}.log | tail -1)  $(grep -oE 'accept +=  *[0-9.]+%' $S/nan/mtp_w${w}.log | tail -1)  text $(md5sum < $S/nan/mtp_w${w}.txt | cut -c1-8)"
done
