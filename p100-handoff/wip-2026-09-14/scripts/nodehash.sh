#!/usr/bin/env bash
# Whole-forward-pass determinism: hash every computed node's output (GGML_CUDA_DBG_NODEHASH) and
# compare runs bit-for-bit. Any data race in any op this model runs shows up as the first
# differing node. (1) prefill: 16k context, 1 chunk, -ub 2048, production defaults, 3 runs.
# (2) decode with the MTP draft, greedy, 2 runs.
cd /home/kaden/llama-opt
B=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad/snap-oopnh/bin
export LD_LIBRARY_PATH=$B
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad/nh
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
echo "library $(sha256sum $B/libggml-cuda.so | cut -c1-16) ($(date +%T))"
echo "stray llama processes: $(pgrep -f 'bin/llama-' | wc -l)"
ulimit -c 0
for i in 1 2 3; do
  GGML_CUDA_P2P=1 GGML_CUDA_DBG_NODEHASH=1 $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 16384 -b 16384 -ub 2048 --chunks 1 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 2>&1 | grep -E 'NODEHASH|Final estimate' > $S/pf$i.log
  echo "prefill run $i: $(grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' $S/pf$i.log)  nodes: $(grep -c NODEHASH $S/pf$i.log) ($(date +%T))"
done
head -c 3000 p100-handoff/ppl-orig.txt > $S/prompt.txt
for i in 1 2; do
  GGML_CUDA_P2P=1 GGML_CUDA_DBG_NODEHASH=1 timeout 1800 $B/llama-speculative-simple -m $M -f $S/prompt.txt -n 48 \
      --temp 0 --top-k 1 --seed 42 --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 > $S/dec$i.txt 2> $S/dec$i.full.log
  grep NODEHASH $S/dec$i.full.log > $S/dec$i.log
  echo "decode run $i: nodes $(wc -l < $S/dec$i.log), $(grep -oE 'accept +=  *[0-9.]+%' $S/dec$i.full.log | tail -1), text $(md5sum < $S/dec$i.txt | cut -c1-8) ($(date +%T))"
done
python3 - "$S" <<'PY'
import re, sys, itertools
S = sys.argv[1]
def load(f):
    rows, cnt = [], {}
    for line in open(f"{S}/{f}"):
        m = re.search(r"NODEHASH dev=(\d+) seq=\d+ i=(\d+) op=(\S+) name=(.*?) ne=(\S+) h=([0-9a-f]+)\s*$", line)
        if not m: continue
        dev = int(m.group(1)); k = cnt.get(dev, 0); cnt[dev] = k + 1
        rows.append(((dev, k), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)))
    return {r[0]: r[1:] for r in rows}
def compare(files, label):
    runs = [load(f) for f in files]
    for a, b in itertools.combinations(range(len(runs)), 2):
        ra, rb = runs[a], runs[b]
        keys = sorted(set(ra) & set(rb))
        first = None; ndiff = 0; struct = 0
        for k in keys:
            if ra[k][:4] != rb[k][:4]: struct += 1; continue
            if ra[k][4] != rb[k][4]:
                ndiff += 1
                if first is None: first = (k, ra[k])
        print(f"{label} runs {a+1} vs {b+1}: {len(keys)} common nodes (only-in-one: {len(set(ra)^set(rb))}), structural mismatches {struct}, hash mismatches {ndiff}")
        if first: print(f"   first differing node: dev={first[0][0]} #{first[0][1]} graph-index {first[1][0]} op {first[1][1]} name {first[1][2]} ne {first[1][3]}")
compare(["pf1.log", "pf2.log", "pf3.log"], "prefill")
compare(["dec1.log", "dec2.log"], "decode")
PY
echo "=== done ($(date +%T))"
