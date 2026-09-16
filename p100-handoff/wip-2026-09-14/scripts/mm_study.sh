#!/usr/bin/env bash
# Prefill matmul algorithm study (sm_60): cuBLAS COMPUTE_16F with CUBLAS_GEMM_DEFAULT_TENSOR_OP (the
# upstream call, GGML_CUDA_MM16_ALGO=-1) against the blocked ALGO6 (new default) and ALGO5.
# Part 1: speed, interleaved and cooled. Part 2: paired per-chunk perplexity against an all-fp32
# reference (fp32 matmuls + fp32 attention accumulation), 4096 x 30 at -ub 2048 (the study_v3 setup).
cd /home/kaden/llama-opt
B=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad/snap-mm/bin
export LD_LIBRARY_PATH=$B
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad/mm
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
sha=$(sha256sum $B/libggml-cuda.so | cut -c1-16); echo "library $sha ($(date +%T))"
echo "stray llama processes: $(pgrep -f 'bin/llama-' | wc -l)"
cool() { local t0=$(date +%s); until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le 48 ]; do sleep 10; done
         echo "   (cooled in $(( $(date +%s) - t0 ))s: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | paste -sd/))"; }
ulimit -c 0
if [ "${SKIP_SPEED:-0}" != 1 ]; then
echo "=== speed: pp512/pp1024/pp2048 at -b 2048 -ub 2048 (matmul n = prompt size), 2 rounds, order rotated"
for round in 1 2; do
  if [ $round = 1 ]; then order="-1 6 5"; else order="5 6 -1"; fi
  for a in $order; do
    cool
    env GGML_CUDA_MM16_ALGO=$a GGML_CUDA_P2P=1 $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
        -p 512,1024,2048 -n 0 -b 2048 -ub 2048 -r 3 2>&1 | grep -oE "pp[0-9]+ +\| +[0-9.]+ ± [0-9.]+" | sed "s/^/   algo $a: /"
  done
done
fi
echo "=== quality: 4096 x 30, -b 4096 -ub 2048 ($(date +%T))"
p() { cool; env $3 $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub 2048 -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 $4 > $S/$1.log 2>&1
      echo "   $1: $(grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' $S/$1.log) ($(date +%T))"; }
p m_ref32   ""  "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM_PREC=32"
p m_ctl32   ""  "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM_PREC=32" "-ub 1024"
p m_def16   ""  "GGML_CUDA_MM16_ALGO=-1"
p m_a6      ""  "GGML_CUDA_MM16_ALGO=6"
p m_mm32    ""  "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32"
[ "$(sha256sum $B/libggml-cuda.so | cut -c1-16)" = "$sha" ] && echo "library unchanged during study" || echo "LIBRARY CHANGED DURING STUDY -- INVALID"
python3 - "$S" <<'PY'
import re, sys, math
S = sys.argv[1]
def pc(f):
    av = []
    for line in open(f"{S}/{f}.log"):
        m = re.match(r"^\s*(\d+)\s+([0-9.]+|nan)\s+([0-9.]+|nan)\s+([0-9.]+|nan)\s*$", line)
        if m: av.append(float(m.group(3)))
    return [ (k+1)*av[k] - k*(av[k-1] if k else 0.0) for k in range(len(av)) ]
r = pc("m_ref32")
print(f"\nall-fp32 reference: PPL {math.exp(sum(r)/len(r)):.4f} over {len(r)} chunks")

print(f"   {'variant':44s} {'PPL':>7s} {'mean dNLL':>10s} {'se':>9s} {'t':>6s}")
for name, f in (("CONTROL all-fp32 at -ub 1024 (reassociation)", "m_ctl32"), ("fp16 matmul DEFAULT_TENSOR_OP (upstream)", "m_def16"), ("fp16 matmul ALGO6 (new)", "m_a6"), ("fp32 matmul, fp16 attention", "m_mm32")):
    v = pc(f); n = min(len(v), len(r)); d = [v[i]-r[i] for i in range(n)]
    if any(x != x for x in d): print(f"   {name:44s}  NaN present"); continue
    mean = sum(d)/n; sd = math.sqrt(sum((x-mean)**2 for x in d)/(n-1)); se = sd/math.sqrt(n)
    print(f"   {name:44s} {math.exp(sum(v[:n])/n):7.4f} {mean:+10.6f} {se:9.6f} {mean/se:+6.2f}")
a, b = pc("m_def16"), pc("m_a6"); n = min(len(a), len(b)); d = [b[i]-a[i] for i in range(n)]
mean = sum(d)/n; sd = math.sqrt(sum((x-mean)**2 for x in d)/(n-1)); se = sd/math.sqrt(n)
print(f"   ALGO6 minus DEFAULT_TENSOR_OP, paired: {mean:+.6f} (se {se:.6f}, t {mean/se:+.2f})")
PY
echo "=== done ($(date +%T))"
