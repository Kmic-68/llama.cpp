#!/usr/bin/env bash
# Everything that needs the model, once /mnt/fast is back:
#  (A) peer fix v2 (= the final build) event runs   (B) decode speed off / v1 / v2 / final
#  (C) final gate + release A/B + decode at depth + 262k server   (D) OOP accum speed
#  (E) all-fp32 reference, control and fp32-matmul row re-run on the race-free build
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
R=/mnt/fast/p100-llamacpp-release/build
B=$S/snap-final2/bin
PF=$S/snap-peerfix2/bin
F=$S/final
waitmodel() { until [ -r $M ]; do sleep 5; done; }
cool() { local lim=${1:-48} t0=$(date +%s); until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le $lim ]; do sleep 10; done
         echo "   (cooled ${lim}C in $(( $(date +%s) - t0 ))s: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | paste -sd/))"; }
ulimit -c 0
waitmodel
echo "=== queue_model start ($(date +%T)); final lib $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16), knob lib $(sha256sum $PF/libggml-cuda.so.0.21.0 | cut -c1-16)"
echo "llama processes: $(ps -eo args | grep -cE '^[^ ]*bin/llama-')"

if [ "${SKIP_A0:-0}" != 1 ]; then
I=$S/snap-inject/bin
mkdir -p $S/inject
rowsof() { grep -E '^ +[0-9]+ +[0-9na.]+ +[0-9na.]+ +[0-9na.]+ *$' $1 | awk '{print $3}' | paste -sd' '; }
echo "=== (A0) race injection (knob build + TEMP delay, lib $(sha256sum $I/libggml-cuda.so.0.21.0 | cut -c1-16)): device 1's all-reduce ADD delayed by dummy sgemms on its stream ($(date +%T))"
echo "   prefill, fp32 matmuls (partials not f16-exact: every exchange uncompressed), P2P, 2 chunks; expected rows without a race: 1.596804 1.370962"
pinj() { local name=$1; shift; waitmodel
  env "$@" GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$I $I/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub 2048 --chunks 2 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/inject/$name.log 2>&1
  echo "   $name: rows [$(rowsof $S/inject/$name.log)]  $(grep 'ADD graphs delayed' $S/inject/$name.log | tail -1) ($(date +%T))"; }
pinj f32_nodelay_w2 GGML_CUDA_PEER_WAIT_DST=2 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
pinj f32_delay_w0   GGML_CUDA_PEER_WAIT_DST=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
pinj f32_delay_w2   GGML_CUDA_PEER_WAIT_DST=2 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
pinj f32_delay_w1   GGML_CUDA_PEER_WAIT_DST=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
echo "   prefill, production fp16 matmuls (exchanges >= 512 rows compressed and already guarded), P2P, 2 chunks"
pinj f16_nodelay_w2 GGML_CUDA_PEER_WAIT_DST=2
pinj f16_delay_w0   GGML_CUDA_PEER_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
pinj f16_delay_w2   GGML_CUDA_PEER_WAIT_DST=2 GGML_CUDA_TEMP_DELAY_ADD_DEV=1
echo "   MTP decode (production flags), 64 tokens, delay 2 x sgemm 2400 per ADD"
minj() { local name=$1; shift; waitmodel
  env "$@" GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$I timeout 900 $I/llama-speculative-simple -m $M --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 \
      -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" -n 64 --temp 0 --top-k 1 --seed 42 -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      > $S/inject/$name.txt 2> $S/inject/$name.log
  echo "   $name: text $(md5sum < $S/inject/$name.txt | cut -c1-8) ($(wc -c < $S/inject/$name.txt) bytes)  $(grep -oE 'accept +=  *[0-9.]+%' $S/inject/$name.log | tail -1)  $(grep 'ADD graphs delayed' $S/inject/$name.log | tail -1) ($(date +%T))"; }
minj mtp_nodelay_w2 GGML_CUDA_PEER_WAIT_DST=2
minj mtp_delay_w0   GGML_CUDA_PEER_WAIT_DST=0 GGML_CUDA_TEMP_DELAY_ADD_DEV=1 GGML_CUDA_TEMP_DELAY_N=2400 GGML_CUDA_TEMP_DELAY_REPS=2
minj mtp_delay_w2   GGML_CUDA_PEER_WAIT_DST=2 GGML_CUDA_TEMP_DELAY_ADD_DEV=1 GGML_CUDA_TEMP_DELAY_N=2400 GGML_CUDA_TEMP_DELAY_REPS=2
fi

if [ "${SKIP_A:-0}" != 1 ]; then
echo "=== (A) final build (peer fix v2 + per-exchange compression type check): event runs, fp32 matmuls, no P2P, 15 chunks ($(date +%T))"
for r in 1 2 3 4 5 6; do
  waitmodel
  GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$B $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub 2048 --chunks 15 \
      -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 > $S/nan/p_w2_r$r.log 2>&1
  echo "   v2 run $r: $(grep -cE '^ +[0-9]+ +nan' $S/nan/p_w2_r$r.log) nan rows, $(grep -oE 'PPL = [0-9.]+' $S/nan/p_w2_r$r.log) ($(date +%T))"
done
python3 - "$S" <<'PY' | tee $S/nan/events_summary.txt
import re, sys, collections, glob, os
S = sys.argv[1]
def rows(f):
    return [m.group(3) for m in (re.match(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$", l) for l in open(f)) if m]
runs = {os.path.basename(f)[2:-4]: rows(f) for f in sorted(glob.glob(f"{S}/nan/p_w[012]_r*.log"))}
full = {k: v for k, v in runs.items() if len(v) == 15}
print(f"   complete runs: {len(full)} of {len(runs)} (excluded: {sorted(set(runs) - set(full))})")
maj = [collections.Counter(v[i] for v in full.values()).most_common(1)[0][0] for i in range(15)]
bad = False
for w, name in (("0", "no fix"), ("1", "v1 record+wait"), ("2", "v2 existing marker (final build)")):
    ks = sorted(k for k in full if k.startswith(w + "_"))
    firsts = [next((i + 1 for i in range(15) if full[k][i] != maj[i]), None) for k in ks]
    n_ev = sum(f is not None for f in firsts)
    print(f"   {name:34s}: {n_ev} of {len(ks)} runs with an event; first divergent chunk per run {firsts}")
    if w == "2" and n_ev: bad = True
print("   V2_EVENT" if bad else "   V2_CLEAN")
PY
if grep -q V2_EVENT $S/nan/events_summary.txt; then echo "!!! v2 showed an event -- queue stopped for review ($(date +%T))"; exit 1; fi
fi

if [ "${SKIP_B:-0}" != 1 ]; then
echo "=== (B) decode speed, production flags (P2P): no fix / v1 / v2 (knob build) / final build, interleaved ($(date +%T))"
runv() { local v=$1; shift
  if [ "$v" = final ]; then env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$B "$B/$@"
  else env GGML_CUDA_PEER_WAIT_DST=$v GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$PF "$PF/$@"; fi; }
for v in 0 1 2 final final 2 1 0; do
  waitmodel; cool 46
  runv $v llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 3 2>&1 | grep -oE "tg256 +\| +[0-9.]+ ± [0-9.]+" | sed "s/^/   $v: /"
done
for v in 0 1 2 final final 2 1 0; do
  waitmodel; cool 46
  runv $v llama-speculative-simple -m $M --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 \
      -p "Here is a quick sort implementation in C++. Just code, no comments:\n\n#include" -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      > $S/nan/mtp3_$v.txt 2> $S/nan/mtp3_$v.log
  echo "   $v MTP: $(grep -oE 'speed: +[0-9.]+ t/s' $S/nan/mtp3_$v.log | tail -1)  $(grep -oE 'accept +=  *[0-9.]+%' $S/nan/mtp3_$v.log | tail -1)  text $(md5sum < $S/nan/mtp3_$v.txt | cut -c1-8)"
done
fi

if [ "${SKIP_C:-0}" != 1 ]; then
export LD_LIBRARY_PATH=$B
echo "=== (F1) final build tg256, cool cards, -r 5 (baseline 17.51) ($(date +%T))"
waitmodel; cool 45
GGML_CUDA_P2P=1 $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5 2>&1 | grep -E "tg256|error"
echo "=== (F2) perplexity gate, ppl-orig.txt -c 4096 (2.6209 +/- 0.0199) ($(date +%T))"
waitmodel
$B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 2>&1 | grep -E "Final estimate|error"

echo "=== (F3) release vs final, interleaved, cooled ($(date +%T))"
run() { local v=$1; shift; waitmodel; cool 48
  case $v in
    release) env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$R $R/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 "$@" ;;
    final)   env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$B $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 "$@" ;;
  esac 2>&1 | grep -oE "(pp[0-9]+( @ d[0-9]+)?|tg[0-9]+( @ d[0-9]+)?) +\| +[0-9.]+ ± [0-9.]+|error.*|failed.*" | sed "s/^/   $v: /"
}
for v in release final final release; do run $v -p 512,1024,2048 -n 0 -b 2048 -ub 2048 -r 3; done
for v in release final; do run $v -p 4096 -n 0 -r 2; done         # default -b 2048 -ub 512: eight 512-row ubatches
for v in final release; do run $v -ub 2048 -b 2048 -p 2048 -n 0 -d 16384 -r 2; done

echo "=== (F4) decode at depth, final build, no MTP ($(date +%T))"
for d in 0 20000; do run final -p 0 -n 128 -d $d -r 2; done

echo "=== (F5) production server at 262144 context, MTP, 20k-token prompt, n_predict 256 ($(date +%T))"
waitmodel; cool 48
GGML_CUDA_P2P=1 $B/llama-server -m $M -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -c 262144 -b 262144 -ub 2048 -np 1 \
    --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 -ubd 256 \
    --host 127.0.0.1 --port 8089 --no-warmup > $F/srv.log 2>&1 &
SRV=$!
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8089/health >/dev/null 2>&1 && break; kill -0 $SRV 2>/dev/null || break; sleep 2; done
if kill -0 $SRV 2>/dev/null; then
  echo "   loaded; VRAM idle per card: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd/) MiB"
  ( while kill -0 $SRV 2>/dev/null; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' '; sleep 0.5; done ) > $F/vram.txt &
  MON=$!
  for rep in 1 2; do
  python3 - "$S" <<'PY'
import json,sys,urllib.request,time
S=sys.argv[1]; p=open(S+"/mine/long_prompt.txt").read()
req=urllib.request.Request("http://127.0.0.1:8089/completion",data=json.dumps({"prompt":p,"n_predict":256,"temperature":0,"cache_prompt":False,"ignore_eos":True}).encode(),headers={"Content-Type":"application/json"})
t0=time.time(); r=json.loads(urllib.request.urlopen(req,timeout=3600).read().decode()); t=r.get("timings",{})
print("   prompt %s tok @ %.1f t/s | gen %s tok @ %.2f t/s | draft %s accepted %s | wall %.0f s | %r" % (t.get("prompt_n"), t.get("prompt_per_second",0), t.get("predicted_n"), t.get("predicted_per_second",0), t.get("draft_n"), t.get("draft_n_accepted"), time.time()-t0, r.get("content","")[:60]))
PY
  done
  kill $MON 2>/dev/null; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  awk '{if($1>a)a=$1; if($2>b)b=$2} END{printf "   PEAK VRAM per card: GPU0 %d MiB, GPU1 %d MiB of 16384 (GPU0 includes Sunshine ~392 MiB)\n",a,b}' $F/vram.txt
else
  echo "   server died at startup:"; grep -iE "error|fail|oom|alloc" $F/srv.log | tail -5
fi
grep -iE "cudaMalloc failed|out of memory|GGML_ASSERT" $F/srv.log | head -3
unset LD_LIBRARY_PATH
fi

if [ "${SKIP_D:-0}" != 1 ]; then
echo "=== (D) GEMM accum_O in place (snap-mm) vs out of place (snap-oop), pp2048 @ d16384, interleaved ($(date +%T))"
for v in mm oop oop mm; do
  waitmodel; cool 46
  env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$S/snap-$v/bin $S/snap-$v/bin/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -b 2048 -p 2048 -n 0 -d 16384 -r 3 2>&1 \
    | grep -oE "pp2048 @ d16384 +\| +[0-9.]+ ± [0-9.]+" | sed "s/^/   $v: /"
done
fi

if [ "${SKIP_E:-0}" != 1 ]; then
echo "=== (E) all-fp32 reference, control and fp32-matmul row on the final (race-free) build, 4096 x 30, -ub 2048 ($(date +%T))"
mkdir -p $S/mm2
p() { waitmodel; cool; env LD_LIBRARY_PATH=$B $3 $B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -c 4096 -b 4096 -ub 2048 -sm tensor -fa 1 -ngl 99 -ctk q4_0 -ctv q4_0 --ppl-output-type 1 $4 > $S/mm2/$1.log 2>&1
      echo "   $1: $(grep -oE 'PPL = [0-9.]+ \+/- [0-9.]+' $S/mm2/$1.log) ($(date +%T))"; }
p m_ref32 "" "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM_PREC=32"
p m_ctl32 "" "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_FA_GEMM_PREC=32" "-ub 1024"
p m_mm32  "" "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32"
p m_a6    "" ""
python3 - "$S" <<'PY'
import re, sys, math
S = sys.argv[1]
def pc(path):
    av = []
    for line in open(path):
        m = re.match(r"^\s*(\d+)\s+([0-9.]+|nan)\s+([0-9.]+|nan)\s+([0-9.]+|nan)\s*$", line)
        if m: av.append(float(m.group(3)))
    return [(k+1)*av[k] - k*(av[k-1] if k else 0.0) for k in range(len(av))]
old = lambda f: pc(f"{S}/mm/{f}.log")
new = lambda f: pc(f"{S}/mm2/{f}.log")
for f in ("m_ref32", "m_ctl32", "m_a6"):
    a, b = old(f), new(f); n = min(len(a), len(b))
    diff = [i + 1 for i in range(n) if abs(a[i] - b[i]) > 1e-9]
    print(f"   {f}: old (09-14 build) vs new (final build), {n} chunks: {'identical' if not diff else 'DIFFER at chunks ' + str(diff)}")
r = new("m_ref32")
print(f"\n   all-fp32 reference (final build): PPL {math.exp(sum(r)/len(r)):.4f} over {len(r)} chunks")
print(f"   {'variant':46s} {'PPL':>7s} {'mean dNLL':>10s} {'se':>9s} {'t':>6s}")
rows = (("CONTROL all-fp32 at -ub 1024 (reassociation)", new("m_ctl32")), ("fp16 matmul DEFAULT_TENSOR_OP (upstream, 09-14)", old("m_def16")),
        ("fp16 matmul ALGO6 (final build)", new("m_a6")), ("fp32 matmul, fp16 attention (final build)", new("m_mm32")))
for name, v in rows:
    n = min(len(v), len(r)); d = [v[i] - r[i] for i in range(n)]
    if n < 2 or any(x != x for x in d): print(f"   {name:46s}  NaN present or no data"); continue
    mean = sum(d)/n; sd = math.sqrt(sum((x-mean)**2 for x in d)/(n-1)); se = sd/math.sqrt(n)
    print(f"   {name:46s} {math.exp(sum(v[:n])/n):7.4f} {mean:+10.6f} {se:9.6f} {mean/se:+6.2f}")
def paired(a, b, label):
    n = min(len(a), len(b)); d = [b[i]-a[i] for i in range(n)]
    mean = sum(d)/n; sd = math.sqrt(sum((x-mean)**2 for x in d)/(n-1)); se = sd/math.sqrt(n)
    print(f"   {label}: {mean:+.6f} (se {se:.6f}, t {mean/se:+.2f})")
paired(old("m_def16"), new("m_a6"), "ALGO6 minus DEFAULT_TENSOR_OP, paired")
paired(new("m_mm32"), new("m_a6"), "fp16 matmul (ALGO6) minus fp32 matmul, both fp16 attention, paired")
PY
fi
if [ "${SKIP_G:-0}" != 1 ]; then
echo "=== (G) final build: FLASH_ATTN_EXT eval and full op suite ($(date +%T))"
LD_LIBRARY_PATH=$B $B/test-backend-ops test -o FLASH_ATTN_EXT > $F/fa2.log 2>&1; echo "   FA exit=$? ($(date +%T))"; grep -E 'tests passed|backends passed|FAIL' $F/fa2.log | head -6
LD_LIBRARY_PATH=$B $B/test-backend-ops test > $F/full2.log 2>&1; echo "   full exit=$? ($(date +%T))"; grep -E 'tests passed|backends passed|FAIL' $F/full2.log | head -8
fi
if [ "${SKIP_H:-0}" != 1 ]; then
echo "=== (H) in-place upstream kernels, 2000 launches each, GPU1, final build ($(date +%T))"
for c in softmax_ip rmsnorm_ip norm_ip; do
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$B timeout 1800 $S/stress/stress $c 2000 2>&1 | grep -E '^case|^  iter' | tail -12
done
fi
echo "=== queue_model done ($(date +%T))"
