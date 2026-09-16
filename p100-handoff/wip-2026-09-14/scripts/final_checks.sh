#!/usr/bin/env bash
# Final build (no knobs, no debug hooks): the CLAUDE.md gate, A/B vs the shipped release, decode at
# depth, and the production server at 262144 context (VRAM, prompt and generation speed).
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
F=$S/final
B=$S/snap-final/bin
R=/mnt/fast/p100-llamacpp-release/build
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
waitmodel() { until [ -r $M ]; do sleep 5; done; }
cool() { local lim=${1:-48} t0=$(date +%s); until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le $lim ]; do sleep 10; done
         echo "   (cooled ${lim}C in $(( $(date +%s) - t0 ))s: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | paste -sd/))"; }
ulimit -c 0
export LD_LIBRARY_PATH=$B
echo "=== final build $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16), HEAD $(git rev-parse --short HEAD) ($(date +%T))"
echo "stray llama processes: $(pgrep -f 'bin/llama-' | wc -l)"

echo "=== (F1) tg256, cool cards, -r 5 (baseline 17.51)"
waitmodel; cool 45
GGML_CUDA_P2P=1 $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5 2>&1 | grep -E "tg256|error"
echo "=== (F2) perplexity gate, ppl-orig.txt -c 4096 (2.6209 +/- 0.0199) ($(date +%T))"
waitmodel
$B/llama-perplexity -m $M -f p100-handoff/ppl-orig.txt -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 2>&1 | grep -E "Final estimate|error"

echo "=== (F3) release vs final, interleaved, cooled ($(date +%T))"
run() { local v=$1; shift; waitmodel; cool 48
  case $v in
    release) env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=$R $R/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 "$@" ;;
    final)   env GGML_CUDA_P2P=1 $B/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 "$@" ;;
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

echo "=== (F6) FLASH_ATTN_EXT eval and full op suite ($(date +%T))"
$B/test-backend-ops test -o FLASH_ATTN_EXT > $F/fa.log 2>&1; echo "   FA exit=$?"; grep -E 'tests passed|backends passed|FAIL' $F/fa.log | head -4
$B/test-backend-ops test > $F/full.log 2>&1; echo "   full exit=$?"; grep -E 'tests passed|backends passed|FAIL' $F/full.log | head -6
echo "=== final checks done ($(date +%T))"
