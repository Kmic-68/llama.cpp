#!/usr/bin/env bash
# Is decode slow at depth? (server showed 3.8 t/s for 16 tokens after a 20k prompt, MTP on)
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad/mine
M=/mnt/fast/models/Qwen3.8-27B-Q6_K.gguf
cool() { until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -n | tail -1)" -le 46 ]; do sleep 10; done; }
echo "=== (1) llama-bench decode, no MTP: depth 0 vs 20000 ($(date +%T))"
for d in 0 20000; do
  cool
  GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m $M -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 128 -d $d -r 2 2>&1 | grep -oE "tg128( @ d[0-9]+)? +\| +[0-9.]+ ± [0-9.]+|error.*|failed.*"
done
echo "=== (2) server, production flags, 20k prompt, n_predict 256: MTP on vs off ($(date +%T))"
for spec in mtp none; do
  if [ $spec = mtp ]; then SP="--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 -ngld 99 -ubd 256"; else SP=""; fi
  cool
  env GGML_CUDA_P2P=1 ./build-opt/bin/llama-server -m $M -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -c 262144 -b 262144 -ub 2048 -np 1 $SP \
      --host 127.0.0.1 --port 8089 --no-warmup > $S/srvd_$spec.log 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:8089/health >/dev/null 2>&1 && break; kill -0 $SRV 2>/dev/null || break; sleep 2; done
  if ! kill -0 $SRV 2>/dev/null; then echo "   $spec: server died at startup"; grep -iE "error|fail" $S/srvd_$spec.log | tail -3; continue; fi
  for np in 256 256; do
  python3 - "$S" "$np" <<'PY'
import json,sys,urllib.request,time
S=sys.argv[1]; n=int(sys.argv[2]); p=open(S+"/long_prompt.txt").read()
req=urllib.request.Request("http://127.0.0.1:8089/completion",data=json.dumps({"prompt":p,"n_predict":n,"temperature":0,"cache_prompt":False,"ignore_eos":True}).encode(),headers={"Content-Type":"application/json"})
t0=time.time(); r=json.loads(urllib.request.urlopen(req,timeout=3600).read().decode()); t=r.get("timings",{})
print("   prompt %s tok @ %.1f t/s | gen %s tok in %.0f ms = %.2f t/s | draft %s acc %s | wall %.0f s" % (t.get("prompt_n"), t.get("prompt_per_second",0), t.get("predicted_n"), t.get("predicted_ms",0), t.get("predicted_per_second",0), t.get("draft_n"), t.get("draft_n_accepted"), time.time()-t0))
PY
  done
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  echo "   ^ $spec"
done
echo "=== done ($(date +%T))"
