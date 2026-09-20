#!/bin/bash
# Do model + MTP + vision all fit at FULL context?
#
# Known baseline without vision: -ub 2048, graphs off, full 259229-token prefill bottoms out at
# 757 MiB free on GPU0. The mmproj is 629 MB, so if it lands on GPU0 and stays resident this
# should not fit -- but that is a prediction, and extrapolating a VRAM peak has burned this
# project before (FINDINGS item 10). So: load it, then drive a real full-depth prefill and take
# the MINIMUM over the whole guard log, not the samples that happen to be on screen.
#
# $1 = context size (default 262144)
set -u
ulimit -c 0
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
REL=/mnt/fast/p100-llamacpp-release
CTX=${1:-262144}
UB=${2:-2048}
export LD_LIBRARY_PATH=$REL/build:${LD_LIBRARY_PATH:-}
export GGML_CUDA_P2P=1
export GGML_CUDA_GRAPHS_PRE_VOLTA=0
export GLOG=$S/guard_vis${CTX}_ub${UB}.log; : > $GLOG

while pgrep -f "port 809[0-9]" >/dev/null 2>&1; do sleep 20; done
sleep 5

$REL/build/llama-server -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
  --mmproj /mnt/fast/models/mmproj-Qwen3.8-27B-Q8_0.gguf \
  -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
  -c $CTX -b 32768 -ub $UB -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
  -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
  --jinja --temp 0.3 --top-k 20 \
  --host 127.0.0.1 --port 8091 > $S/vis${CTX}_ub${UB}.log 2>&1 &
SRV=$!; echo "server pid $SRV" > $S/vis${CTX}_ub${UB}.pid
sh $S/guard.sh $SRV 200 &

for i in $(seq 1 240); do
    grep -q "listening on" $S/vis${CTX}_ub${UB}.log && break
    kill -0 $SRV 2>/dev/null || { echo "CTX=$CTX DIED AT LOAD"; grep -iE "error|out of memory|alloc" $S/vis${CTX}_ub${UB}.log | tail -5; exit 1; }
    sleep 2
done
echo "CTX=$CTX UB=$UB loaded $(date +%H:%M:%S)"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sed 's/^/  free after load: /'
grep -iE "mmproj|clip|vision" $S/vis${CTX}_ub${UB}.log | head -4

python3 -u - "$CTX" <<'PY' 2>&1 | tee $S/vis_results_${CTX}_ub${UB}.txt
import json,urllib.request,sys,time
S="/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad"
U="http://127.0.0.1:8091"; CTX=int(sys.argv[1])
def post(p,b,t=21600):
    try:
        r=urllib.request.urlopen(urllib.request.Request(U+p,json.dumps(b).encode(),
            {"Content-Type":"application/json"}),timeout=t)
        return r.status,json.loads(r.read())
    except Exception as e:
        return 0,str(e)
full=open(S+"/p262.txt").read()
# fill the context as far as it goes: ~3.06 chars/token measured on this corpus
prompt = full if CTX >= 262144 else full[:int(len(full)*(CTX/262144.0)*0.95)]
t0=time.time()
st,d = post("/completion",{"prompt":prompt,"n_predict":64,"temperature":0.3,"top_k":20,
                           "cache_prompt":True,"ignore_eos":True})
if isinstance(d,dict) and "timings" in d:
    t=d["timings"]
    print("CTX=%d prompt_n=%d prefill=%.2f t/s decode=%.2f t/s wall=%.0fs"%(
        CTX,t["prompt_n"],t["prompt_per_second"],t["predicted_per_second"],time.time()-t0))
else:
    print("CTX=%d PREFILL FAILED: %s"%(CTX,str(d)[:200]))
PY

echo "=== acceptance ==="
grep -oE "draft acceptance = [0-9.]+ \( *[0-9]+ accepted / *[0-9]+ generated\), mean len = *[0-9.]+" $S/vis${CTX}_ub${UB}.log | tail -2
MIN=$(grep -oE "gpu0_free=[0-9]+" $GLOG | cut -d= -f2 | sort -n | head -1)
echo "CTX=$CTX UB=$UB min_gpu0_free=${MIN}MiB  (guard samples GPU0 only -- it is the one carrying Sunshine)"
kill $SRV 2>/dev/null; sleep 12; kill -9 $SRV 2>/dev/null
echo "vision$CTX done"
