#!/usr/bin/env bash
# Keeps the LCB run alive unattended: restarts llama-server if it dies,
# restarts the harness (with --resume) if it dies, exits when the run is done.
set -uo pipefail
S="$1"; TARGET="${2:-100}"
SRV='env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=/home/kaden/llama-opt/build-opt/bin /home/kaden/llama-opt/build-opt/bin/llama-server -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -c 36864 --parallel 1 --spec-type draft-mtp --host 127.0.0.1 --port 8080 --no-warmup'

log(){ echo "$(date +%H:%M:%S) $*" >> "$S/supervise.log"; }

while true; do
  done_n=$(python3 -c "import json,sys;print(len([r for r in json.load(open('$S/lcb_run.json')) if 'ok' in r]))" 2>/dev/null || echo 0)
  if [ "$done_n" -ge "$TARGET" ]; then log "COMPLETE $done_n/$TARGET"; break; fi

  if ! curl -s --max-time 20 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok; then
    log "server unhealthy -> restarting"
    pid=$(ps -eo pid,comm --no-headers | awk '$2=="llama-server"{print $1}')
    [ -n "$pid" ] && kill $pid 2>/dev/null; sleep 10
    setsid bash -c "$SRV > $S/srv-run.log 2>&1" < /dev/null > /dev/null 2>&1 &
    for i in $(seq 1 60); do sleep 5; curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q ok && break; done
    log "server back up"
  fi

  if ! ps -eo comm,args --no-headers | grep -q "[l]cb/bench.py"; then
    log "harness not running ($done_n/$TARGET done) -> resuming"
    setsid bash -c "python3 tools/lcb/bench.py --port 8080 --data $S/lcb_v6.jsonl --out $S/lcb_run.json --n $TARGET --seed 1234 --max-tokens 32000 --workers 1 --resume >> $S/lcb_run.log 2>&1" < /dev/null > /dev/null 2>&1 &
    sleep 30
  fi
  sleep 120
done
