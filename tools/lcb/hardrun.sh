#!/usr/bin/env bash
# Extend the hard tier from n=7 toward n=30. Resumable, supervised, deadline-bounded.
set -uo pipefail
S="$1"; DEADLINE="$2"; TARGET="${3:-30}"
B=/home/kaden/llama-opt/tools/lcb/bench.py
log(){ echo "$(date +%H:%M:%S) $*" >> "$S/hardrun.log"; }
health(){ curl -s --max-time 20 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok; }
SRVCMD='env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=/home/kaden/llama-opt/build-opt/bin /home/kaden/llama-opt/build-opt/bin/llama-server -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -c 73728 --parallel 1 --spec-type draft-mtp --host 127.0.0.1 --port 8080 --no-warmup'
log "START target=$TARGET deadline=$(date -d @$DEADLINE '+%H:%M')"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  d=$(python3 -c "import json;print(len([r for r in json.load(open('$S/lcb_hard.json')) if 'ok' in r]))" 2>/dev/null || echo 0)
  [ "$d" -ge "$TARGET" ] && { log "DONE $d/$TARGET"; break; }
  if ! health; then
    log "server down -> restart"
    p=$(ps -eo pid,comm --no-headers | awk '$2=="llama-server"{print $1}'); [ -n "$p" ] && kill $p; sleep 10
    setsid bash -c "$SRVCMD > $S/srv-hard.log 2>&1" < /dev/null > /dev/null 2>&1 &
    for i in $(seq 1 60); do sleep 5; health && break; done
    log "server up"
  fi
  if ! ps -eo comm,args --no-headers | grep -q "[b]ench.py"; then
    log "resume at $d/$TARGET"
    setsid bash -c "python3 $B --port 8080 --data $S/lcb_v6.jsonl --out $S/lcb_hard.json --n $TARGET --seed 1234 \
      --difficulty hard --max-tokens 64000 --workers 1 --resume >> $S/hard.log 2>&1" < /dev/null > /dev/null 2>&1 &
    sleep 45
  fi
  sleep 120
done
log "STOPPED"
