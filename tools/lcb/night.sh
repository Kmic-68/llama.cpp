#!/usr/bin/env bash
# Phase A: easy+medium at 32k (they terminate; gives n for a real interval).
# Phase B: hard at 64k (tests whether hard problems pass when not truncated).
# Hard deadline so phase B cannot eat the morning.
set -uo pipefail
S="$1"; DEADLINE="$2"   # DEADLINE as epoch seconds
B=/home/kaden/llama-opt/tools/lcb/bench.py
log(){ echo "$(date +%H:%M:%S) $*" >> "$S/night.log"; }

health(){ curl -s --max-time 20 http://127.0.0.1:8080/health 2>/dev/null | grep -q ok; }
revive(){
  health && return 0
  log "server down -> restart"
  p=$(ps -eo pid,comm --no-headers | awk '$2=="llama-server"{print $1}'); [ -n "$p" ] && kill $p; sleep 10
  setsid bash -c 'env GGML_CUDA_P2P=1 LD_LIBRARY_PATH=/home/kaden/llama-opt/build-opt/bin /home/kaden/llama-opt/build-opt/bin/llama-server -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -c 73728 --parallel 1 --spec-type draft-mtp --host 127.0.0.1 --port 8080 --no-warmup > '"$S"'/srv-big.log 2>&1' < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 60); do sleep 5; health && break; done
  log "server back"
}

phase(){ # name out difficulty n maxtok
  local name=$1 out=$2 diff=$3 n=$4 mt=$5
  log "PHASE $name start (diff=$diff n=$n max_tokens=$mt)"
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    d=$(python3 -c "import json;print(len([r for r in json.load(open('$S/$out')) if 'ok' in r]))" 2>/dev/null || echo 0)
    [ "$d" -ge "$n" ] && { log "PHASE $name done ($d/$n)"; return 0; }
    revive
    if ! ps -eo comm,args --no-headers | grep -q "[b]ench.py"; then
      log "PHASE $name resume at $d/$n"
      setsid bash -c "python3 $B --port 8080 --data $S/lcb_v6.jsonl --out $S/$out --n $n --seed 1234 \
        --difficulty $diff --max-tokens $mt --workers 1 --resume >> $S/$name.log 2>&1" < /dev/null > /dev/null 2>&1 &
      sleep 45
    fi
    sleep 90
  done
  log "PHASE $name hit deadline"
}

phase em   lcb_em.json   easy,medium 60 32000
phase hard lcb_hard.json hard         8 64000
log "NIGHT COMPLETE"
