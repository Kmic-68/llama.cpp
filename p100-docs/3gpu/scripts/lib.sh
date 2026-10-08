#!/usr/bin/env bash
# Run harness. Sourced by the stage scripts.
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   DEADLINE   optional epoch seconds; no new run is launched at or after it
#   FAN_HWMON_NAME / FAN3_FILE / FAN_SVC / FAN_TEMP_LIM  fan check (see fan_check; host-specific, adjust or disable)
#   EXTRA_MOUNTS  extra docker -v arguments
# Later additions: fan check (watchdog, cool-down and pre-run), launch deadline, stop on a new log error
# or kernel Xid, CPU governor/MHz logging, server runs (run_server).
P1=${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SRC=${SRC:-$P1/../src}
IMG=${IMG:-p100-llamacpp-test:latest}
MODELS=${MODELS:-$HOME/models}
RES=$P1/results
mkdir -p "$RES"
BASE_FILE=${BASE_FILE:-$P1/idle_baseline.txt}   # optional "idx temp" lines; without the file the limit is a flat 45C
ALERT=$P1/ALERT
STOPFILE=$P1/STOP_REASON
DEADLINE=${DEADLINE:-}            # epoch seconds; no new run is launched at or after it
EXTRA_MOUNTS=${EXTRA_MOUNTS:-}    # extra docker -v args (word-split)

FAN_HWMON_NAME=${FAN_HWMON_NAME:-nct6776}   # hwmon chip that reports the GPU shroud fan on the test host
FAN_HWMON=$(for d in /sys/class/hwmon/hwmon*; do [ "$(cat $d/name 2>/dev/null)" = "$FAN_HWMON_NAME" ] && echo $d; done)
FAN3_FILE=${FAN3_FILE:-$FAN_HWMON/fan3_input}   # shroud fan
FAN_SVC=${FAN_SVC:-gpu-fan-control.service}   # the test host's fan-control unit; set FAN_CHECK=0 to skip the fan check
FAN_TEMP_LIM=${FAN_TEMP_LIM:-50}

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$P1/run.log"; }

# fan check: prints the reason and returns 1 if the fan service is not active, or the shroud fan reads 0 RPM
# while any GPU is above FAN_TEMP_LIM
fan_check() {
    local st rpm hot
    [ "${FAN_CHECK:-1}" = 0 ] && return 0
    st=$(systemctl is-active "$FAN_SVC" 2>/dev/null)
    [ "$st" = active ] || { echo "FANSVC $FAN_SVC is '$st'"; return 1; }
    rpm=$(cat "$FAN3_FILE" 2>/dev/null)
    [ -n "$rpm" ] || { echo "FAN3 unreadable ($FAN3_FILE)"; return 1; }
    if [ "$rpm" -eq 0 ]; then
        hot=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | tr -d ' ' \
              | awk -F, -v L="$FAN_TEMP_LIM" '$2>L{printf "GPU%s=%sC ", $1, $2}')
        [ -n "$hot" ] && { echo "FAN3 0 RPM with ${hot}"; return 1; }
    fi
    return 0
}

alert() { echo "$(date +%T) $*" >> "$ALERT"; }

# wait until every GPU is <= 45C (or <= its idle baseline + 2C if BASE_FILE exists); cap 10 min, then proceed and record
cooldown() {
    local tag=$1 t0=$(date +%s) ok temps why capped=""
    while :; do
        if ! why=$(fan_check); then alert "$why (during cool-down before $tag)"; log "ALERT $why"; return 1; fi
        temps=$(nvidia-smi --query-gpu=index,temperature.gpu --format=csv,noheader,nounits | tr -d ' ')
        ok=1
        while IFS=, read -r i t; do
            b=""; [ -f "$BASE_FILE" ] && b=$(awk -v i="$i" '$1==i{print $2}' "$BASE_FILE"); b=${b:-43}
            lim=$(( b + 2 > 45 ? b + 2 : 45 ))
            [ "$t" -le "$lim" ] || ok=0
        done <<< "$temps"
        [ $ok -eq 1 ] && break
        [ $(( $(date +%s) - t0 )) -ge 600 ] && { capped=" CAPPED(10min)"; break; }
        sleep 5
    done
    log "COOL $tag waited=$(( $(date +%s) - t0 ))s${capped} start_temps=$(echo $temps | tr '\n' ' ') fan3=$(cat $FAN3_FILE)rpm"
    return 0
}

smi_start() {
    nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv -l 1 > "$1" 2>&1 & SMI_PID=$!
    # CPU governor and per-core MHz, 1 Hz
    ( while :; do echo "$(date +%s),$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor),$(awk '/^cpu MHz/{printf "%d ", $4}' /proc/cpuinfo)"; sleep 1; done ) > "${1%.smi.csv}.cpu.csv" 2>&1 & CPU_PID=$!
}
smi_stop()  { kill $SMI_PID $CPU_PID 2>/dev/null; wait $SMI_PID $CPU_PID 2>/dev/null; }

# watchdog: >80C, busy (>50%) with SM < 1000 MHz for 5 s, no GPU memory for 600 s while running, fan check,
# new kernel Xid -> kill the container and write ALERT. WD_NOMEM=0 disables the no-GPU-memory check (CPU-only jobs)
watchdog_start() {
    local name=$1
    ( lowclk=0; nomem=0; n=0; t0=$(date +%s)
      while docker ps --format '{{.Names}}' | grep -qx "$name"; do
        q=$(nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
        hot=$(awk -F, '$2>80{print "GPU"$1"="$2"C"}' <<< "$q")
        slow=$(awk -F, '$4>50 && $3<1000{print "GPU"$1":"$3"MHz@"$4"%"}' <<< "$q")
        mem=$(awk -F, '{s+=$5} END{print s}' <<< "$q")
        if [ -n "$hot" ]; then alert "TEMP>80 $hot"; docker kill "$name"; break; fi
        if [ -n "$slow" ]; then lowclk=$((lowclk+1)); else lowclk=0; fi
        if [ $lowclk -ge 5 ]; then alert "LOWCLOCK $slow"; docker kill "$name"; break; fi
        if [ "${mem:-0}" -lt 100 ]; then nomem=$((nomem+1)); else nomem=0; fi
        if [ "${WD_NOMEM:-1}" = 1 ] && [ $nomem -ge 600 ]; then alert "HANG no GPU memory for 600s"; docker kill "$name"; break; fi
        if ! why=$(fan_check); then alert "$why"; docker kill "$name"; break; fi
        n=$((n+1))
        if [ $((n % 30)) -eq 0 ]; then
            x=$(sudo -n journalctl -k -q --since "@$t0" 2>/dev/null | grep -i -E "xid|nvrm.*error" | head -2)
            if [ -n "$x" ]; then alert "KERNEL $x"; docker kill "$name"; break; fi
        fi
        sleep 1
      done ) & WD_PID=$!
}

# pre-run checks shared by run and run_server. returns 0 = go, 1 = skip (done), 97 = deadline, 99 = alert
prerun() {
    local tag=$1
    [ -f "$ALERT" ] && { log "ALERT present, refusing to run $tag"; return 99; }
    if [ -f "$RES/$tag.exit" ] && [ "$(cat $RES/$tag.exit)" = "0" ]; then log "SKIP $tag (already done)"; return 1; fi
    if [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "DEADLINE reached, not launching $tag"; return 97; fi
    local why; if ! why=$(fan_check); then alert "$why (before $tag)"; log "ALERT $why"; return 99; fi
    cooldown "$tag" || return 99
    if [ -n "$DEADLINE" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then log "DEADLINE reached after cool-down, not launching $tag"; return 97; fi
    return 0
}

# post-run log check: OOM-only failures are recorded as SIZEFAIL (queue continues); any other new error, or a
# non-zero exit, writes ALERT (queue stops)
postrun() {
    local tag=$1 ex
    ex=$(cat "$RES/$tag.exit" 2>/dev/null)
    [ -f "$ALERT" ] && { log "ALERT: $(cat $ALERT)"; return 98; }
    grep -n -i -E "error|assert|abort|out of memory|illegal|xid|nan|segmentation|core dumped" "$RES/$tag.err" 2>/dev/null \
        | grep -v -i -E "error_rate|stderr|ggml_cuda_init|no error" | head -8 > "$RES/$tag.errgrep"
    if [ -s "$RES/$tag.errgrep" ] || [ "$ex" != "0" ]; then
        if grep -q -i -E "out of memory|failed to allocate|cudaMalloc failed|unable to allocate" "$RES/$tag.errgrep" "$RES/$tag.err" 2>/dev/null; then
            log "SIZEFAIL $tag exit=$ex: $(head -2 $RES/$tag.errgrep | tr '\n' ' ')"; echo "$tag" >> "$P1/sizefail.txt"; return 0
        fi
        alert "LOGERROR $tag exit=$ex: $(head -2 $RES/$tag.errgrep | tr '\n' ' ' | cut -c1-300)"
        log "ALERT: $(tail -1 $ALERT)"; return 98
    fi
    return 0
}

# run <tag> <docker extra args...> -- <command...>
run() {
    local tag=$1; shift
    local extra=() ; while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
    prerun "$tag"; local pr=$?; [ $pr -eq 1 ] && return 0; [ $pr -ne 0 ] && return $pr
    smi_start "$RES/$tag.smi.csv"
    log "RUN $tag :: ${extra[*]} :: $*"
    docker run -d --gpus all --name p1run -e GGML_CUDA_P2P=1 "${extra[@]}" \
        -v $SRC:/workspace/src -v $MODELS:/models:ro -v $P1:/phase1 $EXTRA_MOUNTS -w /workspace/src $IMG "$@" > /dev/null
    watchdog_start p1run
    docker wait p1run > "$RES/$tag.exit"
    docker logs p1run > "$RES/$tag.out" 2> "$RES/$tag.err"
    docker rm p1run > /dev/null
    kill $WD_PID 2>/dev/null; wait $WD_PID 2>/dev/null
    smi_stop
    log "END $tag exit=$(cat $RES/$tag.exit)"
    postrun "$tag"
}

# run_server <tag> <docker extra args...> -- <server command...>; the client command is taken from $CLIENT and
# runs on the host once /health answers on port $SPORT. The server is stopped when the client exits.
SPORT=${SPORT:-18080}
run_server() {
    local tag=$1; shift
    local extra=() ; while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
    prerun "$tag"; local pr=$?; [ $pr -eq 1 ] && return 0; [ $pr -ne 0 ] && return $pr
    smi_start "$RES/$tag.smi.csv"
    log "RUN(server) $tag :: ${extra[*]} :: $* :: client: $CLIENT"
    docker run -d --gpus all --name p1run --network host -e GGML_CUDA_P2P=1 "${extra[@]}" \
        -v $SRC:/workspace/src -v $MODELS:/models:ro -v $P1:/phase1 $EXTRA_MOUNTS -w /workspace/src $IMG "$@" > /dev/null
    watchdog_start p1run
    local up=0 i
    for i in $(seq 1 600); do
        docker ps --format '{{.Names}}' | grep -qx p1run || break
        curl -s -m 2 "http://127.0.0.1:$SPORT/health" | grep -q '"ok"' && { up=1; break; }
        sleep 1
    done
    local crc=1
    if [ $up -eq 1 ]; then
        log "SERVER UP $tag after ${i}s"
        bash -c "$CLIENT" > "$RES/$tag.client.out" 2> "$RES/$tag.client.err"; crc=$?
    else
        log "SERVER DID NOT COME UP $tag"
    fi
    docker stop -t 30 p1run > /dev/null 2>&1
    docker wait p1run > /dev/null 2>&1
    docker logs p1run > "$RES/$tag.out" 2> "$RES/$tag.err"
    docker rm p1run > /dev/null
    kill $WD_PID 2>/dev/null; wait $WD_PID 2>/dev/null
    smi_stop
    echo $(( up == 1 && crc == 0 ? 0 : 1 )) > "$RES/$tag.exit"
    log "END $tag exit=$(cat $RES/$tag.exit) (client rc=$crc)"
    # server logs contain request lines; only the client's stderr and fatal server messages count as errors
    grep -i -E "error|exception|traceback" "$RES/$tag.client.err" 2>/dev/null | head -3 > "$RES/$tag.errgrep"
    grep -i -E "out of memory|assert|abort|illegal|segmentation|CUDA error" "$RES/$tag.err" 2>/dev/null | head -3 >> "$RES/$tag.errgrep"
    [ -f "$ALERT" ] && { log "ALERT: $(cat $ALERT)"; return 98; }
    if [ -s "$RES/$tag.errgrep" ] || [ "$(cat $RES/$tag.exit)" != 0 ]; then
        alert "LOGERROR $tag: $(head -2 $RES/$tag.errgrep | tr '\n' ' ' | cut -c1-300)"; log "ALERT: $(tail -1 $ALERT)"; return 98
    fi
    return 0
}
