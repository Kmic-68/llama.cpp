#!/bin/sh
# Kills ONLY the llama-server pid passed in, if GPU0 free memory falls under the floor.
PID="$1"; FLOOR="${2:-200}"
while kill -0 "$PID" 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
    free=$((total-used))
    echo "$(date +%H:%M:%S) gpu0_free=${free}MiB" >> "$GLOG"
    if [ "$free" -lt "$FLOOR" ]; then
        echo "$(date +%H:%M:%S) FLOOR BREACH free=${free} -- killing $PID" >> "$GLOG"
        kill "$PID"
        exit 9
    fi
    sleep 5
done
echo "$(date +%H:%M:%S) server exited; guard done" >> "$GLOG"
