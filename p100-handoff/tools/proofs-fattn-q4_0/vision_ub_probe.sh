#!/bin/bash
# What -ub does model + MTP + vision need at full context?
#
# -ub 2048 died AT LOAD with 127 MiB free on GPU0, so the load-time reservation alone is the
# binding constraint and a load probe is a valid filter (it is NOT a valid peak -- the prefill
# peak is higher, FINDINGS item 10 -- so the winner still gets a full-depth run afterwards).
# With CUDA graphs off the mask costs ~0.84 MiB per ubatch unit, so each halving should hand
# back roughly 860 / 1290 / 1505 MiB against 2048.
set -u
ulimit -c 0
S=${S:-$(cd "$(dirname "$0")" && pwd)/run}; mkdir -p "$S"
REL=/mnt/fast/p100-llamacpp-release
export LD_LIBRARY_PATH=$REL/build:${LD_LIBRARY_PATH:-}
export GGML_CUDA_P2P=1
export GGML_CUDA_GRAPHS_PRE_VOLTA=0

printf "%-8s %-14s %-14s %s\n" "-ub" "free GPU0" "free GPU1" "outcome"
for UB in 1024 512 256; do
    export GLOG=$S/guard_probe$UB.log; : > $GLOG
    $REL/build/llama-server -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      --mmproj /mnt/fast/models/mmproj-Qwen3.8-27B-Q8_0.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 32768 -ub $UB -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
      -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
      --jinja --host 127.0.0.1 --port 8092 > $S/probe$UB.log 2>&1 &
    SRV=$!
    sh $S/guard.sh $SRV 200 &
    ok=""
    for i in $(seq 1 240); do
        grep -q "listening on" $S/probe$UB.log && { ok=yes; break; }
        kill -0 $SRV 2>/dev/null || break
        sleep 2
    done
    if [ -n "$ok" ]; then
        f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
        f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
        printf "%-8s %-14s %-14s %s\n" "$UB" "${f0} MiB" "${f1} MiB" "loaded"
    else
        printf "%-8s %-14s %-14s %s\n" "$UB" "-" "-" "DIED AT LOAD"
    fi
    kill $SRV 2>/dev/null; sleep 10; kill -9 $SRV 2>/dev/null; sleep 8
done
echo "probe done"
