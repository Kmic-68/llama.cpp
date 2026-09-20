#!/bin/bash
# Flash-attention kernel time vs KV depth for the q4_0 cache, on whatever is in build-opt.
# $1 = label written into the output filename.
set -u
ulimit -c 0
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=/home/kaden/llama-opt/build-opt/bin
LABEL=${1:-run}

# Wait for the gate to release the GPU. Match on the log's own completion marker, never on
# pgrep -- the pattern matches this script's command line and the wait never ends.
for i in $(seq 1 900); do
    grep -q "gate.sh --full" $S/gate_audit.log 2>/dev/null && break
    sleep 4
done
sleep 20

cd /home/kaden/llama-opt
GGML_CUDA_P2P=1 $B/test-backend-ops perf -o FLASH_ATTN_EXT > $S/faperf_$LABEL.raw 2>&1
echo "exit $?" >> $S/faperf_$LABEL.raw

# The q4_0 rows that matter: type_KV=q4_0, DKQ=256, nh=2, nr23=[6,1]
grep -E "hsk=256.*type_KV=q4_0" $S/faperf_$LABEL.raw \
  | sed -E 's/.*kv=([0-9]+).*nb=([0-9]+).*/kv=\1 nb=\2 &/' > $S/faperf_$LABEL.q4_0 2>/dev/null
echo "faperf $LABEL done"
