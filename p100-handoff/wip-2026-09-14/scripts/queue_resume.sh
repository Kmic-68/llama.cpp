#!/usr/bin/env bash
# Resume 09-16: the two checks left from queue_final3 -- the virtual-device race test on the
# Q4_0 model (the Q6_K one OOMs at four CUDA contexts) and the full op suite on snap-final3.
cd /home/kaden/llama-opt
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
B=$S/snap-final3/bin
ulimit -c 0
echo "=== resume $(date +%T): final3 $(sha256sum $B/libggml-cuda.so.0.21.0 | cut -c1-16)"
QPID=999999 bash $S/queue_virt2.sh
echo "=== (J3b) full op suite on final3 ($(date +%T))"
LD_LIBRARY_PATH=$B $B/test-backend-ops test > $S/final/full3.log 2>&1; echo "   full exit=$? ($(date +%T))"
grep -E 'tests passed|backends passed|FAIL' $S/final/full3.log | head -8
echo "=== queue_resume done ($(date +%T))"
