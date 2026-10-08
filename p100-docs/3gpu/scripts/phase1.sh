#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
# orchestrator: wait for G1 (already running), then G2, G3, G4; stops on ALERT
P1=${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}; cd $P1
until grep -q "=== G1 done" run.log || [ -f ALERT ]; do sleep 30; done
for s in g2 g3 g4; do
    [ -f ALERT ] && { echo "[$(date +%T)] PHASE1 STOPPED (ALERT before $s)" >> run.log; exit 1; }
    ./$s.sh || { echo "[$(date +%T)] PHASE1 STOPPED ($s exited $?)" >> run.log; exit 1; }
done
echo "[$(date +%T)] PHASE1 DONE" >> run.log
