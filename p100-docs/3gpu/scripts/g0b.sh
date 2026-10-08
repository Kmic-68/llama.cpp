#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
# runs after G4: compile (no measurement running) then the two extra G0 microbenchmarks through the harness
cd "${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
until grep -q "PHASE1 DONE\|PHASE1 STOPPED (g4 resume)" run.log; do sleep 30; done
grep -q "PHASE1 STOPPED (g4 resume)" run.log && exit 1
[ -f ALERT ] && exit 1
source ./lib.sh
log "=== G0b start (compile)"
docker run --rm -v $P1/p2pbench:/w -w /w $IMG nvcc -O3 -arch=sm_60 -Wno-deprecated-gpu-targets -o p2pbench2 p2pbench2.cu > $RES/g0b_build.log 2>&1 || { log "G0b BUILD FAILED"; exit 1; }
run g0b_latency_all2all -- timeout 600 /phase1/p2pbench/p2pbench2
log "=== G0b done"
