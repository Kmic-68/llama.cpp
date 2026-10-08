#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
# Final stage as run: XL pp65536 tensor only (layer pp65536 was skipped after a power-cap stop), then the G0b extras
cd "${BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source ./lib.sh
log "=== FINISH (final stage) start"
run g4_pp65536_tensor_XL -- ./build-opt/bin/llama-bench -m /models/Qwen3.8-27B-UD-Q6_K_XL.gguf -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -b 32768 -ub 2048 -ngl 99 -ts 1/1/1 -p 65536 -n 0 -r 3 -o csv
rc=$?; [ $rc -ge 98 ] && { log "PHASE1 STOPPED (finish)"; exit 1; }
log "=== G0b start (compile)"
docker run --rm -v $P1/p2pbench:/w -w /w $IMG nvcc -O3 -arch=sm_60 -Wno-deprecated-gpu-targets -o p2pbench2 p2pbench2.cu > $RES/g0b_build.log 2>&1 || { log "G0b BUILD FAILED"; log "PHASE1 STOPPED (finish)"; exit 1; }
run g0b_latency_all2all -- timeout 600 /phase1/p2pbench/p2pbench2
rc=$?; [ $rc -ge 98 ] && { log "PHASE1 STOPPED (finish)"; exit 1; }
log "=== G0b done"
log "PHASE1 DONE (final stage)"
