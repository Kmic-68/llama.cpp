#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
log "=== G0 start"
run g0_peer -- timeout 300 /phase1/p2pbench/p2pbench
run g0_nopeer -- timeout 300 /phase1/p2pbench/p2pbench --no-peer
log "=== G0 done"
