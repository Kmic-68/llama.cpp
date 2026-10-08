#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
# G4 split mode, 3 GPUs: tensor vs layer, interleaved per test, one invocation per test so cards cool between
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
XL=/models/Qwen3.8-27B-UD-Q6_K_XL.gguf; Q8=/models/Qwen3.8-27B-Q8_0.gguf
COMMON="-fa 1 -ctk q4_0 -ctv q4_0 -b 32768 -ub 2048 -ngl 99 -ts 1/1/1"
log "=== G4 start"
k=0
for t in "tg512:-p 0 -n 512" "pp512:-p 512 -n 0" "pp2048:-p 2048 -n 0" "pp16384:-p 16384 -n 0" "pp65536:-p 65536 -n 0"; do
    tn=${t%%:*}; targs=${t#*:}
    for a in XL Q8; do m=$XL; [ "$a" = Q8 ] && m=$Q8
        if [ $((k % 2)) -eq 0 ]; then modes="tensor layer"; else modes="layer tensor"; fi; k=$((k+1))
        for sm in $modes; do
            run "g4_${tn}_${sm}_${a}" -- ./build-opt/bin/llama-bench -m "$m" -sm $sm $COMMON $targs -r 3 -o csv || [ $? -lt 98 ] || exit 1
        done
    done
done
log "=== G4 done"
