#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
# G3 per-type matvec at the model's real shapes: fork's per-op GPU profiler, decode (n=1) and verify width (pp5)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
XL=/models/Qwen3.8-27B-UD-Q6_K_XL.gguf; Q8=/models/Qwen3.8-27B-Q8_0.gguf
COMMON="-sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -b 32768 -ub 2048 -ngl 99 -ts 1/1/1"
log "=== G3 start"
for a in XL Q8; do m=$XL; [ "$a" = Q8 ] && m=$Q8
    run "g3_opprof_tg64_tp3_${a}" -e GGML_CUDA_OP_PROFILE=1 -- ./build-opt/bin/llama-bench -m "$m" $COMMON -p 0 -n 64 -r 1 --no-warmup -o csv || [ $? -lt 98 ] || exit 1
    run "g3_opprof_pp5_tp3_${a}"  -e GGML_CUDA_OP_PROFILE=1 -- ./build-opt/bin/llama-bench -m "$m" $COMMON -p 5 -n 0 -r 20 --no-warmup -o csv || [ $? -lt 98 ] || exit 1
done
log "=== G3 done"
