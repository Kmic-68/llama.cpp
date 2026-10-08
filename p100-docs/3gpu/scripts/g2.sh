#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
# G2 prefill: pp2048 at depth 0 / 16384 / 65536 (one invocation per depth so cards cool between), then nvprof traces
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
XL=/models/Qwen3.8-27B-UD-Q6_K_XL.gguf; Q8=/models/Qwen3.8-27B-Q8_0.gguf
COMMON="-sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -b 32768 -ub 2048 -ngl 99"
CFGS=("tp3::1/1/1" "p12:CUDA_VISIBLE_DEVICES=1,2:1/1" "p01:CUDA_VISIBLE_DEVICES=0,1:1/1")
log "=== G2 start"
k=0
for d in 0 16384 65536; do for c in "${CFGS[@]}"; do
    IFS=: read -r name envs ts <<< "$c"; e=(); [ -n "$envs" ] && e=(-e "$envs")
    if [ $((k % 2)) -eq 0 ]; then arms="XL Q8"; else arms="Q8 XL"; fi; k=$((k+1))
    for a in $arms; do m=$XL; [ "$a" = Q8 ] && m=$Q8
        run "g2_pp2048_d${d}_${name}_${a}" "${e[@]}" -- ./build-opt/bin/llama-bench -m "$m" $COMMON -ts "$ts" -p 2048 -n 0 -d $d -r 3 -o csv || [ $? -lt 98 ] || exit 1
    done
done; done
log "=== G2 traces start"
for spec in "tp3::1/1/1|XL|16384" "tp3::1/1/1|Q8|16384" "p12:CUDA_VISIBLE_DEVICES=1,2:1/1|XL|16384" "tp3::1/1/1|XL|65536" "tp3::1/1/1|Q8|65536"; do
    c=${spec%%|*}; rest=${spec#*|}; a=${rest%|*}; d=${rest#*|}
    IFS=: read -r name envs ts <<< "$c"; e=(); [ -n "$envs" ] && e=(-e "$envs")
    m=$XL; [ "$a" = Q8 ] && m=$Q8
    run "g2_nvprof_d${d}_${name}_${a}" "${e[@]}" -- /usr/local/cuda/bin/nvprof --print-gpu-trace --csv --normalized-time-unit us \
        --log-file /phase1/results/g2_nvprof_d${d}_${name}_${a}.csv \
        ./build-opt/bin/llama-bench -m "$m" $COMMON -ts "$ts" -p 2048 -n 0 -d $d -r 1 --no-warmup -o csv || [ $? -lt 98 ] || exit 1
done
log "=== G2 done"
