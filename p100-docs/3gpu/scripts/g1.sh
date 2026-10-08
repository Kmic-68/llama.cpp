#!/usr/bin/env bash
# Variables (set in the environment before running; defaults in lib.sh):
#   BENCH_DIR  directory holding these scripts, results/ and run.log (default: this script's directory)
#   SRC        llama.cpp checkout with a build in build-opt/ (default: $BENCH_DIR/../src)
#   MODELS     directory with the .gguf files, mounted read-only at /models (default: $HOME/models)
#   IMG        Docker image with the CUDA toolchain the build was made in (default: p100-llamacpp-test:latest)
#   The P2P programs are expected built in $BENCH_DIR/p2pbench (see README.md).
# G1 decode: throughput (ABBA interleaved), then nvprof traces for attribution
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
XL=/models/Qwen3.8-27B-UD-Q6_K_XL.gguf
Q8=/models/Qwen3.8-27B-Q8_0.gguf
COMMON="-sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -b 32768 -ub 2048 -ngl 99"
# name : env : ts
CFGS=("tp3::1/1/1" "p12:CUDA_VISIBLE_DEVICES=1,2:1/1" "p01:CUDA_VISIBLE_DEVICES=0,1:1/1")
bench() { # tag cfgentry modelvar extra
    local tag=$1 c=$2 m=$3; shift 3
    IFS=: read -r name envs ts <<< "$c"
    local e=(); [ -n "$envs" ] && e=(-e "$envs")
    run "$tag" "${e[@]}" -- ./build-opt/bin/llama-bench -m "$m" $COMMON -ts "$ts" "$@"
}
log "=== G1 throughput start"
ORDER=()
for c in "${CFGS[@]}"; do ORDER+=("$c|XL" "$c|Q8"); done
for round in 1 2; do
    if [ $round -eq 1 ]; then seq=("${ORDER[@]}"); else seq=(); for ((i=${#ORDER[@]}-1;i>=0;i--)); do seq+=("${ORDER[$i]}"); done; fi
    for item in "${seq[@]}"; do
        c=${item%|*}; a=${item#*|}; name=${c%%:*}
        m=$XL; [ "$a" = Q8 ] && m=$Q8
        bench "g1_tg512_${name}_${a}_r${round}" "$c" "$m" -p 0 -n 512 -r 3 -o csv || [ $? -lt 98 ] || exit 1
    done
done
log "=== G1 traces start"
for c in "${CFGS[@]}"; do for a in XL Q8; do
    name=${c%%:*}; m=$XL; [ "$a" = Q8 ] && m=$Q8
    for n in 4 36; do
        IFS=: read -r cn envs ts <<< "$c"; e=(); [ -n "$envs" ] && e=(-e "$envs")
        run "g1_nvprof_${name}_${a}_n${n}" "${e[@]}" -- /usr/local/cuda/bin/nvprof --print-gpu-trace --csv --normalized-time-unit us \
            --log-file /phase1/results/g1_nvprof_${name}_${a}_n${n}.csv \
            ./build-opt/bin/llama-bench -m "$m" $COMMON -ts "$ts" -p 0 -n $n -r 1 --no-warmup -o csv || [ $? -lt 98 ] || exit 1
    done
done; done
log "=== G1 done"
