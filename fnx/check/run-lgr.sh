#!/usr/bin/env bash
# run-lgr.sh NAME [VAR=VALUE ...]: the engine's logits on the check prompts, scored against llama.cpp's CPU backend.
# Each prompt goes through the prompt path (FNX_PREFILL=1), then the reference's continuation in windows of T rows
# (default 5, MTP's verify width). Env from the production profile (PROFILE, default fnx-math/profile.json), then
# the VAR=VALUE overrides. Writes $D/lgr/NAME-PROMPT.lgr and appends the scores to $D/results.txt.
set -uo pipefail
here=$(dirname "$(readlink -f "$0")")
D=${D:-/mnt/fast/p100-scratch/fnx-math}
PROFILE=${PROFILE:-$D/profile.json}
BIN=${BIN:-$here/../build/split/split_fnx_run}
M=${M:-/mnt/fast/models/Qwen flash Next/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
T=${T:-5}
PROMPTS=${PROMPTS:-code prose}
name=$1; shift
mapfile -t envs < <(python3 -c "import json,sys; [print(f'{k}={v}') for k,v in json.load(open(sys.argv[1]))['env'].items()]" "$PROFILE")
for p in $PROMPTS; do
    out=$D/lgr/$name-$p.lgr
    ( cd "$here/../engine" && env "${envs[@]}" FNX_MAX_CTX=${FNX_MAX_CTX:-4096} FNX_PREFILL=${PREFILL:-1} "$@" \
        "$BIN" "$M" logits "$D/prompts/$p.bin" "$D/lgr/ref-cpu-$p.lgr" "$T" "$out" ) > "$D/lgr/$name-$p.log" 2>&1 \
        || { echo "$name $p: fnx_run failed (see $D/lgr/$name-$p.log)"; tail -3 "$D/lgr/$name-$p.log"; continue; }
    r=$(python3 "$here/lgrcmp.py" "$D/lgr/ref-cpu-$p.lgr" "$out")
    echo "$(date +%m-%d_%H:%M) $name T=$T ${*:-} | $r" | tee -a "$D/results.txt"
done
