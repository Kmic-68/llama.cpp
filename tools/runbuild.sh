#!/usr/bin/env bash
# Run a binary from a build snapshot AGAINST ITS OWN LIBRARIES.
#
# The snapshots are not self-contained: their binaries carry an absolute
#   RUNPATH = /home/kaden/llama-opt/build-opt/bin
# so invoking $SNAP/bin/llama-perplexity silently loads libggml-cuda.so from build-opt,
# i.e. whatever build happens to be checked out there. That made a pre-fix binary run the
# fixed kernel and produce results identical to seven significant digits.
# RUNPATH (unlike RPATH) loses to LD_LIBRARY_PATH, so setting it is the whole fix.
#
# Usage: runbuild.sh <label> <binary> [args...]
set -uo pipefail
L=$1; B=$2; shift 2
D=/mnt/fast/p100-scratch/build-$L/bin
[ -x "$D/$B" ] || { echo "runbuild: no $B in build-$L" >&2; exit 2; }
exec env LD_LIBRARY_PATH="$D" "$D/$B" "$@"
