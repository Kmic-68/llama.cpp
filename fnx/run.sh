#!/bin/bash
# run.sh DATA [--port 8080] [--host 127.0.0.1] [--config FILE] [--strata DIR]: starts Strata's server with the run config
# setup.sh wrote (DATA/profile.json). The engine loads in about 5 minutes; then GET /health says "loaded": true and the
# OpenAI-compatible API is at http://HOST:PORT/v1. Stop it with Ctrl+C, then check with nvidia-smi that the cards are free.
set -euo pipefail
say() { echo "run.sh: $*"; }
die() { echo "run.sh: $*" >&2; exit 1; }

[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || die "usage: bash run.sh DATA [--port 8080] [--host 127.0.0.1] [--config FILE] [--strata DIR]"
DATA=$1
shift
PORT=8080 HOST=127.0.0.1 CFG='' STRATA=''
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=${2:?--port N}; shift 2 ;;
    --host) HOST=${2:?--host ADDRESS}; shift 2 ;;
    --config) CFG=${2:?--config FILE}; shift 2 ;;
    --strata) STRATA=${2:?--strata DIR}; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done
abspath() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }
DATA=$(abspath "$DATA")
CFG=$(abspath "${CFG:-$DATA/profile.json}")
STRATA=$(abspath "${STRATA:-$DATA/strata-v0.2.9}")
PY=$STRATA/.venv/bin/python
[ -f "$CFG" ] || die "no $CFG: run setup.sh first"
[ -x "$PY" ] && [ -f "$STRATA/serve/server.py" ] || die "no Strata server with its .venv in $STRATA: run setup.sh first"

# what would stop the load, said before the 5 minutes rather than after
EXE=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["exe"])' "$CFG")
[ -x "$EXE" ] || die "the engine $EXE (\"exe\" in $CFG) is missing"
missing=$(ldd "$EXE" | grep "not found" || true)
[ -z "$missing" ] || die "the engine cannot find: $missing; set LD_LIBRARY_PATH to the CUDA 12 libraries it was built with"
avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || true)
kind=$("$PY" -c 'import json, sys; c = json.load(open(sys.argv[1])); m = c.get("model_name", ""); print("q4" if "q4_k_xl" in m else "edge0" if "edge0" in m else c["env"].get("FNX_PLE_RAM", "1"))' "$CFG")
# what the engine holds once loaded: UD-Q4_K_XL, Edge0, the 64 GB mode (fill_profile.py --ram 64) or the default
case $kind in q4) holds=70 ;; edge0) holds=80 ;; 0) holds=46 ;; *) holds=85 ;; esac
if [ -n "$avail" ] && [ "$avail" -lt $((holds - 5)) ]; then
  say "WARNING: $avail GiB of RAM available; the engine holds about $holds GiB once loaded"
fi
if command -v nvidia-smi >/dev/null; then
  cards=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["gpu"])' "$CFG")
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i "$cards" || true)
  busy=$(echo "$used" | awk -F', ' '$2 > 1000 {print "card " $1 ": " $2 " MiB"}')
  [ -z "$busy" ] || say "WARNING: in use already: $busy"
fi

say "Strata's server on http://$HOST:$PORT with $CFG (the engine loads in about 5 minutes)"
cd "$STRATA"
exec "$PY" serve/server.py --engine strata --config "$CFG" --port "$PORT" --host "$HOST"
