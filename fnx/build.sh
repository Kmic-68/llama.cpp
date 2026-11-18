#!/bin/bash
# build.sh: builds build/split/strata-split-2, the all-cards split engine for two Tesla P100 (sm_60), with the flags of
# the production binary. Linux only.
#
#   bash build.sh                                   # nvcc from PATH, host compiler g++-13 if present
#   CUDA=/usr/local/cuda-12.4 JOBS=8 bash build.sh   # a toolkit root and the parallel jobs
#   PF_JOB=64 bash build.sh                         # the default until 17 Nov 2026: prompt-path jobs of 64 rows
#
# Environment:
#   PF_JOB    rows of one expert per prompt-path job, a compile-time constant: 96 (default: our production for all three
#             models, UD-Q4_K_XL since 16 Nov 2026, Edge0-145B and IQ3_S since 17 Nov) or 64 (our production until then:
#             the same bits, long prompts read slower). A binary built with either value runs every model.
#   CUDA      toolkit root (its bin/nvcc is used); empty: nvcc from PATH. Must be CUDA 12.x: CUDA 13 dropped sm_60.
#   HOSTCXX   host C++ compiler for nvcc and the link; default g++-13, else g++. CUDA 12.4 accepts gcc up to 13.
#   JOBS      parallel compile jobs (default 8). Each nvcc job takes 1-2 GB of RAM; never build beside a loaded engine
#             on a box whose RAM the engine fills (it pins most of it).
#   B         build directory (default ./build; another one needs profile/fill_profile.py --exe)
#   GG        llama.cpp checkout (default ./third_party/llama.cpp); fetched at the pinned commit when missing.
#
# Steps: fetch and check the pinned llama.cpp headers (only headers are used; ggml is not built), regenerate
# engine/src/iq3s_x.cuh from them and compare, configure, build, then refuse a binary that can call a CUDA peer-access
# function (the engine exchanges sums through pinned host memory only), and warn if it cannot find its CUDA libraries.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
B=${B:-$HERE/build}
JOBS=${JOBS:-8}
PF_JOB=${PF_JOB:-96}
GG=${GG:-$HERE/third_party/llama.cpp}
LLAMA_REPO=${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp.git}
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d   # the commit Strata v0.2.9 and v0.2.10 both pin (third_party/ggml/VERSION.txt)

say() { echo "build.sh: $*"; }
die() { echo "build.sh: $*" >&2; exit 1; }

case $PF_JOB in 64|96) ;; *) die "PF_JOB=$PF_JOB: 64 or 96 (the P100 kernels' jobs fill at most two tiles of 48 rows)";; esac

# --- toolchain
. "$HERE/scripts/toolchain.sh"
pick_toolchain
say "nvcc $CUDA_VER ($NVCC), host compiler $("$HOSTCXX" -dumpfullversion 2>/dev/null || echo '?') ($HOSTCXX), $(cmake --version | head -1)"

# --- the pinned llama.cpp headers
if ! git -C "$GG" rev-parse -q --verify HEAD >/dev/null 2>&1; then   # missing, or a fetch that was cut short
  say "fetching llama.cpp $LLAMA_COMMIT into $GG"
  mkdir -p "$GG"
  [ -d "$GG/.git" ] || git -C "$GG" init -q
  git -C "$GG" remote get-url origin >/dev/null 2>&1 || git -C "$GG" remote add origin "$LLAMA_REPO"
  git -C "$GG" fetch -q --depth 1 origin "$LLAMA_COMMIT"
  git -C "$GG" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
have=$(git -C "$GG" rev-parse HEAD)
[ "$have" = "$LLAMA_COMMIT" ] || die "$GG is at $have, not the pinned $LLAMA_COMMIT"
[ -z "$(git -C "$GG" status --porcelain -- ggml)" ] || die "$GG has local changes under ggml/"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
bash "$HERE/scripts/gen_iq3s_x.sh" "$GG" "$tmp" >/dev/null || die "scripts/gen_iq3s_x.sh failed on $GG"
cmp -s "$tmp" "$HERE/engine/src/iq3s_x.cuh" || die "engine/src/iq3s_x.cuh does not match what $GG's vecdotq.cuh gives"
say "llama.cpp headers at $LLAMA_COMMIT; engine/src/iq3s_x.cuh matches them"

# --- configure and build (the production build: Release, sm_60, 2 cards, prompt-path jobs of 96 rows)
cmake -S "$HERE" -B "$B" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" -DCMAKE_CXX_COMPILER="$HOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=60 -DSTRATA_SPLIT_ARCH=60 -DSTRATA_SPLIT_CARDS=2 -DSTRATA_SPLIT_PF_JOB="$PF_JOB" \
  -DSTRATA_GGML_DIR="$GG" ${CUDA:+-DCUDAToolkit_ROOT="$CUDA"}
say "building with $JOBS jobs, prompt-path jobs of $PF_JOB rows (a full build took about 20 minutes on 2 x Xeon Bronze 3106; kernels_p2.cu alone about 17)"
nice -n 10 cmake --build "$B" --target strata-split-2 -j "$JOBS"

# --- checks on the result
EXE=$B/split/strata-split-2
[ -x "$EXE" ] || die "no $EXE"
peers=$(nm -D --undefined-only "$EXE" | grep -i peer || true)
if [ -n "$peers" ]; then
  echo "$peers" >&2
  die "$EXE can call a CUDA peer-access function; this engine must not"
fi
missing=$(ldd "$EXE" | grep "not found" || true)
if [ -n "$missing" ]; then
  echo "$missing" >&2
  say "WARNING: $EXE cannot find the libraries above; add the toolkit's lib64 to LD_LIBRARY_PATH when you run it"
fi
say "built $EXE"
sha256sum "$EXE"
