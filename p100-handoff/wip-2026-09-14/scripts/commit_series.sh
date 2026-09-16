#!/usr/bin/env bash
# The four code commits of attempt 153, each with the tree as it was at that step.
# Never stages CLAUDE.md or ppl.txt (the user's own uncommitted edits).
set -euo pipefail
cd /home/kaden/llama-opt
W=p100-handoff/wip-2026-09-14/code
M=p100-handoff/wip-2026-09-13/msgs

stage() {   # $1 = repo path, $2 = file whose content to stage
    local blob
    blob=$(git hash-object -w "$2")
    git update-index --cacheinfo "100644,$blob,$1"
}

check_clean_msg() {   # refuse to commit a message that still has placeholders
    grep -q '@@' "$1" && { echo "placeholder left in $1"; exit 1; } || true
}

for m in 6_oop_accum 7_matmul_algo6 8_peer_race 9_compress_type 10_samedev_copy; do check_clean_msg "$M/$m.txt"; done

# 6 -- out-of-place accum_O (fattn-gemm.cu only; ggml-cuda.cu stays at HEAD)
stage ggml/src/ggml-cuda/fattn-gemm.cu "$W/fattn-gemm.cu.oop-accum"
git commit -q -F "$M/6_oop_accum.txt"
echo "6: $(git log -1 --format='%h %s')"

# 7 -- Pascal prefill matmuls request ALGO6
stage ggml/src/ggml-cuda/ggml-cuda.cu "$W/ggml-cuda.cu.final-matmul-only"
git commit -q -F "$M/7_matmul_algo6.txt"
echo "7: $(git log -1 --format='%h %s')"

# 8 -- the uncompressed peer copy waits on the destination's work marker
stage ggml/src/ggml-cuda/ggml-cuda.cu "$W/ggml-cuda.cu.final-matmul+peer-v2"
git commit -q -F "$M/8_peer_race.txt"
echo "8: $(git log -1 --format='%h %s')"

# 9 -- compression decided per exchange from the matmul's compute type
stage ggml/src/ggml-cuda/ggml-cuda.cu "$W/ggml-cuda.cu.final-matmul+peer-v2+compress-type"
git commit -q -F "$M/9_compress_type.txt"
echo "9: $(git log -1 --format='%h %s')"

# 10 -- the same-GPU copy between two virtual devices gets the same wait
stage ggml/src/ggml-cuda/ggml-cuda.cu "$W/ggml-cuda.cu.final-matmul+peer-v2+compress-type+samedev"
git commit -q -F "$M/10_samedev_copy.txt"
echo "10: $(git log -1 --format='%h %s')"

echo
echo "worktree vs HEAD (should be only CLAUDE.md, OPTLOG.md, docs and untracked):"
git status --short
echo
echo "code files identical to HEAD:"
git diff --stat HEAD -- ggml src tests common | tail -2
