#!/usr/bin/env bash
# Replace the release bundle's prebuilt binaries, patch series and diffs with the current HEAD.
#
# The 2026-09-06 bundle carries two silent data races (the GEMM attention softmax, and an
# uncompressed peer copy overtaking the all-reduce's reader) that were found and fixed afterwards,
# so its binaries must not stay the "ready to run" option. The old bundle is moved aside, not
# deleted: build-<stamp>/, patches-<stamp>/, diffs-<stamp>/.
#
# Run this only after the gates pass on the build in build-opt/ (tools/gate.sh, the FA eval and the
# full op suite). Usage: ./refresh-build.sh [/mnt/fast/p100-llamacpp-release] [stamp]
set -euo pipefail

REL="${1:-/mnt/fast/p100-llamacpp-release}"
STAMP="${2:-2026-09-06}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BASE=$(cat "$REL/diffs/UPSTREAM-BASE-SHA.txt" 2>/dev/null || cat "$REL/diffs-$STAMP/UPSTREAM-BASE-SHA.txt")
HEAD_SHA=$(cd "$REPO" && git rev-parse HEAD)

[ -d "$REL" ] || { echo "not found: $REL  (is /mnt/fast mounted?)"; exit 1; }
touch "$REL/.wtest" 2>/dev/null || { echo "$REL is not writable (mounted ro?)"; exit 1; }
rm -f "$REL/.wtest"
[ -x "$REPO/build-opt/bin/llama-bench" ] || { echo "no build in $REPO/build-opt/bin"; exit 1; }
if ! (cd "$REPO" && git diff --quiet -- ggml src tests common); then
    echo "working tree has uncommitted code changes -- commit them first, so the patch series matches the binaries"
    (cd "$REPO" && git diff --stat -- ggml src tests common)
    exit 1
fi

echo "==> base $BASE, HEAD $HEAD_SHA"

echo "==> moving the old bundle aside"
for d in build patches diffs; do
    if [ -d "$REL/$d" ] && [ ! -d "$REL/$d-$STAMP" ]; then
        mv "$REL/$d" "$REL/$d-$STAMP"
        echo "    $d -> $d-$STAMP"
    else
        echo "    $d: backup already exists or nothing to move"
    fi
done

echo "==> new binaries from build-opt (this takes a minute: ~450 MB)"
mkdir -p "$REL/build"
cp -a "$REPO/build-opt/bin/." "$REL/build/"
sha256sum "$REL/build/libggml-cuda.so.0.21.0" | cut -c1-16 | sed 's/^/    libggml-cuda /'

echo "==> new patch series (every commit since upstream)"
mkdir -p "$REL/patches"
(cd "$REPO" && git format-patch --no-signature --quiet -o "$REL/patches" "$BASE..HEAD" >/dev/null)
echo "    $(ls "$REL/patches" | wc -l) patches"

echo "==> new diffs"
mkdir -p "$REL/diffs"
(cd "$REPO" && git diff "$BASE" HEAD -- ggml src tests common tools > "$REL/diffs/all-code.diff")
(cd "$REPO" && git diff --stat "$BASE" HEAD -- ggml src tests common tools > "$REL/diffs/all-code.stat")
(cd "$REPO" && git diff "$BASE" HEAD > "$REL/diffs/everything.diff")
echo "$HEAD_SHA" > "$REL/diffs/HEAD-SHA.txt"
echo "$BASE" > "$REL/diffs/UPSTREAM-BASE-SHA.txt"
printf 'HEAD %s %s\nupstream base %s\n' "$HEAD_SHA" "$(cd "$REPO" && git log -1 --format=%s)" "$BASE" > "$REL/diffs/UPSTREAM-BASE.txt"
tail -1 "$REL/diffs/all-code.stat" | sed 's/^/    /'

echo
echo "done. Now run ./sync.sh to refresh the docs and append the correction notices."
