#!/usr/bin/env bash
# Replace the release bundle's binaries and diffs with the current HEAD's build.
#
# The previous build/ and diffs/ are packed into archive/<old-stamp>-bundle.tar.zst first, where
# <old-stamp> is the date the outgoing release was built. The stamp is required, not defaulted:
# a default once gave two different releases the same name.
#
# The fork tracks upstream by merging, so the diff base is the merge-base with upstream/master,
# not the original fork point. There is no per-commit patch series any more: after a merge,
# `git format-patch base..HEAD` would emit every upstream commit too.
#
# Run this only after the gates pass on build-opt/ (tools/gate.sh --full).
# Usage: ./refresh-build.sh <old-stamp, e.g. 2026-09-19> [/mnt/fast/p100-llamacpp-release]
set -euo pipefail

OLD_STAMP="${1:?usage: refresh-build.sh <stamp of the release being replaced> [bundle dir]}"
REL="${2:-/mnt/fast/p100-llamacpp-release}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

[ -d "$REL" ] || { echo "not found: $REL  (is /mnt/fast mounted?)"; exit 1; }
touch "$REL/.wtest" 2>/dev/null || { echo "$REL is not writable (mounted ro?)"; exit 1; }
rm -f "$REL/.wtest"
[ -x "$REPO/build-opt/bin/llama-bench" ] || { echo "no build in $REPO/build-opt/bin"; exit 1; }
if ! (cd "$REPO" && git diff --quiet HEAD -- ggml src tests common tools); then
    echo "uncommitted code changes -- commit them first, so the diffs match the binaries"
    (cd "$REPO" && git diff --stat HEAD -- ggml src tests common tools)
    exit 1
fi

BASE=$(cd "$REPO" && git merge-base HEAD upstream/master)
HEAD_SHA=$(cd "$REPO" && git rev-parse HEAD)
echo "==> upstream base $BASE, HEAD $HEAD_SHA"

ARCH="$REL/archive/$OLD_STAMP-bundle.tar.zst"
OLD=()
for d in build diffs patches; do [ -d "$REL/$d" ] && OLD+=("$d"); done
if [ ${#OLD[@]} -gt 0 ]; then
    [ -e "$ARCH" ] && { echo "$ARCH already exists; pick a different stamp"; exit 1; }
    echo "==> packing the outgoing release (${OLD[*]}) into archive/$(basename "$ARCH")"
    mkdir -p "$REL/archive"
    want=$(cd "$REL" && find "${OLD[@]}" ! -type d | wc -l)
    tar --zstd -cf "$ARCH" -C "$REL" "${OLD[@]}"
    got=$(tar --zstd -tf "$ARCH" | grep -vc '/$')
    if [ "$want" -ne "$got" ]; then
        echo "ARCHIVE INCOMPLETE: $want files on disk, $got in the tarball. Nothing removed."
        exit 1
    fi
    echo "    verified $got files ($(du -h "$ARCH" | cut -f1)); removing the loose copies"
    for d in "${OLD[@]}"; do rm -rf "${REL:?}/$d"; done
fi

echo "==> new binaries from build-opt"
mkdir -p "$REL/build"
cp -a "$REPO/build-opt/bin/." "$REL/build/"
# build-opt keeps older versioned libraries (libfoo.so.0.21.0 beside .so.0.24.0) that nothing links
# to any more; drop every real .so file no symlink resolves to
targets=$(find "$REL/build" -maxdepth 1 -type l -name '*.so*' -exec readlink -f {} \;)
for f in $(find "$REL/build" -maxdepth 1 -type f -name '*.so.*'); do
    grep -qxF "$f" <<<"$targets" || { rm -f "$f"; echo "    removed stale $(basename "$f")"; }
done
sha256sum "$(readlink -f "$REL/build/libggml-cuda.so")" | cut -c1-16 | sed 's/^/    libggml-cuda /'

echo "==> diffs against upstream $BASE"
mkdir -p "$REL/diffs"
(cd "$REPO" && git diff "$BASE" HEAD -- ggml src tests common tools > "$REL/diffs/all-code.diff")
(cd "$REPO" && git diff --stat "$BASE" HEAD -- ggml src tests common tools > "$REL/diffs/all-code.stat")
(cd "$REPO" && git diff "$BASE" HEAD > "$REL/diffs/everything.diff")
echo "$HEAD_SHA" > "$REL/diffs/HEAD-SHA.txt"
echo "$BASE" > "$REL/diffs/UPSTREAM-BASE-SHA.txt"
printf 'HEAD %s %s\nupstream base %s %s\n' \
    "$HEAD_SHA" "$(cd "$REPO" && git log -1 --format=%s)" \
    "$BASE" "$(cd "$REPO" && git log -1 --format='%cs %s' "$BASE")" > "$REL/diffs/UPSTREAM-BASE.txt"
tail -1 "$REL/diffs/all-code.stat" | sed 's/^/    /'

echo
echo "done. Now run ./sync.sh to refresh the docs and wrappers."
