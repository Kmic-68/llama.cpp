#!/usr/bin/env bash
# Assemble the release bundle on /mnt/fast from this repo.
#
# The bundle's prose is authored in ./bundle/ and installed wholesale, so it is never a base
# document with corrections stacked underneath it -- that layering is what made the 09-05 bundle
# hard to read. Edit ./bundle/, run this, done.
#
# If /mnt/fast is mounted READ-ONLY (fuseblk,ro), Windows left the NTFS volume dirty (Fast
# Startup / hibernation). Two ways back to rw, in order of safety:
#   1. Boot Windows, `powercfg /h off` as admin, full Shut down (not Restart), boot Linux.
#   2. With nothing using the mount:
#        sudo umount /mnt/fast && sudo ntfsfix -d /dev/nvme0n1p2 && sudo mount /mnt/fast
#      (used on 2026-09-15; ntfsfix clears the dirty flag and drops the hibernation state, so
#      any Windows fast-startup session that was suspended on that volume is lost.)
#
# Run ./refresh-build.sh first if the binaries, patches and diffs also need rebuilding.
#
# Usage:  ./sync.sh [/mnt/fast/p100-llamacpp-release]
set -euo pipefail

REL="${1:-/mnt/fast/p100-llamacpp-release}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

[ -d "$REL" ] || { echo "not found: $REL  (is /mnt/fast mounted?)"; exit 1; }
if ! touch "$REL/.wtest" 2>/dev/null; then
    echo "$REL is not writable. /mnt/fast is probably mounted ro -- see the header of this script."
    exit 1
fi
rm -f "$REL/.wtest"

echo "==> authored documentation"
install -Dm644 "$HERE/bundle/README.md"   "$REL/README.md"
install -Dm644 "$HERE/bundle/CHANGES.md"  "$REL/CHANGES.md"
for f in "$HERE"/bundle/docs/*.md; do
    install -Dm644 "$f" "$REL/docs/$(basename "$f")"
    echo "    docs/$(basename "$f")"
done
install -Dm644 "$HERE/docs/AUDIT-2026-09-12.md" "$REL/docs/AUDIT-2026-09-12.md"

echo "==> captured environment"
{
    echo "# Environment (captured $(date -u '+%Y-%m-%d %H:%M UTC'))"
    echo
    echo "## GPUs"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    echo
    echo "## CUDA"
    nvcc --version 2>/dev/null | tail -2 || echo "nvcc not on PATH"
    echo
    echo "## CPU / RAM"
    lscpu | grep -E '^(CPU\(s\)|Model name|Thread|Core)' || true
    free -g | head -2
    echo
    echo "## OS / kernel"
    uname -srm
    lsb_release -d 2>/dev/null || true
    echo
    echo "## Models"
    du -h --apparent-size /mnt/fast/models/*.gguf 2>/dev/null || true
} > "$REL/docs/ENVIRONMENT.md"

echo "==> logs and handoff"
install -Dm644 "$REPO/OPTLOG.md"                    "$REL/logs/OPTLOG.md"
install -Dm644 "$REPO/p100-handoff/VERIFICATION.md" "$REL/docs/handoff/VERIFICATION.md"
install -Dm644 "$REPO/p100-handoff/RESUME-HERE.md"  "$REL/docs/handoff/RESUME-HERE.md"
# Two shas, because they can legitimately differ: docs-only commits move HEAD without
# invalidating build/. diffs/HEAD-SHA.txt is written by refresh-build.sh and describes the binaries.
{
    echo "source HEAD at last sync: $(cd "$REPO" && git rev-parse HEAD)"
    if [ -f "$REL/diffs/HEAD-SHA.txt" ]; then
        echo "build/ was built from:    $(tr -d '[:space:]' < "$REL/diffs/HEAD-SHA.txt")"
    fi
} > "$REL/docs/handoff/HEAD-sha.txt"
(cd "$REPO" && git log --oneline -40) > "$REL/docs/handoff/commit-log.txt"
for f in lcb_em.json lcb_hard.json lcb_greedy_BROKEN.json lcb-report.html; do
    [ -f "$REPO/p100-handoff/$f" ] && install -Dm644 "$REPO/p100-handoff/$f" "$REL/docs/handoff/$f"
done

echo "==> archive the superseded 2026-09-06 release"
# Kept, not deleted: the before/after numbers in OPTLOG are against these binaries. One tarball
# instead of three top-level directories that look like part of the current release.
ARCH="$REL/archive"
install -Dm644 "$HERE/bundle/archive-README.md" "$ARCH/README.md"
OLD=()
for d in build-2026-09-06 patches-2026-09-06 diffs-2026-09-06; do
    [ -d "$REL/$d" ] && OLD+=("$d")
done
if [ ${#OLD[@]} -gt 0 ]; then
    if [ -f "$ARCH/2026-09-06-bundle.tar.zst" ]; then
        echo "    archive already exists; leaving ${OLD[*]} in place -- remove by hand if intended"
    else
        want=$(cd "$REL" && find "${OLD[@]}" ! -type d | wc -l)
        tar --zstd -cf "$ARCH/2026-09-06-bundle.tar.zst" -C "$REL" "${OLD[@]}"
        got=$(tar --zstd -tf "$ARCH/2026-09-06-bundle.tar.zst" | grep -vc '/$')
        echo "    packed ${OLD[*]} -> archive/2026-09-06-bundle.tar.zst ($(du -h "$ARCH/2026-09-06-bundle.tar.zst" | cut -f1))"
        if [ "$want" -ne "$got" ]; then
            echo "    ARCHIVE INCOMPLETE: $want files on disk, $got in the tarball. Originals kept."
            exit 1
        fi
        echo "    verified $got files; removing the loose copies"
        for d in "${OLD[@]}"; do rm -rf "${REL:?}/$d"; done
    fi
else
    echo "    already archived"
fi

echo "==> removing superseded files"
# logs/commit-log.txt froze at session 7 and is replaced by docs/handoff/commit-log.txt, which is
# regenerated above. OPTLOG-README.md is a tombstone pointing at ../OPTLOG.md, a path that exists
# in the repo but not in the bundle, where the log is at logs/OPTLOG.md.
for stale in docs/handoff/OPTLOG-README.md logs/commit-log.txt; do
    if [ -e "$REL/$stale" ]; then rm -f "$REL/$stale"; echo "    removed $stale"; fi
done

echo "==> checking the counts the prose claims"
# These drifted badly once (README said 48 patches when there were 165, and 139 OPTLOG entries
# when the highest attempt was 153). Assert them instead of trusting prose.
check() {   # $1 = what, $2 = actual, $3 = file, $4 = regex capturing the claimed number
    claimed=$(grep -oE "$4" "$3" | head -1 | grep -oE '[0-9]+' | head -1)
    if [ "$claimed" = "$2" ]; then
        echo "    ok: $1 = $2"
    else
        echo "    MISMATCH: $1 is $2 but $(basename "$3") says $claimed"
        MISMATCH=1
    fi
}
MISMATCH=0
n_patch=$(ls "$REL"/patches/*.patch 2>/dev/null | wc -l)
n_attempt=$(grep -oE '^## Attempt [0-9]+' "$REL/logs/OPTLOG.md" | awk '{print $3}' | sort -n | tail -1)
n_code=$(cd "$REPO" && git log --format='%s' "$(tr -d '[:space:]' < "$REL/diffs/UPSTREAM-BASE-SHA.txt")..$(tr -d '[:space:]' < "$REL/diffs/HEAD-SHA.txt")" \
         | grep -cE '^(cuda|fix|perf|precision|revert|spec|tests?):')
check "patch files"   "$n_patch"   "$REL/README.md"   '[0-9]+ `git am`-able commits'
check "OPTLOG attempts" "$n_attempt" "$REL/README.md" '[0-9]+ attempts'
check "code commits"  "$n_code"    "$REL/CHANGES.md"  '[0-9]+ code commits'
[ "$MISMATCH" = 0 ] || { echo "    fix bundle/*.md and re-run"; exit 1; }

echo
echo "done. HEAD $(cd "$REPO" && git rev-parse --short HEAD)."
echo "Binaries, patches and diffs are refreshed separately by ./refresh-build.sh"
