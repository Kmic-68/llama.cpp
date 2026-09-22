#!/usr/bin/env bash
# Assemble the release bundle on /mnt/fast from this repo.
#
# The bundle's prose is authored in ../../p100-docs/ (top level, so it is browsable on the fork's
# GitHub page) and installed wholesale, so it is never a base
# document with corrections stacked underneath it -- that layering is what made the 09-05 bundle
# hard to read. Edit p100-docs/, run this, done.
#
# If /mnt/fast is mounted READ-ONLY (fuseblk,ro), Windows left the NTFS volume dirty (Fast
# Startup / hibernation). Two ways back to rw, in order of safety:
#   1. Boot Windows, `powercfg /h off` as admin, full Shut down (not Restart), boot Linux.
#   2. With nothing using the mount:
#        sudo umount /mnt/fast && sudo ntfsfix -d /dev/nvme0n1p2 && sudo mount /mnt/fast
#      (used on 2026-09-15; ntfsfix clears the dirty flag and drops the hibernation state, so
#      any Windows fast-startup session that was suspended on that volume is lost.)
#
# Run ./refresh-build.sh first if the binaries and diffs also need rebuilding.
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
DOCS="$REPO/p100-docs"
install -Dm644 "$DOCS/bundle-README.md" "$REL/README.md"
install -Dm644 "$DOCS/CHANGES.md"       "$REL/CHANGES.md"
for f in BUILD FINDINGS QUICKSTART; do
    install -Dm644 "$DOCS/$f.md" "$REL/docs/$f.md"
    echo "    docs/$f.md"
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
install -Dm644 "$REPO/HANDOFF.md"                  "$REL/docs/handoff/HANDOFF.md"
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

echo "==> bin/ wrappers"
"$HERE/make-wrappers.sh" "$REL"

echo "==> archive README"
# refresh-build.sh packs each outgoing release into archive/; this only installs the index.
install -Dm644 "$REPO/p100-docs/archive-README.md" "$REL/archive/README.md"

echo "==> removing superseded files"
# Files earlier syncs installed that no longer exist in the repo: the pre-merge handoff notes, the
# community notes (folded into FINDINGS), and the per-commit patch series.
for stale in docs/handoff/OPTLOG-README.md docs/handoff/RESUME-HERE.md docs/COMMUNITY-NOTES.md logs/commit-log.txt; do
    if [ -e "$REL/$stale" ]; then rm -f "$REL/$stale"; echo "    removed $stale"; fi
done

echo "==> checking the docs"
# Placeholders like @@TG@@ are filled from measured results before a release. Refuse to ship one.
if grep -rl '@@[A-Z_]*@@' "$REL"/*.md "$REL"/docs/*.md "$REL"/archive/README.md 2>/dev/null; then
    echo "    unfilled placeholders in the files above -- fill p100-docs/*.md and re-run"
    exit 1
fi
echo "    ok"

echo
echo "done. HEAD $(cd "$REPO" && git rev-parse --short HEAD)."
echo "Binaries and diffs are refreshed separately by ./refresh-build.sh"
