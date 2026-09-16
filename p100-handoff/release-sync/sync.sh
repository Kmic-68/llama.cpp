#!/usr/bin/env bash
# Sync the audit record and the refreshed logs into the release bundle on /mnt/fast.
#
# If /mnt/fast is mounted READ-ONLY (fuseblk,ro), Windows left the NTFS volume dirty (Fast
# Startup / hibernation). Two ways back to rw, in order of safety:
#   1. Boot Windows, `powercfg /h off` as admin, full Shut down (not Restart), boot Linux.
#   2. With nothing using the mount:
#        sudo umount /mnt/fast && sudo ntfsfix -d /dev/nvme0n1p2 && sudo mount /mnt/fast
#      (used on 2026-09-15; ntfsfix clears the dirty flag and drops the hibernation state, so
#      any Windows fast-startup session that was suspended on that volume is lost.)
#
# Run ./refresh-build.sh first if the bundle's binaries and patches also need updating.
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

echo "==> new audit document"
install -Dm644 "$HERE/docs/AUDIT-2026-09-12.md" "$REL/docs/AUDIT-2026-09-12.md"

echo "==> refreshed logs and handoff"
install -Dm644 "$REPO/OPTLOG.md"                   "$REL/logs/OPTLOG.md"
install -Dm644 "$REPO/p100-handoff/VERIFICATION.md" "$REL/docs/handoff/VERIFICATION.md"
for f in lcb_em.json lcb_hard.json lcb_greedy_BROKEN.json lcb-report.html; do
    [ -f "$REPO/p100-handoff/$f" ] && install -Dm644 "$REPO/p100-handoff/$f" "$REL/docs/handoff/$f"
done

echo "==> correction notices on the docs that carry now-false claims"
note_once() {   # $1 = file, $2 = marker, $3 = notice file
    [ -f "$1" ] || { echo "    (missing) $1"; return; }
    grep -q "$2" "$1" 2>/dev/null && { echo "    (already noted) $1"; return; }
    printf '\n' >> "$1"
    cat "$3" >> "$1"
    echo "    appended to $1"
}

note_once "$REL/docs/FINDINGS.md"   "AUDIT-2026-09-12" "$HERE/notices/FINDINGS-note.md"
note_once "$REL/docs/QUICKSTART.md" "AUDIT-2026-09-12" "$HERE/notices/QUICKSTART-note.md"
note_once "$REL/README.md"          "AUDIT-2026-09-12" "$HERE/notices/README-note.md"

echo
echo "done. The bundle's binaries, patches and diffs are refreshed by ./refresh-build.sh"
echo "(HEAD $(cd "$REPO" && git rev-parse --short HEAD))."
