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
install -Dm644 "$REPO/OPTLOG.md"                    "$REL/logs/OPTLOG.md"
install -Dm644 "$REPO/p100-handoff/VERIFICATION.md" "$REL/docs/handoff/VERIFICATION.md"
# RESUME-HERE.md and the HEAD sha were missed until 2026-09-17: the bundle's copies still named
# session 7 and commit aa22ccee0, five sessions and 105 commits behind what build/ actually is.
install -Dm644 "$REPO/p100-handoff/RESUME-HERE.md"  "$REL/docs/handoff/RESUME-HERE.md"
(cd "$REPO" && git rev-parse HEAD) > "$REL/docs/handoff/HEAD-sha.txt"
(cd "$REPO" && git log --oneline -40) > "$REL/docs/handoff/commit-log.txt"
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

echo "==> stale counts and names in the 2026-09-05 prose"
# Plain factual drift, not findings -- a notice at the bottom of the file does not help a reader
# who trusts the table at the top. Idempotent: each one is skipped once the new text is in place.
fix_once() {    # $1 = file, $2 = old literal, $3 = new literal, $4 = what
    [ -f "$1" ] || { echo "    (missing) $1"; return; }
    if ! grep -qF "$2" "$1"; then
        grep -qF "$3" "$1" && echo "    (already fixed) $4" || echo "    (NOT FOUND, check by hand) $4"
        return
    fi
    python3 - "$1" "$2" "$3" <<'PY'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
b = open(p).read()
open(p, "w").write(b.replace(old, new))
PY
    echo "    fixed: $4"
}

R="$REL/README.md"
# general.name is "Qwen3.8 27B Abliterated"; the *architecture* string in the gguf is qwen35,
# which is where "Qwen3.5-27B" came from.
fix_once "$R" '**Qwen3.5-27B Q6_K**' '**Qwen3.8-27B Q6_K**' "README model name"
fix_once "$R" '**31-32 t/s** (**1.8x**)' '**30.6-30.9 t/s** (**1.75x**)' "README decode headline (measured 30.64)"
fix_once "$R" '**perplexity 2.6186-2.6199 +/- 0.0199**' \
              '**perplexity 2.6097-2.6204 +/- 0.0199**' "README perplexity range"
fix_once "$R" '| 48 `git am`-able commits against upstream `f280b2698` |' \
              '| 165 `git am`-able commits against upstream `f280b2698` |' "README patch count"
fix_once "$R" 'with numbers** — 139 entries' 'with numbers** — 153 attempts' "README OPTLOG size"
fix_once "$REL/docs/COMMUNITY-NOTES.md" 'the Qwen3.5-27B shape' 'the Qwen3.8-27B shape' \
         "COMMUNITY-NOTES model name"
fix_once "$REL/docs/QUICKSTART.md" 'Expect **2.6186 +/- 0.0199**.' \
         'Expect **2.6097 +/- 0.0198** (gate band 2.6209 +/- 0.0199).' "QUICKSTART expected perplexity"
fix_once "$R" '| prefill (`pp2048`) | 222.6 t/s | **~440 t/s** (**2.0x**) |' \
              '| prefill (`pp2048`) | 222.6 t/s | **411.5 t/s** (**1.85x**) |' "README prefill headline"
fix_once "$REL/docs/BUILD.md" '# expect 2.6186 +/- 0.0199' \
         '# expect 2.6097 +/- 0.0198, band 2.6209 +/- 0.0199' "BUILD expected perplexity"

echo "==> serving section on QUICKSTART"
note_once "$REL/docs/QUICKSTART.md" "## Serving it for real work" "$HERE/notices/QUICKSTART-serving.md"

echo
echo "done. The bundle's binaries, patches and diffs are refreshed by ./refresh-build.sh"
echo "(HEAD $(cd "$REPO" && git rev-parse --short HEAD))."
