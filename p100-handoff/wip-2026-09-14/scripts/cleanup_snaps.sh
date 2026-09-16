#!/usr/bin/env bash
# Delete the binary snapshots in the scratch directory once the work is committed.
# Explicit file deletes, never rm -rf (CLAUDE.md).
set -uo pipefail
S=/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad
KEEP="${KEEP:-}"          # e.g. KEEP="snap-final3" to hold one back
before=$(df -h / | tail -1 | awk '{print $4}')
for d in snap-vc2 snap-mm snap-oop snap-ipnh snap-oopnh snap-nanip snap-peerfix snap-peerfix2 snap-inject snap-inject2 snap-final snap-final2 snap-final3; do
    case " $KEEP " in *" $d "*) echo "keeping $d"; continue;; esac
    [ -d "$S/$d" ] || continue
    n=$(find "$S/$d" -type f -o -type l | wc -l)
    find "$S/$d" -type f -delete
    find "$S/$d" -type l -delete
    find "$S/$d" -depth -type d -empty -exec rmdir {} +
    echo "deleted $d ($n files)"
done
echo "free on /: $before -> $(df -h / | tail -1 | awk '{print $4}')"
