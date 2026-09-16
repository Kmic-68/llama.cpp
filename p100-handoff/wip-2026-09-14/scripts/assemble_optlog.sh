#!/usr/bin/env bash
# Append the rest of attempt 151, attempt 152 and attempt 153 to OPTLOG.md.
set -euo pipefail
cd /home/kaden/llama-opt
A=p100-handoff/wip-2026-09-13/OPTLOG-attempt151-152-draft.md
B=p100-handoff/wip-2026-09-14/OPTLOG-attempt153-draft.md

for f in "$A" "$B"; do
    if grep -q '@@' "$f"; then echo "placeholders left in $f:"; grep -n '@@' "$f"; exit 1; fi
done
if grep -q '^## Attempt 152' OPTLOG.md; then echo "OPTLOG.md already has attempt 152 -- refusing to append twice"; exit 1; fi

if [ -n "$(tail -c 1 OPTLOG.md)" ]; then printf '\n' >> OPTLOG.md; fi   # ensure a trailing newline
printf '\n' >> OPTLOG.md
cat "$A" >> OPTLOG.md
printf '\n' >> OPTLOG.md
cat "$B" >> OPTLOG.md
echo "appended; OPTLOG.md is now $(wc -l < OPTLOG.md) lines"
grep -n '^## Attempt 15[123]' OPTLOG.md
