#!/usr/bin/env python3
"""preveri_naloge.py (2026-11-18, q4hitri): checks koda_naloge.py offline, without a model.
- every FUNKCIJE reference passes its tests, and its tests fail on an empty answer;
- every UREJANJE original FAILS its tests (so the tests see whether the edit was made) and the original with ref_edit
  applied PASSES them;
- the "orodje" originals have no tab or trailing spaces that a model could not copy back from the numbered read.
Prints one line per task and "KONEC uspeh" / "KONEC napaka N"."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import koda_naloge as K

bad = 0
ids = set()
for fid, prompt, tests, ref in K.FUNKCIJE:
    assert fid not in ids, fid
    ids.add(fid)
    ok, err = K.run_code(K.code_of("```python\n" + ref + "\n```"), tests)
    empty, _ = K.run_code("", tests)
    good = ok and not empty
    bad += not good
    print(f"F {fid:12s} ref {'ok' if ok else 'NAPAKA'} prazno {'pade' if not empty else 'USPE'} {'' if ok else err[-200:]}")
for uid, mode, path, original, instr, tests, pairs in K.UREJANJE:
    assert uid not in ids, uid
    ids.add(uid)
    orig_ok, _ = K.run_code(original, tests)
    try:
        edited = K.apply_ref(original, pairs)
        ref_ok, err = K.run_code(edited, tests)
    except AssertionError as e:
        ref_ok, err = False, f"ref_edit: {e}"
    clean = all("\t" not in l and l == l.rstrip() for l in original.split("\n"))
    good = ref_ok and not orig_ok and clean
    bad += not good
    print(f"U {uid:12s} {mode:6s} izvirnik {'pade' if not orig_ok else 'USPE'} ref {'ok' if ref_ok else 'NAPAKA'}"
          f"{'' if clean else ' NECISTO'} vrstic {original.count(chr(10))} {'' if ref_ok else err[-300:]}")
print(f"nalog {len(K.FUNKCIJE)} + {len(K.UREJANJE)}")
print("KONEC uspeh" if not bad else f"KONEC napaka {bad}")
