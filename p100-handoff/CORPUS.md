# The perplexity corpus drifted — read this before trusting any PPL number

## What happened

`CLAUDE.md` states the gate as **2.6209 +/- 0.0199** and calls anything outside
2.601-2.641 a correctness failure. That number is real and was correctly
measured — on a corpus that no longer exists in the working tree.

The corpus was built with:

    cd ~/llama.cpp
    cat README.md docs/*.md docs/**/*.md 2>/dev/null | head -c 800000 > /tmp/ppl.txt

That recipe is **not stable over time** — it reads whatever the docs happen to
say at that commit. The docs changed in the Aug 24 upstream pull:

| commit | date | corpus bytes |
|---|---|---|
| `6d0549831` (clone) | Aug 18 | **420,098**  <- the gate was measured on this |
| `d59d455fd` | Aug 19 | 420,098 |
| `f280b2698` (Aug 24 pull) | Aug 24 | **422,246**  <- `./ppl.txt` is this |
| `b44f8fe6f` | Aug 29 | 422,246 |

So `./ppl.txt` (422,246 bytes) is a *different document* from the one the gate
was calibrated against, and it reads **2.7554 +/- 0.0215** instead. Nothing
regressed; the ruler changed.

## Proof no code regressed

Same corpus (422,246), three builds, identical to every digit and every chunk:

| build | commit | PPL |
|---|---|---|
| upstream stock | `f280b2698` | 2.7554 +/- 0.02151 |
| prior session's `build-faq` | ~`b44f8fe6f` | 2.7554 +/- 0.02151 |
| this work | `2c1f89b12` | 2.7554 +/- 0.02151 |

## The original corpus, reconstructed

`ppl-orig.txt` in this folder is the Aug-18 corpus rebuilt from git history,
420,098 bytes. Regenerate it with:

    cd ~/llama.cpp
    { git cat-file -p 6d0549831:README.md
      for f in $(git ls-tree -r --name-only 6d0549831 -- docs | grep -E '^docs/[^/]+\.md$' | sort); do git cat-file -p "6d0549831:$f"; done
      for f in $(git ls-tree -r --name-only 6d0549831 -- docs | grep -E '^docs/[^/]+/[^/]+\.md$' | sort); do git cat-file -p "6d0549831:$f"; done
    } | head -c 800000 > ppl-orig.txt

## Reference values (original 420,098-byte corpus, -c 4096, -fa on, -sm tensor)

| KV config | PPL | per-chunk [1] |
|---|---|---|
| f16 / f16 | 2.6175 +/- 0.0199 | |
| q8_0 / q8_0 | 2.6170 +/- 0.0199 | |
| q8_0 / q4_0 | 2.6204 +/- 0.0199 | |
| **q4_0 / q4_0** | **2.6209 +/- 0.01994** | 4.9923 |

New 422,246-byte corpus, q4_0/q4_0: **2.7554 +/- 0.02151**, chunk [1] 4.8657.
The two differ from chunk 1 onward — the signature of different input text,
not different arithmetic.

## Rule going forward

**Pin the corpus, never regenerate it.** A perplexity gate is only meaningful
against a fixed document. Use `ppl-orig.txt` (420,098 bytes) with the
2.6209 target, or `./ppl.txt` (422,246) with the 2.7554 target — but never
compare a number from one to a target from the other.


## Session 7 confirmation (2026-09-05)

Verified directly rather than inferred: `./ppl.txt` yields **2.7566 +/- 0.0215 on any
build**, including one with the session's kernel path disabled at runtime
(`GGML_CUDA_FA_TILE_Q4_0=0`), which reproduces the identical figure. So a run against
`./ppl.txt` can never match CLAUDE.md's stated 2.6209 gate, on any build, stock or not.

**Use `p100-handoff/ppl-orig.txt`** -> 2.6186 +/- 0.0199, inside the 2.6209 +/- 0.0199 gate.

The root `HANDOFF.md`'s 2.7554 is the `./ppl.txt` figure from session 1 and is correct for
that corpus.
