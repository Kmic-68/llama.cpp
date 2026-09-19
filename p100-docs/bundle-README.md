# llama.cpp CUDA optimizations for 2x Tesla P100 (sm_60)

Kernel work targeting Pascal, which upstream llama.cpp largely leaves on generic paths.
Measured on **2x Tesla P100-PCIE-16GB**, tensor-split, **Qwen3.8-27B Q6_K**, q4_0 KV cache.

Built from upstream `f280b2698` (2026-08-25) plus 198 commits, 63 of which touch code.

## Results

| | upstream | here |
|---|---|---|
| decode `tg256` | 17.51 t/s | **30.64 ± 0.19** (1.75x) |
| prefill `pp2048` | 222.6 t/s | **411.5** (1.85x) |
| decode @ 229k context, plain | — | **21.5 t/s** |
| decode @ 229k context, MTP | — | **23.2 t/s** |
| perplexity, `-c 4096` | — | **2.6097 ± 0.0198** (gate band 2.6209 ± 0.0199) |
| `test-backend-ops` | — | FLASH_ATTN_EXT 3961/3961, full suite 14593/14593, both GPUs |

Accuracy is *better* than upstream, not traded away: against an all-fp32 reference, upstream's
prefill matmuls sit at +0.00414 nats/token and this build at +0.00343.

**If you reproduce the perplexity number, use `tools/gate.sh`.** The band belongs to one specific
corpus; a different one returns 2.7566 on *any* build including stock, and that mix-up has twice
been misread here as a correctness failure.

## Start here

| | |
|---|---|
| **`docs/QUICKSTART.md`** | **start here** — putting it on PATH, and the flags to run |
| `CHANGES.md` | every code change, grouped, with what it measured |
| `docs/FINDINGS.md` | what worked, what failed, and how the measurements lied |
| `docs/BUILD.md` | rebuilding from source |

## Layout

| path | contents |
|---|---|
| **`bin/`** | **wrappers to put on PATH** — same 91 programs, with `LD_LIBRARY_PATH` set for you, plus `qwen-server` for the tuned configuration |
| `build/` | the real binaries and libraries. Prefer `bin/`; see QUICKSTART |
| `patches/` | 198 `git am`-able commits against upstream `f280b2698`, one per commit in apply order |
| `diffs/` | `all-code.diff` (the same delta squashed into one file), `everything.diff` (incl. docs), and the base/HEAD SHAs |
| `docs/` | the documentation above, plus `ENVIRONMENT.md`, `COMMUNITY-NOTES.md`, the numerical audit, and `handoff/` |
| `logs/OPTLOG.md` | every attempt, kept and reverted, with numbers — 174 attempts |
| `tools/` | `gate.sh` and the gate corpus — use these, not a hand-typed perplexity command |
| `archive/` | the superseded 2026-09-06 release. **Those binaries carry two data races**; see `archive/README.md` |

## Honesty

This is a **private fork**, not an upstream contribution, and nothing here has been through
upstream review. Several changes are specific to this model shape (head size 256, GQA ratio 6)
and are gated so other configurations take the stock path — which means they are untested
elsewhere, not proven safe there.

Two of the correctness fixes ship **on inspection, with no experiment behind them**, and one
defect is known and unexplained. Both are in `CHANGES.md` under "Known gaps"; neither affects a
normal two-GPU run.

`logs/OPTLOG.md` records the failures alongside the wins, including several measurements that
were wrong and had to be retracted. If you only read one thing before trusting a number, read
"How the measurements lied" in `docs/FINDINGS.md`.
