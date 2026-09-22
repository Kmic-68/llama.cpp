# llama.cpp for 2x Tesla P100: release bundle

Prebuilt binaries of the [P100 fork](https://github.com/Kmic-68/llama.cpp/tree/p100-optimizations)
for two Tesla P100-PCIE-16GB cards (sm_60), tuned for Qwen3.8-27B Q6_K with a q4_0 KV cache.
The build includes upstream llama.cpp up to `f46bc30cb` (2026-09-22).

| | upstream at the fork point | this build |
|---|---|---|
| decode, `tg256` | 17.51 t/s | **30.6 t/s** |
| perplexity (gate corpus, `-c 4096`) | — | **2.6101 ± 0.0198** (band 2.6209 ± 0.0199) |

## Start here

Put `bin/` on PATH, then run `qwen-server`:

    export PATH="/mnt/fast/p100-llamacpp-release/bin:$PATH"
    qwen-server

`docs/QUICKSTART.md` explains every flag, the VRAM budget, and the vision variant.

## Layout

| path | contents |
|---|---|
| `bin/` | wrappers for every program, with `LD_LIBRARY_PATH` set to this bundle, plus `qwen-server`. Use these, not `build/` |
| `build/` | the real binaries and libraries |
| `CHANGES.md` | every code change, with what it measured |
| `docs/` | `QUICKSTART.md`, `FINDINGS.md`, `BUILD.md`, `ENVIRONMENT.md`, and the numerical audit |
| `diffs/` | the whole fork as one diff against its upstream base (`all-code.diff`), plus the base and HEAD SHAs |
| `logs/OPTLOG.md` | every attempt, kept or reverted, with numbers |
| `tools/` | `gate.sh` and its perplexity corpus. Use it to check a build |
| `archive/` | superseded releases. See `archive/README.md` before running any of them |

This is a personal fork, and none of it has been through upstream review. Shape-specific
changes (head size 256, GQA ratio 6) are gated, so other models take the stock path. That means
they're untested elsewhere, not proven safe there.
