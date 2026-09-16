## Numerical audit (2026-09-12) and two data races (2026-09-13, 2026-09-15)

All 19 changed compute files were audited against the standard *"bit-identical to upstream, or
strictly fewer roundings"*. Most changes pass, several with exhaustive machine proof. One was
reverted (cuBLAS `ALGO3`), several defects were fixed, and the follow-up work found **two silent
data races** that `test-backend-ops` cannot see:

1. the GEMM attention softmax wrote probabilities over the scores it was still reading
   (2026-09-13), and
2. an uncompressed tensor-parallel peer copy could overwrite the all-reduce's reduction buffer
   before the destination had read it (2026-09-15) — this one corrupts **decode and MTP**, not
   only long prefill.

**`build/` was rebuilt on 2026-09-16 with both fixes** (plus a 10x more accurate — and at 512-1024
rows much faster — prefill matmul kernel). The bundle exactly as it shipped on 2026-09-06 is kept
in `build-2026-09-06/`, `patches-2026-09-06/` and `diffs-2026-09-06/`; **those binaries carry both
races**. `patches/` and `diffs/` now match `build/`.

**Read `docs/AUDIT-2026-09-12.md` before trusting any accuracy claim in this bundle** — it also
lists the documentation errors it found here.

### Measured on the 2026-09-16 build

Interleaved against the 2026-09-06 binaries in `build-2026-09-06/`, both cards cooled to ≤ 48 °C
before every run, `-sm tensor -fa 1 -ctk q4_0 -ctv q4_0 GGML_CUDA_P2P=1`:

| workload | 2026-09-06 bundle | this build |
|---|---|---|
| prefill pp512 (`-b 2048 -ub 2048`) | 326.9 | **390.1** (+19.3%) |
| prefill pp1024 | 375.5 | **412.7** (+9.9%) |
| prefill pp2048 | 424.2 | 411.5 (-3.0%) |
| prefill pp4096 at the default `-ub 512` | 316.4 | **373.9** (+18.2%) |
| prefill pp2048 at 16384 depth | 372.3 | 363.2 (-2.5%) |
| decode tg256 (upstream baseline 17.51) | 30.85 | 30.64 |
| decode tg128 at 20000 depth | — | 28.35 |
| MTP decode, 256 tokens, greedy | — | 55.0 |
| perplexity, `ppl-orig.txt`, `-c 4096` (band 2.6209 ± 0.0199) | 2.6204 (source build of 09-13) | **2.6097 ± 0.0198** |
| `test-backend-ops` | — | FLASH_ATTN_EXT 3961/3961, full suite 14593/14593, both GPUs |

At the production operating point (`llama-server -c 262144 -b 262144 -ub 2048 -np 1` with the MTP
draft, 19966-token prompt): prompt 350-366 t/s, generation 30.1 t/s, peak VRAM **16137 MiB on
GPU0** (including ~392 MiB of Sunshine) and **15745 MiB on GPU1** of 16384.

The losses at `-ub 2048` are the price of the correctness work (the `ALGO3` revert, the
out-of-place softmax, PV `ALGO4`, and the copy wait, ~0.7% of decode); the gains are the blocked
matmul accumulator, which is also 10x more accurate.
