## Correction (2026-09-12 audit, extended 09-13 and 09-15) — see `AUDIT-2026-09-12.md`

Finding 7 above and the GEMM-attention notes are partly superseded. **The binaries in `build/`
were rebuilt on 2026-09-16 and carry every fix below**; the bundle as it shipped on 2026-09-06 is
kept in `build-2026-09-06/`, `patches-2026-09-06/` and `diffs-2026-09-06/`, and it has both data
races.

- **The GEMM attention path is ON by default** (`GGML_CUDA_FA_GEMM=0` disables it), not opt-in.
- **Both** its GEMMs use `CUBLAS_COMPUTE_16F`. The statement that PV uses `CUBLAS_COMPUTE_32F`
  is stale — it described an earlier revision.
- Its QK^T accumulates k=256 in fp16 where upstream `fattn-tile.cuh:604` uses an **fp32**
  `KQ_acc`. Measured (2026-09-13): the dominant error was not QK^T but PV, and mostly a cuBLAS
  algorithm choice — `CUBLAS_GEMM_DEFAULT` picks a long-chain fp16 kernel above n ≈ 6000. PV now
  asks for `ALGO4`: 3.4x lower op error at the production batch. At the model level the fp16 path
  is not distinguishable from fp32 attention (paired per-chunk perplexity, |t| < 2).
- **Two silent data races, both in the 2026-09-06 binaries, both fixed in the new ones:**
  1. *The GEMM softmax wrote the probabilities over the scores it was still reading.* It
     corrupts a few attention rows per long prompt and in fp16 can NaN the output. Proven with an
     in-op self-check (3-6 of 2240 launches differ in place, 0 of ~6700 out of place).
  2. *An uncompressed tensor-parallel peer copy could overwrite the all-reduce's reduction buffer
     before the destination's ADD had read it* (introduced with the dedicated copy stream,
     a4d1103c5). This one hits **decode, MTP and short prompts**, not just long prefill: with the
     race forced deterministically, MTP decode produced different text at 43% draft acceptance
     instead of 79%. Unforced it appeared in 3 of 10 fp32-matmul perplexity runs — twice as NaN,
     once silently. The fix makes the copy wait on the destination's work marker; it costs 0.7%
     of decode.
- **Prefill matmuls on Pascal now request `CUBLAS_GEMM_ALGO6`.** `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
  picks cuBLAS's long-chain fp16 accumulator from ~256 rows up (NMSE ≈ 2.2e-8·k, up to 2e-4 at
  these shapes) and is also the slower kernel at 512-1024 rows. ALGO6 is 10x more accurate at
  every shape measured and **+63% on pp512, +30% on pp1024**, level at pp2048.
- **The f16 all-reduce compression is now decided per exchange**, from the matmul's cuBLAS compute
  type, instead of from a probe of the first exchange only. No change for a model whose row-split
  projections are all quantized (this one); it matters for mixed-type models and for architectures
  that force `GGML_PREC_F32` on those projections (GLM4, GLM4_MOE, JAIS2).
- The fp16 VKQ fold is *more* roundings, not fewer — all of them at higher precision, so the
  error still falls ~sqrt(N/nbatch_fa). The change is right; the old label was wrong.
- Two free guards that were missing are now restored (Q pre-scaled by `scale*0.25`, and
  `FATTN_KQ_MAX_OFFSET`), and the q4_0 tile dequant no longer produces ±inf for |d| >= 8192.
