## Flags this page was missing (2026-09-12 audit, extended 09-15) — see `AUDIT-2026-09-12.md`

### The accuracy mode, in one flag

    GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32

On this card that is the whole accuracy/speed trade. Prefill matmuls stop rounding their inputs and
their accumulation to fp16, which is **the entire measurable distance between this fork and an
all-fp32 run**: paired per-chunk perplexity over 4096 × 30 tokens moves +0.00343 nats/token → -0.00026,
i.e. from "measurably worse than fp32" (t 7.4) to "indistinguishable from it" (t -1.4); perplexity
2.6191 → 2.6095. It costs **~40% of prefill throughput** (pp512 391 → 217 t/s, pp2048 414 → 255)
and **nothing at all on decode** (tg256 30.78 → 30.77), because decode never takes the cuBLAS path.

Use it when the output matters more than the wait; leave it off for interactive work.


| flag | why it matters |
|---|---|
| `GGML_CUDA_FA_GEMM=0` | The cuBLAS-GEMM attention path is **ON by default** at `Q->ne[1] >= 128 && K->ne[1] >= 4096`; `=0` falls back to the tile kernel. In the **2026-09-06** binaries (now in `build-2026-09-06/`) that path also carried a data race, and `=0` was the way to avoid it. The binaries in `build/` were rebuilt on 2026-09-16 with the fix, so this flag is now a plain precision/speed choice: the tile kernel is ~5x more accurate per op but slower at depth. |
| `GGML_CUDA_FA_GEMM_PREC=32` | fp32 accumulation in the GEMM attention path — exact products, more precise than the tile kernel. -11% pp2048 at 16k depth, -27% at 65k. The fp16 default is not distinguishable from it in perplexity. |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=1` | Worth +6.7% on the MTP path at depth. Builds before the 2026-09-12 fix carried a latent hazard here: the q8_1 activation cache pointer was captured into graph kernel parameters and graph invalidation did not track it. Fixed with a buffer generation counter, and proven at runtime: MTP output with graphs on is byte-identical to graphs off across 61 graph replays. |
| `GGML_CUDA_P2P=1` | Peer-to-peer copies for the tensor-parallel all-reduce. Keep it on: without it the exchanges are staged through the host, which is slower *and* widened the copy race that the 2026-09-06 binaries carry. |

**If you are still running the 2026-09-06 binaries**, note that the copy race above affects
**decode and MTP**, where no flag avoids it. Use `build/` (rebuilt 2026-09-16) or rebuild from the
patch series in `patches/`.

Also: `CLAUDE.md`'s `-DP100_NWARPS/-DP100_ROWS/-DP100_MC_*` build flags are **no longer read**.
The mmvq geometry lives in `mmvq.cu`, gated on `__CUDA_ARCH_LIST__ == 600` — building for any
second architecture silently disables the whole Pascal geometry block.
