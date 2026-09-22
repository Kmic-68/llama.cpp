# Changes

Every code change in the fork, grouped by what it touches. Measurements are on 2x Tesla
P100-PCIE-16GB, Qwen3.8-27B Q6_K, q4_0 KV cache, `-sm tensor`. [`../OPTLOG.md`](../OPTLOG.md) has
the attempt-by-attempt record, including everything that was reverted.

## 1. `mul_mat_vec_q`: the decode matvec

Most of decode time is spent here. Together these took decode from **17.51 to ~31 t/s**.

| commit | change | effect |
|---|---|---|
| `b44f8fe6f` | Pascal launch geometry (warps and rows per block) | the first sm_60 tuning pass |
| `2bb2264dd` | drop `__vsubss4` from the Q6_K/Q3_K dot products | sm_60 emulates it in 9 instructions. A bias trick folds the subtraction into shifts that were needed anyway |
| `4d9dbeb34` | stage the weights `x` through shared memory | every block type is 2 mod 4 bytes, so direct reads need split 16-bit loads. Staging keeps global reads aligned |
| `a277ff94f`, `e97421a3d` | `vdr` 2, then 4, for Q6_K; geometry moved into the source | scales and q8_1 metadata fetched once per four lanes |
| `be811a6d1` | stage in 16-byte units | ~4x fewer global loads and shared stores |
| `62d9e35f1` | accumulate a whole `vdr` group as an integer before scaling | fewer int→float conversions; rounds less often, so slightly more accurate |
| `090f53560` | reuse the q8_1-quantized activation across calls | it was being requantized per call |
| `c718d1860` | stage the q8_1 activation through shared memory | **the largest single win.** Every block re-reads the activation, so it, not the weights, sets the cost |
| `dd0e5289b`, `f3cb02935`, `2c1f89b12` | multi-column (speculative) path: per-warp rows, 16-row blocks, block-wide staging | |
| `c3aaef65e` | vectorised Q6_K dequant | bit-identical, 11 → 6 memory instructions per thread |

The DP4A emulation (sm_60 has no `__dp4a`) is 8 instructions via PRMT + XMAD.H1, and bit-exact.

## 2. Flash attention: the tile kernel

| commit | change | effect |
|---|---|---|
| `3f49203da` | **dequantize the q4_0 KV tile straight into shared memory** | `launch_fattn` was converting the *entire* KV cache to f16 on every call: 4.15 ms per call at 262144 context. **9242 → 6213 µs**, and 512 MiB per GPU of staging freed. Numerically equal to upstream's conversion for every finite scale |
| `b574f0b98` | don't reserve f16 staging when the tile kernel reads q4_0 directly | the 512 MiB |
| `88211649b` | `hfma2` in the tile loader | |
| `5fc820f9d`, `2127ac5bf`, `da3bddaeb` | fold the whole GQA-6 group into one block (vec, tile, two-column) | the KV cache is read once per group instead of once per head |
| `8599fe022` | exact-fit tile for the MTP verify batch | a 5-token verify was padded to two 4-token tiles, so 37% of the work was padding. A 36-column tile fits. **1.24x** on that shape |
| `5ea4b2712` | `nbatch_K = 128` on the narrow GQA-6 tiles | 1691 → 1518 µs. Applied only where it helps: it's 17% worse on the wide tile |
| `edc7980bf` | **fold fp16 accumulation into fp32 once per tile** | the output accumulated over the whole cache in `half2`: a quarter-million adds in an 11-bit mantissa. **8.7x more accurate at depth, for 2.4%**, and the error no longer grows with context |
| `43543917b` | give `launch_fattn` the vec kernel's real KV tile size | short contexts were using a twelfth of the SMs |
| `961e63c18` | read each q4_0 byte once in the tile loader | two threads each loaded the same byte and kept half. **−13.1% decode time at 262144** |
| `c6f5211f4` | magic-number dequant (OR the nibble into `1024.0h`, subtract 1032) | no convert instruction. **−16.5% at 262144**, bit-identical over all 65536 scales × 256 bytes |

Reverted for accuracy: `7c77a2b80` accumulated the KQ dot product in `half2`. It was 20% faster
and perplexity couldn't see it, but it rounds measurably more (RMS error ×1.22, as predicted by
where the second rounding lands). Details in FINDINGS.

## 3. GEMM attention for long-context prefill

A cuBLAS-GEMM attention path for pre-Volta, on by default at batch ≥ 128 and KV ≥ 4096
(`GGML_CUDA_FA_GEMM=0` turns it off). At these shapes the tile kernel reaches 18.6% of fp16 peak,
while cuBLAS reaches 13-15 TFLOPS.

| commit | change |
|---|---|
| `738022bda` | the path itself |
| `0f5b88954`, `dbbee401a`, `0b0e16a03` | alias P onto S; PV GEMM in f16; one GEMM call instead of a GQA batch |
| `d85d55edd` | skip all-masked chunks, decided on the GPU |
| `a7cdad458` | PV requests cuBLAS `ALGO4`. The default picks a long-chain fp16 kernel; ALGO4 has 3.4x lower error |
| `fbf220c10` | decline mask shapes the path can't honour |
| `619b6031e` | restore the fp16 overflow guards; q4_0 tile bias in integer |

## 4. Tensor parallel across the two cards

| commit | change | effect |
|---|---|---|
| `27961ce6c` | peer copies on a dedicated stream, so both directions overlap | also introduced a race, fixed in §6 |
| `e5c264b71` | ship partials as f16 when lossless; pipeline the delta-net reduction | half the bytes over PCIe |
| `dce17bf1b` | decide f16 compression per exchange, from the matmul's actual compute type | matters for mixed-type models and for architectures that force fp32 there |
| `1b29f55de` | walk `gated_delta_net` addresses instead of recomputing them | |

Upstream's internal AllReduce stays off on Pascal. It measured 17% slower here, because these are
PCIe cards and it assumes NVLink.

## 5. cuBLAS precision

| commit | change | effect |
|---|---|---|
| `fccdafca1` | Pascal fp16 prefill matmuls request `CUBLAS_GEMM_ALGO6` | the default picks a long-chain fp16 accumulator from ~256 rows up. ALGO6 is **10x more accurate at every shape measured**, and 63% faster at pp512 |
| `55e496262` | revert ALGO3 for wide GEMMs | it reassociated with no precision argument behind it |

## 6. Correctness fixes

Four of these fix bugs the fork itself introduced. Neither data race was visible to
`test-backend-ops`.

| commit | bug | evidence |
|---|---|---|
| `cb6024e6b` | the GEMM attention softmax wrote probabilities over scores it was still reading | in-op self-check: 3-6 of 2240 launches differed in place, 0 of ~6700 out of place |
| `13b24fe27` | the same path accumulated its output in place | made out of place |
| `b67848c64` | a tensor-parallel peer copy could overwrite the all-reduce buffer before the other card had read it. It hit decode and MTP, not only prefill | with the race forced, MTP produced different text at 43% acceptance instead of 79%. The fix costs 0.7% of decode |
| `5d479e6f8` | CUDA graphs captured the q8_1 buffer pointer without tracking it | graphs-on output is byte-identical to graphs-off across 61 replays |
| `c1f7f4b00` | 16 bytes of slack for the mmvq staging over-read | |
| `24290a858` | mmvq row guards bounded by the wrong stride | |
| `9e99d468f` | fastdiv domain guards at 2^32 instead of 2^31 | |
| `fe0e5c811` | restore bit-identical output in the norm kernels | |
| `194190ef7` | the same-GPU copy between two virtual devices didn't wait for its reader | fixed by reading the code; no test exercises it |
| `fd560af8d` | an empty tensor-parallel slice was zeroed by multiplying by 0.0f, which keeps NaNs | fixed by reading the code; the branch never runs on this model |

## 7. CUDA graphs and speculative decoding

| commit | change |
|---|---|
| `b302163d6` | allow CUDA graphs on Pascal, opt-in with `GGML_CUDA_GRAPHS_PRE_VOLTA=1`. +6.7% on the MTP path, −2% on plain decode |
| `74de4a1bd` | `-ubd`: a separate ubatch for the draft context |

## 8. Tests

`282918f2e`, `cffc2b191`, `2f50214e1`, `232797c00`, `33ff1a5ba` and `a8f1ee60d` add FLASH_ATTN_EXT
cases at the shapes this fork runs, out to the real 262144 context. That shape had no coverage
before, which is how the fp16 accumulation error in §2 went unnoticed.

## 9. Upstream merges

### 2026-09-22: upstream `f46bc30cb`

502 upstream commits since the fork point `f280b2698`, including new model architectures,
sparse flash attention, and reworked speculative decoding. Four files conflicted. Resolving them
safely took more than the conflicts:

- **`fattn-tile.cuh` (merged cleanly, but wrong).** Upstream added a `use_sparse` parameter to
  `launch_fattn` before `warp_size`. Our GQA-6 tile launch passed `warp_size` positionally, so
  the `int` bound to `use_sparse` and switched sparse attention on in Qwen's q4_0 attention path.
  It compiled without a warning. On this model it would have aborted on an assert at the first
  attention call; on a model that sets a sparse KV hint, it would have run the wrong kernel.
  Fixed, and every `launch_fattn` call was checked for arity.
- **`convert.cu`.** Upstream landed the same vectorised f32↔f16 cast as ours (`17455ce35`). Theirs
  is kept and ours dropped. Same per-element cast, so the output is bit-identical.
- **`ggml-cuda.cu`.** Upstream's cuBLAS compute-type rule now depends on the batch width (`src1`),
  for BF16 on older GPUs. The fork had moved that rule into a helper that the tensor-parallel
  compression check also calls. The helper now takes `src1`, so both callers agree.
- **`fattn-vec.cuh`, `test-backend-ops.cpp`.** Both sides' changes kept.
- `GGML_CUDA_FA_ALL_QUANTS` is deprecated upstream. Builds use `GGML_CUDA_FA_QUANTS=all`.

Checked unchanged: the DP4A emulation (byte-identical), the Pascal mmvq geometry and staging,
the Q6_K/Q3_K dot products, the direct-q4_0 tile path, GEMM attention dispatch, and `-ubd`.

**The output changed slightly, on purpose.** Upstream `5fdfa6282` corrects the gated delta-net
q/k normalization from `x / max(‖x‖, eps)` to the reference `x · rsqrt(Σx² + eps)`, the form
flash-linear-attention, transformers, vLLM and SGLang all use. That moves the logits by a mean
KL divergence of 0.0015 and leaves perplexity unchanged. With that one change reverted as a
diagnostic, the merged build reproduces the pre-merge logits exactly (KLD < 1e-5, 100% identical
top tokens), so everything the fork computes survived the merge unchanged. Decode speed is also
the same either way.

| gate | before (shipped build) | after |
|---|---|---|
| `tg256`, cold cards, back to back | 30.76 ± 0.18 | 30.57 ± 0.20 |
| perplexity | 2.6097 ± 0.0198 | 2.6101 ± 0.0198 |
| KL divergence vs before, 8 chunks | — | 0.0015 mean, 98.6% same top token (all from `5fdfa6282`) |
| `test-backend-ops` | 14593/14593 | 16180/16180, both GPUs |

## Known gaps

- **`GGML_CUDA_DEVICES` above the physical GPU count isn't reproducible.** At 3 virtual devices,
  4 of 8 identical runs gave NaN. It follows the GEMM attention path (`GGML_CUDA_FA_GEMM=0` is
  stable 5 of 5). Two physical GPUs are bit-stable, so nothing that ships is affected. OPTLOG
  attempt 153 §8c has the data.
- **Deepest prefill is ~10% below its best measurement.** `pp2048` at `-d 262144` measured 95.1
  t/s during tuning and 85.4 later. Possibly thermal; not bisected.
- **The remaining decode gap at depth is structural.** At 229k, attention is 23.7 of 46.6 ms per
  token. The f16 KV path runs at the bandwidth limit, while q4_0 is dominated by dequant work.
  Closing that needs a new register-resident kernel, not tuning.

## Scope

One model (Qwen3.8-27B: head size 256, GQA ratio 6), one quant family, two PCIe P100s.
Shape-specific changes are gated so other configurations take the stock path.
