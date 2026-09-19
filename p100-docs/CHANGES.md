# Change log

Every code change in this fork, grouped by what it touches. 63 code commits on top of upstream
`f280b2698` (2026-08-25); the other 135 of the 198 are documentation and logs.

Each patch file in `patches/` is one commit, numbered in apply order. `diffs/all-code.diff` is
the same thing squashed into one file if you only want to read it.

Numbers are measured on 2x Tesla P100-PCIE-16GB, Qwen3.8-27B Q6_K, q4_0 KV cache, `-sm tensor`.
The full attempt-by-attempt record, including everything that failed, is `logs/OPTLOG.md`.

---

## 1. `mul_mat_vec_q` — the decode matvec

This is ~85% of decode time, so it got the most attention. **Decode 17.51 → 30.64 t/s.**

| commit | change | effect |
|---|---|---|
| `b44f8fe6f` | Pascal launch geometry, warps and rows per block | the initial sm_60 tuning pass |
| `2bb2264dd` | drop `__vsubss4` from Q6_K/Q3_K `vec_dot` | sm_60 has no such instruction; it was being emulated |
| `4d9dbeb34` | stage the weight tile `x` through shared memory | first large win |
| `a277ff94f`, `e97421a3d` | `vdr` 2 then 4 for Q6_K, geometry retuned to 2x2 | |
| `be811a6d1` | stage `x` in 16-byte units | wider loads, fewer instructions |
| `62d9e35f1` | accumulate a whole `vdr` group as integer before scaling | removes a float multiply per group |
| `090f53560` | cache the q8_1-quantized activation across calls | the activation was being requantized per call |
| `c718d1860` | stage the q8_1 activation through shared memory | **the single largest win.** The activation is re-read by every block, so its cost scales with block count — it is the bottleneck, not the weights |
| `dd0e5289b`, `f3cb02935`, `2c1f89b12` | multi-column path: per-warp rows, 16-row blocks, block-wide staging | |
| `c3aaef65e` | vectorized q6_K dequant | |
| `17455ce35` | vectorized contiguous f32↔f16 convert | |

The DP4A emulation (sm_60 has no `__dp4a`) is **8 instructions via PRMT + XMAD.H1 and
bit-exact**. Do not regress it.

## 2. Flash attention — the tile kernel

| commit | change | effect |
|---|---|---|
| `3f49203da` | **dequantize the q4_0 KV tile straight into shared memory** | `launch_fattn` was converting the *entire* KV cache to f16 on *every call* — 4.15 ms at 262144 context, ~66 ms per forward pass, to re-convert a cache that changed by a few positions. **9242 → 6213 µs**, and it frees 512 MiB per GPU of staging. Numerically identical to `to_fp16` for every finite block scale (not bit-identical: zero signs differ, and `d == ±inf` gives a signed infinity here against upstream's NaN). **The most broadly useful change here** — it applies to any pre-Volta GPU with a quantized KV cache |
| `b574f0b98` | stop reserving f16 staging when the tile kernel reads q4_0 directly | the 512 MiB |
| `88211649b` | dequantize with `hfma2` in the tile loader | |
| `5fc820f9d`, `2127ac5bf`, `da3bddaeb` | fold the whole GQA-6 group into one block (vec, tile, and 2-column paths) | |
| `8599fe022` | **exact-fit tile width for the MTP verify shape** | the ladder had `ncols1` of 1/2/4/8 only, so a 5-token verify was padded into two 4-token tiles — **37% of the attention work was padding**. `cols_per_block` must be a multiple of the GQA fold *and* `ncols/nwarps` a power of two; 36 satisfies both at 9 warps. **1.24x on the verify shape** |
| `5ea4b2712` | `nbatch_K = 128` on the narrow GQA-6 tiles | halves the K-chunk loop at head size 256: **1691 → 1518 µs**. Config-specific — 17% *worse* on the 36-wide tile, so it is applied only to the narrow ones |
| `edc7980bf` | **bound fp16 accumulation error** | `VKQ` accumulated over the entire KV cache in a `half2` register — a quarter-million adds in an 11-bit mantissa at 262144 context, and this shape had no eval coverage at all. Now folds into an fp32 running sum once per tile. **8.7x the accuracy at depth for 2.4%** on the decode shape, and the error stops growing with context |
| `43543917b` | give `launch_fattn` the vec kernel's real KV tile size | |
| `961e63c18` | **read each q4_0 byte once in the tile loader** | one read of `qs[m]` carries both values it encodes — `m` in the low nibble, `m+16` in the high — but the old loader handed those to different threads, each loading the same bytes and discarding half of each. q4_0's real DRAM traffic was 2× its useful bytes, which is why it was no faster than an f16 cache moving 2.8× the data. **−13.1% decode at 262144.** Same value into the same slot as before: verified by simulating both index schemes for every thread across 304 configurations |
| `c6f5211f4` | **magic-number dequant** | OR the nibble into the mantissa of `1024.0h` (`0x6400`, ulp exactly 1) to get `1024+q` with no convert instruction, then subtract 1032 to land on `q-8`. Both steps exact in fp16. **−16.5% at 262144**, and **bit-identical** to the `__int2half_rn` form — 0 differences over all 65536 scales × 256 byte values |

**A change that was reverted on accuracy grounds.** `7c77a2b80` accumulated the KQ dot product
in `half2` with `__hfma2` and widened once per group instead of once per product. It is worth
**−20.3%** on this kernel at 262144 and perplexity cannot tell the difference (2.6097 either
way), but it rounds measurably more, so it is not in this build. Per lane, upstream computes
`fl16(a) + fl16(b)` and sums in fp32 while the `hfma2` form computes `fl16(b + fl16(a))`; both
round twice, but the second rounding moves off a product and onto the pair's *sum*, about √2
larger in magnitude and so about twice the error variance. That predicts an RMS ratio of
√(3/2) = 1.2247, and measurement over 2^20 random 256-dimension dot products gives 1.2247–1.2251
across four input distributions. There is no cheap way to have both on Pascal: any scheme that
keeps the accumulation in fp32 costs at least HMUL2 + two widens + two adds per 2 MACs, which is
what upstream already costs. See `FINDINGS.md`.

**On the fp16 accumulation fix:** the widely-circulated P100 fix is to extend the sm_61
`FAST_FP16_AVAILABLE` exemption to sm_60. That is a bigger hammer — it also converts `Q_tmp`,
`KQ` and `KV_tmp` to float, does not build here (the 36-wide tile would need 50176 B of shared
memory against a 48 KiB limit), and costs **+17.5% to +90%** depending on shape. The per-tile
fold gets most of the accuracy for ~2%.

**Do not also "fix" `fattn-vec.cuh`.** Its `half2 VKQ[...]` declaration looks like the identical
bug but is dead code on NVIDIA — it sits under `V_DOT2_F32_F16_AVAILABLE`, which is defined only
for HIP targets. Every CUDA build already accumulates in `float2`. Applying the fold there was
tried and cost tg256 ~31 → 26.0 t/s for no accuracy gain.

## 3. GEMM attention — long-context prefill

A cuBLAS-GEMM attention path for pre-Volta, **on by default** at `Q->ne[1] >= 128 && K->ne[1] >= 4096`.
`GGML_CUDA_FA_GEMM=0` falls back to the tile kernel.

| commit | change | effect |
|---|---|---|
| `738022bda` | the path itself | |
| `0f5b88954` | alias `P` onto `S` | saves a buffer |
| `dbbee401a` | run the PV GEMM in f16 | |
| `0b0e16a03` | issue the attention GEMMs as one call instead of a GQA batch | |
| `d85d55edd` | skip all-zero mask chunks, decided on the GPU | |
| `a7cdad458` | PV requests cuBLAS `ALGO4` instead of `GEMM_DEFAULT` | `GEMM_DEFAULT` picks a long-chain fp16 kernel above n ≈ 6000. **3.4x lower op error** at the production batch |
| `fbf220c10` | decline mask shapes the path cannot honour | |
| `619b6031e` | restore the fp16 overflow guards (Q pre-scaled by `scale*0.25`, `FATTN_KQ_MAX_OFFSET`); q4_0 tile bias in integer | the q4_0 tile dequant no longer produces ±inf for \|d\| >= 8192 |

## 4. Tensor-parallel across the two cards

| commit | change | effect |
|---|---|---|
| `27961ce6c` | run peer copies on a dedicated stream so both directions overlap | **also introduced a data race — see §6** |
| `e5c264b71` | ship tensor-parallel partials as f16; pipeline the delta-net reduction | halves the bytes crossing PCIe |
| `dce17bf1b` | decide f16 compression **per exchange**, from the matmul's cuBLAS compute type | was probed once from the first exchange only. No change for a model whose row-split projections are all quantized (this one); it matters for mixed-type models and for architectures that force `GGML_PREC_F32` there (GLM4, GLM4_MOE, JAIS2) |
| `1b29f55de` | walk `gated_delta_net` addresses instead of recomputing them | |

Keep `GGML_CUDA_P2P=1`. Without it the exchanges stage through the host, which is slower *and*
widens the race in §6.

**The internal AllReduce is slower on Pascal (-17%)** and is correctly gated off upstream for
`cc < Volta`. These are PCIe cards; the pipelined AllReduce assumes NVLink.

## 5. cuBLAS precision

| commit | change | effect |
|---|---|---|
| `fccdafca1` | **Pascal fp16 prefill matmuls request `CUBLAS_GEMM_ALGO6`** | `CUBLAS_GEMM_DEFAULT_TENSOR_OP` picks cuBLAS's long-chain fp16 accumulator from ~256 rows up (NMSE ≈ 2.2e-8·k, up to 2e-4 at these shapes) and is *also* the slower kernel at 512-1024 rows. ALGO6 is **10x more accurate at every shape measured and +63% on pp512, +30% on pp1024**, level at pp2048 |
| `55e496262` | **revert** of `22c96afb3` (ALGO3 for wide f16 GEMMs) | reassociation with no precision argument behind it |

ALGO6 is also more accurate than what upstream does: against an all-fp32 reference, upstream's
`DEFAULT_TENSOR_OP` sits at +0.00414 nats/token and ALGO6 at +0.00343.

## 6. Correctness fixes

Four of these close bugs this fork introduced itself. `test-backend-ops` could not see either
data race — both passed the full suite for weeks.

| commit | bug | how it was proven |
|---|---|---|
| `cb6024e6b` | **The GEMM attention softmax wrote probabilities over the scores it was still reading.** Corrupts a few attention rows per long prompt; in fp16 it can NaN the output | in-op self-check: 3-6 of 2240 launches differ in place, 0 of ~6700 out of place |
| `13b24fe27` | the same path accumulated its running output in place | made out of place |
| `b67848c64` | **An uncompressed tensor-parallel peer copy could overwrite the all-reduce's reduction buffer before the destination's ADD had read it** (introduced by `27961ce6c`). Hits **decode, MTP and short prompts**, not just long prefill | with the race forced deterministically, MTP decode produced different text at 43% draft acceptance instead of 79%. Unforced, it appeared in 3 of 10 fp32-matmul perplexity runs — twice as NaN, once silently. The fix costs **0.7% of decode** |
| `5d479e6f8` | CUDA graphs captured the q8_1 buffer pointer and graph invalidation did not track it | fixed with a buffer generation counter; MTP output with graphs on is byte-identical to graphs off across 61 replays |
| `c1f7f4b00` | 16 bytes of slack for the sm_60 mmvq staging over-read | |
| `24290a858` | mmvq row guards bounded by `stride_col_dst` instead of `nrows_x` | |
| `9e99d468f` | fastdiv domain guards were 2^32, should be 2^31 | |
| `fe0e5c811` | restore bit-identical output in the norm kernels | |
| `194190ef7` | the same-GPU copy between two virtual devices did not wait for the destination's reader | **fixed on inspection, not measured** — see "Known gaps" |
| `fd560af8d` | a device whose graph slice came out empty had its output zeroed by multiplying by `0.0f`, under its own `// FIXME 0.0f * NaN == NaN`. Nothing computed that buffer, so it held whatever the allocator left | **fixed on inspection, not measured** — the branch never executes on this model |

## 7. CUDA graphs and speculative decoding

| commit | change | effect |
|---|---|---|
| `b302163d6` | allow CUDA graphs on pre-Volta, opt-in via `GGML_CUDA_GRAPHS_PRE_VOLTA=1` | upstream disables them by architecture alone. **+6.7% on the MTP path, -2% on single-token decode** — hence opt-in |
| `74de4a1bd` | let the draft context use its own ubatch (`-ubd`) | without it the draft inherits `-ub` and reserves a second 1024 MiB copy of the KQ mask; the full-context config OOMs |

## 8. Tests

`282918f2e`, `cffc2b191`, `2f50214e1`, `232797c00`, `33ff1a5ba`, `a8f1ee60d` add FLASH_ATTN_EXT
coverage at the shapes this fork actually runs: decode-shaped long context, the ubatch tradeoff
at nb=1024/512, the full-cache dequant cost in isolation, the GEMM path's prefill shapes, and an
eval sweep out to the real 262144 operating context.

That last one exists because **the 262144 shape had no coverage at all** before this work, which
is why the fp16 accumulation error in §2 went unnoticed.

---

## What this build measures

| | upstream `f280b2698` | this build |
|---|---|---|
| decode `tg256` | 17.51 t/s | **30.64 ± 0.19** (1.75x) |
| prefill `pp2048`, matched `-ub 512` | 222.6 t/s | **380.4 ± 0.1** (1.71x) |
| prefill `pp2048`, best (`-ub 2048`) | — | **411.5** |
| perplexity, `tools/perplexity-gate-corpus.txt`, `-c 4096` | — | **2.6097 ± 0.0198** (gate band 2.6209 ± 0.0199) |
| `test-backend-ops -o FLASH_ATTN_EXT` | — | 3961/3961, both GPUs |
| `test-backend-ops` full suite | — | 14593/14593, both GPUs |

At the production operating point (`llama-server -c 262144 -b 262144 -ub 2048 -np 1` with the MTP
draft, 19966-token prompt): prompt 350-366 t/s, generation 30.1 t/s, peak VRAM **16137 MiB on
GPU0** (including ~392 MiB of Sunshine) and **15745 MiB on GPU1**, of 16384 each. That prompt is
only 8% of the allocated context — see "Watch VRAM" in QUICKSTART for what a genuinely full one
costs.

### Prefill as context fills

`llama-bench -d <depth> -p 2048 -ub 512`, measured on the shipping build:

| depth | `pp2048` | vs. depth 0 |
|---|---|---|
| 0 | 380.4 ± 0.1 | — |
| 65536 | 222.4 ± 0.6 | 0.58x |
| 131072 | 147.6 ± 12.1 | 0.39x |
| 262144 | 85.4 ± 0.7 | 0.22x |

Decode at depth is measured through the server instead, because `llama-bench -n 128` is too short
to be trusted there (FINDINGS §2): **21.5 t/s plain and 23.2 with the MTP draft at 229k tokens**,
against 30.6 at an empty cache.

Allocating a large `-c` costs decode nothing on its own — `-c 4096` and `-c 262144` both measure
30.7 t/s with a near-empty cache. **Depth is what costs**, not the allocation.

## Known gaps

**Two commits ship on inspection, with no experiment behind them.** `194190ef7` guards a copy
branch that only executes when two virtual devices share one physical GPU — and the only mode
that produces that, `GGML_CUDA_DEVICES`, is itself unreliable (below), so no configuration that
exercises it gives a trustworthy number. `fd560af8d` fixes a branch that never executes on this
model (verbose run: 0 occurrences). Both are correct by reading; neither is proven.

**`GGML_CUDA_DEVICES` above the physical GPU count is not trustworthy.** That flag emulates N
devices round-robin over the real GPUs. At 3 virtual devices, 4 of 8 identical runs produced NaN,
and the runs that completed disagreed in the fourth decimal. Ruled out: both copy guards (it
happens with them on and off), the zero-slice branch, the CUDA memory pool, and `accum_O` write
coverage. It follows the GEMM attention path — with `GGML_CUDA_FA_GEMM=0` the same command is
identical 5 of 5. **Two physical GPUs are bit-stable**, so nothing that ships is affected; this is
a defect in a debug-only mode. `logs/OPTLOG.md` attempt 153 §8c has the table.

**Prefill at full depth is ~10% slower than when that path was tuned.** `pp2048` at `-d 262144`
measured 95.14 t/s during the GEMM-attention work and **85.44 ± 0.68** on the shipping build —
tight error bars on both, so it is not noise. Part may be thermal (77-78 °C at the end of a long
re-measurement sweep; the earlier figure's conditions were not recorded), and it is the same sign
as an already-documented -3.0%. Not bisected. It affects only the deepest prefills; the rest of
the depth curve above matches its original measurements.

**The remaining performance gap is structural.** Plain decode at 229k is 46.6 ms/token, of which
flash attention is 23.7. At the decode shape the f16 KV path runs at 480 GB/s — the bandwidth
limit — while q4_0 runs at 99 GB/s. q4_0 reads 4x fewer bytes and is still slower, so ~1200 of its
1518 µs is dequant overhead. It is not the loads and not the memory path. The fix is to
dequantize K into registers and accumulate all columns per thread, skipping the shared-memory
round trip — a new kernel, not a tuning knob. The parameter space is closed.

## Scope

One model (Qwen3.8-27B, head size 256, GQA ratio 6), one quant family, two PCIe P100s, one driver.
Shape-specific changes are gated so other configurations take the stock path — which means they
are untested elsewhere, not proven safe there. This is a private fork; nothing here has been
through upstream review.
