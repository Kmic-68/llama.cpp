# Change log

Every code change in this fork, grouped by what it touches. 59 code commits on top of upstream
`f280b2698` (2026-08-25); the other 106 of the 165 are documentation and logs.

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
| `d8984de00` | drop `__vsubss4` from Q6_K/Q3_K `vec_dot` | sm_60 has no such instruction; it was being emulated |
| `c7e7faf60` | stage the weight tile `x` through shared memory | first large win |
| `f07b913a7`, `0732c729e` | `vdr` 2 then 4 for Q6_K, geometry retuned to 2x2 | |
| `9fc142d12` | stage `x` in 16-byte units | wider loads, fewer instructions |
| `560442c89` | accumulate a whole `vdr` group as integer before scaling | removes a float multiply per group |
| `246515a32` | cache the q8_1-quantized activation across calls | the activation was being requantized per call |
| `bbdac166e` | stage the q8_1 activation through shared memory | **the single largest win.** The activation is re-read by every block, so its cost scales with block count — it is the bottleneck, not the weights |
| `c9de43286`, `4ee099082`, `134a4f4a5` | multi-column path: per-warp rows, 16-row blocks, block-wide staging | |
| `58c8a73ed` | vectorized q6_K dequant | |
| `5d1fafb01` | vectorized contiguous f32↔f16 convert | |

The DP4A emulation (sm_60 has no `__dp4a`) is **8 instructions via PRMT + XMAD.H1 and
bit-exact**. Do not regress it.

## 2. Flash attention — the tile kernel

| commit | change | effect |
|---|---|---|
| `3ee12f08c` | **dequantize the q4_0 KV tile straight into shared memory** | `launch_fattn` was converting the *entire* KV cache to f16 on *every call* — 4.15 ms at 262144 context, ~66 ms per forward pass, to re-convert a cache that changed by a few positions. **9242 → 6213 µs**, and it frees 512 MiB per GPU of staging. Bit-exact with `to_fp16`. **The most broadly useful change here** — it applies to any pre-Volta GPU with a quantized KV cache |
| `2a6907ab8` | stop reserving f16 staging when the tile kernel reads q4_0 directly | the 512 MiB |
| `f0393db54` | dequantize with `hfma2` in the tile loader | |
| `9c7a8865b`, `362c6172f`, `ce20ad9a7` | fold the whole GQA-6 group into one block (vec, tile, and 2-column paths) | |
| `2c5405fef` | **exact-fit tile width for the MTP verify shape** | the ladder had `ncols1` of 1/2/4/8 only, so a 5-token verify was padded into two 4-token tiles — **37% of the attention work was padding**. `cols_per_block` must be a multiple of the GQA fold *and* `ncols/nwarps` a power of two; 36 satisfies both at 9 warps. **1.24x on the verify shape** |
| `0f759f41e` | `nbatch_K = 128` on the narrow GQA-6 tiles | halves the K-chunk loop at head size 256: **1691 → 1518 µs**. Config-specific — 17% *worse* on the 36-wide tile, so it is applied only to the narrow ones |
| `aa22ccee0` | **bound fp16 accumulation error** | `VKQ` accumulated over the entire KV cache in a `half2` register — a quarter-million adds in an 11-bit mantissa at 262144 context, and this shape had no eval coverage at all. Now folds into an fp32 running sum once per tile. **8.7x the accuracy at depth for 2.4%** on the decode shape, and the error stops growing with context |
| `da56cb516` | give `launch_fattn` the vec kernel's real KV tile size | |

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
| `bdcb3f7bf` | the path itself | |
| `98de4588f` | alias `P` onto `S` | saves a buffer |
| `6af7418c6` | run the PV GEMM in f16 | |
| `2d100c5cc` | issue the attention GEMMs as one call instead of a GQA batch | |
| `0e15f03c4` | skip all-zero mask chunks, decided on the GPU | |
| `508b1af17` | PV requests cuBLAS `ALGO4` instead of `GEMM_DEFAULT` | `GEMM_DEFAULT` picks a long-chain fp16 kernel above n ≈ 6000. **3.4x lower op error** at the production batch |
| `eb56246ec` | decline mask shapes the path cannot honour | |
| `95db6b009` | restore the fp16 overflow guards (Q pre-scaled by `scale*0.25`, `FATTN_KQ_MAX_OFFSET`); q4_0 tile bias in integer | the q4_0 tile dequant no longer produces ±inf for \|d\| >= 8192 |

## 4. Tensor-parallel across the two cards

| commit | change | effect |
|---|---|---|
| `a4d1103c5` | run peer copies on a dedicated stream so both directions overlap | **also introduced a data race — see §6** |
| `e83a7913a` | ship tensor-parallel partials as f16; pipeline the delta-net reduction | halves the bytes crossing PCIe |
| `0b92a60d3` | decide f16 compression **per exchange**, from the matmul's cuBLAS compute type | was probed once from the first exchange only. No change for a model whose row-split projections are all quantized (this one); it matters for mixed-type models and for architectures that force `GGML_PREC_F32` there (GLM4, GLM4_MOE, JAIS2) |
| `ed42ad15d` | walk `gated_delta_net` addresses instead of recomputing them | |

Keep `GGML_CUDA_P2P=1`. Without it the exchanges stage through the host, which is slower *and*
widens the race in §6.

**The internal AllReduce is slower on Pascal (-17%)** and is correctly gated off upstream for
`cc < Volta`. These are PCIe cards; the pipelined AllReduce assumes NVLink.

## 5. cuBLAS precision

| commit | change | effect |
|---|---|---|
| `d7866f67f` | **Pascal fp16 prefill matmuls request `CUBLAS_GEMM_ALGO6`** | `CUBLAS_GEMM_DEFAULT_TENSOR_OP` picks cuBLAS's long-chain fp16 accumulator from ~256 rows up (NMSE ≈ 2.2e-8·k, up to 2e-4 at these shapes) and is *also* the slower kernel at 512-1024 rows. ALGO6 is **10x more accurate at every shape measured and +63% on pp512, +30% on pp1024**, level at pp2048 |
| `6707b9be5` | **revert** of `f8edbf816` (ALGO3 for wide f16 GEMMs) | reassociation with no precision argument behind it |

ALGO6 is also more accurate than what upstream does: against an all-fp32 reference, upstream's
`DEFAULT_TENSOR_OP` sits at +0.00414 nats/token and ALGO6 at +0.00343.

## 6. Correctness fixes

Four of these close bugs this fork introduced itself. `test-backend-ops` could not see either
data race — both passed the full suite for weeks.

| commit | bug | how it was proven |
|---|---|---|
| `8978d3018` | **The GEMM attention softmax wrote probabilities over the scores it was still reading.** Corrupts a few attention rows per long prompt; in fp16 it can NaN the output | in-op self-check: 3-6 of 2240 launches differ in place, 0 of ~6700 out of place |
| `72b108c36` | the same path accumulated its running output in place | made out of place |
| `aef09316d` | **An uncompressed tensor-parallel peer copy could overwrite the all-reduce's reduction buffer before the destination's ADD had read it** (introduced by `a4d1103c5`). Hits **decode, MTP and short prompts**, not just long prefill | with the race forced deterministically, MTP decode produced different text at 43% draft acceptance instead of 79%. Unforced, it appeared in 3 of 10 fp32-matmul perplexity runs — twice as NaN, once silently. The fix costs **0.7% of decode** |
| `c65d3d8a1` | CUDA graphs captured the q8_1 buffer pointer and graph invalidation did not track it | fixed with a buffer generation counter; MTP output with graphs on is byte-identical to graphs off across 61 replays |
| `c5b226581` | 16 bytes of slack for the sm_60 mmvq staging over-read | |
| `2c0d39158` | mmvq row guards bounded by `stride_col_dst` instead of `nrows_x` | |
| `7d004be91` | fastdiv domain guards were 2^32, should be 2^31 | |
| `f30fee9a8` | restore bit-identical output in the norm kernels | |
| `1148b877a` | the same-GPU copy between two virtual devices did not wait for the destination's reader | **fixed on inspection, not measured** — see "Known gaps" |
| `c2dfae805` | a device whose graph slice came out empty had its output zeroed by multiplying by `0.0f`, under its own `// FIXME 0.0f * NaN == NaN`. Nothing computed that buffer, so it held whatever the allocator left | **fixed on inspection, not measured** — the branch never executes on this model |

## 7. CUDA graphs and speculative decoding

| commit | change | effect |
|---|---|---|
| `e339c6243` | allow CUDA graphs on pre-Volta, opt-in via `GGML_CUDA_GRAPHS_PRE_VOLTA=1` | upstream disables them by architecture alone. **+6.7% on the MTP path, -2% on single-token decode** — hence opt-in |
| `0d1ea109c` | let the draft context use its own ubatch (`-ubd`) | without it the draft inherits `-ub` and reserves a second 1024 MiB copy of the KQ mask; the full-context config OOMs |

## 8. Tests

`f7fb6fca3`, `32e8305ff`, `ba6bcac64`, `e5bd6c5a3`, `2dcd8cafd`, `d3446ed2f` add FLASH_ATTN_EXT
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
| prefill `pp2048` | 222.6 t/s | **411.5** (1.85x) |
| perplexity, `tools/perplexity-gate-corpus.txt`, `-c 4096` | — | **2.6097 ± 0.0198** (gate band 2.6209 ± 0.0199) |
| `test-backend-ops -o FLASH_ATTN_EXT` | — | 3961/3961, both GPUs |
| `test-backend-ops` full suite | — | 14593/14593, both GPUs |

At the production operating point (`llama-server -c 262144 -b 262144 -ub 2048 -np 1` with the MTP
draft, 19966-token prompt): prompt 350-366 t/s, generation 30.1 t/s, peak VRAM **16137 MiB on
GPU0** (including ~392 MiB of Sunshine) and **15745 MiB on GPU1**, of 16384 each.

## Known gaps

**Two commits ship on inspection, with no experiment behind them.** `1148b877a` guards a copy
branch that only executes when two virtual devices share one physical GPU — and the only mode
that produces that, `GGML_CUDA_DEVICES`, is itself unreliable (below), so no configuration that
exercises it gives a trustworthy number. `c2dfae805` fixes a branch that never executes on this
model (verbose run: 0 occurrences). Both are correct by reading; neither is proven.

**`GGML_CUDA_DEVICES` above the physical GPU count is not trustworthy.** That flag emulates N
devices round-robin over the real GPUs. At 3 virtual devices, 4 of 8 identical runs produced NaN,
and the runs that completed disagreed in the fourth decimal. Ruled out: both copy guards (it
happens with them on and off), the zero-slice branch, the CUDA memory pool, and `accum_O` write
coverage. It follows the GEMM attention path — with `GGML_CUDA_FA_GEMM=0` the same command is
identical 5 of 5. **Two physical GPUs are bit-stable**, so nothing that ships is affected; this is
a defect in a debug-only mode. `logs/OPTLOG.md` attempt 153 §8c has the table.

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
