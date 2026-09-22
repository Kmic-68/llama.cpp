# Findings

What worked, what didn't, what transfers to other Pascal cards, and the measurement mistakes
that cost the most time. The change list is [CHANGES.md](CHANGES.md); the raw record is
[`../OPTLOG.md`](../OPTLOG.md).

## What binds on a P100

- **The decode matvec is bound by global memory instructions, not bandwidth.** A variant with
  fewer total load/store operations was 5% *slower* because it swapped shared loads for global
  ones. Global loads cost far more than shared ones.
- **The activation, not the weights, is the bottleneck.** Every block re-reads the q8_1
  activation, so its cost scales with block count. Staging it in shared memory was the largest
  single win.
- **Occupancy isn't the limiter.** Forcing fewer registers makes it monotonically slower, even
  with zero spills. So did three separate attempts to raise occupancy in the attention kernel.
- **Achievable bandwidth is ~605 GB/s per card**, not the 732 on the spec sheet. Decode reaches
  ~490 GB/s effective. ECC costs nothing on HBM2, so leave it on.
- **Alignment is a wall.** Every quantized block is 2 mod 4 bytes, so a 4-byte quant word needs
  two loads. Four designs moved that cost around, and none removed it. Only repacking the weights
  would.

## What worked

**Dequantize the KV cache into shared memory, not global.** Upstream converts a quantized KV cache
to f16 in full on every attention call. At 262144 context that's 4.15 ms per call, ~66 ms per
token, to re-convert a cache that changed by a few positions. Dequantizing each tile as it's
staged removed it and freed 512 MiB per card. The result is numerically equal to upstream's for
every finite scale. It isn't bit-identical: some zeros differ in sign, and an infinite scale
gives a signed infinity where upstream gives NaN.

**Fit the tile to the batch.** The attention tile ladder offered widths of 1, 2, 4 and 8 queries
per KV head. A 5-token MTP verify was padded into two 4-token tiles, so 37% of the work was
padding. The width has to be a multiple of the GQA fold (6), and columns per warp must be a
power of two. 36 satisfies both.

**Bound fp16 accumulation error.** The attention output accumulated over the whole KV cache in a
`half2` register, and the error grew as the square root of the context:

| KV length | `half2` accumulator | per-tile fp32 fold |
|---|---|---|
| 4096 | 3.3e-06 | 2.9e-06 |
| 65536 | 2.8e-05 | 3.1e-06 |
| 262144 | not measured | 3.0e-06 |

Folding into fp32 once per tile keeps the fast inner loop and costs 2.4%. The fix often suggested
for P100s, extending the sm_61 `FAST_FP16_AVAILABLE` exemption to sm_60, converts far more to
fp32. It doesn't fit the 36-wide tile in shared memory, and costs 17-90%.

Don't also "fix" `fattn-vec.cuh`. Its `half2 VKQ` declaration looks like the same bug, but it's
compiled only for HIP. CUDA builds already accumulate in `float2`. Trying it cost 16% of decode
for nothing.

**Pick the cuBLAS algorithm on Pascal.** The default picks a long-chain fp16 accumulator from
~256 rows up. `ALGO6` is 10x more accurate at every shape measured, and faster at 512-1024 rows.

**CUDA graphs work on Pascal.** Upstream disables them by architecture. They're +6.7% on the
speculative path but −2% on plain decode, so they ship opt-in.

## What was reverted, and why it's interesting

| attempt | result | lesson |
|---|---|---|
| KQ dot product accumulated in `half2` | −20% time, reverted | Perplexity couldn't tell (2.6097 either way), but per-lane error rose ×1.22, exactly as predicted. Both forms round twice; the second rounding just lands on a larger value. *Where* rounding happens matters, not only how often |
| internal AllReduce on Pascal | −17% | it assumes NVLink, and these are PCIe cards |
| doubling the shared-memory stage | slower | shared memory then limits occupancy (14 → 9 blocks/SM) |
| wide loads in the q4_0 dequant | +7% time | a q4_0 block is 18 bytes, so its quants are never 8-byte aligned |
| vec kernel thread remap | abandoned | passed 3949/3949 op tests and still produced NaN in real inference. Run perplexity first, not last |
| MoE `mmid` threshold tuning | no effect | this model is dense. Check the path runs before tuning it |
| `n_draft` 4 → 6 at depth | +1.9% | acceptance falls from 81% to 70% |

## Two races the op suite couldn't see

Both came from the fork's own work, both are fixed, and both passed all ~14600 `test-backend-ops`
cases for weeks. The suite runs ops one at a time with host syncs between them, which is exactly
the condition under which a cross-stream race can't happen.

1. The GEMM attention softmax wrote probabilities over scores it was still reading. It corrupted
   a few rows per long prompt and could NaN in fp16.
2. A tensor-parallel peer copy could overwrite the all-reduce buffer before the other card's ADD
   had read it. This hit decode and MTP. With the race forced, acceptance fell from 79% to 43%.

Races need repeated unsynchronised runs, deliberate delay injection, or an in-op self-check.

## Serving at long context

**Checkpoints only help if the next query extends them.** Qwen3.8 has 48 recurrent layers, and a
recurrent state can't be rewound. Restoring a saved slot and resending the original prompt still
reprocesses everything, because the saved state includes generated tokens. The recipe: prefill
with `"n_predict": 0`, save, then restore and send the same text plus a suffix. At full depth that
processed 10 tokens instead of 259,229.

## How the measurements lied

These cost more time than any kernel bug.

1. **The wrong corpus.** The perplexity band belongs to one file. Another reads 2.7566 on every
   build, stock included, and it was twice mistaken for a regression. The gate now lives in
   `tools/gate.sh`, so the corpus can't drift from the number.
2. **Warm cards.** One build read 30.8 cold and 24.9 straight after a perplexity run. Discard a
   warmup, and interleave A/B within one session.
3. **`-n 128` at long context** amortises a 2-3 s first-token cost over too few tokens. It
   produced 12.2 t/s when the truth was 21.5. Use 512 or more.
4. **A stale constant.** An old note said bandwidth was ~196 GB/s. The real figure is ~490, and
   the wrong one "proved" a target impossible.
5. **A short prompt's VRAM peak.** "~250 MiB of headroom" was measured at 8% context fill. At a
   full prompt the margin was negative. Take the *minimum* of free memory over a whole long run;
   early samples look safe and mean nothing.
6. **The wrong library.** Copied builds load `build-opt`'s library through RUNPATH. Two different
   kernels once agreed to seven digits, which should have been the tell.
7. **Stale objects.** Editing a CUDA header doesn't always rebuild the template instances that
   include it, so a benchmark measured the previous binary. Touch the includers after a header edit.
8. **Skipped cases.** `test-backend-ops perf` silently skips large-KV cases when VRAM is taken,
   and still prints "passed".
9. **A negative result is only valid for its workload.** CUDA graphs were rejected twice on plain
   decode before they measured +6.7% on the speculative path.
10. **A clean merge isn't a correct merge.** Upstream's new `launch_fattn` parameter shifted one
    of our arguments into a `bool` with no compiler warning. See CHANGES §9.

## What transfers to other Pascal cards

- **Direct KV-cache dequantization** helps any pre-Volta GPU with a quantized KV cache, and more
  as context grows. It's the change most worth proposing upstream.
- **cuBLAS algorithm choice.** Check it on any pre-Volta card before trusting the default.
- **CUDA graphs** on Pascal help workloads that issue many small kernels.
- **The DP4A emulation** applies to all of sm_60. sm_61 cards (GTX 10-series, P40) have real DP4A.

What doesn't transfer directly: the tile configurations assume head size 256 with GQA 6, the
draft-length advice follows from that tile geometry, and the AllReduce result is a PCIe result.

## Where the remaining time is

Plain decode at 229k context is 46.6 ms per token: 22.9 for weights and everything else, 23.7 for
attention. At that shape an f16 KV cache runs at the bandwidth limit (480 GB/s), while q4_0 manages
99 GB/s despite reading a quarter of the bytes. The gap is dequant work and the shared-memory round
trip. Closing it needs a kernel that dequantizes K into registers, which is new work, not tuning.
