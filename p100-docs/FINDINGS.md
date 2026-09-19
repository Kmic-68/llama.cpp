# Findings — what worked, what failed, and how the measurements lied

The distillation. The full record with numbers is `../logs/OPTLOG.md` (153 attempts); the change
list is `../CHANGES.md`.

---

## What worked

### 1. The q8_1 activation, not the weights, bounds `mul_mat_vec_q`
The obvious assumption is that a quantized matvec is bound by reading weights. It is not: the
activation is re-read by every block, so its cost scales with the number of blocks, and staging
it cooperatively was the single largest early win. Weights are streamed once and are
comparatively cheap.

### 2. sm_60 has no DP4A, and the emulation is the hot path
Pascal lacks `__dp4a`. The emulation was reduced to **8 instructions via PRMT + XMAD.H1**, and it
is bit-exact. Everything in the matvec inner loop is measured against that budget.

### 3. Dequantize the KV cache into shared memory, not into global
`launch_fattn` converts a quantized KV cache to f16 **in full, on every call** — a fixed
**4.15 ms** at 262144 context, ~66 ms per forward pass across 16 attention layers, to re-convert
a cache that changed by a few positions. Dequantizing each tile directly into the shared memory
the kernel already stages removed it: **nb=6 9242 → 6213 µs**, and it frees 512 MiB per GPU of
staging. The dequant is bit-exact with `to_fp16`, so perplexity is unchanged to the last digit.

**This is the most broadly useful change here** — it applies to any pre-Volta GPU with a
quantized KV cache, and the cost it removes scales with context length.

### 4. Tile widths must exactly fit the batch
The flash-attn tile ladder offered `ncols1` of 1/2/4/8 only. A 5-token speculative verify was
padded into two 4-token tiles — **37% of the attention work was padding**. `cols_per_block` must
be a multiple of the GQA fold **and** `cpw = ncols/nwarps` must be a power of two; **36 satisfies
both** at 9 warps. Worth 1.24x on the verify shape.

**The rule that follows:** choose `n_draft` so that `nb = n_draft+1` exactly fills a tile. One
token past a boundary buys a whole extra tile width and wastes it — at 262144 context that is the
difference between `n_draft=3` (50.5 ms of attention per pass) and `n_draft=4` (78.4 ms).

**But the rule is not the whole answer in practice.** It only dominates when attention dominates
the pass, which is true at full depth and progressively less true as context shortens. At mixed
depths with a real acceptance rate, `n_draft=4` measured better. See QUICKSTART.

### 5. `nbatch_K = 128` on the narrow tiles
Halves the K-chunk loop from 4 to 2 at head size 256: **1691 → 1518 µs** at the decode shape,
reproducible to 0.1%. Config-specific — **17% worse** on the 36-wide tile, so it is applied only
to the narrow ones.

### 6. CUDA graphs work on Pascal
Upstream disables them for `cc < Volta` on architecture alone. Enabling them is **+6.7% on the
speculative path** and **-2% on single-token decode**, so it ships opt-in via
`GGML_CUDA_GRAPHS_PRE_VOLTA=1`.

An early version captured the q8_1 activation buffer pointer into graph kernel parameters without
tracking it for invalidation. Fixed with a buffer generation counter, and proven at runtime: MTP
output with graphs on is byte-identical to graphs off across 61 graph replays.

### 7. fp16 accumulation in flash-attn was silently lossy, and the error grew with context

`VKQ` — the attention output — accumulated over the **entire** KV cache in a `half2` register: a
quarter-million adds in an 11-bit mantissa at 262144 context. This shape had **no eval coverage at
all**, so it had never been compared against the CPU reference. Measured NMSE:

| kv | half2 accumulator | **per-tile fp32 fold (the fix)** |
|---|---|---|
| 512 | 3.185e-06 | 2.85e-06 |
| 4096 | 3.310e-06 | 2.89e-06 |
| 16384 | 8.357e-06 | 2.59e-06 |
| **65536** | **2.773e-05** | **3.10e-06** |
| 131072 | not measured | 3.55e-06 |
| **262144** (the real operating context) | not measured | **3.00e-06** |

Tolerance is 5.000e-04, so the fixed path keeps ~150x of headroom at the operating point. The
sweep runs to 262144 deliberately: the point of the fix is that the error stops tracking context
length, and that is worth measuring at the depth actually used rather than extrapolating from
65536. Both GPUs agree to within scatter.

The error grew as sqrt(context). The fix accumulates in half2 *within* a KV tile and folds into an
fp32 running sum once per tile, so the inner loop keeps its single `HMUL2` and pays one conversion
per 64 products. **8.7x the accuracy at depth for 2.4% on the decode shape**, and the error is now
flat with context rather than growing.

Note the fold is *more* roundings, not fewer — all of them at higher precision, so the error still
falls ~sqrt(N/nbatch_fa). The change is right; an earlier description of it here was not.

**Note for other P100 owners:** the widely-circulated fix is to extend the sm_61
`FAST_FP16_AVAILABLE` exemption to sm_60. That is a bigger hammer — it also turns `Q_tmp`, `KQ`
and `KV_tmp` to float. On tuned configs here it does not even build (the 36-wide tile needs
50176 B of shared memory against a 48 KiB limit), and measured as an accuracy-equivalent change it
costs **+17.5% to +90%** depending on shape. The per-tile fold gets most of the accuracy for ~2%,
and keeps the fp16 path that makes the P100 worth using.

**Do not also "fix" `fattn-vec.cuh`.** It declares `half2 VKQ[ncols][(D/2)/nthreads_V]` and looks
like the identical bug. It is dead code on NVIDIA: that declaration sits under
`V_DOT2_F32_F16_AVAILABLE`, which is defined only for `GGML_USE_HIP` on RDNA/CDNA/gfx906 targets.
Every CUDA build already takes the `#else` branch and accumulates in `float2`. Applying the fold
there was tried and cost tg256 ~31 → 26.0 t/s for no accuracy gain, because taking a reference to
a local array forces it out of registers in a kernel with no register headroom. This macro is
**not** `FAST_FP16_AVAILABLE`, which *is* defined on sm_60 and is the one the community post is
about.

### 8. cuBLAS algorithm choice is worth more than precision mode on Pascal

`CUBLAS_GEMM_DEFAULT_TENSOR_OP` picks cuBLAS's long-chain fp16 accumulator from ~256 rows up
(NMSE ≈ 2.2e-8·k, up to 2e-4 at these shapes), and it is *also* the slower kernel at 512-1024
rows. Requesting `CUBLAS_GEMM_ALGO6` is **10x more accurate at every shape measured and +63% on
pp512, +30% on pp1024**, level at pp2048. Both faster and more accurate, which is rare enough to
be worth stating twice.

The same effect drove `ALGO4` for the GEMM attention PV product: `GEMM_DEFAULT` picks a long-chain
fp16 kernel above n ≈ 6000, and `ALGO4` is 3.4x lower op error at the production batch.

An earlier attempt at this (`ALGO3` for wide f16 GEMMs) was **reverted** — it was reassociation
with no precision argument behind it.

---

### 10. A peak-VRAM number must be `min` over the whole watchdog log, never the samples on screen

Two runs, identical config (`-c 262144 -b 262144 -ub 512`, `--spec-type none`, same
259229-token prompt). I watched the first one's guard log go 2748 -> 2722 -> 2666 MiB during
early prefill, concluded it "never came close to the floor", and wrote that into the shipped
config comment as the justification for `-ub 512`. The actual minimum over the full log was
**273 MiB**. The footprint climbs steeply only at the very end of prefill, so early samples look
reassuring and mean nothing. The second run, differing only by 136 MiB of other GPU use, bottomed
out at 193 MiB and was killed.

Always: `grep -oE "gpu0_free=[0-9]+" guard.log | cut -d= -f2 | sort -n | head -1`.

And do not use the load-time reservation as a proxy for the prefill peak: at `-c 65536` the
ubatch scaling suggested 5.76 bytes per `token*ubatch`, while the measured full-depth prefill
peak implies ~42 bytes -- a factor of 7 the wrong way.

### 11. On a hybrid model you cannot rewind the KV cache, so a checkpoint only helps if the query EXTENDS it

`/slots/{id}?action=save` and `restore` work and are fast -- 4.94 GB of 259k-token state in
2.3 s, against a 34-minute prefill. But restoring and then sending the *original prompt* still
reprocessed all 259229 tokens, with `f_sim_best = 1.000` in the log: a perfect prefix match.

The reason is that the saved state held prompt + 63 generated tokens, so matching the shorter
prompt required *truncating* the cache. This model has 48 recurrent gated-delta-net layers among
its 65 blocks, so `common_context_can_seq_rm` returns FULL (whole sequences only) -- a recurrent
state cannot be rewound, and the server's only option is to discard everything and start over.

The working recipe:
  1. prefill with `"n_predict": 0`, so the saved tokens are exactly the prompt with no tail
  2. `action=save`
  3. afterwards: `action=restore`, then query with `<the exact same text> + <any suffix>`

Verified at full depth: the extending query processed **10 tokens instead of 259229**.
`scratchpad/slots/full262_exact.bin` (`n_saved = 259229`) is such a checkpoint.

## What failed (do not repeat without new information)

| attempt | result | why it is interesting |
|---|---|---|
| **Internal AllReduce on Pascal** | **-17%** | It is gated off for `cc < Volta` because its poll uses `__nanosleep`. That poll is a `volatile` load, so a `clock64()` spin makes it run — and it is simply slower. These are **PCIe** cards; the pipelined AllReduce assumes NVLink. Upstream's gate is right for a reason it never states |
| **Occupancy, three ways** | neutral or worse | 384 threads, occupancy 2→4, doubled warps. The tile kernel is **not latency-bound**, which is the single most useful negative here |
| **`cpw` (Q-column reuse)** | identical to 0.005% | 96 vs 192 threads. Doubling shared-read reuse per thread changes nothing, so it is not shared-read-bandwidth bound either |
| **Wide loads in the dequant** | **+7% (worse)** | Replacing 8 scalar byte loads with one unaligned 8-byte load. A q4_0 block is 18 bytes so `qs` is never 8-byte aligned; the scalar loads coalesce and the wide load does not |
| **`nbatch_K = 256`** | +44% (worse) | Halving the K loop once pays; twice does not |
| **Fused-MoE `mmid` threshold** | **inert** | Spent real time on it before checking: this model has **no `ffn_*_exps` tensors**. It is dense, there are no `MUL_MAT_ID` nodes, and the code path never executes |
| **`n_draft` 4 → 6 at depth** | +1.9% | Acceptance falls 81% → 70% and cancels the amortization |
| **Thread-mapping inversion (vec kernel)** | abandoned | Passed 3949/3949 op tests and still NaN'd real inference — an out-of-bounds write the op suite cannot see. **The op suite is not sufficient validation for this kernel; run perplexity first, not last** |

---

## Two data races the op suite could not see

Both were introduced by this fork's own work, both are fixed, and both passed the full
`test-backend-ops` suite for weeks while live.

1. **The GEMM attention softmax wrote probabilities over the scores it was still reading.** It
   corrupts a few attention rows per long prompt and in fp16 can NaN the output. Proven with an
   in-op self-check: 3-6 of 2240 launches differ in place, 0 of ~6700 out of place.

2. **An uncompressed tensor-parallel peer copy could overwrite the all-reduce's reduction buffer
   before the destination's ADD had read it**, introduced along with the dedicated copy stream.
   This one hits **decode, MTP and short prompts**, not only long prefill: with the race forced
   deterministically, MTP decode produced different text at 43% draft acceptance instead of 79%.
   Unforced, it appeared in 3 of 10 fp32-matmul perplexity runs — twice as NaN, once silently. The
   fix makes the copy wait on the destination's work marker; it costs 0.7% of decode.

The lesson is item 9 below.

---

## Where the remaining gap is

Plain decode at 229k is **46.6 ms/token = 21.5 t/s**:

| | ms/token |
|---|---|
| weights + everything else | 22.9 (**490 GB/s effective, 67% of peak**) |
| **flash attention** | **23.7** |

The decisive diagnostic: at the decode shape the **f16** KV path runs at **480 GB/s — the
bandwidth limit** — while **q4_0 runs at 99 GB/s**. q4_0 reads 4x fewer bytes and is still slower,
so ~1200 of its 1518 µs is dequant overhead. It is **not** the loads (disproved) and **not** the
memory path (f16 proves it is fine). It is dequant arithmetic plus the shared-memory round trip.

**The remaining fix is structural**, and the parameter space is closed: dequantize K into
registers and accumulate all columns per thread, skipping the shared round trip entirely. That is
a new kernel for the low-column shape, not a tuning knob.

---

## How the measurements lied — read this before trusting a number

These cost more time than any kernel bug.

1. **A negative result is only valid for the workload it was measured on.** CUDA graphs were
   rejected **twice** on single-token `llama-bench`, which issues few kernels. On the speculative
   path they are worth +6.7%.
2. **`-n 128` is far too short at long context.** It amortizes a ~2-3 s fixed startup over ~30
   passes. This produced a plain-decode figure of **12.2 t/s** when the truth was ~21.5 — a
   **1.8x error** that misdirected an entire session. Use >= 512 tokens.
3. **A stale constant poisoned every budget.** The project notes recorded "achieved bandwidth
   ~196 GB/s"; the real figure is **~490 GB/s**. That single number produced a published
   conclusion that a throughput target was *physically impossible*, which had to be retracted.
4. **Cold-start and thermal skew are severe.** The same configuration measured **26.4 t/s
   first-in-batch and 30.1 warm**; cross-session swings reach 13%. Always discard a warmup run and
   interleave A/B *within one session*.
5. **`test-backend-ops perf` silently skips large-kv cases when VRAM is occupied.** A running
   server made every 262144 case vanish while still printing "2/2 backends passed".
6. **A single profile at long context is prefill-dominated** (590 GPU-seconds of prefill against
   13.5 s of decode). Difference two runs' **call counts**, which are exact integers.
7. **Check which corpus a perplexity target belongs to.** The gate here is 2.6209 ± 0.0199, but
   only against one specific file; another gives 2.7566 on *any* build including stock. This is
   not hypothetical — it has caused two separate reverts of correct work. Put the gate in a script
   rather than in prose, so the corpus cannot drift away from the number. Here that is
   `../tools/gate.sh`.
8. **Editing a CUDA header does not necessarily rebuild the template instances that include it.**
   A `cmake --build` after editing `fattn-vec.cuh` rebuilt plenty of other objects and **zero** vec
   instances, so the next benchmark measured the previous binary. Follow any kernel-header edit
   with `grep -rl "<header>" ggml/src/ggml-cuda/ | xargs touch`.
9. **A peak-VRAM number is only valid at the context fill it was measured at.** "Peaks at 16137
   MiB, ~250 MiB of headroom" was measured with a 19966-token prompt against a 262144 context —
   8% full — and was quoted for weeks as the budget for the configuration. It is not a budget:
   the attention mask is sized `n_kv x n_tokens` where `n_kv` is the *used* cache, so it grows as
   the prompt fills, and at a genuinely full prompt the margin is *negative* and the server dies
   mid-prefill. (First diagnosed here as the GEMM-attention workspace, which was wrong — that path
   chunks at a fixed 2048 and is constant in depth. Reading the allocation sites took two minutes
   and would have saved a wrong fix.) Sample free VRAM over a long prefill
   rather than reading a peak off a short one; the shape of the curve is the finding, and here it
   was flat for 20 minutes before it moved at all. See "Watch VRAM" in QUICKSTART.
10. **`test-backend-ops` cannot see a data race.** Both races above passed the full suite — 14593
   cases — for weeks. The suite runs ops one at a time with host synchronization between them,
   which is exactly the condition under which a cross-stream race does not occur. Races need
   unhooked repeated runs, deliberate delay injection, or an in-op self-check. A green suite is
   necessary and nowhere near sufficient.
