# Numerical Verification of the P100 CUDA Patches

Base `f280b2698` → HEAD `2c1f89b12`. Audited 2026-08-31 by three independent
adversarial reviewers, each instructed to *falsify* the bit-exactness claim.

> **Scope note (2026-09-01).** This paper covers the patch set up to
> `2c1f89b12` only. Six further code changes landed afterwards
> (`17455ce35..c4908ecb4`); their numerical status is summarised below and
> argued in full in `OPTLOG.md` attempts 71-80. They were **not** re-audited by
> the adversarial-reviewer process described here.
>
> | change | commit | status |
> |---|---|---|
> | vectorised f32<->f16 convert | `17455ce35` | bit-exact, index remap only |
> | vectorised q6_K dequant | `c3aaef65e` | bit-exact, **machine-proven**: both index mappings replayed over 4096 random superblocks, 1048576 elements, 0 mismatches, 0 unwritten |
> | concurrent peer copies | `27961ce6c` | bit-exact, scheduling only -- no arithmetic touched |
> | cuBLAS ALGO3 | `22c96afb3` | **NOT bit-exact** -- different kernel, different f16 k-accumulation order. PPL 2.6209 -> 2.6214 (0.03 sigma) |
> | f16 all-reduce | `e5c264b71` | bit-exact *here*, and guarded by a one-time runtime probe that verifies every element of the first exchange is f16-representable before any copy is compressed |
> | pipelined delta-net reduction | `e5c264b71` | bit-exact -- `warp_reduce_sum(float2)` applies the same per-component offsets in the same order as the scalar form |
> | delta-net addressing walked | `1b29f55de` | bit-exact, addressing only |
>
> Verification used for these: `test-backend-ops` per-op, a 2-chunk perplexity
> signature check (chunk [1] must read 4.9738), and the full 30-chunk gate.
> Each bit-exact change was confirmed to reproduce the *preceding build*
> digit-for-digit, which is a stronger test than the gate band alone.
>
> The same caution the paper makes below still applies: a matching perplexity
> does **not** prove bit-exactness. Where "bit-exact" is claimed above it rests
> on a structural argument (same expression, same accumulation order, only the
> thread->work assignment or the addressing changes), plus digit-for-digit
> reproduction of the prior build -- not on the gate.

## Conclusion

**The patch set is not byte-identical to stock — confirmed empirically, not
just by inspection.** 2966 of 3847 tensors in a forward pass differ, including
the final logits, with the first divergence at the layer-0 QKV projection.
Two changes on the hot path regroup floating-point summations. The arithmetic *rewrites* are bit-exact and
machine-proven; the *reduction geometry* changes are not. An earlier claim in
`OPTLOG.md` that bit-exactness was "confirmed empirically" by perplexity
returning to 2.7554 was an invalid inference and is retracted — five
significant figures averaged over ~30 chunks cannot detect a low-bit change.

## Method

Three techniques, in decreasing order of strength:

1. **Exhaustive CPU proof.** Reimplement both the stock and patched integer
   routines in standalone C, then enumerate the entire input domain. Used for
   the dp4a emulation, the q6_K/q3_K unpack, `fastdiv`, and the binbcast index
   mapping. This is a proof, not a sample.
2. **Static reduction-order analysis.** Float addition is not associative, so
   any change to *which* terms a thread accumulates, or to how many partial
   trees are combined, changes the result bits. Traced through `calc_nwarps` →
   `blocks_per_iter` → K-loop stride, and through `nbatch_fa` →
   `parallel_blocks` → `gridDim.y` → KV partition.
3. **Whole-graph tensor hashing.** FNV-1a over the full bytes of all 3847
   tensors in a forward pass, via an env-gated hook in the eval callback
   (`tools/tensor-hash-harness.patch`). Captured from both a stock build and
   the optimized build of the same tree, on identical prompts, and diffed
   line-for-line. The optimized build was first confirmed deterministic across
   two runs, so any difference observed is attributable to the patch, not to
   run-to-run nondeterminism.

## Empirical result (whole-graph diff)

Stock `f280b2698` vs HEAD `2c1f89b12`, same prompt, `-sm layer -fa 1 -ctk q4_0
-ctv q4_0`. Graph structure identical in both (3847 tensors, same order), so
the line-for-line comparison is valid.

| path | prompt | tensors differing | first divergence |
|---|---|---|---|
| single-token (`ncols_dst == 1`) | 1 token | **2966 / 3847 (77%)** | `node_13` — layer-0 QKV projection |
| multi-column (`ncols_dst == 4`) | 4 tokens | **2978 / 3847 (77%)** | `node_13` — same |

`result_output` (the final logits) differs on both paths:
`8efc56de4b4baf6e` → `8778091a86bd6765` single-token.

The divergence begins at the **first matrix multiply** and everything
downstream inherits it. The 881 tensors that still match are exactly those
that never pass through `mul_mat_vec_q`: `conv_states`, `state_predelta`,
`conv_state_update`, and the per-layer `cache_s_*` recurrent state. Every
family downstream of a matmul differs — `norm`, `ffn_up`, `ffn_gate`,
`ffn_swiglu`, `ffn_out`, `Kcur`, `l_out`, `attn_residual`, `linear_attn_out`.

This is the predicted signature: the refuted changes all live in
`mul_mat_vec_q` and flash-attention, and the confirmed-exact ones
(`rms_norm`, `binbcast`, the unpack, dp4a) cannot originate a divergence —
`norm` tensors differ only because their *inputs* already did.

## Magnitude: is the model silently worse?

Bit-difference is not degradation. Raw f32 tensors were dumped from both
builds on an identical prompt and compared element-wise.

**Error at the source is one rounding.** Relative RMS error at each layer's
output, stock vs optimized:

| layer | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 63 |
|---|---|---|---|---|---|---|---|---|---|
| rel. RMS | 9.8e-08 | 4.9e-04 | 2.1e-03 | 3.7e-03 | 6.0e-03 | 8.5e-03 | 1.9e-02 | 3.5e-02 | 4.4e-02 |

Layer 0 is **9.8e-08** — fp32 machine epsilon is 1.19e-07. The kernels agree
to within a single float rounding. Growth from there is smooth and geometric
(~1.09×/layer) with no discontinuity, which is the signature of chaotic
amplification of rounding noise in a deep network, not of a defect. A real
indexing or bounds bug shows up as a jump, or as a large layer-0 error.

**At the output:**

| metric | value |
|---|---|
| logits differing (bitwise) | 248319 / 248320 |
| RMS diff / RMS signal | 1.94e-02 |
| KL(stock ‖ optimized) | 1.97e-03 nats |
| argmax | **identical** |
| top-10 | **identical, same order** |

**Control — the same measurement against a change already in use.** Switching
the KV cache between `q4_0` and `f16` (same build) perturbs the model
*more* than the entire patch set does:

| | patch set | q4_0 vs f16 KV cache |
|---|---|---|
| layer-63 rel. RMS | 4.4e-02 | 7.8e-02 |
| output RMS ratio | 1.94e-02 | 3.39e-02 |
| KL | 1.97e-03 | 5.14e-03 |

The kernel changes move the model **~2.6× less than the KV quantization the
benchmark already runs with**. Both preserve argmax and the full top-10.

**Greedy text does eventually diverge.** At `--temp 0` the two builds emit
identical text for roughly 35 lines, then split into different phrasing of the
same content. This is expected: over hundreds of sequential argmax decisions,
some token eventually falls within 2e-03 KL of a tie and the trajectories
separate. It is not evidence of degradation — the same thing happens between
any two batch sizes on stock llama.cpp.

**Quality is unchanged where it is measurable.** Measured on the ORIGINAL
420,098-byte corpus (`ppl-orig.txt`) that the gate was calibrated against:

    llama-perplexity -f ppl-orig.txt -sm tensor -fa on -ngl 99 -c 4096 \
      -ctk q4_0 -ctv q4_0

| build | PPL |
|---|---|
| reference (prior session, `build-faq`) | 2.6209 +/- 0.01994 |
| **this work, incl. both fixes** | **2.6209 +/- 0.01994** |

All 30 per-chunk values identical, `[1]4.9923 ... [30]2.6209`. This meets the
`CLAUDE.md` gate exactly.

On the *current* 422,246-byte `./ppl.txt` every build reads 2.7554 +/- 0.02151
instead -- upstream stock `f280b2698`, the prior session's `b44f8fe6f`, and this
work all identical to every digit. That corpus is a different document, not a
different result; see `CORPUS.md`. Nothing regressed at any point.

### Task-level check: LiveCodeBench v6 (2026-09-12)

Perplexity and KL answer "did the kernels move the distribution". They do not
answer "can the model still do the work". That was measured directly against a
**published** number rather than A/B against stock: `livecodebench/code_generation_lite`
`test6.jsonl` (175 problems, contests 2026-01-04..2026-04-06, 112 AtCoder stdin +
63 LeetCode functional, 40 tests/problem), judged in a `bwrap` sandbox, sampling
at the vendor-specified thinking-mode settings (**temp 0.6, top_p 0.95, top_k 20,
min_p 0** — greedy is explicitly forbidden for this model and produces endless
reasoning; see the failure note below).

| slice | n | token cap | pass@1 | 95% Wilson |
|---|---|---|---|---|
| easy | 27 | 32768 | **27/27 = 100.0%** | 88..100 |
| medium | 33 | 32768 | 26/33 = 78.8% | 62..89 |
| **easy+medium** | **60** | **32768** | **53/60 = 88.3%** | **77.8..94.2** |
| hard | 8 | 65536 | 4/8 = 50.0% | 21.5..78.5 |
| all measured | 68 | — | 57/68 = 83.8% | 73.3..90.7 |

**Published Qwen3.8-27B LCB v6 is 90.3%, which is INSIDE the easy+medium
interval.** 772,886 completion tokens, ~8.8 h wall on the two cards.

Two things this measurement is not:

1. **Not a stock A/B.** It says the optimized build performs at the published
   level; it does not isolate the kernels, and it cannot — the interval is 16
   points wide, and the whole patch set moves the model 2.6x less than the q4_0
   KV cache the benchmark already runs with (above). A task benchmark is a far
   blunter instrument than the KL number; it is here to catch gross breakage,
   not low-bit drift.
2. **Not the official full v6.** `test6` is one of six files and skews hard
   (43 easy / 52 medium / 80 hard against the full 1,055-problem set's easier
   mix). Weighted to the `test6` difficulty mix the measured rates give **70.8%**;
   the full v6 mix is easier than that. Running all 1,055 is 3.5-6 days on this
   hardware.

**The dominant failure mode is the token cap, not wrong answers.** 6 of the 7
easy+medium failures and 2 of the 4 hard failures hit the cap mid-reasoning with
no answer emitted. Excluding them: easy+medium **53/54 = 98.1%**, hard **4/6 =
66.7%**, weighted **83.7%**. This was verified not to be a repetition loop — a
truncated trace was probed directly and held **354 unique sentences of 355**.
Two hard problems that scored 0/40 at a 32k cap scored **40/40** when re-run at
64k (`abc397_e` finished at 44,130 tokens; LeetCode `3762` at 51,473). So the
true pass rate sits in a bracket whose lower end is the table above and whose
upper end is the no-truncation column; a budget under ~48k understates this
model on hard problems.

**Harness:** `tools/lcb/` (`bench.py`, sandbox judge `driver.py`, offline
`rejudge.py`, `analyze.py`, unattended supervisors). Raw completions and
per-problem verdicts: `lcb_em.json`, `lcb_hard.json`.

**Three harness bugs scored correct code as failure — all fixed, all worth knowing:**

| bug | symptom | fix |
|---|---|---|
| greedy decoding | 5 of 5 hard problems emitted the entire 32k budget as reasoning and never answered: **0/5** | vendor sampling settings above. Broken run kept as `lcb_greedy_BROKEN.json` |
| `bwrap --ro-bind / /` | judge could not create its own mount point: `Can't create file at /payload.json: Read-only file system` — **every** problem failed | bind the payload to `/tmp/payload.json`, inside the tmpfs |
| `sys.stdin` as bare `StringIO` | solutions calling `sys.stdin.buffer.read()` died with `AttributeError` | `TextIOWrapper(BytesIO(...))`. `rejudge.py` recovered `arc195_a`, moving the sample 82.1% -> 85.7% |

The judge was then validated against hand-written solutions before any model
output was scored: **40/40 correct accepted, 0/40 wrong accepted**, in both
stdin and functional modes. Two residual `KeyError: 'Solution'` failures
(`3771`, `3794`) are genuine — the model emitted a bare function instead of the
required class.


## Results

| Change | Claim | Verdict | Evidence |
|---|---|---|---|
| dp4a emulation (PRMT+XMAD) | bit-exact | **CONFIRMED** | 22,466,048 cases, 0 mismatches |
| q6_K / q3_K unpack, `__vsubss4` removed | bit-exact | **CONFIRMED** | exhaustive per-byte; saturation provably unreachable |
| `rms_norm` row in registers | bit-identical | **CONFIRMED** | strided ownership preserved; zero-padding appended, never interleaved |
| `binbcast` contiguous fast path | bit-identical | **CONFIRMED** | 384 predicate-satisfying shapes, 0 divergences |
| `fastdiv` in cpy/unary/concat | exact | **REFUTED** (unreachable here) | exact only for n ≤ 2^31; guards admit 2^32−1 |
| q8_1 activation cache | never stale | **CONFIRMED** | 8 attack vectors enumerated, all closed |
| `calc_nwarps` 4→2, `ncols_dst==1` | bit-identical | **REFUTED** | halves K-loop stride → different partial-sum grouping |
| `VDR_Q6_K_Q8_1_MMVQ` 1→4 | bit-identical | **REFUTED** | four lanes folded into one int accumulator; scale reassociated |
| flash-attn KV tile fix | bit-identical | **REFUTED** | changes `parallel_blocks` in 21.5% of D=64 and 36.6% of D=256 configs |
| multi-column `split_rows` | *not* bit-identical | **CONFIRMED** (as stated) | no correctness bug found |
| whole build, end to end | model still performs at the published level | **CONFIRMED** | LiveCodeBench v6 `test6` easy+medium **53/60 = 88.3%** [77.8..94.2]; published 90.3 inside the interval |

### Why the three refutations are "better", not "worse"

- **`VDR = 4`** replaces four separately-rounded float lanes with one exact
  integer accumulator (|acc| ≤ 262144 < 2^24, so int→float is exact). That is
  strictly *fewer* roundings.
- **`nwarps` 4→2** is a regrouping, not a precision change: same terms, two
  partial trees instead of four. Neither ordering is privileged.
- **flash-attn** changes the KV partition. The patched tile size is arguably
  the more correct one, since the kernel steps KV by `nthreads`, not `D`.

None of these lose precision. None is byte-identical.

## Defects found

Ordered by severity. None affects the measured benchmark, all are reachable
outside it.

1. **OOB write (MoE).** On the `MUL_MAT_ID` layout `stride_col_dst` is
   `ne0*ne1`, not `nrows_x`, so the row clamp and write guard don't bound.
   Stock was immune at 1 row/block; this patch forces 2. Odd `nrows_x` writes
   into the next expert's output slot.
2. **`fastdiv` guards off by 2×.** `cpy.cu:253`, `concat.cu:74`,
   `unary.cu:286/389` assert `≤ UINT32_MAX`; the safe bound is `2^31`. A
   tensor with >2^31 elements silently misdirects multi-gigabyte reads and
   writes. Needs ≥4 GB in one tensor, so unreachable on 27B-Q6_K, but this
   code is **not** gated on sm_60. One-line fix each. (The bound bug is
   upstream's idiom; this patch propagates it to three new sites.)
3. **OOB read.** The fused-gate path indexes with the unclamped row index.
4. **Build breakage off sm_60.** `y_slots` is defined only inside the Pascal
   `#ifdef` but used unconditionally.
5. **`cudaMalloc`/`cudaFree` on the hot path**, replacing the pool allocator —
   illegal during graph capture, forces an implicit device sync, invisible to
   VRAM accounting.
6. **Cache not stream-indexed.** Keyed `[device]` where pools are
   `[device][stream]`. Unreachable today (`curr_stream_no` is always 0), breaks
   if concurrent streams are wired up.
7. **Committed probe switches.** `P100_NOY`, `P100_MEMONLY`, `P100_NOUNPACK`
   in `vecdotq.cuh` guard live `#if` blocks and are documented as breaking
   results. Currently 0.
8. **Unexplained.** `ggml-cuda.cu:773` adds `&& tensor->data != nullptr` to
   the quantized-padding zero-init, undocumented. Either it papers over a
   null-`data` crash elsewhere or it is dead.

## Limits of this verification

- The whole-graph diff establishes **that** the outputs differ and **where**
  the divergence starts. It does not measure **how much** they differ — FNV-1a
  is a hash, so a 1-ULP change and a catastrophic one look the same. The
  "no precision loss" claim therefore rests on the component-level proofs and
  on the perplexity gates, not on this diff.
- Both captures used `-sm layer`. The kernels are identical under `-sm tensor`;
  only work distribution differs.
- Perplexity and `test-backend-ops` are tolerance-based and cannot establish
  bit-exactness. They remain valid as *correctness* gates.

## Reproducing

Exhaustive proofs: `tools/proofs-fastdiv-dp4a/`, `tools/proofs-norm-binbcast/`
(CPU-only, `gcc`, no GPU). Hash harness: apply
`tools/tensor-hash-harness.patch`, rebuild `llama-eval-callback`, run with
`LLAMA_TENSOR_HASH=1 ... -sm layer`, and diff the `HASH` lines.

---

## Long-context work (attempts 89-90) — verification status

**Scope note:** the audit above covers up to `2c1f89b12`. This section covers
`738022bda` (cuBLAS-GEMM flash attention) and `0f5b88954` (P/S aliasing).

| claim | evidence | status |
|---|---|---|
| correctness of the new attention path | `test-backend-ops -o FLASH_ATTN_EXT`: **3949/3949**, re-run after every change including the aliasing | **verified** |
| does not perturb short context | d=0, same thermal state: 427.05 +/- 0.44 (off) vs 425.19 +/- 2.19 (on) | **verified** — gated to KV >= 4096 |
| perplexity in band | CLAUDE.md gate, c=4096, ppl-orig.txt: **2.6219 +/- 0.01996** (band 2.6010-2.6408) | **verified** |
| q4_0 KV at depth is sound | A/B at c=16384: tile 2.6035 +/- 0.02713 vs GEMM 2.6047 +/- 0.02719 = **0.04 sigma** | **verified** — this is the only test that exercises the per-chunk dequant |
| +17.9% / +19.3% at depth | d=131072 and d=65536, both arms back to back in one thermal state | **verified**, though run-to-run variance at depth is large (an earlier pair read +9.5%) |
| removes the 512 MiB f16 KV staging | `get_alloc_size` no longer requests it | **request removed; SAVING NOT DEMONSTRATED** |
| 262144 behaviour (throughput, VRAM, decode) | — | **NEVER MEASURED.** Every 262144 figure in these docs is extrapolation |

### On the 512 MiB specifically

Peak VRAM measured **identical** with the path on and off at d=65536
(13493/13237 MiB) and d=131072 (14071/13813 MiB). The unconfirmed explanation is
that ggml sizes one compute buffer by peak *concurrently live* allocations, and at
ub=2048 the FFN intermediates (~142 MB each) exceed the FA staging until it grows
past them near 262144. Do not repeat the 512 MiB claim as fact without measuring
at `-c 262144`.

### Numerical caveat carried by this path

QK^T uses `CUBLAS_COMPUTE_16F` (k=256; the tile kernel likewise keeps KQ in half).
PV uses `CUBLAS_COMPUTE_32F` because it sums thousands of positive terms. **The
f16-PV variant has never been tested** — fp32 was chosen from caution, not
measurement, and it is the single biggest remaining performance lever.

### Update 2026-09-13 — the GEMM attention path, re-verified (OPTLOG attempts 151-152)

Three statements above are now known to be wrong, and one was hiding a real bug.

| earlier statement | status |
|---|---|
| "the tile kernel likewise keeps KQ in half" | **False.** Tile accumulates KQ in fp32 (`fattn-tile.cuh:604`); it keeps the products in half. |
| "PV uses `CUBLAS_COMPUTE_32F`" / "the f16-PV variant has never been tested" | **Stale.** Both GEMMs ship `COMPUTE_16F`. The fp16 PV was measured this session — it is the path's dominant error, and `CUBLAS_GEMM_DEFAULT` made it 10x worse than it needs to be (below). |
| "correctness of the new attention path: 3949/3949, re-run after every change including the aliasing" | **Insufficient.** The aliasing change (0f5b88954) introduced a data race that `test-backend-ops` cannot see: it compares against CPU with a tolerance, once, on random data. |

**The race.** The softmax kernel wrote probabilities over the scores it had just read, in one
buffer. On this toolchain the store can land before the load of the same element, with or
without `__restrict__`. Found through run-to-run variation of fp32 perplexity on identical
input (3.3144 to 3.3272 on one 16k chunk), localized with per-call input/output hashes (a call
with bit-identical inputs produced a different output) and per-stage checksums (only the
softmax's second pass disagreed), and proven with an in-op self-check that runs the kernel twice
on identical inputs: in place 3-6 mismatched launches per ~2240, out of place 0 of ~6700. An
affected launch is not a rounding difference — summed |ΔP| of 3954 to 1.6e7 against
probabilities ≤ 1/8. fp16 showed no mismatch in 15360 checked launches but is the same pattern,
and there the read-back value overflows half and NaNs the output; one 4k perplexity run did go
NaN and did not reproduce. **Fixed: the probabilities get their own buffer** (~1% of the op,
50 MB per GPU at `-ub 2048`). Every build since 0f5b88954, including the release, has the race.

**Determinism is now a checked property.** fp32 GEMM: 3.3165 on all 10 runs across the
out-of-place builds. fp16 GEMM: 3.3185 every run. tile: 3.3134 every run.

**Precision.** NMSE against the CPU fp32 reference at the path's shapes (D=256, 2 KV heads,
GQA 6, q4_0 KV, kv 4096-65536):

| | nb=512 | nb=2048 |
|---|---|---|
| fp16 GEMM, release (PV `GEMM_DEFAULT`) | 1.14-1.24e-5 | 3.98-4.25e-5 |
| **fp16 GEMM, now (PV `ALGO4`)** | 1.18-1.29e-5 | **1.20-1.24e-5** |
| fp32 GEMM (`GGML_CUDA_FA_GEMM_PREC=32`) | 1.49-1.69e-6 | 1.46-1.64e-6 |
| tile kernel | 2.18-2.43e-6 | 2.27-2.37e-6 |

Perplexity effect: see the paired studies in OPTLOG attempt 152.

**262144 context, measured** (was "never measured"): `llama-server -c 262144 -b 262144 -ub 2048
-np 1` with the MTP draft, 19966-token prompt: peak VRAM 15999 MiB on GPU0 (incl. Sunshine) and
15743 MiB on GPU1, prompt 334.8 t/s (in-place build; the out-of-place fix adds 50 MB of GEMM scratch per GPU).

### Update 2026-09-15/16 — a second race, and two rows of the scope table corrected (OPTLOG attempt 153)

| earlier statement | status |
|---|---|
| `27961ce6c` concurrent peer copies: "bit-exact, scheduling only -- no arithmetic touched" | **False.** The scheduling is what broke: a copy could land in the all-reduce's reused buffer before the previous exchange's ADD had read it. Sporadic silent corruption or NaN on every uncompressed exchange — decode, MTP, prompt tails under 512 tokens, fp32 configurations. Fixed; below. |
| `e5c264b71` f16 all-reduce: "guarded by a one-time runtime probe that verifies every element of the first exchange is f16-representable" | **True but incomplete.** Later exchanges were assumed to be like the first. A matmul with F32/BF16 weights or `GGML_PREC_F32` produces non-f16 partials. Now screened per exchange by compute type. For this model nothing changes — every all-reduced projection (`attn_output`, `ffn_down`, `ssm_out`, `nextn.eh_proj`) is Q6_K. |
| "The optimized build was first confirmed deterministic across two runs" (Method, 3) | Still true for what it checked. But a determinism check that synchronizes cannot see a race *between* devices: node-by-node hashing of 2026-09-14 found 0 differences in 33254 prefill and 77324 decode nodes, while unhooked runs of the same build were not deterministic. |

**The race.** The tensor-parallel all-reduce (`ggml-backend-meta.cpp`, `allreduce_fallback`)
copies each partial into the destination's reduction buffer — the same buffer at every layer — and
folds it in with an ADD on the destination's compute stream. The fork's dedicated copy stream
orders a copy against its source only, so a source already at the next layer could overwrite the
buffer before the destination's previous ADD had read it. The f16 path guarded that reuse; the
uncompressed path did not. **Fix:** the copy stream also waits on the destination's work marker,
which `graph_compute` records after every graph, the ADD included.

**Evidence.** Unforced, on identical input (15 chunks, fp32 matmuls, no P2P, no debug hooks): 3 of
10 runs diverged from the per-chunk majority — two into NaN (from chunks 4 and 6), one silently
(from chunk 3, every later chunk off by ~2e-5 nats). With the fix: 0 of 6, and 0 of 4 for the
serialising variant that was not kept. Forced, with a build that delays one device's all-reduce ADD
by dummy sgemms on its own stream — no host synchronize, so the other device keeps running ahead —
the failure becomes deterministic and reaches the production decode path:

| 2 chunks / 64 MTP tokens, P2P on | no delay | delayed, unfixed | delayed, fixed |
|---|---|---|---|
| prefill, fp32 matmuls (uncompressed exchanges) | 1.596804 / 1.370962 | **1.863645 / 1.618642** | 1.596804 / 1.370962 |
| prefill, fp16 matmuls at `-ub 2048` (compressed, already guarded) | 1.603656 / 1.376813 | 1.603656 / 1.376813 | 1.603656 / 1.376813 |
| MTP decode, greedy | text `9f1947a0`, accept 78.8% | **text `72c8d044`, accept 43.4%** | text `9f1947a0`, accept 78.8% |

Cost of the fix: tg256 31.04 → 30.81 (-0.7%), MTP 55.31 → 54.98 (-0.6%), identical output.

**Prefill matmuls.** Upstream's `CUBLAS_GEMM_DEFAULT_TENSOR_OP` picks cuBLAS's long-chain fp16
kernels at this model's prefill shapes (NMSE ≈ 2.2e-8·k, up to 2e-4); the blocked ALGO6 is 10x
more accurate at every shape and faster at 512-1024 rows (pp512 +63%, pp1024 +30%, pp2048 level).
Paired per-chunk perplexity, 4096 × 30 at `-ub 2048`, against an all-fp32 run (2.6102): upstream's
fp16 matmuls +0.00414 nats/token (t 6.7), ALGO6 +0.00343 (t 7.4), **fp32 matmuls with fp16
attention -0.00026 (t -1.4, i.e. indistinguishable from all-fp32)**. fp16 matmuls minus fp32
matmuls, paired: +0.00369 (t 8.2). So the measurable distance from fp32 is the matmuls' fp16
inputs and accumulation, not the fp16 GEMM attention; `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` buys it
back, `GGML_CUDA_FA_GEMM_PREC=32` does not change the model-level number (it is 8x at the op
level).

**The final build** (HEAD `fd560af8d`): perplexity **2.6097 ± 0.0198** on `ppl-orig.txt` at `-c 4096`
(band 2.6209 ± 0.0199), tg256 **30.64 ± 0.19** on cool cards (baseline 17.51),
`test-backend-ops -o FLASH_ATTN_EXT` **3961/3961** and the full suite **14593/14593**, 3/3 backends
on both GPUs. Two further fixes landed with it that no two-device measurement can move, and both
commit messages say so: the same-GPU copy guard (two physical GPUs never take that branch) and
`GGML_OP_FILL` for a device whose slice came out empty (a verbose two-device run logs that path 0
times). The configuration that would exercise the first — more virtual devices than GPUs — is
itself unreliable on this model; see OPTLOG attempt 153 §8c.
