# Numerical Verification of the P100 CUDA Patches

Base `f280b2698` → HEAD `134a4f4a5`. Audited 2026-08-31 by three independent
adversarial reviewers, each instructed to *falsify* the bit-exactness claim.

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

Stock `f280b2698` vs HEAD `134a4f4a5`, same prompt, `-sm layer -fa 1 -ctk q4_0
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
