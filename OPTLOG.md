# P100 (sm_60) CUDA kernel optimization log

Bench (unless noted):
```
GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
  -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 3
```
Correctness gate:
```
./build-opt/bin/llama-perplexity -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf -f ./ppl.txt \
  -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0
```
Required PPL: 2.6209 +/- 0.0199.

Build flags for every entry below:
`-DP100_NWARPS=8 -DP100_ROWS=4 -DP100_MC_NWARPS=4 -DP100_MC_ROWS=2`

---

## Profiling baseline (SASS, `mul_mat_vec_q<Q6_K, ncols_dst=1>`)

Inner loop body = **412 instructions**. Mix:

| op | n | note |
|---|---|---|
| XMAD | 142 | only **32** are the dp4a emulation; **110 are address arithmetic** |
| IADD | 38 | |
| PRMT | 36 | 32 = dp4a sign-extend, 4 = `__vsubss4` |
| LDG | 32 | |
| LOP3 | 32 | |
| MOV32I | 25 | |
| other | 107 | |

Pascal has no IMAD: every 32x32->64 multiply costs 4-6 XMADs. `sizeof(block_q6_K)=210`
and `sizeof(block_q8_1)=36` are not powers of two, and `vec_dot` recomputes
`(const block_q6_K *) vbq + kbx` from scratch on every call. That address math, not the
dot product, is the single largest consumer of issue slots.

Measured cost of `__vsubss4` on sm_60 (standalone cubin): **9 instructions** vs 1 for a
plain `IADD32I`. Pascal and all later NVIDIA GPUs emulate the SIMD-video intrinsics.

---

## Attempts

| # | change | t/s | PPL | verdict |
|---|---|---|---|---|
| 0 | baseline (HEAD b44f8fe6f) | **17.45 +/- 0.01** | 2.6209 (given) | reference |
| 1 | **eliminate `__vsubss4` from Q6_K + Q3_K mmvq vec_dot** | **17.71 +/- 0.02** | 2.7554 (bit-exact vs baseline) | **KEPT** |
| 2 | + hoist row base pointers in `mul_mat_vec_q` (kill address multiplies) | 16.69 +/- 0.00 | not run | REVERTED |
| 3 | + attempt 2 with `__launch_bounds__` min 4 blocks/SM (64 reg) | 15.59 +/- 0.01 | not run | REVERTED |
| 4 | + attempt 2 with `__launch_bounds__` min 3 blocks/SM (80 reg, 0 spill) | 16.68 +/- 0.01 | not run | REVERTED |
| - | re-verify after reverting 2-4 (attempt 1 only) | **17.72 +/- 0.01** | | confirms 1 |

Run-to-run noise on this bench is ~0.01-0.02 t/s (17.71 vs 17.72 on identical builds).

---

## IMPORTANT: the perplexity gate value in CLAUDE.md does not match this tree

CLAUDE.md requires `2.6209 +/- 0.0199`. Measured on this repo with `./ppl.txt`, 30 chunks,
`-c 4096 -ctk q4_0 -ctv q4_0`:

- **stock kernel (HEAD b44f8fe6f, no changes): PPL = 2.7554 +/- 0.02151**
- attempt 1 applied:                          PPL = 2.7554 +/- 0.02151

All 30 per-chunk values are identical between the two runs
(`[1]4.8657,[2]3.9010,[3]3.7738,...,[30]2.7554`). So 2.7554 is simply what this
repo + this `ppl.txt` produce; the 2.6209 figure in CLAUDE.md came from a different
tree or corpus. **Working gate for this session: 2.7554 +/- 0.0215.**

Attempt 1 was additionally verified correct by two stronger checks than perplexity
(perplexity runs at batch 2048, which goes through MMQ and barely exercises mmvq at all):

- standalone sm_60 cubin comparing old vs new `vec_dot` over 4,194,304 random
  `(vl, vh, u, scales, d, d8)` tuples: **0 mismatches, bit-identical floats**, Q6_K and Q3_K
- `test-backend-ops -o MUL_MAT -b CUDA0`: **1193/1193 passed**, including the `n=1`
  mmvq path for q6_K and q3_K

---

## What the failed attempts establish about this kernel

This is the useful result of attempts 2-4. `mul_mat_vec_q` on sm_60 is **not** issue-bound,
**not** occupancy-bound, and does not respond to instruction-count reduction:

| lever pulled | effect on kernel | effect on t/s |
|---|---|---|
| -11% inner-loop instructions (attempt 1) | 412 -> 368 instr | +1.5% |
| -36% inner-loop instructions, XMAD 142 -> 72 (attempt 2) | 264 instr, 96 reg | **-5.8%** |
| 2x occupancy: 2 -> 4 blocks/SM, 64 reg (attempt 3) | 32 warps/SM | **-12%** |
| same occupancy as baseline, 3 blocks/SM, 0 spill (attempt 4) | 80 reg | -5.9% |

Attempts 2 and 4 differ only in register count and land within noise of each other, so the
loss is caused by the pointer-strength-reduction *itself*, not by register pressure. Replacing
the induction-variable index `kbx_offset + i*stride_row_x + kbx` with an incrementing pointer
creates a loop-carried dependency: ptxas can no longer compute future iterations' addresses
ahead of time, so it keeps fewer loads in flight. Attempt 3 makes the same point from the other
side -- capping registers at 64 frees occupancy but destroys per-thread memory-level
parallelism, and costs more than the occupancy gains.

**Conclusion: the binding constraint is memory latency hidden by per-thread MLP (number of
independent loads in flight), not warp count and not ALU throughput.** Optimizations that
reduce outstanding loads, or that serialize address generation, lose -- even when they
strictly reduce work. The next lever should be *fewer/wider load instructions* per byte
fetched, not fewer arithmetic instructions.

---

## Breakthrough: cooperative staging of x through shared memory

### Diagnosis that led to it

nvprof metrics are unavailable (`RmProfilingAdminOnly: 1`), so the bottleneck was found by
calibrated microbenchmarks instead. Measured on this machine (GPU1, ECC on):

- streaming read ceiling: **605 GB/s** (not the 732 GB/s spec number)
- streaming read by per-thread load width: 16B 604, 8B 598, 4B **533**, 2B **328** GB/s

Achieved bandwidth by quant type in mmvq (m=4096,n=1,k=14336): q8_0 332, q4_0 321, q5_K 225,
q6_K 208, q2_K 126 GB/s. Even q8_0 -- the simplest possible vec_dot -- reaches only 55% of
streaming, so the cap is structural.

Throughput tracks **bytes fetched per load instruction**:
a microbenchmark at 64 B/load hits 442 GB/s; q6_K mmvq at 26 B/load hits 208 GB/s (ratios
2.46 vs 2.12). **The kernel is load-ISSUE bound**, not bandwidth bound and not ALU bound.

Root cause: every ggml block size is 2 mod 4 (block_q6_K 210, block_q8_0 34, block_q4_0 18,
block_q3_K 110) because each block carries a 2-byte ggml_half beside a multiple-of-4 payload.
So `get_int_b2` must issue two 16-bit loads per quant word, and the scales and block scale
cost a load each: ~7 load instructions per block, ~26 bytes per load.

Modelled fetch strategies (48.2 MB, extraction actually performed):

| strategy | us | GB/s | vs current |
|---|---|---|---|
| packed 210B via `get_int_b2` (current) | 186 | 259 | 1.00x |
| aligned 32-bit only (needs padded stride) | 147 | 328 | 1.27x |
| **cooperative -> shared memory -> extract** | **98** | **491** | **1.90x** |

A padded block stride was investigated and REJECTED: mmvq and mmq share a tensor's physical
layout (chosen per call by batch size in `ggml_cuda_mul_mat`), so it cannot be scoped to mmvq;
a full CUDA repack buffer type is ~800-1200 lines with a silent-wrong-answer failure mode.
Staging needs none of it -- it reads from the 4-byte-aligned base *below* the data and keeps
the misalignment inside shared memory, so the global side is aligned with the weights untouched.

### Implementation note that decided the result

First cut was SLOWER (244 us vs 204). SASS showed 28 `LD.E ..., P0` -- a runtime loop bound
(`for k = threadIdx.x; k < nint; k += warp_size`) made ptxas emit predicated *generic* loads
for the staging reads instead of `LDG`, discarding the entire benefit. Fixing it to a
compile-time trip count plus an explicit `__ldg` gave `LD.E 0, LDG 13, LDS 56, STS 12`.

Registers went **71 -> 62** and smem to 10496 B, so occupancy improved as well (4 blocks/SM).

### Results (isolated kernel, us/run, m=4096 n=1 k=14336)

| type | before | after |
|---|---|---|
| q6_K | 204.1 | **164.5** |
| q3_K | 286.1 | **151.4** |
| q8_0 | 187.8 | **152.1** |
| q2_K | 152.8 | **142.6** |
| q5_K | 145.0 | 144.1 |
| q4_0 | 94.1 | 96.1 |
| q4_K | ~141 | 130.8 |

| # | change | t/s | verdict |
|---|---|---|---|
| 5 | cooperative shared-memory staging of x in `mul_mat_vec_q` | **20.43 +/- 0.03** | **KEPT** (PPL 2.7554 +/- 0.02151, identical to stock) |

Correctness: `test-backend-ops -o MUL_MAT -b CUDA0` passed 1193/1193 on 4 consecutive runs.
Each warp owns its own `x_stage` slice, so there is no cross-warp sharing to race on.

### Where the remaining time goes (post-staging)

Isolated q6_K kernel is now **164.5 us**. The calibrated fetch model says a staged fetch of this
data volume costs ~98 us, so the split is roughly **98 us fetch + 66 us compute**. The fetch is
therefore already at its modelled floor for this design.

Consequences for further work:
- Even a *free* fetch would leave 66 us, capping this kernel design at ~34 t/s.
- Geometry re-tuned after staging (nwarps x rows): best cell 4x8 = 163.45 us vs current 8x4 =
  165.56 us, i.e. 1.3% and mixed across types. Launch-geometry tuning is exhausted.
- Remaining lever inside the design: the staged image keeps the source misalignment, so
  `get_int_b2` still issues two `LDS.U.U16` per quant word. SASS LSU mix is LDG 13 + LDS 56 +
  STS 12 = 81 ops. Aligning the staged copy (funnel-shift during staging) would halve the LDS
  to ~28, giving ~53 LSU ops, an estimated 1.2-1.3x on the fetch and ~23-26 t/s overall.
  It requires an aligned read path in `get_int_b2` and per-block padding of the smem stride.
- Reaching 40 t/s would need the weights in a wide-load-friendly layout (16-byte loads), i.e.
  the rejected CUDA repack buffer type (~800-1200 lines, silent-wrong-answer risk).

---

## Attempts after staging (all reverted)

| # | change | q6_K us/run | t/s | verdict |
|---|---|---|---|---|
| 6 | re-tune nwarps x rows on staged kernel | best 163.45 (4x8) vs 165.56 (8x4) | - | REVERTED (1.3%, mixed across types) |
| 7 | strip misalignment while staging + branch-free funnel-shift `get_int_b2` | 218.79 | - | REVERTED |

Attempt 7 was the plan to halve the 56 `LDS.U.U16`. It failed because ptxas never proved the
staged pointer was 4-byte aligned, so the `sh == 0` select did not fold: SASS went to
`LDS 40, SHF 29, LD.E 16`, registers 62 -> 79, instructions 636 -> 768. Paying for a
funnel-shift everywhere without collapsing any load is strictly worse.

## Ceiling analysis: why 40 t/s is not reachable

Decisive measurement -- the dp4a/float math was stubbed out of `mul_mat_vec_q` while keeping
**all** staging and field extraction:

| variant | q6_K us/run |
|---|---|
| full kernel | 164.5 |
| **loads only, zero arithmetic** | **114.5** |

So the kernel is 114.5 us of memory work + 50 us of arithmetic (70/30).

- Even with **completely free arithmetic** the kernel cannot go below 114.5 us, which is
  20.43 * 164.5/114.5 = **29.4 t/s**.
- 40 t/s would require the whole kernel in 84 us -- *below the memory-only floor*. It is
  arithmetically unavailable, independent of how good the dot product gets.
- The memory side already runs at 48.17 MB / 114.5 us = **421 GB/s, 70% of this machine's
  605 GB/s streaming ceiling**, and the modelled optimum for this access pattern is 491 GB/s
  (81%). There is roughly 1.17x left in the fetch and maybe 2x in the arithmetic, i.e. a
  realistic hard ceiling near **25-27 t/s** for this kernel design, with substantial work.

A weight repack buys nothing here: cooperative staging already reaches 491 GB/s modelled,
essentially equal to the padded + 16-byte-load ceiling of 499 GB/s. The layout problem is
solved; what remains is the arithmetic cost of emulating DP4A on a GPU that lacks it.

Note on the premise: the 60 t/s figure assumes the 732 GB/s spec bandwidth. **ECC is enabled**
on both cards and the measured streaming ceiling is 605 GB/s, so the true pure-streaming bound
for a 22.4 GB model on two cards is ~54 t/s before any compute or overhead. Disabling ECC
(`nvidia-smi -e 0`, reboot) would recover roughly 10-20% of memory bandwidth -- that is a
user/system decision, deliberately not made here.

## Attempt 8: software pipeline (register prefetch) -- REVERTED, but it sharpens the model

Staging is otherwise strictly serial (load, barrier, compute, barrier), so the hypothesis was
that global load latency is exposed and can be hidden by issuing the next iteration's `__ldg`s
before the current dot products. Implemented by holding the next blocks in registers
(no second smem buffer needed).

| type | committed | prefetch pipeline |
|---|---|---|
| q6_K | 164.7 | 163.8 |
| q3_K | 151.4 | 158.7 |
| q4_0 | 96.1 | 98.1 |
| q4_K | 131.1 | 127.9 |
| q8_0 | 152.1 | 151.8 |

A wash, with registers 62 -> 75 and LDG 13 -> 21. **Reverted.**

The negative result is the useful part: the kernel is **not latency bound, it is LSU-ISSUE
bound.** At 4 blocks/SM the 32 resident warps already hide the load latency, so prefetching
buys nothing -- it does not reduce the number of load instructions, which is what actually
limits it. This also retires "overlap the arithmetic behind the memory" as a strategy: there is
no stall to fill.

Per-iteration LSU mix is **LDG 13 + LDS 56 + STS 12 = 81 ops**, and the 56 shared loads
dominate. They are 56 rather than 28 because the staged image keeps the source misalignment, so
`get_int_b2` reads each quant word as two `LDS.U.U16`.

**The one remaining kernel lever** is therefore to make those single 32-bit shared loads:
strip the misalignment while staging (nearly free -- the data is already in registers on its way
to shared memory) and give `vec_dot_*_q8_1` a compile-time "source is 4-byte aligned" template
parameter so `get_int_b2` can use `((const int *) x)[i32]` directly. Attempt 7 failed only
because it tried to let ptxas *infer* the alignment at run time; asserting it at compile time
avoids both the funnel-shift cost and the register growth.
Estimated LDS 56 -> 32, LSU 81 -> 57 (1.42x), i.e. roughly **25 t/s**.

---

## Attempt 9: templated aligned `get_int_b2` -- REVERTED

Gave `vec_dot_*_q8_1` a `bool aligned4` template parameter and stripped the misalignment while
staging (neighbour word via warp shuffle). Result: q6_K 193.8 us (vs 164.6) and a correctness
failure. Two lessons:

- The saving was far smaller than estimated. Only `vl`/`vh` go through `get_int_b2`; the two
  `scales` bytes and the block scale do not. So alignment can remove at most ~8 of the 81 LSU
  ops (LDS went 56 -> 48, not 56 -> 32), while the shuffles cost 32 instructions. The lever is
  smaller than the cost of pulling it.
- The correctness bug was a divergent `__shfl_sync` with a full mask, called under
  `if (threadIdx.x == warp_size-1)`. All lanes in the mask must execute the shuffle.

## Attempt 10: vdr = 2 for Q6_K + retuned launch geometry -- **KEPT**

`VDR_Q6_K_Q8_1_MMVQ` 1 -> 2, so each thread handles two consecutive int32 of the quant data.
For even `iqs`, `bq8_offset`, `scale_offset` and `vh_shift` are identical for `iqs` and `iqs+1`
(all three are floor-divisions with an even modulus), so the two halves share one `scales`
pointer, one `bq6_K->d`, and one pair of q8_1 `.ds` scales -- all previously fetched twice.
Per block: **57.5 vs 81 LSU ops, 29% fewer.** This does *not* widen the ql/qh loads; the 210-byte
block stride keeps those at 2-byte alignment regardless.

vdr=2 halves the threads per block-group, which moved the launch-geometry optimum. Swept against
the real benchmark (the isolated single-shape proxy was misleading -- it showed vdr=2 as 1.12x
while the full model showed no change until the geometry was retuned):

| nwarps x rows | t/s (tg256) |
|---|---|
| 8 x 4 (old tuning) | 20.35 |
| 4 x 4 | 22.97 |
| **2 x 4** | **23.31** |
| 2 x 2 | 22.67 |
| 1 x 4 | 22.22 |
| 1 x 2 | 22.04 |
| 2 x 8 / 4 x 8 | 19.96 / 18.78 |

Attribution (all tg256): vdr=1 8x4 = 20.35, vdr=1 4x4 = 21.05, vdr=2 4x4 = 22.95. Both the
vdr change and the geometry change contribute.

### The tuning now lives in the source, not in build flags

`-DP100_NWARPS/-DP100_ROWS/-DP100_MC_*` are no longer read. The measured Pascal values are
compiled in, gated on `__CUDA_ARCH_LIST__ == 600` so that MMVQ_PARAMETERS_GENERIC -- which
Ampere and later also fall through to -- is untouched. `__CUDA_ARCH_LIST__` is the correct test
because it is visible to **both** the host and device passes, and `calc_nwarps` feeds both
`__launch_bounds__` and the host-side launch configuration, which must agree.

A flagless build now reproduces the tuned result, so the "17% cliff with no warning" from
forgetting the flags is gone.

| # | change | t/s | verdict |
|---|---|---|---|
| 10 | vdr=2 for Q6_K + Pascal geometry 2x4 baked into source | **23.28 +/- 0.02** | **KEPT** |

## Attempt 11: vdr = 4 for Q6_K + geometry 2x2 -- **KEPT**

Extends the vdr=2 idea. The index-sharing condition holds for groups of 4 as well: verified
that for every `iqs` that is a multiple of 4, all four consecutive indices share
`bq8_offset`, `scale_offset` and `vh_shift`, and that the ql index, qh index and q8_1 lane
index are each consecutive with no wrap (`iqs % 8` is 0 or 4, so `qh_idx + l` and
`(iqs + l) % QI8_1` stay inside their arrays). `vec_dot_q6_K_q8_1` was rewritten to loop over
`vdr` so the factor is a single constant.

vdr = 8 was rejected on analysis: `scale_offset` changes at `iqs % 16 == 4`, so a group of 8
would need two scale reads, cutting the benefit to ~8% for a much larger register footprint.

Raising vdr raises register pressure, which keeps moving the geometry optimum, so the sweep
has to be redone each time -- and always against the real model, never the isolated shape
(vdr=4 is *better* than vdr=2 in isolation, 139.2 vs 143.7 us, but worse on the model at the
old geometry: 21.60 vs 23.28):

| vdr=4, nwarps x rows | regs | t/s (tg256) |
|---|---|---|
| **2 x 2** | 87 | **24.35** |
| 1 x 2 | 87 | 23.96 |
| 4 x 2 | 87 | 22.03 |
| 2 x 1 | 75 | 21.70 |
| 2 x 4 | 103 | 21.59 |
| 1 x 4 | 103 | 21.36 |
| 4 x 4 | 102 | 19.97 |
| 4 x 1 | 74 | 18.32 |
| 8 x 2 | 87 | 17.63 |
| 8 x 1 | 74 | 15.84 |

| # | change | t/s | verdict |
|---|---|---|---|
| 11 | vdr=4 for Q6_K + Pascal geometry 2x2 | **24.33 +/- 0.04** | **KEPT** |

Split after this change (isolated q6_K, m=4096 n=1 k=14336): full 143.7 us -> loads-only
121.3 us at vdr=2, i.e. vdr=2 already cut the arithmetic from 50 us to 22 us. Memory is now
~84% of the kernel, so remaining work has to come off the load path.

## Attempt 12: stage in 16-byte units (uint4) -- **KEPT**

The staging loop was moving 32 bits per lane, i.e. 128 bytes per warp per instruction, and at
vdr=4 that is 7 load + 7 store instructions per row per iteration. The warp's run of blocks is
contiguous, and shared memory already tolerates an arbitrary byte offset (that is what `mis[]`
is for), so the run can simply be fetched from the 16-byte-aligned address *below* it and the
offset carried into the extraction unchanged. One `uint4` instruction moves 512 bytes per warp.

Global loads and shared stores both drop ~4x (7 rounds -> 2). Registers also fell 87 -> 78.

| | q6_K iso us/run | t/s (tg256) |
|---|---|---|
| 32-bit staging | 131.5 | 24.33 |
| **uint4 staging** | **117.0** | **26.21** |

Geometry re-swept afterwards; 2x2 still optimal (2x4 24.69, 4x2 24.76, 1x2 26.00, 4x4 23.49).

| # | change | t/s | verdict |
|---|---|---|---|
| 12 | uint4 (128-bit) staging | **26.21 +/- 0.05** | **KEPT** |

---

# Final state: 17.45 -> 26.26 t/s (+50%)

| commit | change | t/s |
|---|---|---|
| (baseline) | HEAD b44f8fe6f | 17.45 |
| d8984de0 | remove `__vsubss4` from Q6_K/Q3_K vec_dot | 17.72 |
| c7e7faf6 | cooperative shared-memory staging of x | 20.35 |
| f07b913a | vdr=2 for Q6_K + Pascal geometry into source | 23.28 |
| 0732c729 | vdr=4 for Q6_K + geometry 2x2 | 24.33 |
| 9fc142d1 | uint4 (16-byte) staging | **26.26** |

Correctness at every kept step: `test-backend-ops -o MUL_MAT -b CUDA0` 1193/1193, and
**PPL 2.7554 +/- 0.02151, identical to the stock kernel** (the reference for this repo and
this `ppl.txt`; see the note above about CLAUDE.md's 2.6209).

## Effect across quant types (isolated kernel, m=4096 n=1 k=14336, us/run)

| type | staged-only (c7e7faf6) | final | |
|---|---|---|---|
| q6_K | 164.5 | **116.3** | -29% |
| q3_K | 151.4 | **142.8** | -6% |
| q8_0 | 152.1 | 150.6 | -1% |
| q4_0 | 96.1 | 95.4 | -1% |
| q4_K | 131.1 | 134.1 | +2% |
| q5_K | 144.1 | 146.8 | +2% |
| q2_K | 142.6 | 144.9 | +2% |

The ~2% regressions on q4_K/q5_K/q2_K come from the launch geometry, which is Pascal-wide and
was tuned against the Q6_K model (the only model available here). `calc_nwarps` already takes
`type`, so a per-type Pascal table would remove them; it needs a model of each type to tune
against, since the isolated single-shape proxy proved misleading.

## Why this stops around 26-27 t/s

Final measured split of the q6_K kernel: **95 us memory + ~21 us arithmetic**. Even with free
arithmetic the kernel cannot beat ~32 t/s, and 30 t/s needs the memory side cut as well.

The memory side is stuck on one structural fact: **every ggml block size is 2 mod 4**
(block_q6_K 210, block_q8_0 34, block_q4_0 18, block_q3_K 110), because each block carries a
2-byte `ggml_half` beside a multiple-of-4 payload. A 4-byte quant word at a 2-byte-aligned
address costs two memory instructions instead of one, and that cost cannot be moved, only
relocated:

- read it directly from global -> two 16-bit global loads (the original code)
- stage it and read from shared -> two 16-bit shared loads (current code)
- strip the misalignment while staging -> the funnel shift needs the neighbouring word, which
  costs a shuffle or a second load per word (attempts 7 and 9, both measured slower)
- read wider (`LDS.64`/`LDS.128`) -> the 16 useful bytes still straddle two aligned units, so
  the rotation reappears

Attempts 7 and 9 both confirmed this empirically. The only escape is weights that are actually
aligned in global memory, i.e. a repacked CUDA buffer layout (~800-1200 lines, and it must be
shared with the MMQ path -- see `padded_stride_study.md`). With 16-byte-aligned weights the
staging could be dropped entirely: each thread would read its 16 bytes of `ql` and `qh` with one
128-bit load each, ~5 memory instructions per vec_dot against 19 today. That is the remaining
lever, and it is a much larger win than the 1.27x the earlier padded-stride study estimated,
because it composes with vdr=4.

---

## Attempts 13-15: the non-mmvq 24% -- all REVERTED

After the mmvq work, `mul_mat_vec_q` is 76.3% of GPU time and everything else is 23.7%,
spread over ~15 kernels of 1-4% each. Reaching 30 t/s from 26.26 needs 12.5% of total time,
so this became worth attacking.

| # | change | t/s | verdict |
|---|---|---|---|
| 13 | enable CUDA graphs on Pascal (gate was `cc < VOLTA`, undocumented) | 25.97 | REVERTED |
| 14 | 256-thread instead of 1024-thread `rms_norm_f32` block | 26.23 | REVERTED (neutral) |
| 15 | 32-bit byte-offset indexing for q8_1 blocks in vec_dot (sizeof 36) | XMAD 197 -> 199 | REVERTED (compiler already did it) |

Attempt 13 is the interesting one: the arch gate on CUDA graphs carries no comment and sm_60
does support them, but capturing and re-validating the graph costs slightly more than the
launch overhead it saves here. Attempt 14 confirms these kernels are latency bound, not
reduction bound -- a 5120-element RMS norm takes 9.5 us regardless of block size, because
at batch 1 it is one block on one SM and the duration is mostly fixed overhead.

Profile after all kept changes (GPU compute, model load excluded, 1148 ms total):

| share | kernel |
|---|---|
| 76.3% | `mul_mat_vec_q<q6_K, ncols=1>` |
| 3.6% | `rms_norm_f32<1024>` (9.5 us x 4386) |
| 3.5% | `quantize_q8_1` (2.4 us x 16898, one per mmvq) |
| 2.7% | `k_bin_bcast` |
| 2.4% | `flash_attn_ext_vec` |
| 11.5% | ~10 further kernels, each < 1.5% |

The remainder is latency-bound elementwise and normalisation work at batch 1, where every
kernel costs 2-9 us almost regardless of how little it does. Halving all of it would be worth
~12% and would take a dozen separate optimisations; CUDA graphs were the one change that could
have addressed it wholesale, and it does not pay here.

## Final position

**26.26 t/s, +50% over the 17.45 baseline.** mmvq's memory path now runs at ~507 GB/s of the
machine's 605 GB/s streaming ceiling (84%) and its arithmetic is within a few instructions of
minimal for a GPU without DP4A, so the kernel itself is close to done. 30 t/s needs either
aligned weights in global memory (the repack -- see the alignment analysis above) or a broad
attack on the batch-1 launch-latency tail.

---

# Is 40 t/s reachable? No -- proof from measured quantities

All inputs below are measured on this machine, not spec sheets.

- weights 22.42 GB, tensor-split -> **11.21 GB read per GPU per token**
- streaming read ceiling, measured, ECC on: **605 GB/s** per GPU (spec is 732; 83% is normal)
- at 26.26 t/s = 38.1 ms/token, split 29.1 ms mmvq / 9.0 ms everything else (nvprof)

```
40 t/s              = 25.0 ms/token
  - 9.0 ms non-mmvq = 16.0 ms available for mmvq
  11.21 GB / 16.0 ms = 702 GB/s per GPU   vs a 605 GB/s ceiling   -> IMPOSSIBLE
```

Even granting a *perfect* mmvq -- zero arithmetic, running at the full pure-streaming rate while
still unpacking 6-bit quants, which is not achievable -- the ceiling is:

```
11.21 GB / 605 GB/s = 18.5 ms  ->  54.0 t/s with zero compute AND zero other kernels
  + the measured 9.0 ms of non-mmvq work      ->  36.3 t/s ABSOLUTE CEILING
```

And with the arithmetic cost that actually exists, the measured mmvq memory floor puts the
practical ceiling at **30.5 t/s**.

**40 t/s cannot be reached by any kernel optimization on this hardware.** It requires reducing
*bytes read per token*, which means speculative decoding (MTP) or a smaller quant -- both
explicitly out of scope for this goal. The bandwidth-derived 60 t/s figure in the original brief
assumed the 732 GB/s spec number and no compute or non-mmvq time; the honest equivalent is 54 t/s
of pure streaming, 36.3 t/s once the rest of decode is counted.

Delivered: **17.45 -> 26.26 t/s, +50%**, which is 86% of the 30.5 t/s practical ceiling.

### The impossibility proof's key assumption, verified

The ceiling above assumes every weight is read every token. Checked directly against the GGUF
metadata rather than assumed: `general.architecture = qwen35`, 866 tensors, `block_count = 65`,
`embedding_length = 5120`, `feed_forward_length = 17408`, with SSM keys (`ssm.state_size = 128`,
`ssm.inner_size = 6144`) confirming the hybrid attention/gated-delta-net design -- and
**no `expert_count` key, so the model is dense**. There is no active-experts subset that would
reduce bytes per token.

Independently corroborated by measurement: mmvq takes 29.1 ms/token, and at the kernel's measured
414 GB/s per GPU across two GPUs that is ~24 GB moved per token, matching the full 22.42 GB
weight set. The assumption holds, so the 702 GB/s-vs-605 GB/s contradiction stands.

## Attempt 16: pricing the q8_1 activation loads -- NOT WORTH DOING

`quantize_row_q8_1_cuda` is called only from mmvq and `quantize_mmq_q8_1_cuda` only from mmq, so
mmvq's q8_1 activation buffer is exclusively its own and could legally be padded from 36 to 48
bytes per block. That would make `qs` 16-byte aligned, turning the four consecutive `u` loads
per `i` (4 consecutive int32 = 16 contiguous bytes at vdr=4) into a single `uint4`.

Before doing the invasive version (a padded CUDA-only q8_1 struct threaded through all 23
`vec_dot_*_q8_1` signatures plus the quantize path), the ceiling was measured directly by
collapsing the four loads into one -- deliberately wrong results, timing only:

| | q6_K iso us/run |
|---|---|
| baseline | 116.61 |
| four `u` loads collapsed to one (upper bound) | 114.28 |

**2%.** The activation is tiny and L1/L2-resident, so those loads are already nearly free. The
padding work would buy ~26.8 t/s. Not done.

## Every remaining lever, now measured rather than estimated

| lever | measured result |
|---|---|
| q8_1 padding for 128-bit activation loads | 2% upper bound |
| CUDA graphs on Pascal | -1% (capture cost exceeds launch saving) |
| `rms_norm` 256- vs 1024-thread block | neutral (latency bound, not reduction bound) |
| 32-bit q8_1 pointer math | no-op (compiler already did it) |
| aligning the staged copy (3 variants) | structurally break-even: rotation cost and load saving scale with the same bytes |
| **weight repack to aligned global layout** | **the only real one: ~34 t/s, ~800-1200 lines, shared with the MMQ path, silent-wrong-answer failure mode** |

The optimisation space reachable without the repack is exhausted at **26.26 t/s**, 86% of the
30.5 t/s practical ceiling and 72% of the 36.3 t/s absolute one.

## Attempt 17: retune the multi-column (MTP) path -- nothing to tune

The `ncols_dst` 2-8 geometry was still carrying values tuned for the pre-staging kernel, and
this user runs MTP in production, so it was worth re-sweeping. Confirmed via nvprof that these
shapes really do run `mul_mat_vec_q<q6_K, ncols_dst=4>` and not MMQ.

| nwarps x rows | n=2 | n=4 | n=8 |
|---|---|---|---|
| 4x2 (current) | 167.89 | 273.14 | 483.25 |
| 2x2 | 167.63 | 273.37 | 483.33 |
| 2x4 | 167.84 | 273.49 | 483.98 |
| 4x4 | 167.85 | 273.06 | 483.81 |
| 1x2 | 167.78 | 273.50 | 483.34 |
| 8x2 | 167.86 | 273.28 | 483.33 |

Byte-identical across every configuration. The reason is visible in the scaling: n=4 costs only
2.3x n=1 for 4x the output, because the weights are read once regardless of the column count.
The multi-column path is therefore **compute bound, not load bound**, which is exactly why the
geometry -- which only shapes the memory access pattern -- has no effect on it. Left as is.

Useful consequence for MTP: the marginal cost of a draft column is low (~55 us per extra column
against 116 us for the first), so speculative decoding amortises well on this kernel.

## Attempt 18: fold the scale and the int->float conversion out of the vdr loop -- **KEPT**

With vdr=4 the inner body was `sumf += d8[i] * (dp4a(...) * sc)` for each of the 8 (l, i) pairs.
But `scales[4*i]` and the q8_1 scale are **constant across the whole vdr group** -- that is the
same sharing property vdr exploits for the loads. So the integer accumulator can absorb all four
`l` values first (dp4a already takes an accumulator, so chaining is free) and the scaling
collapses to once per `i`:

- the integer multiply by `sc`, **three XMADs each on Pascal, which has no IMAD**, goes from 8 per
  vec_dot to 2
- the int->float conversion goes from 8 to 2

Safe because peak `|acc|` is vdr*4*128*128 = 262144, well inside float's exactly-representable
integer range, so folding the group before the conversion loses nothing. It also rounds twice
per `i` instead of four times, so it is marginally *more* accurate -- PPL came back unchanged.

| | total instr | XMAD | I2F | regs | q6_K iso | t/s |
|---|---|---|---|---|---|---|
| before | 678 | 197 | 16 | 78 | 116.4 us | 26.24 |
| after | **624** | **161** | **8** | **66** | **108.3 us** | **27.03** |

Geometry re-swept afterwards (registers fell 78 -> 66); 2x2 still optimal: 2x4 26.42,
4x2 26.37, 1x2 26.91, 4x4 25.26, 1x4 25.95.

| # | change | t/s | verdict |
|---|---|---|---|
| 18 | integer accumulator across the vdr group | **27.03 +/- 0.04** | **KEPT** (PPL 2.7554, unchanged) |

## Attempts 19-21: rotate the staged copy to 4-byte alignment -- REVERTED (third and final try)

With the split now at 95.2 us memory / 13.7 us arithmetic, arithmetic is 62% of the *instructions*
but only 12.5% of the *time*. That says the ALU is idle and the memory pipe binds, so trading LSU
work for ALU work should win -- which reopened the alignment idea a third time.

Implemented properly this time: each block gets its own 16-byte-aligned slot (padded to a power of
two so the block index is a shift), the source misalignment is rotated away with funnel shifts on
the way in, and `get_int_b2` is templated on a compile-time `aligned4` flag so `mul_mat_vec_q`
reads the staged copy with 32-bit shared loads while the MoE kernel keeps the unaligned path.

It works mechanically -- `LDS.U.U16` 34 -> 2, replaced by 16 `LDS.32` -- and still loses:

| variant | LDG | LDS | LSU | regs | q6_K iso | t/s |
|---|---|---|---|---|---|---|
| **kept (unaligned reader)** | **8** | **34** | **46** | **66** | **108.9 us** | **27.03** |
| rotated + aligned, src_u4 = dst+1 | 14 | 18 | 36 | 71 | 111.4 us | 26.66 |
| rotated + aligned, src_u4 = dst | 14 | 22 | 40 | 71 | 110.3 us | 26.85 |
| + clamped indices to keep LDG.128 | 14 | 22 | 40 | 68 | 114.1 us | 26.50 |

**The correction this forces: LDG and LDS are not interchangeable.** The last row has *fewer* total
LSU operations than the kept version (40 vs 46) and is still 5% slower, because per-block staging
turns one contiguous run into four separate base addresses and global loads cost far more than
shared ones. "LSU-bound" was too coarse a model; it is the *global* load count that binds.

Three independent attempts (9, 19-21) now agree: the 2-mod-4 alignment tax cannot be removed
profitably inside shared memory. Only aligned data in global memory would do it.

## Attempt 22: compile-time staging bound -- REVERTED

`nu4 = (m + nblk*blck_size + 15)/16` costs a 64-bit multiply by a non-power-of-two plus a divide
per row. Replacing it with the compile-time `stage_u4` cut the loop body 421 -> 383 and address
XMADs 56 -> 38, but staged ~3% more bytes every iteration and measured 26.65. Using the
compile-time *product* instead (keeping `m` runtime) still measured 26.72 against 27.03, with
registers 66 -> 70. The compiler's original form wins; left alone.

## Attempts 23-24: wide shared reads with funnel-shift at extraction -- REVERTED

The better form of the alignment idea: leave staging alone (so no extra *global* loads, the
mistake in attempts 19-21) and instead read the vdr contiguous quant words with vdr+1 aligned
32-bit shared loads plus funnel shifts. For vdr=4 that is 5 shared loads instead of 8, the shift
is constant for the whole group, and the shifts land on the idle ALU.

| variant | LDG | LDS | LD.E | regs | q6_K iso | t/s |
|---|---|---|---|---|---|---|
| **kept** | **8** | **34** | **0** | **66** | **108.9 us** | **27.03** |
| wide reads via uintptr_t arithmetic | 14 | 6 | (generic) | 86 | 116.9 us | 25.61 |
| wide reads, provenance preserved | 14 | 26 | 0 | 77 | 109.5 us | 26.86 |

The first version was sabotaged by a subtle bug worth recording: **casting a shared-memory pointer
through `uintptr_t` and back loses the address space**, so ptxas emitted generic loads instead of
`LDS`. Deriving the aligned pointer from the original with `char *` arithmetic fixes it (`LD.E`
back to 0, `LDS` 6 -> 26) and recovers most of the loss -- but the `vlv`/`vhv` arrays cost 11
registers, and at 2 warps/block that outweighs the 8 shared loads saved.

Every remaining variant now trades one resource for another and nets negative: cutting shared
loads costs registers or global loads, cutting instructions costs registers, cutting staged bytes
costs global traffic. 27.03 is a deep local optimum for this kernel structure.
