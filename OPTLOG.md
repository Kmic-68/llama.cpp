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

## Attempt 36: q8_1 activation-quantization cache (KEPT)
`quantize_q8_1` was called exactly once per `mul_mat_vec_q` (16898 times), but q/k/v share one
normed activation and gate/up share another, so ~40% of those launches recomputed an identical
buffer. Added a per-device cache to `ggml_backend_cuda_context` keyed on
(src1 node, src1 data ptr, src0->type, byte size), invalidated at the start of every
`ggml_backend_cuda_graph_compute`. Backing store is a persistent `cudaMalloc` that only grows.
- 27.03 -> **27.49 t/s**
- test-backend-ops -o MUL_MAT: 1193/1193
- PPL 2.7554 +/- 0.02151 (identical to stock)
- KEPT

## Session 2026-08-30 (interrupted, safe stopping point)

State: HEAD = 246515a32, **27.49 t/s**. Working tree has two UNCOMMITTED, NON-KEPT edits:
- `ggml/src/ggml-cuda/norm.cu` — 4x unroll of both rms_norm loops PLUS temporary `DBG_NORM`
  debug printfs. Neutral (27.46 vs 27.49) and has debug cruft: **`git checkout -- ggml/src/ggml-cuda/norm.cu`**.
- `ggml/src/ggml-cuda/mmvq.cu` — geometry moved into named constants
  `P100_MMVQ_NWARPS_1 2` / `P100_MMVQ_ROWS_1 2`. Behaviourally identical to HEAD; keep or revert.
Then rebuild so the binary matches the source.

### Attempt 36: q8_1 activation-quantization cache — KEPT (commit 246515a32)
27.03 -> 27.49 t/s. PPL 2.7554 +/- 0.02151 (identical). test-backend-ops MUL_MAT 1193/1193.
quantize_q8_1 launches dropped from 1-per-mmvq to ~0.52-per-mmvq.

### Attempt 37: rms_norm 4x loop unroll — REVERTED
rms_norm_f32<1024> is 130 calls/token/GPU at 9.4 us (ncols=5120, nrows=1, ONE block of 1024
threads on one SM) = 1.22 ms/token = 3.4% of the token. Unrolling both loops 4x to overlap the
loads only moved the kernel 9.60 -> 9.36 us and the metric 27.49 -> 27.46. The 9.4 us is NOT
loop memory latency; the real cause is still unidentified (40 KB of traffic in 9.4 us is ~4 GB/s).

### Attempt 38: PROBE — remove all mmvq arithmetic (P100_MEMONLY)
Replaced `ggml_cuda_dp4a(...)` with `acc[i] += (vil4|vih4) ^ u`, deleting ~64 instructions per
loop iteration while keeping every load. Result: **27.72 t/s (+0.8%)**.
**=> mul_mat_vec_q is DRAM-bandwidth bound, not issue bound. All inner-loop instruction-count
work is dead: FP16/HFMA2 rewrites, cheaper dp4a, cheaper unpack, LOP3 folding. Do not pursue.**

### Attempt 39: PROBE — 4-byte-aligned shared reads (P100_ALIGNPROBE)
Replaced the two 16-bit `get_int_b2` shared loads with one aligned 32-bit load (wrong data, right
cost), halving LDS from 40 to 20 per iteration: **27.43 t/s**. No gain. The 2-mod-4 alignment tax
inside shared memory costs nothing. Corollary: the earlier "LSU-bound" model is wrong too.

### Attempt 40: geometry re-sweep on top of uint4 staging — all worse, 2x2 stays
| nwarps x rows_per_cuda_block | t/s   |
|-----------------------------|-------|
| 2 x 2 (current)             | 27.48 |
| 2 x 4                       | 26.88 |
| 2 x 1                       | 24.55 |
| 4 x 1                       | 22.02 |
rows_per_cuda_block=1 is catastrophic => q8_1 activation re-reads are expensive despite being
L2-resident; the y-vector reuse across 2 rows is load-bearing.

### Measured time budget at 27.49 t/s (36.4 ms/token, per GPU)
- mul_mat_vec_q      24.8 ms  (11.2 GB of weights => **453 GB/s**, vs 605 GB/s measured streaming ceiling)
- all other kernels   7.5 ms
- GPU idle           ~4.1 ms  (**11%** — largest single remaining pool)
Derived from nvprof: 255458 mmvq launches for 256 tokens (499/token/GPU); mmvq = 65% of GPU
activities at -n 256, 76% at -n 64.

### Tail breakdown (ms per token per GPU, sums to 7.5)
rms_norm_f32<1024> 1.22 | k_bin_bcast 0.92 | flash_attn_ext_vec 0.88 | quantize_q8_1 0.61 |
gated_delta_net 0.45 | k_get_rows_float_vec 0.44 | PtoP memcpy 0.34 | unary_gated 0.34 |
l2_norm 0.33 | cpy_scalar 0.32 | rms_norm<256> 0.30 | rest ~1.3
~920 kernel launches per token per GPU; most tail kernels sit near a ~1.5-3 us floor.

### Next lead when work resumes (in priority order)
1. **The 4.1 ms/token GPU idle (11%, worth ~+3 t/s).** Closing it entirely would give ~30.8 t/s,
   which is exactly the goal. CUDA graphs are hard-disabled on Pascal in
   `ggml_cuda_graph_set_enabled` (`cc < GGML_CUDA_CC_VOLTA`, ggml-cuda.cu ~4244). Lifting that was
   tried once and measured worse (25.97 vs 26.21), but that predates several changes and the pool
   is now the biggest one. I was mid-measurement of whether the CPU launch thread is saturated
   (sample utime+stime from /proc/<pid>/stat over a 10 s window during generation) when
   interrupted — that measurement decides whether the idle is CPU launch cost (=> graphs / fewer
   launches) or cross-device synchronisation in the tensor-split path (=> different fix).
2. **rms_norm_f32<1024>** — find why 40 KB takes 9.4 us on one block, then fix (float4 loads, or a
   different block size). Worth ~+0.6 t/s. Unrolling alone is not the answer.
3. mmvq DRAM efficiency: 453 of 605 GB/s. Geometry is exhausted; whatever the 25% gap is, it is
   not instructions, not LDS, and not block shape.

## Attempt 41: PROBES — where mul_mat_vec_q's time actually goes
Three probes, each keeping the memory traffic and deleting one thing, measured with nvprof at
-n 256 -r 1 (mmvq total over 255458 launches, both GPUs):
| variant                                        | mmvq total | vs baseline |
|------------------------------------------------|-----------:|------------:|
| baseline                                        |   12.684 s |          -- |
| staging only (no dot product, no y loads)       |   10.446 s |      -17.6% |
| dot product kept, q8_1 activation replaced by a constant | 10.975 s | **-13.5%** |
| dp4a deleted, all loads kept (attempt 38)       |        n/a |       -0.8% |
So the gap between the kernel's 453 GB/s and the card's 605 GB/s streaming ceiling is almost
entirely the **q8_1 activation loads**, not the weights, not the arithmetic. The staging pattern
on its own reaches 552 GB/s (91% of the ceiling).

## Attempt 42: cooperative staging of the q8_1 activation — KEPT
Root cause: within a warp, consecutive lanes read 32-bit words 16 bytes apart inside a 36-byte
`block_q8_1` and then jump to the next block, so each of the 8 activation loads per iteration
fans out into many transactions. The bytes are L2-resident (every block of the grid reads the
same activation), so this costs request throughput, not bandwidth. In SASS the activation was
8 LDG.E.CI + 2 LDG.E.CI.U16 per iteration against only 4 LDG.E.CI.128 for the weights.
Fix: stage the warp's contiguous run of q8_1 blocks into shared memory with coalesced uint4
loads, exactly like the weights, and read it back from shared. Gated to ncols_dst == 1 on
Pascal, and to runs of at most 2048 bytes.
- **27.49 -> 28.98 t/s (+5.4%)**
- test-backend-ops -o MUL_MAT: 1193/1193
- PPL 2.7554 +/- 0.02151 (identical to stock)
- KEPT

## Attempt 43: block-wide (instead of per-warp) q8_1 staging — REVERTED
The warps of a block cover a contiguous run of x blocks, so one staged copy could serve the whole
block and halve the activation load instructions for the same shared memory. It requires
__syncthreads instead of __syncwarp, and a uniform loop over the block's base block.
28.98 -> **28.21**. The block-wide barrier costs more than the saved loads. REVERTED.

## Attempt 44: geometry re-sweep after the activation staging — 2x2 still optimal
| nwarps x rows_per_cuda_block | t/s   |
|-----------------------------|-------|
| 2 x 2 (current)             | 28.98 |
| 1 x 2                       | 28.72 |
| 2 x 4                       | 26.71 |

## Attempt 45: PROBE — 4-byte-aligned x reads from shared, re-run
Halving the 36 LDS.U.U16 per iteration is now worth **+0.35%** (28.98 -> 29.08), up from zero
before the activation staging but still not worth the repack. The kernel is not LDS bound.

## Attempt 46: keep the rms_norm row in registers — KEPT
rms_norm_f32<1024> runs with grid (1,1,1) and 1024 threads on ncols=5120: one block owns the row,
so nothing covers its memory latency, and it read the row twice (sum of squares, then scale).
When the row fits in a fixed number of registers per thread (max_regs = 8), hold it there and skip
the second read. Falls back to the strided loops for longer rows.
- **28.98 -> 29.27 +/- 0.10 t/s** (a variant that also did l2_norm and norm measured 29.29, within
  noise, so only the rms_norm change was kept)
- RMS_NORM 51/51, RMS_NORM_MUL_ADD 30/30, NORM 50/50
- PPL 2.7554 +/- 0.02151 (identical to stock)
- KEPT

## Attempts 47-51: mmvq restructures around the activation staging — ALL REVERTED
The no-activation probe still shows 30.70 t/s, so ~5% of the token is the residual activation
cost. Five restructures aimed at it, none paid:
| variant                                                              | t/s   |
|----------------------------------------------------------------------|-------|
| baseline (2 warps x 2 rows, per-warp activation stage)                | 29.32 |
| block-wide activation stage (__syncthreads instead of __syncwarp)     | 28.21 |
| warps own distinct rows + block-wide stage, 4 rows/block              | 29.19 |
| same, 2 rows/block                                                    | 29.17 |
| 4 rows/block with the weight stage chunked 2 rows at a time           | 27.12 |
| 1 warp x 4 rows (4x less activation traffic, 13 blocks/SM)            | 27.58 |
| register-carried prefetch pipeline (one iteration ahead)              | 28.74 |
The 1x4 result is the informative one: it moves a quarter of the activation bytes and is still
6% slower, so what binds is **warps resident per SM**, not L2 traffic or load instructions.
Shared memory is the occupancy limiter (6144 B/block -> 10 blocks/SM), which is why every variant
that spends more shared memory to save traffic loses. mul_mat_vec_q is a firm local optimum here.

## The real find: sm_60 has no integer divider
64-bit division in per-element index math costs dozens of instructions on Pascal, and several tail
kernels did four to eight of them per element. Replacing them with the existing 32-bit
multiply-shift helpers (init_fastdiv_values/fastdiv/fast_div_modulo) is worth more than anything
left in mul_mat_vec_q.

## Attempt 52: contiguous fast path for the elementwise binary kernels — KEPT
When every operand has the destination's shape and is contiguous (residual adds, elementwise
gates) there is no broadcasting to resolve, so skip the generic kernel's per-element fastmodulo
and its two-elements-per-thread stride loop entirely.
k_bin_bcast 3.07 -> 2.09 us. **29.32 -> 29.44 t/s.** Block size 64/128/256 all equivalent.

## Attempt 53: fastdiv in cpy_scalar — KEPT
Eight 64-bit divisions per element. 4.85 -> 2.54 us. **29.44 -> 29.65 t/s.**

## Attempt 54: fastdiv in the gated unary kernels and in concat_cont — KEPT
Two 64-bit divisions per element each. unary_gated 2.76 -> 2.57 us, concat_cont 4.40 -> 3.95 us.
**29.65 -> 29.74 t/s.**

## Attempt 55: float4 loads in the norm kernels — KEPT
rms_norm_f32<1024> runs as a single block on a 5120-wide row, so the SM's request throughput -- not
bandwidth -- is what limits it; one float4 request carries four times the payload of a float one.
Guarded on ncols % 4 == 0, 16-byte alignment, and (for the fused mul/add) the operand having the
same width so the fastmodulo is the identity.
rms_norm_f32<1024> 6.83 -> ~5 us. **29.74 -> 29.98 t/s** (-r 3), 29.81 +/- 0.20 on a -r 5 rerun.
The same treatment for l2_norm_f32 and norm_f32 is worth ~0.2 t/s (29.78 without vs 29.98 with).

### Note on perplexity
This batch measures **PPL 2.7565 +/- 0.02153** against the stock 2.7554 +/- 0.02151 -- the first
change in the session not to reproduce the stock value exactly. The cause is the norm kernels'
reassociated FP32 reduction (four floats per register now sum in a different order); the index-math
changes are bit-exact. The shift is 0.04% of the value and 5% of the error bar, and
test-backend-ops passes RMS_NORM 51/51, RMS_NORM_MUL_ADD 30/30, L2_NORM 20/20, NORM 50/50,
GROUP_NORM 2/2, ADD 99/99, MUL 91/91, CPY 246/246, CONCAT 177/177, SWIGLU 24/24, SET_ROWS 159/159.

### Time budget at ~29.9 t/s (per token per GPU)
mul_mat_vec_q 23.0 ms | other kernels 6.0 ms | GPU idle ~4.6 ms

## Attempt 56: flash-attn vec was told the wrong KV tile size — KEPT
`ggml_cuda_flash_attn_ext_vec_case_impl` passes `D` to `launch_fattn` as nbatch_fa, but the vec
kernel steps its KV loop by `nthreads`, not `D`. launch_fattn uses nbatch_fa only to compute
`ntiles_KV`, which caps how many blocks may split the KV range, so wherever nthreads < D the
parallelism is understated -- on Pascal nthreads is 128 against D = 256, so it is halved. With a
256-long KV that made ntiles_KV = 1, pinning parallel_blocks at 1 and running the whole attention
on **12 blocks of a 56-SM GPU** for 45-51 us a call. Passing `nthreads` is simply the accurate
value and helps at every context length.
- flash_attn_ext_vec 44.8 -> ~30 us
- **29.9 -> 30.11 +/- 0.20 t/s**
- test-backend-ops -o FLASH_ATTN_EXT: 3949/3949
- PPL 2.7565 +/- 0.02153 (unchanged from the previous commit -- this change is bit-neutral)
- KEPT

**30 t/s reached.** 17.51 -> 30.11 = +72%.

## Attempt 57: lower vdr for q6_K to buy occupancy — REVERTED
Every failed mmvq restructure lost the same way: it spent shared memory to save traffic and gave
up blocks per SM. vdr sets how many quant words a thread takes, so lowering it shrinks both stages
(vdr=2 would put shared memory at ~3.3 kB and occupancy at 19 blocks/SM against 10). It does not
help -- the loss from the shorter dot product swamps the extra occupancy:
| VDR_Q6_K_Q8_1_MMVQ | t/s   |
|--------------------|-------|
| 4 (current)        | 30.11 |
| 2                  | 27.00 |
| 1                  | 21.34 |
Both variants pass test-backend-ops MUL_MAT 1193/1193. vdr = 4 stands.

## Where 40 t/s stands
At 30.11 t/s the token costs 33.2 ms, of which mul_mat_vec_q is 23.0 ms per GPU (11.2 GB of
weights at ~487 GB/s), other kernels ~5.5 ms and launch/sync overhead ~4.6 ms. 40 t/s is 25 ms, so
**even a free tail and zero gaps cap the current mmvq at 43.5 t/s** -- the target needs the matmul
itself down near its 20.3 ms staging-only floor (552 GB/s) *and* the remaining 10 ms of tail and
overhead cut to under 5. Getting there is not a matter of one more kernel: it needs either fewer
launches (roughly 2150 per token per GPU today) or a weight layout that streams closer to the
605 GB/s wall.

## Attempt 58: restore bit-identical output — KEPT
The float4 reductions in the norm kernels changed the summation order (thread t took a contiguous
quad instead of the reference's strided columns t, t+B, ...), so the per-thread partials and the
reduction tree differed and perplexity moved 2.7554 -> 2.7565. Reverting just those three blocks,
keeping the register caching, restores **PPL = 2.7554 +/- 0.02151, exactly stock**. That also
confirms empirically what was until now only an argument: the fastdiv index-maths changes, the
contiguous elementwise fast path and the flash-attn tile-size fix are all bit-exact.

Three attempts to buy the speed back without touching the arithmetic, all no better:
| variant                                                             | t/s   |
|---------------------------------------------------------------------|-------|
| bit-exact (register cached, strided ownership)                       | 29.94 |
| bit-exact reduction + float4 second pass (elementwise, so bit-exact) | 29.40 |
| bit-exact + float4 contiguous elementwise kernel                     | 29.62 |
The second-pass idea fails because it must re-read x; the register caching was worth more than the
wider access. The elementwise float4 fails because it uses four times fewer threads, dropping these
small launches from 20 blocks to 5 -- they are launch-floor bound, so fewer blocks costs more than
wider loads save.

## Measurement caveat — read this before trusting any small delta above
GPU0 also serves Sunshine and the desktop. When the machine is in use it draws cycles from GPU0,
and because -sm tensor makes both GPUs rendezvous every layer, the whole token rate follows. Three
*identical* back-to-back runs measured 29.32, 27.97 (+/- 1.25) and 25.16 (+/- 1.87) while the
machine was being used, against 29.94 +/- 0.10 for the same binary when it was idle. Temperatures
were 67-69 C with no throttle flags, so this is contention, not thermal.

**Any A/B difference below about 0.5 t/s in this log is inside that noise** unless it was taken on
an idle machine with repeats. The large results (the activation staging at +5.4%, the fastdiv work,
the flash-attn fix) are well clear of it; the 0.17 t/s between the bit-exact and float4 norm builds
is not, and should be treated as unmeasured rather than as a real cost.

## Sustained throughput, and a correction to the noise diagnosis above

The "measurement caveat" section above blames desktop/Sunshine contention for the run-to-run
variance and states "contention, not thermal". **That is wrong.** It was inferred from a sample
taken at 67 C before the cards had saturated. Sampling the throttle reasons *during* a sustained
load shows what actually happens:

| state | GPU0 | GPU1 |
|---|---|---|
| start of load | 66 C, 1328 MHz, no flags | 69 C, 1328 MHz, no flags |
| ~1 min in | 72 C, 1265 MHz, **sw_power_cap active** | 76 C, 1265 MHz, **sw_power_cap active** |
| ~2 min in | 76 C, 1252 MHz, sw_power_cap | 79 C, 1139 MHz, **sw_thermal_slowdown active** |
| steady state | 79 C, ~1150-1320 MHz, sw_power_cap | 79 C, **949 MHz**, sw_thermal_slowdown |

Both cards hit the 175 W power cap first, then GPU1 hits thermal slowdown at 79 C and drops to
around 950 MHz - roughly 70% of its 1328 MHz boost. GPU1 runs hotter and throttles harder than
GPU0 at every point, which points at airflow rather than anything in software.

Measured, same binary, same command:

| condition | t/s |
|---|---|
| tg256, cards cold (53/54 C) | **29.59 +/- 0.20** |
| tg256, cards at steady state (77/78 C) | **24.77 +/- 2.36** |
| tg2048, from steady state | **22.74 +/- 1.23** |
| tg4096, from cold (so partly inflated) | 24.19 +/- 2.87 |

So the honest headline is two numbers, not one: **~29.6 t/s burst, ~23-25 t/s sustained**, and the
gap is entirely power and cooling. Note tg256-hot (24.77) and tg2048-hot (22.74) are close, so the
growing KV cache costs far less than the throttling does - the flash-attn path scales better with
context than the thermal envelope does with time.

This is not addressable in kernels, and CLAUDE.md forbids touching nvidia-smi power/clock settings.
Better airflow over GPU1 specifically would recover most of it.

**Consequence for every A/B number in this log:** they were taken at whatever thermal state the
machine happened to be in. Comparisons made back to back within one command are roughly fair
(both sides hot); comparisons made minutes apart are not. Anything below ~0.5 t/s should be
re-measured from a controlled thermal state before being believed. The large results are far
enough clear of this to stand.

# Multi-column path (speculative decoding / MTP, small batches)

MTP with the model's built-in head measured **32.94 t/s at 88% acceptance** -- only 1.11x over
non-speculative decode, which is poor for that acceptance rate. Profiling the MTP run shows why:
`mul_mat_vec_q<q6_K, ncols=7>` is **36.7% of GPU time at 196 us a call**, against 99 us for the
one-column kernel that streams the same weights. Everything done earlier this session was gated to
`ncols_dst == 1`, so the hottest kernel in this workload ran the *unoptimised* generic path.

## A benchmark that is actually usable
`llama-bench -p 512 -n 0 -b 7 -ub 7` is 84% `mul_mat_vec_q<ncols=7>`, deterministic, and reports
+/- 0.05 instead of tg256's +/- 0.2. Baseline **61.08 t/s, kernel 188.13 us**.

Note the probe technique used earlier is *invalid* on a speculative workload: replacing the
activation with a constant drove acceptance to 0%, so every draft was rejected and the run fell
back to one-column stepping -- the workload itself depends on the model being correct. Prompt
processing does fixed work regardless, hence the switch.

## Attempt 59: extend the activation staging to ncols_dst > 1 — REVERTED
The probe said deleting the activation makes the kernel **3.7x faster** (188 -> 50.9 us), so
staging looked like the answer. It gained ~1% (185.3 us). The probe conflated two things: it
removes the fan-out *and* the traffic, and staging only fixes the fan-out.

The real problem is **volume**. Every one of the ~2560 blocks re-reads the whole activation:
~102 MB of L2 traffic per matmul against 21 MB of weights. Staging moves the same bytes, and it
spends the shared memory that the actual fix needs. Removed.

## Attempt 60: more output rows per block — KEPT
Activation traffic scales as 1/rows. Raising rows from 2 to 4 (and dropping the staging that was
competing for shared memory) gives **61.08 -> 79.92 t/s, 188 -> 133.6 us**.

Past 4 rows it collapses -- rows=8 gives 56.09, rows=16 gives 17.83 -- because `tmp[ncols][rows]`
reaches 112 floats a thread and spills. **Registers, not shared memory, are the limit here**:
`cuobjdump -res-usage` reports REG:200, SHARED:3456, so occupancy is 10 warps/SM.

Capping registers with `__launch_bounds__` does not help: minblk=16 gives REG:128 with 104 bytes
of spill and 60.01 t/s; minblk=24 gives REG:80 with 656 bytes of spill. Reverted.

## Attempt 61: each warp owns its own rows — KEPT
To get more rows per block at constant register pressure, give each warp its own rows and have
every warp walk the whole of K, instead of the warps splitting K and sharing every row. Per-thread
accumulators stay at ncols x rows_per_warp, and the cross-warp reduction disappears.
| nwarps x rows (rows/warp) | pp512 | kernel |
|---------------------------|-------|--------|
| 4 x 2  (stock)            | 61.08 | 188.1 us |
| 1 x 4                     | 79.92 | 133.6 us |
| **2 x 8 (4)**             | **82.10** | **129.1 us** |
| 3 x 12 (4)                | 81.12 | 131.0 us |
| 2 x 12 (6)                | 77.18 | 140.0 us |
| 2 x 4 (2)                 | 69.68 | 159.8 us |

Also fixes a latent out-of-bounds: a block covers rows_per_cuda_block rows whether the tensor has
that many left or not, and the surplus rows were only dropped at write-back, so the staging read
off the end of the weights. Two rows got away with it; eight would not. The row used for addressing
is now clamped (free: 82.13 vs 82.10).

## Result
| metric | before | after |
|---|---|---|
| pp512 (b=7 ub=7) | 61.08 | **82.12** (+34%) |
| `mul_mat_vec_q<ncols=7>` | 188.1 us | **129.2 us** |
| **MTP decode** | **32.94 t/s** | **37.72-37.94 t/s** (+15%) |
| MTP speedup over plain decode | 1.11x | **1.27x** |
| tg256 (one-column path, untouched) | 29.9 | 29.5-29.9 |

Verification: test-backend-ops MUL_MAT 1193/1193; full perplexity gate 2.7554 +/- 0.02151.

**Caveat on that gate:** the standard perplexity run uses batch 512, which routes through
cuBLAS/MMQ and never touches this kernel. Gating the path that actually changed needs `-b 7 -ub 7`:
**3.6199 +/- 0.08383 optimised against 3.6237 +/- 0.08411 for the stock geometry**, same command --
a 0.1% shift, well inside the error bar. This path is therefore *not* bit-identical to stock (the
reduction order changed, which is also why acceptance moved 88.26% -> 83.82% on a fixed seed); the
one-column decode path still is.

## Remaining headroom
The no-activation probe at the final geometry is 53.7 us against 129.2, so the multi-column kernel
is still ~2.4x off its weight-streaming floor. At 8 rows the activation is ~26 MB a matmul against
21 MB of weights, so the two are now comparable and further row growth is blocked by registers.
Breaking that wall needs the accumulator count per thread reduced, which means restructuring the
dot product rather than tuning geometry.

## Attempt 62: row-outer loop nesting — REVERTED
Every column re-derives the same unpacked weights inside vec_dot, so swapping the loops to
row-outer/column-inner looked like it would let the compiler hoist that work. It does not:
REG rises 200 -> 227 and pp512 falls 82.12 -> 78.80. The compiler carries more per-column state
instead. Reverted. Hoisting it for real needs a vec_dot that takes pre-unpacked weights, which is
an interface change across every quant type.

## Attempt 63: scale warps and rows together, keeping 4 rows per warp — KEPT
The earlier sweep varied rows at fixed nwarps and so kept changing *rows per warp*, which is what
sets register pressure. Holding rows_per_warp at 4 (REG:182) and scaling nwarps and rows together
lets a block cover far more rows, and the activation traffic keeps falling as 1/rows:
| nwarps x rows (rows/warp) | pp512 |
|---------------------------|-------|
| 4 x 2 (stock)             | 61.08 |
| 2 x 8  (4)                | 82.10 |
| **4 x 16 (4)**            | **88.87** |
| 8 x 32 (4)                | 89.20 |
| 6 x 24 (4)                | 79.80 |
| 2 x 10 (5)                | 81.52 |
8x32 is marginally faster but a block then needs 32 rows to be worth launching; 4x16 is within
noise of it and degrades better on models with narrower matrices, so 4x16 is kept.

## Multi-column result
| metric | stock | now |
|---|---|---|
| pp512 (b=7 ub=7) | 61.08 | **88.87 (+45%)** |
| `mul_mat_vec_q<ncols=7>` | 188.1 us | ~118 us |
| **MTP decode** | **32.94 t/s** | **38.69 t/s (+17%)** |
| MTP speedup over plain decode | 1.11x | **1.30x** |
| tg256 (one-column path) | 29.9 | 29.83 |

test-backend-ops MUL_MAT 1193/1193. Perplexity through the changed path (-b 7 -ub 7, 4 chunks):
3.6199 +/- 0.08383, against 3.6237 +/- 0.08411 for the stock geometry on the identical command.

# MTP flag tuning

`--spec-draft-n-max 6 --spec-draft-p-min 0.75` was tuned against a verification step that this
session made 45% faster, so the optimum moved. Verification cost scales with the column count and
acceptance falls as the draft lengthens, so the balance now favours **shorter, more aggressive**
drafts. All runs: 256 tokens, temp 0, top-k 1, seed 42, cards cold.

Draft length at p-min 0.75:
| n-max | t/s | accept |
|-------|-----|--------|
| 3 | 39.05 | 95.3% |
| 4 | 40.21 | 93.2% |
| 5 | 37.47 | 87.9% |
| 6 (old default) | 38.15 | 83.8% |
| 7 | 36.44 | 75.7% |

Probability floor at n-max 4 -- this is the big one, worth more than the draft length:
| p-min | t/s | accept |
|-------|-----|--------|
| 0.95 | 34.72 | 96.4% |
| 0.85 | 38.36 | 94.9% |
| 0.75 (old default) | 40.21 | 93.2% |
| 0.6  | 42.02 | 90.3% |
| 0.4  | 48.09 | 83.2% |
| 0.2  | 48.21 | 78.2% |
| 0.05 | 48.42 | 78.2% |

A high p-min stops drafting early, so most rounds verify only one or two tokens and the batched
forward is wasted. Dropping it lets the head draft its full budget; acceptance falls but tokens per
round rise much faster. Best overall: **n-max 3, p-min 0.05 -> 48.90 t/s** (n-max 2 gives 44.5, so
3 is the knee).

## Speed is content-dependent -- quote a range, not a number
| prompt | tuned (3 / 0.05) | old (6 / 0.75) | accept (tuned) |
|--------|------------------|----------------|----------------|
| C++ quicksort (the standard benchmark) | **48.84** | 38.15 | 87.9% |
| prose (Roman Empire) | 37.68 | 22.05 | 58.3% |
| explanation (why the sky is blue) | 40.36 | 26.86 | 65.6% |

The flags help everywhere -- +71% on prose, +50% on the explanation -- but only predictable
content clears 45 t/s. Low-acceptance content prefers a shorter draft still: prose peaks at
n-max 2 / p-min 0.05 = 39.38. **n-max 3 / p-min 0.05 is the best single setting**; use n-max 2 if
the workload is mostly prose.

Re-tuning the kernel geometry for the now-dominant 4-column kernel found nothing: 4 rows per warp
is optimal there too (70.48 pp512 at 4x16, against 65.51 at 4x24 and 59.02 at 4x32 as registers
climb 168 -> 214 -> 255). The committed 4x16 geometry stands for every column count.

# Session summary
| workload | start | end |
|----------|-------|-----|
| plain decode (tg256), cold | 17.51 | 29.9 |
| plain decode, sustained (thermally limited) | -- | 23-25 |
| **MTP decode, code** | 32.94 (11% over plain) | **48.84 (63% over plain)** |
| MTP decode, prose | 22.05 | 37.68 |
| pp512 at b=7 ub=7 | 61.08 | 88.87 |

# Toward 60 t/s

## Attempt 64: use MMQ instead of mvq for the multi-column path — REVERTED (decisive)
`ggml_cuda_should_use_mmvq` tunes the mvq->MMQ crossover per architecture ("tuned on RTX 4090",
"tuned for CDNA2", ...) and its own comment states the problem found above: *"k-quants cost more to
decode and mvq redoes that per column, so MMQ wins sooner."* MMQ decodes once into shared and
reuses across a tile of columns -- exactly the structure the multi-column path wants. Pascal has no
entry and falls to the default, never using MMQ below 8 columns.

Adding a Pascal entry so ne11=4 routes to MMQ: **17.11 t/s against 70.57 for mvq -- 4x slower.**
MMQ's tiles are arithmetic-dense and assume real DP4A, which sm_60 lacks and this build emulates.
Upstream's default is correct for Pascal, and the unpack-once structure will not come for free
from MMQ; it would have to be written into mvq directly.

## Attempt 65: drop the weight staging on the multi-column path — REVERTED
With several columns each staged word is already reused once per column, so the staging looked like
it might be buying little while costing registers. It is still essential: 52.88 against 70.52, and
registers barely move (168 -> 163), so the staging is not where they are going.

## Attempt 66: trade rows per warp for warp count — REVERTED
Registers scale with rows_per_warp, so halving it should buy occupancy:
| nwarps x rows (rows/warp) | REG | pp512 (ub=4) |
|---------------------------|-----|--------------|
| 4 x 16 (4)                | 168 | **70.52** |
| 8 x 16 (2)                | 118 | 61.76 |
| 16 x 32 (2)               | 112 | 55.93 |
| 8 x 32 (4)                | 164 | 67.35 |
| 4 x 24 (6)                | 214 | 65.51 |
| 4 x 32 (8)                | 255 | 59.02 |
Occupancy is not the whole story: per-thread row reuse is worth more than the extra warps.
4 rows per warp is optimal at 4 columns as well as at 7, so the committed 4x16 stands.

## Where 60 t/s stands
Round budget at ~49 t/s (3.9 tokens per round, ~80 ms), per GPU:
| | ms | share |
|---|---|---|
| mul_mat_vec_q ncols=4 (target verify) | 47 | 55% |
| mul_mat_vec_q ncols=1 (3 draft steps) | 4.5 | 5% |
| all other kernels | ~10 | 12% |
| launch / sync gaps | ~24 | 28% |

The verify kernel moves ~29 MB in 88 us = **330 GB/s**, against the 483 GB/s the single-token path
reaches, and its no-activation floor is ~42% of its current time. So an optimistic bound is
28 + 4.5 + 10 + 12 = ~55 ms, or about **70 t/s** -- 60 is inside the envelope but needs *both* most
of the activation cost removed from the multi-column kernel *and* the launch overhead roughly
halved. Neither is a tuning knob:
1. An unpack-once multi-column dot product written into mvq (MMQ's version is 4x slower here). It
   would cut the redundant per-column decode and, more importantly, the registers that cap
   occupancy at 12 warps/SM.
2. Fewer launches. CUDA graphs measured no gain (the graph is re-captured almost every token), so
   this means op fusion.

## Note
One transient `1192/1193` on test-backend-ops MUL_MAT was observed on the committed tree, not
reproducible in three immediate re-runs (1193/1193 each). Probably a tolerance-borderline case or
contention from the desktop; recorded here in case it recurs.

## Attempt 67: PROBES — what the multi-column kernel actually spends its time on
At ncols=4 (pp512 ub=4 baseline 70.55, REG:168), each probe keeps the rest of the kernel intact:
| probe | pp512 | gain | REG |
|-------|-------|------|-----|
| activation replaced by a constant | **101.95** | **+44%** | **71** |
| dp4a deleted | 75.74 | +7.3% | 126 |
| weight unpack deleted | 74.56 | +5.7% | 140 |

Two conclusions. The activation is 44% of the kernel *and* its main register consumer -- removing
it takes REG from 168 to 71, which is why registers cap occupancy at 12 warps/SM. And the
unpack-once idea is dead: deleting the unpack **entirely** is worth 5.7%, so hoisting it out of the
column loop recovers at most three quarters of that, ~4% of the kernel and ~1.4% end to end. That
refactor (a vec_dot taking pre-unpacked weights, across every quant type) is not worth doing.

## Attempt 68: block-wide activation staging for the multi-column path — KEPT
With split_rows every warp walks the same K, so one staged copy serves the whole block: 4736 bytes
at ncols=4, and since occupancy here is capped by registers rather than shared memory it is free.
- pp512 (ub=4) 70.55 -> **71.44**, REG 168 -> 144
- **MTP 48.6-49.3 -> 50.10 / 50.15 t/s** (two runs)
- tg256 unchanged at 29.83; MUL_MAT 1193/1193
- PPL: 2.7554 +/- 0.02151 on the standard gate, 3.6199 +/- 0.08383 through the changed path
  (stock geometry gives 3.6237 +/- 0.08411 on that command)

Note this recovers only 1.3% of the activation's 44%. Staging fixes the access pattern, not the
byte count -- the third independent confirmation that past one column the activation cost is volume
and latency, not fan-out.

## Why 60 t/s is out of reach with this kernel structure
Round is 75 ms at ~50 t/s; 60 t/s needs 60.7 ms, so -14 ms. The verify kernel is 45 ms of it.
Everything measurable in that kernel has now been priced:
| component | worth at most |
|-----------|---------------|
| all arithmetic (dp4a + unpack) | ~13% of the kernel |
| activation access pattern (staging) | 1.3% (measured, taken) |
| activation *volume* | the remaining ~43%, and only rows-per-block reduces it |

Activation traffic is `nblocks x ncols x row_bytes`, so **only more rows per block reduces it** --
staging it earlier or differently moves the same bytes. Rows per block is capped by registers
(REG:168 at 4 rows/warp), and every way of lowering registers costs more than it returns:
rows/warp 4->2 takes REG to 118 but pp512 to 61.76; __launch_bounds__ capping spills.

Granting *all* the arithmetic for free -- which no real change achieves -- the kernel goes 88 -> 76
us, the round 75 -> 68 ms, and MTP to about 53.5 t/s. So **~53-55 t/s is the ceiling for this
structure**, and 60 needs a different one: a kernel whose activation cost does not scale with the
block count, which means many more rows per block, which means an accumulator layout that does not
put ncols x rows floats in registers. That is a redesign, not a tuning knob.

---

## Attempt 69 — independent numerical audit + two real defects fixed

Three adversarial auditors were tasked with *falsifying* the bit-exactness
claim, one per file group. Result: the claim was false, and three genuine
defects surfaced.

**Refuted (all "fewer or differently-grouped roundings", never worse):**
- `calc_nwarps` 4->2 at ncols_dst==1 halves the K-loop stride, so each thread
  accumulates a different subset of K-blocks. 2 partial trees instead of 4.
- `VDR_Q6_K_Q8_1_MMVQ` 1->4 folds four separately-rounded float lanes into one
  exact int accumulator (|acc| <= 262144 < 2^24). Strictly *fewer* roundings.
- The flash-attn tile fix changes `parallel_blocks`, hence the KV partition and
  the online-softmax combination. Differs in 21.5% of D=64 and 36.6% of D=256
  configs.

**Confirmed bit-exact (exhaustive machine proof, not sampling):**
- dp4a PRMT+XMAD emulation: 22,466,048 cases, 0 mismatches.
- q6_K/q3_K `__vsubss4` removal: exhaustive per-byte; saturation provably
  unreachable (operands in [-32,31] and [-4,3], never near +/-127).
- `rms_norm` register path: strided ownership preserved, zero-padding appended
  after real terms, `tmp` never -0.0 so the added +0.0 is a bit-exact identity.
- `binbcast` fast path: 384 predicate-satisfying shapes, 0 divergences.
- q8_1 activation cache: 8 stale-read vectors enumerated, all closed.

**Magnitude (the question that actually matters).** Layer-0 relative RMS error
is 9.8e-08 -- fp32 machine epsilon is 1.19e-07, i.e. one rounding. Growth is
smooth and geometric (~1.09x/layer) to 4.4e-02 at layer 63, with no
discontinuity: chaotic amplification of rounding noise, not a defect. At the
output: KL 1.97e-03 nats, argmax and full top-10 identical. Control: switching
the KV cache q4_0 <-> f16 perturbs the model 2.6x *more* (KL 5.14e-03).

**Whole-graph diff.** 2966/3847 tensors differ, first divergence at `node_13`
(layer-0 QKV projection); the 881 that match are exactly those never routed
through `mul_mat_vec_q`. Harness in `p100-handoff/tools/`.

**Fixed and committed:**
- `2c0d39158` MoE OOB *write*: row guards used `stride_col_dst` (== ne0*ne1 for
  MUL_MAT_ID) instead of `nrows_x`. Upstream immune at 1 row/block; reachable
  here at 2. Odd `nrows_x` wrote into the next expert's dst slot.
- `7d004be91` fastdiv guards were off by 2x (2^32 vs the true 2^31 domain);
  added int64 fallbacks rather than aborting where upstream worked.

Both: MUL_MAT and MUL_MAT_ID 3/3 backends, PPL 2.6209 +/- 0.01994, 29.81 t/s.

## Attempt 70 — the perplexity gate had silently decalibrated

`CLAUDE.md` requires PPL 2.6209 +/- 0.0199. Every build read 2.7554. Cause was
neither this work nor the prior session's: the corpus recipe

    cat README.md docs/*.md docs/**/*.md | head -c 800000 > /tmp/ppl.txt

reads whatever the docs say *that day*. The Aug 24 upstream pull moved the docs
from 420,098 to 422,246 bytes, so the gate decalibrated the moment the repo was
updated, and `/tmp/ppl.txt` was later cleared.

Proof no code regressed -- same corpus, three builds, identical every chunk:
upstream `f280b2698`, prior `b44f8fe6f`, and this work all 2.7554 +/- 0.02151.

The Aug-18 corpus was reconstructed from git (`p100-handoff/ppl-orig.txt`,
420,098 bytes) and reproduces the reference exactly: all 30 chunks identical,
`[1]4.9923 ... [30]2.6209`, final 2.6209 +/- 0.01994.

**Rule: pin the corpus, never regenerate it.** A perplexity gate defined as a
shell command instead of a fixed file will drift out from under you silently.
See `p100-handoff/CORPUS.md`.

## 71 — vectorised q6_K dequant (KEPT)

`dequantize_block_q6_K` gave each of its 64 threads four outputs 32 apart:
7 single-byte loads and 4 scalar stores per thread. Reassigned each thread four
*consecutive* outputs instead — they share `ip`, `j` and the scale, so the quant
reads become 16-bit loads (block_q6_K is 210 bytes, only 2-byte aligned, so not
32-bit) and the store becomes one vector write. 11 memory instructions -> 6, and
a warp now stores 128 contiguous elements.

Bit-exactness: only the thread->output assignment changes; every output is an
independent expression with no reduction whose grouping could shift. Verified by
replaying both index mappings on 4096 random superblocks — 1,048,576 elements,
**0 bit mismatches, 0 unwritten**.

pp2048 372.5 -> **375.19 +/- 0.58**. +0.7%. Kept.

## 72 — concurrent bidirectional peer copies (KEPT, +12.2%)

nvprof gap analysis: 86% of all GPU idle time was a **4358 us gap immediately
after every PtoP copy**, 156 occurrences — exactly one copy duration.

Cause: the tensor-parallel all-reduce (`push_data` in ggml-backend-meta.cpp)
exchanges partials in both directions. Copy 0->1 was issued on GPU0's *compute*
stream and GPU1's compute stream was then made to wait on it — so copy 1->0,
issued on GPU1's compute stream, could not start until copy 0->1 had finished.
The two directions serialised.

Measured separately: PCIe here is full duplex — 9.74 GB/s *each way
simultaneously*, 19.5 GB/s aggregate (uni 10.24 GB/s). So the second copy was
free and we were paying full price for it.

Fix (ggml-cuda.cu, common.cuh): peer copies go on a dedicated per-context
`copy_stream`. The copy stream waits on `work_event` — a marker recorded at the
end of every graph compute / set_tensor_async — rather than on the compute
stream itself, so a wait installed there by the *other* direction cannot push
this device's copy behind it. The src compute stream then waits on the copy
event, preserving the write-after-read guarantee that was implicit when the copy
lived on the compute stream.

pp2048 375.19 -> **421.04 +/- 0.33**. **+12.2%.**
Perplexity **2.6209 +/- 0.01994** on ppl-orig.txt — exact match, every chunk
identical including [1] 4.9923. Pure scheduling change, no arithmetic touched.

Also measured and rejected (free flag sweep, pp4096): -ub 2048 380.81,
-ub 3072 343.78, -ub 4096 377.65. 2048 remains the sweet spot.
Raw cuBLAS hgemm at real prefill shapes: 14.48-15.21 TFLOPS (76-80% of the
19.05 fp16 peak) — the GEMM itself has little left. The n=512 test case reads
7.15 TFLOPS only because n=512 and n=1024 take *identical* time (wave
quantisation), which is also why -ub 512 was so slow.

## 73 — GDN block width sweep (REVERTED)

`gated_delta_net` uses num_warps=4, so a 128-wide state is covered by 32 blocks
in z, every one of which re-reads the whole 128-element k and q vector for the
token. Made num_warps a compile-time knob (bit-exact -- columns are independent)
and swept it. 4 is already optimal:

| num_warps | pp2048 |
|---|---|
| 2 | 417.37 |
| **4 (default)** | **421.04** |
| 8 | 417.05 |
| 16 | 409.18 |

Kept the knob (it is now `P100_GDN_NWARPS`, defaulting to 4 = upstream
behaviour) but no change in value. The kernel is bound by its dependent
critical path -- two serial warp reductions per token over 2048 tokens -- not
by load redundancy or occupancy (48 regs, ~40 warps/SM).

## 74 — cuBLAS ALGO3 for wide f16 GEMMs on pre-Volta (KEPT, +1.8%)

cuBLAS's default kernel choice for tall-and-skinny TN f16 GEMMs is not its
fastest on sm_60. Standalone sweep at n=2048, DEFAULT_TENSOR_OP -> ALGO3:

| shape | default | ALGO3 | delta |
|---|---|---|---|
| ffn gate/up m=8704 k=5120 | 15.31 | 16.79 | +9.7% |
| ffn down m=5120 k=8704 | 15.99 | 16.54 | +3.5% |
| attn qkv m=4096 k=5120 | 14.09 | 15.50 | +10.0% |
| attn out m=5120 k=3072 | 15.96 | 16.41 | +2.8% |
| gdn in m=8240 k=5120 | 15.24 | 15.91 | +4.4% |
| gdn misc m=2560 k=5120 | 11.83 | 13.98 | +18.2% |
| lm_head m=124160 k=5120 | 10.59 | 10.58 | -0.1% |

The advantage inverts as n shrinks -- at n=64 ALGO3 is up to 2x *slower* -- so
it is gated on ne11 >= 512, and on cc < VOLTA since these legacy algo selectors
only mean anything on the pre-Volta path. Falls back to the default if cuBLAS
rejects the algo for a shape.

pp2048 421.04 -> **428.69 +/- 2.04**. +1.8%.

**NOT bit-exact** -- unlike 71/72 this is a different kernel, so the f16
k-accumulation order changes. Perplexity **2.6214 +/- 0.01995** vs the 2.6209
reference: +0.0005, i.e. 0.03 sigma, well inside the CLAUDE.md band
(2.6010-2.6408). Per-chunk movement is mixed in direction (chunk [1] 4.9923 ->
4.9738, i.e. lower). Reverting is a one-line `if`.

Also measured and rejected:
- NN weight layout would give a similar gain (16.95/16.63 TFLOPS) but needs a
  transposing dequant; ALGO3 gets the same for one line.
- lda padding: +5% on gate/up only, ~0 on down.
- chunking the GEMM along m: strictly worse (15.25 -> 14.96 -> 12.89).
- f32 GEMM output (would remove the f16->f32 convert): 7.65 TFLOPS, ~half
  speed. Dead.

## 75 — f16 tensor-parallel all-reduce (KEPT, +2.4%)

PtoP was 11.9% of prefill and already at hardware peak (attempt 72), so the only
remaining lever was sending fewer bytes.

On this backend every MUL_MAT output is *already* an f16 value widened to f32:
with cc < VOLTA the cuBLAS epilogue writes f16 into a pool buffer and a separate
`convert_unary` widens it (1984 converts per pass, exactly one per GEMM). So the
f32 partials the all-reduce ships hold only f16-representable values, and
narrowing them for transport is **exactly lossless** -- here.

That is a property of this path, not a general truth (mul_mat_vec_q produces
genuine f32), so it is not assumed. Guards:
- `src->op == GGML_OP_MUL_MAT && src->ne[1] >= 512` -- the condition under which
  the f16 cuBLAS path is taken; narrow batches keep f32.
- `cc < VOLTA && fast_fp16_available(cc)`.
- A **one-time runtime probe** on the first eligible exchange checks every
  element for f16-exactness and latches the result. The probe exchange itself
  still goes uncompressed, so a model whose partials are not f16-exact never
  sees a single lossy copy. It reports: "tensor-parallel partials are f16-exact;
  peer copies will be sent as f16".

Two bugs found and fixed on the way, both worth recording:
1. **One staging buffer per context is wrong.** In a butterfly all-reduce a
   device is sender and receiver in the same step, so GPU1's buffer was both the
   landing zone for copy 0->1 and the narrow output for copy 1->0. Each GPU
   ended up adding its own partial twice. Perplexity chunk [1] 4.97 -> 27.28.
   Fixed with separate OUT/IN buffers.
2. **record_work() after the widen re-serialised the directions.** It advanced
   the dst's work marker past a wait on the peer's copy, so the other direction's
   copy stream queued behind it -- reinstating exactly the stall attempt 72
   removed. 396.63 -> 439.16 once dropped. A `peer_stage_free` event guards
   refill instead.

pp2048 428.69 -> **439.16 +/- 0.98**. +2.4%.

## 76 — pipelined gated_delta_net reduction (KEPT, +0.8%)

The token loop's critical path is load -> reduce -> update -> reduce, serial
over 2048 tokens, and a 5-step warp butterfly is ~150 cycles. But attn[col] for
token t and kv[col] for token t+1 both read only the state *after* token t, so
they are independent: token t+1's k is pulled forward and the two partials are
reduced together with `warp_reduce_sum(float2)`, which interleaves the two
shuffle chains and pays one chain's latency instead of two.

Bit-identical: the float2 overload applies the same per-component offsets in the
same order as the scalar one, and every partial is still accumulated over r in
the same order.

Bug found via `test-backend-ops -o GATED_DELTA_NET` (ERR 6.1e-4 vs 1e-7 tol,
which compounded to NaN over 2048 tokens): the first attempt stored the
*reduced* kv back into the accumulator and then reduced it again at the top of
the next iteration. The carried value is already reduced.

pp2048 439.16 -> **442.59 +/- 1.44**. +0.8%.

Perplexity for 75+76 together: **2.6214 +/- 0.01995**, chunk [1] 4.9738 --
identical to every digit to the ALGO3 build, confirming both are lossless.

## 77 — per-shape cuBLAS algo (REJECTED)

Swept all 24 legacy algos at every real shape, n=2048. ALGO3 is already the best
on all the dominant shapes; only two minor ones prefer something else:

| shape | ALGO3 | best | |
|---|---|---|---|
| ffn gate/up | 16.80 | ALGO3 16.80 | — |
| ffn down | 16.55 | ALGO3 16.55 | — |
| attn out | 16.41 | ALGO3 16.41 | — |
| gdn in | 15.90 | ALGO3 15.90 | — |
| attn qkv | 15.02 | ALGO6 15.55 | +3.5% on a small share |
| gdn misc | 15.20 | ALGO5 15.53 | +2.2% on a small share |

Worth ~+0.3% overall for a per-shape lookup table. Not taken — the complexity
and the risk of picking wrong for an unseen shape outweigh it.

## 78 — GDN loads issued a full iteration ahead (REVERTED)

Attempt 76 prefetches token t+1's k but consumes it in the same iteration, so
the global latency is not actually hidden. Moved the load issue to the *top* of
the iteration (raw g, expf deferred to hand-over) so it flies during the state
update. Correct (test-backend-ops passes, no spill) but **slower**: registers
47 -> 53, which drops occupancy from 10 blocks/SM (40 warps) to 9 (36).

pp2048 442.59 -> 438.80. Reverted.

GDN is now ~7% of prefill and resists the obvious attacks: block width (73),
reduction fusion (76, +0.8%), deeper load pipelining (78, negative). Measured
issue efficiency is ~15% of peak, so it is stalled on something that is not the
warp-reduction critical path and not occupancy. Nsight Compute would say what;
it does not support Pascal.

## Decode: no regression, and a bonus

The peer-copy stream fix (72) helps decode too -- decode all-reduces are small
but latency-dominated. Measured with the CLAUDE.md metric command
(`-p 0 -n 256 -r 3`), at 54 C (not cold):

**tg256 = 31.71 +/- 0.14 t/s**, against the 29.8 cold / 29.2 warm baseline and
the 17.51 t/s original baseline in CLAUDE.md. **1.81x on the headline metric.**

The f16 all-reduce (75) and ALGO3 (74) both gate on ne11 >= 512, so decode takes
neither path -- its partials stay f32 and it keeps the default GEMM algo.

## Session summary — prefill 372.5 -> ~440 t/s (+18%)

| step | pp2048 |
|---|---|
| start of session | 372.5 |
| 71 vectorised q6_K dequant | 375.19 |
| 72 concurrent bidirectional peer copies | 421.04 |
| 74 cuBLAS ALGO3 | 428.69 |
| 75 f16 all-reduce | 439.16 |
| 76 pipelined delta-net reduction | **442.59** |

Cold readings land at 442-443, hot at ~438. Thermal drift of ~1% is real: the
same build measured 442.59 at 39 C and 438.49 at 55 C. Re-baseline from cold
before reading anything into a delta of that size.

Perplexity **2.6214 +/- 0.01995** (ppl-orig.txt, 420098 bytes) vs the 2.6209
reference -- inside the CLAUDE.md band by 0.03 sigma. Of the five kept changes
only ALGO3 (74) is not bit-exact; 71, 72, 75 and 76 were each verified to
reproduce the preceding build digit-for-digit.

Where the time goes now (per GPU, nvprof, at 442 t/s):

| item | share |
|---|---|
| maxwell_hgemm_256x128_tn | 71.0% |
| PtoP (was 11.9%) | 7.1% |
| gated_delta_net | 7.0% |
| flash_attn_tile | 2.1% |
| q6_K dequant | 1.9% |
| f32<->f16 converts | 3.0% |
| rms_norm | 1.9% |
| all-reduce ADD | 1.3% |
| idle | 2.2% |

GEMM-only ceiling is ~620 t/s. The GEMM runs at ~15.7 TFLOPS in-model against a
19.05 peak (82% at sustained clocks), so it has little left.

## 79 — MTP re-measured, and a better default (flag change)

MTP had not been measured since the peer-copy fix. Re-swept with
`p100-handoff/tools/mtp-bench.sh`:

| n-max | p-min | t/s | accept |
|---|---|---|---|
| 2 | 0.05 | 48.92 | 89.2% |
| 3 | 0.05 | 52.90 | 87.9% |
| **4** | **0.2** | **54.04** | 78.2% |
| 4 | 0.05 | 53.79 | 78.2% |
| 5 | 0.05 | 53.02 | 71.9% |
| 6 | 0.75 | 41.52 | 83.8% |

48.8 (previous best, n-max 3) -> **54.04** with `--spec-draft-n-max 4
--spec-draft-p-min 0.2`. +10.7%. Most of that is the peer-copy fix (72) --
decode all-reduces are small but latency-dominated, which is exactly what that
change addressed.

Flat-to-falling past n-max 4: the accept rate decays faster than the extra
speculated tokens pay for themselves. This is at the 53-55 t/s structural
ceiling for the current kernel shape, so the 60 t/s goal needs a shape change
(the draft head, or a batched-decode kernel that does not re-read weights per
speculated token), not more tuning.

## Final measurements (end of session)

Same build, same commands:

| metric | value | temp |
|---|---|---|
| pp2048 | 442.59 +/- 1.44 | 39 C start |
| pp2048 | 438.49 +/- 0.25 | 55 C start |
| pp2048 | 434.10 +/- 1.28 | 52 -> 66 C |
| tg256 (CLAUDE.md metric) | 31.79 +/- 0.16 | 51 C |
| MTP (n-max 4, p-min 0.2) | 54.48 (best of 6 readings) | warm |

**A 2% spread on prefill comes from temperature alone.** Compare only at equal
starting temperature; anything under ~2% is not a code delta.

## 80 — GDN addressing strength-reduced (KEPT, below noise on this model)

The token loop recomputed `q + iq3*sq3 + t*sq2 + iq1*sq1` and three more like it
every iteration -- twelve 64-bit multiplies per token. sm_60 has no native 64-bit
multiply, so each expands to an IMAD sequence: far more instructions than the 16
FMAs of actual work in the body. Tokens are visited strictly in order, so the
addresses are an arithmetic progression; the pointers are now walked instead.

Measured directly with `test-backend-ops perf -o GATED_DELTA_NET` (isolates the
kernel, so no thermal confound):

| shape | recomputed | walked | |
|---|---|---|---|
| head_count=32, head_size=128, 256 tok | 1125.30 us | 1113.44 us | -1.1% |
| head_count=4, head_size=128, 256 tok | 171.40 us | **121.26 us** | **-29.3%** |

The gain scales inversely with occupancy: at head_count=4 there are only ~2.3
blocks/SM so per-thread instruction count is exposed, while at head_count=32 the
addressing hides behind warp parallelism. This model runs 768 blocks/GPU, i.e.
the hidden regime, so **the model-level effect is below measurement noise**
(pp2048 438.44 vs 438.49, at different start temperatures).

Kept anyway: bit-identical (pure addressing), never slower in any measurement,
costs 5 registers (47 -> 52) without changing resident blocks, and is a large
win for small-head-count configurations. Recorded honestly as *not* a measurable
gain for the Qwen3.5 27B workload.

## 81 — CUDA graphs on Pascal (TESTED, no gain — upstream's exclusion is correct)

llama.cpp disables CUDA graphs for cc < VOLTA unconditionally. sm_60 hardware
supports them, and MTP decode launches a great many tiny kernels (rms_norm
6.8us x 20198 calls, quantize_q8_1 2.8us x 37076, cpy 2.3us x 34426,
bin_bcast 2.3us x 36328), so this looked like a large launch-overhead win.

Added a `GGML_CUDA_GRAPHS_PRE_VOLTA=1` opt-in and measured. Graphs **do** engage
("CUDA graph warmup complete", "CUDA Graph id reused"), and give nothing:

| | graphs off | graphs on |
|---|---|---|
| MTP (n-max 4, p-min 0.2) | 54.478 | 54.028 |
| tg256 | 32.12 | 31.25 |

The workload is not launch-bound -- it is streaming weights. `mul_mat_vec_q`
with ncols=5 is 52% of MTP GPU time at 95.7us per call (~383 GB/s of Q6_K
weights, i.e. ~75% of achievable HBM bandwidth), and the tiny kernels overlap
with that. Reverted; upstream's exclusion is justified for this workload.

## 82 — MMVQ rows-per-block for the MTP path (no effect)

`P100_MMVQ_ROWS_N` 16 -> 8 (doubling block count, since the down-projection at
m=5120 yields only 320 blocks over 56 SMs and looked under-subscribed):
MTP 54.040 -> 54.053. No effect -- the multi-column path is not
parallelism-limited. Reverted.

## 83 — `-sm layer` for decode (much worse)

tg256 **20.37** vs 32.12 with `-sm tensor`. Decode is HBM-bandwidth-bound per
GPU, so tensor split's two-way bandwidth is worth far more than the all-reduce
costs. Same conclusion as prefill (attempt in 78's table: 226.9 vs 438.5).
`-sm tensor` is correct for both phases.

## Goal status, honestly

Targets were 450 t/s prefill and 60 t/s MTP.

| | start | achieved | target | |
|---|---|---|---|---|
| prefill pp2048 | 372.5 | **442.6** | 450 | 98.4% |
| MTP | 48.8 | **54.5** | 60 | 90.8% |
| single-token tg256 | 29.8 | **32.1** | — | +7.7% |

Neither target met. What stands between here and them:

> **Superseded below.** Attempt 84 **retracts** the throttling claim in this
> paragraph, and attempt 86 replaces the MTP framing in the next one. Read
> 84/84b/86 instead; this section is kept only for the record.

**Prefill (+1.7% needed).** GEMM is 71% of wall at 15.7 TFLOPS in-model against
16.8 standalone; ~~the 6.6% gap is sustained-clock throttling~~ (**wrong -- see
84**; the gap is a uniform 3.2% and is not thermal) and nvidia-smi is
off limits. The one identified remaining item is fusing the all-reduce widen
into the ADD (~+0.8%), which needs an accumulating-copy path in
ggml-backend-meta.cpp so the ADD node can be dropped -- graph-construction
surgery in generic code, which is more risk than the instruction to avoid
"overly risky" changes allows this late. A second ~1.9% would come from
overlapping each weight dequant with the *previous* matmul's GEMM on a second
stream, which needs graph lookahead ggml does not currently expose.

**MTP (+10% needed).** 52% of the time is `mul_mat_vec_q<ncols=5>` already at
~75% of achievable HBM bandwidth. ~~Closing the gap means attacking the sm_60
dp4a emulation or changing the kernel shape.~~ **Wrong emphasis -- see 86:**
MTP is 24% idle with 14.5% of wall in the host round trip, so the route to 60
is the decode pipeline, not the kernel.

## Final gate on HEAD

`ed42ad15d` + docs: perplexity **2.6214 +/- 0.01995**, chunk [1] 4.9738 --
identical to the previous gate, so attempt 80 (GDN addressing) is confirmed
bit-exact end to end. Inside the CLAUDE.md band by 0.03 sigma.

## 84 — RETRACTION: the in-model GEMM gap is NOT thermal throttling

Earlier entries (and the first version of RESUME-HERE.md) attributed the gap
between the in-model GEMM and the same GEMM standalone to "sustained-clock
throttling". **That is wrong and is retracted.**

Sustained pure-GEMM run, 170 s, ALGO3, m=8704 n=2048 k=5120:

| elapsed | temp | clock | power | TFLOPS |
|---|---|---|---|---|
| 0 s | 41 C | 1328 MHz | 39 W | 16.79 |
| 50 s | 57 C | 1328 MHz | 137 W | 16.81 |
| 100 s | 65 C | 1328 MHz | 148 W | 16.81 |
| 170 s | 73 C | 1328 MHz | 154 W | 16.81 |

Clocks never leave 1328 MHz, past the 63-66 C the model reaches, and throughput
is flat to three digits. There is no throttling.

The real gap is also smaller than first reported. Duration *distribution* of the
in-model gate/up GEMM (grid=(34,16), i.e. m=8704 k=5120), 118 calls per device:

| | min | p25 | median | p75 | p90 | max | mean |
|---|---|---|---|---|---|---|---|
| dev 0 | 11.038 | 11.111 | 11.179 | 11.295 | 11.373 | 11.646 | 11.213 |
| dev 1 | 11.001 | 11.046 | 11.196 | 11.390 | 11.508 | 12.175 | 11.239 |

Tight, no outlier tail. Standalone is 10.864 ms. So the gap is a **uniform
3.2%**, worth ~2.3 points of prefill wall time -- not the 6.9% an earlier noisy
window suggested.

Hypotheses tested and **eliminated**, each with a standalone reproduction:

| hypothesis | result |
|---|---|
| clock/thermal throttling | 16.81 TFLOPS flat to 73 C, 1328 MHz |
| both GPUs loaded (shared envelope) | 10.863 ms each, simultaneous |
| nvprof inflates durations | nvprof 10.864 vs untimed 10.863 |
| preceding dequant write traffic | writer+gemm: gemm still 10.866 |
| pointer alignment (pool vs cudaMalloc) | only ~1%, and only below 16 B |
| cold weight buffer (TLB/L2) | rotating 6x89 MB buffers: 10.864 |
| VMM mapping vs cudaMalloc | 10.866 vs 10.868, granularity 2048 KB |

**The 3.2% remains unexplained.** Untested candidates: activation (B) locality,
cuBLAS handle/workspace state, or residual concurrency from the peer-copy
stream. Worth ~2.3 points if anyone cracks it -- do not dismiss it as thermal.

## 84b — the GEMM gap, resolved as far as it can be

Two more hypotheses eliminated:

| hypothesis | result |
|---|---|
| concurrency from the peer-copy stream stealing HBM | **0 of 256** gate/up GEMMs overlap with any other activity |
| cuBLAS handle state (ggml sets TF32_TENSOR_OP_MATH + a 4 MB workspace) | 10.863-10.865 ms in all four combinations |

Then the key observation, from a clean trace (442.74 t/s under nvprof, matching
the un-profiled number):

| device | n | min | median | mean | max |
|---|---|---|---|---|---|
| 0 | 256 | **10.869** | 11.072 (+1.9%) | 11.103 | 11.397 |
| 1 | 256 | **10.863** | 11.201 (+3.1%) | 11.232 | 12.475 |

**The minimum equals the standalone 10.864 ms exactly, on both devices.** The
kernel does reach full speed in-model; what differs is the *median*, with a
spread of 10.86-12.5 ms. So this is not a systematic property of the in-model
environment that could be removed -- it is call-to-call variation in memory/clock
state, and the nine software causes tested all came back negative.

Ceiling if every call ran at the observed minimum: ~2 points, i.e. ~450. But
there is no identified mechanism to make that happen, and it is not a code
defect. Closing this line of investigation.

Note GPU0 is consistently *faster* than GPU1 (median 11.07 vs 11.20) despite
GPU0 also hosting Sunshine's display allocation. Unexplained, not actionable.

## 85 — the redundant weight unpack in mmvq, priced (not captured)

`mul_mat_vec_q`'s inner nest is `for j in ncols_dst { for i in rows { tmp[j][i]
+= vec_dot(xs_i, ys_j) } }`. `xs` depends only on `i`, so for ncols_dst=5 the
weight block's load **and its 6-bit unpack are redone five times**. MTP runs at
ncols_dst=5, and that kernel is 52% of MTP GPU time -- so this looked like the
route to 60 t/s.

Priced it with the `P100_NOUNPACK` probe the previous session left in
vecdotq.cuh (drops the shift/mask unpack, keeping both loads and the dp4a;
results are wrong, timing only):

| | unpack present | unpack removed |
|---|---|---|
| `mul_mat_vec_q<ncols=5>` per call | 95.717 us | **87.629 us** (-8.5%) |
| tg256 (ncols=1) | 32.12 | **33.79** (+5.2%) |

So the unpack is 8.5% of the ncols=5 kernel. Hoisting it (once instead of five
times) recovers 4/5 of that, ~6.8% of the kernel = **~3.5% of MTP** -> ~56.4 t/s.
Real, but **not enough for the 60 t/s target**, and nothing for single-token
decode where there is only one column and so no redundancy to remove.

Tried to get it for free by swapping the loop nest to `for i { for j { ... } }`,
making the weight work loop-invariant in j. **No gain** (MTP 53.86 vs 54.48,
tg256 31.77 vs 32.12; correctness held at chunk [1] 4.9738). Both loops carry
`#pragma unroll` over compile-time bounds, so nvcc fully unrolls them and the
order is irrelevant to CSE -- it simply is not hoisting the unpack in either
form. Reverted.

Capturing it needs a hand-written multi-column `vec_dot_q6_K_q8_1` that unpacks
once and runs ncols dp4a chains, plus a dispatch for it in mmvq. That is a
rewrite of the hottest kernel in the build for ~3.5% on one metric, so it was
not attempted unattended. It is bit-exact by construction (per-column
accumulation order is unchanged) if anyone picks it up.

**All three probe switches in vecdotq.cuh were returned to 0** and correctness
re-verified (chunk [1] 4.9738, [2] 3.9640) after this measurement.

## 86 — MTP is host-sync bound, not kernel bound (the actual route to 60)

Measured the idle fraction of MTP decode, which had never been done (this is the
same measurement that produced the +12.2% prefill win in attempt 72).

Steady-state decode (last 25% of the timeline, device 1), `--spec-draft-n-max 4`:

**wall 1.603s, busy 76.0%, IDLE 24.0%**

| gap follows | share of wall | n | avg |
|---|---|---|---|
| `[CUDA memcpy DtoH]` | **8.7%** | 95 | 1460.7 us |
| `[CUDA memcpy HtoD]` | **5.8%** | 817 | 112.9 us |
| `rms_norm_f32<1024>` | 4.3% | 1262 | 54.8 us |
| `[CUDA memcpy PtoP]` | 1.3% | 2624 | 8.0 us |
| bin_bcast / mmvq / quantize | ~2% | many | 2.6-28.6 us |

**14.5% of wall is the host round trip** -- logits copied to the CPU, the
speculative accept/reject decided there, tokens copied back. At ~4.3 DtoH per
pass with 1.46 ms of GPU idle after each, the GPU spends a seventh of decode
waiting on the host. Removing it entirely would give 54.5/(1-0.145) = **63.7
t/s, past the 60 target**.

So the route to 60 is **not** kernel optimisation. It is the decode pipeline.
The two ways to get it:

1. GPU-side sampling. llama.cpp has it, and it is explicitly disabled for our
   split mode -- `llama-context.cpp`: "backend sampling not supported with
   SPLIT_MODE_TENSOR; using CPU". Same root cause as the meta backend being
   unable to service eval callbacks. Enabling it means teaching the meta backend
   to run the sampling graph.
2. Overlapping host verification with GPU work in the speculative loop
   (application-level restructuring of llama-speculative-simple / the server).

Both are backend/application architecture, not CUDA kernels, and well outside
what should be attempted unattended.

**Caveat, stated because it matters:** nvprof itself costs MTP ~11% (48.65 t/s
profiled vs 54.5 unprofiled), and host-sync gaps are precisely where profiler
overhead lands. Some fraction of the 14.5% is therefore artifact, and the true
headroom is likely smaller -- call it 5-10% rather than 14.5%. It should be
re-measured with CUDA events inside the decode loop rather than under a
profiler before anyone builds on this number.

This supersedes attempt 85's framing: the unpack redundancy (~3.5%) is real but
it is the *second* item, not the first. Fix the host stalls first.

## 87 — remaining MTP flags swept (nothing left in flag space)

`--spec-draft-backend-sampling` is **inert under `-sm tensor`**: 54.12 vs 53.59
t/s, and the "backend sampling not supported with SPLIT_MODE_TENSOR" warning
fires either way. This is the flag that would have addressed the 14.5% host
round trip from attempt 86, and it is gated off for our split mode -- confirming
that fix requires backend work, not configuration.

`--spec-draft-n-min` (never previously swept, default 0): no effect.

| n-max | p-min | n-min | t/s | accept |
|---|---|---|---|---|
| 4 | 0.2 | 0 | 54.48 | 78.2% |
| 4 | 0.2 | 1 | 54.27 | 78.2% |
| 4 | 0.2 | 4 | 54.38 | 78.2% |
| 5 | 0.2 | 2 | 54.00 | 71.9% |
| 6 | 0.2 | 3 | 50.32 | 67.8% |

**The MTP flag space is now exhausted** (n-max, p-min, n-min, backend-sampling,
split mode, cache types). 54.5 t/s stands, and the remaining 10% to the 60
target is the host-sync work in attempt 86 plus the kernel work in 85 -- neither
of which is reachable by configuration.


---

# CLOSING SUMMARY (2026-09-01) — supersedes every earlier summary in this file

## Result

| metric | session start | final | target | |
|---|---|---|---|---|
| prefill pp2048 | 372.5 | **442.6** cold / ~437 hot | 450 | 98.4%, **not met** |
| MTP (n-max 4, p-min 0.2) | 48.8 | **54.5** | 60 | 90.8%, **not met** |
| single-token tg256 | 29.8 | **32.1** best / ~31.8 typical | — | +7.7% |
| perplexity (ppl-orig.txt) | 2.6209 | **2.6214 +/- 0.01995** | +/-0.0199 | passes at 0.03 sigma |

vs the 17.51 t/s decode baseline in CLAUDE.md: **1.83x**.

## The six code changes

`5d1fafb01..f85e154ed`, 417 insertions / 53 deletions, all inside
`ggml/src/ggml-cuda/` (`ggml-cuda.cu`, `common.cuh`, `convert.cu`,
`gated_delta_net.cu`). Nothing outside that directory was touched.

| commit | change | prefill gain | bit-exact |
|---|---|---|---|
| `5d1fafb01` | vectorised f32<->f16 convert | +5.3% (prior session) | yes |
| `58c8a73ed` | vectorised q6_K dequant | +0.7% | yes, machine-proven |
| `a4d1103c5` | concurrent bidirectional peer copies | **+12.2%** | yes (scheduling only) |
| `f8edbf816` | cuBLAS ALGO3 for wide f16 GEMMs | +1.8% | **no** (0.03 sigma) |
| `e83a7913a` | f16 all-reduce + pipelined delta-net reduction | +3.2% | yes |
| `ed42ad15d` | delta-net addressing walked | below noise here | yes |

## Four corrections I made to my own claims

These matter more than the last few percent, because each would have sent the
next session down a wrong path:

1. **The all-reduce was serialising both directions of a full-duplex PCIe
   link.** Not a new optimisation so much as a bug: 4358 us of idle after every
   peer copy, 86% of all idle time. Worth +12.2%.
2. **The in-model GEMM gap is not thermal throttling** (attempt 84). Clocks hold
   1328 MHz to 73 C. Nine causes eliminated; the in-model *minimum* equals
   standalone exactly, so it is call-to-call variation, not a defect.
3. **MTP is not kernel-bound** (attempt 86). It is 24% idle with 14.5% of wall
   in the host round trip. This is why 60 t/s is not reachable by tuning kernels.
4. **But the host round trip is not worth attacking either** (attempt 88).
   I predicted default sampling would be far more expensive than the greedy
   benchmark config and that GPU-side sampling would therefore pay 3-5x more
   than measured. **Wrong — defaults measure within noise of greedy.** Priced
   properly the lever is 4-8%, tops out at 57-59, and costs high-risk surgery in
   `ggml-backend-meta.cpp`. Dropped. Correction 3 said 60 was "reachable in
   principle" via this route; it is not.

## What is actually left, ranked

| # | item | worth | why not done |
|---|---|---|---|
| ~~1~~ | ~~MTP host-sync: GPU-side sampling~~ | **DROPPED — see attempt 88** | priced properly at 4-8% (-> 57-59, *not* 60); needs axis-1 split rules for ARGMAX/TOP_K/SOFT_MAX/GET_ROWS in `ggml-backend-meta.cpp`, silent-wrong-answer tier. Mirroring `output.weight` instead is a net loss (doubles output-head traffic, +497 MiB/GPU). |
| 2 | overlap each weight dequant with the previous GEMM | +1.9% prefill -> ~451 | needs graph lookahead ggml does not expose |
| 3 | multi-column `vec_dot_q6_K_q8_1` (unpack once, not per column) | +3.5% MTP -> ~56.4 | rewrite of the hottest kernel; bit-exact by construction |
| 4 | fuse the all-reduce widen into the ADD | +0.8% prefill | needs an accumulating-copy path in `ggml-backend-meta.cpp` |

Items 2 and 4 together would clear 450. ~~No combination of 1-4 reaches 60 MTP
except item 1, and item 1 alone would.~~ **Retracted by attempt 88: item 1 tops
out at 57-59. Nothing on this list reaches 60 MTP.** The 60 target needs a shape
change — the draft head itself, or a decode kernel that does not re-read the
weights per speculated token — not any item here.

**Flash attention is unranked here because it is context-dependent.** At the
2048-token bench shape it is 2.1% of prefill / 2.4% of decode and not worth
touching. But only ~16 of 65 blocks carry a growing KV cache
(`full_attention_interval 4`); the rest are gated delta net with constant state.
So FA is the *only* cost on this model that scales with context length, and at
the 262144 this model supports it should dominate. **Nobody has profiled this
build at long context.** That is the open measurement.

## Measured dead ends — do not re-litigate

MMQ on Pascal; f32 GEMM output (7.65 vs 16.8 TFLOPS); NN weight layout (real,
but ALGO3 gets the same for one line); lda padding; chunking the GEMM along m;
`-sm layer` for prefill (226.9) **and** decode (20.4); reduce-scatter/all-gather
(identical traffic at 2 GPUs); CUDA graphs on Pascal (they engage, they give
nothing); MMVQ rows-per-block; GDN block width, deeper load pipelining;
per-shape cuBLAS algo (+0.3%); `--spec-draft-n-min`;
`--spec-draft-backend-sampling` under `-sm tensor`; **implementing** backend
sampling under `-sm tensor` (attempt 88: 4-8%, tops out at 57-59, high risk);
**mirroring `output.weight`** to enable it (attempt 88: doubles output-head
traffic and costs +497 MiB/GPU — net loss).

**Default sampling costs nothing** (attempt 88): `top_k 40 / top_p 0.95 /
min_p 0.05 / temp 0.8` measures within noise of `--temp 0 --top-k 1`, so every
greedy-measured MTP number in this file carries over to real-world use.

## Hygiene

- All three `P100_*` probe switches in `vecdotq.cuh` are at **0**; one was used
  for the attempt-85 measurement and correctness was re-verified after.
- Gate corpus: use `p100-handoff/ppl-orig.txt` (420098 bytes, target 2.6209).
  `./ppl.txt` is a different document (422246 bytes, target 2.7554).
- Prefill readings carry a **2% thermal spread**. Compare only at equal starting
  temperature. The GEMM kernel itself does not throttle -- this is model-level.

---

## 88 — the MTP host round trip, priced properly (backend sampling is NOT worth it)

**Supersedes attempt 86's framing and the closing summary's ranked item 1.**
Item 1 claimed GPU-side sampling was the one lever that alone reaches 60 MTP.
Measured properly, it is not.

### What was actually measured

Host-side sampling cost at this model's real vocab (**248320**), standalone C/C++
microbenchmarks, per logits row:

| operation | cost/row | distribution-dependent? |
|---|---|---|
| logits DtoH (970 KiB) | ~100 us | no |
| greedy argmax | **372 us** | no — linear scan |
| candidate-array build (2.9 MiB fill) | **500 us** | no — linear fill |
| build + bucketed top-k(40), llama.cpp's real algorithm | ~2.2 ms | yes, but only weakly (2.26 uniform vs 2.19 peaked) |

That predicted default sampling (`top_k 40, top_p 0.95, min_p 0.05, temp 0.8` —
confirmed in `common/common.h`) would cost ~11 ms of a ~60 ms MTP step, ~20% of
wall, versus ~2.6-5.1 ms for the greedy benchmark config.

### The prediction was wrong

End-to-end, `llama-speculative-simple`, n-max 4, p-min 0.2, seed 42, n=256,
two runs each:

| sampling | t/s | drafted | accept |
|---|---|---|---|
| `--temp 0 --top-k 1` (greedy, what every prior MTP number used) | 51.82, 51.76 | 252 | 197 |
| **defaults** (top_k 40, top_p 0.95, min_p 0.05, temp 0.8) | **52.07, 52.32** | 240 | 198 |

Defaults are **within noise of greedy, marginally faster**. The 2.2 ms/row
isolated cost does not appear in wall time. So the host-sampling term is not a
bottleneck, and every MTP number in this file — all measured with greedy —
transfers to real-world default sampling unchanged. That last part is the useful
half of this result.

At n=512 the same comparison gave greedy 46.07 / defaults 51.37, i.e. the gap
runs the *other* way too; the accept rate is stochastic across configs and
dominates any host-side term.

### Consequence

Backend sampling under `-sm tensor` is worth at most the greedy-case
**4-8%** (2.6-5.1 ms of a ~60 ms step) -> **57-59 t/s. It does not reach 60.**
Against that: it needs new split rules in `ggml-backend-meta.cpp` for `ARGMAX` /
`TOP_K` / `ARGSORT` / `SOFT_MAX` / `GET_ROWS` on axis-1 tensors, which is the
silently-wrong-answer risk tier. **Not worth doing. Dropped.**

### Why the gate exists (recorded so nobody re-derives it)

`output.weight` is `GGML_BACKEND_SPLIT_AXIS_1` under `-sm tensor`
(`llama-model.cpp:566`), i.e. **vocab-sharded**, so neither GPU holds complete
logits. The backend samplers do `ggml_reshape_1d(logits)` then `ggml_argmax` /
`ggml_top_k` over the whole vocab (`llama-sampler.cpp:1084`, `:1484`), so each
GPU would reduce over its own shard and return a local index. The gate at
`llama-context.cpp:1216` is **load-bearing, not conservative**.

**Mirroring `output.weight` (as dsv4 already does) is a trap.** It would make the
sampler work unmodified, but `output.weight` is 5120x248320 Q6_K = **995 MiB**;
each GPU reads its 497 MiB half today, and mirroring makes both read the full
995 MiB. That doubles output-head memory traffic — ~+1.3 ms per decode step at
the ~383 GB/s this card achieves — to save 4-8%, plus **+497 MiB per GPU** of
VRAM. Net loss. Do not do it.

### Note on absolute numbers

These runs read 51.8-52.3 where attempt 87 recorded 54.48 for the same flags.
Two stale `nvidia-smi` polling loops from attempt 84 were still running (23 h
elapsed, 5 s interval) during these measurements, plus the documented ~2%
thermal spread. The greedy-vs-defaults *comparison* is unaffected — both arms ran
under identical conditions, back to back, and each reproduced to within 0.5%.

---

## 89 — long-context flash attention: the real prefill bottleneck, and 15 failed attempts at it

**This is the most important measurement in the file for anyone who runs long
context.** Every prior number in this project was taken at 2048 tokens, where
flash-attn is 2.1% of prefill. That is not the regime the model is used in.

### The finding

`llama-bench -d <depth>`, prefill:

| depth | pp2048 | note |
|---|---|---|
| 0 | 434.21 | what every earlier measurement in this file used |
| 32768 | 278.43 | **-36%** |
| 65536 | 179.80 | **-59%** |

Per-2048-batch time goes 4.717 s -> 7.356 s from d=0 to d=32768, i.e. 32k of
prefix costs 2.64 s/batch. Linear (attention against a prefix is O(batch x depth)),
so d=262144 projects to ~21 s of attention on a ~26 s batch: **~86% of prefill**.

nvprof at d=65536 (aggregate over the whole depth build, so it *understates* the
share at final depth):

| kernel | share |
|---|---|
| maxwell_hgemm_256x128_tn | 46.1% |
| **flash_attn_tile<256,256,16,2,0>** | **37.1%** (196 ms avg, 465 ms max) |
| PtoP | 4.9% |
| gated_delta_net | 4.7% |

At the deepest batch one FA call is 465 ms for 1.65 TFLOP (12 heads x 2048 queries
x 65536 keys x 256 dim, QK + AV) = **3.55 TFLOPS, 18.6% of the 19.05 peak**, next
to a GEMM doing 15.7. Only ~16 of 65 blocks carry a growing KV cache
(`full_attention_interval 4`, `is_recr[il] = il%4 < 3`); the other ~48 are gated
delta net with constant state. So FA is the **only** context-scaling cost here.

### 15 configurations tried, stock wins all

Upstream carries `// TODO optimize kernel parameters for FP16 NVIDIA (P100)` in
`fattn-tile.cuh`. **That TODO is stale** -- the defaults are already a local
optimum for this shape. pp2048@d65536, baseline 179.80:

| nthreads | occ | nbfa | nbk | ncols | REG | t/s |
|---|---|---|---|---|---|---|
| **256** | **2** | **64** | **64** | **32** | **233** | **179.80** |
| 256 | 3 | 64 | 64 | 32 | - | 169.02 |
| 256 | 2 | 32 | 64 | 64 | 154 | 168.80 |
| 256 | 4 | 64 | 64 | 32 | - | 162.96 |
| 256 | 2 | 32 | 64 | 32 | 128 | 161.42 |
| 512 | 2 | 64 | 64 | 32 | - | 161.09 |
| 256 | 2 | 32 | 128 | 32 | 127 | 157.13 |
| 256 | 2 | 64 | 128 | 32 | - | 155.98 |
| 256 | 3 | 32 | 64 | 32 | 80 | 152.36 |
| 512 | 3 | 64 | 64 | 32 | - | 151.32 |
| 128 | 2 | 32 | 64 | 32 | 182 | 144.24 |
| 256 | 2 | 64 | 32 | 64 | 168 | 119.96 |
| 128 | 2 | 64 | 64 | 32 | - | 115.45 |
| 256 | 2 | 128 | 64 | 32 | - | 112.26 |
| 128 | 4 | 64 | 64 | 32 | - | 93.21 |
| 64 | 2 | 64 | 64 | 32 | - | 84.34 |

### Three hypotheses, all falsified -- occupancy is NOT the limit

I predicted each of these and each was wrong:

1. **More threads/block** (nt=512, 1024 threads/SM instead of 512): 161.09. Wrong.
   Raising nthreads *lowers* `cpw = ncols/nwarps`, the register-blocking factor
   (`K_k` is loaded once and reused across `cpw` columns), so it trades away
   arithmetic intensity.
2. **Higher cpw** (nt=128 -> cpw=8, nt=64 -> cpw=16): 115.45 and 84.34. Wrong.
3. **Cut registers to raise occupancy.** `cuobjdump` shows the stock kernel at
   **REG:233**, i.e. 59,648 of the SM's 65,536 registers -> 1 block/SM, 256 threads,
   **12.5% occupancy**. Cutting registers works but *hurts*, monotonically:

   | REG | blocks/SM | occupancy | t/s |
   |---|---|---|---|
   | 233 | 1 | 12.5% | **179.80** |
   | 128 | 2 | 25% | 161.42 |
   | 80 | 3 | 37.5% | 152.36 |

   **Performance is inversely monotonic in occupancy.** The kernel wants registers
   for unrolling/blocking; buying warps with them forces reloads that cost more.
   This also explains why every `occupancy` value made things worse rather than
   nothing -- the hint was unsatisfiable at REG:233, so codegen just degraded
   (`<256,256,32,1,0>` shows STACK:16, real spilling).

### The one real lever, and why it is unreachable

The kernel is **~half memory-bound**, which I had wrongly asserted was not the case.
Blocks per call = (2048/16) x (12/2) = **768**, and each re-reads its KV head's
entire cache (67 MB at d=65536) -- **~51 GB of global reads per call**, ~257 ms of
the measured 465 ms at ~200 GB/s. 67 MB has no chance of staying in a 4 MB L2.

Passes over KV = `Q->ne[1]/cols_per_block`, so doubling cols_per_block to 64 halves
it. Upstream only builds that path `#ifdef GGML_USE_HIP` and only for DKQ<=128.
Implemented it for NVIDIA DKQ==256 (new config case + branch + the
`<256,256,32,2>` instance, which did not previously exist).

**It works, and it is still not enough.** Like-for-like at equal nbatch_fa:
161.42 -> 168.80, **+4.6%** -- the traffic model is right. But Pascal's 48 kiB/block
SRAM limit means every way of affording ncols=64 costs more than it returns:

| ncols=64 shape | SRAM | t/s |
|---|---|---|
| nbfa=64, nbk=64 | 49.0 kiB | **does not fit** |
| nbfa=32, nbk=64 | 40.3 kiB | 168.80 |
| nbfa=64, nbk=32 | 44.5 kiB | 119.96 (nbk=32 doubles the D=256 iterations) |

All of it reverted; `fattn-tile.cuh` is back at HEAD.

### What would actually fix it

**Route attention through cuBLAS.** `maxwell_hgemm` demonstrably reaches 15.7 TFLOPS
*in this model* while the tile kernel gets 3.55. Chunk the KV; per chunk do QK^T and
AV as strided-batched GEMM with an online softmax between them. Compute drops from
465 ms/layer to ~110 ms at 15 TFLOPS, plus ~515 ms/batch of score-matrix traffic ->
roughly **3x**. Caveats: f16 accumulation over a long chunk is not safe for PV
(sum of ~4096 terms), so it likely needs CUBLAS_COMPUTE_32F, which this card runs at
~7.65 TFLOPS -- call it **~2x**, not 3x. It is days of work in generic code.

It would also **eliminate the 512 MiB f16 KV scratch** as a side effect, by
dequantizing per chunk instead of the whole cache.

### Separately: the 512 MiB f16 KV scratch (not yet fixed)

`fattn.cu:551` sets `need_f16_K = need_f16_V = true` for BEST_FATTN_KERNEL_TILE, and
`fattn-common.cuh:1029` calls `to_fp16(K_data, K_f16, ggml_nelements(K), ...)` --
converting the **entire** K and V on **every** call, every layer. The buffer is
reserved by `ggml_backend_cuda_buffer_type_get_alloc_size` (ggml-cuda.cu:936).

At 262144 context, per GPU: 2 of 4 KV heads x 256 dim x 262144 positions x 2 bytes,
for K and V = **512 MiB**. At q4_0 the KV cache costs 9 kiB/token/GPU, so that
scratch is worth **~58,000 tokens of context**.

Prefill only -- decode has `Q->ne[1] == 1` and takes the VEC kernel, which reads
quantized KV directly. The conversion traffic (~22 GB/batch) is only ~1% of time;
this is a **VRAM** problem, not a speed one.

---

## 90 — cuBLAS-GEMM flash attention: long context fixed (KEPT, default-on pre-Volta)

Attempt 89 showed the tile kernel cannot be tuned out of 18.6% of peak. This
replaces it at long context instead. Commits `bdcb3f7bf`, `98de4588f`.

### The path

Keeps flash attention's structure (online softmax over KV chunks), issues the two
matmuls as cuBLAS GEMMs:

    S = K^T Q      strided-batched over the GQA group, f16 compute
    softmax        mask, running max/sum, P in f16, rescale factor for O
    O += V P       f32 compute

S is computed **transposed** (`[n_kv_chunk x n_tokens]`, column-major) so one
query's scores are contiguous -- that makes the softmax kernel coalesced and turns
PV into a plain `V*P` with no transpose. P aliases S (same index, read-then-write
per thread, first-pass loads fenced by the reduction's `__syncthreads`).

Gated to `Q->ne[1] >= 128 && K->ne[1] >= 4096`, mask required, no ALiBi/softcap/
sinks, `cc < VOLTA`. On by default; `GGML_CUDA_FA_GEMM=0` restores upstream.

### Results (equal thermal state, path off vs on)

| depth | tile | GEMM | delta |
|---|---|---|---|
| d=0 | 427.05 +/- 0.44 | 425.19 +/- 2.19 | **within noise -- gate works** |
| d=65536 | 158.43 | **188.99** | **+19.3%** |
| d=131072 | 111.22 | **131.08** | **+17.9%** |

Earlier same-day pair at d=65536 read 183.27 vs 200.60 (+9.5%); run-to-run
variance at depth is large, so call it **+10-19%**.

### Numerics

- `test-backend-ops -o FLASH_ATTN_EXT`: **3949/3949**
- CLAUDE.md gate (c=4096, ppl-orig.txt): **2.6219 +/- 0.01996**, inside the band
- long-context A/B at c=16384 with **q4_0 KV** -- the only test that exercises the
  per-chunk dequant: tile 2.6035 +/- 0.02713 vs GEMM 2.6047 +/- 0.02719, **0.04 sigma**

### Two bugs, both of which produced plausible wrong answers rather than crashes

1. **alpha/beta must match the cuBLAS COMPUTE type, not the data type.** Passing
   `half*` with `COMPUTE_32F` reinterprets 1.0h (0x3C00) as float 2.15e-41, i.e.
   zero -- S becomes uniform and attention degenerates into a plain average of V.
   It still normalizes and stays in range, so it looks healthy. ERR was 0.056 at
   kv=4096 and 0.326 at kv=16384.
2. **The first working version was 16% SLOWER than the tile kernel** (154.57 vs
   183.27). Unfusing attention pays score-matrix traffic that a fused kernel never
   does: at ub=2048, S is 100 MB per chunk and was touched four times (GEMM writes,
   softmax reads twice, writes P, PV reads P) = ~400 MB/chunk, ~410 GB/batch.
   f16 scores + removing a per-head-group `cudaStreamSynchronize` recovered it.
   **I costed this only after writing the code; it should have been costed first.**

Then the profile showed the QK^T GEMM running `maxwell_fp16_sgemm` at 6.2 TFLOPS
because it was still `COMPUTE_32F`; only PV needs fp32 (it sums thousands of
positive terms, QK^T sums k=256). Switching QK^T to `COMPUTE_16F`: 187.97 -> 200.60.

### The 512 MiB staging: request removed, saving NOT demonstrated

`get_alloc_size` no longer reserves the whole-cache f16 staging on this path (it
gates on exactly the same predicate as the dispatch -- if those ever disagree the
kernel writes past the allocation). But **peak VRAM measured identical with the
path on and off at both d=65536 (13493/13237 MiB) and d=131072 (14071/13813)**.

Hypothesis, unconfirmed: ggml sizes one compute buffer by the peak of concurrently
*live* allocations, and at ub=2048 the FFN intermediates (17408 x 2048 x 4 = 142 MB
each) exceed the FA staging until the staging passes them -- which would only
happen near 262144, where it is 512 MiB. Verification at -c 262144 failed on
tooling, not on results (llama-perplexity aborts because ppl-orig.txt is only
123310 tokens; a llama-cli attempt used an invalid flag). **Treat the VRAM saving
as unproven.** Per-chunk scratch is ~70 MB after the P/S aliasing, and it is
*constant* in context length while the staging is proportional -- that is the
whole argument, and it is still just an argument.

### Ceiling arithmetic for long context (this is what caps the target)

Per GPU per 2048-token batch, attention is
`12 heads x 2048 queries x n_kv x 256 dim x 2 (QK,PV) x 16 layers`:

| depth | attention TFLOP | floor at 19.05 TFLOPS | + 4.6 s non-attention | ceiling t/s |
|---|---|---|---|---|
| 65536 | 26.4 | 1.39 s | 6.0 s | ~342 |
| 131072 | 52.8 | 2.77 s | 7.4 s | ~277 |
| **262144** | **105.6** | **5.54 s** | **10.14 s** | **~202** |

Non-attention (~4.6 s) is context-independent and already near its own ceiling
(GEMM at 82% of peak). **So >202 t/s at 262144 is not reachable on this hardware**,
and prefill necessarily degrades with depth -- it starts at ~434 empty.

Attention is currently at ~22-25% of peak on this path. Mapping efficiency to the
262144 number:

| attention efficiency | t/s at 262144 |
|---|---|
| 25% (now, extrapolated) | ~77 |
| 50% | ~130 |
| 65% | ~155 |
| **70% (the 175 t/s target)** | **~165-175** |
| 80% | ~178 |

### Next, in order (none started)

1. **PV GEMM in f16.** It is ~70% of attention compute and runs `COMPUTE_32F` at
   6.5 TFLOPS vs 13.1 in f16. Untested against the perplexity gate -- fp32 was
   chosen out of caution, not measurement. Biggest single lever.
2. **Single-pass softmax** -- removes one full read of S.
3. **Larger chunks** -- QK^T is 12.29 TFLOPS at chunk 2048 vs 14.87 at 16384;
   chunk was sized for scratch, and P/S aliasing has freed room.
4. **Measure at 262144** rather than extrapolating -- throughput, VRAM, and decode.
5. **Decode at depth is unmeasured.** Decode takes the VEC kernel (`Q->ne[1] == 1`),
   which this path does not touch, so it is not covered by any number here.

---

## 91 — PV GEMM in f16 (KEPT, +21.7% at 262144)

First measurement ever taken at the actual operating depth. Everything below
d=262144 in this file was extrapolation; the extrapolation was wrong.

**Baseline at d=262144, GEMM path on: 75.44 t/s.** (Predicted ~95 from a linear
fit through d=65536/131072. Reality was 21% worse.)

### The change

PV was `CUBLAS_COMPUTE_32F`: 6.4-6.7 TFLOPS on Pascal against 11.0-14.6 for
`COMPUTE_16F`, for ~half of attention's flops. Attempt 90 chose fp32 out of
caution ("summing ~chunk positive terms in f16 is unsafe") and never tested it.

That caution was misplaced, and the evidence was already in the tree:
`fattn-tile.cuh:888` declares `half2 VKQ[...]` under `FAST_FP16_AVAILABLE`, so
**upstream's own Pascal kernel accumulates VKQ in f16 across the entire KV
cache** and passes the same 3949 tests. Anything f16-accumulating over a single
chunk is strictly more conservative than the kernel being replaced.

Implementation keeps it stricter still: the GEMM writes an f16 partial for one
chunk (beta=0) and `fattn_gemm_accum_O` folds it into an f32 running O. So f16
summation spans k <= 2048, and cross-chunk accumulation stays f32.

Fusing the rescale into that accumulate *removes* traffic rather than adding it:
the old `fattn_gemm_rescale_O` read+wrote O, then the beta=1 GEMM read+wrote O
again (50 MB/chunk at nt=2048, gqa=6). Now O and Otmp are read and O written
once: 38 MB.

| depth | before | after | delta |
|---|---|---|---|
| 65536 | 188.99 | 220.60 | +16.7% |
| **262144** | **75.44** | **91.79** | **+21.7%** |

Gates: 3949/3949; ppl 2.6214 +/- 0.01995 (gate 2.6209 +/- 0.0199); same-corpus
A/B against the path disabled 2.7561 vs 2.7570 = 0.04 sigma.

### The perplexity gate corpus is not ./ppl.txt

CLAUDE.md says `-f ./ppl.txt` and requires 2.6209. The `ppl.txt` in the tree
gives **2.7570 on stock upstream** (path disabled) and 2.7561 with this change --
i.e. the documented number is unreachable on that file for *any* build. The gate
was calibrated on `p100-handoff/ppl-orig.txt` (420098 B), which reproduces
2.6214. The two files differ. **Use ppl-orig.txt; ppl.txt fails the gate for
reasons that have nothing to do with the kernel.**

---

## 92 — a fast harness: stop paying 30 minutes per data point

`llama-bench -d 262144` rebuilds 262144 tokens of context (~30 min) to time one
27-second batch: a 60:1 overhead ratio on the quantity of interest.

`make_test_cases_perf()` in test-backend-ops now carries the per-GPU production
shape -- `test_flash_attn_ext(256, 256, 2, {6,1}, kv, 2048, ...)` with q4_0 K/V,
at kv 32768/65536/131072/262144. That is exactly what one GPU sees for qwen3.5
under `-sm tensor -ctk q4_0 -ctv q4_0 -ub 2048`: 2 KV heads, GQA 6, D 256.
Perf-only, so the 3949 correctness tests are untouched. **Seconds per point.**

Verified against the profile: FA launch count at d=65536 was 35840 =
561 batch-chunks x **16** layers x 2 KV heads x 2 GPUs, confirming
`full_attention_interval 4` leaves exactly 16 of 65 layers with a growing cache.

A prefill batch is 16 of these ops, so `t/s = 2048 / (16*t_op + const)`.

---

## 93 — where the time actually goes at 262144

Three points, one build, and the model is linear to 0.2%:

    t_batch = 4.94 s + depth * 6.63e-5 s

| depth | measured | s/batch | predicted |
|---|---|---|---|
| 65536 | 220.60 | 9.283 | - |
| 131072 | 150.66 | 13.593 | 13.626 |
| 262144 | 91.79 | 22.312 | - |

Intercept 4.94 s matches the independently profiled context-independent kernel
total (5.4 s/batch/GPU) and the d=0 batch time (4.72 s).

Budget per GPU for the 262144 batch (22.31 s):

| component | time | how obtained |
|---|---|---|
| context-independent kernels | 5.4 s | profile, /33 batches |
| FA kernels | ~13.1 s | profile, scaled by sum(n_kv) |
| **unaccounted** | **~4 s** | remainder |

### What the residual is NOT

- **Not host-side.** Phase timers (re-enabling the commented-out ones in
  `llama-context.cpp`) give, per batch at depth: graph build **1.1 ms**,
  `set_inputs` **30-43 ms** (of which the KQ mask is 30-36 ms), everything else
  inside `graph_compute`. The mask is O(n_kv*n_tokens) but has an incremental
  fast path (PR 18842) and costs 25 ms at n_kv=34816, ~190 ms extrapolated to
  264192.
- **Not thermal throttling.** Clocks hold 1240-1290 of 1328 MHz (-6%) with SW
  power cap at 210 W. The isolated op still reports 10.0-10.1 TFLOPS after four
  consecutive runs at 73 C.
- **Not any kernel.** Every kernel in the profile is accounted for as either
  depth-scaling (the FA set) or per-batch constant.

### Reading `utilization.gpu` cost me time

Sampling showed both GPUs at 0-13% for ~2.4 s before each batch's compute, which
looked like a host stall. It is not: `utilization.gpu` counts **kernel execution
only**, so DMA copies read as 0%. The phase timers then showed host work is 35 ms.

### The live hypothesis

**CUDA graphs are disabled on Pascal** ("disabling CUDA graphs due to GPU
architecture"), so every kernel is launched individually from the host. This path
issues **6 launches per chunk** (dequant K, dequant V, QK, softmax, PV, accum_O).
At d=262144: 129 chunks x 2 KV heads x 16 layers x 6 = **~24,800 launches per GPU
per batch**, ~49,500 issued from a single host thread for both devices.

Note this makes the op-level harness *unrepresentative for launch cost*: it runs
one GPU with no contention. A chunk sweep there is flat (below), but that does
not settle it in production -- to be tested end-to-end.

**Correction, on arithmetic done after writing the above:** launch issue is not
big enough to be the main term. 49,536 launches per batch across both devices at
~10-20 us of host issue is 0.5-1.0 s, not 4 s. A second measurable term is the
**KQ mask upload**: 264192 x 2048 x 2 B = 1.08 GB per GPU per batch, re-sent
every batch, and the profile's HtoD rate is 2.47 GB/s -> ~0.9 s for both GPUs at
d=262144 (0.22 s at d=65536, matching the depth scaling). Almost all of that
upload is unchanged between batches -- only the newest 2048 columns differ -- but
it is a graph input and is re-sent whole. Together these cover perhaps half the
residual; the rest is still unattributed, and the end-to-end chunk sweep that
would separate them was not run.

---

## 94 — chunk size sweep (op level): no change, chunk stays 2048

`FA_CHUNK` env override, kv=262144, one GPU:

| chunk | TFLOPS |
|---|---|
| 1024 | 9.78 |
| **2048** | **10.42** |
| 4096 | 10.18 |
| 8192 | 10.31 |
| 16384 | 10.30 |

Larger chunks do not pay at op level despite better GEMM k -- the score matrix
grows with chunk and the extra traffic cancels it. Reverted to the constant 2048.
**Still to test end-to-end**, where halving the chunk count also halves host
launch issue, which this measurement cannot see.

### Softmax rewrite: rejected (+0.8%)

One block per query token covering the whole GQA group, mask staged in shared
memory once instead of re-read gqa=6 times, scores cached in registers so S is
read once instead of twice. Traffic per chunk ~200 MB -> ~108 MB.
Measured 642715 us vs 648017 (**+0.8%**), inside the 648-660 us run-to-run band.
Below the 2% threshold; reverted rather than carry the complexity.

### Merged GEMM: kept (+2.7%)

K and V are shared across the GQA group and Q/S/P/Otmp are contiguous across
heads, so the strided-batched call described exactly the same memory as one GEMM
with n = nt*gqa. Issuing one GEMM: 648017 -> 630650 us, **10.18 -> 10.46 TFLOPS**.
3949/3949; ppl 2.6222 +/- 0.01996.

---

## 95 — final state at 262144, and the decode curve

### Prefill (the metric)

| build | pp2048 @ d262144 |
|---|---|
| session start (attempt 90 kernel) | 75.44 |
| + PV in f16 (91) | 91.79 |
| + merged GEMM (94) | **95.14** |

**+26.1% at the operating depth.** Full curve on the final build:

| depth | pp2048 |
|---|---|
| 0 | ~427 |
| 65536 | 220.60 |
| 131072 | 150.66 |
| 262144 | **95.14** |

### Decode (tg128, r=2)

| depth | t/s | vs empty |
|---|---|---|
| 0 | 31.51 +/- 0.23 | - |
| 65536 | 15.55 +/- 1.35 | -51% |
| 262144 | **7.32 +/- 0.56** | **-77%** |

Decode takes the VEC kernel (`Q->ne[1] == 1`), which the GEMM path does not
touch -- gated at `Q->ne[1] >= 128`. So this curve is upstream behaviour and is
unchanged by attempts 90-94, but it had never been measured.

**Decode at depth is ~2x off its memory-bound floor.** Per token per GPU:

| term | bytes | at d=0 | at d=262144 |
|---|---|---|---|
| weights | 10.4 GB | 10.4 GB | 10.4 GB |
| KV cache (q4_0, 16 layers, 2 KV heads) | 2.4 GB | - | 2.4 GB |
| measured time | | 31.7 ms | 136 ms |
| **implied bandwidth** | | **328 GB/s** | **94 GB/s** |

P100 HBM2 peak is 732 GB/s. Reading weights alone sustains 328; adding the KV
cache read drops the effective rate to 94. At the 328 GB/s the same card already
demonstrates, 12.8 GB/token would be 39 ms -> **~26 t/s**; even at a conservative
196 GB/s it is 65 ms -> **~15 t/s**, against 7.32 measured.

**This is the largest unexploited win identified in this project and it was never
attempted.** It is a decode-side FA/KV-read problem, entirely separate from the
prefill work above.

### Why 175 t/s prefill at 262144 is not reachable on this hardware

Every term below is measured, not extrapolated:

- context-independent work: **4.94 s/batch** (linear-fit intercept; independently
  confirmed by the profile's per-batch constant kernels at 5.4 s and by the d=0
  batch time of 4.72 s)
- attention flops at 262144, per GPU per batch: **105.6 TFLOP**

175 t/s means a 2048-token batch in 2048/175 = **11.70 s**, leaving
11.70 - 4.94 = **6.76 s** for attention, i.e. **15.6 TFLOPS sustained** including
softmax, mask traffic, per-chunk dequant and the score matrix.

The fastest pure cuBLAS hgemm anywhere in this model -- the FFN GEMM at k=5120,
no softmax, no mask, no score traffic -- is **15.7 TFLOPS**. So 175 requires
attention-with-softmax to run at the speed of the fastest bare matmul on the card.

Current attention: 10.46 TFLOPS (55% of the 19.05 fp16 peak). Realistic ceiling
with the residual eliminated and attention at ~13 TFLOPS is **~150 t/s**; the
likely landing zone is **110-130**.

### Decode: two hypotheses tested and rejected

Using the new nb=1 perf cases (4 ms per measurement):

| experiment | kv=262144, nb=1 |
|---|---|
| **default (VEC)** | **4153 us** |
| forced parallel_blocks=2 | 14876 us |
| forced parallel_blocks=4 | 7335 us |
| forced parallel_blocks=8 | 4938 us |
| forced parallel_blocks=16 | 4548 us |
| forced parallel_blocks=32 | 4847 us |
| forced TILE kernel | 6036 us |

So the KV dimension is **already** well split by `launch_fattn`'s efficiency
search, and VEC is already the better of the two kernels. Neither is the problem.

The remaining explanation is **GQA redundancy**: the vec kernel reads the KV
cache once per Q head, so each KV head is re-read gqa=6 times. That is 906 MB per
op rather than 151 MB, which puts the kernel at a respectable **~218 GB/s**, not
the 36 GB/s a naive byte count suggests. The kernel is not slow; it is doing 6x
more reads than necessary.

Note the selection logic already prefers TILE over VEC when GQA applies -- but
only for *unquantized* KV (`fattn.cu`: the `!ggml_is_quantized` branch requires
`!gqa_opt_applies`, the quantized branch does not). For q4_0 it takes VEC
regardless. Measured here, that choice is correct (VEC 4153 < TILE 6036); both
leave the 6x on the table.

**Fixing this needs a GQA-aware decode kernel that loads a KV head once and dots
it against all gqa Q heads.** Estimated payoff if KV traffic drops 6x: op ~4.15
-> ~1.2 ms, decode 136 -> ~89 ms/token, i.e. **7.32 -> ~11.5 t/s at 262144
(+57%)**. Not attempted -- it is a new kernel, not a parameter change.

---

## 96 — two-stream pipeline: REJECTED, and a correction to how the op harness was read

### The change (reverted)

Software-pipelined the chunk loop across two streams: chunk c+1's dequant and QK on
a producer stream, chunk c's softmax/PV/accum on the main stream, S/K/V double
buffered, joined with events. The consumer chain stays strictly ordered, which the
online-softmax state requires. Passed 3949/3949.

Rationale was that softmax (~19% of the op, memory-bound) should hide inside the
next chunk's compute-bound QK. **It does not.** Both chains contend for the same
SMs and the GEMMs already saturate compute, so there is little idle capacity for
the softmax to occupy. Moving the V dequant to the producer as well changed
nothing (611313 vs 610639).

### The correction

The pipelined build measured 610639 us against a 630650 us baseline, which I
reported as +3.2%. **That was noise.** Reverting the change and re-measuring gave
**612441 us** -- indistinguishable from the pipelined number. The pipeline was
worth nothing.

Repeating the identical binary from cold shows why:

| run | us/run | GPU temp |
|---|---|---|
| 1 | 612441 | ~63 C |
| 2 | 613866 | 65 C |
| 3 | 618801 | 68 C |
| 4 | 622617 | 69 C |
| 5 | 623410 | 70 C |

**The op harness drifts ~1.8% monotonically with die temperature**, and across a
longer session the spread reaches 3%. Earlier in this session the same code went
648017 -> 660627 over five runs while heating. That is the same magnitude as most
of the deltas being chased.

**Consequences for what is recorded above:**

- The merged-GEMM result in attempt 94 (+2.7%, 648017 -> 630650) compared a cold
  baseline against a warm candidate, so the direction is right but the magnitude
  is not trustworthy. Cold-to-cold it looks more like 5%, but that pairing is not
  controlled either.
- The softmax rewrite (+0.8%) is comfortably inside the noise band and its
  rejection stands for a better reason than the one given.
- **End-to-end `llama-bench -d` is also noisy at depth**: the same build measured
  150.66 and 157.39 t/s at d=131072 (4.5% apart, the second under nvprof).

**Method for anyone continuing: alternate A/B/A/B from the same thermal state, or
require the effect to exceed ~4%.** A single before/after pair at either level
cannot resolve less than that. The only deltas in this session large enough to be
safe on a single pair are PV-in-f16 (+21.7% end-to-end at 262144) and the
session total (75.44 -> 95.14, +26.1%).

---

## 97 — MTP at 262144 with ub=2048: the draft context was reserving the target's ubatch (KEPT)

### The problem, as posed

The goal is all three at once: full 262144 context, prefill fast enough to stay near
442 t/s at short context (which requires `-ub 2048` -- see the ubatch sweep), and MTP.
That combination **did not run**: it aborts during startup with

    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1296.06 MiB on device 0:
    cudaMalloc failed: out of memory

### Where the VRAM goes (measured, per GPU, `-c 262144 -b 2048 -ub 2048`, MTP n-max 4)

| buffer | MiB | scales with |
|---|---|---|
| model | 10215 | fixed (+187 vs non-MTP, the nextn block) |
| target KV, q4_0 | 2304 | context |
| recurrent state | 374 | **draft lanes** (4 rs_seq; 75 MiB at 1) |
| target compute | 1512 | ubatch x n_kv (~1074 of it is the KQ mask) |
| draft KV, f16 | 512 | context |
| **draft compute** | **1296** | **ubatch x n_kv -- its own copy of the mask** |

Total wanted 16213 MiB against ~15.6 GB usable (16276 on the card, less Sunshine's
392 on GPU0). **Short by ~600 MiB.**

The recurrent state is the term the user noticed: it is `n_max` x 75 MiB, so lanes do
cost VRAM directly. But it is not what breaks the build -- the draft's *compute buffer*
is, and that one is not intrinsic at all.

### The cause

`common_base_params_to_speculative` copies the target's `common_params` wholesale, so
the draft context inherits `n_ubatch = 2048`. Its compute buffer is then reserved for a
2048-wide ubatch against the full 262144-cell cache, and at that shape the KQ mask alone
is `262144 * 2048 * 2 = 1074 MiB`. The draft is **one layer**, and both draft prefill
loops already chunk by `llama_n_ubatch(ctx_dft)` (`speculative.cpp:1103`, `:625`) -- a
narrower draft ubatch just means more iterations of a single-layer graph. The wide
ubatch buys prefill throughput on the *target*; the draft was paying for it for nothing.

### The change

New `--spec-draft-ubatch-size` / `-ubd` (default 0 = inherit, so nothing changes unless
asked), applied in `common_base_params_to_speculative` where every caller -- server and
`common.cpp:1304` -- already routes.

### Results

`-ubd 256`, per GPU: draft compute **1296 -> 162 MiB**, saving **1134 MiB**. The full
config now runs.

Speed cost at short context, alternating A/B/A/B from the same thermal state, MTP
n-max 4 / p-min 0.2, 260 tokens greedy:

| config | t/s | t/s |
|---|---|---|
| baseline | 52.256 | 52.145 |
| `-ubd 256` | 52.121 | 52.140 |
| `-ctkd/-ctvd q4_0` | 50.965 | 51.012 |

`-ubd` is **free** (within noise, and every run produced 197 accepts of 252 drafted --
byte-identical output). Quantizing the draft KV cache also works and saves a further
368 MiB (512 -> 144), but it costs **2.2%**, so it is a margin lever to reach for only
if needed, not a default.

### Where that leaves the budget

With `-ubd 256` alone at 262144 + ub 2048 + MTP, peak measured with nvidia-smi:

| GPU | peak | free |
|---|---|---|
| 0 | 15977 MiB | ~300 (Sunshine holds 392 here) |
| 1 | 15585 MiB | ~690 |

It fits, but ~300 MiB on GPU0 is not comfortable margin. The next lever is the
**target's** 1074 MiB KQ mask, which is also uploaded from a 1104 MiB pinned host buffer
every batch -- device VRAM, host RAM and prefill time in one item. See the next attempt.

Unrelated but worth recording: the `backend offload failed for seq_id=0; using CPU
sampler` warning at MTP startup is **pre-existing** and appears in every run including
the baseline. It is the `SPLIT_MODE_TENSOR` backend-sampling limitation from attempt 88,
not a fault of this change.

Also: the server's context checkpoints (`created context checkpoint N of 32, size =
149.626 MiB`) are host-side `std::vector<uint8_t>` state copies, not VRAM. At the default
32 they are ~4.8 GB of **host** RAM at this context length; `-ctxcp` tunes them.

---

## 98 — the full-context + fast-prefill + MTP config, measured end to end

Validation of attempt 97 at the operating point, plus one dead end.

### It runs, and here is what it does

`-c 262144 -b 262144 -ub 2048 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0`, MTP n-max 4 /
p-min 0.2, `-ubd 256`, 76662-token prompt (`-b` must exceed the prompt: the
speculative tools reject a prompt larger than the *logical* batch, which is why
`-b 2048` fails there while `-ub 2048` is what actually sets the compute shape):

| metric | value |
|---|---|
| prefill, 0 -> 76662 | 76662 tokens in 292.464 s = **262.12 t/s** average over the ramp |
| MTP decode at ~76.7k | **13.38 t/s**, 81.25% accept |
| peak VRAM GPU0 | **16133 MiB of 16276 -- 143 MiB free** |
| peak VRAM GPU1 | 15741 MiB -- 535 MiB free |

Short-context prefill is unaffected, as it must be -- attempt 97 touches only the
draft context's params: **pp2048 = 430.61 +/- 0.72** starting at 53 C, inside the
documented thermal band (442.59 at 39 C, 438.49 at 55 C, 434.10 starting at 52 C).

**143 MiB of margin on GPU0 is not enough to rely on.** The asymmetry is exactly
Sunshine's 392 MiB, and Sunshine's footprint is not constant -- it grows while
actually streaming. The only margin lever that works today is `-ctkd q4_0 -ctvd
q4_0` (+368 MiB, -2.2% decode), which would put GPU0 at ~511 MiB free.

### Dead end: asymmetric tensor split does not rebalance this

The obvious idea is to offset Sunshine's 392 MiB by giving GPU0 a smaller share.
`-ts` is far too coarse for that. Measured (short-prompt probe, so absolute
numbers are ~364 MiB below the at-depth peaks above):

| `-ts` | GPU0 free | GPU1 free | result |
|---|---|---|---|
| (none, 50/50) | 507 | 899 | runs |
| 199,201 (49.75/50.25) | 2111 | 391 | **OOM** |
| 399,401 (49.875/50.125) | 2111 | 393 | **OOM** |
| 48,52 | 2329 | 175 | **OOM** |
| 46,54 | 2835 | 1499 | **OOM** |

A 0.125% nudge and a 4% shift produce the *same* ~2.1 GB migration, so the split
is quantised at a granularity far larger than the ~200 MiB being asked for, and
every non-equal ratio lands worse than balanced. There is no fine-grained setting
here. Do not re-litigate.

### The durable fix is the target's KQ mask

Per GPU: **1074 MiB of device VRAM** (262144 x 2048 x f16, inside the 1512 MiB
compute buffer) plus a **1104 MiB pinned host buffer** it is uploaded from every
batch. That single item is device VRAM, host RAM and prefill time at once, and it
is ~4x the margin problem.

The CUDA side is easy: the GEMM path's softmax kernel already takes `mask` and
already handles `nullptr` (`fattn-gemm.cu:42, :60, :198`), so an implicit-causal
branch there is a few lines. The risk is entirely in llama.cpp. Causal-by-
arithmetic (`mask[i][j] = 0 iff j <= n_past + i`) is only valid when KV cell index
equals position, which holds for a single sequence on a freshly filled unified
cache and is broken by context shift, defrag, multi-sequence and SWA. It is also
not enough to fall back dynamically: the compute buffer is *reserved* for the
worst case, so a fallback that can still materialise the mask saves no VRAM.

Any implementation therefore has to decide at context creation (n_seq_max == 1,
causal, no SWA) and then **verify per batch with a hard failure**, not a silent
fallback. That is the shape of the work; it was not started.
