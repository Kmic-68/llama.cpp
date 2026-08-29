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
