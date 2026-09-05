# P100 (sm_60) CUDA kernel optimization — handoff

> **STALE — this is the session-1 document (2026-08-29), kept because its analysis of
> `mul_mat_vec_q` is still correct and still load-bearing.**
>
> For current state read **`p100-handoff/RESUME-HERE.md`**. Short version as of
> 2026-09-05 (HEAD `606215cbf`): `tg256` is **32.11 t/s** and MTP decode at 262144
> context is **23.2 t/s**. Most work since this document has been long-context and
> speculative-decode, not `mul_mat_vec_q`.
>
> One correction: the **PPL 2.7554** below is the figure for `./ppl.txt`. That is *not*
> the corpus CLAUDE.md's 2.6209 gate refers to — that one is
> `p100-handoff/ppl-orig.txt`, which gives **2.6186**. `./ppl.txt` yields 2.7566 on any
> build including stock, so a run against it can never match the documented gate. See
> `p100-handoff/CORPUS.md`.

**Result: 17.45 → 27.03 t/s (+55%)** on the CLAUDE.md metric
(`qwen3.8-27B-Q6_K`, 2× Tesla P100, `-sm tensor -fa 1 -ctk/-ctv q4_0`, tg256).

All changes are kernel-level and quant/model-agnostic. Verified across q6_K, q3_K, q4_K, q5_K,
q4_0, q5_0, q8_0, q2_K, iq4_nl. `test-backend-ops -o MUL_MAT` 1193/1193 and
**PPL 2.7554 ± 0.02151 (identical to stock) at every kept step.**

---

## 1. What is committed

| commit | change | t/s |
|---|---|---|
| `d8984de0` | remove `__vsubss4` from Q6_K/Q3_K vec_dot | 17.72 |
| `c7e7faf6` | cooperative shared-memory staging of x | 20.35 |
| `f07b913a` | vdr=2 for Q6_K; Pascal geometry moved into source | 23.28 |
| `0732c729` | vdr=4 for Q6_K; geometry retuned to 2×2 | 24.33 |
| `9fc142d1` | uint4 (16-byte) staging | 26.21 |
| `560442c8` | integer accumulator across the vdr group | **27.03** |

### 1.1 `__vsubss4` removal
No NVIDIA GPU has had SIMD-video hardware since Kepler; ptxas emulates `__vsubss4` in 9 SASS
instructions. Both K-quant call sites subtract a power-of-two bias from packed sub-byte values
whose high bits are zero, so saturation is unnecessary. `b - 32` is the sign-extension of a 6-bit
value from bit 5: flip bit 5, shift left 2, and the sign lands in bit 7 — a valid packed int8 for
dp4a equal to `4*(b-32)`. Both shifts fold into the extract shifts that were needed anyway; the
factor of 4 is undone once in the float scale. Q3_K is the same idea from bit 2 with a shift of 5.

### 1.2 Cooperative staging (the big one)
Every ggml block size is **2 mod 4** (q6_K 210, q8_0 34, q4_0 18, q3_K 110) because each block
carries a 2-byte `ggml_half` beside a multiple-of-4 payload. So `get_int_b2` must issue two 16-bit
loads per quant word: ~7 load instructions and ~26 bytes per block.
Instead the warp pulls its whole contiguous run of blocks in with coalesced loads from the
**4-byte-aligned address below the data**, and extraction reads from shared memory where the odd
alignment is free. The misalignment is *preserved* in shared memory rather than removed — that is
what keeps the global side aligned without repacking the weights.

### 1.3 vdr = 4
For `iqs` a multiple of 4, `bq8_offset`, `scale_offset` and `vh_shift` are identical across all
four indices (they are floor-divisions whose moduli, 8 and 4, are multiples of 4), and the ql
index, qh index and q8_1 lane index are each consecutive without wrapping (`iqs % 8` is 0 or 4).
So the scales, block scale and q8_1 `.ds` values are fetched once per four lanes.
**vdr=8 is not viable**: `scale_offset` changes at `iqs % 16 == 4`.

### 1.4 uint4 staging
Staging moved 32 bits per lane (128 B/warp/instruction). A `uint4` moves 512 B. Global loads and
shared stores each dropped ~4×. Shared memory already tolerates an arbitrary byte offset (that is
what `mis[]` carries), so the run is fetched from the 16-byte-aligned address below it.

### 1.5 Integer accumulator
`scales[4*i]` and the q8_1 scale are constant across the vdr group, and `dp4a` already takes an
accumulator. So all four `l` values accumulate in an integer *before* any float work, collapsing
the integer multiply by the scale (3 XMADs each — Pascal has no IMAD) and the int→float conversion
from 8 per vec_dot to 2. Peak `|acc|` = vdr·4·128·128 = 262144, inside float's exact integer range.
Rounds twice per `i` instead of four times, so slightly *more* accurate.

### 1.6 Tuning lives in the source now
`-DP100_NWARPS/-DP100_ROWS/-DP100_MC_*` are **no longer read**. The measured Pascal values are
compiled in, gated on `__CUDA_ARCH_LIST__ == 600` so `MMVQ_PARAMETERS_GENERIC` — which Ampere and
later also fall through to — is untouched. `__CUDA_ARCH_LIST__` is the correct test because it is
visible to **both host and device passes**, and `calc_nwarps` feeds both `__launch_bounds__` and
the host launch configuration, which must agree. A flagless build now reproduces the tuned result;
the old "17% cliff from forgetting the flags" is gone.

---

## 2. The mental model — what actually binds

Measured, not assumed. This is the most valuable part of the handoff.

- **Machine streaming ceiling: 605 GB/s per GPU** (ECC on; spec is 732, and 83% is normal
  achievable). Not 732. Do not plan against the spec number.
- **ECC is free on P100.** HBM2 has dedicated ECC storage — `nvidia-smi` shows 16269 of 16384 MiB,
  a 115 MiB driver reserve. In-band ECC (GDDR5 cards) would cost ~1024 MiB. **Keep ECC on**;
  disabling it buys nothing.
- **Bandwidth by per-thread load width**: 16 B → 604, 8 B → 598, 4 B → 533, **2 B → 328 GB/s.**
- **The kernel is bound by the number of *global* memory instructions**, not by bandwidth, not by
  ALU, not by warp occupancy. Corrected twice during the session:
  - "issue-bound" was too coarse — a 36% instruction cut *lost* 6%.
  - "LSU-bound" was also too coarse — a variant with **fewer total LSU ops (40 vs 46) was 5%
    slower** because it traded shared loads for global ones. **LDG ≫ LDS in cost.**
- **Current split** (q6_K, m=4096 n=1 k=14336): full **108.9 µs** = **95.2 µs memory + 13.7 µs
  arithmetic**. Arithmetic is 62% of the *instructions* but only 12.5% of the *time*.
- **Registers are the recurring killer.** At nwarps=2 the kernel sits at 66-67. Nearly every
  optimization that reduces one resource raises registers and nets negative.
- **Occupancy is NOT the limiter, in either direction.** Forcing *fewer* registers hurts
  monotonically even with zero spill (67→27.01, 64→26.63, 61→26.68, 56→26.34), and raising
  occupancy by other means loses too. Per-thread resources matter more than warp count here.
  Note the two limits: at 64 threads/block registers allow 14 blocks/SM and shared memory allows
  17, so **registers bind and shared memory has headroom — until you double it, at which point
  smem binds at 9 blocks.**
- Decode profile: mmvq **76%** of GPU time; the other 24% is ~15 latency-bound kernels of 1-4%
  each (rms_norm 3.6%, quantize_q8_1 3.5%, k_bin_bcast 2.7%, flash_attn 2.4%).
  GPU is ~95% busy, so launch overhead is not the problem.

---

## 3. Ceilings

```
weights 22.42 GB, tensor-split      -> 11.21 GB per GPU per token
mmvq memory floor                   -> 11.21 / 605 GB/s = 18.5 ms/token
```

| scenario | t/s |
|---|---|
| now | 27.03 |
| free arithmetic, everything else unchanged | **30.9** |
| perfect repack (memory at the streaming floor) | ~30.5 |
| + arithmetic halved | ~33 |
| + non-mmvq tail halved as well | ~39 |
| zero arithmetic AND zero other kernels | 54 |

**40 t/s is not reachable by kernel work.** It needs ~702 GB/s/GPU against a 605 GB/s ceiling
unless the arithmetic *and* essentially the whole non-mmvq tail also vanish. The model is dense
(GGUF has no `expert_count`; `qwen35`, 65 blocks, hybrid attention/SSM), so every weight is read
every token — confirmed independently by mmvq moving ~24 GB/token at its measured rate.
Reducing **bytes per token** (MTP, or a smaller quant) is the only lever that changes this.

---

## 4. What was tried and failed — do not repeat without new information

| attempt | result | why |
|---|---|---|
| Hoist row base pointers / kill address multiplies | 16.69 (−6%) | 4 live 64-bit pointers → regs 71→96, 2 blocks/SM |
| `__launch_bounds__` forcing 3 or 4 blocks/SM | 15.6-16.7 | capping registers destroys per-thread MLP |
| `#pragma unroll 2 / 4` on the block loop | −4 / −13% | regs 71→127→165 |
| Register-prefetch software pipeline | wash | **not latency bound** — 32 warps already hide it |
| Aligned `get_int_b2` (4 separate designs) | all slower | see §5 |
| CUDA graphs on Pascal (gate is `cc < VOLTA`, undocumented) | −1% | capture/validate costs more than it saves |
| `rms_norm` 256- vs 1024-thread block | neutral | latency bound, not reduction bound |
| 32-bit byte-offset indexing for q8_1 | no-op | compiler already did it |
| Compile-time staging bound | −1.4% | stages ~3% more bytes every iteration |
| Force 64/61/56 registers via `__launch_bounds__` | 26.6 / 26.7 / 26.3 | monotonically worse **with zero spill** — ptxas rematerialises instead |
| Double-buffer the shared stage (remove the WAR barrier) | 26.38 | smem 3712 → 7168 B makes *shared memory* the occupancy limiter (14 → 9 blocks/SM) |
| Padded/repacked global layout | not attempted | ~800-1200 lines, must be shared with MMQ, and only reaches ~30.5 |
| Multi-column (MTP) geometry re-sweep | zero sensitivity | that path is compute bound; weights read once regardless of column count |
| Lift the Pascal exclusion on mmvq GLU fusion | 25.49 | upstream's "not universally faster on Pascal" **still holds against the reworked kernel**: the fused kernel reads the *gate* matrix unstaged, and staging it too would double smem to 7424 B, making shared memory the occupancy limiter (14 -> 9 blocks/SM). Fusion's real upside is small anyway -- it saves the intermediate round-trip, ~18 MB/token against 22.4 GB of weights |
| q8_1 activation padding for 128-bit `u` loads | 2% ceiling (measured) | activation is L1/L2-resident; those loads are already nearly free |

---

## 5. The alignment wall (four attempts, all negative)

A 4-byte quant word at a 2-byte-aligned address costs two memory instructions. That cost can be
**relocated but not removed**:

- read from global directly → two 16-bit global loads (original code)
- stage it, read from shared → two 16-bit shared loads (current code)
- strip the misalignment while staging → funnel shift needs the neighbouring word, so a shuffle
  or a second load per word, **and per-block slots turn one contiguous run into several base
  addresses, raising the global load count** (26.50-26.85 vs 27.03)
- funnel-shift at extraction instead (vdr+1 aligned shared loads for vdr words) → LDS 34→26 but
  the staged arrays cost 11 registers (26.86 vs 27.03)

**Pitfall worth remembering:** casting a shared-memory pointer through `uintptr_t` and back
**loses the address space** — ptxas silently emits generic loads instead of `LDS` (cost us 25.61
until spotted). Derive aligned pointers with `char *` arithmetic from the original pointer.

The only real escape is weights actually aligned in global memory.

---

## 6. Process notes

- **Build times**: `mmvq.cu` alone ≈ 90 s. Touching `vecdotq.cuh` or changing any `-D` flag
  rebuilds ~200 template instances ≈ 25 min. Keep experiments inside `mmvq.cu` where possible; put
  sweep knobs in the source rather than in flags.
- **Fast isolated benchmark** (1.8 s):
  `./build-opt/bin/test-backend-ops perf -o MUL_MAT -b CUDA0 -p 'type_a=q6_K,type_b=f32,m=4096,n=1,k=14336'`
- **Always confirm on the real model.** The isolated single-shape proxy is misleading: vdr=4 looked
  *better* than vdr=2 in isolation while being 7% *worse* on the model at the old geometry.
- **Raising vdr moves the launch-geometry optimum**; re-sweep every time, against the real model.
- **nvprof metrics are unavailable** (`RmProfilingAdminOnly: 1`, needs root). Timing works;
  `--print-gpu-summary` works. Everything here was derived from timings + SASS + microbenchmarks.
- **Correctness gates**, in order of sensitivity: a standalone bit-exactness cubin (used for
  `__vsubss4`: 4.2M random tuples), then `test-backend-ops -o MUL_MAT -b CUDA0` (1193 cases vs the
  CPU backend — this catches layout bugs immediately), then perplexity. Perplexity runs at batch
  2048, which goes through **MMQ and barely exercises mmvq at all**, so it is the *weakest* gate
  for decode-path changes despite being the slowest.
- **The PPL figure in CLAUDE.md (2.6209) does not match this repo.** The stock, unmodified kernel
  gives **2.7554 ± 0.02151** on `./ppl.txt`, all 30 chunk values identical. `ppl.txt` is the
  llama.cpp README, not wikitext — that is the discrepancy. Use 2.7554 as the reference.

---

## 7. Where to go next

1. **MTP** — the only lever that changes bytes/token. Marginal cost of a draft column is ~55 µs
   against 116 µs for the first, so it amortises well. On the previously measured 1.275× that is
   ~34 t/s, for zero risk. This is the highest-value remaining action.
2. **Per-type Pascal geometry.** The launch geometry is Pascal-wide but was tuned against the only
   model available (q6_K), costing ~2% on q4_K/q5_K/q2_K. `calc_nwarps` already takes `type`;
   fixing this needs models of those types to tune against.
3. **Repack to an aligned global layout** — ~30.5 t/s, ~800-1200 lines, must be shared with the
   MMQ path (`ggml_cuda_mul_mat` picks mmvq vs MMQ per call on the same tensor, so it cannot be
   scoped to decode). Failure mode is *silently wrong dot products* that can still pass the
   perplexity band — build a bit-exactness harness first.
4. **Algorithm choice was checked, not assumed.** `mmvf` handles only F16/F32 weights and the old
   dequantize-mul-mat-vec path is gone from modern llama.cpp, so there is no ready-made
   dequantize-and-FMA alternative to exploit P100's fast FP16 (2:1, the only Pascal with it, and
   the only one without DP4A). Writing one from scratch would change the numerics materially --
   FP16 accumulation over 14336 elements -- so it needs a perplexity plan, not just a kernel.
5. **The batch-1 latency tail** (24% of decode, ~15 kernels). No single big win; CUDA graphs were
   the one change that could have addressed it wholesale and it measures negative.
