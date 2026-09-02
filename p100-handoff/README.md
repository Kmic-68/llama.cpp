# llama.cpp CUDA kernel optimisations for Tesla P100 (sm_60)

2x Tesla P100-PCIE-16GB, tensor-split, Qwen3.5 27B Q6_K, q4_0 KV cache.

**Current numbers (2026-09-01, HEAD `836e9fdc4`)** -- these supersede everything
else in this file, which describes the state at the end of session 2:

| workload | CLAUDE.md baseline | now |
|---|---|---|
| plain decode `tg256` (the CLAUDE.md metric command) | 17.51 t/s | **32.1** best / ~31.8 typical (**1.83x**) |
| speculative decode (MTP, `-n-max 4 -p-min 0.2`) | 32.94 | **54.5** |
| prefill `pp2048 -b 2048 -ub 2048` | 222.6 (at `-ub 512`) | **442.6** cold / ~437 hot |
| perplexity (`ppl-orig.txt`, q4_0/q4_0) | 2.6209 | **2.6214 +/- 0.01995** |

Read `RESUME-HERE.md` first. The full attempt log is `../OPTLOG.md` -- start
from its CLOSING SUMMARY. `full-kernel.diff` is the whole kernel delta against
upstream `f280b2698`; `session3.diff` is just the 2026-08-31/09-01 work.

<details>
<summary>Historical: state at the end of session 2 (kept for reference)</summary>

| workload | before | after |
|----------|-------|-------|
| plain decode (`tg256`, cold cards) | 17.51 t/s | **29.83** (+70%) |
| plain decode, sustained (thermally limited) | — | 23–25 |
| **speculative decode (MTP)** | **32.94** | **50.1** (+52%) |
| MTP speedup over plain decode | 1.11x | **1.68x** |
| prompt processing at batch 7 (`pp512 -b 7 -ub 7`) | 61.08 | 88.87 (+45%) |

</details>


## Long context (added later; read this before the numbers below)

Everything below this section is measured at **2048 tokens of context**. The
machine's actual workload is **262144**, where prefill is a completely different
number because attention work is O(batch x depth).

| depth | tile kernel | cuBLAS-GEMM path (`bdcb3f7bf`) |
|---|---|---|
| 0 | 427 hot | 425 (gated off below KV 4096) |
| 65536 | 158.43 | **188.99** |
| 131072 | 111.22 | **131.08** |
| 262144 | ~61 (extrapolated) | ~77 (extrapolated, **never measured**) |

Cause: on Pascal there are no tensor cores, so long context falls to
`flash_attn_tile` at **3.55 TFLOPS of a 19.05 peak (18.6%)** beside a cuBLAS GEMM
doing 15.7. Only ~16 of 65 blocks have a growing KV cache
(`full_attention_interval 4`), so attention is the only context-scaling cost.

**Hard ceiling at 262144 is ~202 t/s** (105.6 TFLOP of attention per GPU per
2048-token batch = 5.54 s at full peak, plus ~4.6 s of context-independent work).
Target is 175, needing attention at ~70% of peak against ~25% today. Route and
arithmetic: OPTLOG attempt 90. Decode at depth is **unmeasured** -- it uses the
VEC kernel, which this path does not touch.

Everything here is general CUDA kernel work — nothing keys off this model or quant. Verified
quant-agnostic across q6_K, q3_K, q4_K, q5_K, q4_0, q5_0, q8_0, q2_K and iq4_nl. The norm,
elementwise, copy and flash-attention changes are architecture-general; the largest effects land on
GPUs that, like Pascal, lack an integer divider and DP4A.

## Contents

| path | what |
|------|------|
| `full-kernel.diff` | every kernel change against upstream `f280b2698` |
| `session2.diff` | the second session's changes alone (on top of `b44f8fe6f`) |
| `session3.diff` | the 2026-08-31/09-01 changes alone (on top of `5d1fafb01`) |
| `session4-longcontext.diff` | the cuBLAS-GEMM attention path alone (on top of `9183630c8`) |
| `RESUME-HERE.md` | **start here** -- current state and ranked next steps |
| `VERIFICATION.md` | numerical audit (covers up to `134a4f4a5`; later changes noted at the top) |
| `CORPUS.md` | why the perplexity gate corpus drifted, and which target goes with which file |
| `patches/` | the same as `git am`-able commits, in order |
| `commit-log.txt` | commit messages with per-file stats |
| `OPTLOG.md` | **every attempt, kept and reverted, with numbers** — the real record |
| `HANDOFF.md` | the first session's distilled notes |
| `bench/` | the benchmark harness actually used to iterate |
| `tools/` | the CPU sampling profiler and measurement recipes |

Reapply on a fresh checkout:

    git checkout f280b2698
    git am /path/to/p100-handoff/patches/*.patch

## Build, run, verify

    cmake -B build-opt -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-opt --config Release -j 14

Geometry is chosen in source now (`calc_nwarps` / `calc_rows_per_block` in `mmvq.cu`, under
`GGML_CUDA_MMVQ_PASCAL`); the old `-DP100_*` flags in `CMAKE_CUDA_FLAGS` are gone.

    # plain decode
    GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m <model> -sm tensor -fa 1 \
      -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5

    # speculative decode, tuned flags (the MTP head is inside the weights; no -md needed)
    GGML_CUDA_P2P=1 ./build-opt/bin/llama-speculative-simple -m <model> \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.05 -ngld 99 \
      -n 256 --temp 0 --top-k 1 --seed 42 -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p "..."

Correctness gates used at every step:

    ./build-opt/bin/test-backend-ops -o MUL_MAT -b CUDA0          # 1193/1193
    ./build-opt/bin/test-backend-ops -o FLASH_ATTN_EXT -b CUDA0   # 3949/3949
    ./build-opt/bin/llama-perplexity -m <model> -f ./ppl.txt -sm tensor -ngl 99 \
      -c 4096 -ctk q4_0 -ctv q4_0                                 # 2.7554 +/- 0.02151

## Numerical status — read this before trusting the gate

- **The single-token decode path is bit-identical to stock.** Perplexity is exactly
  2.7554 +/- 0.02151, the same value stock produces. An earlier revision added float4 loads to the
  norm *reductions*, which reordered the summation and moved it to 2.7565; that was reverted.
  Floating-point addition does not associate, so the reduction keeps the reference's ownership
  (thread t accumulates columns t, t+block_size, ... in that order) and only the register caching,
  which is order-preserving, was kept.
- **The multi-column path is not bit-identical.** Its reduction order changed (each warp now owns
  its own rows, so the cross-warp reduction is gone). It is numerically sound but not bit-equal.
- **The standard perplexity gate does not test the multi-column path.** It runs at batch 512, which
  routes through cuBLAS/MMQ and never touches that kernel. Gate it with `-b 7 -ub 7`:
  **3.6199 +/- 0.08383** against **3.6237 +/- 0.08411** for the stock geometry on the identical
  command — a 0.1% shift, well inside the error bar.
- `ppl.txt` is the llama.cpp README, not wikitext, which is why the absolute number does not match
  the 2.6209 in CLAUDE.md. Stock reproduces 2.7554 exactly on it, so the comparison is valid.

## Burst vs sustained — quote both

| condition | t/s |
|-----------|-----|
| tg256, cards cold (53/54 C) | 29.59 +/- 0.20 |
| tg256, cards at steady state (77/78 C) | 24.77 +/- 2.36 |
| tg2048, from steady state | 22.74 +/- 1.23 |

Under sustained load both cards hit the 175 W cap, then GPU1 hits `sw_thermal_slowdown` at 79 C and
falls to ~949 MHz against a 1328 MHz boost. GPU1 is consistently hotter and throttles harder than
GPU0, which points at airflow. With `-sm tensor` the two cards rendezvous every layer, so the
slower one paces both. Nothing in software addresses this; better airflow over GPU1 would recover
most of it, and the 175 W cap (against a 250 W default) then becomes the next limit.

MTP speed is also content-dependent — acceptance drives it:

| prompt | tuned flags | old flags (6 / 0.75) | acceptance |
|--------|-------------|----------------------|------------|
| C++ quicksort | **48.8–50.1** | 38.15 | 87.9% |
| explanation | 40.36 | 26.86 | 65.6% |
| prose | 37.68 | 22.05 | 58.3% |

## The findings that mattered

### 1. The q8_1 activation, not the weights, bounds `mul_mat_vec_q`
Within a warp the activation reads are 32-bit words 16 bytes apart inside a 36-byte `block_q8_1`,
then jump to the next block, so each of the eight loads per iteration fans out into many
transactions. Every block of the grid reads the *same* activation, so it is L2-resident: this costs
request throughput, not bandwidth. Probes (each keeps the traffic and deletes one thing):

| variant | mmvq total | vs baseline |
|---------|-----------:|------------:|
| baseline | 12.684 s | — |
| staging only (no dot product, no activation) | 10.446 s | -17.6% |
| dot product kept, activation replaced by a constant | 10.975 s | **-13.5%** |
| every dp4a deleted, all loads kept | — | -0.8% |

Staging the warp's contiguous run of q8_1 blocks into shared memory with coalesced `uint4` loads,
the same way the weights already were: **+5.4%**.

### 2. sm_60 has no integer divider
Several tail kernels did four to eight **64-bit divisions per element** of index maths, each
dozens of instructions. Replacing them with ggml's existing multiply-shift helpers
(`init_fastdiv_values` / `fastdiv` / `fast_div_modulo`) was worth more than anything left in the
matmul: `cpy_scalar` 4.85 -> 2.54 us, `k_bin_bcast` 3.07 -> 2.09 us, `concat_cont` 4.40 -> 3.95 us.
Worth checking on any older architecture.

### 3. A flash-attention bug, not Pascal-specific
`ggml_cuda_flash_attn_ext_vec_case_impl` passes `D` to `launch_fattn` as the KV batch size, but the
vec kernel steps its KV loop by `nthreads`. `launch_fattn` uses that number to cap how many blocks
may split the KV range, so wherever `nthreads < D` the parallelism is understated — Pascal runs 128
threads against `D` = 256, so it was halved. With a 256-long KV that left one KV tile, pinning the
split at 1 and running the whole attention on **12 blocks of a 56-SM GPU**. Passing `nthreads` is
simply the accurate value. Probably worth upstreaming.

### 4. Past one column, the activation cost is *volume*, not access pattern
Speculative decoding verifies several draft tokens in one batched forward, landing on
`mul_mat_vec_q` with `ncols_dst > 1` — 37% of GPU time in an MTP run, and untouched by all of the
above, which was gated to one column. Activation traffic is `nblocks x ncols x row_bytes`, so
**only more rows per block reduces it**; staging moves the same bytes and is worth ~1.3%.

The trap: rows appeared capped at 8 because the sweep varied `rows` at fixed warp count, which also
varied *rows per warp* — and rows-per-warp is what sets register pressure. Holding it at 4 and
scaling warps and rows together keeps REG at 182 while a block covers 16 rows:

| nwarps x rows (rows/warp) | pp512 (ub=7) |
|---------------------------|--------------|
| 4 x 2 (stock) | 61.08 |
| 1 x 4 | 79.92 |
| 2 x 8 (4) | 82.10 |
| **4 x 16 (4)** | **88.87** |
| 8 x 32 (4) | 89.20 |
| 2 x 12 (6) | 77.18 |

### 5. The MTP flags were tuned against the old kernel
`p-min` matters more than draft length. A high floor stops drafting early, so most rounds verify one
or two tokens and waste the batched forward:

| p-min (at n-max 4) | t/s | accept |
|--------------------|-----|--------|
| 0.95 | 34.72 | 96.4% |
| 0.75 (old default) | 40.21 | 93.2% |
| 0.4 | 48.09 | 83.2% |
| 0.05 | 48.42 | 78.2% |

Best: **`--spec-draft-n-max 3 --spec-draft-p-min 0.05`**. With `--temp 0 --top-k 1` speculative
decoding is exact, so these cost nothing in quality.

## What was tried and lost — do not repeat

`OPTLOG.md` has all of it with numbers. The expensive ones:

- **Five restructures of the one-column kernel** aimed at the residual activation cost: block-wide
  staging (28.21), warps owning distinct rows (29.19), chunked weight staging (27.12), 1 warp x 4
  rows (27.58), register-carried prefetch (28.74), against 29.32. The 1x4 result is the informative
  one — it moves a quarter of the activation bytes and is still 6% slower, so what binds there is
  warps resident per SM.
- **MMQ for the multi-column path.** Upstream tunes the mvq->MMQ crossover per architecture and its
  comment names the exact problem ("k-quants cost more to decode and mvq redoes that per column").
  On Pascal MMQ is **4x slower** (17.11 against 70.57): its tiles are arithmetic-dense and assume
  real DP4A. Upstream's default is right here.
- **The unpack-once refactor.** Deleting the weight unpack *entirely* is worth 5.7%, so hoisting it
  out of the column loop recovers ~4% of the kernel, ~1.4% end to end. Not worth a vec_dot
  interface change across every quant type. (Measured before building it — the probe cost 20
  minutes and saved hours.)
- **Register capping** via `__launch_bounds__`: 128 regs spills 104 bytes and gives 60.01; 80 regs
  spills 656.
- **Lowering vdr** to buy occupancy: 27.00 at vdr=2, 21.34 at vdr=1, against 30.11 at vdr=4.
- **CUDA graphs on Pascal**: no gain — the graph is re-captured roughly once per token because this
  model's node properties change every step, so it is never actually replayed.
- Aligned `get_int_b2` (four separate designs), `#pragma unroll` variants, double-buffering the
  shared stage, the row-outer loop nesting (REG *up* to 227), lifting the Pascal exclusion on mmvq
  GLU fusion, dropping the weight staging at multiple columns (52.88 against 70.52).

## Measurement pitfalls that cost real time

- **Probes that break correctness are invalid on a speculative workload.** Replacing the activation
  with a constant drove MTP acceptance to 0%, so every draft was rejected and the run collapsed to
  one-column stepping — the workload depends on the model being right. Use prompt processing, which
  does fixed work regardless.
- **A probe can conflate two effects.** "Delete the activation" removes the fan-out *and* the
  traffic. It said 3.7x was available; staging, which fixes only the fan-out, returned 1%.
- **Thermal state dominates small deltas.** Three *identical* back-to-back runs measured 29.32,
  27.97 and 25.16 while the cards saturated. Any A/B difference below ~0.5 t/s is inside that noise
  unless measured from a controlled thermal state.
- **Never run two GPU jobs at once here.** They OOM (25 GB model, 32 GB total) and corrupt each
  other's numbers.
- **nvprof inflates the gaps between kernels.** Read kernel durations from it, not utilisation.
- **`uintptr_t` round-trips lose the shared-memory address space** and ptxas silently emits generic
  loads instead of `LDS`. Derive aligned pointers with `char *` arithmetic from the original
  pointer.
- **Staging loops need a compile-time trip count and an explicit `__ldg`.** With a runtime bound
  ptxas emits predicated *generic* loads and the staging is pointless.

## Hardware facts (measured, not spec)

- Streaming ceiling **605 GB/s** per GPU (spec 732). By load width: 16 B = 604, 8 B = 598,
  4 B = 533, **2 B = 328**.
- Every ggml block size is 2 mod 4 (q6_K = 210, q8_0 = 34, q4_0 = 18) because each block carries a
  2-byte `ggml_half` beside a multiple-of-4 payload, so `get_int_b2` issues two 16-bit loads per
  quant word. The staging *preserves* the misalignment in shared memory rather than removing it,
  which is what keeps the global side aligned without repacking the weights.
- No DP4A, no IMAD (32x32->64 costs 4-6 XMADs), no SIMD-video (`__vsubss4` is 9 SASS instructions).
  Fast FP16 2:1, unique among Pascal.
- Nsight Compute does not support Pascal. nvprof does.
- `GGML_CUDA_P2P=1` is worth ~5%. `-sm tensor` massively beats `-sm layer` (30 against ~18).
- Keep ECC on: P100 HBM2 has dedicated ECC storage, costing ~115 MiB of 16384, not ~1 GB.

## Where the remaining headroom is

At ~50 t/s an MTP round is 75 ms: verify kernel 45 ms (55%), draft steps 4.5 ms, other kernels
~11 ms, launch/sync gaps ~15 ms. Everything in the verify kernel has been priced:

| component | worth at most |
|-----------|---------------|
| all arithmetic (dp4a + unpack) | ~13% of the kernel |
| activation access pattern (staging) | 1.3%, taken |
| activation **volume** | the remaining ~43%, and only rows-per-block reduces it |

Rows-per-block is capped by registers (REG:168 at 4 rows/warp, 12 warps/SM), and every way of
lowering registers costs more than it returns. Granting *all* the arithmetic for free — which no
real change achieves — puts MTP at about 53.5 t/s. **So ~53–55 t/s is the ceiling for this kernel
structure.** Going past it needs a design whose activation cost does not scale with block count:
many more rows per block, which means not holding `ncols x rows` floats in registers. That is a
redesign with genuinely uncertain payoff, not a tuning knob.

The other 15 ms is launch and sync overhead (~2150 kernel launches per token per GPU on the plain
path). CUDA graphs do not help here, so that means op fusion.
