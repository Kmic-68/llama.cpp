# Resume point — long context is the live problem; short context is done

All work is committed. Tree is clean apart from your own `CLAUDE.md` edit and
untracked `ppl.txt` / `p100-handoff/`.

## READ THIS FIRST: prefill is a curve, and the target is 175 t/s at 262144

Every number in this file below this section is measured at **2048 tokens of
context**, where flash attention is 2.1% of prefill. **That is not the regime this
model is used in.** The user runs 262144. Prefill degrades with depth because
attention work is O(batch x depth) -- this is physics, not a defect:

| depth | before (tile kernel) | now (GEMM path) |
|---|---|---|
| 0 | 434 / 427 hot | 425 (unchanged -- path is gated off below KV 4096) |
| 32768 | 278 | not re-measured |
| 65536 | 158.43 | **188.99** (+19.3%) |
| 131072 | 111.22 | **131.08** (+17.9%) |
| 262144 | ~61 (extrapolated) | **~77 (extrapolated, NEVER MEASURED)** |

**Hard ceiling at 262144 is ~202 t/s** -- attention there is 105.6 TFLOP per GPU
per 2048-token batch, which is 5.54 s even at this card's full 19.05 TFLOPS, plus
~4.6 s of context-independent work. See OPTLOG attempt 90 for the table.

**Current target: 175 t/s at 262144**, which needs attention at ~70% of peak; it
is at ~22-25% now. Route is OPTLOG attempt 90's "Next, in order" list, starting
with the PV GEMM in f16 (~70% of attention compute, currently fp32 at 6.5 TFLOPS
against 13.1 available). Also open: **decode at 262144 has never been measured**,
and it takes the VEC kernel, which the new path does not touch.

## Where things stand at SHORT context (2048)

| metric | start of session | now |
|---|---|---|
| pp2048 (`-b 2048 -ub 2048`) | 372.5 | **442.6** cold / ~437 hot |
| tg256 (CLAUDE.md metric cmd) | 29.8 | **32.1** best / ~31.8 typical |
| MTP decode (n-max 4, p-min 0.2) | 48.8 | **54.5** |
| perplexity (ppl-orig.txt) | 2.6209 | **2.6214 +/- 0.01995** |

Against CLAUDE.md's original 17.51 t/s decode baseline that is **1.83x**.

Goals were 450 t/s prefill and 60 t/s MTP. **Neither was met**: prefill landed
at 442.6 (98.4%), MTP at 54.5 (90.8%). Both remaining gaps are structural --
see "What is left" below, which says exactly what stands in the way and how much
each is worth.

## Committed this session

Six code commits (+ eleven docs/log commits), `5d1fafb01..f85e154ed`:

| commit | what | gain |
|---|---|---|
| `58c8a73ed` | vectorised q6_K dequant | +0.7% pp |
| `a4d1103c5` | **concurrent bidirectional peer copies** | **+12.2% pp** |
| `f8edbf816` | cuBLAS ALGO3 for wide f16 GEMMs | +1.8% pp |
| `e83a7913a` | **f16 all-reduce** + pipelined delta-net reduction | +3.2% pp |
| `ed42ad15d` | delta-net addressing walked, not recomputed | below noise here* |

Later session (long context), `9183630c8..98de4588f`:

| commit | what | gain |
|---|---|---|
| `bdcb3f7bf` | **cuBLAS-GEMM flash attention for pre-Volta** | **+17.9% @ d=131072, +19.3% @ d=65536**, 0 at short context |
| `98de4588f` | alias P onto S in that path | -50 MB scratch |

`bdcb3f7bf` adds `ggml/src/ggml-cuda/fattn-gemm.{cu,cuh}` and touches
`fattn.cu` (dispatch + `get_alloc_size`). On by default for `cc < VOLTA`;
`GGML_CUDA_FA_GEMM=0` restores upstream behaviour. Gated to KV >= 4096 so short
context is provably unaffected (425.19 on vs 427.05 off, within noise).

\* -29% on the kernel at head_count=4, -1.1% at head_count=32; this model runs
in the regime where it hides behind warp parallelism. Kept because it is
bit-exact and never slower.

Code touched, whole session:

| file | +/- |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | peer-copy stream, f16 all-reduce, ALGO3 |
| `ggml/src/ggml-cuda/common.cuh` | copy stream, work event, staging buffers |
| `ggml/src/ggml-cuda/convert.cu` | vectorised q6_K dequant |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | fused reduction, walked addressing, nwarps knob |

Nothing outside `ggml/src/ggml-cuda/` was modified.

**The big one is `a4d1103c5`.** The tensor-parallel all-reduce was serialising
its two directions: copy 0->1 went on GPU0's compute stream and GPU1's compute
stream was then made to wait on it, so copy 1->0 -- issued on GPU1's compute
stream -- queued behind that wait. nvprof showed it as a 4358us gap after every
PtoP copy, exactly one copy duration, 86% of all idle time. PCIe here is full
duplex (9.74 GB/s *each way* simultaneously), so the opposing copy was free and
we were paying full price for it. This one also carries most of the decode and
MTP gains, since decode all-reduces are small but latency-dominated.

## Numerical status

**Five of the six kept changes are bit-exact** and were each verified to
reproduce the previous build digit-for-digit (chunk [1] and all 30 chunks).

Only **ALGO3 (`f8edbf816`) is not**: a different cuBLAS kernel means a different
f16 k-accumulation order. It moved perplexity 2.6209 -> 2.6214, i.e. 0.03 sigma,
well inside the required 2.6010-2.6408 band, with mixed per-chunk direction.
Reverting it is a one-line `if` in `ggml-cuda.cu` if you ever want strict
bit-parity with upstream, at the cost of ~1.8%.

The f16 all-reduce is lossless *here* and does not assume it: a one-time runtime
probe checks every element of the first exchange for f16-exactness and latches
the answer, and that probe exchange itself still goes uncompressed, so a model
whose partials are not f16-exact never sees a lossy copy. It logs
"tensor-parallel partials are f16-exact; peer copies will be sent as f16".

All three `P100_*` probe switches in `vecdotq.cuh` are back at **0** (one was
used for a measurement in attempt 85) and correctness was re-verified after.

## Measure like this

    GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m <model> \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 2048 -n 0 -b 2048 -ub 2048 -r 3

**`-ub 2048` matters a lot** and belongs in your server flags, not just the
bench. Swept: 512 is catastrophic (cuBLAS takes the same time for n=512 as
n=1024 -- wave quantisation), 3072 is bad (non-power-of-2), 4096 is slightly
worse than 2048.

**Re-baseline from cold.** Note this is *model-level* drift (many kernels, PtoP,
barriers) -- the GEMM kernel itself does not throttle at all. Same build, same
command, three readings:
442.59 at 39 C, 438.49 at 55 C, 434.10 starting at 52 C and ending at 66 C.
That is a 2% spread from temperature alone. Anything under ~2% is not a code
delta -- always compare at the same starting temperature.

## What is left, and what is not

Per-GPU profile at 442 t/s: GEMM 71.0%, PtoP 7.1%, gated_delta_net 7.0%,
converts 3.0%, flash-attn 2.1%, q6_K dequant 1.9%, rms_norm 1.9%, all-reduce ADD
1.3%, idle 2.2%. GEMM-only ceiling ~620 t/s.

**That profile is at 2048 context, and the user runs 262144.** Only ~16 of the
65 blocks carry a growing KV cache (`full_attention_interval 4`); the other ~49
are gated delta net with constant state. So flash-attn is the **only** cost on
this model that scales with context length — everything else in that profile is
context-independent. At 2048 it is 2.1% and not worth touching; at 262k it should
dominate, and the 2.1% figure is actively misleading if quoted at long context.
**DONE -- see OPTLOG attempts 89 and 90 and the section at the top of this file.**
Profiled at d=65536: flash_attn_tile was 37.1% of GPU time, one call doing
1.65 TFLOP in 465 ms = **3.55 TFLOPS, 18.6% of the 19.05 peak**, beside a cuBLAS
GEMM doing 15.7. Fixed by replacing the path (attempt 90), not by tuning it
(attempt 89: 15 configs, stock wins all).

Ranked by what is actually still available (all context-independent, so the
above outranks them at 262k):

1. **Fuse the all-reduce widen into the ADD** (~+1%). Today the f16 partial is
   widened to f32 into `node_tmp` (1.4%) and then the meta backend's ADD reads
   it back (1.3%). One "accumulate f16 into f32" kernel would drop a whole
   41.9 MB pass. Needs a new accumulating-copy path in `ggml-backend-meta.cpp`
   so the ADD node can be skipped -- that is why it was not done.
2. **gated_delta_net** (7.0%). Resisted three attacks: block width, reduction
   fusion (+0.8%, kept), deeper load pipelining (negative -- registers 47->53
   costs more occupancy than it wins). Measured issue efficiency ~15% of peak,
   so it stalls on something that is neither the reduction critical path nor
   occupancy. Nsight Compute would answer this; it does not support Pascal.
   nvprof metrics (`--metrics stall_memory_dependency,stall_exec_dependency`)
   is the next tool to reach for.
3. **Overlap each weight dequant with the previous matmul's GEMM** (~+1.9%).
   dequant is memory-bound and the GEMM is compute-bound, and dequant(W2) does
   not depend on GEMM(W1) -- but they are on one stream and ggml executes node
   by node, so this needs graph lookahead the backend does not expose today.
4. Per-shape cuBLAS algo: ~+0.3%, measured, judged not worth the risk.

The GEMM runs at ~15.9 TFLOPS in-model against 16.80 standalone -- a uniform
**3.2%** gap worth ~2.3 points. An earlier draft of this file blamed thermal
throttling; **that was wrong and is retracted**. A 170 s pure-GEMM run holds
1328 MHz and 16.81 TFLOPS all the way to 73 C, hotter than the model ever gets.
Seven causes have been tested and eliminated with standalone reproductions
(throttling, dual-GPU load, nvprof overhead, preceding write traffic, pointer
alignment, cold buffers, VMM mapping) -- see OPTLOG attempt 84. The cause is
still unknown; it is real and it is not thermal.

**Dead ends, measured -- do not re-litigate:**
- MMQ on Pascal (no DP4A, ~4x ALU disadvantage).
- f32 GEMM output to remove the widen: 7.65 vs 16.8 TFLOPS. Halves the GEMM.
- NN weight layout: real (+10%) but needs a transposing dequant; ALGO3 gets the
  same for one line.
- lda padding: helps gate/up only, ~0 elsewhere.
- Chunking the GEMM along m: strictly worse.
- `-sm layer` for prefill: 226.9 vs 438.5.
- Reduce-scatter/all-gather instead of full-exchange all-reduce: identical
  traffic for 2 GPUs.
- **CUDA graphs on Pascal**: they do engage if you lift the `cc < VOLTA` guard,
  and give nothing (MTP 54.48 -> 54.03, tg256 32.12 -> 31.25). The workload
  streams weights; it is not launch-bound. Upstream's exclusion is correct.
- MMVQ rows-per-block 16 -> 8 for the multi-column path: no effect.
- `-sm layer` for decode: 20.37 vs 32.12. Tensor split wins both phases.

## Decode (60 t/s goal still open)

Single-token **32.1 t/s** best, ~31.8 typical. MTP re-measured after this
session's changes with `p100-handoff/tools/mtp-bench.sh <n-max> <p-min>`.
**The whole flag space is now swept** -- n-max, p-min, n-min, backend-sampling,
split mode, cache types:

| n-max | p-min | n-min | t/s | accept |
|---|---|---|---|---|
| 2 | 0.05 | 0 | 48.92 | 89.2% |
| 3 | 0.05 | 0 | 52.90 | 87.9% |
| **4** | **0.2** | **0** | **54.48** | 78.2% |
| 4 | 0.05 | 0 | 53.79 | 78.2% |
| 4 | 0.2 | 1 | 54.27 | 78.2% |
| 4 | 0.2 | 4 | 54.38 | 78.2% |
| 5 | 0.05 | 0 | 53.02 | 71.9% |
| 5 | 0.2 | 2 | 54.00 | 71.9% |
| 6 | 0.2 | 3 | 50.32 | 67.8% |
| 6 | 0.75 | 0 | 41.52 | 83.8% |

**Use `--spec-draft-n-max 4 --spec-draft-p-min 0.2`** -- the old default of
n-max 3 leaves ~2% on the table. 48.8 -> 54.5 is +11.7%, and the peer-copy fix
(`a4d1103c5`) is most of it.

`--spec-draft-n-min` does nothing. `--spec-draft-backend-sampling` is **inert
under `-sm tensor`** (54.12 vs 53.59, and the "not supported with
SPLIT_MODE_TENSOR" warning fires either way) -- which matters, because that is
exactly the flag that would have addressed the host round trip below.

The curve is flat-to-falling past n-max 4: accept rate decays faster than the
extra speculated tokens pay for themselves.

### MTP is host-sync bound -- but do NOT start here (see OPTLOG attempt 88)

> **This whole subsection is superseded.** Attempt 88 priced the host round trip
> end-to-end instead of trusting nvprof. Default sampling (`top_k 40 / top_p 0.95
> / min_p 0.05 / temp 0.8`) measures **within noise of `--temp 0 --top-k 1`**
> (52.07/52.32 vs 51.82/51.76 t/s), so the host-sampling term is not a
> bottleneck. GPU-side sampling is worth **4-8%, landing 57-59 — not 60** — and
> needs axis-1 split rules for ARGMAX/TOP_K/SOFT_MAX/GET_ROWS in
> `ggml-backend-meta.cpp`, which is the silent-wrong-answer risk tier.
> **Dropped.** Mirroring `output.weight` to sidestep it is a trap: it doubles
> output-head memory traffic (995 MiB read per GPU instead of 497) and costs
> +497 MiB/GPU. Net loss. The text below is kept for its profiling detail only.
>
> One useful consequence: every MTP number in these docs was measured greedy,
> and attempt 88 shows that **transfers to real-world default sampling
> unchanged**.

Steady-state MTP decode is **24% idle**, and **14.5% of wall is the host round
trip** (95 DtoH events with 1.46 ms of GPU idle after each, plus 817 HtoD):
logits to the CPU, speculative accept/reject there, tokens back. Removing it
gives 54.5/(1-0.145) = **63.7 t/s, past the 60 target**. So 60 is reachable, but
through the decode *pipeline*, not kernel tuning:

1. ~~GPU-side sampling~~ -- **dropped, attempt 88.** For the record, the gate at
   `llama-context.cpp:1216` is load-bearing, not conservative: `output.weight` is
   `SPLIT_AXIS_1` (`llama-model.cpp:566`), i.e. vocab-sharded, so neither GPU
   holds complete logits, while the backend samplers do `ggml_reshape_1d` then
   `ggml_argmax`/`ggml_top_k` over the whole vocab (`llama-sampler.cpp:1084`,
   `:1484`). Each GPU would reduce over its own shard and return a local index.
2. Overlap host verification with GPU work in the speculative loop.

**Caveat: nvprof costs MTP ~11% (48.65 profiled vs 54.5 not), and host-sync gaps
are exactly where that overhead lands.** Re-measure with CUDA events inside the
decode loop before building on 14.5%; true headroom is probably 5-10%.

**Second MTP idea, measured and costed:** the weight block's load and
6-bit unpack are redone once per column, five times over at ncols_dst=5. Priced
with the `P100_NOUNPACK` probe: `mul_mat_vec_q<ncols=5>` 95.717 -> 87.629 us
(-8.5%), tg256 32.12 -> 33.79 (+5.2%). Hoisting it recovers 4/5 of that,
**~+3.5% MTP -> ~56.4 t/s** -- real, but still short of 60, and nothing for
single-token decode where there is only one column. Swapping the loop nest to
make the weight work loop-invariant does *not* let nvcc capture it (both loops
are fully unrolled, so order is irrelevant to CSE; measured, reverted).
Capturing it needs a hand-written multi-column `vec_dot_q6_K_q8_1` plus dispatch
-- bit-exact by construction, but a rewrite of the hottest kernel in the build.

Profiled: **52% of MTP GPU time is `mul_mat_vec_q<ncols=5>`**, 95.7us per call,
moving Q6_K weights at ~383 GB/s -- roughly 75% of achievable HBM bandwidth on
this card. The remaining 25% is the sm_60 dp4a emulation (8 instructions, already
bit-exact and heavily optimised by earlier sessions). So **60 t/s needs a shape
change, not tuning**: the draft head itself, or a decode kernel that does not
re-read the weights per speculated token.

## Still open from the earlier audit (unchanged, none fixed)

- `cudaMalloc`/`cudaFree` on the mmvq hot path
- q8_1 activation cache keyed `[device]` where pools are `[device][stream]`
- three live probe switches in `vecdotq.cuh`
- `ggml-cuda.cu` `&& tensor->data != nullptr` -- undocumented

## Corpus warning

Use `p100-handoff/ppl-orig.txt` (420,098 bytes) with the 2.6209 target. `./ppl.txt`
is a *different* document (422,246 bytes) whose target is 2.7554. See CORPUS.md.
