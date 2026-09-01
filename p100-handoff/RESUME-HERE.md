# Resume point — prefill 442 t/s, decode 31.7 t/s

All work is committed. Tree is clean apart from your own `CLAUDE.md` edit and
untracked `ppl.txt` / `p100-handoff/`.

## Where things stand

| metric | start of session | now |
|---|---|---|
| pp2048 (`-b 2048 -ub 2048`) | 372.5 | **442.6** cold / 434.1 hot |
| tg256 (CLAUDE.md metric cmd) | 29.8 | **31.7** |
| MTP decode (best settings) | 48.8 | **54.0** |
| perplexity (ppl-orig.txt) | 2.6209 | **2.6214 +/- 0.01995** |

Against CLAUDE.md's original 17.51 t/s decode baseline that is **1.81x**.

Prefill goal was 450; landed at 442.6, i.e. 98.4% of it.

## Committed this session

| commit | what | gain |
|---|---|---|
| `58c8a73ed` | vectorised q6_K dequant | +0.7% |
| `a4d1103c5` | **concurrent bidirectional peer copies** | **+12.2%** |
| `f8edbf816` | cuBLAS ALGO3 for wide f16 GEMMs | +1.8% |
| `e83a7913a` | **f16 all-reduce** + pipelined delta-net reduction | +3.2% |

The big one is `a4d1103c5`. The tensor-parallel all-reduce was serialising its
two directions: copy 0->1 went on GPU0's compute stream and GPU1's compute
stream was made to wait on it, so copy 1->0 queued behind that wait. nvprof
showed it as a 4358us gap after every PtoP copy -- exactly one copy duration,
86% of all idle time. PCIe here is full duplex (9.74 GB/s *each way*
simultaneously), so the second copy was free and we were paying full price.

## Numerical status

Four of the five kept changes are **bit-exact** and were each verified to
reproduce the previous build digit-for-digit (chunk [1] and all 30 chunks).
Only **ALGO3 (`f8edbf816`) is not**: a different cuBLAS kernel means a different
f16 k-accumulation order. It moved perplexity 2.6209 -> 2.6214, i.e. 0.03 sigma,
well inside the required 2.6010-2.6408 band, with mixed per-chunk direction.
Reverting it is a one-line `if` in `ggml-cuda.cu` if you ever want strict
bit-parity with upstream at the cost of ~1.8%.

The f16 all-reduce is lossless *here* and does not assume it: a one-time runtime
probe checks every element of the first exchange for f16-exactness and latches
the answer, and that probe exchange itself still goes uncompressed. It logs
"tensor-parallel partials are f16-exact; peer copies will be sent as f16".

## Measure like this

    GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m <model> \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 2048 -n 0 -b 2048 -ub 2048 -r 3

**`-ub 2048` matters a lot** and belongs in your server flags, not just the
bench. Swept: 512 is catastrophic (cuBLAS takes the same time for n=512 as
n=1024 -- wave quantisation), 3072 is bad (non-power-of-2), 4096 is slightly
worse than 2048.

**Re-baseline from cold.** The same build, same command, three readings:
442.59 at 39 C, 438.49 at 55 C, 434.10 starting at 52 C and ending at 66 C.
That is a 2% spread from temperature alone. Anything under ~2% is not a code
delta -- always compare at the same starting temperature.

## What is left, and what is not

Per-GPU profile at 442 t/s: GEMM 71.0%, PtoP 7.1%, gated_delta_net 7.0%,
converts 3.0%, flash-attn 2.1%, q6_K dequant 1.9%, rms_norm 1.9%, all-reduce ADD
1.3%, idle 2.2%. GEMM-only ceiling ~620 t/s.

Ranked by what is actually still available:

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
3. Per-shape cuBLAS algo: ~+0.3%, measured, judged not worth the risk.

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

## Decode (60 t/s goal still open)

Single-token 31.7 t/s. MTP re-measured after this session's changes with
`p100-handoff/tools/mtp-bench.sh <n-max> <p-min>`:

| n-max | p-min | t/s | accept |
|---|---|---|---|
| 2 | 0.05 | 48.92 | 89.2% |
| 3 | 0.05 | 52.90 | 87.9% |
| **4** | **0.2** | **54.04** | 78.2% |
| 4 | 0.05 | 53.79 | 78.2% |
| 5 | 0.05 | 53.02 | 71.9% |
| 6 | 0.75 | 41.52 | 83.8% |

**Use `--spec-draft-n-max 4 --spec-draft-p-min 0.2`** -- the old default of
n-max 3 leaves ~2% on the table. 48.8 -> 54.0 is +10.7%, and the peer-copy fix
(`a4d1103c5`) is most of it.

That sits right at the 53-55 t/s structural ceiling estimated for the current
kernel shape, and the curve is flat-to-falling past n-max 4 (accept rate decays
faster than the extra tokens pay). **60 t/s needs a shape change, not tuning** --
the draft head itself, or a batched-decode kernel that does not re-read the
weights per speculated token.

## Still open from the earlier audit (unchanged, none fixed)

- `cudaMalloc`/`cudaFree` on the mmvq hot path
- q8_1 activation cache keyed `[device]` where pools are `[device][stream]`
- three live probe switches in `vecdotq.cuh`
- `ggml-cuda.cu` `&& tensor->data != nullptr` -- undocumented

## Corpus warning

Use `p100-handoff/ppl-orig.txt` (420,098 bytes) with the 2.6209 target. `./ppl.txt`
is a *different* document (422,246 bytes) whose target is 2.7554. See CORPUS.md.
