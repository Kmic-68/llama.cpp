# Handoff

Where the work stands, and what's worth doing next. For the project rules and gates, see
`CLAUDE.md`. For the results and changes, see `p100-docs/`.

## State (2026-09-22)

- Branch `p100-optimizations`, merged with upstream `f46bc30cb`. See CHANGES §9 for what the
  merge touched and how it was checked.
- `tg256` 30.6 t/s (upstream at the fork point: 17.51). Perplexity 2.6101 ± 0.0198 on the gate corpus.
- The release bundle at `/mnt/fast/p100-llamacpp-release` is built from this HEAD.
  `diffs/HEAD-SHA.txt` there is authoritative.
- 176 attempts are logged in `OPTLOG.md`. Its CLOSING SUMMARY (line ~1896) is from 2026-09-01 and
  predates the long-context work. The later attempts are the current story.

## Open threads

1. **Register-resident q4_0 attention for decode at depth.** The largest remaining win. At 229k
   context attention is half of each token (23.7 of 46.6 ms), and q4_0 runs at 99 GB/s where f16
   hits 480, so the cost is dequant plus the shared-memory round trip. A kernel that dequantizes K
   into registers and accumulates all columns per thread would attack it. That's a new kernel.
2. **`GGML_CUDA_DEVICES` above the physical GPU count isn't reproducible** (NaN in 4 of 8 runs at
   3 virtual devices). It follows the GEMM attention path. It's debug-only, and two physical GPUs
   are bit-stable. OPTLOG attempt 153 §8c.
3. **Fuse the all-reduce widen into the ADD** (~+1% prefill). It needs an accumulating-copy path
   in `ggml-backend-meta.cpp`.
4. **`gated_delta_net`** is 7% of prefill and at ~15% issue efficiency. It resisted three attempts,
   and there's no profiler here to say why.
5. **Deepest prefill regressed ~10%** (95.1 → 85.4 t/s at `-d 262144`). Possibly thermal; not
   bisected.
6. **A smaller flash-attention build.** Upstream's default `GGML_CUDA_FA_QUANTS` covers the serving
   configuration and would shrink the CUDA library, which is VRAM on each card. It's unmeasured.

## Closed: don't re-sweep without new information

| axis | result |
|---|---|
| attention occupancy (384 threads, occupancy 2-4, doubled warps) | neutral or worse, three ways |
| `nbatch_K` 64/128/256 | 128 is best on narrow tiles; 256 is +44% |
| `nbatch_fa` 32/64/128 | 64 |
| Q-column reuse (`cpw` 1 vs 2) | identical to 0.005% |
| wide loads in the q4_0 dequant | scalar wins by 7% |
| mmvq register caps, unrolling, prefetch pipelines | all worse; not latency-bound |
| double-buffered mmvq staging | shared memory then limits occupancy |
| internal AllReduce on Pascal | −17% (PCIe) |
| MMQ on Pascal | no DP4A, ~4x ALU disadvantage |
| `-sm layer`, for prefill or decode | tensor split wins both |
| f32 GEMM output | halves GEMM throughput |

## Notes for kernel work

- The mmvq geometry and staging live in `mmvq.cu`, under `GGML_CUDA_MMVQ_PASCAL`. Build times:
  `mmvq.cu` alone is ~90 s, while touching `vecdotq.cuh` rebuilds ~200 instances (~25 min).
  Put sweep knobs in the source, not in `-D` flags.
- Fast isolated benchmark (a few seconds):
  `test-backend-ops perf -o MUL_MAT -b CUDA0 -p 'type_a=q6_K,type_b=f32,m=4096,n=1,k=14336'`.
  Always confirm on the real model. The isolated shape has pointed the wrong way before.
- Casting a shared-memory pointer through `uintptr_t` loses the address space, and ptxas silently
  emits generic loads. Derive aligned pointers with `char *` arithmetic.
- Correctness harnesses and proofs are in `p100-handoff/tools/` (bit-exactness replays for the
  fastdiv, DP4A, q4_0 dequant and norm changes). `p100-handoff/VERIFICATION.md` is the numerical
  audit.
