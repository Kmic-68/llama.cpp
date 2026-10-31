# llama.cpp for 2x Tesla P100

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA work for Pascal (sm_60),
which upstream mostly leaves on generic paths. It is tuned for two
Tesla P100-PCIE-16GB cards, tensor-split, running Qwen3.8-27B Q6_K with a q4_0 KV cache. Other quant
types (Q5_K_M, Q4_K_M, IQ*, ...) get Pascal paths too.

It tracks upstream by merging. The last merge was upstream `f46bc30cb`
(2026-09-22), so current model architectures are supported.

## Results

| | upstream at the fork point | this fork |
|---|---|---|
| decode, `tg256` (no MTP) | 17.51 t/s | **33.2 t/s** |
| decode with MTP, 2k context | — | **54 t/s** |
| decode with MTP, 260k context | — | **29-35 t/s** |
| prefill, `pp2048` at 0 context | ~250 t/s | **493 t/s** |
| prefill at 260k context (vision loaded) | — | **123 t/s** filling, **153 t/s** for a question on a loaded context |
| perplexity (gate corpus, `-c 4096`) | — | **2.6101** |

[QUICKSTART.md](QUICKSTART.md) has the full table from 2k to 260k and the exact server command.

The speed didn't cost accuracy. The P100 multiplies in fp16 at twice its fp32 rate, and this fork
uses that everywhere the work is compute-bound (prefill matmuls, the MTP verify matvec, decode and
verify attention), but never accumulates long sums in fp16: partial sums move into fp32 every few
dozen values. Against an all-fp32 run, the prefill path now reads KLD 0.00119 where upstream's fp16
cuBLAS reads 0.00152, and fp32 itself scatters 0.0006-0.001 just from summing in a different order.
Perplexity 2.6101, all-fp32 2.6095; at this size perplexity can't separate them, KLD can. [CHANGES.md §11](CHANGES.md) has the method.

## Documents

| | |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | how to run it, and what each flag is for |
| [CHANGES.md](CHANGES.md) | every code change, grouped by subsystem, with what it measured |
| [FINDINGS.md](FINDINGS.md) | what worked, what failed, what transfers to other Pascal cards, and the measurement traps |
| [BUILD.md](BUILD.md) | building from source and verifying a build |
| [MOE.md](MOE.md) | large MoE models whose experts don't fit in VRAM (Flash-Next): the expert cache, threaded dispatch, and the config finder |
| [THREE-GPU.md](THREE-GPU.md) | measurements on three P100s from a contributor: NCCL, graphs, cancel latency |
| [`../OPTLOG.md`](../OPTLOG.md) | the full record: every attempt, kept or reverted, with numbers |

## What's in it, briefly

- **Decode matvec.** `mul_mat_vec_q` is rebuilt around one fact: the bottleneck is the q8_1
  activation that every block re-reads, not the weights.
- **Flash attention.** The kernel no longer converts the whole quantized KV cache to f16 on every
  call, which cost 4.15 ms per call at 262144 context. Tile widths fit the speculative batch
  exactly instead of padding it, and fp16 accumulation is folded to fp32 once per tile.
- **Long-context prefill.** Attention runs as two hand-written fp16 GEMM kernels with exact fp32
  folds, reading the q4_0 cache directly, register-renamed in SASS to avoid bank conflicts. The
  causal mask is per-row prefix lengths instead of an n_kv × n_tokens matrix, which saves ~1 GiB per
  card at 262k and lets vision run at `-ub 2048`. The tensor-parallel exchange overlaps the matmuls.
- **Tensor parallel.** Partials cross PCIe as f16 when that's lossless, on a dedicated copy stream.
- **fp16 math with fp32 accumulation.** Prefill matmuls (`gemm-fold.cu`), the 2-5 token verify
  matvec (`mmvq-f16.cu`) and decode/verify attention (`fattn-q4p.cuh`) multiply on the fp16 pipe
  and fold their sums into fp32, with quantized values entering as exact integers.
- **Every other quant type.** Pascal single-token dot products and fp16 multi-token matvecs for all
  weight types, not just Q6_K. Q5_K_M with MTP: 56-64 t/s for one request (was 34-40), 30-38 t/s
  each for two at once (was 7-8). Bit-identical or more accurate than before in every tested case.
  [CHANGES.md §15](CHANGES.md).
- **MTP speculative decoding** with sampled drafts and the lossless speculative-sampling rule, a
  fixed-width verify, a draft length that follows the context depth, and a K/V-only catch-up.

## Caveats

This is a personal fork, not an upstream contribution, and none of it has been through upstream
review. Some changes are specific to this model's shape (head size 256, GQA ratio 6). They're
gated so other shapes take the stock path, which means they're untested elsewhere, not proven
safe there.

The commit messages are AI-written and say so in their trailers. llama.cpp's `AGENTS.md` forbids
that for upstream contributions, so anything proposed upstream would need a human author.

The fork shipped two data races of its own making for weeks before they were found. Both are
fixed, and both passed the full `test-backend-ops` suite the whole time. [FINDINGS.md](FINDINGS.md)
explains why the suite can't catch that class of bug.

## License

MIT, same as upstream. See [LICENSE](../LICENSE).
