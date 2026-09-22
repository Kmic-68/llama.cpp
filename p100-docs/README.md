# llama.cpp for 2x Tesla P100

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA work for Pascal (sm_60),
which upstream mostly leaves on generic paths. It is tuned on one machine: two
Tesla P100-PCIE-16GB cards, tensor-split, running Qwen3.8-27B Q6_K with a q4_0 KV cache.

It tracks upstream by merging. The last merge was upstream `f46bc30cb`
(2026-09-22), so current model architectures are supported.

## Results

| | upstream at the fork point | this fork |
|---|---|---|
| decode, `tg256` | 17.51 t/s | **30.6 t/s** |
| prefill, `pp2048` at `-ub 512` | 222.6 t/s | **380.4 t/s** (before the 2026-09-22 merge) |
| decode at 229k context | — | 21.5 t/s plain, 23.2 with MTP |
| perplexity (gate corpus, `-c 4096`) | — | **2.6101 ± 0.0198** (band 2.6209 ± 0.0199) |

The speed didn't cost accuracy. Measured against an all-fp32 run, this fork's prefill matmuls are
slightly *closer* to fp32 than upstream's (+0.00343 nats/token against +0.00414). Upstream's
default cuBLAS algorithm on Pascal is both slower and less accurate.

## Documents

| | |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | how to run it, and what each flag is for |
| [CHANGES.md](CHANGES.md) | every code change, grouped by subsystem, with what it measured |
| [FINDINGS.md](FINDINGS.md) | what worked, what failed, what transfers to other Pascal cards, and the measurement traps |
| [BUILD.md](BUILD.md) | building from source and verifying a build |
| [`../OPTLOG.md`](../OPTLOG.md) | the full record: every attempt, kept or reverted, with numbers |

## What's in it, briefly

- **Decode matvec.** `mul_mat_vec_q` is rebuilt around one fact: the bottleneck is the q8_1
  activation that every block re-reads, not the weights.
- **Flash attention.** The kernel no longer converts the whole quantized KV cache to f16 on every
  call, which cost 4.15 ms per call at 262144 context. Tile widths fit the speculative batch
  exactly instead of padding it, and fp16 accumulation is folded to fp32 once per tile.
- **Long-context prefill** uses a cuBLAS-GEMM attention path.
- **Tensor parallel.** Partials cross PCIe as f16 when that's lossless, on a dedicated copy stream.
- **Pascal fp16 matmuls** request cuBLAS `ALGO6`, which is 10x more accurate and faster at the
  common sizes.

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
