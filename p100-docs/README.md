# llama.cpp for 2x Tesla P100 (sm_60)

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA kernel work aimed at
Pascal, which upstream largely leaves on generic paths. Branched from `f280b2698` (2026-08-25).

Measured on 2x Tesla P100-PCIE-16GB, tensor-split, Qwen3.8-27B Q6_K, q4_0 KV cache:

| | upstream | here |
|---|---|---|
| decode `tg256` | 17.51 t/s | **30.64 ± 0.19** (1.75x) |
| prefill `pp2048` | 222.6 t/s | **411.5** (1.85x) |
| decode @ 229k context | — | **21.5 t/s** plain, 23.2 with MTP |
| perplexity, `-c 4096` | — | 2.6097 ± 0.0198 (gate band 2.6209 ± 0.0199) |
| `test-backend-ops` | — | FLASH_ATTN_EXT 3961/3961, full suite 14593/14593, both GPUs |

Accuracy is *better* than upstream, not traded for the speed: against an all-fp32 reference,
upstream's prefill matmuls sit at +0.00414 nats/token and this fork at +0.00343, because upstream's
default cuBLAS algorithm on Pascal picks a long-chain fp16 accumulator that is both less accurate
and slower than `ALGO6` at 512-1024 rows.

## Read these

| | |
|---|---|
| **[CHANGES.md](CHANGES.md)** | **every code change, grouped by subsystem, with what it measured** |
| [QUICKSTART.md](QUICKSTART.md) | the flags to run, and why each one is there |
| [FINDINGS.md](FINDINGS.md) | what worked, what failed, and how the measurements lied |
| [BUILD.md](BUILD.md) | building it |
| [COMMUNITY-NOTES.md](COMMUNITY-NOTES.md) | what generalizes to other Pascal cards, what does not |
| [`../OPTLOG.md`](../OPTLOG.md) | the full record — 153 attempts, kept and reverted, with numbers |
| [`../p100-handoff/`](../p100-handoff/) | the raw engineering record: harnesses, logs, per-session notes |

## Reproducing the numbers

Every measurement above comes from `tools/gate.sh`, which is in this repo:

    ./tools/gate.sh

It runs the perplexity and throughput gates with the corpus they belong to. **Use it rather than
a hand-typed perplexity command** — the band belongs to one specific corpus, and a different wiki
dump returns ~2.7566 on *any* build including stock llama.cpp. That mix-up has twice been misread
here as a correctness failure, which is why the gate lives in a script instead of in prose.

The numerical-accuracy claims come from `test-backend-ops` against its CPU fp32 reference; the
per-token accuracy comparisons are paired per-chunk perplexity over 4096 x 30 tokens, with the
t-statistics reported alongside them in [CHANGES.md](CHANGES.md) and [FINDINGS.md](FINDINGS.md).

Two measurement traps will bite you if you benchmark this yourself: cold-start and thermal skew
reach 13% across sessions (discard a warmup run, interleave A/B within one session), and `-n 128`
is far too short at long context. Both are in [FINDINGS.md](FINDINGS.md).

## The short version of what's in it

The decode matvec (`mul_mat_vec_q`, ~85% of decode time) is rebuilt around the observation that
the **q8_1 activation, not the weights, is the bottleneck** — it is re-read by every block.
Flash attention stops **re-converting the whole quantized KV cache to f16 on every call**, which
was costing 4.15 ms per call at 262144 context; tile widths are made to fit the speculative batch
exactly instead of padding 37% of the work; and fp16 accumulation over a quarter-million adds is
folded to fp32 per tile, which is 8.7x more accurate at depth for ~2%. There is a cuBLAS-GEMM
attention path for long-context prefill, CUDA graphs enabled on Pascal, and a tensor-parallel
all-reduce that ships partials as f16.

## Honesty

**This is a private fork made public, not an upstream contribution**, and nothing here has been
through upstream review. Several changes are specific to this model shape (head size 256, GQA
ratio 6) and are gated so other configurations take the stock path — they are untested elsewhere,
not proven safe there.

**The commit messages are AI-written**, and the commits say so in their `Co-Authored-By` trailers.
llama.cpp's `AGENTS.md` prohibits AI-written commit messages and PR descriptions for
contributions, and requires that a contributor understand and defend a change without AI
assistance. Private forks are exempt, which is what this is. Anything proposed upstream from here
would need rewriting by a human author who can defend it.

**Two data races shipped in this fork for weeks before being found**, both introduced by its own
optimization work, both now fixed, and both of which passed the full 14593-case `test-backend-ops`
suite the entire time. That suite runs ops one at a time with host synchronization between them,
which is exactly the condition under which a cross-stream race does not occur. FINDINGS has the
detail; it is the most useful thing here for anyone doing similar work.

**Two fixes ship on inspection with no experiment behind them, and one defect is known and
unexplained** (`GGML_CUDA_DEVICES` above the physical GPU count is not reproducible). See "Known
gaps" in CHANGES.md. Neither affects a normal two-GPU run.

`OPTLOG.md` records the failures alongside the wins, including several measurements that were
wrong and had to be retracted.

## License

MIT, same as upstream llama.cpp. See [LICENSE](../LICENSE).
