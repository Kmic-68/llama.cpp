# Notes for anyone reusing this

## What is likely to generalize

- **The KV-cache dequantization fix** (`launch_fattn` converting the whole cache to f16 on every
  call) affects **any pre-Volta GPU with a quantized KV cache**, and the cost scales with context
  length. This is the most broadly useful change here.
- **cuBLAS algorithm selection.** On Pascal, `CUBLAS_GEMM_DEFAULT_TENSOR_OP` picks a long-chain
  fp16 accumulator that is both less accurate and slower than `ALGO6` at 512-1024 rows. Worth
  checking on any pre-Volta card before assuming the default is sensible.
- **CUDA graphs on Pascal.** Upstream disables them by architecture alone; they help workloads
  that issue many small kernels.
- **The DP4A emulation** (8 instructions, PRMT + XMAD.H1, bit-exact) applies to all of sm_60.

## What is specific to this setup

- Tile configurations are scoped to **head size 256 with GQA ratio 6** — the Qwen3.8-27B shape
  under a 2-way tensor split. Other shapes take the stock path by design.
- The `n_draft` recommendation follows from *that* tile geometry. On a different head size the
  boundary moves; the rule (`n_draft+1` should exactly fill a tile) is what transfers, not the
  number — and see QUICKSTART for why the rule alone does not settle it in practice either.
- These are **P100-PCIe** cards. At least one result (the internal AllReduce being slower) is a
  direct consequence of PCIe rather than NVLink.

## If you want to upstream any of this

llama.cpp's `AGENTS.md` requires that a contributor fully understand and be able to defend a
change without AI assistance, and it **prohibits AI-written commit messages and PR descriptions**.
The commit messages in this series are AI-written, so they would need rewriting by a human author.
Private forks are explicitly exempt from those rules, which is what this is.

The dequantization fix is the piece most worth proposing upstream, since it is a clear
correctness-preserving win for a whole GPU generation rather than a shape-specific tuning.

## Reproducing the measurements

See `../tools/MEASURE.md`. Three things save hours:

- `test-backend-ops perf -o FLASH_ATTN_EXT -p "<params regex>"` measures one shape in ~7 s instead
  of ~7 min for the suite.
- `speculative.n_max` is a **per-request** field on the server, and the prompt cache covers both
  target and draft contexts — so you can prefill a 229k prompt **once** and then measure each
  configuration in ~20 s instead of re-prefilling for 25 minutes.
- Cold-start and thermal skew reach 13% across sessions. Discard a warmup run and interleave A/B
  within one session, or you will measure the cards rather than the code.
