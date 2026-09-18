# Archive

## `2026-09-06-bundle.tar.zst`

The release as it shipped on 2026-09-06: `build/`, `patches/` (48 commits) and `diffs/` from that
date, under their original names.

**Those binaries carry two data races.** Both were found afterwards and are fixed in the current
`build/`:

1. The GEMM attention softmax wrote probabilities over the scores it was still reading. Corrupts
   a few attention rows per long prompt; in fp16 it can NaN the output. `GGML_CUDA_FA_GEMM=0`
   avoided it.
2. An uncompressed tensor-parallel peer copy could overwrite the all-reduce's reduction buffer
   before the destination had read it. This one hits **decode, MTP and short prompts**, and **no
   flag avoids it**.

It is kept only so the before/after measurements in `../logs/OPTLOG.md` can be reproduced. Do not
run it for real work.

To extract:

    tar --zstd -xf 2026-09-06-bundle.tar.zst
