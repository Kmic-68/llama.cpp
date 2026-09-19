# Archive

Two superseded releases, each packed whole. Neither is the build to run — that is `../build/`.

## `2026-09-06-bundle.tar.zst`

The release as it shipped on 2026-09-06: `build/`, `patches/` (48 commits) and `diffs/` from that
date, under their original names.

**Those binaries carry two data races.** Both were found afterwards and are fixed in every later
build here:

1. The GEMM attention softmax wrote probabilities over the scores it was still reading. Corrupts
   a few attention rows per long prompt; in fp16 it can NaN the output. `GGML_CUDA_FA_GEMM=0`
   avoided it.
2. An uncompressed tensor-parallel peer copy could overwrite the all-reduce's reduction buffer
   before the destination had read it. This one hits **decode, MTP and short prompts**, and **no
   flag avoids it**.

It is kept only so the before/after measurements in `../logs/OPTLOG.md` can be reproduced. Do not
run it for real work.

## `2026-09-16-bundle.tar.zst`

The release as it shipped on 2026-09-16: `build-2026-09-16/`, `patches-2026-09-16/` (165 commits)
and `diffs-2026-09-16/`.

This one is **not** race-carrying — patches 0154 and 0160-0163 are the fixes for both races above.
It is simply superseded: the current `../patches/` continues the same linear history to 198
commits, so 0001-0165 there are these same commits.

### A naming trap, now fixed

Until 2026-09-19 this generation sat loose in the bundle root as `build-2026-09-06/`,
`patches-2026-09-06/` and `diffs-2026-09-06/` — the *2026-09-06* stamp, because
`refresh-build.sh` defaults `STAMP=2026-09-06` and reuses it on every refresh. So two different
releases carried one date: the 48-commit one inside the tarball above, and this 165-commit one in
the directories beside it. They were never duplicates — the `llama-bench` binaries differ — and
`sync.sh`'s "archive already exists; leaving ... in place" message meant the loose copies were
being kept, not that they were already packed.

If you refresh the bundle again, pass an explicit stamp:

    ./refresh-build.sh /mnt/fast/p100-llamacpp-release 2026-09-19

To extract either one:

    tar --zstd -xf 2026-09-16-bundle.tar.zst
