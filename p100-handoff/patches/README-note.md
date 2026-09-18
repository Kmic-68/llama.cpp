# About `patches/`

These numbered patches are the *earlier* patch set, up to `2c1f89b12`, kept for
reference. They are **already committed** in this tree -- do not re-apply them.

`PENDING-oob-and-fastdiv-fixes.patch` is likewise already committed
(`24290a858` and `9e99d468f`) and can be deleted whenever you like.

The six changes made on 2026-08-31/09-01 (`17455ce35..c4908ecb4`) are **not**
represented here; they exist only as commits. To see them:

    git log --oneline 17455ce35..c4908ecb4 -- ggml/
    git diff 17455ce35 c4908ecb4 -- ggml/src/ggml-cuda/

Files touched by that later work, all under `ggml/src/ggml-cuda/`:
`ggml-cuda.cu`, `common.cuh`, `convert.cu`, `gated_delta_net.cu`.
