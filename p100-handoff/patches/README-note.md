# About `patches/`

These numbered patches are the *earlier* patch set, up to `134a4f4a5`, kept for
reference. They are **already committed** in this tree -- do not re-apply them.

`PENDING-oob-and-fastdiv-fixes.patch` is likewise already committed
(`2c0d39158` and `7d004be91`) and can be deleted whenever you like.

The six changes made on 2026-08-31/09-01 (`5d1fafb01..f85e154ed`) are **not**
represented here; they exist only as commits. To see them:

    git log --oneline 5d1fafb01..f85e154ed -- ggml/
    git diff 5d1fafb01 f85e154ed -- ggml/src/ggml-cuda/

Files touched by that later work, all under `ggml/src/ggml-cuda/`:
`ggml-cuda.cu`, `common.cuh`, `convert.cu`, `gated_delta_net.cu`.
