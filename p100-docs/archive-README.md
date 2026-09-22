# Archive

Superseded releases, each packed whole. None of them is the build to run. That's `../build/`.

| archive | what it is |
|---|---|
| `2026-09-06-bundle.tar.zst` | the first release (48 commits on upstream `f280b2698`). **It carries two data races** that were fixed afterwards: the GEMM attention softmax (avoidable with `GGML_CUDA_FA_GEMM=0`) and a tensor-parallel peer copy that corrupts decode and MTP, which no flag avoids. Kept only so OPTLOG's before/after numbers can be reproduced |
| `2026-09-16-bundle.tar.zst` | race-free, on the same upstream base. Superseded by later tuning |
| `2026-09-19-bundle.tar.zst` | the last release on upstream `f280b2698`, before the first upstream merge. It's the "before" in CHANGES §9 |

To unpack one:

    tar --zstd -xf 2026-09-19-bundle.tar.zst

Run unpacked binaries with `LD_LIBRARY_PATH` pointing at their own directory. Otherwise they
load whatever `libggml-cuda.so` their RUNPATH names.
