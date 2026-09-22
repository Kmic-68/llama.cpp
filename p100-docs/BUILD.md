# Building

## Get the source

    git clone -b p100-optimizations https://github.com/Kmic-68/llama.cpp
    cd llama.cpp

The release bundle also carries the fork as one diff against the upstream commit it last merged:
`diffs/all-code.diff`, with the base SHA in `diffs/UPSTREAM-BASE-SHA.txt`. To apply it to a
clean upstream checkout:

    git checkout $(cat /mnt/fast/p100-llamacpp-release/diffs/UPSTREAM-BASE-SHA.txt)
    git apply /mnt/fast/p100-llamacpp-release/diffs/all-code.diff

## Configure and build

    cmake -B build-opt -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_FA_QUANTS=all -DCMAKE_BUILD_TYPE=Release
    cmake --build build-opt --config Release -j 14

A full build is ~1.3 GB and takes ~25 minutes.

- **Build for sm_60 alone.** The Pascal tuning in `mmvq.cu` is gated on
  `__CUDA_ARCH_LIST__ == 600`, so adding any second architecture silently switches it all off.
- **`GGML_CUDA_FA_QUANTS=all`** compiles flash attention for every K/V type pair. It replaces
  `GGML_CUDA_FA_ALL_QUANTS`, which upstream deprecated. Upstream's default set
  (`q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`) covers the serving configuration too, and builds a
  smaller library.
- **`GGML_CUDA_NCCL=OFF`.** NCCL isn't used, and the internal AllReduce is slower on PCIe Pascal.
- The old `-DP100_NWARPS`, `-DP100_ROWS`, `-DP100_MC_NWARPS` and `-DP100_MC_ROWS` flags are no
  longer read. The values live in `mmvq.cu`.

**Watch the library size.** The flash-attention template instances multiply by head size and KV
type, and the CUDA library is loaded into VRAM on each card. Growing `libggml-cuda.so` from 374 to
530 MB once pushed the 262144 context back into `cudaMalloc` failure. If the long-context
configuration stops starting after a change, check this first.

**After editing a `.cuh`,** touch the files that include it. The build doesn't always rebuild the
template instances:

    grep -rl "fattn-vec.cuh" ggml/src/ggml-cuda/ | xargs touch

## Verify

    ./tools/gate.sh           # decode benchmark, perplexity, flash-attention op tests
    ./tools/gate.sh --full    # the same, plus the full op suite (~25 minutes)

Expect `tg256` around 30.6 t/s on cold cards, and perplexity 2.6101 ± 0.0198, inside the band
2.6209 ± 0.0199.

Two cautions about what a pass means:

- **Run perplexity early.** One change passed 3949/3949 op tests and still produced NaN in real
  inference.
- **The op suite can't see races.** It runs ops one at a time with host syncs between them. Two
  races in this fork passed it for weeks.

## Which build is running

`--version` reports the commit that was HEAD when the build tree was last *configured*, not
built, so it can lag. `diffs/HEAD-SHA.txt` in the bundle is the authoritative record. Rebuilds of
identical source differ in a few bytes of `.symtab`; compare the `.text`, `.rodata` and
`.nv_fatbin` sections, not whole files.
