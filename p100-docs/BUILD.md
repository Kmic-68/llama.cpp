# Building from source

## Reapply onto upstream

The patch series applies to upstream llama.cpp at the SHA in `../diffs/UPSTREAM-BASE-SHA.txt`:

    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp
    git checkout f280b26983ad0fdb705a0d9ebf0503e76f2899b0
    git am /mnt/fast/p100-llamacpp-release/patches/*.patch

Or apply the whole delta at once:

    git apply /mnt/fast/p100-llamacpp-release/diffs/all-code.diff

(`all-code.diff` is kernel and test changes only. `everything.diff` includes the documentation
and logs as well.)

## Configure and build

    cmake -B build-opt -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build-opt --config Release -j 14

Notes:

- **`-DCMAKE_CUDA_ARCHITECTURES=60` matters.** Some tuning is guarded on
  `__CUDA_ARCH_LIST__ == 600` so it applies only to an sm_60-exclusive build; a multi-arch build
  silently takes generic paths.
- **`-DGGML_CUDA_FA_ALL_QUANTS=ON`** is needed for the q4_0 KV cache paths. It also multiplies
  flash-attn template instantiations, which matters below.
- `-DGGML_CUDA_NCCL=OFF` — NCCL is not used here. The internal AllReduce was tested on Pascal and
  is **slower** than the fallback (see FINDINGS).
- Earlier `-DP100_*` tuning flags are **gone**. The values live in the source now, so a plain
  build cannot silently miss them.

## Watch the library size

Flash-attn instantiations are multiplied by head size and KV type. Adding shapes carelessly grew
`libggml-cuda.so` from **374 MB to 530 MB**, which was enough to push the 262144-token context
back into `cudaMalloc` failure at load. New tile configs here are scoped to `DKQ == DV == 256` for
that reason. If the long-context config stops starting, check this first.

## Verify

    # correctness
    ./build-opt/bin/test-backend-ops                 # expect 3/3 backends

    /mnt/fast/p100-llamacpp-release/tools/gate.sh    # perplexity + metric, with the right corpus

`gate.sh` expects **2.6097 ± 0.0198** against a band of 2.6209 ± 0.0199, and `tg256` around
**30.6 t/s**.

Run it by hand only if you must, and mind the corpus:

    ./build-opt/bin/llama-perplexity -m <model> \
      -f /mnt/fast/p100-llamacpp-release/tools/perplexity-gate-corpus.txt \
      -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0

**Run perplexity before trusting a kernel change, not after.** One change in this series passed
3949/3949 operation tests and still produced NaNs in real inference.

**And note what the op suite cannot do.** Two data races in this fork passed the full 14593-case
suite for weeks. It runs ops one at a time with host synchronization between them, which is
exactly the condition under which a cross-stream race does not occur. A green suite is necessary
and nowhere near sufficient.

## `--version` lags, and that is not a stale build

`llama-cli --version` on these binaries reports commit `dce17bf1b`, three commits behind what
they were actually built from. llama.cpp stamps the build-info string at **cmake configure**
time, not at each build, so it pins to whatever HEAD was when the build tree was last configured.

Verify by content instead. The shipped `libggml-base.so` and `libggml-cuda.so` are byte-identical
to the build tree they came from, and the last two fixes are present in them — for example the
zero-slice fix adds a log string you can grep for:

    strings -a build/libggml-base.so | grep "has a zero-sized slice of"

`diffs/HEAD-SHA.txt` is the authoritative record of what `build/` was built from.

## Reproducibility note

A rebuild of identical source is not byte-identical to a previous one: `build-opt` and a snapshot
of the same tree differ in exactly 8 bytes, an nvcc temp-derived symbol in `.symtab`. The
build-id and the `.text`, `.rodata` and `.nv_fatbin` hashes match. Compare sections, not whole
files.
