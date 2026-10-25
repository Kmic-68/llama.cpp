# Quickstart

For two Tesla P100-16GB cards (or other Pascal sm_60 cards) running Qwen3.8-27B Q6_K with its
built-in MTP head. Build first: see [BUILD.md](BUILD.md).

## Run the server

One command for text and vision. For vision, add the projector line shown below it.

    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 \
    LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
    ./build-opt/bin/llama-server \
      -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 2048 -ub 2048 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
      -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 \
      --jinja --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
      --host 0.0.0.0 --port 8080

**For vision**, add:

      --mmproj /path/to/mmproj-Qwen3.8-27B-Q8_0.gguf

Nothing else changes. Until 2026-10-01 vision needed `-ub 1024`; a smaller attention mask made
room for `-ub 2048` with the projector loaded.

The release bundle's `qwen-server` wrapper runs exactly this command; pass `--mmproj <file>` to it
for vision.

## What to expect

Prefill, and decode with MTP, at each context depth. Measured 2026-10-01 with `tools/depth-bench.py`:
one conversation grown from 2k to 260k with vision loaded, prefill timed on each ~30k-token chunk,
decode averaged over two questions per depth, cards hot throughout. The previous release is shown
for comparison.

| depth | prefill | decode | prefill, 09-26 release |
|---|---|---|---|
| 2k | 454 t/s ¹ | 54 t/s | 316 t/s (first request) |
| 32k | 423 t/s | 55 t/s | 342 t/s |
| 62k | 342 t/s | 46 t/s | 257 t/s |
| 92k | 257 t/s | 44 t/s | 194 t/s |
| 122k | 206 t/s | 38 t/s | 162 t/s |
| 152k | 175 t/s | 37 t/s | 141 t/s |
| 182k | 156 t/s | 38 t/s | 125 t/s |
| 212k | 140 t/s | 32 t/s | 118 t/s |
| 242k | 127 t/s | 29 t/s | 108 t/s |
| 260k | 123 t/s | 33 t/s | 100 t/s |

¹ On a warm server. The first request after startup reads 375 t/s: it also pays ~1.5 s of one-time
setup (loading cuBLAS kernels and allocating buffers). The GPU itself prefills ~500 t/s at short
context (`llama-bench` `pp2048`: 493); a server request adds the final partial batch, the MTP
catch-up and a checkpoint save.

Filling the whole 260k context takes ~25 minutes of prefill (09-26 release: ~29). A short question
on top of an already-loaded 260k context runs faster than the fill rate: ~153 t/s for 1.5k tokens.

MTP decode depends on how predictable the text is: code and factual answers run faster than
creative writing. Cards that have been under sustained load read ~5-10% lower.

## What the flags do

| flag | why |
|---|---|
| `-sm tensor` | splits every layer across both cards. Needed to fit the full context |
| `-fa 1` | flash attention. Tensor split requires it |
| `-ctk q4_0 -ctv q4_0` | q4_0 KV cache. An f16 cache doesn't fit at 262k |
| `-c 262144` | the model's full context. Reserving it costs nothing until it fills |
| `-np 1` | one server slot. Each slot allocates its own full KV cache |
| `-b 2048` | the logical batch. With `-ub` dividing it, the GPU passes are the same at any `-b`: a 112k-token prompt and three greedy answers gave byte-identical text and identical MTP draft counts at `-b 32768` and `-b 2048` (2x P100, deterministic drafting). What it does change is how long a dropped client keeps the server busy, because cancellation is checked between batches: a 64k prompt abandoned after 60 s held the slot ~33 s more at 32768, ~2 s at 2048 ([THREE-GPU.md](THREE-GPU.md) reported the same). An older version of this row said MTP needed 32768; that no longer holds |
| `-ub 2048` | tokens per GPU pass. 2048 is the fastest prefill that still leaves VRAM headroom at 262k, with or without vision |
| `--spec-type draft-mtp` | speculative decoding with the model's built-in MTP head |
| `--spec-draft-n-max 4 --spec-draft-p-min 0.2` | draft up to 4 tokens; stop below 20% confidence. Use 3 if you mostly work past ~150k context |
| `-ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0` | the draft layer on the GPU, with its own small ubatch and a q4_0 cache. Without `-ubd 64` the draft runs out of memory at full context |
| `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0` | the model card's sampling. The gguf doesn't carry min-p, so without the flag llama.cpp's 0.05 applies |
| `GGML_CUDA_P2P=1` | direct copies between the cards instead of through host memory |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=3` | CUDA graphs for the single-token MTP draft steps only. Full graphs (`1`) run out of VRAM at full context |
| `LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20` | the draft samples instead of taking its top token, and the verify uses the speculative-sampling rule. The output distribution is unchanged; more drafts get accepted |

## VRAM

Figures are **per card**, at a full 262k context, with nothing else on the GPU. GPU0 is the
tighter card because the vision projector loads onto it.

| configuration | GPU0 free at full context |
|---|---|
| text only, `-ub 2048` | ~1.7-1.9 GiB (estimated: the vision figure plus the projector's 600-850 MiB) |
| vision, `-ub 2048` | ~1.1 GiB (measured 732-792 MiB with ~392 MiB of desktop streaming also on GPU0) |

VRAM use grows as the context fills, so check it with a full prompt, not a short one. If GPU0
also drives a display or runs other programs, subtract what they use. Lowering `-ub` is the
fix, and it costs only prefill speed: 2048 → 1024 → 512. With vision, `-mmdev CUDA1` puts the
projector on the second card instead and frees ~850 MiB on GPU0.

## Precision switches

Everything defaults to the fast path, which is at least as accurate as stock. These exist for
A/B testing.

| variable | effect |
|---|---|
| `GGML_CUDA_GEMM_FOLD=0` | prefill matmuls on stock cuBLAS fp16 instead of the fold kernel (fp16 products, fp32 sums) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | prefill matmuls fully in fp32. ~40% slower, no measurable accuracy gain |
| `GGML_CUDA_FA_GEMM=0` | turns off the GEMM attention path for long prefill (back to the tile kernel) |
| `GGML_CUDA_FA_FOLD=0` | long-prefill attention on cuBLAS (fp16 over 2048-key chunks) instead of the fold kernels (fp16 over 128 keys, fp32 across) |
| `GGML_CUDA_FA_PV2=0` | the older fold PV kernel (same accumulation, ~14% slower at 262k) |
| `GGML_CUDA_FA_SASS=0`, `GGML_CUDA_GEMM_FOLD_SASS=0` | the compiler's build of the fold attention and fold GEMM kernels instead of the register-renamed SASS (bit-identical, slightly slower) |
| `GGML_CUDA_GDN_CHUNKED=0` | the gated delta net as a per-token recurrence during prefill instead of chunked (same accuracy vs fp64, slower) |
| `LLAMA_KQ_MASK_COMPACT=0` | the full n_kv x n_tokens f16 attention mask instead of per-row prefix lengths (bit-identical; needs ~1 GiB more per card at 262k and `-ub 2048`, so `-ub 2048` with vision no longer fits) |
| `GGML_CUDA_AR_P2P=0` | the tensor-parallel exchange back to copy-then-add (the one-kernel P2P version is bit-identical and faster) |
| `GGML_CUDA_FUSE_FFN_GLU=0` | the FFN gate, up and SwiGLU as three kernels again (bit-identical) |
| `LLAMA_SPEC_BLOCK_VERIFY=0` | per-token draft verification instead of block verification (same output distribution; block accepts more) |
| `LLAMA_MTP_DRAFT_VOCAB=0` | the MTP draft scores the full vocabulary instead of a small copy of the common tokens. Saves ~112 MiB per card, drafts get slower, output is unchanged |

## Benchmarking

    GGML_CUDA_P2P=1 ./build-opt/bin/llama-bench -m /path/to/Qwen3.8-27B-Q6_K.gguf \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 5

Expect ~32 t/s (plain decode, no MTP). Measure on cool cards: right after a long run, P100s can
read up to 20% low. To check accuracy, run `tools/gate.sh`; perplexity should land near 2.61.
