# Large MoE models on 2x P100: the expert cache

This section covers mixture-of-experts models whose experts don't fit in VRAM, for example
Qwen3.8-Flash-Next UD-IQ4_XS (94 GB of weights on two 16 GB cards) at 262144 context. The rest of
the fork (the dense 27B path) is unchanged by it: with `--moe-lru-slots` unset, none of this code
runs, and the dense path keeps its single-threaded dispatch (see "Threaded dispatch").

Credit: the expert cache, its CUDA ops and the threaded dispatch are wispoffates' work
(p100-series fork), with the tensor-split support by lich-77. The slot-lending idea and the
"measure the machine, then pick the config" approach come from Strata. This fork ported them,
fixed a prefill non-determinism (top-k ties), added copy-engine fills, tail assembly and the
config finder, and made every default exact (see "Exactness").

## Results (Flash-Next UD-IQ4_XS, 262144 context, q4_0 KV, `-sm tensor`)

| config | prefill t/s (20k prompt) | decode t/s | VRAM per card |
|---|---|---|---|
| decode-first: 145 slots, `-ub 256` | ~87 | 26.9 | ~14.4 GB |
| balanced (finder pick for an agent workload) | ~500 | ~23-24 | ~14.3 GB |

At 128k (139264 context, `-ub 3072`, 75 + 80 borrowed slots with the slot loan below, copy-engine fills, measured 11-05):
prefill 489 t/s for a full 131072-token prompt, decode 32-34 t/s, GPU0 peak ~14.4 GB. Decode speed depends on the text
being generated: a continuation that keeps missing the expert cache decodes at ~25 t/s even with the loan. Decode drops to ~16 t/s when the generated
tokens' per-layer-embedding rows are not in the page cache (the PLE table is read from disk; watch major faults).

Sampling is the model card's (temp 1.0, top-k 20, top-p 0.95, min-p 0). Upstream llama.cpp with
`--n-cpu-moe` runs this model at a few t/s.

## How it works

Each MoE layer has 512 experts; a token uses 10. Most of them don't fit in VRAM, so each layer gets
**S slots** in VRAM (a pool) and the rest stay in pinned host RAM. The cache is exclusive: an expert
lives either in a slot or on the host, never both, so S slots cost S experts of VRAM and nothing
is stored twice.

Four graph ops per layer, all on the GPU:

1. **LRU_DECIDE**: reads the router's top-k, finds which experts are already in a slot, and picks
   least-recently-used victims for the misses.
2. **LRU_FILL**: swaps each missed expert in from host RAM and writes the victim back. With
   `GGML_CUDA_LRU_CE=1` the copies run on the copy engines, staged per weight kind (up, gate, down)
   in the order the matmul reads them, so compute starts before the whole expert has arrived.
3. **LRU_ASSEMBLE** (prefill): a 3072-token ubatch touches nearly every expert, so instead of filling
   slots it assembles the host experts the batch needs into a contiguous buffer. With
   `LLAMA_MOE_LRU_TAIL=1` only the host experts are assembled; the MoE matmul reads pooled experts
   straight from the pool. `LLAMA_MOE_LRU_PREFETCH=1` assembles layer l+1 while layer l computes.
4. **LRU_HOST_MERGE**: merges the results back into the layer output.

Decode time per token is a fixed part (about 32 ms here) plus about 0.11 ms per expert miss, so
more slots help decode, with diminishing returns. Prefill time per token is compute plus
transfer of the (512 - S) host experts once per ubatch, so a bigger `-ub` helps prefill. Both
compete for the same VRAM: slots against activations and assembly buffers. That trade-off is
what the config finder decides.

The prefill matmul for the assembled experts is a Pascal tile GEMM (`ggml-cuda/tile-mm.cu`):
fp16 HFMA2 with short chains folded into fp32, the same rule as the rest of this fork.

**Threaded dispatch** (`ggml-backend-meta.cpp`, `ggml-cuda/ar-rank.cu`): with `-sm tensor` on two
GPUs, one host thread per device issues that device's work, and the AllReduce is a kernel that
pairs the GPUs over P2P without host barriers, fused with the next RMS norm. Flash-Next decode is
launch-bound (many small ops per token): 24.0 to 26.9 t/s. It is on for every MoE model, cached or not (Qwen3.6-35B-A3B
Q2_K_XL tg128 73.4 -> 83.9 t/s, pp512 -4%, outputs identical; wispoffates spotted it) (`GGML_META_THREADS=0` turns it off, `=1` forces it for any model). Dense models don't get it by default: on the 27B
it skips the single-threaded path's matmul-overlapped prefill exchange (pp2048 497 vs 462 t/s) for the same decode
speed and little VRAM at depth (~110 MiB per card at 128k).

## Slot loan (opt-in)

The compute buffer is sized by prefill (4.6 GB per card at `-ub 3072`, 128k context) and sits idle while decoding, when the
cache needs the memory most. `LLAMA_MOE_LRU_LEND=N` adds N slots per layer whose memory is also the compute buffer: the pools
are allocated in VMM chunks and ggml-cuda maps the borrowed chunks a second time, behind a small head
(`LLAMA_MOE_LRU_LEND_HEAD`, MiB, default 768) that holds the decode graphs. Decode uses every slot; before a larger batch the
borrowed slots are emptied (they hold copies, the experts keep their host slots), and refill as decode runs. Outputs are
identical with and without it. It needs the exclusive cache with `GGML_CUDA_LRU_STASH=0`; give it about the compute buffer's
size (N x per-slot bytes x layers) and take the base slot count down so prefill's own peak still fits.

## The config finder

`p100-handoff/moe-autoconfig/` measures the machine once, then picks slots, `-ub`, prefetch and
tail for the workload:

    # once per model and machine (~40 min): a routing trace, then five short probes
    python3 calibrate.py --model /path/to/model-00001-of-0000N.gguf --bin ./build-opt/bin
    # any time: reads free VRAM per card and MemAvailable now, prints the server command
    python3 autoconfig.py --calib models/<model>/<host>/calib.json --workload agent

It minimizes time per generated token, `R / prefill + 1 / decode`, where R is prompt tokens per
generated token (chat 3, agent 10, rag 30, or `--ratio`). So "1 t/s less decode for 100 t/s more
prefill" is decided by arithmetic, not by taste. VRAM, decode and prefill are least-squares fits
of the calibration points; the routing trace gives the miss rate for any slot count. A
`--profile` file can reserve VRAM for other programs (a desktop streamer) and RAM headroom.

## Exactness

Every default gives the same result as the uncached model, up to fp32 summation order:

- the cache only moves weights; the same kernels compute on them (seeded outputs are identical
  for any slot count);
- top-k ties are broken by expert index, so long prefills are deterministic;
- the AllReduce sums in fp32. wispoffates' 16-bit wire for prefill reductions is available as
  `GGML_CUDA_AR_WIRE=2` but changes results, so it is off here; prefill measured the same speed
  without it (501 t/s at 100 slots).

Not ported (yet): his fp16 single-token matvec kernels (`mmvq-gp100`, `mmvq-h1`), worth about
2 t/s of Flash-Next decode on his tree; the QSA cache gather (KLD 0.0156, output-changing).

## Environment

| variable | default | what it does |
|---|---|---|
| `--moe-lru-slots N` | 0 (off) | slots per layer (or `LLAMA_MOE_LRU_SLOTS=<file>` per layer) |
| `LLAMA_MOE_LRU_EXCLUSIVE=1` | | exclusive cache (recommended) |
| `GGML_CUDA_LRU_DMA=1`, `GGML_CUDA_LRU_ASYNC=1`, `GGML_CUDA_LRU_CE=1` | | async copy-engine fills |
| `LLAMA_MOE_LRU_PREFETCH`, `LLAMA_MOE_LRU_TAIL` | 0 | prefill assembly ahead / host experts only |
| `LLAMA_UB_KV_BUDGET` | 0 | caps ubatch x context so a big `-ub` fits at full context |
| `GGML_META_THREADS` | on for MoE models, else off | threaded two-GPU dispatch |
| `GGML_META_AR_MAX_BYTES` | 32 MiB | largest in-kernel AllReduce |
| `LLAMA_MOE_LRU_LEND`, `LLAMA_MOE_LRU_LEND_HEAD` | 0, 768 | slot loan: borrowed slots per layer, head MiB |

`autoconfig.py` prints the full set for the chosen config.
