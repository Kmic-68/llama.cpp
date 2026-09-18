# Quickstart — the flags that matter

Binaries are built for **sm_60 only** and expect the driver in `ENVIRONMENT.md` (580.173.02).

## Put it on PATH

Add this one line to `~/.bashrc`:

    export PATH="/mnt/fast/p100-llamacpp-release/bin:$PATH"

Then `llama-server`, `llama-bench`, `llama-cli` and the other 88 tools just work from anywhere,
and `qwen-server` starts the tuned configuration below.

**Use `bin/`, not `build/`.** They hold the same 91 programs, but `bin/` are one-line wrappers
that set `LD_LIBRARY_PATH` to this bundle before exec'ing. The real binaries carry a RUNPATH
pointing at the tree they were compiled in, so run directly from `build/` they load *that* tree's
`libggml-cuda.so` if it still exists — a different build, silently, with no error. Check any time
with:

    ldd $(which llama-server) | grep ggml-cuda

Every path should say `/mnt/fast/p100-llamacpp-release/build`.

## Serving (the configuration in daily use)

With `bin/` on PATH, the whole thing is:

    qwen-server

Arguments are appended and override the defaults, so `qwen-server --port 9000` moves the port and
`QWEN_MODEL=/path/to.gguf qwen-server` swaps the model. What it runs:

    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=1 \
    llama-server \
      -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 262144 -ub 2048 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
      -ngld 99 -ubd 256 \
      --jinja --temp 0.3 --top-k 20 \
      --host 0.0.0.0 --port 8080 \
      --tools all \
      --mcp-servers-config ~/mcp-servers.json

### Performance flags

| flag | why |
|---|---|
| `-sm tensor` | tensor-split across both cards; required for this model to fit at 262144 |
| `-fa 1` | flash attention; **required** by `SPLIT_MODE_TENSOR` |
| `-ctk q4_0 -ctv q4_0` | q4_0 KV cache. f16 will not fit at this context |
| **`-np 1`** | **required.** The server auto-sizes its slot count and each slot allocates its own 262144 KV cache. Without this, startup dies with `cudaMalloc failed` on 512 MiB while the GPUs are nearly empty |
| `-b 262144` | admission limit; must exceed the prompt. Separate from `-ub` |
| `-ub 2048` | sets the compute shape and prefill speed |
| **`-ubd 256`** | draft context ubatch. Without it the draft inherits `-ub 2048`, reserves a second 1024 MiB copy of the KQ mask, and the whole config OOMs |
| `GGML_CUDA_P2P=1` | peer-to-peer between the two cards. Keep it on — without it the exchanges stage through the host, which is slower |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=1` | CUDA graphs on Pascal: **+6.7% on the speculative path, -2% on single-token decode**. Set it for MTP workloads, leave it off otherwise |

### Serving flags (these do not touch the CUDA path)

| flag | why |
|---|---|
| `--jinja` | use the model's own chat template from the gguf. Required for tool calls to be formatted as the model was trained; Qwen3.8's template is not the built-in default |
| `--temp 0.3 --top-k 20` | sampling. Low but non-zero — a deliberate choice for reliable tool-call formatting, not a tuned value |
| `--tools all` | enable the server's built-in tool handlers |
| `--mcp-servers-config` | MCP servers exposed as tools |
| `--host 0.0.0.0 --port 8080` | listen on the LAN rather than loopback |

### Watch VRAM

This configuration peaks at **16137 MiB on GPU0** of 16384 — about 250 MiB of headroom, and that
already counts ~392 MiB held by an unrelated desktop-streaming process. A mid-session
`cudaMalloc failed` is this, not a bug. The levers are `-ub 2048` → `1536`, or a smaller `-c`.

## `--spec-draft-n-max`: 3 or 4 depends on your depth

Both are right at their own operating point.

**3** comes from tile geometry: the draft plus the token being verified is `nb = n_draft+1`, the
flash-attn tile ladder has fixed widths, and `nb = 4` fills one exactly while `nb = 5` does not.
At 262144 context that boundary is worth **50.5 ms against 78.4 ms** of attention per verify pass.

**4** wins when attention does not dominate the pass — shallower contexts, and a real acceptance
rate rather than greedy. The extra draft token, when accepted, saves an entire forward pass. In
day-to-day agent use with mixed prompt depths, 4 measured better.

**No single benchmark settles it**, because speculative decoding is not deterministic in real
use: acceptance depends on the sampled tokens, so two runs of one prompt at `--temp 0.3` do not
do the same amount of work. Compare distributions over several runs at the depth you operate at.

(That is sampler variance. It is separate from numerical determinism: with a fixed seed and
greedy sampling this build is bit-reproducible run to run on two physical GPUs.)

## Short context / plain decode

    GGML_CUDA_P2P=1 llama-bench -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 256 -r 3

Leave `GGML_CUDA_GRAPHS_PRE_VOLTA` **unset** here — it costs ~2% on this workload.

## Is speculative decoding worth it?

**At short context, yes** — MTP is worth ~1.7x. **At 229k context it is worth almost nothing**
(23.2 t/s against 21.5-23.7 plain), because the verify pass pays nearly the same attention cost
as the token it saves. If your workload is long-context, the simpler plain-decode config is
within noise of the speculative one.

## The accuracy mode, in one flag

    GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32

On this card that is the whole accuracy/speed trade. Prefill matmuls stop rounding their inputs
and accumulation to fp16, which is **the entire measurable distance between this fork and an
all-fp32 run**: paired per-chunk perplexity over 4096 × 30 tokens moves from +0.00343 nats/token
to -0.00026 — from "measurably worse than fp32" (t 7.4) to "indistinguishable from it" (t -1.4);
perplexity 2.6191 → 2.6095.

It costs **~40% of prefill throughput** (pp512 391 → 217 t/s, pp2048 414 → 255) and **nothing on
decode** (tg256 30.78 → 30.77), because decode never takes the cuBLAS path.

Use it when the output matters more than the wait; leave it off for interactive work.

## Other precision flags

| flag | what it does |
|---|---|
| `GGML_CUDA_FA_GEMM=0` | The cuBLAS-GEMM attention path is **on by default** at `Q->ne[1] >= 128 && K->ne[1] >= 4096`; `=0` falls back to the tile kernel, which is ~5x more accurate per op but slower at depth. A plain precision/speed choice |
| `GGML_CUDA_FA_GEMM_PREC=32` | fp32 accumulation in the GEMM attention path. -11% pp2048 at 16k depth, -27% at 65k, and **buys nothing measurable at the model level** — the fp16 default is not distinguishable from it in perplexity |

## Correctness check

    /mnt/fast/p100-llamacpp-release/tools/gate.sh

Use the script. It carries the right corpus, and the corpus is the part that drifts — see
`FINDINGS.md`, "How the measurements lied", item 7.

If you run it by hand anyway:

    llama-perplexity -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      -f /mnt/fast/p100-llamacpp-release/tools/perplexity-gate-corpus.txt \
      -sm tensor -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0

Expect **2.6097 ± 0.0198** (gate band 2.6209 ± 0.0199). Use *that* corpus — a different wiki dump
gives ~2.7566 on any build including stock, which looks like a regression and is not.

## Build flags that no longer exist

`-DP100_NWARPS`, `-DP100_ROWS`, `-DP100_MC_NWARPS`, `-DP100_MC_ROWS` are **no longer read**. The
mmvq geometry lives in `mmvq.cu`, gated on `__CUDA_ARCH_LIST__ == 600` — building for any second
architecture silently disables the whole Pascal geometry block.
