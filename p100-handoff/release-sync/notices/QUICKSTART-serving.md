## Serving it for real work (2026-09-17)

The configuration at the top of this page is the *benchmark* configuration: one model, one client,
greedy, nothing else attached. This is the same thing as an actual agent-serving process, as run
day to day:

    LD_LIBRARY_PATH=/mnt/fast/p100-llamacpp-release/build \
    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=1 \
    /mnt/fast/p100-llamacpp-release/build/llama-server \
      -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 262144 -ub 2048 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.2 \
      -ngld 99 -ubd 256 \
      --jinja --temp 0.3 --top-k 20 \
      --host 0.0.0.0 --port 8080 \
      --tools all \
      --mcp-servers-config ~/mcp-servers.json

Every performance flag matches the table above; the additions are serving and quality flags that
do not touch the CUDA path at all:

| flag | why |
|---|---|
| `--jinja` | use the model's own chat template from the gguf. Required for tool calling to be formatted the way the model was trained, and Qwen3.8's template is not the built-in default |
| `--temp 0.3 --top-k 20` | sampling, not kernels. Low but non-zero — see the note on determinism below |
| `--tools all` | enable the server's built-in tool handlers |
| `--mcp-servers-config` | MCP servers the server exposes as tools |
| `--host 0.0.0.0 --port 8080` | listen on the LAN rather than loopback |
| `LD_LIBRARY_PATH=.../build` | **required** when invoking the binaries by absolute path. They carry a RUNPATH pointing at the original build tree, so without this they silently load whichever `libggml-cuda.so` is there instead of the one in `build/` |

### `--spec-draft-n-max`: this page says 3, real use says 4

These are not in conflict; they are two different operating points, and the page only ever
described one of them.

**The case for 3** (FINDINGS §4) is a tile-geometry argument measured at 262144 context: the draft
plus the token being verified is `nb = n_draft+1`, the flash-attn tile ladder has fixed widths, and
`nb = 4` fills one exactly while `nb = 5` does not. At full depth that boundary is worth **50.5 ms
against 78.4 ms of attention per verify pass** — the extra token buys a whole extra tile width and
wastes most of it.

**The case for 4** is that the 28 ms is only decisive when attention dominates the pass, which is
true at 262k and progressively less true as the context shortens. The extra draft token, when
accepted, saves an entire forward pass; whether that trade wins depends on the acceptance rate of
the actual traffic, and on how deep the conversation typically is. In day-to-day agent use — mixed
prompt depths, tool results coming back, `--temp 0.3` rather than greedy — `n-max 4` was the better
setting.

**Neither number is a general answer, and a single benchmark run cannot settle it**, because
speculative decoding is not deterministic in real use: acceptance depends on the sampled tokens, so
two runs of the same prompt at `--temp 0.3` do not do the same amount of work. Anything measured
here needs several runs at the depth you actually operate at, compared on distribution rather than
on a single number. Treat 3 as the answer for long-context benchmark throughput and 4 as the
default for interactive work, and re-measure if your traffic shifts.

(This is separate from the *numerical* determinism the rest of this bundle claims. With a fixed
seed and greedy sampling the model's arithmetic is reproducible run to run on two physical GPUs —
that was verified. Sampling variance is a property of the sampler, not of the kernels.)
