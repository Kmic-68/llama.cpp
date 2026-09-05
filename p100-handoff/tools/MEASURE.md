# Measurement harnesses (session 7)

Two harnesses replaced ~27-minute measurement cycles with seconds. Use them.

## `fa-fast.sh` — kernel-level, ~7 s per shape

`test-backend-ops` accepts **`-p <params regex>`**. Filtering to the shapes that matter runs
one case in ~7 s instead of ~7 min for the whole flash-attn suite:

    ./build-opt/bin/test-backend-ops perf -o FLASH_ATTN_EXT \
      -p "kv=262144,nb=1,.*type_K=q4_0"

Gotchas:
- The timing line and the case description are on **separate output lines**. A grep requiring
  both on one line silently returns nothing.
- `perf` **silently skips large-kv cases when VRAM is occupied**. A running llama-server made
  every kv=262144 case vanish while still printing "2/2 backends passed". Check for a live
  server first.

## `server-sweep.py` — end-to-end, ~20 s per config after one prefill

`speculative.n_max` (and `n_min`, `p_min`, `type`) are **per-request JSON fields**, and the
server's prompt cache covers both the target and draft contexts. So prefill 229k **once**
(~25 min) and then sweep configurations against the cached prefix at ~20 s each.

Start the server with **`-np 1`** — it auto-sizes its slot count and each slot allocates its
own 262144 KV cache, so without it startup dies with `cudaMalloc failed` on 512 MiB while the
GPUs are nearly empty.

    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=1 ./build-opt/bin/llama-server \
      -m /mnt/fast/models/Qwen3.8-27B-Q6_K.gguf \
      -ngl 99 -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 \
      -c 262144 -b 262144 -ub 2048 -np 1 \
      --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.2 -ngld 99 -ubd 256 \
      --port 8099 --host 127.0.0.1

Verify depth in the server log: `n_tokens = 229610, truncated = 0`.

Do **not** pass `--slot-save-path` at this context — a 229k KV cache is ~20 GB and will fill
the disk. Set `ulimit -c 0`: a crashing server writes a multi-GB core that fills `/` and makes
every subsequent command fail with ENOSPC.

## Generation length

**`-n 128` is far too short at 229k.** It amortises a ~2-3 s fixed startup over ~30 passes and
understated plain decode as **12.2 t/s** when the real figure is ~21.5 — a 1.8x error that
misdirected a whole session. Use **>= 512**.
