# Methodology

## Build and image

- **Source:** this fork at `ae35056eb` (tag `p100-optimizations-b11515-ae35056`), unmodified.
- **Image:** a CUDA 12.9.1 development image (nvcc 12.9, NCCL 2.27.3 headers and libraries present). The build and
  every run happen inside it; the container gets all GPUs (`--gpus all`), the source tree and the model directory
  (read-only).
- **Build `build-opt`** (configuration "A"): `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_BUILD_TYPE=Release`,
  native CPU flags, `GGML_CUDA_NCCL=OFF`.
- **Build `build-nccl`** (configuration "B"): the same source and flags with `-DGGML_CUDA_NCCL=ON -DGGML_NATIVE=ON`, in
  a separate build directory. A first attempt with `-DGGML_NATIVE=OFF` crashed with an illegal instruction on this
  CPU (no AVX2) and was discarded.
- llama-bench prints `build_commit=unknown` in the container because git refuses the bind-mounted checkout; the
  checkout was verified at `ae35056eb` on the host.
- Host: NVIDIA driver 580.178.04, IOMMU off, performance cpufreq governor, all three cards behind one host bridge.

## Protocol

- **One run = one process in a fresh container.** The model is loaded from disk every time.
- **Cool-down before every run:** wait until every GPU is at or below the limit, with a 10-minute cap; the wait and
  the start temperatures are logged. The limit was ≤45 °C or idle baseline + 2 °C (round 1), ≤45 °C (round 2), and
  ≤48 °C with the fans at full speed (round 3 after its first three results, and rounds 4 to 6).
- **Warmup:** llama-bench's built-in warmup (discarded) on every throughput run. Server runs send a discarded warmup
  request first. Round 1's nvprof and op-profile runs used `--no-warmup`.
- **Repetitions:** llama-bench `-r 3` per invocation unless stated; the tables give mean ± stdev of those
  repetitions. MTP cells in round 3 are three separate invocations.
- **Decode test:** `-p 0 -n 512` (tg512). **Prefill tests:** `-p 2048 -n 0`, with `-d 16384` or `-d 65536` for
  depth, and `-p 16384` / `-p 65536` for long prompts. llama-bench has no `-c`; its context is prompt + generation
  + depth.
- **Interleaving:** arms alternate per test (A B, then B A), or run in rotated order per round, so that drift does
  not favour one arm.
- **Telemetry:** `nvidia-smi` at 1 Hz for every run (temperature, SM clock, power, utilization, memory, clock
  event reasons), plus the cpufreq governor and per-core MHz at 1 Hz from round 2, and fan RPM and PWM every 2–5 s.
- **Watchdog:** a run is killed and the queue stops on GPU temperature above 80 °C, SM clock below 1000 MHz for 5 s
  while busy, no GPU memory in use for 10 minutes, a failed fan check, or a new kernel Xid. A new error in a run's
  stderr also stops the queue (round 1 only flagged it); an out-of-memory failure is recorded as "does not fit".
- **Nothing else runs on the machine** during a measured run (no builds, quantization or large copies).
- **Model load:** default (mmap) in round 1; `-lm none` from round 2 on. On this 16 GB-RAM machine mmap adds
  noise and a mean penalty to decode (RESULTS.md section 2.3) and roughly doubles load time.

## Command lines

Every command actually run is in `results/round1/commands.txt` and in the `RUN` lines of `results/round*/run_log.txt`
(`$BENCH_DIR` and `$SCRATCH` stand for local directories). The container is started as:

```
docker run --gpus all -e GGML_CUDA_P2P=1 [-e ...] \
    -v <llama.cpp checkout>:/workspace/src -v <models>:/models:ro -v <bench dir>:/phase1 \
    -w /workspace/src <image> <command>
```

Common arguments below: `COMMON = -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 -b 32768 -ub 2048`.

**Throughput (configuration A; B uses `build-nccl/bin` and adds `-e NCCL_P2P_LEVEL=SYS`):**

```
GGML_CUDA_GRAPHS_PRE_VOLTA=3 ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 0 -n 512 -r 3 -o csv
GGML_CUDA_GRAPHS_PRE_VOLTA=3 ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 2048 -n 0 [-d 16384|-d 65536] -r 3 -o csv
```

Round 1 ran the same commands without `GGML_CUDA_GRAPHS_PRE_VOLTA` and without `-lm none`; 2-GPU arms added
`CUDA_VISIBLE_DEVICES=1,2` or `0,1` and `-ts 1/1`; layer split replaced `-sm tensor` with `-sm layer`.

**MTP** (adapted from the fork's `p100-handoff/bench/mtp.sh`: same flags, with the model, a prompt file, `-n 256`,
`-c 16384`, `-b 32768 -ub 2048`, `-fit off`, `-lm none` and the graph variable added):

```
GGML_CUDA_GRAPHS_PRE_VOLTA=3 ./build-opt/bin/llama-speculative-simple -m XL.gguf --spec-type draft-mtp \
    --spec-draft-n-max N --spec-draft-p-min P -ngld 99 -f PROMPT -n 256 --temp 0 --top-k 1 --seed 42 \
    -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 -c 16384 -b 32768 -ub 2048 -fit off -lm none
```

The three prompts are raw chat-template text ending in an empty think block: a short chat question (about 40
tokens), a code-generation request (about 60 tokens), and a summarization request over the first 30,500 bytes of
`docs/build.md` (8,009 tokens). `-fit off` avoids the tool's auto-fit, which is not implemented for tensor split.

**Decode budget** (round 2) and **kernel profile** (round 3):

```
nvprof --print-gpu-trace [--print-api-trace] --csv --normalized-time-unit us --log-file trace.csv \
    ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 0 -n 36 -r 1 -o csv
GGML_CUDA_OP_PROFILE=2 ./build-opt/bin/llama-bench -lm none -m XL.gguf COMMON -p 0 -n 64 -r 1 --no-warmup -o csv
LLAMA_TL=1 ./build-opt/bin/llama-server -m XL.gguf COMMON -np 1 -fit off -lm none -c 4096 ...   # host timeline
```

Analysis: `scripts/analyze_i1.py` (budget), `scripts/analyze_i4.py` (kernel table), `scripts/analyze_prefill.py`
and `scripts/analyze_decode.py` (round 1 attribution), `scripts/analyze_overlap.py` (layer-split overlap). The token
boundary in a trace is the device-0 logits copy to host; the last 32 of 36 tokens are analysed.

**NCCL quality** (round 3; corpus = the fork's gate corpus `p100-handoff/ppl-orig.txt`):

```
# baseline, non-NCCL build, at -ub 2048 and at -ub 5
./build-opt/bin/llama-perplexity -m XL.gguf -f ppl-orig.txt -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 \
    -c 4096 -b 4096 --chunks 8 -lm none -fit off -ub 2048 --kl-divergence-base base_ub2048.bin
# NCCL build against that baseline
NCCL_P2P_LEVEL=SYS build-nccl/bin/llama-perplexity <same arguments> -ub 2048 --kl-divergence --kl-divergence-base base_ub2048.bin
# repeatability: one 16,384-token chunk, saved logits hashed after each of 8 runs
NCCL_P2P_LEVEL=SYS build-nccl/bin/llama-perplexity ... -c 16384 -b 16384 -ub 2048 --chunks 1 --kl-divergence-base rep.bin
```

The server check sent seven chat requests to one server (a prompt, a different prompt, the first again, a follow-up
turn, a prompt of about 7,600 tokens, then a short one) with the default prompt cache, at `-c 32768` and `-c 65536`.

**P2P and the shootout:** `scripts/p2pbench/` (see `scripts/README.md`).

## Round 4

**Server flags** (context fit and agent battery; A uses `build-opt`, B uses `build-nccl` with `NCCL_P2P_LEVEL=SYS`):

```
GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
./bin/llama-server -m XL.gguf --jinja --cache-ram 0 --no-cache-idle-slots --parallel 1 -c 262144 \
    -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 -b 32768 -ub 2048 -fit off -lm none \
    --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0
```

**Context fit:** the server above at `-c 262144` and `-c 131072`. `scripts/context_fit_client.py` tokenizes
wikitext-2 raw test through the server, sends the first 129,000 token ids as one completion request with the prompt
cache off, generates 64 tokens, and records the server's own timings and the number of prompt tokens it processed.
Peak memory is the maximum of the 1 Hz `nvidia-smi` log. Order A, B, B, A. The Q8_0 check ran at `-c 262144` only.

**Q8_0 against the XL:** llama-bench `-r 3` for tg512, pp2048 and pp2048 at `-d 16384`, on A and B, order alternated
per test (Q8_0 XL XL Q8_0). MTP: `llama-speculative-simple`, n-max 3, p-min 0.0, greedy, 256 tokens, the three
prompts of round 3, three invocations per cell. `LLAMA_SPEC_SAMPLE_TEMP` and `LLAMA_SPEC_DRAFT_TOPK` were set for
these runs.

**MTP under sampling:** the same program and prompts with `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0`,
`LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20`, n-max 3, p-min 0.0 / 0.5 / 0.75, `--seed 1`, `2`, `3`.

**KLD at `-c 16384`** (corpus = wikitext-2 raw test, `wiki.test.raw`, unmodified):

```
./build-opt/bin/llama-perplexity -m XL.gguf -f wiki.test.raw -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 \
    -c 16384 -b 16384 --chunks 4 -lm none -fit off -ub 2048 --kl-divergence-base a2048.bin      # also -ub 5 -> a5.bin
NCCL_P2P_LEVEL=SYS build-nccl/bin/llama-perplexity <same arguments> -ub 2048 --kl-divergence-base b2048.bin
kldpos a2048.bin b2048.bin 1024      # and: kldpos a2048.bin a5.bin 1024
```

Three save runs, then `scripts/kldpos.cpp` on pairs of saved files: overall mean, median, 99.9% and maximum KLD,
top-token agreement, and the same per 1,024-position bucket. The tool was checked against llama-perplexity's own
figure on the `-c 4096` files of round 3: mean KLD 0.002322 against 0.002313, the same 98.394% top-token agreement
(the small difference is the files' 16-bit quantization of log-probabilities). llama-perplexity scores positions
n_ctx/2 to n_ctx-2 of each chunk, so 4 chunks at `-c 16384` give 32,764 scored tokens.

**Agent battery.** A tool-calling coding agent, driven by a separate A/B evaluation harness on a second machine,
uses the server above over the LAN through its OpenAI-compatible endpoint. The harness and its tasks are not in this
repository.

- Tasks (9): `err_python_env`, `err_replay_patch`, `err_ambiguous_edit`, `err_case_search`, `err_hidden_search`,
  `err_big_output`, `err_multi_dir`, `err_inline_script`, `err_big_file_read`. Each starts the agent in a fresh
  working directory with one instruction, and a checker grades the final answer or the files left behind.
- Limit: 600 s per task; a task still running then is killed and counted as failed.
- The agent's configured context is 65,536 tokens; the harness sets no sampler parameters, so the server's apply.
- One server process per block, started after the cool-down and stopped at the end of the block. Blocks: 1 A, 1 B,
  2 B, 2 A, 3 A, 3 B, each one repetition of all nine tasks; then 4 B, 4 A, 5 A, 5 B, 6 B, 6 A, each one extra
  repetition of `err_inline_script` and `err_big_file_read`. Tasks run one at a time.
- Prefill and decode rates are sums over every request in the server logs (`prompt eval time` and `eval time`
  lines): tokens divided by time. Pass or fail is the harness's verdict.

## Rounds 5 and 6

Both rounds use configuration B (the NCCL build with `NCCL_P2P_LEVEL=SYS`, graphs `=3`, `-lm none`) and the XL. The
protocol is as before: cool-down to 48 °C or below, a discarded warmup, 1 Hz telemetry and the watchdog. From round 5
the 1 Hz `nvidia-smi` log also records every clock-event reason as its own column (`clocks_event_reasons.*`), and CPU
package energy is read from RAPL once a second.

**NCCL tuning.** One run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING` (pp2048 plus 32 generated tokens)
to read the default algorithm, protocol and channels. Then one variable at a time, each as three llama-bench
invocations (`-p 0 -n 512`, `-p 2048 -n 0`, `-p 2048 -n 0 -d 16384`, all `-r 3`), with a default run before the
first setting, after every third setting and at the end. Every run of the sweep, default included, had
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT` writing to a file, which logs at initialization only. The two best single
settings were then run together beside one more default run.

**Power cap.** `nvidia-smi -pl N` on all three cards before a run, 125 W restored and read back after the sweep and
by an exit trap. Busy sample: utilization above 50%. Power-cap fraction: share of a card's busy samples with
`clocks_event_reasons.sw_power_cap` active. The 175 W step was conditional on three checks made on the 150 W runs:
a gain of at least 3% on one test, every card below 75 °C, and a projected peak (3 x 175 W plus the CPU package
peak) with 25% headroom under the 1000 W supply.

**Server arms.** All server tests use these flags, with `--parallel`, `--kv-unified` and `-b` varied as stated:

```
GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_VOLTA=3 NCCL_P2P_LEVEL=SYS LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20 \
build-nccl/bin/llama-server -m XL.gguf --jinja --cache-ram 0 --no-cache-idle-slots --parallel 1 -c 262144 \
    -sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -ngl 99 -ts 1/1/1 -b 32768 -ub 2048 -fit off -lm none \
    --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0 \
    --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0
```

- Long prompts are 64,000 (or 20,000) token ids from wikitext-2 raw test, sent to `/completion` with the prompt
  cache off except in the soak; a different stretch of the corpus each time.
- Time to first token is the first generated text on the stream, reasoning text included.
- Concurrent clients (round 5): scenario (a) starts the second client 5 s after the first; (b) starts two chats
  together; aggregate t/s is generated tokens of both divided by the wall time of the pair.
- Soak: 90 minutes, one client, the request mix of RESULTS.md section 1.15; drift is the mean of the first
  15 minutes against the last 15 minutes per request kind.

**`-ub`.** Round 5: llama-bench `-p 2048 -n 0` at depth 0 and `-d 16384`, `-ub` 512, 1024, 2048 and 4096, `-b 32768`,
`-r 3`, order reversed for the second test. Round 6: the same with `-p 8192` for 2048 against 4096; peak memory is
the maximum of the 1 Hz log.

**KLD at `-c 65536`.** An out-of-tree program links the build's `libllama`, loads the model with the settings used
everywhere else (tensor split 1/1/1, all layers offloaded, flash attention, q4_0 KV, `-lm none`), decodes one
65,536-token context in 2,048-token batches with logits requested at every position, and writes each position's
log-probabilities with `llama-perplexity`'s 16-bit quantization and record layout. It keeps no more than one batch
of logits in host memory. Three files were written (non-NCCL `-ub 2048`, NCCL `-ub 2048`, non-NCCL `-ub 5`) and
compared with `scripts/kldpos.cpp` in 4,096-position buckets. The program was first checked against
`llama-perplexity --kl-divergence-base` at `-c 4096` on the same build: positions 2,048–4,094 are in both files and
all 2,047 records are byte-identical. Repeatability: three more NCCL runs without saving, comparing a 64-bit hash of
all records.

**Client drop during prefill (round 6).**

1. The client sends a 64,000-token streaming request and notes the time.
2. 78.0 s later it shuts the socket down and closes it.
3. It then reads the server container's log through the container runtime every 0.5 s (no request to the server)
   until the line `release: ... task N | stop processing` appears.
4. It sends one short chat.

All reported times are differences between server-log timestamps: the drop is placed on the log's clock as the
request's `launch_slot_` line plus 78.0 s; "cancel task" and "release" are the server's own lines for that task;
"next request accepted" is the `launch_slot_` line of the short chat. In the polling variants a second connection
requests `/slots` every 0.2 s, or `/health` every 5 s, from before the long request until after the short chat.
`-b 32768` was measured on the serving configuration, restarted for a clean state; `-b 2048` and `-b 4096` in a test
container with the same flags, six servers alternating the two values, a cool-down before every scenario.

Round 6's first `-b` sweep (RESULTS.md section 1.16) also dropped a request at each value, but sent the next chat at
the moment of the drop and polled `/slots` every 0.2 s in parallel. Its drop timings are therefore measurements of
the polling effect only.

## The pure Q6_K file

The fork's documents describe their reference as a pure Q6_K of Qwen3.8-27B. No such file is published, so one was
made locally:

```
./build-opt/bin/llama-quantize --allow-requantize --pure Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-pure-Q6_K.gguf Q6_K
```

- Source: the unsloth pure Q8_0 (29,047,086,048 bytes), which carries the MTP block.
- Result: 22,431,001,568 bytes, 20.88 GiB of tensor data, all 506 quantized tensors Q6_K (including
  `output.weight`, `token_embd.weight`, `ssm_alpha` and `ssm_beta`), no Q8_0 tensors, MTP block present
  (`blk.64`, `nextn_predict_layers = 1`). 10.5 minutes of CPU time, with no GPU run in parallel.
- **This is not the fork's file.** It matches the description in size and tensor types, but it was requantized from
  Q8_0 rather than quantized from full precision, and no importance matrix was used. It was used for speed
  comparison only; its quality was not measured.
