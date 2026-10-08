# Limitations

Read this before quoting a number from this directory.

## Scope

- **One machine.** Every number comes from a single host: three P100s on PCIe 3.0 x8 behind one host bridge, a
  Sandy Bridge-E CPU, 16 GB of RAM, a 125 W power cap. Nothing here was reproduced on a second system. The host-bound
  decode result in particular depends on a 2011 CPU.
- **One model family and three quants** (Qwen3.8-27B: unsloth UD-Q6_K_XL, unsloth Q8_0, and a local pure Q6_K).
- **One build** (`ae35056eb`), run in a container. llama-bench reports `build_commit=unknown` inside the container;
  the checkout was verified by hand.

## Open questions

- **2 GPUs against 3 GPUs is unresolved.** The only 2-GPU runs were in round 1, with CUDA graphs off and mmap
  loading. Both settings distort decode on this host, and the 2-GPU arms were not repeated. Round 1 showed 2 GPUs ahead
  of 3; that result should not be relied on in either direction. No 2-GPU run was made with NCCL.
- **`-c 262144` was run with a half-full context only.** One 129,000-token prompt per configuration completed
  with over 5 GB free per card (RESULTS.md section 1.5). A prompt near 262,144 tokens was not run, and each cell is
  a single run.
- **The agent battery is small.** Nine tasks from one harness, 3 repetitions each (6 for two tasks), one model
  file, one agent. Pass rates differ between A and B by two runs out of 33. The wall-time difference is consistent
  across tasks; the pass-rate difference is not established.
- **`err_big_file_read` hits the 600 s limit on both configurations** (6 of 6 runs on A, 4 of 6 on B). Why was not
  investigated. It dominates the battery's mean wall time and all of its timeouts.
- **The earlier battery run quoted for context used a different quant file** (UD-Q5_K_XL) and older settings. It is
  not a controlled baseline.
- **The served settings have been checked with single requests only.** The settings of RESULTS.md section 1.21
  (`-b 2048`, 150 W) were started as a service and answered a short chat, a second request and a 64,000-token
  prompt. The agent battery and the 90-minute soak were run at `-b 32768` and 125 W, not on these settings.
- **Client-drop timings cover one point in a batch.** Every drop was made 78 s into a 64,000-token prefill. At
  `-b 2048` and `-b 4096` the slot was released at the same token count, so the difference between the two is not
  measured, and the worst case (one full batch) is inferred. `-b 8192` was not run without polling.
- **Why `/slots` polling delays the disconnect check is a reading of the code, not a measurement.** The effect is
  measured (about 80 s against 0.01 s); the explanation in section 1.19 was not confirmed by changing the code. Only
  one polling rate (5 per second) and one endpoint were tried; `/metrics` is disabled on these servers, and `/props`
  was not tried.
- **Which clients call a server's status endpoints cannot be read from its log** at the default log level: it
  records slot activity, not requests.
- **Two slots do not help a client that arrives during a long prefill** (section 1.12). Why the second slot's prompt
  is not interleaved was not investigated.
- **The power-cap sweep is llama-bench only.** Efficiency per watt uses GPU power from `nvidia-smi` and CPU package
  power from RAPL; the rest of the machine is not measured. The 175 W step was run for minutes, not hours.
- **NCCL tuning tried one variable at a time** on three llama-bench tests, plus one combination. Larger buffer
  sizes, other algorithms' parameters and server workloads were not tried.

## NCCL

- **Long-context KLD was measured up to `-c 65536`**: one 65,536-token context of wikitext-2 with every position
  scored (section 1.14), `-c 16384` (positions 8192–16382, 32,764 tokens) and `-c 4096` on the fork's gate corpus
  (16,376 tokens scored). Beyond 65,536 it is **not measured**. The 65k run is a single context of one corpus, made
  with an out-of-tree dump tool because stock llama-perplexity needs about 32 GB of host memory there. The two context sizes used different corpora, so the
  rise in mean KLD from 4k to 16k (0.0014 to 0.0031) is not a clean measure of growth with context.
- **NCCL is not bit-exact for decode.** It is repeatable (identical hashes across runs) but differs from the
  non-NCCL path: 3.3% of saved-logit bytes differ, and greedy generations diverge. Any check that compares bytes or
  greedy text against a non-NCCL reference fails under NCCL.
- **Prefill exchanges of 26 or more tokens use BF16** in the NCCL path as written (131,072 elements at 3 GPUs,
  5,120 per token). Decode, MTP draft and MTP verify exchanges stay in f32.
- Why a different f32 summation order grows to a mean KLD of 0.002 was not investigated. The comparison offered
  (`-ub 5` against `-ub 2048` on the non-NCCL build gives 0.0023) shows the size is not specific to NCCL; it does
  not explain it.
- NCCL chooses a host-memory transport by default on this topology; the results use `NCCL_P2P_LEVEL=SYS`. The
  host-memory path was measured only for speed, and its behaviour with a small `/dev/shm` was not tested.

## MTP

- **MTP settings were compared on three fixed prompts, 256 tokens**: greedy in round 3, and with the serving
  sampler in round 4 (three seeds, n-max 3 only, p-min 0.0 / 0.5 / 0.75). Under sampling the seed-to-seed standard
  deviation is up to 6 t/s, so only the p-min ranking is supported, not small differences between A and B. The
  fork's own default (n-max 4, p-min 0.2) was not in either grid.
- **Round 4's greedy MTP cells had the sampled-draft environment variables set** and are 4–9% below round 3's.
  Compare within a round only.
- Under greedy decoding, repetitions measure timing noise only: acceptance is fixed per configuration and prompt.
- Differences between the NCCL and non-NCCL MTP results come partly from different generated text, not only speed.

## Attribution and profiling

- **The kernel table is approximate.** Kernel times come from nvprof with graphs on; shapes come from the fork's op
  profiler, which only records with graphs off. Per-shape times were scaled per quant type to the nvprof totals
  (Q8_0 ×0.959, Q6_K ×1.034). The op profiler prints only its top 45 entries per device, so Q5_K is under-listed
  (0.51 of 1.36 ms) and 4% of matvec time is not attributed to a shape. "Lost to the ceiling" uses the fork's
  605 GB/s figure, not a ceiling measured here.
- **nvprof inflates host-side time** (token time ×1.4 with graphs on, ×2.1 with graphs off). Kernel durations were
  used as measured; idle time was rescaled to the unprofiled token time and split in the trace's proportions. The
  split between "exchange-adjacent" and "other" idle is therefore an estimate.
- **Round 1's decode exchange share (19%) is superseded.** With graphs off the idle it counted was mostly the GPUs
  waiting for the host. With graphs on the same definition gives 11.3%.
- **The AllReduce shootout is a standalone program**, not the fork's exchange. Its (a) arm replicates the generic
  pattern with peer copies and add kernels; it matches the in-model prefill latency (32.7 against 31.0 ms) but is not
  the same code. Savings per token or per step are projections from measured exchange counts.

## Protocol deviations

- **Cool-down** was "all GPUs ≤45 °C or ≤ idle baseline + 2 °C" in round 1, "≤45 °C" in round 2, and "≤48 °C" with
  the fans at full speed for most of round 3 and all of rounds 4 to 6. Start temperatures are recorded per run. The first three round-3
  results were taken under the ≤45 °C rule.
- **nvprof and op-profile runs in round 1 used `--no-warmup`.**
- **llama-bench has no `-c`**; its context is prompt + generation + depth.
- **Round 2 and round 3 loaded the model with `-lm none`; round 1 used the default (mmap).**
- **Round 5 script fault:** a harness function overwrote the loop variable holding the test name, so four runs of
  the power-cap sweep ran llama-bench's default test. They are excluded and were repeated (RESULTS.md section 1.11).
- **Round 6 used fewer repetitions** in its `-b` sweep (1 to 3 per scenario), and the `-b 2048` cost comparison at
  150 W was not interleaved and cooled down only before the first repetition.
- **Rounds 5 and 6 mix two power caps.** Round 5 and the round 6 `-b` sweep and `-ub` retest ran at 125 W; the
  client-drop tests, the `-b 2048` cost comparison and the served settings are at 150 W.
- Several runs were skipped or killed (out of memory on 2 GPUs with Q8_0 at depth 65536; a watchdog stop on a
  power-capped clock dip under layer split). RESULTS.md section 3.8 lists them.

## Other

- **The pure Q6_K is a local requantization of the unsloth Q8_0** (`--allow-requantize --pure`, no imatrix). It
  matches the fork's described reference in size and tensor types, but it is not the fork's file, and requantizing
  from Q8_0 is not the same as quantizing from full precision. Its quality was not measured.
- **Upstream issue 29466** (second-request assert under `-sm tensor`) did not reproduce in four scripted server runs,
  the agent battery, or 56 short-then-long request pairs of the 90-minute soak. That is not evidence that it is fixed.
- **Raw data is incomplete for rounds 2 to 6**: see [results/README.md](results/README.md).
- Times in logs are the test host's local time (EDT).
