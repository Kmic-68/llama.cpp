# 3x Tesla P100: measurement data

Measurements of this fork at tag `p100-optimizations-b11515-ae35056` (`ae35056eb`) on **three** Tesla P100 cards
under `-sm tensor`. The fork's own documents cover two cards; this directory adds data for a three-card machine. It
contains measurements and the scripts that produced them. It changes no source, kernel or build file.

## Claims and confidence

| Claim | Confidence | Basis and limits |
|---|---|---|
| With CUDA graphs off (`GGML_CUDA_GRAPHS_PRE_VOLTA` unset), 3-GPU decode is limited by host enqueue time, not the GPUs | High | Unprofiled server timeline: 43.6 of 45.3 ms per token inside `llama_decode`; nvprof API counts. One machine, a 2011 CPU |
| `GGML_CUDA_GRAPHS_PRE_VOLTA=3` raises 3-GPU plain decode from 22.4 to 29.5 t/s | High | Interleaved runs, spread ≤0.8% |
| mmap loading adds noise and a mean penalty on a host with less RAM than the model | High for this host | 5 interleaved rounds; depends on RAM size |
| An NCCL build with `NCCL_P2P_LEVEL=SYS` raises 3-GPU prefill by 51–82% and plain decode by 5.6% | High for speed | 3 repetitions per cell, repeated in round 4. Prefill exchanges of 26+ tokens run in BF16 |
| The NCCL prefill gain carries through to an agent workload: mean task time 124 s against 183 s | Moderate | Nine tasks, 3 repetitions, alternating blocks; every task faster. One harness, one model file |
| Both configurations run `-c 262144` on 3 cards, with over 5 GB free per card | Moderate | One 129,000-token prompt per configuration; a full context was not run |
| Q8_0 decodes 7% slower than the XL with graphs on, and prefills the same | High | 3 repetitions per cell on both configurations |
| NCCL's output differs from the non-NCCL path by no more than changing `-ub` does | Moderate to high | Mean KLD 0.0014 against 0.0023 for `-ub 5` vs `-ub 2048` at `-c 4096`; 0.0031 against 0.0070 at `-c 16384`; 0.0009 against 0.0015 over all 65,535 positions of one 65,536-token context, with no growth by position. One corpus per size |
| No NCCL environment setting beats NCCL's defaults by 3% on this topology | High | One variable at a time and one combination, 3 tests, 6 default runs; the best (1 channel) gives +1.6% prefill |
| The 125 W power cap limits throughput: 150 W gives +6.5% decode and +5% prefill, 175 W +7.7% and +9% | High for llama-bench | 2–3 alternated invocations per cap; decode per GPU watt falls 12% and 21% |
| `-np 2` works with tensor split, NCCL and MTP, but a second client still waits for a long prefill to finish | Moderate | One server per arm, 3 repetitions; cause not investigated |
| A request dropped during prefill holds the slot until the `llama_decode` call in flight ends: 72–74 s at `-b 32768`, about 4 s at `-b 2048` | High for the points measured | Server-log timestamps, 3 repetitions each; one drop point, so the worst case per `-b` is inferred |
| Polling `/slots` 5 times a second delays the server's disconnect check to the end of prefill (about 80 s), at any `-b` | High for the effect, low for the cause | 9 runs with polling against 9 without; the explanation is from reading the code |
| `-b 2048` costs 0.3–0.8% of 64k prefill speed and nothing in decode | Moderate | 125 W sweep and a 150 W pair; few repetitions |
| The server ran 90 minutes of mixed requests with no failure, drift or memory growth | Moderate | One soak, one client, 349 requests |
| NCCL decode is repeatable but not bit-identical to the non-NCCL path | High | Saved-logit hashes: 8 of 8 identical runs; 3.3% of bytes differ against the non-NCCL build |
| The generic 3-GPU exchange takes 54–57% of prefill time | High | nvprof attribution; prefill is not host-bound |
| Out-of-tree one-shot and ring all-reduce kernels beat the replicated generic pattern and are bit-identical across the 3 cards | Moderate | A standalone benchmark, not the fork's code path; projected savings are projections |
| About 8.6 of 21.2 ms/token of matvec time is above the fork's 605 GB/s ceiling | Low to moderate | Shapes from the op profiler (graphs off), times from nvprof (graphs on), scaled per quant type |
| MTP with n-max 3, p-min 0.0 is the best measured setting | Moderate | Fastest under greedy decoding and under the serving sampler (3 seeds). Three prompts; the fork's n-max 4 / p-min 0.2 was not in the grid |
| 2 GPUs against 3 GPUs | **Unresolved** | The 2-GPU runs were made with graphs off and mmap on, and were not repeated |

Full list: [LIMITATIONS.md](LIMITATIONS.md).

## Hardware and software

- 3x Tesla P100-PCIE-16GB, each on a PCIe 3.0 x8 link, all behind one host bridge (no NVLink, no PCIe switch).
- Single-socket Sandy Bridge-E host (Core i7-3930K, 6 cores), **16 GB RAM**, model on a SATA SSD.
- 125 W power cap per card for rounds 1 to 5 and part of round 6; 150 W for the client-drop tests and the settings
  now served. NVIDIA driver 580.178.04, CUDA 12.9.1, NCCL 2.27.3 (in the build image).
- Build `ae35056eb`, `-DCMAKE_CUDA_ARCHITECTURES=60`, Release, native CPU flags, run in a Docker container.
- Model: Qwen3.8-27B. "XL" = unsloth `UD-Q6_K_XL` (23.55 GiB of tensor data: 12.6 GiB Q8_0, 9.9 GiB Q6_K,
  1.0 GiB Q5_K). Also unsloth pure Q8_0 and a locally requantized pure Q6_K.

## Headline results

3 GPUs, XL, `-sm tensor -fa 1 -ctk q4_0 -ctv q4_0 -b 32768 -ub 2048`, `GGML_CUDA_P2P=1`. "A" = graphs `=3` and
`-lm none`; "B" = A plus the NCCL build with `NCCL_P2P_LEVEL=SYS`. Mean ± stdev of 3 repetitions.

| test, t/s | graphs off, mmap (round 1) | A | B |
|---|---:|---:|---:|
| tg512 | 21.40 | 29.44 ± 0.09 | 31.09 ± 0.06 |
| pp2048 at depth 0 | 274.6 | 275.94 ± 0.23 | 499.75 ± 0.11 |
| pp2048 at depth 16384 | 260.1 | 259.64 ± 0.38 | 444.91 ± 0.58 |
| pp2048 at depth 65536 | 211.1 | 211.75 ± 0.27 | 318.66 ± 0.59 |
| MTP (n-max 3, p-min 0.0, greedy), mean of 3 prompts | — | 48.36 | 47.30 |
| MTP (n-max 3, p-min 0.0, sampled, 3 seeds), mean of 3 prompts | — | 41.3 | 43.6 |
| pp129000 through `llama-server` at `-c 262144` (one run) | — | 211.0 | 315.9 |

Agent battery (nine tool-calling tasks, 600 s limit, `-c 262144`, MTP on, sampled; section 1.9):

| | A | B |
|---|---:|---:|
| mean wall time per task, first 3 repetitions | 183 s | 124 s |
| tasks passed, first 3 repetitions | 23 of 27 | 24 of 27 |
| tasks passed, excluding the task that times out on both | 26 of 27 | 26 of 27 |
| server prefill over the battery | 242.2 t/s | 377.9 t/s |
| server decode with MTP over the battery | 48.1 t/s | 48.9 t/s |
| peak memory per card | 11.0 GiB | 11.1 GiB |

Rounds 5 and 6 (configuration B; sections 1.10 to 1.21):

| | |
|---|---|
| NCCL tuning | no setting beats the default by 3%; 1 channel gives +1.6% prefill |
| Power cap, tg512 / pp2048 at depth 0 | 125 W: 31.25 / 506.0 t/s; 150 W: 33.29 / 530.4; 175 W: 33.66 / 549.0 |
| `-ub 4096` against 2048, 8,192-token prompt | +1.3% and +0.8%, for 2.4 GB more per card |
| NCCL against non-NCCL, 65,536-token context | mean KLD 0.00089, same top token 98.62%; four identical NCCL hashes |
| Second client arriving 5 s into a 64k prefill, time to first token | 163 s on one slot; 164–192 s on two |
| Two short chats at once, aggregate | 37 t/s on one slot; 45–47 t/s on two |
| Slot release after a client drops mid-prefill, nothing else calling | 72–74 s at `-b 32768`; about 4 s at `-b 2048` and `-b 4096` |
| The same with `/slots` polled 5 times a second | about 80 s at every `-b` |
| 64k prefill, `-b 2048` against `-b 32768`, 150 W | 403.4 against 404.8 t/s |
| 90-minute soak | 349 requests, none failed; prefill flat; peak memory constant |

Settings now served on the test machine: configuration B, `-c 262144`, `-b 2048 -ub 2048`, `-lm none -fit off`,
one slot, MTP n-max 3 / p-min 0.0, 150 W per card (section 1.21).

Other measurements in [RESULTS.md](RESULTS.md):

- **Where a decode token goes** (graphs on): matvec 21.2 of 33.8 ms; 128 exchanges of 20 KB per token; exchange
  work plus adjacent idle 3.8 ms.
- **P2P on this board:** 5.1–5.2 GB/s per pair in both directions at 20 MB; 3.1–4.4 GB/s per card with all three
  sending at once.
- **Exchange share of prefill:** 57% at depth 16384, 54% at depth 65536 (generic 3-GPU path).
- **AllReduce shootout (standalone):** at 20 KB, 49.4 µs for the replicated generic pattern against 36.9 µs
  (one-shot, f32); at 40 MB, 32.7 ms against 18.0 ms (ring, f32).
- **Layer split** prefills long prompts faster than tensor split without NCCL (373.9 against 242.0 t/s at pp65536)
  and decodes slower (15.7 against 22.4 t/s, graphs off).
- **Pure Q6_K against the XL:** +2.4% decode, no difference in prefill (graphs off).
- **Q8_0 against the XL (graphs on):** tg512 27.73 against 29.81 t/s (A) and 29.06 against 31.23 t/s (B); prefill
  equal; about 1.1 GiB more memory per card.
- **Thermal soak, 30 min:** peak 59 / 66 / 64 °C, no thermal throttling, no throughput drift.
- **NCCL quality:** see the table above and RESULTS.md sections 1.3 and 1.8.

## Files

| | |
|---|---|
| [RESULTS.md](RESULTS.md) | All results; final numbers first (Parts 1 to 1d), superseded round-1 decode numbers last |
| [METHODOLOGY.md](METHODOLOGY.md) | Protocol, command lines, build, how the pure Q6_K was made |
| [LIMITATIONS.md](LIMITATIONS.md) | What these numbers do not show |
| [results/](results/) | Per-run output, telemetry and profiler summaries; see its README for what was left out |
| [scripts/](scripts/) | Run harness, P2P microbenchmarks, the AllReduce shootout, trace analyzers, the per-position KLD tool |

## How this was produced

The runs were planned, executed and written up with an AI coding assistant (Claude Code) working on the test machine
under my direction. I chose what to measure and reviewed the results; the scripts, the analysis and these documents
were drafted by the assistant. Treat the numbers as measurements from one machine and check anything you rely on.
