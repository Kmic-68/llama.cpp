# Results: 3x Tesla P100, tensor split

All numbers are from one machine (see [README.md](README.md)) on build `ae35056eb`, model Qwen3.8-27B. "XL" is the
unsloth `UD-Q6_K_XL` quant; "Q8_0" is the unsloth pure Q8_0; "pure Q6_K" is a local requantization (see
[METHODOLOGY.md](METHODOLOGY.md)). Unless stated: 3 GPUs, `-sm tensor -ts 1/1/1 -fa 1 -ctk q4_0 -ctv q4_0
-b 32768 -ub 2048 -ngl 99`, `GGML_CUDA_P2P=1`, mean ± stdev of llama-bench repetitions. Read
[LIMITATIONS.md](LIMITATIONS.md) before quoting anything.

Six measurement rounds are reported. They are ordered here by how much weight the numbers deserve, not by date:

- **Part 1 (round 3, final):** graphs on (`GGML_CUDA_GRAPHS_PRE_VOLTA=3`), `-lm none`, with and without NCCL.
- **Part 1b (round 4):** `-c 262144` fit, Q8_0 against the XL, MTP under sampling, NCCL KLD at `-c 16384`, and an
  agent battery on A and B.
- **Part 1c (round 5):** NCCL tuning, the power cap, concurrent clients and `-np 2`, `-ub`, NCCL KLD at
  `-c 65536`, and a 90-minute server soak.
- **Part 1d (round 6):** `-b`, a client drop during prefill with and without `/slots` polling, and the settings
  now served.
- **Part 2 (round 2):** the decode budget, the AllReduce shootout, noise, layer split, MTP grid, `-ub`, the pure
  Q6_K, the thermal soak, and the first NCCL runs. Mixed settings; each table says which.
- **Part 3 (round 1, decode numbers superseded):** P2P characterization and prefill attribution (still valid), and
  the original decode numbers, which were taken with CUDA graphs off and mmap loading.

Two configurations recur: **A** = graphs `=3`, `-lm none`, no NCCL (the stock `build-opt`); **B** = A plus a build with
`-DGGML_CUDA_NCCL=ON` run with `NCCL_P2P_LEVEL=SYS`.

---

# Part 1. Final numbers: graphs on, `-lm none`, with and without NCCL


## 1.1 Stack matrix

llama-bench rows: mean ± stdev of 3 reps in one invocation, A/B order alternated per test. MTP rows: 3 separate invocations per prompt and config, ABBA; mean ± stdev across the 3.

| test | A | B | B / A | round 1 | A vs round 1 | **B vs round 1** |
|---|---:|---:|---:|---:|---:|---:|
| tg512 t/s | 29.44 ± 0.09 | 31.09 ± 0.06 | **1.056** | 21.40 | +37.6% | **+45.3%** |
| pp2048 @0 t/s | 275.94 ± 0.23 | 499.75 ± 0.11 | **1.811** | 274.6 | +0.5% | **+82.0%** |
| pp2048 @16384 t/s | 259.64 ± 0.38 | 444.91 ± 0.58 | **1.714** | 260.1 | −0.2% | **+71.1%** |
| pp2048 @65536 t/s | 211.75 ± 0.27 | 318.66 ± 0.59 | **1.505** | 211.1 | +0.3% | **+51.0%** |
| MTP chat t/s (acceptance) | 41.05 ± 0.02 (48.1%) | 42.21 ± 0.08 (50.8%) | 1.028 | — | | |
| MTP code t/s (acceptance) | 58.31 ± 0.08 (83.3%) | 53.75 ± 0.02 (75.3%) | **0.922** | — | | |
| MTP summ ~8k t/s (acceptance) | 45.72 ± 0.04 (61.3%) | 45.93 ± 0.01 (60.8%) | 1.005 | — | | |
| MTP mean of the 3 prompts | 48.36 | 47.30 | 0.978 | — | | |

- **Decode:** the full stack (graphs + no mmap + NCCL P2P) is **+45% over round 1** (21.40 → 31.09 t/s). Graphs and `-lm none` deliver +37.6% of that; NCCL adds 5.6% on top. Round 2's +17–18% for NCCL was measured with graphs off, where NCCL's saving in host calls counted for more.
- **Prefill:** config A is identical to round 1 (graphs `=3` only touches single-token graphs, and prefill was never mmap-sensitive). NCCL P2P gives **+82% / +71% / +51%** at depth 0 / 16k / 64k. The gain shrinks with depth because attention, not the exchange, grows with depth. **These prefill numbers use BF16 exchanges** (section 1.3 gives the KLD).
- **MTP decode is not faster under NCCL on average (0.978×).** Chat +2.8%, summarization +0.5%, **code −7.8%**.
  - **The acceptance rates differ between A and B on the same greedy prompt** (code 83.3% vs 75.3%; drafted and accepted counts differ on all three prompts).
  - So NCCL's f32 decode exchange does not produce the same logits as the butterfly: the generated text takes a different path. The code result is a different continuation with fewer accepted drafts, not slower kernels. This is the first direct evidence for section 1.3's bit-exactness question: **decode under NCCL is not bit-identical to the non-NCCL build.**
- **Run-to-run:** within a config the three reps repeat to ≤0.2% and the acceptance counts are identical, so each config is deterministic with itself.
- **No round 1 MTP number exists.** Against section 2.6 (single runs, config A equivalent: 40.90 / 57.91 / 45.42) config A reproduces within 0.4–0.7%.
- **Conditions:** tg512 A/B and pp2048@0 B started under the ≤45 °C rule; all later runs under ≤48 °C with fans at full (35–48 °C; per-run start temperatures are in `results/`). Round 1 and 1.5 numbers were taken at ≤45 °C starts (≤ baseline + 2 in round 1).

## 1.2 MTP settings with repetition (greedy)

Cells: mean ± stdev t/s over 3 runs (draft acceptance, decode ms per accepted draft token). Acceptance and ms/accepted are identical across the 3 runs of a cell (greedy, fixed seed).

**Config A (graphs `=3`, `-lm none`, no NCCL)**

| n-max | p-min | chat | code | summ ~8k | mean t/s |
|---|---|---:|---:|---:|---:|
| 3 | 0.0 | 41.05 ± 0.02 (48%, 41.2) | 58.31 ± 0.08 (83%, 24.0) | 45.72 ± 0.04 (61%, 33.9) | 48.36 |
| 3 | 0.5 | 33.83 ± 0.17 (61%, 52.7) | 55.63 ± 0.04 (88%, 25.2) | 37.95 ± 0.05 (69%, 42.7) | 42.47 |
| 3 | 0.75 | 30.76 ± 0.22 (79%, 65.2) | 47.14 ± 1.78 (92%, 32.1) | 34.51 ± 0.09 (81%, 52.6) | 37.47 |
| 4 | 0.0 | 40.51 ± 0.05 (43%, 39.0) | 59.39 ± 0.08 (77%, 22.3) | 45.98 ± 0.05 (55%, 31.6) | **48.62** |
| 4 | 0.5 | 31.36 ± 0.68 (52%, 57.1) | 57.90 ± 0.07 (78%, 22.9) | 33.75 ± 0.05 (55%, 48.7) | 41.00 |
| 4 | 0.75 | 30.90 ± 0.92 (73%, 62.6) | 48.53 ± 1.36 (88%, 27.7) | 34.70 ± 0.08 (80%, 49.0) | 38.04 |

**Config B (A + NCCL, `NCCL_P2P_LEVEL=SYS`)**

| n-max | p-min | chat | code | summ ~8k | mean t/s |
|---|---|---:|---:|---:|---:|
| 3 | 0.0 | 42.21 ± 0.08 (51%, 39.3) | 53.75 ± 0.02 (75%, 26.9) | 45.93 ± 0.01 (61%, 33.7) | **47.30** |
| 3 | 0.5 | 32.89 ± 0.11 (54%, 56.9) | 47.35 ± 0.08 (86%, 30.8) | 39.45 ± 0.02 (70%, 40.9) | 39.89 |
| 3 | 0.75 | 31.89 ± 0.11 (76%, 62.3) | 48.97 ± 0.02 (92%, 29.6) | 38.19 ± 0.08 (88%, 45.4) | 39.69 |
| 4 | 0.0 | 37.59 ± 0.01 (38%, 44.3) | 57.54 ± 0.33 (73%, 23.3) | 43.36 ± 0.03 (50%, 34.6) | 46.16 |
| 4 | 0.5 | 35.42 ± 0.17 (55%, 45.6) | 51.56 ± 0.12 (80%, 26.7) | 37.63 ± 0.09 (59%, 41.7) | 41.54 |
| 4 | 0.75 | 31.80 ± 0.11 (75%, 61.6) | 50.52 ± 0.09 (90%, 27.0) | 35.27 ± 0.09 (85%, 50.2) | 39.20 |

- **Ranking by mean t/s.** A: (4, 0.0) 48.62 > (3, 0.0) 48.36 > (3, 0.5) 42.47 > (4, 0.5) 41.00 > (4, 0.75) 38.04 > (3, 0.75) 37.47. B: (3, 0.0) 47.30 > (4, 0.0) 46.16 > (4, 0.5) 41.54 > (3, 0.5) 39.89 > (3, 0.75) 39.69 > (4, 0.75) 39.20.
- **Does the round 2 ranking hold? Yes for the conclusion that matters, with one tie flipped.**
  - **p-min 0.0 wins on both configs** by 12–20% over 0.5 and 19–29% over 0.75, far outside the ≤0.2% typical run-to-run spread.
  - **n-max 3 vs 4 at p-min 0.0 is a tie that changes sides:** A has n-max 4 ahead by 0.5% (round 2 had n-max 3 ahead by 1.4% on single runs); B has n-max 3 ahead by 2.5%.
  - **n-max 3, p-min 0.0 is the robust pick** (best on B, within 0.5% of best on A).
- **Repetition shows what the single runs could not:**
  - Timing noise per cell is tiny (≤0.2% in most cells; up to 3.8% on a few A cells at p-min 0.75).
  - The per-prompt differences between cells are real but come from **which text gets generated**: acceptance is a fixed property of (config, n-max, p-min, prompt) under greedy decoding.
  - Per-prompt orderings therefore differ between A and B (e.g. code at (3, 0.5): 55.6 on A, 47.4 on B), because NCCL's decode numerics send the generation down a different path (section 1.1, section 1.3).
- **MTP under NCCL is not faster:** best B cell 47.30 vs best A cell 48.62 (−2.7%). NCCL's value for the stack is prefill, not MTP decode.
- **Caveat:** greedy decoding. The fork's serving configuration samples at temp 1.0 with `LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20`, where acceptance behaves differently (the smoke server run saw 48–68% at n-max 3, p-min 0.0). The p-min conclusion should be confirmed under sampling before it is treated as final for production.

## 1.3 NCCL: KLD against the non-NCCL build, repeatability, server check

**Thresholds, from the code** (`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_backend_cuda_comm_allreduce_nccl`): at 3 backends an exchange is reduced in **FP32 when it has fewer than 131,072 elements, BF16 at or above**. (2 backends: 32,768; 4 or more: 262,144.) Every exchange in this model is n_tokens × 5,120, so the boundary is **25 tokens f32 / 26 tokens BF16**. NCCL is the default path on Linux whenever it is compiled in (`GGML_CUDA_ALLREDUCE` unset), and every exchange goes through it.

| exchange | size | path |
|---|---|---|
| decode (1 token) | 5,120 elements | **f32** |
| MTP draft step (1 token) | 5,120 | **f32** |
| MTP verify (n-max + 1 ≤ 6 tokens) | ≤ 30,720 | **f32** |
| prefill ubatch ≥ 26 tokens (e.g. `-ub 2048`: 10.5 M) | ≥ 133,120 | **BF16** |
| prefill tail or short prompt ≤ 25 tokens | ≤ 128,000 | f32 |

**KLD** (gate corpus `p100-handoff/ppl-orig.txt`, `-c 4096`, 8 chunks, 16,376 tokens scored (the second half of each chunk), q4_0 KV, graphs `=3`):

| comparison | path exercised | PPL (test / base) | mean KLD | median | 99.9% | max KLD | same top token |
|---|---|---|---:|---:|---:|---:|---:|
| A vs A, `-ub 2048` (control) | — | 2.722840 / 2.722833 | −0.000006 | −0.000002 | 0.000004 | 0.000004 | 100.000% |
| **B vs A, `-ub 2048`** | **BF16 (prefill)** | 2.720638 / 2.722833 | **0.001358 ± 0.000031** | 0.000232 | 0.046274 | **0.192963** | 98.699% |
| **B vs A, `-ub 5`** | **f32 (decode-sized)** | 2.722189 / 2.719414 | **0.002104 ± 0.000047** | 0.000332 | 0.071585 | **0.241253** | 98.418% |
| **yardstick: A `-ub 5` vs A `-ub 2048`** (no NCCL involved) | — | 2.719413 / 2.722833 | **0.002313 ± 0.000053** | 0.000368 | 0.081448 | 0.239156 | 98.394% |

- **Is decode bit-exact under NCCL? No.**
  - The f32 path's saved logits differ from the non-NCCL build's: 8,939,339 of the first 268,435,456 bytes (3.3%) differ, and all 7,757 one-MiB chunks of the 8.1 GB file differ somewhere.
  - section 1.1 shows the practical effect: different greedy continuations and acceptance counts on the same prompt.
- **NCCL is deterministic with itself.** The two f32 saves are byte-identical (sha `3e913bdfc6588dd0` both).
- **Repeat-until-diverge** (HANDOFF #6 / OPTLOG 153 §8c method; one 16,384-token chunk, `-ub 2048`, saved logits hashed): **NCCL 8 of 8 identical** (sha `c1f8ab1397003e07`), no NaN, no divergence. Non-NCCL control 3 of 3 identical (sha `bcaf0f2d3f360297`).
- **Server second-request test (upstream issue 29466):** 7 chat requests per server (repeat, different prompt, follow-up turn, ~7.6k-token prompt, short prompt after it), default prompt cache.
  - Completed without an assert or crash on **B with MTP, B without MTP, A with MTP**, and on **B with MTP at `-c 65536`** (peak 9.9 GiB per GPU).
  - **The issue did not reproduce on any of the four**; this test does not show it fixed, only not triggered.
  - Repeating a prompt gave identical draft counts within a server.
- **Yardstick:** the non-NCCL build moves its own logits by KLD 0.0023 (98.39% same top token) when only `-ub` changes from 2048 to 5. That is the same size as NCCL's f32 deviation (0.0021) and larger than its BF16 deviation (0.0014).
- **Verdict: safe.**
  - NCCL's deviation, f32 and BF16 alike, is no larger than what this build already does to itself between ubatch sizes.
  - Perplexity is unchanged within error (ratio 0.9992 ± 0.0005 and 1.0010 ± 0.0006).
  - It is repeatable to the bit.
  - **Not bit-exact:** any gate that compares bytes or greedy text against a non-NCCL reference will fail.
- **Limits of this verdict:** one corpus, `-c 4096`, q4_0 KV; BF16 at 16k context was checked for repeatability, not KLD. Why sub-rounding differences (a different f32 summation order) amplify to KLD 0.002 was not investigated; the yardstick shows the amplification is a property of the build, not of NCCL.

## 1.4 Per-kernel matvec profile, graphs on

- **Tools and correction.**
  - **nvprof sees inside CUDA graphs** (it reports every graph-launched kernel), so kernel durations come from the graphs-on trace (`i4_nvprof_graphs3_n36`, last 32 tokens). Device durations are used as measured; nvprof's inflation (54.3 ms/token profiled vs 33.5 real) is host-side and does not enter this table.
  - nvprof gives no tensor shapes. Those come from the fork's `GGML_CUDA_OP_PROFILE=2`, which only records with graphs off (one run, 64 tokens).
  - The per-shape times from that run are scaled per quant type to the graphs-on totals (Q8_0 ×0.959, Q6_K ×1.034).
  - **Q5_K is left unscaled:** the op profile prints only its top 45 keys per device, so only 0.51 of Q5_K's 1.36 ms is listed.
  - The table covers 20.30 of the 21.18 ms/token matvec total (96%).
- **Columns.** "K × N" is the per-GPU slice (input width × rows). Bytes = K·N·bpw/8 (Q8_0 8.5, Q6_K 6.5625, Q5_K 5.5 bpw). Ceiling 605 GB/s (round 1). All per token, mean of the 3 GPUs.

| # | quant | K x N (per-GPU slice) | tensor (op name) | calls/token/GPU | MiB/call | us/call | GB/s | % of 605 | ms/token | share of matvec | ms lost to ceiling |
| 1 | q8_0 | 5120 x 3200 | node_ | 11.3 | 16.60 | 82.9 | 210 | 35% | 0.940 | 4.4% | 0.614 |
| 2 | q8_0 | 5120 x 15 | node_ | 44.0 | 0.08 | 12.9 | 6 | 1% | 0.567 | 2.7% | 0.561 |
| 3 | q8_0 | 5120 x 18 | node_ | 42.0 | 0.09 | 12.9 | 8 | 1% | 0.540 | 2.6% | 0.534 |
| 4 | q6_K | 5120 x 5888 | ffn_gate | 20.3 | 23.58 | 65.8 | 376 | 62% | 1.338 | 6.3% | 0.507 |
| 5 | q8_0 | 5760 x 5120 | ffn_out | 19.3 | 29.88 | 77.8 | 403 | 67% | 1.504 | 7.1% | 0.503 |
| 6 | q6_K | 5888 x 5120 | ffn_out | 20.0 | 23.58 | 62.7 | 394 | 65% | 1.255 | 5.9% | 0.437 |
| 7 | q8_0 | 1920 x 5120 | linear_attn_out | 22.0 | 9.96 | 35.4 | 295 | 49% | 0.780 | 3.7% | 0.400 |
| 8 | q6_K | 5120 x 5888 | ffn_up | 17.3 | 23.58 | 60.9 | 406 | 67% | 1.056 | 5.0% | 0.347 |
| 9 | q8_0 | 5120 x 5888 | ffn_up | 14.3 | 30.55 | 76.7 | 418 | 69% | 1.099 | 5.2% | 0.341 |
| 10 | q8_0 | 5120 x 5760 | ffn_up | 14.0 | 29.88 | 74.6 | 420 | 69% | 1.044 | 4.9% | 0.319 |
| 11 | q8_0 | 5120 x 3840 | node_ | 9.0 | 19.92 | 68.2 | 306 | 51% | 0.614 | 2.9% | 0.303 |
| 12 | q8_0 | 5120 x 5888 | ffn_gate | 10.3 | 30.55 | 80.5 | 398 | 66% | 0.832 | 3.9% | 0.285 |
| 13 | q8_0 | 5120 x 5760 | ffn_gate | 10.0 | 29.88 | 79.0 | 397 | 66% | 0.790 | 3.7% | 0.272 |
| 14 | q6_K | 5120 x 3200 | node_ | 10.7 | 12.82 | 45.1 | 298 | 49% | 0.481 | 2.3% | 0.244 |
| 15 | q6_K | 5120 x 3840 | node_ | 10.0 | 15.38 | 51.0 | 316 | 52% | 0.510 | 2.4% | 0.243 |
| 16 | q8_0 | 5888 x 5120 | ffn_out | 9.7 | 30.55 | 77.8 | 412 | 68% | 0.752 | 3.6% | 0.240 |
| 17 | q6_K | 5632 x 5120 | ffn_out | 10.0 | 22.56 | 62.5 | 378 | 63% | 0.625 | 3.0% | 0.234 |
| 18 | q6_K | 5120 x 5760 | ffn_gate | 8.7 | 23.07 | 65.0 | 372 | 62% | 0.563 | 2.7% | 0.216 |
| 19 | q6_K | 5120 x 5632 | ffn_gate | 8.0 | 22.56 | 63.9 | 370 | 61% | 0.511 | 2.4% | 0.199 |
| 20 | q8_0 | 2304 x 5120 | linear_attn_out | 11.0 | 11.95 | 38.5 | 326 | 54% | 0.423 | 2.0% | 0.195 |
| 21 | q6_K | 2304 x 5120 | linear_attn_out | 8.0 | 9.23 | 39.9 | 242 | 40% | 0.319 | 1.5% | 0.192 |
| 22 | q5_K | 5888 x 5120 | ffn_out | 3.3 | 19.77 | 83.6 | 248 | 41% | 0.279 | 1.3% | 0.164 |
| 23 | q6_K | 5120 x 5632 | ffn_up | 7.3 | 22.56 | 58.9 | 401 | 66% | 0.432 | 2.0% | 0.145 |
| 24 | q8_0 | 1536 x 5120 | attn_output | 8.7 | 7.97 | 29.9 | 280 | 46% | 0.259 | 1.2% | 0.139 |
| 25 | q8_0 | 5120 x 82773 | result_output | 0.7 | 429.43 | 924.6 | 487 | 80% | 0.616 | 2.9% | 0.120 |
| 26 | q6_K | 5120 x 3072 | Qcur_full | 5.3 | 12.30 | 41.5 | 311 | 51% | 0.221 | 1.0% | 0.108 |
| 27 | q6_K | 5120 x 5760 | ffn_up | 5.3 | 23.07 | 59.2 | 409 | 68% | 0.316 | 1.5% | 0.102 |
| 28 | q8_0 | 5120 x 3072 | Qcur_full | 4.0 | 15.94 | 50.9 | 328 | 54% | 0.204 | 1.0% | 0.093 |
| 29 | q8_0 | 5120 x 5632 | ffn_up | 3.3 | 29.22 | 74.5 | 411 | 68% | 0.248 | 1.2% | 0.079 |
| 30 | q5_K | 5120 x 5888 | ffn_gate | 1.3 | 19.77 | 84.4 | 246 | 41% | 0.113 | 0.5% | 0.067 |
| 31 | q8_0 | 3072 x 5120 | attn_output | 3.3 | 15.94 | 47.1 | 355 | 59% | 0.157 | 0.7% | 0.065 |
| 32 | q8_0 | 5120 x 5632 | ffn_gate | 2.3 | 29.22 | 77.1 | 398 | 66% | 0.180 | 0.8% | 0.062 |
| 33 | q8_0 | 5120 x 82774 | result_output | 0.3 | 429.43 | 926.1 | 486 | 80% | 0.309 | 1.5% | 0.061 |
| 34 | q8_0 | 5120 x 12 | node_ | 4.0 | 0.06 | 13.9 | 5 | 1% | 0.055 | 0.3% | 0.055 |
| 35 | q6_K | 5120 x 6144 | Qcur_full | 2.0 | 24.61 | 68.4 | 377 | 62% | 0.137 | 0.6% | 0.052 |
| 36 | q5_K | 5120 x 3840 | node_ | 1.0 | 12.89 | 62.0 | 218 | 36% | 0.062 | 0.3% | 0.040 |
| 37 | q8_0 | 5120 x 6144 | Qcur_full | 1.3 | 31.88 | 82.8 | 404 | 67% | 0.110 | 0.5% | 0.037 |
| 38 | q5_K | 5120 x 5632 | ffn_gate | 0.7 | 18.91 | 81.2 | 244 | 40% | 0.054 | 0.3% | 0.032 |

Tensor names behind the op labels (from the GGUF): `node_` 5120×3200 / 5120×3840 = `attn_qkv` (delta-net input projection, 5120×10240 whole); `node_` 5120×12 / 15 / 18 = `ssm_alpha` and `ssm_beta` (5120×48 whole); `linear_attn_out` = `ssm_out`; `ffn_out` = `ffn_down`; `Qcur_full` = `attn_q`; `result_output` = `output.weight`.

- **Matvec by quant** (graphs on, per token per GPU): Q8_0 12.02 ms (308 kernels), Q6_K 7.76 ms (161), Q5_K 1.36 ms (27), Q4_K 0.03 ms (1). Total 21.18 ms, matching section 2.1 (21.21).
- **8.6 of the listed 20.3 ms is above the 605 GB/s ceiling.**
- **The three worst weight shapes by milliseconds lost:**
  1. **`attn_qkv`, 5120×10240 (slices 3200 and 3840), 48 delta-net layers: 1.40 ms/token lost** (Q8_0 slices 0.61 + 0.30, Q6_K slices 0.24 + 0.24), running at 35–52% of the ceiling. The Q8_0 5120×3200 slice is the single worst row (210 GB/s).
  2. **`ssm_alpha` / `ssm_beta`, 5120×48 (12–18 rows per GPU), Q8_0: 1.15 ms/token lost.** 90 calls per token per GPU at 12.9 µs each to read 80 KB: 1% of the ceiling. These are latency-bound, not bandwidth-bound: a matvec launch for 15 rows.
  3. **The FFN matrices, 5120×17408 and 17408×5120 (slices 5632–5888):** the largest single rows after those are `ffn_gate` Q6_K 5120×5888 (0.51 ms lost) and `ffn_down` Q8_0 5760×5120 (0.50). Each runs at 62–69% of the ceiling, but together the six FFN classes lose **4.29 ms/token**, half of all the loss.
- **Also below 55% of the ceiling:** `ssm_out` (Q8_0 1920×5120 and 2304×5120: 49–54%, 0.60 ms lost), `attn_output` and `attn_q` slices (46–59%).
- **Close to the ceiling:** only the output head (`output.weight` slice 5120×82774, Q8_0) at 80%.
- **Reading:** the loss splits into three kinds: (a) wide FFN matrices at ~65% of ceiling, a kernel-efficiency gap; (b) mid-size projections (2,000–3,800 rows or columns) at 35–55%, a geometry or occupancy gap; (c) tiny `ssm_alpha/beta` matvecs that should not be separate kernel launches at all.
<!-- P16-LIVE-END -->


---

# Part 1b. Round 4: serving context, Q8_0, MTP under sampling, long-context KLD, agent battery

Same configurations A and B as Part 1, same build, XL unless stated. Raw material: `results/round4/`.

## 1.5 Context fit at `-c 262144`

`llama-server` with the serving flags (MTP on, n-max 3, p-min 0.0, one slot, `-fit off -lm none`). One prompt of
129,000 tokens (wikitext-2 raw test, sent as token ids, prompt cache off), then 64 generated tokens. One run per cell.

| config | `-c` | peak MiB, GPU 0 / 1 / 2 | prompt tokens processed | prefill t/s | decode t/s (MTP) |
|---|---:|---|---:|---:|---:|
| A | 262144 | 11027 / 10929 / 11017 | 129,000 | 211.0 | 27.0 |
| B | 262144 | 11125 / 11027 / 11115 | 129,000 | 315.9 | 26.6 |
| A | 131072 | 10199 / 10137 / 10189 | 129,000 | 210.6 | 27.1 |
| B | 131072 | 10297 / 10235 / 10287 | 129,000 | 315.7 | 26.6 |
| A, Q8_0 | 262144 | 12123 / 12023 / 12123 | 129,000 | 211.1 | — |
| B, Q8_0 | 262144 | 12221 / 12121 / 12221 | 129,000 | 318.1 | — |

- No out-of-memory failure, truncation or context shift in any run.
- Both configurations run `-c 262144` with more than 5 GB free per 16 GB card (XL) and more than 4 GB (Q8_0).
- NCCL adds about 100 MiB per card. Going from `-c 131072` to `-c 262144` adds about 830 MiB per card.
- The prompt filled half of the 262144 context. A prompt near the full context was not run.

## 1.6 Q8_0 against the XL, graphs on

Mean ± stdev of 3 llama-bench repetitions; MTP cells are 3 separate invocations (greedy, n-max 3, p-min 0.0, 256
tokens). Order alternated per test.

| test, t/s | A XL | A Q8_0 | B XL | B Q8_0 |
|---|---:|---:|---:|---:|
| tg512 | 29.81 ± 0.07 | 27.73 ± 0.15 | 31.23 ± 0.06 | 29.06 ± 0.07 |
| pp2048 at depth 0 | 276.25 ± 0.36 | 277.14 ± 0.31 | 502.34 ± 0.55 | 508.09 ± 0.79 |
| pp2048 at depth 16384 | 259.80 ± 0.37 | 261.13 ± 0.23 | 445.04 ± 0.74 | 449.94 ± 0.88 |
| MTP chat | 37.62 ± 0.74 | 35.86 ± 0.04 | 40.03 ± 0.04 | 35.74 ± 0.04 |
| MTP code | 56.09 ± 0.03 | 49.91 ± 0.05 | 55.91 ± 0.12 | 50.17 ± 0.05 |
| MTP summ ~8k | 41.79 ± 0.07 | 39.18 ± 0.04 | 39.99 ± 0.04 | 40.13 ± 0.02 |

- Q8_0 is 7.0% slower than the XL in plain decode on both configurations, and 0.3–1.1% faster in prefill.
- With MTP, Q8_0 is 5–11% slower on five of the six cells and level on one (B, summ).
- Round 1 measured Q8_0 as equal in decode; that was with graphs off, where decode was host-bound (section 3.7).
- **These MTP cells are not comparable with section 1.2.** They were run with `LLAMA_SPEC_SAMPLE_TEMP=1.0
  LLAMA_SPEC_DRAFT_TOPK=20` set in the environment (the serving configuration's draft settings), section 1.2 without.
  The A XL cells here are 4–9% below section 1.2's n-max 3 / p-min 0.0 row.

## 1.7 MTP under sampling

`llama-speculative-simple`, n-max 3, 256 tokens, `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0` with
`LLAMA_SPEC_SAMPLE_TEMP=1.0 LLAMA_SPEC_DRAFT_TOPK=20`. Three prompts, seeds 1, 2, 3 in every cell. Cells: mean ± stdev
t/s over the 3 seeds.

| config | p-min | chat | code | summ ~8k | mean t/s | mean acceptance | ms per accepted draft token |
|---|---|---:|---:|---:|---:|---:|---:|
| A | 0.0 | 35.98 ± 6.05 | 49.18 ± 5.07 | 38.73 ± 2.74 | **41.3** | 50.5% | 46.7 |
| A | 0.5 | 31.59 ± 3.00 | 48.21 ± 5.68 | 33.92 ± 1.11 | 37.9 | 63.1% | 53.2 |
| A | 0.75 | 30.36 ± 1.30 | 43.20 ± 2.74 | 30.60 ± 0.34 | 34.7 | 76.3% | 61.2 |
| B | 0.0 | 38.26 ± 2.96 | 54.78 ± 1.95 | 37.75 ± 2.89 | **43.6** | 54.6% | 46.2 |
| B | 0.5 | 33.52 ± 1.56 | 48.09 ± 3.45 | 33.59 ± 1.25 | 38.4 | 63.0% | 54.1 |
| B | 0.75 | 31.08 ± 1.49 | 45.36 ± 3.62 | 31.99 ± 1.54 | 36.1 | 76.4% | 62.6 |

- p-min 0.0 is the fastest setting under sampling on both configurations and all three prompts, as it was under
  greedy decoding (section 1.2).
- A higher p-min raises acceptance and lowers throughput: fewer tokens are drafted.
- The seed-to-seed standard deviation is up to 6 t/s, larger than the A-to-B difference in most cells.
- The fork's default (n-max 4, p-min 0.2) was not in this grid either.

## 1.8 NCCL KLD at `-c 16384`

Stock `llama-perplexity`, wikitext-2 raw test, `-c 16384`, 4 chunks, q4_0 KV. It scores the second half of each
chunk: positions 8192–16382, 32,764 tokens. Each run saved its logits; pairs of files were compared with
`scripts/kldpos.cpp`.

| pair | mean KLD | median | 99.9% | max | same top token |
|---|---:|---:|---:|---:|---:|
| NCCL against non-NCCL, both `-ub 2048` | 0.003104 | 0.000452 | 0.330 | 14.71 | 98.407% |
| non-NCCL `-ub 5` against non-NCCL `-ub 2048` | 0.007028 | 0.000796 | 0.842 | 14.83 | 98.001% |

Mean KLD per 1,024-position bucket (4,096 tokens per bucket):

| positions | NCCL vs non-NCCL | `-ub 5` vs `-ub 2048` |
|---|---:|---:|
| 8192–9215 | 0.00229 | 0.00458 |
| 9216–10239 | 0.00210 | 0.01112 |
| 10240–11263 | 0.00172 | 0.00294 |
| 11264–12287 | 0.00582 | 0.01300 |
| 12288–13311 | 0.00339 | 0.00593 |
| 13312–14335 | 0.00424 | 0.00703 |
| 14336–15359 | 0.00255 | 0.00739 |
| 15360–16382 | 0.00272 | 0.00423 |

- At 16k the NCCL difference is again smaller than the difference from changing `-ub` on the non-NCCL build.
- Both means are 2–3 times their `-c 4096` values (0.00136 and 0.00231, section 1.3). The corpus also differs
  between the two measurements, so this is not a clean measure of growth with context.
- Within 8192–16382 there is no steady rise with position. Bucket means are dominated by a few tokens (the maximum
  of 14.7 falls in the 11264–12287 bucket of both pairs).
- Perplexity: 5.5947 (A, `-ub 2048`), 5.5880 (B, `-ub 2048`), 5.6259 (A, `-ub 5`), each ± 0.076.
- Positions 0–8191 are not scored by the tool. `-c 65536` was not run: the tool holds every scored token's logits
  in host memory (about 32 GB there, against 16 GB installed).
- **Correction to section 1.3:** the `-c 4096` runs scored 16,376 tokens (the second half of each of 8 chunks), not
  32,768. The KLD values there are unaffected.

## 1.9 Agent battery, A against B

A tool-calling coding agent driven by an evaluation harness, pointed at `llama-server` on the test machine: nine
error-recovery tasks, 600 s limit per task, XL, `-c 262144`, MTP n-max 3 / p-min 0.0, the serving sampler. One
server per block; blocks alternate A B, B A, A B (one repetition of all nine tasks each), then B A, A B, B A (one
extra repetition of the two slowest tasks each). 33 task runs per configuration. The harness and its tasks are not
part of this repository.

| | A | B |
|---|---:|---:|
| passed, all 33 runs | 26 (79%) | 28 (85%) |
| passed, first 3 repetitions of each task (27 runs) | 23 (85%) | 24 (89%) |
| passed, excluding `err_big_file_read` (27 runs) | 26 (96%) | 26 (96%) |
| mean wall time per task, all 33 runs | 223 s | 166 s |
| mean wall time per task, first 3 repetitions | 183 s | 124 s |
| runs killed at the 600 s limit | 6 | 4 |
| server prefill, all 150 requests | 242.2 t/s | 377.9 t/s |
| server decode with MTP, all requests | 48.1 t/s | 48.9 t/s |
| draft acceptance, all requests | 71.2% | 73.3% |
| peak MiB, GPU 0 / 1 / 2 | 11027 / 10929 / 11017 | 11125 / 11027 / 11115 |

| task | n | A passed | A mean wall | B passed | B mean wall |
|---|---:|---:|---:|---:|---:|
| err_python_env | 3 | 3 | 109 s | 3 | 94 s |
| err_replay_patch | 3 | 3 | 104 s | 3 | 69 s |
| err_ambiguous_edit | 3 | 3 | 119 s | 3 | 83 s |
| err_case_search | 3 | 3 | 133 s | 3 | 86 s |
| err_hidden_search | 3 | 2 | 104 s | 2 | 69 s |
| err_big_output | 3 | 3 | 109 s | 3 | 66 s |
| err_multi_dir | 3 | 3 | 103 s | 3 | 74 s |
| err_inline_script | 6 | 6 | 237 s | 6 | 206 s |
| err_big_file_read | 6 | 0 | 600 s | 2 | 433 s |

- **B's prefill advantage shows up in task wall time.** Every task is faster on B (13–39%); the seven 3-repetition
  tasks average 112 s on A and 77 s on B. Decode speed is the same on both.
- **Every killed run is `err_big_file_read`:** 6 of 6 on A, 4 of 6 on B. With six runs per configuration the
  difference between A and B on that task is not established.
- **One run per configuration failed without timing out:** `err_hidden_search` (exit code 0, a wrong answer: the
  agent listed one of the two matching files and missed the one in a hidden directory). A repetition 2, B
  repetition 0.
- No new warning or error appeared in the server logs of either configuration.
- An earlier run of the same battery (3 repetitions, a different quant file, UD-Q5_K_XL, before these settings)
  passed 24 of 27 (89%; 23 of 24 excluding `err_big_file_read`) at a mean of 187 s per task and 207.5 t/s server
  prefill. The model file differs, so it is context, not a controlled comparison.

---

# Part 1c. Round 5: NCCL tuning, power cap, concurrent clients, `-ub`, 65k KLD, soak

Configuration B (NCCL build, `NCCL_P2P_LEVEL=SYS`, graphs `=3`, `-lm none`), XL. Raw material: `results/round5/`.
One script fault occurred in this round; it is described in section 1.11.

## 1.10 NCCL tuning

One variable at a time against the default, `NCCL_P2P_LEVEL=SYS` kept. llama-bench `-r 3` per cell; six default runs
per test, spread through the sweep. Rule set before the runs: a gain counts only if it is at least 3% and more than
2 standard deviations.

**What NCCL picks by default here** (from `NCCL_DEBUG=INFO`): Ring algorithm, 2 channels, P2P direct-pointer
transport. Decode-sized exchanges (20,480 bytes, f32) use protocol LL on 1 channel; prefill-sized exchanges (BF16)
use protocol Simple on both channels.

| setting | tg512 t/s | ratio | pp2048, depth 0, t/s | ratio | pp2048, depth 16384, t/s | ratio |
|---|---:|---:|---:|---:|---:|---:|
| default (6 runs) | 31.27 ± 0.06 | 1.000 | 506.63 ± 0.51 | 1.000 | 448.84 ± 0.74 | 1.000 |
| `NCCL_ALGO=Ring` | 31.28 ± 0.05 | 1.000 | 506.78 ± 0.45 | 1.000 | 449.06 ± 1.16 | 1.000 |
| `NCCL_ALGO=Tree` | 28.47 ± 0.05 | 0.910 | 482.05 ± 0.31 | 0.951 | 430.16 ± 0.17 | 0.958 |
| `NCCL_PROTO=LL` | 31.24 ± 0.08 | 0.999 | 422.44 ± 0.37 | 0.834 | 380.73 ± 0.18 | 0.848 |
| `NCCL_PROTO=LL128` | 29.68 ± 0.07 | 0.949 | 502.01 ± 0.64 | 0.991 | 444.34 ± 0.87 | 0.990 |
| `NCCL_PROTO=Simple` | 28.98 ± 0.05 | 0.927 | 506.69 ± 0.59 | 1.000 | 448.93 ± 1.07 | 1.000 |
| `NCCL_BUFFSIZE=4194304` (the default size) | 31.25 ± 0.06 | 0.999 | 506.74 ± 0.64 | 1.000 | 448.56 ± 0.62 | 0.999 |
| `NCCL_BUFFSIZE=16777216` | 31.22 ± 0.07 | 0.998 | 508.81 ± 0.77 | 1.004 | 449.77 ± 0.74 | 1.002 |
| 1 channel (`NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1`) | 31.24 ± 0.05 | 0.999 | 514.80 ± 0.63 | 1.016 | 453.46 ± 0.82 | 1.010 |
| 2 channels | 31.26 ± 0.06 | 1.000 | 506.19 ± 0.76 | 0.999 | 448.99 ± 1.19 | 1.000 |
| 4 channels | 31.25 ± 0.05 | 0.999 | 501.18 ± 0.71 | 0.989 | 443.87 ± 0.88 | 0.989 |
| 1 channel + 16 MiB buffer | 31.24 ± 0.06 | 0.999 | 515.38 ± 0.25 | 1.017 | 455.18 ± 1.07 | 1.014 |

- **No setting beats the default by the 3% rule.** The closest is 1 channel: +1.6% and +1.0% prefill, decode
  unchanged. It is repeatable (run-to-run spread is 0.1–0.2%) and small.
- Nothing was rejected or ignored. The channel count is adjustable on this topology (NCCL reported 1, 2 and 4
  channels as requested).
- Because no setting qualified, the bit-exactness check planned for a winner was not run.

## 1.11 Power cap

The cards are capped at 125 W on this machine (the cap's minimum; the default is 250 W). During configuration-B runs
at 125 W the driver's power-cap reason was active in up to 80% of busy samples on a card in decode and 30–50% in
prefill (1 Hz samples; busy = utilization above 50%). The cap was then swept with `nvidia-smi -pl` on all three
cards. llama-bench `-r 3`; two invocations per raised cap and three at 125 W, alternated with 125 W.

| cap | tg512 t/s | pp2048, depth 0, t/s | pp2048, depth 16384, t/s | GPU power in decode, sum of 3 | GPU + CPU package peak | hottest card | decode t/s per GPU watt |
|---|---:|---:|---:|---:|---:|---:|---:|
| 125 W | 31.25 ± 0.07 | 506.00 ± 0.38 | 448.28 ± 0.76 | 356 W | 454 W | 54 °C | 0.0877 |
| 150 W | 33.29 ± 0.10 (+6.5%) | 530.42 ± 1.05 (+4.8%) | 471.27 ± 0.68 (+5.1%) | 430 W | 500 W | 58 °C | 0.0775 (−12%) |
| 175 W | 33.66 ± 0.18 (+7.7%) | 548.98 ± 0.32 (+8.5%) | 490.52 ± 0.84 (+9.4%) | 484 W | 562 W | 60 °C | 0.0696 (−21%) |

- 125 to 150 W buys 2.0 t/s of decode for 74 W of GPU power. 150 to 175 W buys 0.4 t/s of decode for 54 W; prefill
  keeps gaining (+3.5% and +4.1%).
- At 175 W decode is no longer power-capped (0–4% of busy samples); prefill still is on some cards (up to 40%).
- CPU package power is from RAPL; the motherboard, memory, drives and fans are not in the telemetry.
- **Script fault.** The run harness's cool-down function overwrote the shell variable that the sweep loop used for the
  test name. The first attempt at the alternated second run per cap therefore ran llama-bench's default test instead
  of the intended one. Those runs are tagged `i2__pl*` in `results/round5/runs.csv` and are excluded. The queue was
  stopped and restarted with the variable renamed, and the proper second runs were added; the table uses those.
  Section 1.13 would have hit the same fault and was run only with the corrected script.

## 1.12 Concurrent clients, and `-np 2`

`llama-server`, `-c 262144`, MTP n-max 3 / p-min 0.0, serving sampler, `-b 32768`. One server per arm; each scenario
3 times. (a) client 1 sends a 64,000-token prompt, client 2 a short chat 5 s later. (b) two short chats at once.
(c) the 64,000-token request is dropped by its client halfway through prefill.

| | `-np 1` | `-np 2 --kv-unified` | `-np 2 --no-kv-unified` |
|---|---:|---:|---:|
| context per slot | 262144 | 262144 (shared) | 131072 |
| peak MiB per card | 11173 / 11075 / 11163 | 12383 / 12285 / 12371 | 11849 / 11751 / 11837 |
| (a) second client, time to first token | 162.5 s | 191.9 s | 163.8 s |
| (a) first client, 64k prompt total | 166.6 s (386 t/s) | 198.6 s (325 t/s) | 170.6 s (379 t/s) |
| (b) two chats at once, aggregate | 37.2 t/s | 45.4 t/s | 47.3 t/s |
| (b) per-client decode | 37–50 t/s, one after the other | 23–32 t/s, together | 23–32 t/s, together |
| single client, chat decode (2 requests) | 49.0 t/s | 41.0 t/s | 46.8 t/s |
| single client, 20k-token prompt | 450.3 t/s | 439.4 t/s | 445.3 t/s |

- **`-np 2` works with `-sm tensor`, NCCL and MTP on this build**, with and without `--kv-unified`: no flag was
  rejected and no request failed.
- **It does not let a second client in during a long prefill.** With two slots the second client still waited for
  the first client's whole prefill.
- Two simultaneous short chats get 22–27% more aggregate throughput on two slots, each at about 60% of single-client
  speed.
- Cost of `-np 2`: 1.2 GB more per card with `--kv-unified`, 0.7 GB without; `--kv-unified` also slowed the 64k
  prefill by 16%.
- **The (c) timings of this round are not reported here.** They were taken while the client polled `/slots` five
  times a second, which itself delays the server's disconnect check (section 1.19). The raw (c) records are in
  `results/round5/` for reference.

## 1.13 `-ub` under NCCL

| `-ub` | pp2048, depth 0, t/s | pp2048, depth 16384, t/s |
|---|---:|---:|
| 512 | 437.33 ± 1.24 (−13.5%) | 390.31 ± 0.52 (−12.9%) |
| 1024 | 471.14 ± 0.53 (−6.8%) | 419.45 ± 0.16 (−6.4%) |
| 2048 | 505.55 ± 0.62 | 447.99 ± 0.68 |
| 4096 | 505.99 ± 0.28 (+0.1%) | 448.46 ± 1.00 (+0.1%) |

- A 2,048-token prompt is one micro-batch at both 2048 and 4096, so this table cannot separate them; section 1.17
  repeats the comparison with an 8,192-token prompt.
- None of these values changes which exchanges use BF16: the switch is at 131,072 elements with 3 backends
  (`ggml-cuda.cu`), 26 tokens of 5,120 values, and every prefill micro-batch here is far above it.

## 1.14 NCCL KLD at `-c 65536`

One 65,536-token context of wikitext-2 raw test (articles concatenated, unmodified), every position scored (65,535
tokens). Stock `llama-perplexity` holds every scored token's logits in host memory (about 32 GB here, against 16 GB
installed), so an out-of-tree tool decoded the context in 2,048-token batches and wrote each position's 16-bit
log-probabilities to disk in `llama-perplexity`'s record format; `scripts/kldpos.cpp` compared pairs of files. The
tool reproduces `llama-perplexity`'s saved file exactly at `-c 4096` (0 of 2,047 records differ).

| positions | NCCL vs non-NCCL: mean KLD | max | same top token | non-NCCL `-ub 5` vs `-ub 2048`: mean KLD | max | same top token |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.000891 | 1.687 | 98.622% | 0.001516 | 0.555 | 98.245% |
| 0–4,095 | 0.000688 | 0.137 | 98.779% | 0.001287 | 0.203 | 98.511% |
| 4,096–16,383 | 0.000873 | 0.449 | 98.649% | 0.001587 | 0.555 | 98.446% |
| 16,384–32,767 | 0.000902 | 0.213 | 98.499% | 0.001482 | 0.188 | 98.047% |
| 32,768–65,534 | 0.000919 | 1.687 | 98.654% | 0.001535 | 0.532 | 98.236% |

- NCCL's difference is about 0.6 of the non-NCCL batch-size difference in every bucket, and does not grow with
  position after the first 4k.
- Repeatability: three more NCCL runs at `-c 65536` and the saved run gave the same hash of all records, 4 of 4.
- One corpus and one context. The dump tool is not included (see `results/README.md`).

## 1.15 Server soak, 90 minutes

One slot, `-c 262144`, MTP on, serving sampler. One client, one request at a time: short chats and code requests
alternating, a 20,000-token prompt every sixth step, a 64,000-token prompt about every 10 minutes, each long prompt
preceded by a short request.

| request | n | prefill t/s, first 15 min | last 15 min | decode t/s, first 15 min | last 15 min |
|---|---:|---:|---:|---:|---:|
| chat | 141 | — | — | 44.0 | 42.9 |
| code | 96 | — | — | 45.0 | 43.9 |
| 20k-token prompt | 47 | 448.3 | 447.7 | 40.3 | 38.5 |
| 64k-token prompt | 9 | 385.2 | 384.2 | 38.9 | 40.9 |

- 349 requests, none failed, no crash. Peak memory was 11173 / 11075 / 11163 MiB in every 10-minute window.
- Prefill was flat. Decode on chat and code requests was 2.5% lower in the last 15 minutes than the first.
- Cards peaked at 61 / 64 / 61 °C. The server log has four warning lines, all at startup.
- 56 short-then-long request pairs ran without the second-request failure of upstream issue 29466.

---

# Part 1d. Round 6: `-b`, client drops during prefill, and the settings now served

Configuration B, XL, `-c 262144`, MTP n-max 3 / p-min 0.0, one slot. Raw material: `results/round6/`.

## 1.16 `-b` sweep (125 W)

One server per value. (a) a 64,000-token prompt alone, once; (b) the same prompt with a second client's short chat
5 s later, twice; (d) a 512-token generation, three seeds. Fewer repetitions than the other rounds, to fit a
90-minute limit.

| `-b` | (a) 64k prefill t/s | vs 32768 | (b) second client, time to first token | (b) first client total | (d) decode t/s |
|---|---:|---:|---:|---:|---:|
| 32768 | 388.4 | 1.000 | 162.5 s | 166.5 s | 37.8 |
| 8192 | 386.3 | 0.995 | 163.3 s | 167.3 s | 36.4 |
| 4096 | 387.8 | 0.998 | 163.6 s | 167.6 s | 38.2 |
| 2048 | 385.4 | 0.992 | 164.1 s | 168.1 s | 37.1 |

- A smaller `-b` costs under 1% of prefill speed and leaves decode unchanged.
- It does not shorten a second client's wait on one slot.
- This round also timed a dropped request at each `-b`, with `/slots` polled five times a second. Those timings
  showed only the polling effect of section 1.19 and are superseded by sections 1.18 and 1.19.

## 1.17 `-ub` with an 8,192-token prompt (125 W)

| test | `-ub 2048` t/s | peak MiB per card | `-ub 4096` t/s | peak MiB per card | 4096 / 2048 |
|---|---:|---|---:|---|---:|
| pp8192, depth 0 | 498.25 ± 0.89 | 10565 / 10563 / 10581 | 504.63 ± 0.84 | 12971 / 12969 / 12987 | 1.013 |
| pp8192, depth 16384 | 436.68 ± 0.88 | 10659 / 10657 / 10681 | 440.24 ± 0.86 | 13065 / 13063 / 13087 | 1.008 |

- `-ub 4096` gains 1.3% and 0.8% and costs 2.4 GB more per card. `-ub 2048` was kept.

## 1.18 A client drop during prefill: how long the slot stays busy

A 64,000-token streaming request; the client closes the connection 78 s after sending, about half of the prefill at
150 W. Times are differences of server-log timestamps. No other request or call reaches the server until the log
shows the slot released (the client follows the container's log, which makes no call to the server); then one short
chat is sent. 3 repetitions, alternating with the scenario of section 1.19.

| `-b` | drop to the server's "cancel task" line | drop to slot release | drop to next request accepted | prompt tokens in the slot at release |
|---|---|---|---|---:|
| 32768 | 0.01, 0.01, 0.01 s | 72.3, 73.9, 74.0 s | 72.6, 74.2, 74.3 s | 61,948 |
| 4096 | 0.02, 0.01, 0.01 s | 3.8, 4.0, 3.8 s | 4.2, 4.3, 4.2 s | 36,864 |
| 2048 | 0.02, 0.01, 0.02 s | 3.8, 4.3, 4.0 s | 4.3, 4.8, 4.2 s | 36,864 |

- The server notices the drop at once, but acts on a cancel only between `llama_decode` calls, and `-b` sets how
  many prompt tokens go into one call. At `-b 32768` the 64,000-token prompt is four calls (32,768, 29,180, 2,048 and
  4 tokens); the drop fell at the start of the second, which ran to its end.
- At `-b 2048` and `-b 4096` the slot was released about 4 s after the drop. Both released at 36,864 tokens, a
  boundary of both batch sizes, so these runs do not separate them. The expected worst case, one full batch (about
  5 s at 2048, about 10 s at 4096), is inferred, not measured.
- The `-b 32768` rows were taken on the serving configuration itself; the other two in a test container with the
  same settings.

Server log around the drop at `-b 2048` (drop at 2.54.87):

```
2.53.558.528 I slot print_timing: id  0 | task 9 | prompt processing, n_tokens =  34816, progress = 0.54, t =  75.07 s / 463.79 tokens per second
2.54.890.647 W srv          stop: cancel task, id_task = 9
2.58.686.633 I slot      release: id  0 | task 9 | stop processing: n_tokens = 36864, truncated = 0
2.59.124.797 I slot launch_slot_: id  0 | task 29 | processing task, is_child = 0
```

## 1.19 The same drop while another client polls `/slots`

As section 1.18, with a second connection requesting `/slots` five times a second throughout (about 790 answered
requests per run), and separately with `/health` requested every 5 s.

| `-b` | other traffic | drop to "cancel task" | drop to slot release |
|---|---|---|---|
| 32768 | `/slots`, 5 per second | 79.5, 81.2, 80.5 s | 79.6, 81.2, 80.5 s |
| 4096 | `/slots`, 5 per second | 80.0, 79.7, 79.1 s | 80.1, 79.8, 79.2 s |
| 2048 | `/slots`, 5 per second | 79.7, 79.3, 79.8 s | 79.8, 79.4, 79.8 s |
| 32768 | `/health`, every 5 s | 0.01, 0.01, 0.01 s | 73.8, 74.0, 74.1 s |

- **Polling `/slots` five times a second delays the server's disconnect check until prefill has finished**, at every
  `-b`: the "cancel task" line appears only after the first generated token, about 80 s after the drop.
- `/health` every 5 s has no effect.
- Reading of the code (not a measurement): the handler waiting for a request's first result tests for a closed
  connection only when a 1-second wait on a condition variable times out, and every result delivered for any request
  wakes all waiters and restarts that wait. `/slots` requests are answered during a decode, so five a second keep the
  wait from ever timing out. The same code is in upstream llama.cpp master as of 2026-10-05. No code was changed.

## 1.20 Cost of `-b 2048` at 150 W

One server per value; three 64,000-token prompts alone, then three 512-token generations.

| | `-b 2048` | `-b 32768` |
|---|---|---|
| 64k prefill t/s, repetitions 1 / 2 / 3 | 406.7 / 402.5 / 401.0 | 406.7 / 404.4 / 403.1 |
| 64k prefill t/s, mean | 403.4 | 404.8 |
| 512-token decode t/s, seeds 1 / 2 / 3 | 38.3 / 38.7 / 40.1 | 38.6 / 36.1 / 39.4 |

- Prefill is 0.3% lower at `-b 2048`; decode differences are within seed-to-seed spread.
- The two values ran one after the other, and only the first prefill repetition of each followed a cool-down.

## 1.21 Settings now served on the test machine

```
environment: GGML_CUDA_P2P=1  GGML_CUDA_GRAPHS_PRE_VOLTA=3  NCCL_P2P_LEVEL=SYS
             LLAMA_SPEC_SAMPLE_TEMP=1.0  LLAMA_SPEC_DRAFT_TOPK=20
llama-server (NCCL build) -m Qwen3.8-27B-UD-Q6_K_XL.gguf --jinja --cache-ram 0 --no-cache-idle-slots
    --parallel 1 -c 262144 -ngl 99 -sm tensor -ts 1/1/1 -fa 1 -ctk q4_0 -ctv q4_0 -b 2048 -ub 2048
    -lm none -fit off --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0
    -ngld 99 -ubd 64 -ctkd q4_0 -ctvd q4_0 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0.0
power cap: 150 W per card
```

Checks after starting it (single requests, from a second machine on the LAN):

| request | prompt tokens | prefill t/s | generated | decode t/s |
|---|---:|---:|---:|---:|
| short chat | 64 | — | 53 | 40.1 |
| second request right after | 21 | — | 200 | 49.7 |
| 64,000-token prompt | 64,000 | 404.7 | 64 | 37.4 |

- Startup: MTP draft context created, `-lm none` and `-fit off` accepted, five warning lines (no API key, CPU
  sampler under tensor split twice, a reasoning-template note, a notice about a future default port).
- Peak during the checks: 61 / 63 / 60 °C, 11,125 / 11,027 / 11,115 MiB.
- With `-b 32768` at 150 W the same 64,000-token prompt ran at 406.9 t/s.

---

# Part 2. Round 2: where the time goes, and what else was measured


## 2.1 Per-token decode budget, graphs off and on

Method: nvprof `--print-gpu-trace --print-api-trace`, llama-bench `-n 36` (built-in warmup), last 32 tokens, token boundary = the device-0 logits DtoH copy. **Profiler correction:** kernel and copy durations are taken as real (nvprof does not inflate device time); real idle = unprofiled tg512 ms/token − trace busy, split exchange-adjacent vs other in the trace's proportions (round 1 method B). Host time comes from the **unprofiled** server timeline (`LLAMA_TL=1`), because nvprof inflates host API calls (token time ×2.14 graphs-off, ×1.41 graphs-on). Analyzer: `scripts/analyze_i1.py`.

| ms/token (mean of 3 GPUs) | graphs off (round 1 config) | % | graphs on (`GGML_CUDA_GRAPHS_PRE_VOLTA=3`, fork serving default) | % |
|---|---:|---:|---:|---:|
| real token time (tg512, ABBA, 2 runs each) | **44.6** (22.42 t/s) | 100 | **33.8** (29.57 t/s) | 100 |
| matvec (`mul_mat_vec_q`) | 21.38 | 47.9 | 21.21 | 62.7 |
| activation quantize for matvec | 0.88 | 2.0 | 1.19 | 3.5 |
| full attention (FA, rope, KV write, FWHT), 17 layers | 1.04 | 2.3 | 1.70 | 5.0 |
| delta-net (GDN, conv, gating), 48 layers | 0.88 | 2.0 | 1.26 | 3.7 |
| norms / other kernels | 2.15 | 4.8 | 2.62 | 7.8 |
| exchange ops (peer copies + adds) | 1.36 | 3.1 | 1.35 | 4.0 |
| exchange-adjacent idle | 8.42 | 18.9 | 2.44 | 7.2 |
| other idle (launch gaps / host-bound) | 8.57 | 19.2 | 2.16 | 6.4 |
| per-GPU imbalance (max − mean of non-exchange busy) | 0.10 | 0.2 | 0.14 | 0.4 |

- **Exchanges:** 128 per token (512 peer copies + 384 ADDs per token; 4 copies per exchange, generic butterfly), **20 KB f32** per copy (5120 floats).
- **Host (unprofiled, server timeline, median of 510 steps):** graphs off: `llama_decode` enqueue **43.6 ms**, then 0.96 ms waiting for the GPUs; step 45.3 ms. **Decode is host-bound:** the GPUs finish ~1 ms after the host stops enqueuing. Graphs on: enqueue 17.7 ms, then **15.4 ms waiting** for the GPUs; step 33.9 ms. **Now GPU-bound.**
- **Host API per token (profiled counts):** graphs off: 19,174 calls (5,274 kernel launches, 2,048 `cudaStreamWaitEvent`, 1,304 `cudaEventRecord`, 512 `cudaMemcpyPeerAsync`, 905 set/get device). Graphs on: 7,276 calls (387 `cudaGraphLaunch`, 384 kernel launches = the exchange ADDs, 2,048 waits, 1,304 records, 512 peer copies). About 4,250 of the 7,276 are the exchange's per-call path.
- **Host enqueue by device (profiled, by Correlation_ID):** graphs off: GPU0 33.9 / GPU1 18.4 / GPU2 16.1 ms (GPU0 carries the butterfly's fold and swap); graphs on: 24.1 / 7.9 / 5.8 ms.
- **CUDA graphs:** with `=3`, **129 `cudaGraphLaunch` per device per token** (one per segment between the 128 exchanges), 0 captures, instantiates or updates per token (no re-capture), and 0 `GGML_CUDA_GRAPH_DEBUG` update lines. The graph covers all three devices, but split into 129 pieces each; the exchange ADDs stay outside it (1-node graphs are excluded on purpose). Graphs off (round 1 and llama-bench default): bypassed entirely.
- **Inside graphs, the non-matvec kernels read ~1.6 ms/token slower** (attention 1.04→1.70, delta-net 0.88→1.26, quantize 0.88→1.19). This is profiler-measured; treat it with caution.
- **Biggest unexplained piece:** graphs off, it is host enqueue (43.6 of 45.3 ms). Round 1's 19% "exchange-adjacent idle" for decode was largely host-bound waiting, not link time. Graphs on, it is **matvec efficiency**: 21.2 ms/token for ~7.4 GiB per GPU = ~375 GB/s against the 605 GB/s measured ceiling, i.e. ~8 ms/token unrecovered. After matvec: the exchange path (ops + adjacent idle 3.8 ms, 11%) and ~2.2 ms of other idle.
- **Consequence for round 1:** Round 1 decode numbers (21.4–21.6 t/s) were taken with graphs off, i.e. host-bound. 3-GPU decode with the fork's serving default is **29.6 t/s**.

## 2.2 AllReduce shootout (out of tree)

Latency in µs, end to end (event-timed on GPU0, from all three partials ready to all three results done; iterations enqueued behind a 30 ms GPU sleep so host enqueue cost is excluded). `bytes` = f32 vector size.

| bytes (f32) | a meta pattern f32 | a f16 | b one-shot f32 | b f16 | c two-shot f32 | c f16 | d ring f32 | d f16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 KB | 49.4 | 47.0 | 34.8 | **30.5** | 43.6 | 41.6 | 43.7 | 42.4 |
| **20 KB (decode, 5120 f32)** | 49.4 | 49.1 | 36.9 | **31.3** | 44.6 | 41.9 | 44.1 | 42.4 |
| 64 KB | 81.9 | 71.9 | 60.1 | **43.6** | 61.7 | 50.2 | 60.4 | 49.6 |
| 128 KB | 141.3 | 97.5 | 98.8 | 61.4 | 90.1 | 62.9 | 87.6 | **61.9** |
| 256 KB | 239.9 | 141.6 | 171.9 | 106.5 | 152.2 | 98.1 | 144.1 | **95.8** |
| 4 MB | 3302 | 1736 | 2530 | 1414 | 1984 | 1126 | 1795 | **1010** |
| 20 MB | 16361 | 8526 | 12867 | 6819 | 9672 | 5537 | 8835 | **5034** |
| **40 MB (prefill ubatch 2048 × 5120 f32)** | 32678 | 16999 | 25877 | 13592 | 19743 | 11091 | 18045 | **10073** |

- **Bit-identical across the 3 GPUs:** b, c, d **yes at every size and both wires** (fixed summation order, one owner per element; with f16 wire every GPU keeps the wire-rounded value). a f32 is also identical (the single swap ADD commutes exactly). a f16 is not, because GPU2 gets an f16-rounded copy while GPU0/1 keep f32.
- **Error vs a double-precision sum** (max |err| / Σ|x|): f32 ~1.1e-7 for all; f16 wire 4.7e-4 (b) to 1.4e-3 (d, rounds at every hop).
- **Link-bound at large sizes:** ring f32 sustains 2.3 GB/s algorithm bandwidth and f16 4.2 GB/s, against the round 1 all-to-all ceiling of 3.1–4.4 GB/s per GPU. **Block count is not the limiter:** the pre-run picked 2–16 blocks.
- **Kernel-only time** (in-kernel `%globaltimer` after a 3-GPU start barrier, i.e. without launch and event overhead) is ~12 µs below end to end at small sizes; e.g. b f16 at 20 KB: 18.4 µs.
- **The isolated meta pattern matches the model:** a f32 at 40 MB = 32.7 ms against the 31.0 ms per exchange round 1 measured inside prefill.

**Projected savings** (measured exchange counts: 128 per decode token, 128 per 2048-token prefill step; round 1 / section 2.1 step times):

| | per exchange | per decode token (×128) | per pp2048 step (×128) |
|---|---:|---:|---:|
| isolated: a f32 → b f32 (20 KB) | −12.5 µs | −1.6 ms | |
| isolated: a f32 → b f16 (20 KB) | −18.1 µs | −2.3 ms (of 33.8 graphs-on = 6.8%) | |
| **decode, scaled to the in-model cost:** section 2.1 graphs-on exchange path = 3.8 ms/token (29.7 µs/exchange, since copies partly overlap neighbours' compute); b f32 = 0.75× a, b f16 = 0.63× a | | −0.95 ms (f32) / −1.4 ms (f16): **3–4% of the token** | |
| prefill: a f32 → d ring f32 (40 MB) | −14.6 ms | | **−1.87 s of 7.87 s @d16384 (−24%): 260 → ~342 t/s** |
| prefill: a f32 → d ring f16 (40 MB) | −22.6 ms | | **−2.89 s (−37%): 260 → ~413 t/s**, above the 2-GPU 383 t/s |
| prefill ceiling (exchange fully hidden or removed) | | | −4.52 s (−57%) |

- **Measured ceiling, decode:** one-shot f16 31.3 µs end to end (18.4 µs in-kernel) per 20 KB exchange. Against the in-model 3.8 ms/token, the AllReduce algorithm alone recovers at most ~1.4 ms. More would need putting the exchange inside the CUDA graph, which removes 129 graph splits per device and ~4,250 host calls per token, plus the ~2.2 ms of other idle.
- **Measured ceiling, prefill:** ring f16 10.1 ms per 40 MB exchange (link-bound at ~4.2 GB/s). **Ring f32 18.0 ms is the lossless option.** Overlapping the exchange with compute (as the fork's 2-GPU chunked path does) is the only way below that.

## 2.3 Decode noise

| variant | mean t/s (ms/token) | stdev across the 5 runs | pooled within-run stdev |
|---|---:|---:|---:|
| base | 22.62 (44.21) | 0.142 = **0.63%** | 0.85% |
| `taskset -c 0-5` (one thread per physical core) | 22.49 (44.46) | 0.137 = **0.61%** | 0.60% |
| `-t 1` | 22.57 (44.31) | 0.107 = **0.48%** | 0.66% |
| performance governor | not run: already `performance` on all cores; nothing to revert | | |

- **No variant moves the mean or the spread outside the others' noise.** Pinning and thread count do not matter: decode is host-bound on one enqueue thread (section 2.1), and the CPU threadpool is idle.
- **Per-core MHz and governor** were logged at 1 Hz for every run (`results/*.cpu.csv`). The governor stayed `performance`.
- **All variants show 0.5–0.9%, against 8% in round 1** (21.62 ± 1.71 under `-sm tensor`). The only harness difference is the model load mode: Round 1 used the default mmap on a 16 GB-RAM host with a 23.5 GiB model; here `-lm none`. The extra mmap-vs-none A/B at the end of the queue tests this directly.

### Noise against model load mode

XL tg512, 3 GPUs, `-sm tensor`, graphs off, `-r 3`, 5 rounds alternating order.

| load mode | runs (t/s) | mean t/s (ms/token) | stdev across runs | pooled within-run stdev |
|---|---|---:|---:|---:|
| `-lm mmap` (round 1's default) | 19.67, 21.14, 20.96, 17.73, 20.60 | 20.02 (49.95) | 1.36 = **6.8%** | 2.48 = **12.4%** |
| `-lm none` | 22.41, 22.62, 22.44, 22.43, 22.88 | **22.56** (44.33) | 0.20 = **0.9%** | 0.14 = **0.6%** |

- **Confirmed: mmap is round 1's decode noise, and it is also a mean penalty (−11%, +5.6 ms/token).** On this 16 GB-RAM host the 23.5 GiB model cannot stay in the page cache. The CPU-resident pieces (token_embd `get_rows`, 1 GiB Q6_K) are re-faulted from disk during decode.
- **round 1 decode figures (21.4–21.6 t/s, ±8%) were taken under mmap** and understate the clean graphs-off number (22.6) by ~5%. Its decode exchange shares were computed against those inflated token times.
- **Deployments on the test machine should use `-lm none`** (or the server's `-lm none`).
<!-- P15-LIVE-END -->

<!-- P16-LIVE-BEGIN -->

## 2.4 Layer split: long prompts and the pipeline

- **pp65536 `-sm layer`: 373.86 ± 0.60 t/s** (175.3 s/step), against `-sm tensor` 242.0 t/s (round 1 G4): **layer is 1.54× faster** at 64k, as it was 1.59× at pp16384 (round 1).
- **nvprof timeline, pp16384 `-sm layer`** (measured pass, 38.06 s, 429.9 t/s, matching the unprofiled round 1 428.9, so nvprof barely perturbs prefill):
  - GPU busy: GPU0 67.5%, GPU1 68.3%, GPU2 61.5%.
  - Each GPU starts ~3.1 s after the previous one (one ubatch's worth of its layer slice): GPU0 +0 s, GPU1 +3.08 s, GPU2 +6.16 s.
  - Wall time with N GPUs busy at once: 0 GPUs 0.0%, 1: 18.1%, **2: 66.4%, 3: 15.4%**.
  - **Ubatches overlap: 82% of the busy time has two or more GPUs working**, against ~0% for a serial layer pipeline. This is the scheduler's pipeline parallelism: ubatch k+1 runs on GPU0 while ubatch k is on GPU1 and so on. There is no exchange; the only inter-GPU traffic is one activation copy per ubatch per boundary.
- **Server flags that give this in a deployment:**
  - `-sm layer`, with all layers offloaded (`-ngl 99`; the scheduler requires `n_gpu_layers > n_layer`).
  - KV offloaded: the default; no `-nkvo`.
  - No tensor overrides (`-ot`, `--cpu-moe`).
  - **`-b` larger than `-ub`** (e.g. the fork's `-b 32768 -ub 2048`): the pipeline only fills when one batch splits into several ubatches, so prompts shorter than `-ub` get no overlap.
  - The build's `GGML_SCHED_MAX_COPIES` (default 4) sets the pipeline depth.
  - The condition is `llama-context.cpp:429`. llama-bench suppresses the "pipeline parallelism enabled" info line, so the trace above is the evidence.
  - **Trade-off:** layer split decodes slower (round 1: 15.72 vs 21.62 t/s graphs-off), and a single long prompt is the only thing that benefits.

## 2.5 Decode at depth, tensor against layer

| depth | `-sm tensor` t/s (ms/token) | `-sm layer` t/s (ms/token) | tensor / layer |
|---|---:|---:|---:|
| 0 (sections 2.1 and 2.3; round 1 for layer) | 22.4–22.6 (44.2–44.6) | 15.72 (63.6) | 1.43× |
| 16384 | **22.43 ± 0.05** (44.57) | **15.72 ± 0.00** (63.63) | 1.43× |
| 65536 | **22.23 ± 0.08** (44.98) | **14.35 ± 0.00** (69.71) | 1.55× |

- **Tensor-split decode is flat with depth:** −0.9% from 0 to 64k, because the attention cost (17 full-attention layers, section 2.1: ~1 ms/token at depth 0) is hidden behind the 43.6 ms host enqueue (graphs off). With graphs on (GPU-bound, section 2.1) depth would show; not measured here.
- **Layer split** is flat to 16k and loses 9% at 64k, where each GPU processes its layers serially and the attention cost lands on the critical path.
- **Tensor stays ahead for decode at every depth** (1.43–1.55×), while layer is 1.54–1.59× faster for long-prompt prefill (section 2.4). A deployment choosing a split mode trades one for the other.

## 2.6 MTP grid, single runs (greedy)

Cells: decode t/s (draft acceptance, decode ms per accepted draft token = decode time / accepted drafts). One run per cell; the prompt order was interleaved across configs.

| n-max | p-min | chat t/s (acc %, ms/acc) | code t/s (acc %, ms/acc) | summ ~8k t/s (acc %, ms/acc) | mean t/s |
|---|---|---:|---:|---:|---:|
| 3 | 0.0 | 40.90 (48%, 41.4) | **57.91** (83%, 24.2) | 45.42 (61%, 34.1) | **48.08** |
| 3 | 0.5 | 33.15 (61%, 53.5) | 55.35 (88%, 25.3) | 37.55 (69%, 43.1) | 42.02 |
| 3 | 0.75 | 30.05 (79%, 66.2) | 43.84 (92%, 33.0) | 33.81 (81%, 53.5) | 35.90 |
| 3 | 0.9 | 28.12 (89%, 76.0) | 43.07 (97%, 35.6) | 32.70 (93%, 61.4) | 34.63 |
| 4 | 0.0 | 40.27 (43%, 39.3) | 56.34 (77%, 23.5) | **45.66** (55%, 31.8) | 47.42 |
| 4 | 0.5 | 31.18 (52%, 56.0) | 57.47 (78%, 23.1) | 33.03 (55%, 49.8) | 40.56 |
| 4 | 0.75 | 30.78 (73%, 60.7) | 48.87 (88%, 28.0) | 33.93 (80%, 50.2) | 37.86 |
| 4 | 0.9 | 28.65 (89%, 79.8) | 38.20 (92%, 41.9) | 32.21 (96%, 57.5) | 33.02 |
| 5 | 0.0 | 35.07 (34%, 45.5) | 52.66 (62%, 25.1) | 41.19 (46%, 34.9) | 42.97 |
| 5 | 0.5 | 29.77 (47%, 57.7) | 46.51 (72%, 29.0) | 35.11 (54%, 43.1) | 37.13 |
| 5 | 0.75 | 30.49 (70%, 62.1) | 48.20 (86%, 27.7) | 33.84 (84%, 49.1) | 37.51 |
| 5 | 0.9 | 28.81 (83%, 73.4) | 45.57 (92%, 30.1) | 31.64 (94%, 60.2) | 35.34 |

- **Best overall: n-max 3, p-min 0.0, mean 48.1 t/s** (chat 40.9, code 57.9, summarization 45.4). n-max 4 / p-min 0.0 is within noise (47.4).
- **Against plain decode** with graphs `=3` (29.6 t/s, section 2.1), MTP is **1.36–1.96× faster**.
- **p-min is the dominant knob on 3 GPUs:** raising it trades acceptance for fewer drafts and always loses throughput. Acceptance climbs from ~45% to ~90%, while t/s falls 25–35%.
  - Interpretation: each verify step is cheap relative to a token, so drafting aggressively pays. The fork's 2-GPU recommendation is n-max 4 / p-min 0.2; p-min 0.2 was not in this grid.
- **n-max 5 is worse than 3–4 at every p-min** (more rejected drafts).
- **Adaptation of `mtp.sh`:** it hardcodes the model path and prompt and takes only n-max / p-min. I kept its flags (`--spec-type draft-mtp -ngld 99 --temp 0 --top-k 1 --seed 42 -sm tensor -fa 1 -ctk/-ctv q4_0`) and added `-m` (XL), `-f <prompt>`, `-n 256`, `-c 16384`, `-b 32768 -ub 2048`, `-ts 1/1/1`, `-fit off` (avoids its fit abort under tensor split), `-lm none` and `GGML_CUDA_GRAPHS_PRE_VOLTA=3` (the fork's serving default). The three prompts are described in METHODOLOGY.md (chat, code, and a summarization prompt of 8,009 tokens built from `docs/build.md`).

## 2.7 Prefill `-ub` sweep

| `-ub` | pp2048 @ depth 0: t/s (s/step) | vs ub 2048 | pp2048 @ depth 16384: t/s (s/step) | vs ub 2048 |
|---|---:|---:|---:|---:|
| 512 | 254.35 (8.05) | −7.7% | 240.48 (8.52) | −7.2% |
| 1024 | 267.02 (7.67) | −3.1% | 251.60 (8.14) | −2.9% |
| **2048** | **275.48 (7.43)** | — | **259.13 (7.90)** | — |
| 4096 | 275.59 (7.43) | +0.0% | 259.23 (7.90) | +0.0% |

- **`-ub 2048` is the knee.** 4096 adds nothing: a pp2048 test cannot form a ubatch above 2048, and the depth-16384 fill at 4096 does not change the measured step either. Smaller ubatches lose 3–8%.
- The exchange count per 2048 tokens scales with the number of ubatches (128 per ubatch, each ub × 5120 f32). Total exchange bytes are unchanged, so the loss is per-ubatch fixed cost (exchange latency, launches), not bandwidth.
- **The fork's default `-ub 2048` is right for 3 GPUs.** A bigger ubatch would only help prompts longer than 2048, and VRAM limits it at long context.

## 2.8 Pure Q6_K against the XL

- `llama-quantize --allow-requantize --pure Q8_0 → Q6_K` into a scratch directory: 10.5 min CPU (07:40–07:51 EDT, no GPU run in parallel), 21,381 MiB, 6.56 BPW.
- **Fingerprints match the fork's reference on every point:**
  - tensor data **20.88 GiB**;
  - all 506 quantized tensors Q6_K (**0 Q8_0**), including **output.weight Q6_K**, token_embd Q6_K, and **ssm_alpha / ssm_beta Q6_K** (48 each);
  - **MTP block present** (blk.64, 4 nextn tensors, `nextn_predict_layers = 1`).
  - Caveat: no imatrix was used; whether the fork's own file used one is unknown, which could matter for quality but not speed.

| test (3 GPUs, `-sm tensor`, graphs off, ABBA) | pure Q6_K | XL | Q6_K vs XL |
|---|---:|---:|---:|
| tg512 t/s (ms/token) | **23.17 ± 0.14** (43.17) | 22.62 ± 0.10 (44.22) | **+2.4% (−1.05 ms)** |
| pp512 t/s | 257.27 | 256.25 | +0.4% |
| pp2048 t/s | 275.68 | 275.51 | +0.1% |

- **The quant effect is small on 3 GPUs.**
  - **Decode, graphs off, is host-bound** (section 2.1). The 6% fewer weight bytes (20.88 vs 22.24 GiB streamed) show only as −1.05 ms/token, roughly the matvec time saved by the smaller stream. With graphs on (GPU-bound), the gain should be about the same in ms and a larger share of the token; not measured here.
  - **Prefill is exchange- and GEMM-bound**, so the quant makes no difference.
- **The XL is not what held 3-GPU the test machine back against the fork's 2-GPU Q6_K numbers.** The graph setting (section 2.1: +32%) and the exchange (section 2.2) are.

## 2.9 Thermal soak

| | GPU0 | GPU1 | GPU2 |
|---|---:|---:|---:|
| peak temp (max in first 5 min → last 5 min) | 59 °C (56 → 59) | 66 °C (63 → 66) | 64 °C (62 → 64) |
| SM clock while busy, median (min) | 1328 (1126) MHz | 1328 (1151) MHz | 1316 (1139) MHz |
| SM clock median, first 5 min → last 5 min | 1328 → 1316 | 1328 → 1328 | 1328 → 1316 |
| power while busy, median (max sample) | 116 W (173) | 117 W (163) | 117 W (164) |
| throttle reasons while busy | none 87%, **SW power cap 13%** | none 85%, power cap 15% | none 84%, power cap 16% |

- **No thermal throttling** (no HW/SW thermal slowdown bits in any sample). Peak 66 °C, 14 °C under the 80 °C stop. The only clock event is the 125 W software power cap during prefill bursts; minimum busy clock 1126 MHz, above the 1000 MHz watchdog line.
- **Throughput holds: prefill 269.0 → 268.4 t/s (−0.2%), decode 21.74 → 21.77 t/s (+0.1%)**, first 5 min vs last 5 min. Means over the soak: 268.5 / 21.72 t/s.
- **Fans:** the shroud fan (fan3) ran 1478–2547 RPM (median 2436), pwm2 112–207 (median 202 of 255). Fan control tracked the load and never reached the 0-RPM condition.
- **Server vs bench:** server decode (21.7 t/s) is ~4% below llama-bench tg512 (22.6), and server pp2048 is ~2.5% below bench (275.5). The difference is per-request overhead; it does not drift over 30 min.

## 2.10 NCCL, first runs (graphs off)

| test | build-opt (meta butterfly) | NCCL defaults (**SHM via host**) | NCCL `NCCL_P2P_LEVEL=SYS` (**P2P/direct**) |
|---|---:|---:|---:|
| tg512 t/s (ms/token) | 22.97 (43.54) | **27.21** (36.76), +18.5% | 26.84 (37.26), +16.9% |
| pp2048 t/s (s/step) | 275.64 (7.43) | 438.84 (4.67), +59% | **495.64** (4.13), **+80%** |
| pp2048 @16384 t/s (s/step) | 258.99 (7.91) | 411.57 (4.98), +59% | **439.95** (4.66), **+70%** |

- **Transport** (from `NCCL_DEBUG` files): defaults chose `SHM/direct/direct` (through host memory, because the topology is PHB) on all 18 channel connections; `NCCL_P2P_LEVEL=SYS` forces `P2P/direct`.
- **NCCL as shipped uses BF16 for large exchanges.** `ggml_backend_cuda_comm_allreduce_nccl` reduces in FP32 below 131,072 elements at 3 GPUs (decode, 5,120 elements: f32 arithmetic, but not bit-identical to the non-NCCL path; see 1.3) and in **BF16 above it (every prefill exchange)**. **The prefill numbers need a KLD check before any quality claim.**
- **Decode +17–18% at f32**, even graphs-off and host-bound. One grouped `ncclAllReduce` per exchange replaces the butterfly's ~33 host calls per exchange (4 peer copies, 3 ADD launches, events and waits; section 2.1). The gain is host-call reduction, not link time.
- **Prefill +70–80% (BF16, P2P)** lands at 440–496 t/s: **above 2-GPU the test machine (383.5 @16k, 426–433 @0)** and level with the fork's published 2-GPU pp2048 (493). This is consistent with the shootout's f16-ring projection (~413 @16k) plus NCCL's pipelining across its channels.


---

# Part 3. Round 1

## 3.0 What is superseded, and why

Round 1 ran every decode test with **CUDA graphs off** (the default on Pascal when `GGML_CUDA_GRAPHS_PRE_VOLTA` is
unset) and with the default **mmap** model load. Round 2 showed both were wrong for this machine:

- **Graphs off made decode CPU-bound.** The host spends 43.6 of 45.3 ms per token enqueueing about 19,000 CUDA calls;
  the GPUs finish about 1 ms after it stops (section 2.1). With `=3` the same test runs at 29.6 t/s instead of 22.4.
- **mmap added noise and a mean penalty.** With 16 GB of RAM and a 23.5 GiB model, pages are re-read from disk during
  decode: 6.8% spread across runs and −11% on the mean, against 0.9% with `-lm none` (section 2.3).

So in this part: **every decode figure (tg512, the decode exchange share, the 2-GPU against 3-GPU decode comparison,
Q8_0 against XL decode) is superseded.** They are kept for the record. The 2-GPU arms were never re-run with graphs on
and `-lm none`, so the 2-GPU against 3-GPU question is open, not answered in either direction.

**Still valid from round 1:** the P2P characterization (3.1), the prefill numbers and prefill attribution (3.2; prefill
is neither CPU-bound nor mmap-sensitive, and round 3's configuration A reproduces them to 0.5%), and the per-type
matvec rates (3.5).


## 3.1 P2P characterization (valid)

**(g) G0, P2P characterization** (per directed pair, all 6 pairs within ±2% of each other)

| Test | 20 KB | 100 KB | 20 MB |
|---|---|---|---|
| copy engine push (source issues) | 8.3 µs, 2.45 GB/s | 24.0 µs, 4.27 GB/s | **5.21 GB/s** |
| copy engine pull (destination issues) | 11.3-11.9 µs | 27.2-27.7 µs | 5.15 GB/s |
| kernel remote **write** | 9.9-10.2 µs | 27.2-27.3 µs | 4.49 GB/s |
| kernel remote **read** | 11.3-11.9 µs | 27.2-27.7 µs | **5.11-5.17 GB/s** |
| staged (peer access off) | — | 43.5-45.5 µs, 2.3 GB/s | 5.79-5.84 GB/s |
| flag one-way latency (single-thread ping-pong) | **0.90-0.94 µs** | | |

New small-message test (1024-thread block writes N bytes to the peer, fence, flag; peer answers with a flag; round
trip, 950 iterations; median / p99; all pairs):

| Size | Median RT | p99 RT |
|---|---:|---:|
| 0 (flag only, this kernel structure) | 6.1-7.2 µs | 7.2 µs |
| 16 KB | 9.2-10.2 µs | 10.2 µs |
| 64 KB | 18.4-19.5 µs | 19.5 µs |
| 256 KB | 57.3-58.4 µs | 58.4-59.4 µs |

New all-to-all test (all 3 GPUs send to both peers at once; median of 17 reps):

| Mode | 64 KB per peer | 20 MB per peer |
|---|---|---|
| kernel writes, send GB/s per GPU (per link) | 3.11-3.28 (1.56-1.64) | 3.10-3.15 (1.55-1.58) |
| copy engine, send GB/s per GPU (per link) | 2.43-2.89 (1.22-1.45) | GPU0 **4.42** (2.21), GPU1 3.94 (1.97), GPU2 3.95 (1.97) |

The Sandy Bridge-E "slow P2P read" concern **does not apply**: reads match writes. Under all-to-all load each GPU's
x8 link carries two sends and two receives at once; per-GPU egress drops to 3.1-4.4 GB/s (60-85% of a single pair's
5.2), ~2 GB/s per link. GPU1 and GPU2, which share one root-port group, are ~11% below GPU0 with copy engines at 20 MB;
with kernel writes all three are equal. Caveats: `%globaltimer` ticks in ~1.024 µs steps on Pascal, so latency
medians/p99 are quantized; the 0-byte round trip here (6.1 µs) is higher than G0's single-thread ping-pong (1.9 µs RT)
because of the block sync and fences in this kernel structure; all-to-all GPUs are launched from one host thread,
so a few µs of start skew is included at 64 KB.

## 3.2 Prefill: throughput and exchange-share attribution (valid)

### Prefill throughput (pp2048, t/s)

| Depth | 3 GPUs XL | 3 GPUs Q8_0 | 2 GPUs XL (1,2) | 2 GPUs XL (0,1) | 2 GPUs Q8_0 (1,2) | 2 GPUs Q8_0 (0,1) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 274.64 ± 0.67 | 276.07 ± 0.36 | 426.16 ± 1.67 | 432.78 ± 1.68 | 423.39 ± 0.32 | 430.15 ± 0.10 |
| 16384 | 260.14 ± 0.35 | 261.62 ± 0.16 | 380.45 ± 0.45 | 386.64 ± 0.45 | 378.70 ± 0.48 | 383.69 ± 0.24 |
| 65536 | 211.08 ± 0.12 | 211.89 ± 0.42 | 280.19 ± 0.12 | 284.11 ± 0.29 | OOM | OOM |

### Prefill attribution (nvprof, averaged over depth 0..D+2048)

| Trace | GPU busy 0/1/2 | Exchange work 0/1/2 | Exchange latency | Imbalance wait | s_pp |
|---|---|---|---:|---:|---:|
| 3 GPUs XL, D=16384 | 95 / 63 / 71% | 59 / 26 / 33% | 51.4% (31.0 ms/exchange) | 6.0% (3.6 ms) | 57.4% |
| 3 GPUs Q8_0, D=16384 | 96 / 64 / 71% | 59 / 26 / 33% | 51.9% | 6.0% | 57.9% |
| 3 GPUs XL, D=65536 | 93 / 64 / 71% | 53 / 23 / 30% | 46.7% (31.1 ms) | 7.6% (5.1 ms) | 54.2% |
| 3 GPUs Q8_0, D=65536 | 93 / 64 / 71% | 53 / 23 / 30% | 46.8% | 7.6% | 54.4% |
| 2 GPUs XL (1,2), D=16384 | 98 / 98% | 24 / 24% (overlapped with GEMM) | — | — | — |

## 3.3 Exchange share (prefill rows valid; decode rows superseded)

**(a) Exchange share** (definition: exchange time + exchange-adjacent idle, as a share of token or step time)

| Workload (3 GPUs, `-sm tensor`) | Token / step time | Exchange share | Exchange time | How computed | Confidence |
|---|---:|---:|---:|---|---|
| Decode, XL | 46.73 ms/token (21.40 t/s) | **19.2-19.5%** | **9.0-9.1 ms/token** | B and C below agree | moderate |
| Decode, Q8_0 | 46.16 ms/token (21.67 t/s) | **13.0-16.5%** | **6.0-7.6 ms/token** | B and C below | moderate |
| Prefill pp2048 @ depth 16384, XL | 7.873 s/step (260.1 t/s) | **57.4%** | **4.52 s/step** (2.21 ms/token) | D below | high |
| Prefill pp2048 @ depth 65536, XL | 9.703 s/step (211.1 t/s) | **54.2%** | **5.26 s/step** (2.57 ms/token) | D below | high |
| Prefill @16384 / @65536, Q8_0 | 7.828 / 9.665 s/step | 57.9% / 54.4% | 4.53 / 5.26 s/step | D below | high |

How it was computed, from nvprof `--print-gpu-trace` of llama-bench in the container:
- **A (raw, decode):** in the n=36 trace, a steady window of 32 tokens delimited by counting exchange copies
  (512 peer copies per token = 128 exchanges × 4 copies), per-device union of kernels and copies, idle gaps
  classified by whether the next block starts with an exchange op. Result: 31.6% mean, **51.8% on GPU0**, 30.1%
  GPU1, 12.9% GPU2 (XL). **This overstates decode:** under nvprof a token takes 118.9 ms vs 46.7 ms real (host
  side inflated about 2.5×; GPU kernel durations are not).
- **B (calibrated, decode):** real idle per token = real token time (from tg512) − GPU busy (from the trace;
  kernel durations are reliable). Real idle × the trace's exchange-adjacent fraction of idle (per device, then
  averaged) + exchange-op time. XL: 19.1 ms idle × 0.397 + 1.37 ms = **9.0 ms = 19.2%**. Q8_0: 16.2 ms × 0.386 +
  1.37 ms = 7.6 ms = 16.5%.
- **C (2-GPU control, decode):** same quant on 2 GPUs (fork's one-kernel P2P AllReduce active) idles 10.0 ms/token
  (XL) vs 19.1 ms on 3 GPUs. Excess = **9.1 ms = 19.5%** (XL); Q8_0 10.2 vs 16.2 ms → 6.0 ms = 13.0%.
- **D (prefill):** per exchange in the trace, each GPU's "ready" time (start of its outgoing copy; for GPU0, end of
  its last compute kernel before the fold ADD); imbalance = max(ready) − mean(ready); exchange latency = last copy
  end − max(ready). Averaged over the whole depth fill (0 to D+2048), so it is a depth-averaged share applied to the
  measured step time at D. nvprof barely inflates prefill (7.73 s/step in trace vs 7.87 s real at 16k), hence high
  confidence. Breakdown @16384 XL: latency 51.4% + imbalance 6.0%; @65536: 46.7% + 7.6%. ~31 ms latency per
  exchange (40 MB f32 partials, 3 dependent stages, nothing overlapped).

## 3.4 Tensor against layer, round 1 (decode rows superseded)

**(e) G4: tensor vs layer, 3 GPUs, XL** (Q8_0 where measured)

| Test | XL tensor | XL layer | **tensor / layer** | Q8_0 tensor | Q8_0 layer |
|---|---:|---:|---:|---:|---:|
| tg512 | 21.62 ± 1.71 | 15.72 ± 0.95 | **1.38** | 21.98 ± 2.09 | 13.68 ± 1.12 |
| pp512 | 245.90 ± 15.80 | 178.88 ± 57.19 | 1.37 (noisy) | 240.14 ± 18.74 | 208.70 ± 14.80 |
| pp2048 | 275.23 ± 0.79 | 228.75 ± 0.30 | **1.20** | 276.23 ± 0.03 | 228.45 ± 0.72 |
| pp16384 | 269.96 ± 0.15 | **428.88 ± 2.49** | **0.63** | skipped | killed (watchdog) |
| pp65536 | 242.00 ± 0.06 | not run (skipped after the power-cap stop) | — | skipped | skipped |

**Does the answer change at 65k?** It already changes at 16k: layer split processes a 16k prompt **1.59× faster**
than tensor split. **[inferred]** With `-b 32768 -ub 2048` a long prompt is split into many ubatches that the
layer-split scheduler pipelines across the three GPUs (all three busy at once: 63/65/61% utilization, power-capped),
while tensor split pays the 3-stage exchange on every layer. 65k under layer split was not measured (skipped after
the power-cap watchdog stop), so **65k itself is unanswered**; tensor at 65k is 242.0 t/s, and the pipelining effect
should persist, but that needs measuring. Plain answer: **decode and short prompts favour tensor split (1.2-1.4×);
long prompts favour layer split (1.6× at 16k).** Whether llama-server pipelines a single long prompt the same way under
`-sm layer` is not yet verified.

## 3.5 Per-type matvec rates (valid)

**(d) Per-type matvec at the model's per-GPU shapes** (G3, `GGML_CUDA_OP_PROFILE=1`, 3 GPUs; the profiler syncs
every graph and prints only the top 45 entries per device, so read as relative)

| Type | n=1 decode GB/s | % of 605 GB/s ceiling | n=5 verify GB/s | % of ceiling |
|---|---:|---:|---:|---:|
| Q6_K (XL) | 375-380 (largest shapes 403-408) | 62-63% (67%) | 206-210 | 34-35% |
| Q8_0 (XL) | 313-326 (largest 386-393) | 52-54% (64-65%) | 145-154 | 24-25% |
| Q8_0 (pure) | 339-348 | 56-58% | 174-179 | 29-30% |
| Q5_K (XL) | 243-248 | 40-41% | 108-112 | 18% |

Q8_0 is ~15% slower per byte than the fork's tuned Q6_K at n=1, and ~15-30% slower at the MTP verify width; Q5_K is
~35% slower at n=1 and ~2× slower at n=5. Matvec efficiency drops ~12% when slices shrink from 1/2 to 1/3 (2-GPU
~425 GB/s effective vs 3-GPU ~375 GB/s, from the G1 traces).

## 3.6 Per-GPU utilization

**(f) Per-GPU utilization** (mean over seconds when any GPU was >50% busy; GPU0/GPU1/GPU2)

| Run | Util % | Power W | SM MHz | Peak °C |
|---|---|---|---|---|
| tg512 tensor XL | 60 / 58 / 57 | 115 / 113 / 113 | 1324 / 1322 / 1316 | 56 / 63 / 63 |
| pp2048 @16384 tensor XL | 90 / 46 / 46 | 79 / 79 / 82 | 1226 / 1221 / 1225 | 48 / 55 / 53 |
| pp2048 @65536 tensor XL | 88 / 47 / 48 | 80 / 84 / 82 | 1216 / 1209 / 1211 | 51 / 58 / 56 |
| pp16384 tensor XL | 91 / 46 / 46 | 81 / 81 / 77 | 1224 / 1215 / 1215 | 52 / 59 / 58 |
| pp65536 tensor XL | 89 / 48 / 48 | 83 / 81 / 81 | 1210 / 1202 / 1199 | 55 / 61 / 60 |
| pp16384 layer XL | 63 / 65 / 61 | 92 / 92 / 87 | 1114 / 1119 / 1114 | 55 / 62 / 59 |
| pp2048 @16384, 2 GPUs (1,2) | — / 99 / 99 | — / 115 / 115 | — / 1114 / 1110 | — / 60 / 58 |

Uneven utilization seen when serving (about 80% on one card against 53% on the others) is **GPU0 acting as the hub of the generic 3-GPU exchange**: every exchange is 2→0 (fold),
0↔1 (swap), 0→2 (copy-back). GPU0 runs 257 exchange ADDs per decode token vs 129 on GPU1 and 1 on GPU2, and in
prefill spends 53-59% of its time in exchange work vs 23-33% for the others. It shows mainly in **prefill** (90/46/46);
in decode the GPUs are evenly busy (60/58/57). A prefill-heavy serving load shows the same pattern. On 2 GPUs both cards run 99% during prefill (the fork's overlapped exchange).

## 3.7 Superseded decode numbers

Everything in this section was measured with graphs off and mmap loading. The text is the round 1 reading at the time and is kept unchanged for the record; its conclusions about decode, and about 2 GPUs against 3, should not be relied on.

### 2 GPUs against 3 GPUs (graphs off, mmap; superseded)

**(b) 2-GPU vs 3-GPU scaling, same quant** (tensor split; 2-GPU = pairs 1,2 and 0,1; mean of both pairs)

| Test | XL 3 GPUs | XL 2 GPUs | 3 vs 2 | Q8_0 3 GPUs | Q8_0 2 GPUs | 3 vs 2 |
|---|---:|---:|---:|---:|---:|---:|
| tg512 (t/s; ms/token) | 21.40; 46.7 | 23.21; 43.1 | **−7.8% (+3.6 ms)** | 21.67; 46.2 | 21.03; 47.6 | +3.0% (−1.4 ms) |
| pp2048 @0 (t/s; s/step) | 274.6; 7.46 | 429.5; 4.77 | **−36%** | 276.1; 7.42 | 426.8; 4.80 | −35% |
| pp2048 @16384 | 260.1; 7.87 | 383.5; 5.34 | **−32%** | 261.6; 7.83 | 381.2; 5.37 | −31% |
| pp2048 @65536 | 211.1; 9.70 | 282.2; 7.26 | **−25%** | 211.9; 9.67 | **OOM** (does not fit) | — |

Adding the third GPU makes the test machine **slower** on every XL test, and slower on all Q8_0 prefill tests. **This is not
like-for-like against the fork's published numbers** (Q6_K 2-GPU: tg256 ~31-32.6, pp2048 493): the pure Q6_K arm was
not available, and both the test machine quants stream more bytes per token (XL +13.6%, Q8_0 +29% vs pure Q6_K). The 2-GPU XL
pp2048 at depth 0 (426-433 t/s) is within 12-14% of the fork's 493; 2-GPU decode (23 t/s) is 28-30% below its 32.6.

### Q8_0 against XL (graphs off, mmap; superseded for decode)

**(c) Q8_0 vs XL under both split modes (3 GPUs)**

| Decode tg512 | XL | Q8_0 | Q8_0 / XL | Bytes/token Q8_0 / XL |
|---|---:|---:|---:|---:|
| `-sm tensor` | 21.62 ± 1.71 | 21.98 ± 2.09 | 1.02 (within noise) | 1.14 |
| `-sm layer` | 15.72 ± 0.95 | 13.68 ± 1.12 | **0.87** | 1.14 (expected 0.877) |

Under layer split each token streams all weights through the three GPUs one after another and there are no
exchanges, so speed tracks bytes: Q8_0's 14% more bytes predicts 0.877×, measured 0.870×. **Weight streaming is the
bottleneck under layer split.** Under tensor split Q8_0 is as fast as the XL despite 14% more bytes. The traces show
why: Q8_0 matvec costs +2.6 ms/token (23.9 vs 21.3 ms), but its idle is 2.9 ms lower (16.2 vs 19.1 ms). On 3 GPUs
under tensor split the token time is dominated by something other than weight streaming: matvec is only ~46-52% of
the token, idle ~35-41%. Cross-check with section 3.3: ~9 ms of that idle is exchange-attributable (19%); the other ~8-10 ms
appears even on 2 GPUs and does not shrink with less work per GPU. **[inferred]** The fact that a heavier per-GPU
load (Q8_0) *reduces* idle points at host-side enqueue/launch latency being overlapped by longer GPU work, i.e.
host-bound gaps, not only exchange cost. So **Q8_0 is a faithful stand-in for the XL only under tensor split.**

### Decode throughput by configuration (graphs off, mmap; superseded)

| Config | XL r1 | XL r2 | Q8_0 r1 | Q8_0 r2 |
|---|---:|---:|---:|---:|
| 3 GPUs | 21.51 ± 1.63 | 21.29 ± 1.76 | 21.77 ± 2.17 | 21.56 ± 2.16 |
| 2 GPUs pair 1,2 | 22.81 ± 1.44 | 23.22 ± 1.71 | 20.85 ± 1.57 | 20.99 ± 1.72 |
| 2 GPUs pair 0,1 | 23.38 ± 1.73 | 23.44 ± 1.56 | 21.07 ± 1.69 | 21.21 ± 1.41 |

### Decode attribution (graphs off, mmap; superseded)

| | 3 GPUs XL | 2 GPUs XL (1,2) | 3 GPUs Q8_0 | 2 GPUs Q8_0 (1,2) |
|---|---:|---:|---:|---:|
| Real token time | 46.7 ms | 43.5 ms | 46.1 ms | 47.8 ms |
| GPU busy (mean of devices) | 27.6 ms | 33.5 ms | 29.9 ms | 37.6 ms |
| of which matvec | 21.3 ms | 27.9 ms | 23.9 ms | 32.1 ms |
| Idle | 19.1 ms | 10.0 ms | 16.2 ms | 10.2 ms |
| Exchanges per token | 128 (512 peer copies) | 128 (one P2P kernel each) | 128 | 128 |
| Exchange ops GPU0 / GPU1 / GPU2 | 2.19 / 1.08 / 0.82 ms | 0.95 / 0.97 ms | 2.19 / 1.09 / 0.82 ms | 0.96 / 1.00 ms |
| Raw nvprof s_dec (A) mean / GPU0 | 31.6% / 51.8% | 13.2% / 23.6% | 30.3% / 49.2% | 9.5% / 16.6% |

Matvec by type per token, 3 GPUs XL: Q8_0 12.4 ms, Q6_K 7.4 ms, Q5_K 1.3-1.5 ms. Also ~9 host-to-device copies and
~346 activation quantizations per device per token.

## 3.8 Missing or failed runs in round 1

| Run | Status | Reason |
|---|---|---|
| pure Q6_K arm (all of G1-G3, required 2-GPU control) | not run | no file on the test machine; 15 GB free; quantizing not allowed during the phase |
| `g2_pp2048_d65536_p12_Q8`, `_p01_Q8` | exit 139 | CUDA error in `ggml_cuda_pool_vmm::alloc` during prefill: out of memory (13.5 GiB weights per card + 64k KV + buffers). Recorded as does-not-fit |
| `g4_pp16384_layer_Q8` | exit 137 | killed by watchdog: LOWCLOCK GPU2 999 MHz at 100% for 5 s (power cap) |
| `g4_pp16384_tensor_Q8`, `g4_pp65536_tensor_Q8`, `g4_pp65536_layer_Q8` | skipped | removed from the queue |
| `g4_pp65536_layer_XL` | skipped | skipped after the power-cap stop |
