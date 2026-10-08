# Scripts

Out-of-tree tools used for the measurements. Nothing here is built by the fork's CMake.

## Run harness

- `lib.sh`: sourced by the stage scripts. `run <tag> [docker args] -- <command>` waits for the cool-down, starts
  1 Hz `nvidia-smi` and CPU logging, runs the command in a container, runs the watchdog, and stores stdout, stderr
  and the exit code under `results/`. A tag that already finished with exit 0 is skipped, so a stopped queue can be
  restarted. `run_server` does the same around `llama-server` and a client command.
- `g0.sh` (P2P), `g1.sh` (decode, 2 and 3 GPUs), `g2.sh` (prefill at depth), `g3.sh` (op profile), `g4.sh` (tensor
  against layer), `g0b.sh` and `finish.sh` (the P2P extras and the last stage as run), `phase1.sh` (runs g2 to g4
  after g1).

Variables (environment; defaults in `lib.sh`):

| variable | meaning | default |
|---|---|---|
| `BENCH_DIR` | directory with these scripts, `results/` and `run.log` | the scripts' directory |
| `SRC` | llama.cpp checkout with `build-opt/` | `$BENCH_DIR/../src` |
| `MODELS` | directory with the `.gguf` files (mounted at `/models`) | `$HOME/models` |
| `IMG` | Docker image with the CUDA toolchain | `p100-llamacpp-test:latest` |
| `DEADLINE` | epoch seconds after which no run is launched | unset |
| `FAN_CHECK`, `FAN_HWMON_NAME`, `FAN_SVC`, `FAN_TEMP_LIM` | fan check; **host-specific** (an `nct6776` sensor and a local fan-control unit). Set `FAN_CHECK=0` elsewhere | on |
| `WD_NOMEM`, `EXTRA_MOUNTS` | disable the no-GPU-memory check for CPU-only jobs; extra `-v` mounts | on; none |

The stage scripts name the model files `Qwen3.8-27B-UD-Q6_K_XL.gguf` and `Qwen3.8-27B-Q8_0.gguf` under `/models`.
The watchdog uses `sudo journalctl -k` for the Xid check and needs Docker with the NVIDIA runtime.

```
BENCH_DIR=$PWD SRC=/path/to/llama.cpp MODELS=/path/to/models ./g1.sh
```

## P2P microbenchmarks and the AllReduce shootout

`p2pbench/p2pbench.cu` (per-pair bandwidth and flag latency), `p2pbench2.cu` (small-message round trips, all-to-all
load), `p2pbench3.cu` (the 3-GPU AllReduce shootout: the replicated generic pattern, one-shot, two-shot and ring,
f32 and f16 wire, with a bit-identity check). Build with the CUDA toolkit and run on a machine with 3 GPUs that can
peer:

```
nvcc -O3 -arch=sm_60 -o p2pbench  p2pbench.cu
nvcc -O3 -arch=sm_60 -o p2pbench2 p2pbench2.cu
nvcc -O3 -arch=sm_60 -o p2pbench3 p2pbench3.cu
./p2pbench ; ./p2pbench --no-peer ; ./p2pbench2 ; ./p2pbench3        # p2pbench3 quick  for a short run
```

The stage scripts expect the binaries in `$BENCH_DIR/p2pbench/`. `p2pbench3` spins inside kernels while waiting for
peers; a peer that never arrives hangs the run, so use `timeout`.

## Trace analyzers

`analyze_decode.py`, `analyze_prefill.py`, `analyze_g3.py` (round 1), `analyze_i1.py` (decode budget from an nvprof
GPU + API trace), `analyze_i4.py` (per-kernel table from an op profile and an nvprof trace), `analyze_overlap.py`
(GPU overlap under layer split). Python 3, standard library only. Each file's docstring gives its arguments.

The queue scripts for rounds 2 and 3 are not included; their exact command lines are in `results/round2/run_log.txt` and
`results/round3/run_log.txt`.

## Round 4 tools

- `kldpos.cpp`: per-position KL divergence between two files saved by `llama-perplexity --kl-divergence-base`.
  Prints the overall mean, median, 99.9% and maximum KLD and the top-token agreement, then one CSV line per position
  bucket. Build: `g++ -O2 -pthread -o kldpos kldpos.cpp`. Run: `kldpos base.bin test.bin [bucket=1024]`. It reads
  both files with `pread` and needs little memory. It also accepts a second header (`_logitsA`, every position of
  one context) that no program in this directory writes.
- `context_fit_client.py`: sends the first N tokens of a text file to a running `llama-server` as one prompt with the
  prompt cache off and prints the server's timings. `context_fit_client.py fit <corpus> <n_tokens>`.

The round 4 queue and the agent-battery runner are not included: they hold machine-specific paths and addresses.
Their command lines are in `results/round4/run_log.txt`, and the battery procedure is in METHODOLOGY.md.
