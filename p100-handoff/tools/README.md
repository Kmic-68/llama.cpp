# Measurement tools used

## cpu-sampler.c
LD_PRELOAD sampling profiler. `perf` is unusable here (perf_event_paranoid=4) and gdb cannot
attach (yama ptrace_scope=1), so this spawns a sampler thread that finds the busiest thread by
reading /proc/self/task/*/stat and signals it with SIGRTMIN+5, capturing a backtrace per sample.
Two earlier designs failed and are worth knowing about: ITIMER_PROF never delivered a single
signal inside the CUDA process, and a timer_create(CLOCK_THREAD_CPUTIME_ID) armed on the main
thread also produced nothing.

    gcc -O2 -shared -fPIC -o libprof.so cpu-sampler.c -ldl -lrt -lpthread
    LD_PRELOAD=./libprof.so PROF_OUT=/tmp/prof.txt ./build-opt/bin/llama-bench ...
    cut -d';' -f1 /tmp/prof.txt.<pid> | sort | uniq -c | sort -rn | head

Result on this workload: ~72% of CPU samples are inside libcuda.so, with clock_gettime and
sched_yield underneath - the launch thread is spin-waiting on the GPU, not starved for work.

## nvprof
Nsight Compute does not support Pascal. nvprof does.

    nvprof --print-gpu-summary ./build-opt/bin/llama-bench ... -n 64 -r 1
    nvprof --print-gpu-trace --csv --log-file trace.csv ./build-opt/bin/llama-bench ... -n 8 -r 1

The trace gives grid/block/registers/duration per launch, which is how the flash-attn kernel was
caught running on 12 blocks. Beware: nvprof inflates the gaps between kernels, so read kernel
durations from it but not utilisation.

**That warning was checked and it is half right (2026-09-01).** Kernel
*durations* are accurate: the same GEMM reads 10.864 ms under nvprof and
10.863 ms un-profiled. But the *gaps* really do inflate -- nvprof costs MTP
decode ~11% end to end (48.65 t/s profiled vs 54.5 not), and that overhead lands
precisely in the host-synchronisation gaps. So:

- kernel time, per-call cost, grid/occupancy: trust nvprof.
- idle/utilisation on a **prefill** run: usable (442.74 t/s profiled vs 442.6
  not, i.e. no distortion, because prefill barely touches the host).
- idle/utilisation on a **decode/MTP** run: treat as an upper bound. The 14.5%
  host round trip in OPTLOG attempt 86 needs re-measuring with CUDA events
  inside the decode loop before anything is built on it.

## mtp-bench.sh
    ./p100-handoff/tools/mtp-bench.sh <n-max> <p-min>
Runs llama-speculative-simple with MTP and prints t/s plus the accept rate. Best
known settings are `4 0.2` (54.5 t/s). The whole flag space around this is swept
in OPTLOG attempt 79 and 87.

## SASS
    cuobjdump -sass build-opt/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o
Instruction histograms of the mmvq inner loop are how the activation loads were identified
(8 x LDG.E.CI 32-bit for the activation against 4 x LDG.E.CI.128 for the weights).
