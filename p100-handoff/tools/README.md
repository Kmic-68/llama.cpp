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

## SASS
    cuobjdump -sass build-opt/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o
Instruction histograms of the mmvq inner loop are how the activation loads were identified
(8 x LDG.E.CI 32-bit for the activation against 4 x LDG.E.CI.128 for the weights).
