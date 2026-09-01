# Benchmark harness

The metric in CLAUDE.md (`llama-bench ... -p 0 -n 256 -r 3`) is the right *headline* number but a
poor *iteration* number: it takes ~2.5 minutes and its run-to-run spread is +/- 0.2 or worse once
the cards are warm. These are what the session actually iterated on.

| script | what | why |
|--------|------|-----|
| `pp7.sh <tag>` | `llama-bench -p 512 -n 0 -b 7 -ub 7` under nvprof | 84% `mul_mat_vec_q<ncols=7>`, +/- 0.05 |
| `q4.sh` | same at `-b 4 -ub 4`, plus register usage | the column count MTP actually hits |
| `mtp.sh <n-max> <p-min>` | one MTP decode, prints t/s and acceptance | end-to-end check |
| `mtpp.sh <n-max> <p-min> <prompt> <tag>` | as above with a chosen prompt | MTP speed is content-dependent |

Prompt-processing benchmarks are the right proxy for the speculative-decoding kernels: they do a
fixed amount of work at a chosen batch width, and they are deterministic. See "invalid probes"
in the main README for why the MTP run itself cannot be used for probing.

Register usage for a given instantiation:

    cuobjdump -res-usage build-opt/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o \
      | grep -A1 "_Z13mul_mat_vec_qIL9ggml_type14ELi4E"
