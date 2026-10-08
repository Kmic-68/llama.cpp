| setting | tg512 t/s | ratio | pp2048_d0 t/s | ratio | pp2048_d16384 t/s | ratio | mean ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| default (6 runs) | 31.27 ± 0.06 | 1.000 | 506.63 ± 0.51 | 1.000 | 448.84 ± 0.74 | 1.000 | 1.000 | reference |
| algoRing (`NCCL_ALGO=Ring`) | 31.28 ± 0.05 | 1.000 | 506.78 ± 0.45 | 1.000 | 449.06 ± 1.16 | 1.000 | 1.000 | within noise |
| algoTree (`NCCL_ALGO=Tree`) | 28.47 ± 0.05 | 0.910 | 482.05 ± 0.31 | 0.951 | 430.16 ± 0.17 | 0.958 | 0.940 | slower beyond noise |
| bufA1x (`NCCL_BUFFSIZE=4194304`) | 31.25 ± 0.06 | 0.999 | 506.74 ± 0.64 | 1.000 | 448.56 ± 0.62 | 0.999 | 1.000 | within noise |
| bufB4x (`NCCL_BUFFSIZE=16777216`) | 31.22 ± 0.07 | 0.998 | 508.81 ± 0.77 | 1.004 | 449.77 ± 0.74 | 1.002 | 1.002 | within noise |
| combo (`NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_BUFFSIZE=16777216`) | 31.24 ± 0.06 | 0.999 | 515.38 ± 0.25 | 1.017 | 455.18 ± 1.07 | 1.014 | 1.010 | within noise |
| nchA1 (`NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1`) | 31.24 ± 0.05 | 0.999 | 514.80 ± 0.63 | 1.016 | 453.46 ± 0.82 | 1.010 | 1.008 | within noise |
| nchB2 (`NCCL_MIN_NCHANNELS=2 NCCL_MAX_NCHANNELS=2`) | 31.26 ± 0.06 | 1.000 | 506.19 ± 0.76 | 0.999 | 448.99 ± 1.19 | 1.000 | 1.000 | within noise |
| nchC4 (`NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4`) | 31.25 ± 0.05 | 0.999 | 501.18 ± 0.71 | 0.989 | 443.87 ± 0.88 | 0.989 | 0.992 | within noise |
| protoLL (`NCCL_PROTO=LL`) | 31.24 ± 0.08 | 0.999 | 422.44 ± 0.37 | 0.834 | 380.73 ± 0.18 | 0.848 | 0.894 | slower beyond noise |
| protoLL128 (`NCCL_PROTO=LL128`) | 29.68 ± 0.07 | 0.949 | 502.01 ± 0.64 | 0.991 | 444.34 ± 0.87 | 0.990 | 0.977 | slower beyond noise |
| protoSimple (`NCCL_PROTO=Simple`) | 28.98 ± 0.05 | 0.927 | 506.69 ± 0.59 | 1.000 | 448.93 ± 1.07 | 1.000 | 0.976 | slower beyond noise |
