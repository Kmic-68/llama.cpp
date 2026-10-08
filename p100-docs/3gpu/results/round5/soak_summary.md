- `i6_soak_B` exit=0 (cool 5s, start 0,45/1,48/2,43): 349 requests over 90.0 min, 0 failed

  | request kind | n | prefill t/s, first 15 min | last 15 min | decode t/s, first 15 min | last 15 min |
  |---|---:|---:|---:|---:|---:|
  | chat | 141 | 41.8 | 43.2 | 44.0 | 42.9 |
  | code | 96 | 51.9 | 49.3 | 45.0 | 43.9 |
  | short_before_20k | 47 | 33.2 | 33.4 | 42.4 | 43.4 |
  | pp20k | 47 | 448.3 | 447.7 | 40.3 | 38.5 |
  | short_before_64k | 9 | 49.6 | 70.8 | 39.0 | 48.1 |
  | pp64k | 9 | 385.2 | 384.2 | 38.9 | 40.9 |

  | minutes | peak MiB GPU0/1/2 | peak C GPU0/1/2 |
  |---|---|---|
  | 0-10 | 11173/11075/11163 | 60/63/60 |
  | 10-20 | 11173/11075/11163 | 60/63/60 |
  | 20-30 | 11173/11075/11163 | 61/63/60 |
  | 30-40 | 11173/11075/11163 | 60/63/61 |
  | 40-50 | 11173/11075/11163 | 61/64/60 |
  | 50-60 | 11173/11075/11163 | 61/64/61 |
  | 60-70 | 11173/11075/11163 | 60/63/61 |
  | 70-80 | 11173/11075/11163 | 61/64/61 |
  | 80-90 | 11173/11075/11163 | 61/64/61 |
  | 90-100 | 11173/11075/11163 | 60/64/61 |

  - shroud fan RPM during the soak: min 2848, mean 2944, max 3040
  - power/throttle while busy, GPU0/1/2: power-cap reason active 36.7%/36.8%/35.8%; power-brake 0.0%/0.0%/0.0%; thermal 0.0%/0.0%/0.0%; mean W 110/110/113; peak W 159/164/163; peak C 61/64/61; median SM MHz 1189/1189/1189; GPU sum 327 W mean; CPU package 47 W mean / 59 W peak; GPU+CPU peak 471 W
  - server log warning/error lines: 4
    - 1x `srv  llama_server: security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/ll`
    - 1x `set_sampler: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU`
    - 1x `spec common_specu: backend offload failed for seq_id=N; using CPU sampler`
    - 1x `srv          init: chat template supports preserving reasoning, it is enabled by default (may use more tokens,`

