# Two P100s on an AMD board

Measurements of this fork on two Tesla P100-PCIE-16GB cards in an AMD desktop board, by
[@maxlawrie](https://github.com/maxlawrie). Full write-up, scripts and the coding tasks:
[maxlawrie/p100-llamacpp-fork-trial](https://github.com/maxlawrie/p100-llamacpp-fork-trial). Nothing here was
reproduced on the 2-card machine the rest of these documents describe.

## The machine

- ASUS Prime X370-Pro, Ryzen 5 2600 (Zen+, PCIe 3.0), 64 GB DDR4. 3x P100-PCIE-16GB; the two used here are on
  the CPU's lanes at PCIe 3.0 x8 each (`nvidia-smi topo -m`: PHB). The third, on the chipset's PCIe 2.0 x4, was
  left out.
- Ubuntu 26.04, kernel 7.0, driver 580, CUDA 12.9.1, Docker. 140 W power cap per card.
- Build `83a7f0839`, sm_60 only, NCCL off.
- Qwen3.8-27B unsloth `UD-Q6_K`, the QUICKSTART server command (text only), `GGML_CUDA_P2P=1`.

## Results

| | |
|---|---|
| `pp2048`, `-ub 2048`, q4_0 KV, no MTP, cool cards | 452 t/s |
| `tg256`, no MTP | 31.1 t/s |
| server prompt at 4k / 16k / 32k | 392 / 412 / 382 t/s |
| server decode with MTP, code replies | 48-72 t/s, draft acceptance 64-79% |
| VRAM per card at start, `-c 262144` | 13.8 GiB |

At 140 W both cards reach 79-80 °C within ~2.5 minutes of continuous load in this case.

## What transfers

- **On AMD, the IOMMU can silently break `-sm tensor`.** With the kernel's default translated IOMMU domain,
  `nvidia-smi topo -p2p r` reported OK, but every peer write was dropped: the kernel logged 14,300 entries like
  `nvidia 0000:08:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0012 address=0x6800000040 flags=0x0020]`,
  from both cards, at addresses in the other card's BAR1. The cards then wait forever: 100% utilisation at ~40 W,
  `pp2048` 42.6 t/s, `tg256` not finished after 37 minutes. Booting with **`iommu=pt`** fixed it
  (`journalctl -k -b | grep "Default domain type"` should say `Passthrough`). This was on the 8 Oct tag, before
  [#12](https://github.com/Kmic-68/llama.cpp/pull/12) added an exchange for boards without P2P; with
  `GGML_CUDA_P2P=0` that build hung the same way. Not retested on the current branch without `iommu=pt`.
- **Build the branch, not the newest-looking tag.** Tag `p100-optimizations-b11529-e48e240` is from 8 Oct, 809
  commits behind the branch on 1 Nov, and GitHub doesn't list tags by date. That build aborted on a 305-token
  chat request with a tool definition: `ggml-cuda.cu:5398: GGML_ASSERT(!ggml_cuda_gemm_fold_glu_pending()) failed`.
  `GGML_CUDA_FUSE_FFN_GLU=0` didn't help, `GGML_CUDA_GEMM_FOLD=0` did. The branch build ran the hard coding set
  below without a crash, likely thanks to [#5](https://github.com/Kmic-68/llama.cpp/pull/5).
- **The build can take a weak host down.** The ~25-minute all-core CUDA build hard-reset this Ryzen 5 2600 with a
  fatal machine check on a hot day. Building on another machine and moving the image with
  `docker save | zstd | ssh host 'zstd -d | docker load'` works.

## Quality, for context

Eight harder coding tasks, graded by unit tests against the standard library or brute-force references on random
inputs, thinking off, temperature 0, one run each: **5/8** in 170 s. Qwen3.8-Flash-Next IQ3_S on the
[Strata](https://github.com/Niko1221/Strata) engine, on the same rig with 3 cards, scores 7/8 in 272 s. Task
list and failures are in the linked repository.

## Open

- The tool-call request and the easier task set weren't rerun on the branch build (a thermal cut-off stopped
  that run).
- Nothing past 32k of real context; single runs.
