# Foghorn: the full guide

[← back to the README](../README.md) · Everything about the port in one place: numbers, each model, install by hand, switches, other machines, limits.

## Qwen3.8-Flash-Next on two Tesla P100

This is a port of Strata's all-cards split engine (`strata-split-2`) to the NVIDIA Tesla P100 (Pascal, sm_60). It runs
**Qwen3.8-Flash-Next** on **two P100 16 GB** cards with MTP drafting and pictures, from either of two GGUF files, and
since 15 Nov 2026 **Edge0-145B**, Flash-Next with its experts in a latent space. On our server, with this repository's
benchmark (see "Our numbers"; the UD-Q4_K_XL row is from 16 Nov, before 18 Nov's switch and settings):

| Model file and profile | Context | RAM | Fresh writing | Edits (py / md) | Reading 38,862 tokens |
|---|---:|---|---:|---:|---:|
| ISTA-DASLab's GSQ-RCO **IQ3_S** (`setup.sh DATA`) | 524,288 | 96 GB | 66.5 tok/s | 110.7 / 115.8 | 1,012 tok/s |
| the same in the **64 GB mode** (`setup.sh DATA --ram 64`) | 262,144 | 64 GB | 63.8 | 93.6 / 99.2 | 807 |
| Unsloth's **UD-Q4_K_XL** (`setup.sh DATA --model q4_k_xl`) | 524,288 | 96 GB | 50.9 | 62.5 / 63.7 | 795 |
| Edge0's **Edge0-145B NVFP4** (`setup.sh DATA --model edge0`) | 524,288 | 96 GB | 55.0 | 96.6 / 98.9 | 691 |

> **Built for these files, on Tesla P100 cards.** Files with other expert types (Q6_K, Q3_K, Q2_K, ...) do not use
> this port's P100 kernels and run slower than Strata itself, or not at all (see "Known limitations"). On other cards,
> or with other files, use [Strata](https://github.com/Niko1221/Strata). IQ3_S is the faster of the two Flash-Next files:
> UD-Q4_K_XL is the larger quantization (72 GiB of experts against 49), so fewer of its experts fit on the cards, edits
> wait for PCIe, and 96 GB of RAM is tight for it (see "Unsloth's UD-Q4_K_XL"). Edge0-145B is a different model: it writes
> faster than UD-Q4_K_XL and slower than IQ3_S, and reads prompts slower than both; its authors report it a little better
> at agentic coding, and in our tests its Slovenian is clearly weaker (see "Edge0-145B").

**Coder** (`setup.sh DATA --coder`, either file): the router picks only from the half of the experts that ISTA-DASLab's
Coder release keeps. On our server fresh writing and code edits ran 9-11 % faster on IQ3_S, and code edits 43 % faster
on UD-Q4_K_XL. But it is a different, smaller model: it is worse outside code, and in our tests it ignored an
instruction given in Slovenian (see "Coder").

The engine is one executable. [Strata](https://github.com/Niko1221/Strata)'s Python server (v0.2.9) drives it and gives
you the OpenAI-compatible API, the tokenizer and the chat template. This repository holds the engine's sources
as our Edge0 production build was built from them (`e56c3f60` since 17 Nov 2026, with prompt-path jobs of 96 rows as
`build.sh` builds them; `f6088fde` with 64 rows before): our Coder build (`40ef6d53`, our UD-Q4_K_XL production engine
of 13 Nov plus the Coder switch, which when off gives our IQ3_S production engine's bits) plus Edge0's latent experts,
which when off give our UD-Q4_K_XL production engine's bits (see "Our numbers"). The one difference from those sources,
since 18 Nov 2026, is a switch, `FNX_MTP_NORM_STREAMS`, which only the UD-Q4_K_XL profiles set: our UD-Q4_K_XL
production engine since that day (`f28228d1`) is the sources of the one before it (`6ccc87af`) with the same change, not
these sources. Unset, the engine makes the same calls as before (see "Unsloth's UD-Q4_K_XL" and "Known limitations").
The repository also has the build, run configs for each model, and the benchmark we measure with. The models are not
included: you download them yourself (see "Prepare the model files").

- What is ours and what is Strata's: [NOTICE.md](../NOTICE.md), file by file in [docs/PROVENANCE.md](PROVENANCE.md),
  every change in [docs/upstream-v0.2.10.diff](upstream-v0.2.10.diff).
- Engine switches: [docs/SWITCHES.md](SWITCHES.md). How the port came about: [docs/HISTORY.md](HISTORY.md).

## Our numbers

### This release, with the benchmark

[bench/bench.py](../bench/bench.py) through Strata v0.2.9's server, on this folder's sources as they were before 18 Nov's
switch, built with prompt-path jobs of 96 rows (`e56c3f60`; `build.sh`'s default, see "Build"). The IQ3_S columns ran on
17 Nov 2026 (09:24-09:44) with the profiles as they are now in `profile/`, the UD-Q4_K_XL columns on 16 Nov
(19:30-19:50) with its profiles as they were until 18 Nov (`FNX_EXPERT_RAM_GIB=52.5`, `FNX_EXPERT_DYN_GIB=18`,
`FNX_MTP_NORM_STREAMS` unset). The benchmark has not been run with 18 Nov's UD-Q4_K_XL settings; our rounds with them
are in "Unsloth's UD-Q4_K_XL". That build gives the bits of our production engines for IQ3_S (`36409ddc`) and, until
18 Nov, UD-Q4_K_XL (`6ccc87af`, both below) and is our production engine for Edge0-145B. Each profile was written by
`profile/fill_profile.py` at `FNX_ROUTE_PRIOR=0` (exact, see "Exact and lossy") and run once, on a freshly loaded
engine. Writing is at temperature 0 unless marked. These are the benchmark's own inputs, so your run compares with them
directly (it prints them beside yours). Edge0-145B's run, on 17 Nov, is in "Edge0-145B".

| Test | IQ3_S, 512k | IQ3_S, 64 GB mode, 256k | UD-Q4_K_XL, 512k | UD-Q4_K_XL, 256k |
|---|---:|---:|---:|---:|
| Fresh writing: 12 new prompts, 300 tokens each (tok/s) | **66.5** | **63.8** | **50.9** | **50.9** |
| Edit: the model repeats Strata's `tools/collect_profile_traces.py` with a variable renamed (1,197 tokens) | **110.7** | **93.6** | **62.5** | **63.4** |
| Edit: the model repeats 5,000 characters of Strata's `docs/SECOND_GPU.md` with a word replaced (1,228 tokens) | **115.8** | **99.2** | **63.7** | **65.0** |
| Cold read of 38,862 tokens | **1,012** | **807** | **795** | 929 (not cold, see below) |
| Writing right after that read | 61.5 | 57.4 | 49.2 | 48.5 |
| New messages on a running conversation: 2,029 / 3,586 / 5,259 / 7,142 / 11,081 / 14,212 / 20,243 tokens | 729 / 854 / 900 / 842 / 1,000 / 927 / 961 | 628 / 741 / 795 / 736 / 898 / 825 / 862 | 537 / 716 / 767 / 722 / 877 / 779 / 840 | 605 / 808 / 856 / 807 / 969 / 869 / 930 |
| Picture (96 x 96 PNG, encoder on the CPU): time to the answer; writing after it (sampled) | 5.1 s; 52.9 | off | 6.5 s; 46.8 | 3.0 s; 47.0 |
| Load, until the server reports loaded | 308 s | 244 s | 209 s | 201 s |

- **This release's sources with Edge0** (`f6088fde`; before 18 Nov's switch), with `FNX_LATENT` unset, against our
  UD-Q4_K_XL production engine (`c213636c`) on the UD-Q4_K_XL 512k profile at `FNX_ROUTE_PRIOR=0`, in a test window on
  15 Nov (round `izdaja1`): the same first-token logits in 18 of 18 requests (KL 0) and the same speeds (fresh 52.5
  against 51.0, edits 61.9 / 63.7 against 62.0 / 64.0 tok/s). The IQ3_S file could not be compared that day (its first
  shard was not on our server).
- **The same bits as our production.** On 13 Nov the engine that this repository's `build.sh` then built on our server
  (`f85a4aba`, 64-row jobs) ran the IQ3_S columns without pictures: fresh 66.6, edits 111.0 / 116.0, a cold read 954,
  medium reads 700 / 815 / 853 / 803 / 941 / 876 / 904; in the 64 GB mode 63.3, 93.6 / 101.9, 768 and 613 / 717 / 759 /
  710 / 850 / 783 / 817. Our IQ3_S production engine of then (`bf9c7f1f`) with the same profile gave the same
  first-token logits in 27 of 27 requests and the same speeds (66.9, 111.0 / 116.0, 953). The 64 GB mode gave the IQ3_S
  profile's logits in 27 of 27 (for that comparison with the same YaRN setting as the 512k profile; speed does not depend
  on it). The IQ3_S 512k run of 17 Nov above gave 13 Nov's logits bit for bit in 27 of 27 requests; the 64 GB run of
  17 Nov was not compared again. Our IQ3_S production engine since 17 Nov (`36409ddc`: `bf9c7f1f`'s sources with 96-row
  jobs) gave `bf9c7f1f`'s logits in 27 of 27 requests (KL 0; round `n17s1` in "Our rounds"). The picture's first answer
  in the 17 Nov run (5.1 s) was that morning's first with pictures on this file; in our rounds it takes 2.6-2.7 s.
  In 13 Nov's morning window that build with the UD-Q4_K_XL profile of then and our production engine for it
  (`c213636c`) gave the same logits in 29 of 29 requests, the picture's included. This folder's sources since then
  (`40ef6d53`, `c213636c` plus `FNX_EXPERT_MASK`) with the switch unset gave our IQ3_S production engine's logits in 27 of
  27 requests (KL 0) and its speeds (our test chain: fresh 66.96 against 66.86, a 36k read 943.9 against 942.5).
  On 16 Nov the UD-Q4_K_XL runs above gave the first-token logits of 13 Nov's runs on `c213636c` in 29 of 29 requests
  each (512k and 256k). Our production engine for UD-Q4_K_XL since that day (`6ccc87af`: `c213636c`'s sources with
  96-row jobs) gave the 512k run's logits in 29 of 29 and its speeds (fresh 50.7, edits 62.7 / 63.8, a cold read 786,
  medium reads 541 / 722 / 771 / 727 / 878 / 779 / 843, the picture 3.0 s; 45.8). Since 18 Nov it is `f28228d1`
  (`6ccc87af`'s sources with `FNX_MTP_NORM_STREAMS`), which with 18 Nov's settings gave `6ccc87af`'s first tokens in 81
  of 81, 17 of 17 and 7 of 7 requests of our rounds (KL 0; see "Unsloth's UD-Q4_K_XL").
- **The 64 GB mode** ran in a memory cgroup of 58 GiB on our 96 GB server (see "64 GB of RAM").
- **UD-Q4_K_XL:** 10 GiB of RAM stayed available at the lowest in both runs. While the 512k profile loaded, 1.3 GiB went
  to the swap file (more than 20 GiB were available at the time) and swap use stayed there until the engine exited; the
  256k run added none. The 256k profile ran right after the 512k one, so most of the lookup table was still in the page
  cache: its "cold" read is warm (the 512k profile's 795 is the cold one). The picture's first answer took 6.5 s in the
  512k run, the first of that window, and 3.0 s in the three runs after it. On 13 Nov, on the 64-row build with
  `FNX_PF_JOBV=7` and the 2-bit draft experts, the cold read was 765 and the warm one 901, the medium reads 540 / 681 /
  721 / 687 / 813 / 732 / 781 (512k) and 603 / 755 / 793 / 752 / 887 / 805 / 853 (256k), fresh writing 50.3 / 51.1,
  edits 62.6 / 63.5 and 63.0 / 63.7, writing after the picture 42.6 / 43.2. The run of 13 Nov's morning with the
  UD-Q4_K_XL profile of then (256k, `FNX_EXPERT_DYN_GIB=14`, six guessed experts a row in the next-layer copies) gave
  fresh 41.4, edits 45.3 / 43.9, a cold read 726, and ran short of RAM (4-7 GiB available, the swap file full, the
  picture's first answer 48.4 s). See "Unsloth's UD-Q4_K_XL".

### Our rounds

The rounds named (`splitNN`, `q4kN`) are in our measurement logs, which are not part of this release (they are in
Slovenian and mixed with our private setup); the numbers and findings in this guide and the rest of `docs/` are taken from them.
They ran our own chain of tests (the benchmark's fresh prompts and Markdown edit, other texts for the reads and the
Python edit), IQ3_S at `FNX_ROUTE_PRIOR=0`:

| Test | Result | Measured in |
|---|---|---|
| Fresh writing: 12 new prompts, 300 tokens each | **66.7-67.2** tok/s | rounds `split69`, `split71`, `split72`, `split74` |
| Edit: the model repeats a 5,000-character Markdown text with one word replaced (1,228 tokens) | **114.7-116.2** tok/s | the same rounds |
| Edit: the model repeats a 4.6 KB Python file with a variable renamed (1,388 tokens) | **108.5-109.1** tok/s | the same rounds (private input file; the benchmark uses a public one) |
| Cold read of 36,424 tokens | **961** tok/s | `split74`, `split75` (with `FNX_PF_SP`; 920-924 without) |
| Writing right after that read | 56-60 tok/s | `split69`-`split73` |
| New messages on a running conversation: 1,719 / 3,460 / 4,582 / 7,181 / 10,795 / 13,840 / 20,787 tokens | 656 / 808 / 839 / 807 / 946 / 900 / 917 tok/s | `split75` |
| A read deep in a conversation: 58,555 tokens | 941 tok/s | `split75` |
| Picture (96 x 96 PNG, encoder on the CPU): time to the answer | 2.6-2.7 s | `split69` |
| Writing after the picture (sampled) | 52.4-52.6 tok/s | `split69` |
| Load, until the server reports loaded | 305-335 s | `split69`-`split75` |

- **Deep contexts**, measured on an older build (`c16e3b6d`, 11 Nov): reading 0 → 134k tokens 835 tok/s,
  131k → 281k 800 tok/s, 279k → 505k 725 tok/s. Later builds only made reading faster.
- **With `FNX_ROUTE_PRIOR=0.15`** (our production setting, lossy) on the build of 11 Nov (`b1438fe1`, `split67`):
  fresh 68.3, edit py 113.1, md 117.5 tok/s, against 66.6, 105.3 and 111.5 at 0 in the same round.
- **Prompt-path jobs of 96 rows** (since 17 Nov 2026; the table above is from 64-row builds). Round `n17s1`, the same
  chain at `FNX_ROUTE_PRIOR=0`, our production engine of then (`bf9c7f1f`) against its sources with 96-row jobs
  (`36409ddc`, our production engine since): the cold 36k read 954.8 → 1005.0 tok/s (+5.3 %), the reads on a running
  conversation 660 / 810 / 840 / 806 / 950 / 901 / 923 → 669 / 839 / 867 / 830 / 995 / 940 / 965, writing the same
  (fresh 67.3 and 67.1, edits 108.9 / 115.7 in both), the same first-token logits in 27 of 27 requests (KL 0). Kernel
  mode 6, which UD-Q4_K_XL takes, read at 1003.1, so IQ3_S keeps `FNX_PF_JOBV=7`. At our production's
  `FNX_ROUTE_PRIOR=0.15` (round `n17s2`) the read went 955.8 → 1004.8, with fresh writing at 69.6 and 70.2.
- Two requests at once (`--batch 2`, an older build): about 32 tok/s each, 63 together, so no gain over one.

### The machine

| | |
|---|---|
| Server | HPE ProLiant DL380 Gen10 |
| CPUs | 2 x Intel Xeon Bronze 3106 (8 cores, 1.7 GHz, no Hyper-Threading), 2 NUMA nodes |
| RAM | 96 GB DDR4 RDIMM (6 x 16 GB, 3 channels per CPU); Linux sees 91.8 GiB |
| GPUs | 2 x Tesla P100-PCIE-16GB, one on each CPU, each in a PCIe Gen3 x16 slot; power limit 185 W (default 250 W) |
| Host to card | 11.9 GB/s per card, the same from either CPU's memory |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38, kernel parameter `iommu=pt`, transparent huge pages `madvise` |
| Driver, toolkit | NVIDIA 580.178.04; CUDA 12.4.131 (Ubuntu's package), host compiler g++ 13, CMake 4.2.3 |
| Server side | Strata v0.2.9's `serve/` on Python 3.14.4 |

Our server runs Strata v0.2.9 with two local changes that are not on the speed path: a `/slots` endpoint for our own app,
and tool-call parsing fixes backported from Strata 0.2.12. Plain v0.2.9 is what you need.

## 64 GB of RAM

The default profile holds 82-85 GiB once loaded: a pinned RAM copy of every expert (48.8 GiB) and the 26.8 GiB lookup
table (`per_layer_token_embd`, the second shard) copied into RAM. `setup.sh --ram 64` (or `fill_profile.py --ram 64`)
writes a profile for a machine with 64 GB, IQ3_S only:

- `FNX_PLE_RAM=0`: the lookup table is read from its file through the page cache, as Strata's own engine does; each
  decode window or prompt chunk asks for its rows' pages at once.
- `FNX_EXPERT_DYN_GIB=13`: of the experts in VRAM only the coldest 13 GiB keep a RAM copy and take part in the expert
  cache; the other 6.8 GiB stay in VRAM for good, with no copy. RAM then holds 42.0 GiB of experts instead of 48.8.
- `--max-context 262144` (the trained length, no YaRN) and a 2,000 MiB conversation cache. At 524,288 positions the
  K/V pool needs nearly every expert slot to be one it can borrow, so there is nothing to keep in VRAM for good; the load
  says so and stops if the two do not fit together.

We have no 64 GB machine. We ran the profile on our 96 GB server inside a memory cgroup of 58 GiB without swap
(`systemd-run --user --scope -p MemoryMax=58G -p MemorySwapMax=0`), which counts the pinned copies, the engine's own
memory and the page cache, after dropping the lookup table's file from the page cache (as after a reboot):

- **The load** peaked at the cgroup's 58 GiB without an out-of-memory kill: 55.6 GiB that cannot be dropped (42.0 GiB of
  pinned experts and 13.6 GiB the load needs for a while), the rest page cache.
- **Once loaded** the engine holds 45-46.5 GiB (43.4 GiB pinned, 1.5-3 GiB of its own); the page cache took about 3 GB of
  the lookup table by the end of the benchmark.
- **Speed** against the 512k profile on 96 GB (table under "Our numbers"): fresh writing −4 %, edits −15 % and −14 %, a
  cold 38k read −20 %, the reads on a running conversation −10 to −14 %, writing after a long read −7 %. The answers
  are the same: first-token logits identical in 27 of 27 requests (compared on 13 Nov, on the 64-row build). The lookup
  table had been dropped from the page cache before the run (as after a reboot), so these are figures of a cold start.
- On a real 64 GB machine Linux sees about 62 GiB. Leave little else running: the load's 55.6 GiB must fit beside the
  system, and what is left is page cache for the lookup table.

## Unsloth's UD-Q4_K_XL

[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), folder `UD-Q4_K_XL`, four
parts, 111.3 GB. Its routed experts are Q4_K for gate|up (layer 2 Q5_K) and Q5_1 for down (layers 2, 4, 30, 46 and 47
Q8_0): 71.7 GiB in all, against IQ3_S's 48.8. This release has P100 kernels for those four types, for writing (the row
kernel) and for reading prompts (the job kernels), made like the IQ3_S ones: the values become FP16 exactly, then the
block's scales apply. `setup.sh --model q4_k_xl` downloads it with ISTA-DASLab's picture projector and writes
`profile/flash-next-q4_k_xl-512k.template.json` (524,288 positions with YaRN x2, as the IQ3_S profile;
`profile/flash-next-q4_k_xl-256k.template.json` is the same at 262,144 without YaRN, through `fill_profile.py --template`);
[docs/SWITCHES.md](SWITCHES.md#the-ud-q4_k_xl-profiles) lists what differs from the IQ3_S profile and why.

- **96 GB of RAM** at least. The profiles use the 64 GB mode's switches to fit at all: with every expert pinned (73 GiB),
  9 GiB of page cache were left for the 28.8 GB lookup table and a 36k read ran at 146 tok/s. With the table read
  through the page cache and only the experts the cards may give up kept in RAM (`FNX_EXPERT_DYN_GIB=18.430`; 18 until
  18 Nov and in the benchmark runs), RAM holds 70.5 GiB of experts and 10 GiB stayed available at the lowest in the
  benchmark runs. The first long read after a load waits for the disk (795 tok/s at 38,862 tokens); later ones find more
  of the table in the page cache (about 930).
- **Q4_0 draft experts** (`FNX_MTP_EXPERTS=q4_0`, since 13 Nov 2026): the MTP draft layer's experts made from Qwen's BF16
  tensors with `scripts/mtp_q4.py` instead of Strata's 2-bit ones, in the same VRAM. The model checks every drafted token,
  so the answers stay the same token for token; more drafts are accepted, and fresh writing ran 10 % faster (our round
  `mtp3` at `FNX_ROUTE_PRIOR=0`: 46.99 -> 51.84 tok/s; edits, drafted mostly by prompt lookup, equal). With IQ3_S, whose
  distribution sits further from the BF16 model the draft now follows, the gain was about 1 %, so its profile keeps the
  2-bit file. `setup.sh` makes the file. The benchmark above ran with it; its twelve fresh prompts gained less than our
  round's (50.3 tok/s on 13 Nov with the 2-bit file, 50.9 on 16 Nov: fewer rounds, but longer ones).
- **Prompt-path jobs of 96 rows and kernel mode 6** (`build.sh`'s default, `FNX_PF_JOBV=6`; since 16 Nov 2026). The
  P100 job kernels work on two tiles of 48 rows, so a 96-row job fills them and a 64-row one leaves a third empty. A 36k
  read in our test chain, the lookup table in the page cache (round `n16p1`, `FNX_ROUTE_PRIOR=0`): 736.0 tok/s with 64
  rows and mode 7 (our production until then), 864.3 with 96 rows, 976.4 with 96 rows and mode 6 (+32.7 %); medium reads
  +7-20 %, writing unchanged, the same first-token logits in 27 of 27 requests (KL 0). The benchmark's long read is the
  first after a load and waits for the disk, so it shows less (765 → 795); its medium reads from 3.6k tokens gained 5-8 %
  at 512k and 7-9 % at 256k. The profiles' mode 6 on a 64-row build (`PF_JOB=64`; our production engine of 13 Nov, the
  512k profile): a cold read 761 and medium reads 545 / 699 / 749 / 702 / 841 / 745 / 808, between 13 Nov's and the
  96-row build's, the same logits in 29 of 29 requests. The other two models gain less from 96 rows (17 Nov, the same
  36k read): IQ3_S 5.3 % and keeps mode 7 (see "Our rounds"), Edge0-145B 3.2 % with mode 6 (see "Edge0-145B").
- **The draft layer's source normalised per stream** (`FNX_MTP_NORM_STREAMS=1`, a new switch, off unless set; since
  18 Nov 2026). The RMS norm of the MTP draft layer's source (`pre_fc_norm_hidden`, 4 x 2560 weights) takes each of a
  row's four hyper-connection streams on its own, as the hyper-connection norms do, instead of the four as one vector
  of 10,240 values; the tensors do not say which it is. Only the drafts change, and the model checks every drafted
  token, so the answers stay the same. Alone, against our production engine of then (`6ccc87af`; round `n17v1` on
  17 Nov, three loads of each, the 512k profile): fresh writing 50.82 → 52.79 tok/s (+3.9 %; 71.9 → 74.3 % of the
  drafted tokens accepted, 2.46 → 2.64 tokens a round), writing after a 36k read 44.67 → 47.97 (+7.4 %), edits 61.82 →
  60.50 (py, −2.1 %) and 63.81 → 63.32 (md, −0.8 %); the same first tokens in 77 of 77 requests (KL 0). A round takes
  longer with it (the cards' time in fresh writing 41.0 → 42.7 ms) and gives more tokens; edits, drafted mostly by
  prompt lookup, get no more of them (95.8 and 95.5 % accepted). Unset or 0, the engine makes the same call as before.
  With IQ3_S and Edge0-145B it was not measured, and their profiles leave it unset. The idea is from the Colibri
  engine's notes (see "Credits and license").
- **Fewer K/V pages of the pool's own, more expert slots** (`FNX_EXPERT_RAM_GIB=52.071` and `FNX_EXPERT_DYN_GIB=18.430`,
  before 52.5 and 18; settings only, since 18 Nov 2026). The K/V pool keeps 153 pages of its own (29,376 positions)
  instead of 305 (58,560), and the VRAM that frees is 150 expert slots more a card: 5,495 slots of 1.46 MiB (7.9 GiB;
  before 5,345 and 7.6), which the prompt path's buffers borrow while it reads and the pool as the context grows, up to
  2,976 pages (571,392 positions). 17,885 experts instead of 18,035 are in RAM only; RAM still holds 70.5 GiB of
  experts. Alone, on top of the switch above (round `n17k1`, two loads of each): the cards' time for a round of fresh
  writing 42.42 / 42.44 → 41.73 / 41.91 ms (−1.4 %), for a token of an edit 15.77 / 15.72 → 15.46 / 15.51 ms (−1.7 %),
  the same first tokens in 39 of 39 requests (KL 0). 290 slots more (51.671 and 18.830) were faster still (−2.1 % and
  −3.6 %), but at load they left a card only 15-17 MiB above the least the load needs; 150 leave about 220 MiB. A card
  with less to spare (another process on it, for example) stops the load with `FNX_EXPERT_DYN_GIB: VRAM short by …`.
  The price: the pool borrows expert slots from 29.4k positions on, before from 58.5k, and while it does, a conversation
  left idle for 2 minutes is parked to RAM and restored at its next turn (`IDLE_PARK` in `engine/tools/sfnx.cpp`).
- **Both together, against our production before them** (`6ccc87af` with 52.5 and 18; rounds `n18s1`, `n18g1` and
  `n18r1` in the night to 18 Nov, the 512k profile, the same prompts, old and new in the same test window). Fresh
  writing 51.75 → 53.45 tok/s (+3.3 %); edits 61.6-61.9 → 61.7-61.9 (py) and 63.7-64.0 → 64.4-65.1 (md, +1.6 %); writing
  after a 36k read 43.1 → 45.9. Deep reads and the writing after them (`n18g1`): 36k tokens 796.5 / 45.7 → 807.1 / 47.5
  tok/s, 134k 856.8 / 44.4 → 880.3 / 47.7, 284k 801.2 / 41.8 → 826.9 / 42.4, 501k 751.2 / 33.7 → 762.7 / 35.1.
  **Writing at a depth of 58.5k is slower:** 49.2 → 47.1 and 51.8 → 50.9 tok/s (two code tasks, −4.3 % and −1.7 %).
  Which change that is we did not measure: the switch alone made edits 0.8-2.1 % slower, and the pool now borrows
  expert slots from 29.4k positions on (before from 58.5k), though writing after the reads of 36k to 501k tokens above,
  all past 29.4k, was faster. Reading is equal once the lookup table's rows are in the page cache (`n18r1`, loads old,
  new, old, new: a 20,787-token prompt read at 938 / 810 / 939 / 940 tok/s the first time and at 939 / 942 / 940 / 940
  the second). A first read 10-20 % slower after a load is a cold page cache for the 28.8 GB lookup table's rows, not a
  setting. The same first tokens in 81 of 81, 17 of 17 and 7 of 7 requests (KL 0), the same texts at temperature 0 in
  6 of 6, no errors. The 256k profile has the same three values, as our production does; the rounds ran on the 512k
  one. These are our rounds, not `bench/bench.py`: its numbers in "Our numbers" are of 16 Nov.
- **To get them** on a machine with the previous release: stop the server (never build beside a loaded engine), update
  this folder (`git pull`) and run `bash build.sh` again: the switch is new engine code, and an engine built from earlier
  sources ignores `FNX_MTP_NORM_STREAMS` without a word. `setup.sh` keeps an existing `DATA/profile.json`, so in each
  UD-Q4_K_XL run config set `"FNX_EXPERT_RAM_GIB": "52.071"` and `"FNX_EXPERT_DYN_GIB": "18.430"` and add
  `"FNX_MTP_NORM_STREAMS": "1"` to its `env`, or delete the file and run `setup.sh` again with the options you gave it
  before. At the next load the engine's log has `sfnx: the draft layer's source normalised per stream` and, on ours,
  `the K/V pool's own 153 pages (29376 positions)` (the count follows the VRAM free on your cards). The IQ3_S and
  Edge0-145B profiles have not changed.
- **To go back,** in the written JSON: `FNX_EXPERT_RAM_GIB=52.5` and `FNX_EXPERT_DYN_GIB=18` undo the second;
  `FNX_MTP_NORM_STREAMS=0`, or the line removed, undoes the first.
- **What the RAM copies are for.** Of the 71.7 GiB of experts 52.1 GiB live only in RAM; the cards hold the rest. Those
  keep a RAM copy when the prompt path (its staging buffers) or the K/V pool (as the context grows past its own pages)
  may borrow their slots, and when they are dynamic: the cards' expert cache then follows the work. With 5.2 GiB fixed on
  the cards by a general route profile instead (`FNX_EXPERT_DYN_GIB=14` beside `FNX_EXPERT_RAM_GIB=52.5`, 4 GiB less RAM,
  256k only) edits ran 12-14 % slower.
- **Next-layer copies of 2 guessed experts a row** (`FNX_PREFETCH_WANT=2`, IQ3_S 6): every guessed copy competes with a
  window's own missing experts for PCIe. In our rounds at 512k (`q4k5`, `q4k7`) edits went from 67.4-70.3 tok/s at 6 a
  row to 77.4-84.3 at 2 (9 a row: 54-57; none: 72.2-78.1); fresh writing stayed at 55.8-57.5.
- **Slower than IQ3_S**, most of all in edits. In our rounds (our own test chain, with our production's
  `FNX_ROUTE_PRIOR=0.15`) UD-Q4_K_XL at 512k ran fresh writing at 56.9 tok/s against IQ3_S's 69.3 (82 %), edits at
  77.4-84.3 against 115.0-122.9 (68 %), a cold 36k read at 819 against 946 (87 %). On the cards are 27 % of its experts
  (IQ3_S 41 %), and each missing one is 1.33 times bigger: an edit's prompt-lookup windows of up to 16 rows touch about
  100 different experts a layer and wait for the PCIe copies (an edit's round 151 ms against IQ3_S's 99). Shorter lookup
  drafts (`--lookup` 5, 7, 10), NUMA placement of the RAM copies and half-size staging buffers did not help. Those
  reads were on the 64-row build with mode 7; with 96 rows and mode 6 the chain's 36k read at `FNX_ROUTE_PRIOR=0` is 976
  tok/s, against IQ3_S's 1,005 with 96 rows (955 with 64).
- **Deep contexts** (round `q4k4` on 13 Nov, the 64-row build, 512k): reads to 134k / 284k / 501k tokens ran at 809 /
  765 / 737 tok/s and writing there at 49.3 / 48.0 / 38.4 tok/s, close to IQ3_S's (835 / 800 / 725, writing about 53 /
  50 / 37). This release's deep reads (round `n18g1`) are under "Both together" above.
- **Pictures** work with the same projector; the encoder reads the vocabulary from the first part.
- **The same bits as our production:** see "Our numbers". The kernels compute in FP16 like the IQ3_S ones, so against
  other engines the answers are close, not identical.

## Coder

[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
(Apache-2.0) removes half of the routed experts, 256 of 512 in each layer. RCO chose them for code, agentic tool use and
pictures, and the model card says abilities outside that set degrade. We do not use their file. Its allocation list
names the experts it keeps by their index in the full model ("the mask can be reapplied to the original weights").
`FNX_EXPERT_MASK` applies that list to the full IQ3_S or UD-Q4_K_XL weights: the router never picks a dropped expert,
and the ten picks are weighted among themselves as in the pruned model. So this is the Coder's choice of experts with
your file's weights, not ISTA-DASLab's Coder file.

- **Turn it on:** `bash setup.sh /data/flash-next-coder --models /data/flash-next/models --coder` writes a Coder profile
  into its own folder, sharing the model files. `setup.sh` keeps an existing `profile.json`, hence the new folder. By
  hand: `fill_profile.py --coder`, or add `"FNX_EXPERT_MASK": "<this repository>/engine/data/coder_mask.txt"` to a
  profile's `env`.
- **The list:** `engine/data/coder_mask.txt`, one line a layer. `python3 scripts/coder_mask.py ALLOCATION.txt OUT.txt`
  makes it from their
  `tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.rco-allocation.txt` (revision
  `bba88856f8c3c0c9e40e45f82d05e228b814ac43`), byte for byte.
- **Why it is faster:** each token's ten experts come from 256 a layer instead of 512, so more of them are on the cards
  already. The dropped experts go to RAM first, which leaves the cards' slots to the kept ones. RAM use does not change
  (every expert is still loaded), so it needs 96 GB like the full profile. With the 64 GB mode it has not been tried.

**Speed.** Our own test chain (the benchmark's fresh prompts and Markdown edit, other texts for the reads and the Python
edit, see "Our rounds"), not `bench/bench.py`. It ran at our production's `FNX_ROUTE_PRIOR=0.15`, full and Coder in the
same test window on 13 Nov 2026, each once on a freshly loaded engine, tok/s:

| Test | IQ3_S | IQ3_S, Coder | UD-Q4_K_XL, 512k | UD-Q4_K_XL, 512k, Coder (two runs) |
|---|---:|---:|---:|---:|
| Fresh writing: 12 new prompts, 300 tokens each | 68.8 | **75.2** (+9 %) | 55.3 | **60.9-61.6** (+10-11 %) |
| Edit: a 4.6 KB Python file with a variable renamed (1,388 tokens) | 116.7-118.8 | **130.2-131.2** (+11 %) | 77.5-77.6 | **110.2-112.6** (+43 %) |
| Edit: a 5,000-character Markdown text with a word replaced (1,228 tokens) | 121.5-122.5 | 133.0-133.9, did not edit | 79.4-83.2 | 102.6-108.5, did not edit |
| Cold read of 36,424 tokens | 943 | **979** (+4 %) | 695 | 751-886 |
| New messages on a running conversation, 1,719 to 20,787 tokens | 665-941 | 738-980 | 459-781 | 571-906 |
| Picture, asked in Slovenian: the answer, time to it | "Rdeča in modra", 3.4 s | "red, blue", 2.9 s | not run | "red, blue", 3.2 s |

With `FNX_EXPERT_RAM_GIB=26` instead of 29 (more of the kept experts on the cards) IQ3_S Coder was not faster (fresh
73.6). UD-Q4_K_XL's reads vary between runs because its RAM is tight (see "Unsloth's UD-Q4_K_XL").

**What it does to the answers.** It is lossy by design, far more than `FNX_ROUTE_PRIOR` (see "Exact and lossy"):

- **First token:** against the full IQ3_S model, both at 0.15, the first-token logits of our 29 test requests gave KL
  2.2 on average, median 0.15, and the same first token in 23 of 29. UD-Q4_K_XL Coder against full IQ3_S: the same
  first token in 22-23 of 27-29. For scale, in the same windows: full UD-Q4_K_XL (at 0.15) against full IQ3_S (at 0)
  gave KL 0.041 (26 of 27), and our IQ3_S production engine against this build without the mask, both at 0.15,
  KL 0.0034 (29 of 29; at 0 the two give KL 0).
- **The Python edit** (rename a variable): the same text as the full model's, on both files.
- **The Markdown edit,** asked in Slovenian to replace a word in a Markdown text: the full model replaced every
  occurrence, and the Coder returned the text without a single replacement on both files. So its speed in that row is
  the speed of copying the text.
- **The picture question,** asked in Slovenian: the full model answered in Slovenian, the Coder correctly but in English.
- **Code benchmarks:** we have not run any. ISTA-DASLab's card gives, for their own file, SWE-bench Verified 79.20
  (82.80 at BF16) and LiveCodeBench v6 87.43 (the same as BF16). Those are not numbers for the masked IQ3_S or UD-Q4_K_XL.

Use it for code. For general use, and for instructions in languages other than English, keep the full model. We have
not checked it on English prose.

## Edge0-145B

[Edge0/Edge0-145B-NVFP4](https://huggingface.co/Edge0/Edge0-145B-NVFP4) (revision `cf9b3679`, the Edge0 Community Model
License 1.0) is Qwen3.8-Flash-Next with its routed experts moved into a latent space. In each layer D maps a token's 2,560
values into 8 groups of 1,792; each of the 512 experts belongs to one group and works there (gate|up 640 x 1792, down
1792 x 640, in NVFP4); their outputs are summed per group, and U takes the sum back to 2,560. The router, the dense layers,
the lookup table and the MTP draft layer are Flash-Next's byte for byte (we compared them), so the same tokenizer, draft
layer and picture projector work with it. 145B parameters, 6.6B active. Its authors report it a little better than
Flash-Next at agentic coding and about equal at reasoning (below); we have not measured that.

`setup.sh DATA --model edge0` downloads its four files into `DATA/models/edge0-145b/` (57.3 GB: the skeleton GGUF with the
dense layers and the tokenizer, 6.0 GB; `experts.bin`, 47.6 GB; `latent_dense_q8.bin` with D and U, 3.7 GB; `labels.bin`)
and the IQ3_S's second shard (28.8 GB), which holds the lookup table. Give `--models` the IQ3_S's folder to share that
shard. The engine maps Edge0's files as they are (`FNX_LATENT`); nothing is converted. The profile is
`profile/flash-next-edge0-512k.template.json` (524,288 positions with YaRN x2, as the IQ3_S profile);
[docs/SWITCHES.md](SWITCHES.md#the-edge0-145b-profile) lists what differs.

**Speed,** tok/s. The first column is `bench/bench.py` as in "Our numbers" (17 Nov 2026 in a test window, 07:16-07:25,
a freshly loaded engine, the profile written by `profile/fill_profile.py`), on this folder's sources of then built with
prompt-path jobs of 96 rows (`e56c3f60`; see "Build"), our Edge0 production engine since that day. The others are our own
test chain (see "Our rounds"), round `n17e2` of the same morning: that engine, and our production until then (`f6088fde`,
64 rows, `FNX_PF_JOBV=7`). All at `FNX_ROUTE_PRIOR=0`:

| Test | bench.py | Round `n17e2`: 96 rows, mode 6 | Round `n17e2`: 64 rows, mode 7 |
|---|---:|---:|---:|
| Fresh writing: 12 new prompts, 300 tokens each | **55.0** | 55.15 | 54.81 |
| Edit, Python (the benchmark's file; in our rounds our 4.6 KB file, 1,388 tokens) | **96.6** | 94.5 | 94.5 |
| Edit, Markdown (1,228 tokens) | **98.9** | 98.9 | 98.5 |
| Cold read (bench.py 38,862 tokens, our rounds 36,424) | **691** | 689.3 | 668.1 |
| Writing right after that read | 57.1 | 49.7 | 49.0 |
| New messages on a running conversation (bench.py 2,029-20,243 tokens, our rounds 1,719-20,787) | 535 / 612 / 637 / 607 / 686 / 647 / 669 | 497 / 608 / 616 / 603 / 684 / 659 / 670 | 481 / 588 / 601 / 586 / 664 / 639 / 650 |
| Picture (96 x 96 PNG): time to the answer; writing after it | "Rdeča in modra", 5.1 s; 43.3 | "Rdeča in modra", 1.9 s; 42.0 (round `edge0v`, 15 Nov) | |
| Load, until the server reports loaded | 215 s | 220 s | 225 s (round `n17e3`) |

- **Prompt-path jobs of 96 rows and kernel mode 6** (`build.sh`'s default, `FNX_PF_JOBV=6`; since 17 Nov 2026), as
  for UD-Q4_K_XL. In round `n17e2` the 36k read went from 668.1 tok/s (64 rows, mode 7) to 686.1 (96 rows) and 689.3
  (mode 6; +3.2 %), medium reads +2.5-3.4 %, writing unchanged, the same first-token logits in 27 of 27 requests (KL 0).
  It gains less than UD-Q4_K_XL: its prompt path also computes D and U, which the size of the jobs does not touch. The benchmark on
  15 Nov, on the 64-row build with mode 7: fresh 55.1, edits 96.6 / 99.1, a cold read 669, medium reads 519 / 592 / 615 /
  587 / 666 / 627 / 650, the picture 5.1 s; 46.3, load 210 s.

- **Experts on the cards** are what counts, as for UD-Q4_K_XL. With D and U whole on both cards and
  `FNX_EXPERT_RAM_GIB=36`, 4,603 experts were on the cards: fresh 44.9, edits 56.3 / 56.5, a 36k read 575. At 31.5,
  7,099: 50.7, 81.0 / 83.9 and 577. With D and U split by group across the cards (`FNX_LAT_SPLIT=1`, 1.74 GiB a card
  less) and 28, 9,041: 54.9, 94.5 / 98.4 and 658. Below 28 the engine moves experts back to RAM itself (at 26.5: 9,017).
- **Why it is slower than IQ3_S** (in the benchmark 17 % in fresh writing, 13-15 % in edits, 32 % in reading): D and U are
  dense, 73 million multiply-adds a token and layer, more than its ten experts. Each card computes half of them and reads half of their 3.7 GB at every
  step, and a layer has two exchanges more than Flash-Next's.
- **What the engine adds** (`history/patches/edge0_patch*.py`): an NVFP4 type in the P100 kernels for writing and for
  prompts (64-value blocks, four UE4M3 scales, the E2M1 values exact in FP16; the scales are divided by 16 so that FP16
  sums cannot overflow, and the experts' tail scales take it back); the latent branch in writing and in both prompt
  variants; D and U by group across the cards.
- **Checked:** a layer recomputed in NumPy in FP64 from the engine's own inputs (`FNX_LAT_DUMP`) matched with cosine
  0.99992 and a relative error of 1.3 % (FP16 rows and sums); 53-79 % of the draft layer's drafts are accepted in fresh
  writing and 95 % in edits, so the model's outputs are what Flash-Next's draft layer expects. With `FNX_LATENT` unset
  these sources gave our UD-Q4_K_XL production engine's first-token logits in 18 of 18 requests (KL 0, round `izdaja1`)
  and its speeds.
- **Slovenian is weaker than Flash-Next's.** Of 30 short questions in Slovenian (other prompts than the benchmark's),
  14-18 were answered in another language (Croatian or Serbian, or with Chinese words in between), against 5 for
  Flash-Next IQ3_S. The same with the prompt path in FP32 (`FNX_LAT_F32=1`), so it is the model, not the port. Longer
  Slovenian answers mix in words that are not Slovenian. We have not checked its English apart.
- **Pictures** work with ISTA-DASLab's projector: our picture question in Slovenian was answered right.
- **RAM:** 96 GB. It keeps 28 GiB of experts and the 26.8 GiB lookup table in RAM, about 80 GiB once loaded (10-13 GiB
  were available on ours).
- **Its authors' numbers** (Edge0's model card, not ours), Edge0-145B against Qwen3.8-Flash-Next: DeepSWE 1.1 64.6 / 58.7,
  Terminal-Bench 2.1 86.5 / 83.3, HLE 35.4 / 35.9, GPQA Diamond 90.4 / 91.7, IFBench 80.6 / 81.3, AA-LCR 79.3 / 79.1.

## Requirements

- **Linux, x86-64.**
- **Two Tesla P100 16 GB (PCIe).** The engine is built for exactly two cards of compute capability 6.0 and refuses
  others. It never uses peer-to-peer access or NVLink: the cards exchange their sums through pinned host memory.
- **PCIe Gen3 x16 for each card.** Experts that are not in VRAM are copied from RAM while it runs (see "Other machines").
- **96 GB of RAM**, or **64 GB** with the 64 GB mode (IQ3_S only, see "64 GB of RAM"). The default profile keeps a
  pinned RAM copy of every expert (about 47 GiB) and the model's 26.8 GiB lookup table (the second shard) in RAM. Once
  loaded it holds 82-85 GiB of the 91.8 GiB Linux sees on our server; during the load it needs 8.4 GiB more, freed at
  the end. UD-Q4_K_XL and Edge0-145B need 96 GB too (see "Unsloth's UD-Q4_K_XL" and "Edge0-145B").
- **Disk:** 83.6 GB for the IQ3_S shards (UD-Q4_K_XL: 111.3 GB; Edge0-145B: 57.3 GB plus the IQ3_S's 28.8 GB second shard), 0.9 GB for the picture projector, 0.8 GB for the MTP
  draft layer (another 6 GB while you prepare it). The IQ3_S load reads 48 GB from disk: about 100 s of the 5 minutes
  on ours (0.5 GB/s).
- **CUDA 12.x** (CUDA 13 dropped sm_60) with a host compiler it accepts (CUDA 12.4: gcc 13 at most), CMake 3.22 or newer,
  git, and a driver that supports both CUDA 12 and Pascal.
- **Python 3.10 or newer** for Strata's server and tools.
- **`iommu=pt`** (or the IOMMU off) on Intel boards with VT-d: with translation on, our host-to-card copies ran at
  7.6-8 GB/s instead of about 12.

## Install

Three commands, after the requirements above. `/data/flash-next` is any folder with about 95 GB free; everything but the
engine goes there.

```bash
git clone https://github.com/Megla123/foghorn.git && cd foghorn
bash build.sh                          # the engine: build/split/strata-split-2, about 20 minutes
bash setup.sh /data/flash-next         # Strata v0.2.9, the model (85 GB download), tokenizer, MTP layer, pictures, run config
bash run.sh /data/flash-next           # the server on http://127.0.0.1:8080/v1, loaded after about 5 minutes
```

- `bash setup.sh /data/flash-next --ram 64` writes the 64 GB mode's profile; `bash setup.sh /data/flash-next-q4 --model q4_k_xl`
  downloads and sets up Unsloth's UD-Q4_K_XL instead of the IQ3_S file (112 GB), `bash setup.sh /data/edge0 --model edge0`
  Edge0-145B (87 GB, of which the IQ3_S's second shard is 28.8 GB; `--models` the IQ3_S's folder shares it). One `DATA` folder holds one profile, so
  give each model its own folder, or `--models` the same model folder to share the downloads. `--coder` (with either
  file) writes a Coder profile (see "Coder").
- `bash setup.sh /data/flash-next --check` only reports what is there and what is still to do.
- `setup.sh` skips every step whose result is already there, so after an interruption you run it again; downloads
  resume. `--models DIR` uses model files you already have (they are checked against SHA-256), `--no-pictures` leaves
  out the picture encoder and its 0.9 GB projector, `--route-prior 0.15` writes our production setting (lossy, see
  "Exact and lossy"). It finds the two P100 with `nvidia-smi` and names them in the run config.
- `bash run.sh /data/flash-next --port 8080 --host 127.0.0.1` checks the engine's libraries, free RAM and the cards
  before it starts the server. Stop it with Ctrl+C and check with `nvidia-smi` that the cards are free.
- With the server running, `python3 bench/bench.py --strata-dir /data/flash-next/strata-v0.2.9 --wait` measures it
  (see "Benchmark and compare").

The sections below say what each script does and how to do the same by hand. They use these names:

```bash
P100=$HOME/foghorn                     # this repository
DATA=/data/flash-next                  # the folder setup.sh fills
STRATA=$DATA/strata-v0.2.9             # Strata's server and tools, at v0.2.9
MODELS=$DATA/models                    # the downloaded model files
PREP=$DATA/prep                        # the files made from them (tokenizer, MTP draft layer)
```

## Build

```bash
cd "$P100" && bash build.sh            # nvcc from PATH; or: CUDA=/usr/local/cuda-12.4 JOBS=8 bash build.sh
cd "$P100" && PF_JOB=64 bash build.sh  # the default until 17 Nov 2026: prompt-path jobs of 64 rows (the same bits, long prompts read slower)
```

`build.sh`:

1. fetches llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the commit Strata v0.2.9 and v0.2.10 both pin, into
   `third_party/llama.cpp` (only its ggml-cuda headers are used; ggml itself is not built);
2. regenerates `engine/src/iq3s_x.cuh` from that checkout and stops if it differs from the shipped one;
3. configures and builds `build/split/strata-split-2`, with production's settings: Release, `sm_60`, 2 cards,
   `-DSTRATA_SPLIT_PF_JOB=96`, `--use_fast_math --expt-relaxed-constexpr`, and `engine/p100_compat.h` force-included in
   every CUDA source. Prompt-path jobs of 96 rows are our production for UD-Q4_K_XL since 16 Nov 2026 and for IQ3_S and
   Edge0-145B since 17 Nov: against 64 rows the same bits, and a 36k read 32.7 % (with kernel mode 6), 5.3 % and 3.2 %
   faster. `PF_JOB=64` sets `-DSTRATA_SPLIT_PF_JOB=64`, the earlier build (the CMake option's own default). Either
   binary runs every model;
4. refuses a binary that can call a CUDA peer-access function, and warns if `ldd` cannot find the CUDA libraries.

It stops early, with a message, if the toolkit is not CUDA 12 or the host compiler is newer than that toolkit accepts
(set `HOSTCXX=g++-13`). A full build took 17.5 minutes on our two Xeon Bronze 3106 (`kernels_p2.cu` alone most of it).
Each compile job takes 1-2 GB of RAM. Do not build on the machine while the engine is loaded: it pins most of the RAM,
and a build beside it once hung our server.

The rebuilt binary will not be byte-identical to ours, because the source path is compiled in (the default location of
`engine/data/`). [docs/PROVENANCE.md](PROVENANCE.md) lists the SHA-256 of every source file, and the copies here
match them: our Edge0 production engine's sources (`e56c3f60`) with 18 Nov's switch in four files, which we compiled on
17 Nov as `a7717235` but have not run in a round or in the benchmark (see "Known limitations").

## Prepare the model files (what setup.sh does)

Strata v0.2.9 is the server, and its tools prepare the files. `setup.sh` clones it into `$STRATA`; by hand:

```bash
git clone https://github.com/Niko1221/Strata.git "$STRATA"
cd "$STRATA" && git checkout v0.2.9          # commit c62d5e3a15b40ea51c96aae9e8d36928965f53ab
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export STRATA_GGUF_PY=$P100/third_party/llama.cpp/gguf-py     # llama.cpp's gguf-py, fetched by build.sh
```

The commands in this section are Strata's tools as their own help describes them in v0.2.9. Our files were made in
October with an older Strata; we have not run these commands from v0.2.9 ourselves. Run them from `$STRATA`.

**The model.** From [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (the one Strata pins), into `$MODELS`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 54,817,524,224 | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 28,800,138,432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (pictures only) | 907,543,008 | `b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0` |

**Or UD-Q4_K_XL** (`setup.sh --model q4_k_xl`), from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
folder `UD-Q4_K_XL`, revision `084c631ac221016019d6e0a863afe92d90edd3b0`, with the same projector from ISTA-DASLab:

| File | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` | 10,946,618 | `acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf` | 49,720,539,552 | `45792ae869072d8229384c51c7c73dd959e191505252b41dd71852d9746f5ce4` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf` | 32,803,030,176 | `8c76e79d444b16a663d9c8db54e585ed3b1da13eebd2619c37b0c355da803c82` |
| `Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf` | 28,800,138,432 | `5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36` |

The fourth part is the same lookup table as IQ3_S's second shard: the two files differ only in two header bytes (the
part's number and count). The tokenizer folder made from the first part is byte-identical to the one made from the IQ3_S
file.

`setup.sh` downloads them with `curl` (resumable; set `HF_TOKEN` if Hugging Face asks for one) and checks them against
the tables. By hand, for example with the Hugging Face CLI, one file at a time, then `sha256sum`:

```bash
hf download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 --local-dir "$MODELS"
```

**The tokenizer folder** (vocabulary, merges and chat template, taken from the GGUF's metadata):

```bash
.venv/bin/python tools/strata_tokenizer.py --gguf "$MODELS"/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf --out "$PREP"
# writes $PREP/tokenizer/
```

**The MTP draft layer.** The GGUF has no MTP head. Strata's tools read its 31 tensors (5.2 GB) from Qwen's BF16
checkpoint with HTTP range requests (at a pinned revision, checked against SHA-256), pack them and write the engine's
runtime files:

```bash
.venv/bin/python tools/mtp_fetch.py inventory --out "$PREP"/mtp-bf16
.venv/bin/python tools/mtp_fetch.py fetch --out "$PREP"/mtp-bf16
.venv/bin/python tools/mtp_fetch.py verify --out "$PREP"/mtp-bf16
.venv/bin/python tools/mtp_pack.py --src "$PREP"/mtp-bf16 --experts q2_0 --out "$PREP"/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf "$PREP"/mtp-q2_0.gguf --out "$PREP"/mtp-rt
# $PREP/mtp-rt holds dense.bin, dense.txt and experts.bin (Strata's 2-bit draft experts)
.venv/bin/python ../scripts/mtp_q4.py "$PREP"/mtp-bf16 "$PREP"/mtp-rt/experts-q4_0.bin
# the draft experts as Q4_0 from the BF16 tensors (1.4 GB), for FNX_MTP_EXPERTS=q4_0 (the UD-Q4_K_XL profiles);
# $PREP/mtp-bf16 and mtp-q2_0.gguf can go afterwards
```

The SHA-256 of our MTP and tokenizer files are in [docs/HISTORY.md](HISTORY.md#our-prepared-files), to compare.

**The draft vocabulary.** The draft layer may only propose tokens from a subset. `engine/data/draft_vocab_sl.bin` is
the one we run and measured with: sFNX's English and code subset (`draft_vocab_en.bin`, 40,525 tokens) plus 8,621
Slovenian word pieces, 49,146 in all. For English only, `fill_profile.py --draft-vocab "$P100"/engine/data/draft_vocab_en.bin`
takes sFNX's subset; for another language build one with Strata's `tools/draft_vocab.py --corpus` (see its help). The
choice changes speed, never the answers: every drafted token is checked by the full model.

**The picture encoder** (optional). Strata's `strata-vision`, built from the v0.2.9 checkout against the same llama.cpp:

```bash
CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 cmake -S tools/vision -B build-vision -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$P100"/third_party/llama.cpp -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13
cmake --build build-vision --target strata-vision -j 6        # $STRATA/build-vision/bin/strata-vision
```

These are the settings our encoder was built with. The profile runs it on the CPU (`"gpu": false`), so the cards' memory
stays with the model.

## Run (what run.sh does)

`setup.sh` writes the run config to `$DATA/profile.json` with `profile/fill_profile.py` and keeps it on later runs, so
you can edit it. By hand, the script checks that every file it names exists:

```bash
cd "$P100"
python3 profile/fill_profile.py --models "$MODELS" --mtp "$PREP"/mtp-rt --tokenizer "$PREP"/tokenizer \
    --vision-exe "$STRATA"/build-vision/bin/strata-vision --logs "$DATA"/logs --out "$DATA"/profile.json
# without --vision-exe: no pictures; --route-prior 0.15: our production setting (lossy); --gpu 2,3 if the P100 are
# not cards 0 and 1; --exe if you built elsewhere than build/; --draft-vocab for another draft vocabulary;
# --ram 64: the 64 GB mode; --model q4_k_xl: UD-Q4_K_XL (profile/flash-next-q4_k_xl-512k.template.json; its 256k one
# with --template profile/flash-next-q4_k_xl-256k.template.json); --model edge0: Edge0-145B (its four files in
# "$MODELS"/edge0-145b, the IQ3_S's second shard in "$MODELS"); --coder: only the Coder's experts (lossy)
```

`run.sh` then starts Strata's server with it from the v0.2.9 checkout, which by hand is:

```bash
cd "$STRATA"
.venv/bin/python serve/server.py --engine strata --config "$DATA"/profile.json --port 8080 --host 127.0.0.1
```

`LD_LIBRARY_PATH` must lead to the CUDA 12 libraries the engine was built against (ours is unset: Ubuntu's package puts
them in `/usr/lib/x86_64-linux-gnu`). The engine logs to the config's `"log"`. `GET /health` answers `"loaded": true`
after about 5 minutes; then the API is at `http://127.0.0.1:8080/v1`.

**What the profile sets** (every switch in [docs/SWITCHES.md](SWITCHES.md)):

| Key or argument | Value | Meaning |
|---|---|---|
| `"gpu"` | `"0,1"` | the two P100 as `nvidia-smi` numbers them (`setup.sh` finds them) |
| `--native` | shard 1 | the model; shard 2 is found beside it |
| `--mtp`, `--draft-vocab` | the MTP folder, `draft_vocab_sl.bin` | the draft layer and the tokens it may propose |
| `--max-context` | 524288 | with `FNX_ROPE_SCALE=2 FNX_YARN_ORIG_CTX=262144` (YaRN x2); longer requests get HTTP 400 |
| `--spec`, `--lookup` | 4, 15 | MTP draft depth; prompt-lookup drafts of up to 15 tokens (edits) |
| `--conversation-cache-mib` | 4000 | parked conversations in RAM, so returning to one does not read it again |
| `--batch` | 1 | one request at a time |
| `--vision` | | pictures through the encoder |

The `env` block holds 38 `FNX_*` switches. The ones that matter most for speed: `FNX_EXPERT_RAM_GIB=29.0` (how much of
the experts stays in RAM only), `FNX_P100_DMA=1` with `FNX_DMA_SPLIT=2` and `FNX_PREFETCH=1` (copying missing experts),
`FNX_PF_JOBV=7`, `FNX_PF_STAGE_ALL=1`, `FNX_PF_SP=1` and `FNX_PF_F16ACC=1` (reading), `FNX_P100_ATTN_DEC=1` (attention
while writing). The UD-Q4_K_XL profiles differ in ten of them ([docs/SWITCHES.md](SWITCHES.md#the-ud-q4_k_xl-profiles)).
To stay within the trained 262,144 positions, edit the written JSON: `--max-context 262144`, and remove `FNX_ROPE_SCALE`
and `FNX_YARN_ORIG_CTX`.

## Benchmark and compare

With the server running:

```bash
cd "$P100"
python3 bench/bench.py --strata-dir "$STRATA" --url http://127.0.0.1:8080 --wait --out results.json
python3 bench/bench.py --strata-dir "$STRATA" --tests fresh,edit,long,medium,image --out results.json   # with a picture
```

It runs our tests (fresh writing, two edits, a long read, medium reads, optionally a picture), 10-12 minutes on our
machine, and prints your numbers beside our runs of the same benchmark ("Our numbers"): IQ3_S's or
UD-Q4_K_XL's by the model the server reports, the 64 GB mode's with `--ours iq3_s_64gb`. Some prompts are in Slovenian,
as in our runs. The texts are read from the Strata v0.2.9 checkout, so every run reads the same ones. Before you compare:

- **Start the engine fresh and run the benchmark once.** The expert cache learns what it reads, so a second run on the same
  engine is faster.
- Ours are one run each. Our rounds differed by up to 2 % between runs of the same build (long read 903-918), more right
  after a change of settings. All of ours ran on the 96-row build (`build.sh`'s default): IQ3_S's and Edge0-145B's on
  17 Nov with the profiles as they are now, UD-Q4_K_XL's on 16 Nov (`--ours q4_k_xl` is the 512k one, `q4_k_xl_256k` the
  256k one; its long read was not cold). UD-Q4_K_XL's are from before 18 Nov's switch and settings, and the benchmark
  has not been run with those: in our rounds fresh writing was 3.3 % faster with them, edits equal (py) or 1.6 % faster
  (md), writing after a 36k read 3.9-6.5 % faster, reading equal once the lookup table is in the page cache, and writing
  at a depth of 58.5k, which the benchmark does not test, 1.7-4.3 % slower (see "Unsloth's UD-Q4_K_XL"). The IQ3_S
  picture's first answer (5.1 s) came in that morning's first run with pictures on this file; in our rounds it takes
  2.6-2.7 s. On a 64-row build (`PF_JOB=64`) long prompts read slower, UD-Q4_K_XL's most (see "Unsloth's UD-Q4_K_XL").
- Keep `FNX_PROF=0` and nothing else on the cards. Our cards ran at 1,328 MHz under load; the clock follows recent load,
  so short tests can differ by 4-5 %.

## Exact and lossy

Every change in this port was checked against the build before it: at temperature 0 the first-token logits gave KL 0
(we dump them with `FNX_DUMP_FIRST_LOGITS`) and the answers were the same to the character. "Exact" in this repository
means that: the same bits as our previous build.

- **`FNX_ROUTE_PRIOR`** is the one lossy switch. Above 0, past each token's top 2 experts the router prefers experts that
  are already in VRAM. At 0.15 the first-token KL was 0.0095 and a wider check (8 edits, 10 programs, 10 calculations,
  5 JSON tasks, 3 hidden-data tasks, 6 Slovenian answers) came out the same as at 0. The same prompt at temperature 0 no
  longer always gives the same text, because the routing depends on what the cache holds. At 0.3 a long Markdown edit
  made only 2 of 6 replacements, so we stopped at 0.15. The template defaults to 0.
- **`FNX_EXPERT_MASK`** (`--coder`) makes it a different model on purpose, with half of the experts (see "Coder").
  Unset, it changes nothing.
- **Against other engines** the answers are close, not identical: the P100 kernels compute in FP16 where Strata's use
  other arithmetic (`FNX_PF_F16ACC`, the HFMA2 expert kernels). Against our earlier engine on the same cards (Strata
  0.2.9's layer-split engine with our P100 kernels) the first token matched in 21 of 21 requests. The YaRN setting
  changes the position encoding at every length (see "Run" to turn it off).
- The models are quantizations, not the full model: ISTA-DASLab's IQ3_S (Strata's `docs/QUANTS.md` has its numbers) and
  Unsloth's UD-Q4_K_XL. We have not compared their answers' quality.

## Other machines

We have one machine. The points below are what the design implies, with the measurements on our machine they rest on;
none of them was measured on another machine.

- **One CPU instead of two:** should work the same. The engine needs no peer access; it pins each card's host thread to
  the CPU the card hangs from (from `/sys`), which on one CPU is simply that CPU. On our server each card reads both
  CPUs' memory at the same 11.9 GB/s.
- **x8/x8 slots (or a x16 slot wired x8), PCIe Gen3:** about 6 GB/s per card instead of 12. About 60 % of the experts
  live only in RAM, so traffic over PCIe is part of the work: missing experts are copied in for each decode window, and a
  long read stages every missing expert of each layer (up to 568 MiB a layer) for each chunk. Expect reading to slow
  down clearly and writing to slow down less (fewer copies per token). Gen2 x16 is like Gen3 x8.
- **More RAM** does not make IQ3_S faster; the cards' PCIe links are the limit. With UD-Q4_K_XL, about 128 GB would keep
  its lookup table in RAM, so the first long read after a load would run like the later ones; writing would stay the same. With **64 GB** use the 64 GB mode (see
  "64 GB of RAM"); less than that does not load.
- **Power limit:** at 250 W instead of our 185 W, long reads were at most 1-1.5 % faster on our cards.
- **P100 SXM2:** the same chip and should run; NVLink is not used. Not tried.
- **Volta and newer cards** (sm_70 and up): the P100 kernels are used only below sm_70, so a build for those cards is
  Strata v0.2.10's split engine with a profile tuned for 16 GB P100s, older than current Strata. Use Strata there.
- **Other Pascal cards** (P40, GTX 10x0, sm_61): not supported. The engine refuses an architecture it was not built for,
  and its kernels are built on FP16 arithmetic, which those chips run at a small fraction of the P100's rate.
- **Three or four cards:** Strata's code has them; this port has only run on two.
- **An older processor** (no AVX2, e.g. Sandy Bridge or Ivy Bridge Xeons): the engine and Strata's server need none, but
  Strata's picture encoder is built for an AVX2 baseline by default; on Sandy Bridge-EP NVIDIA's driver also runs the
  cards at PCIe Gen2 unless told otherwise. `tune/check.py` checks both (next section).

## Tuning for your machine (tune/)

Two tools, described in [tune/README.md](../tune/README.md):

```bash
python3 tune/check.py --data /data/flash-next --out report.md   # about a minute, no model loaded
python3 tune/tune.py --data /data/flash-next --screen           # 2-3 hours, one load per candidate
```

- **`check.py`** reads the cards, each card's PCIe path up to the CPU, the processor, RAM, NUMA, the IOMMU and the disk
  with the model; measures host-to-card copies from pinned memory and the disk; picks the profile for your RAM and names
  what limits your machine against ours, with advice. Each piece of advice says whether it rests on a measurement on our
  machine or is only what the design implies. The report is written to paste into a GitHub issue. Run it with the cards
  free; the copy test skips a card that is in use.
- **`tune.py`** measures the switches that give the same bits (`FNX_PREFETCH_WANT`, `FNX_PREFETCH_MIN_T`,
  `FNX_DMA_SPLIT`, `FNX_EXPERT_RAM_GIB`, `FNX_EXPERT_DYN_GIB`, `FNX_PLE_RAM`, `--conversation-cache-mib`; the staging
  buffers on request) on your machine, each candidate on a freshly loaded engine with a shortened benchmark, and keeps a
  change only when it wins by more than the noise it measured between two baseline runs and a final run confirms it. It
  checks the RAM before each load, stops when the machine keeps swapping, compares every arm's first-token logits with
  your starting profile's, and writes `DATA/profile.tuned.json`. `FNX_ROUTE_PRIOR` is tried only with `--allow-lossy`;
  `--coder`'s expert mask is never changed.
- **On our machine** (13 Nov 2026, a test window, the IQ3_S 512k profile at `FNX_ROUTE_PRIOR=0`, engine `c213636c`):
  `check.py` measured 12.3 GB/s to each card, the same from the other CPU's memory and with both cards at once, Gen3 x16
  while copying, and found no bottleneck. `tune.py --screen --only prefetch,dma` took 7 loads (about 70 minutes): the
  two baselines gave fresh 66.7-67.0, edits 111.0 / 115.9-116.1 and a 38,862-token read at 961-962 tok/s (noise 0.5 % or
  less, writing after the long read 2.1 %); 2, 3 or 4 guessed next-layer copies a row were within 1 % of the profile's
  6 (9: −5 %), prefetch off, `FNX_PREFETCH_MIN_T=2` and `FNX_DMA_SPLIT=1` were in the noise, `FNX_DMA_SPLIT=0` cost 4 %
  in edits. It kept the profile as it is, and every arm's first-token logits were the baseline's bit for bit (19 of 19).
  It has not run on another machine yet; results from other machines are welcome as issues.

## Known limitations

- **Only these files.** The port's P100 expert kernels cover the expert types IQ3_S, IQ3_XXS, IQ2_S, IQ4_XS, IQ4_NL,
  Q4_0, IQ2_XXS and IQ1_M, since 13 Nov Q4_K, Q5_K, Q5_1 and Q8_0 (UD-Q4_K_XL's), and since 15 Nov NVFP4 for Edge0's
  latent experts (in Edge0's own file layout only, through `FNX_LATENT`; Edge0's Q3_K build is not supported). Files with other expert
  types (Q6_K, Q3_K, Q2_K, ...) get none of them: writing falls back to Strata's original kernels, and reading a prompt
  stops with `p100 moe: expert type N not handled` (with `FNX_PF_MOE_Q8=1` it reads through the writing kernels,
  slowly). On the first release, which had no Q4_K kernels, a tester ran UD-Q4_K_XL beside current Strata on the same
  machine and the port was slower on every test; on IQ3_S the tester got the same results as ours. Of ISTA's other
  files, IQ2_XS ran on an earlier build of this port (7-8 Nov, about 65 tok/s fresh) but not on this one; Q2_0 and
  IQ3_XXS have not been run. Unsloth's other files have not been run.
- Two P100 16 GB and Linux only.
- One request at a time is what we tuned for (`--batch 1`).
- The engine refuses `logprobs`.
- Loading takes about 5 minutes; the first answers after a load are slower while the expert cache settles.
- With a long context and the 4 GB conversation cache, RAM is tight: our server has 6-9 GiB free once loaded. A parked
  conversation takes about 21 KB a token; bigger ones are read again instead of restored.
- The sources' comments refer to our test tools (`mikro/...`), measurement rounds (`splitNN`), ideas by number and our own
  apps; those are not part of this release. A few `FNX_PROF` log lines are in Slovenian. We kept the files exactly as
  they run.
- `history/patches/` holds the last patch scripts we applied, with our build machine's paths. They are a record: their
  changes are already in `engine/`.
- **Edge0-145B:** its Slovenian is clearly weaker than Flash-Next's (see "Edge0-145B"); it runs at 524,288 positions
  only with D and U split across the cards (`FNX_LAT_SPLIT=1`). Its sources were built in our development build
  directory, not with this folder's `build.sh`; `setup.sh --model edge0 --check`, `fill_profile.py`, `run.sh` and
  `bench/bench.py` ran on our server against the real files (linked, not downloaded). Its coding and reasoning numbers
  are its authors', not ours.
- **Coder:** worse than the full model outside code, and in our tests it ignored an instruction given in Slovenian and
  answered a Slovenian question in English (see "Coder"). It has not been run with the 64 GB mode or through
  `bench/bench.py`.
- This folder's `build.sh` built the engine on our server (17.5 minutes with `JOBS=12`, llama.cpp from our Strata v0.2.9
  build tree at the pinned commit), and that engine gave our production engines' bits ("Our numbers"); the profile
  writer and the benchmark ran there as they are. The Coder change came after that: its sources (`maska_patch.py`
  applied to the ones `build.sh` built) were built in the build directory `c213636c` came from, not yet with `build.sh`. `setup.sh` and `run.sh` were run against stand-ins (a fake Strata
  checkout and tools, small model files), not yet against the real downloads.
- **The 96-row build** (`build.sh`'s default since 17 Nov): on 16 Nov `PF_JOB=96 JOBS=12 bash build.sh` built this folder's
  engine of then on our server in 17 minutes (llama.cpp from our Strata v0.2.9 build tree at the pinned commit). The file
  (`a1e9ac07`) is not the same as `e56c3f60`, which came from another build directory, but on the 512k profile it gave
  `e56c3f60`'s first-token logits in 29 of 29 requests of the benchmark and its speeds (fresh 50.4, edits 62.7 / 64.0, a
  cold read 794, medium reads 536 / 721 / 770 / 721 / 876 / 780 / 840, the picture 3.1 s; 44.3). It was run with UD-Q4_K_XL
  only; IQ3_S's and Edge0-145B's numbers are `e56c3f60`'s. `build.sh` has not been run again since 96 became its default
  (the same option, now without `PF_JOB=96`).
- **18 Nov's switch** (`FNX_MTP_NORM_STREAMS`): `build.sh` has not been run on these sources. The same files were
  compiled on our server on 17 Nov with 96-row jobs (`a7717235`), not with this folder's `build.sh`, and that binary
  has not run in a round or in the benchmark. What ran, and is our UD-Q4_K_XL production engine, is `f28228d1`: that
  engine's sources of before (`6ccc87af`'s) with the same patch, 96 rows. Unset, the code makes the calls it made
  before; we did not run this folder's binary against `e56c3f60` to check it. The switch was not measured with IQ3_S
  or Edge0-145B, the benchmark has not been run with 18 Nov's UD-Q4_K_XL settings, and our rounds with them ran on the
  512k profile only.

## Credits and license

- **Strata** by Niko1221 and the Strata contributors (MIT): the engine this is built from, its server and tools.
- **sFNX** by John Fink (MIT, https://github.com/johnfink8/sfnx): the split-every-layer engine Strata ported.
- **ggml / llama.cpp** by the ggml authors (MIT): the GPU block formats and dot products; `engine/src/iq3s_x.cuh` is made
  from its `vecdotq.cuh`.
- **ISTA-DASLab** for the GSQ-RCO model files and the Coder release's list of kept experts (Apache-2.0, see
  [third_party/ista-daslab/](../third_party/ista-daslab/)), **Unsloth** for the UD-Q4_K_XL file, **Qwen** for Qwen3.8-Flash-Next, **Edge0** for Edge0-145B (the
  Edge0 Community Model License 1.0; its files are not included here).
- Ideas taken from others: the 16-bit `dp4a` form (`fnx_dp4a_xmad` in `p100_compat.h`) from KaneQuah/Kmic-68's llama.cpp
  fork (commit `99eb48176d`); the copy-engine thread for missing experts (`FNX_P100_DMA`) from DeepInfer; the register
  constraint in `fnx_dp4a_vmad` from Strata PR #4274; the expert pool floor from Strata 0.2.12 (commit `20885f7a`);
  `FNX_PF_TAIL_MIB=4096` for UD-Q4_K_XL from a tester's recipe for that file; the
  draft layer's source normalised per stream (`FNX_MTP_NORM_STREAMS`) from the notes of the Colibri engine
  (JustVugg/colibri, Apache-2.0: its `docs/qwen38.md`, `Q38_MTP_WIRING=b`), the idea only, none of its code.

MIT, the same license as Strata: see [LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md). The licenses of sFNX and ggml, and of
the Coder's expert list (Apache-2.0), are in [third_party/](../third_party/).
