# How the port came about

Strata's all-cards split engine (from sFNX) is written for Ampere and newer NVIDIA cards and for AMD RDNA. It does not
build for sm_60: the Tesla P100 has no `dp4a`, no `__nanosleep`, no tensor cores, and its FP16 is fast only through
paired `HFMA2` instructions. Before this engine our two P100 ran Strata 0.2.9's layer-split engine with our own P100
kernels (about 47-55 tok/s fresh, 620-650 tok/s long reads), where one card waits while the other works. This engine gives
each card half of every layer, so both read their weights at once.

The port started on 5 November 2026 from Strata v0.2.10 and went into our production on 6 November. Each step below
was measured in an A/B round (`splitNN`) against the build before and, unless marked lossy, checked to give the same bits
(first-token logits KL 0, identical answers at temperature 0). Dates are 2026, times CET; the hashes are the first
digits of each binary's SHA-256.

| Date | Build | Change | Effect |
|---|---|---|---|
| 5-6 Nov | `f0f40dca` | The port: `p100_compat.h` (dp4a through `vmad`, no `nanosleep`); P100 expert kernels for prompts (`p100_moe.cu`, `p100_iq_gemm.cuh`: FP16 tiles) and for writing (`p100_moe_rows.cuh`: a row kernel); P100 attention for prompts (`p100_attn.cuh`); the lookup table locked in RAM; the arena (K/V pages borrow expert slots, up to 524,544 positions); copies of missing experts beside the row kernels; reads of up to 144 tokens through windows; checkpoints every 16,384 rows; YaRN to 512k | in production from 6 Nov 05:30 |
| 6 Nov | `a7a7f0a8` | `logit_bias` (our chat app sends a list that bans foreign scripts) | |
| 6 Nov | `93b71b1f` | next-layer prefetch (`FNX_PREFETCH`), `FNX_ROUTE_PRIOR` (lossy) | edit py 91.0 → 106.9, md 98.5 → 109.8, fresh 63.4 → 65.1 (with λ 0.15) |
| 6 Nov | `a2b6f771` | GDN recurrence kernel with a warp per part (the kernel 2.0-3.4x faster); rows per warp in the dense int8 products by window size | fresh 65.3 → 70.9 (+8.5 %), edits +4.3 % (λ 0.15) |
| 7 Nov | `a381cc7c`, `333ecfe5` | control vectors (`FNX_CVEC`); IQ2_XXS and IQ1_M kernels for ISTA's IQ2_XS file | IQ2_XS: fresh 65.4, edit py 105.5, md 108.5 |
| 9 Nov | `6f47ed0f` | each copied expert's blocks wait only for its own copy (`FNX_DMA_SPLIT=2`) | edits +2-3 % |
| 10 Nov | `e35054e3` | the prompt expert kernel tuned for small jobs (`FNX_PF_JOBV=3`) | long read 714 → 839 (+17.6 %) |
| 10 Nov | `8be80815` | `hc` reads without needless conversions, attention-selection scores in tiles, small launches folded together | long read 838 → 879, fresh +1.6 % |
| 11 Nov | `c16e3b6d` | attention selection: a lower bound first, then work on candidates only (`FNX_QSA_SEL`) | reads deep in the context (279k → 505k) +14.2 % |
| 11 Nov | `1882468e` | crash fix: the conversation cache picked a conversation or checkpoint by an index that parking the current one had just moved (a `segfault` in `load_state`, 9 times since 9 Nov) | |
| 11 Nov | `b1438fe1` | the prompt path stages every missing expert of a layer instead of reading some across PCIe; next-step loads ahead in the expert kernel; 64-row tiles for `down` (`FNX_PF_JOBV=6`, `FNX_PF_STAGE_ALL`, `FNX_PF_PREP`) | long read 835 → 906 (+8.5 %), medium reads +6-8 % |
| 12 Nov | `2ef67b0e` | the byte dot product in one of three exact forms (`vmad`, `prmt`+`xmad`, the compiler's), chosen per kernel and window size | edits +3.3 %, fresh +0.7 % |
| 12 Nov | `5ef3a6ae` | the prompt expert kernel with two weight buffers (`FNX_PF_JOBV=7`); also the code of two router and `hc` ideas that gave nothing (off) | long read +0.6 %; the first release |
| 12 Nov | `bf9c7f1f` | each card reads `hc` only for its own rows of a prompt pass and the cards trade the mixed rows (`FNX_PF_SP`) | long read 920 → 961 (+4.4 %), deep +4.8 %, medium reads from 36k +3.7-4.5 % |
| 13 Nov | **`c213636c`** | P100 kernels for UD-Q4_K_XL's expert types (Q4_K, Q5_K, Q5_1, Q8_0) and `FNX_SH_IN`; the 64 GB mode (`FNX_PLE_RAM`, `FNX_EXPERT_DYN_GIB`, written in this repository); two RAM placement ideas that gave nothing (`FNX_NUMA_EXPERTS`, `FNX_PLE_WARM`, off) | UD-Q4_K_XL: a 36k read 146 → 722-905 tok/s, fresh writing 52.6; IQ3_S unchanged; the first UD-Q4_K_XL release |
| 13 Nov | `c213636c` | UD-Q4_K_XL settings only: a 512k profile (`FNX_EXPERT_DYN_GIB=18`: the K/V pool borrows up to 7.6 GiB of expert slots a card), the 256k one with the same 18, and 2 guessed experts a row in the next-layer copies (`FNX_PREFETCH_WANT=2`, was 6) | UD-Q4_K_XL in the benchmark: fresh 41.4 → 50.3, edits 45.3 / 43.9 → 62.6 / 63.5, long read 726 → 765, the picture's first answer 48.4 → 4.2 s (512k); reads to 501k tokens at 737 tok/s |
| 13 Nov | **`40ef6d53`** | `FNX_EXPERT_MASK`: the router picks only from the experts ISTA-DASLab's Coder release keeps (256 of 512 a layer), on either file's own weights; unset, the same bits as `bf9c7f1f` (KL 0, 27 of 27) | lossy by design. IQ3_S at λ 0.15: fresh 68.8 → 75.2, edit py 117-119 → 130-131, long read 943 → 979; UD-Q4_K_XL at λ 0.15: fresh 55.3 → 60.9, edit py 77.5 → 111; **this release** (GUIDE.md, "Coder") |
| 13 Nov | `c213636c` | UD-Q4_K_XL settings only: the MTP draft layer's experts as Q4_0 from the BF16 tensors (`FNX_MTP_EXPERTS=q4_0`, `scripts/mtp_q4.py`) instead of Strata's 2-bit file, the same VRAM; the same answers token for token | fresh writing 46.99 -> 51.84 (+10.3 %) at `FNX_ROUTE_PRIOR=0`, edits equal; IQ3_S +1 % (kept on 2-bit) |
| 14 Nov | `1949d99c` | Edge0-145B (`FNX_LATENT`, `edge0_patch.py` to `edge0_patch4.py`): the latent experts, D/U and group labels mapped beside the skeleton GGUF; an NVFP4 type in the P100 kernels for writing and for prompts (64-value blocks, four UE4M3 scales, the E2M1 table; the scales divided by 16 so that FP16 sums cannot overflow, the experts' tail scales times 16); the latent branch in writing and in both prompt variants (D, the experts on row `t * 8 + group`, `z`, U into the ring's sum); D and U whole on each card. 512k did not fit (about 100 MiB of VRAM left), 256k did | Edge0 at 256k: fresh 44.9, edits 56.3 / 56.5, a 36k read 575; a layer recomputed in NumPy (FP64) from the engine's own inputs: cosine 0.99990 |
| 15 Nov | `de45a5b6` | Edge0 settings only: `FNX_EXPERT_RAM_GIB` 36 → 31.5. At 36 only 4,603 experts were on the cards (Flash-Next IQ3_S: about 10,000) and the rest of the VRAM went to K/V pages a 256k context never needs | fresh 44.9 → 50.7, edits 56 → 81-84; reading unchanged |
| 15 Nov | **`f6088fde`** | Edge0: D and U by group across the cards (`FNX_LAT_SPLIT`, `edge0_patch6.py`): 1.74 GiB a card less, two exchanges more a layer. The prompt path's FFN half runs in three passes over the cards when it is on, each pass ending at an exchange's wait: the prompt path queues the cards one after the other from one host thread, and a kernel's first launch can wait for the whole context (CUDA 12 loads modules lazily), so a wait queued before the other card's send hung the first try in the prompt warm-up | at 512k with `FNX_EXPERT_RAM_GIB=28` (9,041 experts on the cards): fresh 54.9, edits 94.5 / 98.4, a 36k read 657.8 (+14 %), 512k fits; with `FNX_LATENT` unset the same bits as `c213636c` (KL 0, 18 of 18); **this release** (GUIDE.md, "Edge0-145B") |
| 16 Nov | `6ccc87af` | UD-Q4_K_XL only, no source file changed: prompt-path jobs of 96 rows instead of 64 (a build option, `PF_JOB=96 bash build.sh`; the P100 job kernels work on two tiles of 48 rows, and a 64-row job left a third of them empty) and `FNX_PF_JOBV=6` in its profiles (7 came with the IQ3_S profile and had not been measured for Q4_K and Q5_1) | a 36k read in our chain 736.0 → 864.3 (96 rows) → 976.4 tok/s (mode 6; +32.7 %), medium reads +7-20 %, writing unchanged, the same bits (KL 0, 27 of 27). In the benchmark the cold long read 765 → 795 (it waits for the disk), the warm one 901 → 929, medium reads +5-9 %; this release's sources of then built with `PF_JOB=96` (`e56c3f60`) gave `6ccc87af`'s logits in 29 of 29 requests (GUIDE.md, "Unsloth's UD-Q4_K_XL") |
| 17 Nov | `e56c3f60`, `36409ddc` | Edge0-145B and IQ3_S, no source file changed: 96-row jobs for them too. `e56c3f60` is this release's sources of then (before 18 Nov's switch) at 96 rows, with `FNX_PF_JOBV=6` in Edge0's profile; `36409ddc` is our IQ3_S production engine's sources (`bf9c7f1f`) at 96 rows, with mode 7 kept (6 read at 1003.1 against 1005.0). `build.sh` builds 96 rows by default from here on (`PF_JOB=64` for the earlier build) | a 36k read in our chain: Edge0 668.1 → 686.1 (96 rows) → 689.3 tok/s (mode 6; +3.2 %), IQ3_S 954.8 → 1005.0 (+5.3 %); medium reads +2.5-3.4 % and +1-5 %; writing unchanged, the same bits (KL 0, 27 of 27 each). In the benchmark IQ3_S's cold long read 954 → 1,012 (the 64 GB mode 768 → 807), Edge0's 669 → 691 (GUIDE.md, "Our numbers", "Edge0-145B") |
| 18 Nov | `f28228d1` | UD-Q4_K_XL only. A switch, `FNX_MTP_NORM_STREAMS` (`mtp_norm_patch.py`; the idea from the Colibri engine's notes): the RMS norm of the MTP draft layer's source takes each of a row's four hyper-connection streams on its own instead of the four as one vector. Only the drafts change, which the model checks; unset, the engine makes the same calls as before. And settings: `FNX_EXPERT_RAM_GIB` 52.5 → 52.071 and `FNX_EXPERT_DYN_GIB` 18 → 18.430, so the K/V pool keeps 153 pages of its own instead of 305 and each card has 150 expert slots more; the pool borrows slots from 29.4k positions on, before from 58.5k. `f28228d1` is `6ccc87af`'s sources with the patch, 96 rows; `e56c3f60`'s sources with it (those in `engine/` now) were compiled on 17 Nov as `a7717235`, which has not run in a round | against `6ccc87af` with 52.5 and 18, old and new in the same test window (the 512k profile): fresh writing 51.75 → 53.45 tok/s (+3.3 %), edits equal (py) and +1.6 % (md), writing after a 36k read 43.1 → 45.9 and 45.7 → 47.5, reads of 36k to 501k tokens 1.3-3.2 % faster in one pair of loads (`n18g1`); writing at a depth of 58.5k −1.7 to −4.3 %; reading equal once the lookup table is in the page cache; the same first tokens (KL 0) in 81 of 81, 17 of 17 and 7 of 7 requests (GUIDE.md, "Unsloth's UD-Q4_K_XL") |

## Tried and dropped

Kept in the code where it was cheap, off by default (see [SWITCHES.md](SWITCHES.md)):

- `hc` on 80 blocks for 56 SMs (stuck at the grid barrier) or two blocks per SM (slower); `hc` with f16 dots (fresh −7 %).
- Overlapping copies with compute at every window size (no gain at 2 rows); FP16 weights on the fly when writing (not faster).
- The router with plans per 32 slots, ballot-numbered jobs, 2 or 4 rows per warp, priority streams, another launch order:
  the same bits, no gain in the engine (the router is not on the critical path).
- Dense products in half precision for wide windows (`FNX_DENSE_H`): +0.4 % at most.
- The `xmad` dot product everywhere (fresh −1.9 %: kernels that cross a register step lose occupancy) and in the IQ3_S
  writing kernels (no gain: those are memory-bound).
- Smaller row-kernel blocks, more lanes per row in phase 2, weights past L1, a routing memory for edits: none faster in the
  engine (edits waited for copies; the memory guessed only 27 % right).
- LRU or LFU instead of CLOCK for the expert slots (simulated: 5 % fewer copies), other pool splits, an expert profile
  learned from our own use (slower at the start), refilling the lent slots after a read.
- Huge pages for the lookup table (load 309 s), NUMA placement of RAM (both cards read either CPU's memory at 11.9 GB/s),
  the `performance` governor, C-states off, a higher power limit (+1-1.5 % long reads at 250 W).
- Strata's one-hop sum (`FNX_ONEHOP=2`): +1.4 %, but different last bits.
- For UD-Q4_K_XL: more next-layer copies (9 a row: edits −20 %), half-size staging buffers to save 2 GiB of RAM (reads
  −19-33 %), 384 MiB of VRAM reserve instead of 512 (nothing), shorter lookup drafts (nothing: the same copies per token),
  the GDN state snapshots shrunk for windows of 8 rows (the prompt path borrows that block as its scratch).
- For Edge0: D and U while writing on the P100 row kernel instead of the dense int8 product (`edge0_patch5.py`, not in this
  release): fresh writing 44.9 → 36.1. The row kernel gives each row one thread or a few, which reads D's and U's long rows
  (2,560 and 14,336 values) poorly; the dense product's loads are coalesced. The profile we read it from counted the
  experts' copies in the same phases as D and U.

## The patch scripts

`history/patches/` holds the scripts of the last steps (11-17 Nov) as they were run on our build machine. They edit the
development tree in place and name its paths, so they will not run elsewhere, and their changes are already in `engine/`.
`gen_iq3s_x.sh` there is the original of `scripts/gen_iq3s_x.sh`. The full change against Strata is
[upstream-v0.2.10.diff](upstream-v0.2.10.diff).

## Our prepared files

SHA-256 of the files our production uses, to compare with what Strata's tools give you (GUIDE.md, "Prepare the model files").
Ours were made on 21 Oct 2026 with an older Strata.

| File | SHA-256 |
|---|---|
| `mtp-rt/dense.bin` | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| `mtp-rt/dense.txt` | `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd` |
| `mtp-rt/experts.bin` | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `mtp-rt/experts-q4_0.bin` (13 Nov, `scripts/mtp_q4.py`) | `2775490f13a29eb36868b96b727d4b3c901e7de7d75e0952477d87851a7a2829` |
| `tokenizer/chat_template.jinja` | `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` |
| `tokenizer/merges.txt` | `4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332` |
| `tokenizer/tokenizer.json` | `c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410796a6f30bf` |
| `tokenizer/token_type.json` | `5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b` |
| `tokenizer/vocab.json` | `4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4` |
| `engine/data/draft_vocab_sl.bin` (shipped) | `3e3fa1a67d8dbe0989ff682758eb9467ee827174ab3471086b6794b81e50620b` |
