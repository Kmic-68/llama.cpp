# Patch scripts (a record, not to be run)

The scripts that made the last steps of the port (11-17 November 2026), as they were run on our build machine. Each
edits the development tree in place, checks that every edit matched as often as expected, and some save the original
first. They name that machine's paths (`/srv/build/...`) and our test tools (`mikro/...`), which are not part of this
release. Their changes are already in `engine/`; [docs/HISTORY.md](../../docs/HISTORY.md) says what each step did, and
[docs/upstream-v0.2.10.diff](../../docs/upstream-v0.2.10.diff) holds the whole change against Strata.

| Script | Switch it added | Step |
|---|---|---|
| `nvp_patch.py`, `nvw_patch.py` | `FNX_NVPROF_PASS`, `FNX_NVPROF_WIN` | `nvprof` capture of one prompt pass or a run of windows |
| `rrt_patch.py`, `topk_patch.py`, `prio_patch.py` | `FNX_RR_TIMES`, `FNX_RR_FAST`, `FNX_PRIO` | router timing, a faster top-10, stream priority (no gain) |
| `stage_patch.py`, `prep_patch.py`, `jobv4_patch.py`, `jobv5_patch.py`, `jobv6_patch.py` | `FNX_PF_STAGE_ALL`, `FNX_PF_PREP`, `FNX_PF_JOBV` 4-6 | build `b1438fe1` |
| `xmad_patch.py` | none (the dot-product form per kernel and window size is fixed by constants at its top) | build `2ef67b0e` |
| `moe_x_patch.py`, `gen_iq3s_x.sh` | `FNX_MOE_X`, `FNX_MOE_XD` | the xmad dot product in the IQ3_S writing kernels (no gain, off) |
| `rr_order_patch.py`, `hc_up2_patch.py`, `rr_rq4_patch.py` | `FNX_RR_ORDER`, `FNX_HC_UP2`, `FNX_RR_RQ4` | router and `hc` ideas (no gain, off) |
| `jobv7_patch.py` | `FNX_PF_JOBV=7` | build `5ef3a6ae`, the first release |
| `sp_patch.py`, `sp2_patch.py` | `FNX_PF_SP`, `FNX_PF_SP_MIN` | build `bf9c7f1f`: each card reads `hc` only for its own rows of a prompt pass |
| `q4x_patch.py` | `FNX_SH_IN` (its `FNX_PLE_DISK` was dropped for `FNX_PLE_RAM`) | P100 kernels for UD-Q4_K_XL's expert types (Q4_K, Q5_K, Q5_1, Q8_0) |
| `numa_patch.py` | `FNX_NUMA_EXPERTS` | experts' RAM copy on the card's CPU (no gain, off) |
| `warm_patch.py` | `FNX_PLE_WARM` | read the lookup table once after the load (slower, off) |
| `maska_patch.py` | `FNX_EXPERT_MASK` | build `40ef6d53`: route only to the experts ISTA-DASLab's Coder release keeps |
| `edge0_patch.py` | `FNX_LATENT`, `FNX_LATENT_PLE` | Edge0-145B: its latent experts, D/U and labels beside the skeleton GGUF, the NVFP4 type in the P100 kernels, the latent branch in writing and in the plain prompt variant |
| `edge0_patch2.py` | `FNX_LAT_F32`, `FNX_LAT_CHECK` | the prompt path's D and U with FP16 sums like its other products; a finite-value check |
| `edge0_patch3.py` | none | the latent branch in the overlapped prompt variant too (without it prompts of more than two chunks failed with "ring error 2") |
| `edge0_patch4.py` | `FNX_LAT_DUMP`, `FNX_LAT_DUMP_L` | one layer's inputs and U output to files, for the NumPy check; build `1949d99c` |
| `edge0_patch6.py` | `FNX_LAT_SPLIT` | build `f6088fde`: D and U by group across the cards, the prompt path's FFN half in three passes |
| `mtp_norm_patch.py` | `FNX_MTP_NORM_STREAMS` | build `f28228d1` (`c213636c`'s sources with it, 96-row jobs): the RMS norm of the MTP draft layer's source takes each of a row's four hyper-connection streams on its own instead of the four as one vector (the idea from the Colibri engine's notes) |

The 64 GB mode (`FNX_PLE_RAM`, `FNX_EXPERT_DYN_GIB`, `FNX_EXPERT_DYN_MIN`) was written in this repository, not by a script: our
Q4 build took it from here. All of the above up to `warm_patch.py` are the sources of build `c213636c`; `maska_patch.py`
applied to them gives build `40ef6d53`, and the five `edge0_patch*.py` applied to that, in the order above, give the sources of
build `f6088fde`. `mtp_norm_patch.py` applied to those gives the sources in `engine/` (compiled with 96-row jobs on 17 Nov as
`a7717235`, which has not run in a round); applied to `c213636c`'s, it gives those of build `f28228d1`. `edge0_patch5.py` (D
and U while writing on the P100 row kernel) was slower and was reverted; it is not here (docs/HISTORY.md, "Tried and dropped").
