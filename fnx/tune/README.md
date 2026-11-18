# Tuning for your machine

We have one machine (README, "The machine"), and the profiles in `profile/` are tuned on it. Two tools look at yours:

| Tool | Loads the model | Takes | What it gives |
|---|---|---|---|
| `tune/check.py` | no | about a minute | the hardware around the cards, measured copies and disk reads, the profile to use, the bottlenecks with advice, a report for a GitHub issue |
| `tune/tune.py` | yes, once per candidate | 2-3 hours (about 10 minutes a load) | the exact switches measured on your machine; writes `DATA/profile.tuned.json` |

Both need Python 3.10 and nothing beyond what `setup.sh` installs; `check.py` reads the driver's `libcuda.so.1` directly
(no CUDA toolkit, nothing compiled).

## check.py

```bash
python3 tune/check.py --data /data/flash-next --out report.md
```

Run it with the cards free: the copy test takes about 600 MiB of each card for a few seconds and is skipped on a card
that has more than 1 GiB in use (`--force-copy` runs it anyway). It reads and measures:

- **The cards**: two Tesla P100 16 GB (cc 6.0), the driver (the 580 branch is NVIDIA's last for Pascal), power limit,
  clocks, the NUMA node, the IOMMU group and BAR1 of each card (a P100 needs a 16 GiB BAR1: Above 4G Decoding).
- **The processor**: x86-64, AVX2, and the platform. The engine and Strata's server need no AVX2 (`build.sh` sets no
  `-march`), but Strata's picture encoder is built for an AVX2 baseline by default and stops on an older processor:
  there use `setup.sh --no-pictures` or build `strata-vision` with `-DSTRATA_PORTABLE=OFF`. On Sandy Bridge-EP (Xeon
  E5 v1, Core i7 39xx) NVIDIA's driver runs the cards at PCIe Gen2 unless the module option `NVreg_EnablePCIeGen3=1` is
  set; Ivy Bridge-EP (E5 v2) runs Gen3 on the same boards.
- **Each card's PCIe path**: the card's link and every bridge above it up to the CPU, as they run and at most. An idle
  P100 drops its link to Gen1, so the link is also read while the card copies.
- **Copies** from pinned host memory, as the engine copies missing experts: to each card in 64 MiB and 1 MiB pieces,
  back to the host, from the other CPU's memory, and to both cards at once (two cards behind one PCIe switch share its
  uplink, which a single card's test does not show).
- **RAM**, swap, NUMA nodes and their memory, the IOMMU (`/proc/cmdline`, `/sys/class/iommu`, the cards' IOMMU groups),
  transparent huge pages.
- **The disk with the model**: reads in a row (4 readers of 16 MiB, `O_DIRECT`; the load reads 48 GB) and random
  128 KiB reads (a token's row of the 26.8 GiB lookup table, which the 64 GB mode and the UD-Q4_K_XL profiles read from
  the disk). If other work waits for I/O meanwhile, it says so.

Then it picks the profile by the RAM (96 GB: the default IQ3_S profile; 64 GB: `--ram 64`; less: none) and lists what
limits your machine against ours. `--write-profile OUT` writes `DATA/profile.json` with its suggested switches.

Our machine, for comparison (`check.py` in a test window, 13 Nov 2026): 12.3 GB/s to each card (Gen3 x16, `iommu=pt`;
1 MiB copies 10.8, back to the host 13.1), the same from the other CPU's memory and with both cards at once; the disk
0.50-0.53 GB/s in a row and 1,350-1,700 random 128 KiB reads a second (a logical drive of an HPE Smart Array controller;
a run while another job wrote to the disk gave 34, which is why the report notes I/O pressure).

### What the rules rest on

Each piece of advice is marked **[measured]** when it rests on a measurement on our machine and **[unmeasured]** when it
is what the design implies. The second kind has not been tried on another machine:

| Finding | Advice | Basis |
|---|---|---|
| no AVX2 | pictures off, or the encoder built for this processor | **[unmeasured]** on such a processor; read from Strata v0.2.9's `tools/vision/CMakeLists.txt` |
| Sandy Bridge-EP | `NVreg_EnablePCIeGen3=1`, then check the link while copying | **[unmeasured]** here; NVIDIA's driver documentation |
| a virtual machine | run on the host, or `iommu=pt` there | **[unmeasured]** in a VM; the copy test shows what it costs |
| fewer than 8 CPU threads | a CPU with more | **[unmeasured]**: a spinning host thread per card, the server and 12 embedding threads; ours has 16 at 1.7 GHz and writes 66.6 tok/s |
| governor `powersave` outside intel_pstate / amd-pstate | ondemand, schedutil or performance | **[unmeasured]** |
| BAR1 without an address | Above 4G Decoding in the BIOS | the engine refuses such a card (`card_check.hpp`) |
| the IOMMU translates the cards' DMA | `iommu=pt` (or VT-d off) | [measured] 7.6-8 GB/s instead of 11.9 on ours |
| a card's link below Gen3 x16 while copying, or a narrower bridge above it | slot wiring, riser, BIOS slot speed | [measured] ours run Gen3 x16 |
| below about 10 GB/s to a card (x8, Gen2, a shared switch uplink) | start `tune.py` from fewer guessed next-layer copies: `FNX_PREFETCH_WANT` 2-3, `FNX_PREFETCH_MIN_T` 2 | **[unmeasured]** rule. It comes from UD-Q4_K_XL on ours, where PCIe was the limit: 6 guesses a row → 2 gave edits +15-20 % |
| both cards at once much slower than each alone | slots on different root ports or CPUs | [unmeasured] |
| the other CPU's memory slower than the card's own | nothing: the engine pins each card's thread to its CPU; `FNX_NUMA_EXPERTS=1` also places the RAM copies (same bits) | [measured] the same on ours, `FNX_NUMA_EXPERTS=1` in the noise |
| power limit under 185 W | raise it (`nvidia-smi -pl`) | [unmeasured]; 250 W gave reads at most 1-1.5 % over 185 W on ours [measured] |
| slow disk in a row | only the load is slower | [measured] 0.5 GB/s, about 100 s of the load on ours |
| few random reads, with the lookup table on the disk | an SSD or NVMe | [measured] on ours the first long read after a load ran at 765 tok/s, later ones at about 900 (UD-Q4_K_XL) |

## tune.py

```bash
python3 tune/tune.py --data /data/flash-next --dry-run                 # the check and the plan, nothing loaded
python3 tune/tune.py --data /data/flash-next                           # about 2-3 hours
python3 tune/tune.py --data /data/flash-next --only prefetch,dma --screen
setsid nohup python3 tune/tune.py --data /data/flash-next > /dev/null 2>&1 < /dev/null &   # detached: DATA/tune-*/progress.log
```

It starts from `DATA/profile.json` (the profile `setup.sh` wrote, at `FNX_ROUTE_PRIOR=0`) and:

1. runs `check.py` and prints its verdict; a suggested `FNX_PREFETCH_WANT` is tried first;
2. loads the starting profile twice: the spread between the two runs is your machine's noise;
3. tries one group of switches at a time, each candidate on a freshly loaded engine with `bench.py`'s fresh writing,
   edits and long read (about 4 minutes after the load; `--tests` to change). Each test's threshold is the larger of
   `--min-gain` (2 %; 4 % for fresh writing and writing after the long read, as the cards' clock follows recent load)
   and its noise. A candidate is kept when one test is faster by more than its threshold, the mean of all by at least
   half `--min-gain`, none slower by more than its threshold, and its first-token logits are the starting profile's bit
   for bit. The fastest kept candidate of a group is the bar for the next group;
4. runs the best profile once more, compares its logits with the starting profile's (`FNX_DUMP_FIRST_LOGITS`), and
   writes it to `DATA/profile.tuned.json` only if this run, too, beats the starting profile by the same rule (else it
   writes the starting profile and says so: `END ok-no-gain`). Every arm is in a table in `DATA/tune-*/summary.md`.

| Group (`--only`) | Switches | Candidates |
|---|---|---|
| `prefetch` | `FNX_PREFETCH_WANT`, `FNX_PREFETCH_MIN_T`, `FNX_PREFETCH` | 2, 3, 4, 6, 9 guesses a row and off; then from 2 or 4 rows. With `--screen` the guesses are first compared in one load through `FNX_PREFETCH_FILE` (edits, forth and back), and only those ahead there get a load of their own |
| `dma` | `FNX_DMA_SPLIT` | 0, 1, 2 |
| `experts` | `FNX_EXPERT_RAM_GIB`, `FNX_EXPERT_DYN_GIB` | ±1 GiB in RAM only (less: more experts in VRAM, until the load refuses); ±2 GiB dynamic where the profile sets it, from its whole GiB (UD-Q4_K_XL's 18.430: 20 and 16) |
| `ple` | `FNX_PLE_RAM` | the other setting, when the RAM allows |
| `convcache` | `--conversation-cache-mib` | half, only when an arm came within 4 GiB of the RAM's end (RAM, not speed) |
| `stage` (not by default) | `FNX_PF_STAGE_MIB`, `FNX_PF_TAIL_MIB` | half the staging buffers; 512 MiB more lent slots. Ours are at the size of the largest layer and smaller ones were slower |

All of these give the same bits ("none" under output effect in [docs/SWITCHES.md](../docs/SWITCHES.md)).
`FNX_ROUTE_PRIOR` (lossy) is tried only with `--allow-lossy`.

**Safety.** It refuses to start beside another engine, with the cards in use, or with something answering on its port
(18097, `--port`). Before each load it compares the RAM available with what the profile holds (README's numbers) and
skips a candidate that would not fit. It stops the whole run when swap use grows by more than 512 MiB while an arm's
tests run (`--swap-mib`) or by more than 2 GiB during its load (`--swap-load-mib`): a full machine pages out other
programs' idle memory while the engine pins its own (530 MiB in one of our loads), but a profile that does not fit keeps
swapping while it works. It stops only the server it started and the processes under
it, by PID (Strata runs the engine in a process group of its own). `DATA/tune-*/STOP` ends it after the running arm;
`progress.log` ends with a line `END <result>`.

**A smaller machine on a bigger one.** `--memory-max 58G` runs every arm in a memory cgroup of that size without swap
(`systemd-run --user --scope -p MemoryMax=58G -p MemorySwapMax=0`), as we ran the 64 GB mode (README, "64 GB of RAM").
The cgroup counts the pinned copies, the engine's memory and the page cache.

**On ours** (13 Nov 2026, the IQ3_S 512k profile at `FNX_ROUTE_PRIOR=0`, engine `c213636c`), `--screen --only
prefetch,dma` took 7 loads, about 70 minutes, and kept the profile as it is:

| Arm | Fresh | Edit py | Edit md | Read 38,862 | Write after it | Verdict |
|---|---:|---:|---:|---:|---:|---|
| base1 / base2 | 67.0 / 66.7 | 111.0 / 111.0 | 116.1 / 115.9 | 961 / 962 | 62.0 / 60.7 | noise 0.5 % or less (writing after the read 2.1 %) |
| screen, one load: `FNX_PREFETCH_WANT` 6 / 2 / 3 / 4 / 9 | | edits (mean of py and md): 113.5 / 114.3 / 114.3 / 114.8 / 107.8 | | | | none ahead of 6 by 2 % |
| `FNX_PREFETCH=0` | 66.8 | 111.8 | 114.5 | 961 | 61.7 | in the noise |
| `FNX_PREFETCH_MIN_T=2` | 66.0 | 111.7 | 115.8 | 961 | 64.3 | in the noise |
| `FNX_DMA_SPLIT=0` | 65.6 | 106.5 | 111.2 | 961 | 60.1 | edit md −4.2 % |
| `FNX_DMA_SPLIT=1` | 66.0 | 109.8 | 114.7 | 961 | 61.1 | in the noise |

Every arm's first-token logits were the baseline's bit for bit (19 of 19). An earlier run stopped at its fourth load,
when swap grew by 530 MiB while the engine pinned its memory; that is why the limit during a load is now 2 GiB.

**Keep in mind.** The benchmark's numbers depend on the expert cache, which learns what it reads, so every candidate
gets a fresh load and the order of the arms does not favour any. A change that wins by less than the noise is reported
but not kept; run the group again (`--only`) if a result looks close.
