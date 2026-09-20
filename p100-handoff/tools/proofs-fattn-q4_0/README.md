# Proofs for the q4_0 flash-attention tile loader (attempts 174-175)

These are the programs behind the accuracy claims in FINDINGS.md section 3 and CHANGES.md, plus
the harness that produced the vision-at-full-context numbers in QUICKSTART.md. They are kept
because the claims they support are published, and a published claim nobody can re-run is a
claim nobody can check.

Everything here assumes `cpy_ne == 2` on sm_60. `ggml_cuda_get_max_cpy_bytes()` returns 8 below
Volta and 16 at or above it, so `cpy_ne = that/4` is **2** on a P100, not the 4 that earlier
attempts in OPTLOG assumed. Several of the reasoning errors these programs caught trace back to
that one wrong constant.

## dequant_exhaustive.cu
The whole claim, brute-forced: for all 65536 fp16 scales x all 256 byte values, does the
magic-number dequant (`0x6400 | q`, then subtract 1032) give bit-identical results to the
`__int2half_rn` form it replaced?

    nvcc -arch=sm_60 -o dq dequant_exhaustive.cu && ./dq

Result: 0 differences over all 16777216 cases. Both steps are exact in fp16 -- every integer in
[1024,2048) and every value in [-8,7] is representable -- so there is nothing to round.

## dequant_classify.cu
The same domain against *upstream* `to_fp16`, which evaluates `d*q + (-8*d)` in fp32 and rounds
once. Classifies every disagreement rather than just counting them.

    nvcc -arch=sm_60 -o dc dequant_classify.cu && ./dc

Result: numerically identical for every finite `d` (both round the same exact real `d*(q-8)`
exactly once). The only disagreements are 31759 signs of zero (`q == 8` with `d < 0` gives -0.0
where `8d + (-8d)` gives +0.0) and 30 cases at `d == +-inf`, where this form yields the correctly
signed infinity and upstream yields NaN from `inf - inf`. This is why the docs say "numerically
identical", not "bit-exact" -- the earlier wording was an overclaim and was corrected.

## kq_error.cu
Why the half2 accumulation patch was reverted even though it measured -20.3% on the kernel.
Per lane the two forms are

    upstream: fl16(a) + fl16(b)   summed in fp32
    hfma2   : fl16(b + fl16(a))

Both round twice, but the second rounding moves off a product and onto the pair's sum, whose
magnitude is ~sqrt(2) larger, so its error variance is ~2x a product's: 3 units total against
upstream's 2. Predicted RMS ratio sqrt(3/2) = 1.2247.

    nvcc -arch=sm_60 -o kq kq_error.cu && ./kq

Measured over 2^20 random 256-dim dot products: 1.2251 / 1.2247 / 1.2249 across three input
distributions. Perplexity does not resolve the difference (2.6097 either way) and it sits far
below the q4_0 cache's own error -- it was reverted anyway, because the standing rule is that a
kept change must round no more than the code it replaces.

## loader_map.cpp
Host-side model of the restructured loader's index arithmetic. Written to answer one question:
does the new slot indexing require anything the old one did not?

It does. The restructured loader assumes each tile starts on a q4_0 block boundary
(`J % QK4_0 == 0`); the pre-restructure loader derived block and nibble from the absolute head
dimension and needed only `2*cpy_ne` alignment. That is a *narrower* contract, and nbatch_K
values of 40, 48, 56 and 72 all appear in the config table for head sizes 40..112. Unreachable
today -- only `DKQ == DV == 256` is instantiated, where nbatch_K is 64 or 128 -- so this is a
latent trap, not a live bug. It is now held shut by a `static_assert` in fattn-tile.cuh, so
widening the gate fails to build instead of silently mis-indexing.

## guard.sh
VRAM floor watchdog. `guard.sh <PID> <floor_MiB>` polls GPU0 free memory and kills **only the PID
passed in** if it drops below the floor, logging each sample as `gpu0_free=` to `$GLOG`.

GPU0 is the one carrying Sunshine's permanent 392 MB, so GPU0 starvation takes the desktop down
with it -- that has happened, and this script exists because of it. It never uses `pkill -f`:
three separate incidents in this project traced to a pattern match hitting more than intended,
including the shell running the match itself.

## vision_fit.sh, vision_ub_probe.sh
`vision_ub_probe.sh` is a load-time filter across `-ub` values; `vision_fit.sh <CTX> <UB>` drives
a real full-depth prefill and takes the **minimum** over the whole guard log.

Both steps are necessary and the split is the point. A load probe is a valid filter -- `-ub 2048`
with the mmproj died at load with 127 MiB free, so the load-time reservation alone is binding --
but it is *not* a valid peak: the prefill peak is higher (FINDINGS item 10), and extrapolating a
VRAM peak has burned this project before. So the probe narrows the field and the winner still
earns it over 259229 real tokens.

Result: the mask costs 0.742 MiB per ubatch unit with the mmproj loaded (linear to three digits:
756/1024, 380/512, 190/256). `-ub 1024` fits model + MTP + vision at the full `-c 262144`,
bottoming out at 731 MiB free -- the same margin the shipped text-only `-ub 2048` config already
runs at. The lever is `-ub`, not `-c`.

## fa_perf.sh
Isolates the flash-attention kernel via `test-backend-ops -p` at a given kv depth. Waits on a log
marker rather than polling `pgrep`, which in this project matched the waiting shell's own command
line.

Read these numbers cold. The same binary reads 30.75 +/- 0.19 t/s cold and 24.9 +/- 2.6 hot: the
P100's 175 W cap is enforced and it clocks down off its 1328 MHz ceiling under sustained load, so
a hot run silently measures the cooler, not the kernel.

## Running these later

The three shell scripts wrote into the session scratchpad, which is gone. They now default `S` to
a `run/` directory beside the script; override with `S=/somewhere ./vision_fit.sh 262144 1024`.

`vision_fit.sh` wants `$S/p262.txt`, a ~792 KB prompt that fills the 262144-token context
(259229 tokens, ~3.06 chars/token). It was a line-numbered concatenation of this repo's own
sources and is **not** committed -- for a VRAM test only the token count matters, not the text,
so any ~790 KB of prose or code does the job:

    find tools common src -name '*.cpp' -o -name '*.h' | sort | xargs cat \
      | awk '{printf "%06d %s\n", NR, $0}' | head -c 792713 > run/p262.txt

Do not reuse that file as a perplexity corpus. PPL numbers are only comparable against the exact
corpus they were calibrated on, which is `p100-handoff/ppl-orig.txt` and nothing else -- see
CORPUS.md for how that went wrong once already.
