# What is whose

| Part | Origin | License |
|---|---|---|
| `engine/` (sources) | Strata v0.2.10's `src/split/` (github.com/Niko1221/Strata, commit `ffa80d73133ea5c548aefe57889a16e8085a938f`), modified and extended by us for the Tesla P100 | MIT: Strata's notice and ours, [LICENSE](LICENSE) |
| Strata's `src/split/` itself | a port of sFNX by John Fink (https://github.com/johnfink8/sfnx, commit `f9ba1e9`) | MIT, [third_party/sfnx/](third_party/sfnx/) |
| `engine/src/iq3s_x.cuh` | generated from ggml's `ggml-cuda/vecdotq.cuh` (llama.cpp commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`) with its dot product swapped for ours | MIT, The ggml authors, [third_party/ggml/LICENSE](third_party/ggml/LICENSE) |
| `engine/data/expert-routes.bin`, `engine/data/draft_vocab_en.bin` | sFNX's data files, as Strata v0.2.10 ships them (byte-identical) | MIT (sFNX) |
| `engine/data/draft_vocab_sl.bin` | ours: `draft_vocab_en.bin` plus 8,621 token ids chosen with Strata's `tools/draft_vocab.py --corpus` from 1,483 random articles of the Slovenian Wikipedia (the text is not included) | MIT (ours) |
| `engine/data/coder_mask.txt` | the experts ISTA-DASLab's Coder release keeps, from its allocation list (Hugging Face `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`), rewritten by our `scripts/coder_mask.py` | Apache-2.0, [third_party/ista-daslab/](third_party/ista-daslab/) |
| `engine/p100_compat.h`, `engine/src/p100_*`, the changes to the other files | ours | MIT (ours) |
| `CMakeLists.txt`, `build.sh`, `setup.sh`, `run.sh`, `scripts/`, `profile/`, `bench/`, `docs/`, `history/` | ours; `bench/bench.py` reads its texts from a Strata checkout at run time | MIT (ours) |

[docs/PROVENANCE.md](docs/PROVENANCE.md) says for every file in `engine/` whether it is Strata's unchanged, Strata's
modified, or new, with its SHA-256. [docs/upstream-v0.2.10.diff](docs/upstream-v0.2.10.diff) is every change against
Strata v0.2.10's `src/split/`. The paths in sFNX's `NOTICE.md` (`src/split/...`) are `engine/...` here.

Not included and not redistributed: the model files (ISTA-DASLab's and Unsloth's GGUF, Edge0's Edge0-145B files under the Edge0
Community Model License 1.0, Qwen's checkpoint for the MTP layer), Strata's server and tools, and llama.cpp (`build.sh` fetches
the headers it needs). `setup.sh` downloads the model files from their own repositories; their licenses are theirs.

Ideas taken from others, with thanks: the 16-bit form of the byte dot product (`fnx_dp4a_xmad`) from KaneQuah/Kmic-68's
llama.cpp fork (commit `99eb48176d`); a host thread per card that copies missing experts with the copy engine
(`FNX_P100_DMA`) from DeepInfer; the inline-assembly register constraint in `fnx_dp4a_vmad` from Strata PR #4274; the
expert pool floor from Strata 0.2.12 (commit `20885f7a`); `FNX_PF_TAIL_MIB=4096` for UD-Q4_K_XL from a tester's recipe for that file;
the MTP draft layer's source normalised per stream (`FNX_MTP_NORM_STREAMS`) from the notes of the Colibri engine
(JustVugg/colibri, Apache-2.0: its `docs/qwen38.md`, `Q38_MTP_WIRING=b`), the idea only, none of its code.
