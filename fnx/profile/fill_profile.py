#!/usr/bin/env python3
"""Writes a run config for Strata's server (serve/server.py --config) from one of the templates in this folder.

    python3 profile/fill_profile.py --models /data/flash-next --mtp /data/flash-next/mtp/rt \
        --tokenizer /data/flash-next/tokenizer --vision-exe /opt/strata-v0.2.9/build-vision/bin/strata-vision \
        --logs /var/tmp --out my-profile.json

--models     the folder with the model's files (both IQ3_S shards, or the four UD-Q4_K_XL parts, or Edge0's four files in
             edge0-145b/ beside the IQ3_S's second shard) and mmproj-Qwen3.8-Flash-Next-BF16.gguf
--model      iq3_s (default): ISTA-DASLab's IQ3_S, flash-next-iq3s-512k.template.json; q4_k_xl: Unsloth's UD-Q4_K_XL,
             flash-next-q4_k_xl-512k.template.json (README, "Unsloth's UD-Q4_K_XL"; its 256k one with
             --template profile/flash-next-q4_k_xl-256k.template.json); edge0: Edge0's Edge0-145B-NVFP4,
             flash-next-edge0-512k.template.json (README, "Edge0-145B")
--mtp        the MTP draft layer's runtime folder (dense.bin, dense.txt, experts.bin; README, "The MTP draft layer")
--tokenizer  the tokenizer folder (vocab.json, merges.txt, chat_template.jinja ...; README, "The tokenizer folder")
--vision-exe Strata's strata-vision; leave it out to run without pictures (the "vision" block and --vision go)
--route-prior FNX_ROUTE_PRIOR: 0 (default) is exact; 0.15 is our production setting, faster and lossy (README)
--draft-vocab another draft vocabulary than engine/data/draft_vocab_sl.bin, e.g. engine/data/draft_vocab_en.bin
--exe        the engine, if build.sh was run with another build directory (default build/split/strata-split-2)
--gpu        the two P100 as nvidia-smi numbers them (default "0,1")
--ram        96 (default): the template as it is; 64: the 64 GB mode (README, "64 GB of RAM"; IQ3_S only): FNX_PLE_RAM=0,
             FNX_EXPERT_DYN_GIB=13, --max-context 262144 without YaRN, --conversation-cache-mib 2000
--coder      FNX_EXPERT_MASK=engine/data/coder_mask.txt: the router picks only from the 256 experts a layer that
             ISTA-DASLab's Coder release keeps (README, "Coder"); faster, lossy, for code

Every path the config names is checked; a missing one is reported and nothing is written. Anything else (the context
length, other switches) is edited in the written JSON; docs/SWITCHES.md lists the switches.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent
# per model: its template and its files (the first is --native; the engine finds the others beside it)
MODELS = {
    "iq3_s": ("flash-next-iq3s-512k.template.json",
              [f"Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{i}-of-00002.gguf" for i in (1, 2)]),
    "q4_k_xl": ("flash-next-q4_k_xl-512k.template.json",
                [f"Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf" for i in (1, 2, 3, 4)]),
    # Edge0-145B: the skeleton GGUF is --native, the latent experts and D/U beside it (FNX_LATENT), the lookup table from the
    # IQ3_S's second shard (FNX_LATENT_PLE, checked below)
    "edge0": ("flash-next-edge0-512k.template.json",
              ["model-skeleton.gguf", "experts.bin", "latent_dense_q8.bin", "labels.bin"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True)
    ap.add_argument("--model", choices=sorted(MODELS), default="iq3_s")
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--vision-exe")
    ap.add_argument("--logs", default="/var/tmp")
    ap.add_argument("--route-prior", default="0")
    ap.add_argument("--draft-vocab", help="another draft vocabulary (default engine/data/draft_vocab_sl.bin)")
    ap.add_argument("--exe", default=str(RELEASE / "build" / "split" / "strata-split-2"))
    ap.add_argument("--gpu", default="0,1")
    ap.add_argument("--ram", choices=["96", "64"], default="96", help="64: the 64 GB mode (README, \"64 GB of RAM\")")
    ap.add_argument("--coder", action="store_true", help="only the Coder release's experts (README, \"Coder\")")
    ap.add_argument("--template", help="another template than the model's")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if float(a.route_prior) < 0:
        sys.exit("--route-prior: 0 or more")
    if a.ram == "64" and a.model != "iq3_s":
        sys.exit("--ram 64: only with --model iq3_s (UD-Q4_K_XL's experts alone are 72 GiB)")
    if a.coder and a.model == "edge0":
        sys.exit("--coder: not with --model edge0 (the Coder's list names Flash-Next's experts, not Edge0's latent ones)")
    cards = a.gpu.split(",")
    if len(cards) != 2 or not all(c.strip().isdigit() for c in cards) or cards[0].strip() == cards[1].strip():
        sys.exit("--gpu: two different card numbers, e.g. 0,1")
    template, files = MODELS[a.model]
    values = {"{{RELEASE}}": str(RELEASE), "{{EXE}}": str(Path(a.exe).resolve()),
              "{{MODELS}}": str(Path(a.models).resolve()), "{{MTP}}": str(Path(a.mtp).resolve()),
              "{{TOKENIZER}}": str(Path(a.tokenizer).resolve()), "{{LOGS}}": str(Path(a.logs).resolve()),
              "{{VISION_EXE}}": str(Path(a.vision_exe).resolve()) if a.vision_exe else "", "{{ROUTE_PRIOR}}": a.route_prior}

    def fill(x):
        if isinstance(x, str):
            for k, v in values.items():
                x = x.replace(k, v)
            return x
        if isinstance(x, list):
            return [fill(v) for v in x]
        if isinstance(x, dict):
            return {k: fill(v) for k, v in x.items()}
        return x

    cfg = fill(json.loads(Path(a.template or HERE / template).read_text(encoding="utf-8")))
    cfg["gpu"] = ",".join(c.strip() for c in cards)
    if a.draft_vocab:
        cfg["args"][cfg["args"].index("--draft-vocab") + 1] = str(Path(a.draft_vocab).resolve())
    if not a.vision_exe:
        cfg.pop("vision")
        cfg["args"].remove("--vision")
    if a.ram == "64":
        # the lookup table read through the page cache, only the coldest 13 GiB of the experts in VRAM with a RAM copy; at 524,288
        # positions the K/V pool would need nearly every expert slot, so the trained 262,144 (no YaRN) and a smaller conversation cache
        cfg["env"]["FNX_PLE_RAM"] = "0"
        cfg["env"]["FNX_EXPERT_DYN_GIB"] = "13"
        cfg["args"][cfg["args"].index("--max-context") + 1] = "262144"
        cfg["args"][cfg["args"].index("--conversation-cache-mib") + 1] = "2000"
        for k in ("FNX_ROPE_SCALE", "FNX_YARN_ORIG_CTX"):
            cfg["env"].pop(k)
    if a.coder:
        cfg["env"]["FNX_EXPERT_MASK"] = str(RELEASE / "engine" / "data" / "coder_mask.txt")

    native = Path(cfg["args"][cfg["args"].index("--native") + 1])
    need = [str(native.with_name(f)) for f in files] + [
            cfg["args"][cfg["args"].index("--draft-vocab") + 1],
            cfg["env"]["FNX_EXPERT_PROFILE"], cfg["tokenizer"] + "/vocab.json", cfg["tokenizer"] + "/chat_template.jinja",
            cfg["args"][cfg["args"].index("--mtp") + 1] + "/dense.bin",
            cfg["args"][cfg["args"].index("--mtp") + 1] + ("/experts-q4_0.bin" if cfg["env"].get("FNX_MTP_EXPERTS") == "q4_0" else "/experts.bin"),
            cfg["exe"]]
    if "vision" in cfg:
        need += [cfg["vision"]["exe"], cfg["vision"]["mmproj"]]
    if "FNX_EXPERT_MASK" in cfg["env"]:
        need.append(cfg["env"]["FNX_EXPERT_MASK"])
    if "FNX_LATENT_PLE" in cfg["env"]:
        need.append(cfg["env"]["FNX_LATENT_PLE"])
    missing = [p for p in need if not Path(p).exists()]
    if missing:
        sys.exit("missing, nothing written:\n  " + "\n  ".join(missing))
    Path(a.out).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out} ({a.model}, FNX_ROUTE_PRIOR={a.route_prior}, pictures {'on' if 'vision' in cfg else 'off'}, "
          f"{a.ram} GB of RAM{', the Coder experts' if a.coder else ''})")


if __name__ == "__main__":
    main()
