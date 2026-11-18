#!/usr/bin/env python3
"""Writes engine/data/coder_mask.txt (FNX_EXPERT_MASK) from ISTA-DASLab's Coder allocation list.

    python3 scripts/coder_mask.py ALLOCATION.txt OUT.txt

ALLOCATION.txt is tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.rco-allocation.txt of
ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF (revision bba88856f8c3c0c9e40e45f82d05e228b814ac43). Its second
section names every expert of the unpruned model by its original index, with an empty type for a pruned one. An expert
is kept when its gate, up and down rows all have a type; the script stops if the three disagree. The output has one
line a layer: the layer, then its kept experts.
"""
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\[(\d+)\]:\s*(\S*)\s*$")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    kept = defaultdict(dict)   # (layer, expert) -> {part: kept}
    for line in open(sys.argv[1], encoding="utf-8"):
        m = ROW.match(line)
        if m:
            kept[(int(m[1]), int(m[3]))][m[2]] = bool(m[4])
    if not kept:
        sys.exit("no per-expert rows: not an rco-allocation.txt")
    layers = defaultdict(list)
    for (layer, expert), parts in sorted(kept.items()):
        if len(parts) != 3 or len(set(parts.values())) != 1:
            sys.exit(f"layer {layer} expert {expert}: gate, up and down disagree ({parts})")
        if parts["gate"]:
            layers[layer].append(expert)
    out = ["# ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF (HF, 2026-10-19 build): kept experts per layer, original indices",
           "# from tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.rco-allocation.txt, section 2 (gate = up = down)"]
    out += [" ".join(map(str, [layer] + layers[layer])) for layer in sorted(layers)]
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {sys.argv[2]}: {len(layers)} layers, {sum(map(len, layers.values()))} experts kept")


if __name__ == "__main__":
    main()
