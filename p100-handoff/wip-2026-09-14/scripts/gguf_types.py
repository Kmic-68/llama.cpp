#!/usr/bin/env python3
# Minimal read-only GGUF tensor lister (no numpy): prints a count of tensor types per name suffix,
# so the weight types behind the tensor-parallel row-split matmuls can be checked.
import struct, sys, collections

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
              24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
              39: "MXFP4"}

def main(path):
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
        magic, version = f.read(4), rd("I")[0]
        assert magic == b"GGUF", magic
        n_tensors, n_kv = rd("QQ")
        def rstr():
            n = rd("Q")[0]; return f.read(n).decode("utf-8", "replace")
        sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        def rval(t):
            if t == 8: return rstr()
            if t == 9:
                et, n = rd("I")[0], rd("Q")[0]
                return [rval(et) for _ in range(n)]
            return rd(sizes[t])[0]
        kv = {}
        for _ in range(n_kv):
            k = rstr(); t = rd("I")[0]; v = rval(t)
            kv[k] = v if not isinstance(v, list) or len(v) < 8 else f"[{len(v)} items]"
        print("arch:", kv.get("general.architecture"), " file_type:", kv.get("general.file_type"), " tensors:", n_tensors)
        by_suffix = collections.defaultdict(collections.Counter)
        for _ in range(n_tensors):
            name = rstr(); nd = rd("I")[0]; dims = rd("Q" * nd); t = rd("I")[0]; rd("Q")
            parts = name.split(".")
            suffix = ".".join(parts[2:]) if parts[0] == "blk" else name
            by_suffix[suffix][(GGML_TYPES.get(t, t), "x".join(map(str, dims)))] += 1
        for suffix in sorted(by_suffix):
            print(f"  {suffix:40s} " + ", ".join(f"{ty} {shape} x{c}" for (ty, shape), c in sorted(by_suffix[suffix].items())))

main(sys.argv[1])
