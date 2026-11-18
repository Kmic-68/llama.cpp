#!/usr/bin/env python3
"""The MTP draft layer's routed experts as Q4_0, made from Qwen's BF16 tensors, for FNX_MTP_EXPERTS=q4_0.

    python3 scripts/mtp_q4.py DATA/prep/mtp-bf16 DATA/prep/mtp-rt/experts-q4_0.bin

The first argument is the folder Strata's `tools/mtp_fetch.py fetch` wrote (`tensors/*.bin` and `mtp-manifest.json`); only
its two expert tensors are read: `gate_up_proj` [512, 1280, 2560] (gate rows 0-639, up rows 640-1279) and `down_proj`
[512, 2560, 640], both BF16. The output is what the engine's `MtpFiles::experts_blocks` reads: gate, up, down, each
[512][rows] of 32-value Q4_0 blocks (18 bytes), 1,415,577,600 bytes. The quantizer is Strata's `tools/mtp_pack.py` q4_0
(ggml's `quantize_row_q4_0_ref`). The draft layer takes the same VRAM as with Strata's 2-bit `experts.bin`, which the
engine unpacks to Q4_0 blocks too.

The draft only proposes tokens; the model checks every one, so the answer stays the same token for token (our round
`mtp3`, 13 Nov 2026: identical texts). A draft closer to the model gets more of them accepted: UD-Q4_K_XL wrote 10 %
faster (fresh 46.99 -> 51.84 tok/s at FNX_ROUTE_PRIOR=0), IQ3_S about 1 %.
"""
import sys
from pathlib import Path

import numpy as np

E, F, H, BB = 512, 640, 2560, 18


def bf16(path, shape):
    raw = np.memmap(path, dtype=np.uint16, mode="r")
    if raw.size != int(np.prod(shape)):
        sys.exit(f"{path}: {raw.size * 2} bytes, not the {int(np.prod(shape)) * 2} of BF16 {shape}")
    return raw.reshape(shape)


def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def q4_0(w):
    """ggml quantize_row_q4_0_ref: d = signed max / -8, code = min(15, trunc(x / d + 8.5))."""
    x = w.reshape(-1, 32).astype(np.float32)
    mx = x[np.arange(x.shape[0]), np.abs(x).argmax(axis=1)][:, None]
    d = mx / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    q = np.minimum(15, np.trunc(x * inv + 8.5)).astype(np.uint8)
    out = np.empty((x.shape[0], BB), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.reshape(-1)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = Path(sys.argv[1]) / "tensors", Path(sys.argv[2])
    gu = bf16(src / "mtp.layers.0.mlp.experts.gate_up_proj.bin", (E, 2 * F, H))
    dn = bf16(src / "mtp.layers.0.mlp.experts.down_proj.bin", (E, H, F))
    G = E * F * (H // 32) * BB
    D = E * H * (F // 32) * BB
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        f.truncate(2 * G + D)
        for e in range(E):
            w = f32(np.asarray(gu[e]))
            g, u, d = q4_0(w[:F]), q4_0(w[F:]), q4_0(f32(np.asarray(dn[e])))
            f.seek(e * g.size)
            f.write(g.tobytes())
            f.seek(G + e * u.size)
            f.write(u.tobytes())
            f.seek(2 * G + e * d.size)
            f.write(d.tobytes())
            if e % 128 == 127:
                print(f"{e + 1} of {E} experts", flush=True)
    tmp.replace(out)
    print(f"wrote {out} ({2 * G + D:,} bytes)")


if __name__ == "__main__":
    main()
