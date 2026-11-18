#!/usr/bin/env python3
"""mtp_norm_patch.py (2026-11-16): FNX_MTP_NORM_STREAMS=1, the draft layer's source (pre_fc_norm_hidden, 4 x 2560 weights) normalised
per stream, as the hyper-connection norms do (hc.cuh's rs_of), instead of as one vector of the four streams. The tensors do not say
which it is; Colibri (docs/qwen38.md, Q38_MTP_WIRING=b) measured a higher acceptance of the drafts per stream. Only the drafts change
(the main model verifies each), so the output's distribution stays exact. Off (the default) the engine computes the same bits.
Run from /srv/build/strata-hkrati/upstream/src/split; originals go to /srv/build/strata-hkrati/varne/pred-mtp-norm/.
mtp_norm_patch.py DIR (2026-11-17, a production source's copy: mtp_norm/zgradi_norm.sh): the originals go to DIR instead."""
import shutil
import sys
from pathlib import Path

root = Path.cwd()
safe = Path(sys.argv[1] if len(sys.argv) > 1 else "/srv/build/strata-hkrati/varne/pred-mtp-norm")
files = ["src/kernels.hpp", "src/kernels.cu", "src/engine.cu", "src/prefill.cu"]
if not (root / "src/engine.cu").exists():
    sys.exit("run from upstream/src/split")
if "rms_norm_streams" in (root / "src/kernels.cu").read_text(encoding="utf-8"):
    sys.exit("rms_norm_streams is already there: patched")
safe.mkdir(parents=True, exist_ok=True)
for f in files:
    shutil.copy2(root / f, safe / Path(f).name)

texts = {}


def edit(path, old, new, count=1):
    t = texts.setdefault(path, (root / path).read_text(encoding="utf-8"))
    n = t.count(old)
    if n != count:
        sys.exit(f"{path}: {n} matches, {count} expected:\n{old}")
    texts[path] = t.replace(old, new)


edit("src/kernels.hpp",
     "void rms_norm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, void* st);\n",
     "void rms_norm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, void* st);\n"
     "// rms_norm_rows with each of a row's hc streams of n normalised on its own (w of hc * n): FNX_MTP_NORM_STREAMS\n"
     "void rms_norm_streams(const float* x, const float* w, float* y, int rows, int hc, int n, float eps, void* st);\n")

edit("src/kernels.cu",
     '    ck_launch("rms_norm_rows");\n}\n',
     '    ck_launch("rms_norm_rows");\n}\n'
     "namespace {\n"
     "// y[r][c] = x[r][c] / sqrt(mean(x[r][c]^2) + eps) * w[c], a block per stream (mtp_norm_patch.py)\n"
     "__global__ void k_rms_norm_streams(const float* x, const float* w, float* y, int hc, int n, float eps) {\n"
     "    __shared__ float sh[32];\n"
     "    const float* xr = x + (size_t) blockIdx.x * n;\n"
     "    const float* wr = w + (size_t) (blockIdx.x % hc) * n;\n"
     "    float s = 0.f;\n"
     "    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];\n"
     "    const float r = rsqrtf(block_sum(s, sh) / (float) n + eps);\n"
     "    for (int i = threadIdx.x; i < n; i += blockDim.x) y[(size_t) blockIdx.x * n + i] = xr[i] * r * wr[i];\n"
     "}\n"
     "}  // namespace\n"
     "void rms_norm_streams(const float* x, const float* w, float* y, int rows, int hc, int n, float eps, void* st) {\n"
     "    k_rms_norm_streams<<<rows * hc, 1024, 0, (cudaStream_t) st>>>(x, w, y, hc, n, eps);\n"
     '    ck_launch("rms_norm_streams");\n'
     "}\n")

edit("src/engine.cu",
     "        rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);\n",
     '        static const bool norm_streams = env_flag("FNX_MTP_NORM_STREAMS");   // mtp_norm_patch.py: each stream on its own\n'
     "        if (norm_streams) rms_norm_streams(b.mtp_src, mc.norm_hid, b.mtp_hn, T, HC, N, eps, b.st);\n"
     "        else rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);\n", 2)
edit("src/engine.cu",
     "        b.mtp_src = dalloc<float>((size_t) MAXT * D);\n",
     '        if (c == 0 && env_flag("FNX_MTP_NORM_STREAMS")) std::fprintf(stderr, "sfnx: the draft layer\'s source normalised per stream\\n");\n'
     "        b.mtp_src = dalloc<float>((size_t) MAXT * D);\n")

edit("src/prefill.cu",
     "        rms_norm_rows(p.R, mc.norm_hid, p.xf, C, D, eps, s);\n",
     '        static const bool norm_streams = env_flag("FNX_MTP_NORM_STREAMS");   // mtp_norm_patch.py: as Engine::mtp_enqueue\n'
     "        if (norm_streams) rms_norm_streams(p.R, mc.norm_hid, p.xf, C, HC, N, eps, s);\n"
     "        else rms_norm_rows(p.R, mc.norm_hid, p.xf, C, D, eps, s);\n")

for path, t in texts.items():
    (root / path).write_text(t, encoding="utf-8")
    print("patched", path)
