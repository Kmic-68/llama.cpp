#!/usr/bin/env python3
# hc_up2_patch.py SRC_DIR: idea 62 (2026-11-12). k_hc's up part has 80 shares of 32 columns on a grid of 56 blocks, so 24 blocks take
# a second share, and its up weights are loaded only after the first share's whole up part (load_up at the top of the share). FNX_HC_UP2,
# bits: 1 the next share's weights loaded into the registers right after the share's dots (its scales are already in wsc), so the
# loads run beside the share's tail (rprime, sigmoid, mixed, q8); 2 a later share's weights to L2 (prefetch.global.L2) with the first
# share's loads, before the first grid barrier. The same values: the same sums. Each edit must match as counted.
import sys

src = sys.argv[1]


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1)
    open(p, "w").write(s)


edit("hc.cuh", [(
    "    int8_t* tq = nullptr;   // and its split twin, or null\n"
    "    float* td = nullptr;\n"
    "};\n",
    "    int8_t* tq = nullptr;   // and its split twin, or null\n"
    "    float* td = nullptr;\n"
    "    int up2 = 0;            // FNX_HC_UP2 (idea 62): 1 the next share's up weights after the share's dots, 2 a later share's to L2 early\n"
    "};\n",
    1,
), (
    "    load_up(blockIdx.x);\n"
    "    grid_sync(a.bar, a.err, grid);\n",
    "    load_up(blockIdx.x);\n"
    "    if ((a.up2 & 2) && ulive)\n"
    "        for (int v2 = (int) blockIdx.x + (int) grid; v2 < H_UPS; v2 += (int) grid)\n"
    "#pragma unroll\n"
    "            for (int cc = 0; cc < HC; ++cc) {\n"
    "                const size_t urow = (size_t) cc * N + v2 * H_UPC + udd;\n"
    "                const int8_t* wp = a.w_up.q + urow * LR + uj * 32;\n"
    "                asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(wp));\n"
    "                asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(wp + 16));\n"
    "                if (uj == 0) asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(a.w_up.d + urow * (LR / 32)));\n"
    "            }\n"
    "    grid_sync(a.bar, a.err, grid);\n",
    1,
), (
    "    if (vb != blockIdx.x) {\n"
    "        __syncthreads();   // the last share is done with gs / ms\n"
    "        load_up(vb);\n"
    "    }\n",
    "    if (vb != blockIdx.x) {\n"
    "        __syncthreads();   // the last share is done with gs / ms\n"
    "        if (!(a.up2 & 1)) load_up(vb);   // FNX_HC_UP2 1: loaded after the last share's dots\n"
    "    }\n",
    1,
), (
    "            if (t < H_UPC * H_TPRU && uj == 0) gs[k][cc][udd] = acc[cc];   // the sigmoid below, once per element on\n"
    "        }                                                                      // every lane (here a warp issued it for 2)\n"
    "    }\n",
    "            if (t < H_UPC * H_TPRU && uj == 0) gs[k][cc][udd] = acc[cc];   // the sigmoid below, once per element on\n"
    "        }                                                                      // every lane (here a warp issued it for 2)\n"
    "    }\n"
    "    if ((a.up2 & 1) && vb + (int) grid < H_UPS) load_up(vb + (int) grid);   // the share's own scales are in wsc already\n",
    1,
)])
edit("kernels.cu", [(
    "    if (q8) twin_of(q8, N, a.tq, a.td);\n"
    "    cudaStream_t s = (cudaStream_t) st;\n"
    "    const bool one = T <= hc1_rows(), h16 = hc_f16();\n",
    "    if (q8) twin_of(q8, N, a.tq, a.td);\n"
    "    static const int up2 = [] {\n"
    "        const int u = (int) env_int(\"FNX_HC_UP2\", 0);\n"
    "        if (u < 0 || u > 3) throw std::runtime_error(\"FNX_HC_UP2: a sum of 1 (the next share's up weights early) and 2 (a later share's to L2)\");\n"
    "        if (u) std::fprintf(stderr, \"fnx: FNX_HC_UP2 %d: k_hc's later up shares loaded early\\n\", u);\n"
    "        return u;\n"
    "    }();\n"
    "    a.up2 = up2;\n"
    "    cudaStream_t s = (cudaStream_t) st;\n"
    "    const bool one = T <= hc1_rows(), h16 = hc_f16();\n",
    1,
)])
print("hc_up2_patch: hc.cuh, kernels.cu")
