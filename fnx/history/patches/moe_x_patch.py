#!/usr/bin/env python3
# moe_x_patch.py SRC_DIR: idea 67 (2026-11-12). The IQ3_S expert kernels of the writing path (moe.cuh's k_moe_gu, k_moe_down) with
# the int8 dot products of ggml's vec_dot_iq3_s_q8_1 in p100_compat.h's xmad form (iq3s_x.cuh beside this script, copied to the
# sources): on sm_60 ggml's own are the compiler's from C (ggml_cuda_dp4a under GGML_CUDA_CC_DP4A), about a sixth slower alone
# (mikro/kane/dp4a_hitrost_c.log). The same sums. Off unless the engine is told: FNX_MOE_X (bit T: the gate|up kernel of T rows of
# window), FNX_MOE_XD (1: the down kernel to 8 rows of window, 2: past 8). After xmad_patch.py (the two-form header,
# FNX_DP4A_FORMS). Each edit must match as counted.
import os
import shutil
import sys

src = sys.argv[1]
shutil.copyfile(os.path.dirname(os.path.abspath(__file__)) + "/iq3s_x.cuh", src + "/iq3s_x.cuh")


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want, first) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1) if first else s.replace(a, b)
    open(p, "w").write(s)


edit("kernels.cu", [(
    "template <ggml_type type> __device__ __forceinline__ float vdot(const void* v, const block_q8_1* y, int kbx, int iqs) {\n",
    "#ifdef FNX_DP4A_FORMS\n"
    "#include \"iq3s_x.cuh\"\n"
    "#endif\n"
    "// X: IQ3_S with its int8 dot products in the xmad form (iq3s_x.cuh; the same sums)\n"
    "template <ggml_type type, bool X = false>\n"
    "__device__ __forceinline__ float vdot(const void* v, const block_q8_1* y, int kbx, int iqs) {\n",
    1, False,
), (
    "    else if constexpr (type == GGML_TYPE_IQ3_S) return vec_dot_iq3_s_q8_1(v, y, kbx, iqs);\n",
    "#ifdef FNX_DP4A_FORMS\n"
    "    else if constexpr (type == GGML_TYPE_IQ3_S && X) return vec_dot_iq3_s_q8_1_x(v, y, kbx, iqs);\n"
    "#endif\n"
    "    else if constexpr (type == GGML_TYPE_IQ3_S) return vec_dot_iq3_s_q8_1(v, y, kbx, iqs);\n",
    1, False,
)])

GX = "".join(
    "                case %d: k_moe_gu<TA, G, %d, true><<<grid, MMVQ_THREADS, 0, s>>>(w, jobs, njobs, y, out); return;\n" % (t, t)
    for t in range(1, 17))
edit("moe.cuh", [
    (
        "template <ggml_type type, int G, int TT>\n__device__ __forceinline__ void job_rows(",
        "template <ggml_type type, int G, int TT, bool X = false>\n__device__ __forceinline__ void job_rows(",
        1, False,
    ),
    ("tmp[j] += vdot<type>(wr, ", "tmp[j] += vdot<type, X>(wr, ", 1, False),
    (
        "template <ggml_type type, int U>\n__device__ __forceinline__ float unit_dot(",
        "template <ggml_type type, int U, bool X = false>\n__device__ __forceinline__ float unit_dot(",
        1, False,
    ),
    ("s += vdot<type>(wrow, ", "s += vdot<type, X>(wrow, ", 1, False),
    (
        "template <ggml_type TA, int G, int TT>\n__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu(",
        "template <ggml_type TA, int G, int TT, bool X = false>\n__global__ void __launch_bounds__(MMVQ_THREADS) k_moe_gu(",
        1, False,
    ),
    ("    job_rows<TA, G, TT>(moe_expert(", "    job_rows<TA, G, TT, X>(moe_expert(", 1, False),
    (
        "template <ggml_type TA, bool PRE>\n__global__ void __launch_bounds__(DN_THREADS) k_moe_down(",
        "template <ggml_type TA, bool PRE, bool X = false>\n__global__ void __launch_bounds__(DN_THREADS) k_moe_down(",
        1, False,
    ),
    # k_moe_down's (the first; the second is k_moe_down_big's)
    ("part[j][threadIdx.x] = unit_dot<TA, DN_U>(wr, u, hq[j]);", "part[j][threadIdx.x] = unit_dot<TA, DN_U, X>(wr, u, hq[j]);", 2, True),
    (
        "// the gate|up kernel for a window of T rows (a job holds at most T)\n",
        "// idea 67: IQ3_S's writing kernels with the dots in the xmad form (iq3s_x.cuh). FNX_MOE_X: bit T: the gate|up kernel of T rows\n"
        "// of window; FNX_MOE_XD: 1: the down kernel to 8 rows of window, 2: past 8\n"
        "#ifdef FNX_DP4A_FORMS\n"
        "inline unsigned moe_x() { static const unsigned m = (unsigned) env_int(\"FNX_MOE_X\", 0); return m; }\n"
        "inline unsigned moe_xd() { static const unsigned m = (unsigned) env_int(\"FNX_MOE_XD\", 0); return m; }\n"
        "#endif\n"
        "template <ggml_type TA, bool PRE>\n"
        "void dn_launch(dim3 grid, size_t smem, cudaStream_t s, const MoeW& w, const Job* jobs, const int* njobs, const float* gu,\n"
        "               float* out) {\n"
        "#ifdef FNX_DP4A_FORMS\n"
        "    if constexpr (TA == GGML_TYPE_IQ3_S) {\n"
        "        if (moe_xd() & (PRE ? 2u : 1u)) {\n"
        "            k_moe_down<TA, PRE, true><<<grid, DN_THREADS, smem, s>>>(w, jobs, njobs, gu, out);\n"
        "            return;\n"
        "        }\n"
        "    }\n"
        "#endif\n"
        "    k_moe_down<TA, PRE><<<grid, DN_THREADS, smem, s>>>(w, jobs, njobs, gu, out);\n"
        "}\n"
        "\n"
        "// the gate|up kernel for a window of T rows (a job holds at most T)\n",
        1, False,
    ),
    (
        "const block_q8_1* y, float* out) {\n    switch (T) {\n        case 1: k_moe_gu<TA, G, 1><<<",
        "const block_q8_1* y, float* out) {\n"
        "#ifdef FNX_DP4A_FORMS\n"
        "    if constexpr (TA == GGML_TYPE_IQ3_S) {\n"
        "        if (T >= 1 && T <= 16 && ((moe_x() >> T) & 1u)) {\n"
        "            switch (T) {\n"
        + GX +
        "                default: break;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "#endif\n"
        "    switch (T) {\n        case 1: k_moe_gu<TA, G, 1><<<",
        1, False,
    ),
    (
        "            k_moe_down<TA, true><<<grid, DN_THREADS, smem, s>>>(w, jobs, njobs, (const float*) in, out);\n",
        "            dn_launch<TA, true>(grid, smem, s, w, jobs, njobs, (const float*) in, out);\n",
        1, False,
    ),
    (
        "            k_moe_down<TA, false><<<grid, DN_THREADS, smem, s>>>(w, jobs, njobs, (const float*) in, out);\n",
        "            dn_launch<TA, false>(grid, smem, s, w, jobs, njobs, (const float*) in, out);\n",
        1, False,
    ),
])
print("moe_x_patch: kernels.cu, moe.cuh, iq3s_x.cuh")
