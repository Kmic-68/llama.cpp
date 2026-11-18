#!/usr/bin/env python3
# xmad_patch.py SRC_DIR: idea 63 per kernel (2026-11-12). The int8 dot product of the dense Q8 kernels (dense.cuh) and of the hc read
# (hc.cuh) in the form each kernel is faster with: p100_compat.h's vmad or xmad (both exact; the header must be the two-form one,
# FNX_DP4A_FORMS). Which kernel takes which is in dense.cuh's FNX_Q8S_* and hc.cuh's FNX_HC_X_T; their defaults are the choice
# below, measured in mikro/kane/izbira/prvi/izbira.log (q8s_izbira.cu, hc_izbira.cu; izbira.sh's variant v is production
# b1438fe1's code: all vmad, the loops and the launcher as before). Each edit must match once.
import sys

src = sys.argv[1]
# the choice (bit T: the kernel of T rows of window; izbira.sh's builds override them with -D)
X1, X2, X4 = "0x1c86cu", "0xfed8u", "0x7fe0u"   # xmad: 1 row a warp at 2, 3, 5, 6, 11, 14-16 rows of window; 2 at 3, 4, 6, 7, 9-15; 4 at 5-14
MANY2, TC8 = "0x118u", "0x6200u"    # 2 rows a warp with the wide windows' loop at 3, 4, 8; 4 rows a warp, 8 activation rows at a time at 9, 13, 14
R4_T, TWO_T = "5", "15"             # 8192 rows and more: 4 rows a warp from R4_T rows of window; 2048 and more: two launches from TWO_T
HC_X_T = "7"                        # the hc read: xmad from this many rows of window
C1, C2, C4, HC_LOW_F = "0x82u", "0u", "0u", "0"   # the compiler's own form: 1 row a warp at 1, 7; the hc read under HC_X_T rows: vmad (0)


def edit(name, pairs, counts=None):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b) in enumerate(pairs):
        want = counts[i] if counts else 1
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b)
    open(p, "w").write(s)


edit("kernels.cu", [(
    "        if (rows >= 8192 && T > 6) q8s_launch<4, SEG>(g, rows, cols, yq, T, s);\n",
    "        // dense.cuh's choice (FNX_Q8S_R4_T, FNX_Q8S_TWO_T): a wide window in two launches, 8 rows and then the rest, each by this\n"
    "        // same choice (each launch stays under its step of registers; a row's sum does not depend on the rows beside it)\n"
    "        static_assert(FNX_Q8S_TWO_T > 8, \"FNX_Q8S_TWO_T: the two launches are 8 rows and the rest\");\n"
    "        if (rows >= 2048 && T >= FNX_Q8S_TWO_T) {\n"
    "            Q8Segs h = g;\n"
    "            for (int i = 0; i < Q8S_SEGS; ++i) h.out[i] += (size_t) 8 * g.stride[i];\n"
    "            q8s_run<SEG>(g, rows, cols, y, 8, st);\n"
    "            q8s_run<SEG>(h, rows, cols, (const char*) y + (size_t) 8 * (cols / 32) * sizeof(block_q8_1), T - 8, st);\n"
    "            return;\n"
    "        }\n"
    "        if (rows >= 8192 && T >= FNX_Q8S_R4_T) q8s_launch<4, SEG>(g, rows, cols, yq, T, s);\n",
), (
    "four for big matrices past 6\n"
    "    // rows of window (activation loads shared by more rows), two below that (split38: 6-11 % faster than four on 8192 and 32768\n"
    "    // rows x 2560) and for 2048 to 8191 rows",
    "four for big matrices from\n"
    "    // FNX_Q8S_R4_T rows of window on (dense.cuh; activation loads shared by more rows; the segmented variant past 6), two below\n"
    "    // that and for 2048 to 8191 rows",
), (
    "__device__ __forceinline__ float f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }\n",
    "__device__ __forceinline__ float f16(uint16_t h) { return __half2float(__ushort_as_half(h)); }\n"
    "// the int8 dot product in the form a kernel chose: p100_compat.h's on sm_60 (F: 0 vmad, 1 xmad, 2 the compiler's own from C; the\n"
    "// same sums), elsewhere the card's own instruction\n"
    "template <int F> __device__ __forceinline__ int dp4a_x(int a, int b, int c) {\n"
    "#ifdef FNX_DP4A_FORMS\n"
    "    if constexpr (F == 1) return fnx_dp4a_xmad(a, b, c);\n"
    "    else if constexpr (F == 2) return fnx_dp4a_c(a, b, c);\n"
    "    else return fnx_dp4a_vmad(a, b, c);\n"
    "#else\n"
    "    return __dp4a(a, b, c);\n"
    "#endif\n"
    "}\n",
)])

edit("dense.cuh", [
    (
        "constexpr int Q8S_THREADS = 256, Q8S_SEGS = 3;\n",
        "constexpr int Q8S_THREADS = 256, Q8S_SEGS = 3;\n"
        "\n"
        "// A kernel of T rows of window and R rows a warp: its form of the int8 dot product and its loops, chosen per kernel on GP100\n"
        "// (idea 63, mikro/kane/q8s_izbira.cu; every choice gives the same bits). The xmad form is about twice as fast alone and holds\n"
        "// more registers: it pays where the kernel stays on its step of blocks an SM (64 / 85 / 128 registers at 256 threads).\n"
        "//   FNX_Q8S_X1, _X2, _X4: bit T set: the kernel of T rows with 1, 2, 4 rows a warp takes xmad;\n"
        "//   FNX_Q8S_C1, _C2, _C4: bit T set: it takes the compiler's own form (looked at first);\n"
        "//   FNX_Q8S_MANY2: bit T set: 2 rows a warp take the loop of the wide windows (weights held, 4 activation rows at a time)\n"
        "//                  at T <= 8 too;\n"
        "//   FNX_Q8S_TC8: bit T set: 4 rows a warp take the activations 8 rows at a time;\n"
        "//   FNX_Q8S_R4_T: matrices of 8192 rows and more take 4 rows a warp from this many rows of window on (2 below);\n"
        "//   FNX_Q8S_TWO_T: matrices of 2048 rows and more: from this many rows of window on two launches, 8 rows and the rest\n"
        "//                  (kernels.cu's q8s_run).\n"
        "#ifndef FNX_Q8S_X1\n"
        "#define FNX_Q8S_X1 " + X1 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_X2\n"
        "#define FNX_Q8S_X2 " + X2 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_X4\n"
        "#define FNX_Q8S_X4 " + X4 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_C1\n"
        "#define FNX_Q8S_C1 " + C1 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_C2\n"
        "#define FNX_Q8S_C2 " + C2 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_C4\n"
        "#define FNX_Q8S_C4 " + C4 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_MANY2\n"
        "#define FNX_Q8S_MANY2 " + MANY2 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_TC8\n"
        "#define FNX_Q8S_TC8 " + TC8 + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_R4_T\n"
        "#define FNX_Q8S_R4_T " + R4_T + "\n"
        "#endif\n"
        "#ifndef FNX_Q8S_TWO_T\n"
        "#define FNX_Q8S_TWO_T " + TWO_T + "\n"
        "#endif\n"
        "constexpr int q8s_x(int T, int R) {\n"
        "    if (((R == 1 ? FNX_Q8S_C1 : R == 2 ? FNX_Q8S_C2 : FNX_Q8S_C4) >> T) & 1u) return 2;\n"
        "    return (((R == 1 ? FNX_Q8S_X1 : R == 2 ? FNX_Q8S_X2 : FNX_Q8S_X4) >> T) & 1u) ? 1 : 0;\n"
        "}\n"
        "// activation rows at a time against the step's held weights; 0: all T rows' activations held, the weights row by row.\n"
        "// q8s_tc0: as it was before the choice (the kernels that are not k_q8s keep it)\n"
        "constexpr int q8s_tc0(int T, int R) { return ((T > 8 && R >= 2) || (R >= 4 && T > 4)) ? 4 : 0; }\n"
        "constexpr int q8s_tc(int T, int R) {\n"
        "    if (R == 4 && ((FNX_Q8S_TC8 >> T) & 1u)) return 8;\n"
        "    if (R == 2 && ((FNX_Q8S_MANY2 >> T) & 1u)) return 4;\n"
        "    return q8s_tc0(T, R);\n"
        "}\n",
    ),
    (
        "template <int T, int Q8S_R, bool SPLIT, bool SEG>\n__device__ __forceinline__ void q8s_block(",
        "// X, TC: the dot product's form and the loop (above); k_q8s takes its kernel's, the other users vmad and the loop as before\n"
        "template <int T, int Q8S_R, bool SPLIT, bool SEG, int X = 0, int TC = q8s_tc0(T, Q8S_R)>\n"
        "__device__ __forceinline__ void q8s_block(",
    ),
    (
        "    if constexpr ((T > 8 && Q8S_R >= 2) || (Q8S_R >= 4 && T > 4)) {\n",
        "    if constexpr (TC > 0) {\n",
    ),
    ("        constexpr int TC = 4;\n", ""),
    ("__dp4a(", "dp4a_x<X>("),
    (
        "template <int T, int Q8S_R, bool SPLIT, bool SEG>\n__global__ void __launch_bounds__(Q8S_THREADS) k_q8s(",
        "template <int T, int Q8S_R, bool SPLIT, bool SEG, int X = q8s_x(T, Q8S_R), int TC = q8s_tc(T, Q8S_R)>\n"
        "__global__ void __launch_bounds__(Q8S_THREADS) k_q8s(",
    ),
    (
        "    q8s_block<T, Q8S_R, SPLIT, SEG>(g, rows, cols, y, tq, td, blockIdx.x);\n",
        "    q8s_block<T, Q8S_R, SPLIT, SEG, X, TC>(g, rows, cols, y, tq, td, blockIdx.x);\n",
    ),
], counts=[1, 1, 1, 1, 8, 1, 1])

edit("hc.cuh", [
    (
        "// 8 int8 weights (uint2) . 8 int8 activations (two ints)\n"
        "__device__ __forceinline__ int dot8i(const uint2 w, const int2 x) { return __dp4a((int) w.y, x.y, __dp4a((int) w.x, x.x, 0)); }\n",
        "// the hc read's int8 dots in the xmad form from FNX_HC_X_T rows of window on (idea 63, mikro/kane/hc_dp4a_test.cu on GP100: the\n"
        "// kernel holds 80 registers either way; under that many rows xmad's longer code costs more than its rows gain)\n"
        "#ifndef FNX_HC_X_T\n"
        "#define FNX_HC_X_T " + HC_X_T + "\n"
        "#endif\n"
        "#ifndef FNX_HC_LOW_F\n"
        "#define FNX_HC_LOW_F " + HC_LOW_F + "\n"
        "#endif\n"
        "constexpr int hc_x(int T) { return T >= FNX_HC_X_T ? 1 : FNX_HC_LOW_F; }\n"
        "// 8 int8 weights (uint2) . 8 int8 activations (two ints)\n"
        "template <int X> __device__ __forceinline__ int dot8i(const uint2 w, const int2 x) {\n"
        "    return dp4a_x<X>((int) w.y, x.y, dp4a_x<X>((int) w.x, x.x, 0));\n"
        "}\n",
    ),
    ("(float) dot8i(u[q], ", "(float) dot8i<hc_x(T)>(u[q], "),
    ("= __dp4a((int) w", "= dp4a_x<hc_x(T)>((int) w"),
], counts=[1, 1, 8])
print("xmad_patch: kernels.cu, dense.cuh, hc.cuh")
