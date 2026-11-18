#!/usr/bin/env python3
# q4x_patch.py SRC_DIR (2026-11-12 night, Unsloth's UD-Q4_K_XL of Flash-Next): the P100 expert kernels (the writing path's row kernel,
# p100_moe_rows.cuh; the prompt path's job kernels, p100_iq_gemm.cuh / p100_moe.cu) learn the four expert types of that file: gate|up
# Q4_K (layer 2: Q5_K), down Q5_1 (layers 2, 4, 30, 46, 47: Q8_0). A run (32 values) becomes 8 words of one value a byte plus its scale
# words; the values go to f16 exactly as 1024 + v (the byte in an f16's mantissa under the exponent of 1024, then 1024 off), then
# d sc q - dmin m (K-quants), d q + m (Q5_1), d q (Q8_0), as ggml's dequantize_row_*. A group: one super-block (K-quants) or four blocks.
# Also FNX_SH_IN=0: Q4_K layers take the row path with the shared expert beside it, not the decode kernels' launch with the shared expert
# inside (engine.cu: an older design for Q4_K files); FNX_PLE_DISK=1: the n-gram (PLE) table stays a mapping of its file (read on demand,
# the page cache keeps the hot rows) instead of 28.8 GB locked in RAM. Each edit must match as counted.
import sys

src = sys.argv[1]


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1)
    open(p, "w").write(s)


HELPERS = r'''
// ---- q4x_patch.py: the Q4 XL expert types. A run's words: [0..7] its 32 values one a byte (v < 1024), [8] d | dmin << 16 (K-quants),
// d | m << 16 (Q5_1) or d (Q8_0), [9] the K-quant sub-block's scale | min << 8 (ggml get_scale_min_k4)
template <> struct RunWords<12> { static constexpr int n = 10; };   // Q4_K
template <> struct RunWords<13> { static constexpr int n = 10; };   // Q5_K
template <> struct RunWords<7> { static constexpr int n = 9; };     // Q5_1
template <> struct RunWords<8> { static constexpr int n = 9; };     // Q8_0
template <int TY> inline constexpr bool kQX = TY == 12 || TY == 13 || TY == 7 || TY == 8;
#define P100_U32(p, off) (P100_U16(p, off) | (P100_U16(p, (off) + 2) << 16))
// bit b of the low nibble of x to bit 4 of byte b
__device__ __forceinline__ uint32_t kq_spread4(uint32_t x) { return ((x & 1u) << 4) | ((x & 2u) << 11) | ((x & 4u) << 18) | ((x & 8u) << 25); }
// run ib of the block (K-quants: its sub-block ib; Q5_1, Q8_0: the block itself) at x
template <int TY>
__device__ __forceinline__ void kq_words(const uint8_t* x, const int ib, uint32_t (&w)[RunWords<TY>::n]) {
    if constexpr (TY == 12 || TY == 13) {
        constexpr int QS = TY == 12 ? 16 : 48;
        const int c = ib >> 1, sh = 4 * (ib & 1);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            uint32_t v = (P100_U32(x, QS + 32 * c + 4 * k) >> sh) & 0x0F0F0F0Fu;
            if constexpr (TY == 13) v |= ((P100_U32(x, 16 + 4 * k) >> ib) & 0x01010101u) << 4;
            w[k] = v;
        }
        w[8] = P100_U32(x, 0);
        const uint8_t* q = x + 4;
        uint32_t s, m;
        if (ib < 4) s = q[ib] & 63, m = q[ib + 4] & 63;
        else s = (q[ib + 4] & 15) | ((q[ib - 4] >> 6) << 4), m = (q[ib + 4] >> 4) | ((q[ib] >> 6) << 4);
        w[9] = s | (m << 8);
    } else if constexpr (TY == 7) {
        const uint32_t qh = P100_U32(x, 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = P100_U32(x, 8 + 4 * k);
            w[k] = (q & 0x0F0F0F0Fu) | kq_spread4((qh >> (4 * k)) & 15u);
            w[k + 4] = ((q >> 4) & 0x0F0F0F0Fu) | kq_spread4((qh >> (16 + 4 * k)) & 15u);
        }
        w[8] = P100_U32(x, 0);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = P100_U32(x, 2 + 4 * k);
        w[8] = P100_U16(x, 0);
    }
}
// bytes b0, b1 of w (selector 0x4140: bytes 0 and 1, 0x4342: bytes 2 and 3) as the f16 pair (v0, v1), exact for v < 1024
__device__ __forceinline__ half2 kq_h2(uint32_t w, uint32_t sel) {
    return __hsub2(as_half2(__byte_perm(w, 0u, sel) | 0x64006400u), as_half2(0x64006400u));
}
template <int TY>
__device__ __forceinline__ void kq_finish(const uint32_t (&w)[RunWords<TY>::n], uint4* __restrict__ out) {
    if constexpr (TY == 12 || TY == 13) {   // d sc q - dmin m
        const half2 a = dup_f(h2f(uint16_t(w[8])) * float(w[9] & 255u)), b = dup_f(-h2f(uint16_t(w[8] >> 16)) * float(w[9] >> 8));
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[j] = make_uint4(as_u32(__hfma2(kq_h2(w[2 * j], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j], 0x4342u), a, b)),
                                as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4342u), a, b)));
    } else if constexpr (TY == 7) {   // d q + m
        const half2 a = dup_u16(w[8] & 0xffffu), b = dup_u16(w[8] >> 16);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            out[j] = make_uint4(as_u32(__hfma2(kq_h2(w[2 * j], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j], 0x4342u), a, b)),
                                as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4140u), a, b)), as_u32(__hfma2(kq_h2(w[2 * j + 1], 0x4342u), a, b)));
    } else {   // d q: an int8 plus 128 as above, then 1152 off
        const half2 a = dup_u16(w[8]), o = as_half2(0x64806480u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t u0 = w[2 * j] ^ 0x80808080u, u1 = w[2 * j + 1] ^ 0x80808080u;
            const half2 q0 = __hsub2(as_half2(__byte_perm(u0, 0u, 0x4140u) | 0x64006400u), o), q1 = __hsub2(as_half2(__byte_perm(u0, 0u, 0x4342u) | 0x64006400u), o);
            const half2 q2 = __hsub2(as_half2(__byte_perm(u1, 0u, 0x4140u) | 0x64006400u), o), q3 = __hsub2(as_half2(__byte_perm(u1, 0u, 0x4342u) | 0x64006400u), o);
            out[j] = make_uint4(as_u32(__hmul2(q0, a)), as_u32(__hmul2(q1, a)), as_u32(__hmul2(q2, a)), as_u32(__hmul2(q3, a)));
        }
    }
}
'''

edit("p100_iq_gemm.cuh", [
    ("template <> struct RunWords<29> { static constexpr int n = 3; };   // IQ1_M: the codes, qh | the run's two scales << 16, d\n",
     "template <> struct RunWords<29> { static constexpr int n = 3; };   // IQ1_M: the codes, qh | the run's two scales << 16, d\n" + HELPERS, 1),
    # run_load (global): the row at `row`, run `run`
    ("    if constexpr (TY == 18) {   // IQ3_XXS\n        const block_iq3_xxs* x = reinterpret_cast<const block_iq3_xxs*>(row) + (run >> 3);\n",
     "    if constexpr (TY == 12 || TY == 13) {   // q4x_patch.py\n        kq_words<TY>(row + (run >> 3) * (TY == 12 ? 144 : 176), run & 7, w);\n"
     "    } else if constexpr (TY == 7 || TY == 8) {\n        kq_words<TY>(row + run * (TY == 7 ? 24 : 34), 0, w);\n"
     "    } else if constexpr (TY == 18) {   // IQ3_XXS\n        const block_iq3_xxs* x = reinterpret_cast<const block_iq3_xxs*>(row) + (run >> 3);\n", 1),
    # run_finish
    ("    if constexpr (TY == 18) {\n        const uint32_t aux = w[4];\n        const half2 dd = dup_f(h2f(uint16_t(w[5])) * float(1 + 2 * (aux >> 28)) * 0.25f);\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n            const uint4 sg = __ldg(t.sign7",
     "    if constexpr (kQX<TY>) {\n        kq_finish<TY>(w, out);   // q4x_patch.py\n    } else if constexpr (TY == 18) {\n        const uint32_t aux = w[4];\n        const half2 dd = dup_f(h2f(uint16_t(w[5])) * float(1 + 2 * (aux >> 28)) * 0.25f);\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n            const uint4 sg = __ldg(t.sign7", 1),
    # the group constants
    ("template <int TY> inline constexpr int kSU = TY == 23 ? 10 :",
     "template <int TY> inline constexpr int kSU = TY == 12 ? 10 : TY == 13 ? 12 : TY == 7 ? 7 : TY == 8 ? 10 : TY == 23 ? 10 :", 1),
    ("template <int TY> inline constexpr int kRunsGr = TY == 20 || TY == 2 || TY == 42 ? 4 : 8;",
     "template <int TY> inline constexpr int kRunsGr = TY == 20 || TY == 2 || TY == 42 || TY == 7 || TY == 8 ? 4 : 8;", 1),
    ("template <int TY> inline constexpr int kGrBytes = TY == 18 ? 98 :",
     "template <int TY> inline constexpr int kGrBytes = TY == 12 ? 144 : TY == 13 ? 176 : TY == 7 ? 96 : TY == 8 ? 136 : TY == 18 ? 98 :", 1),
    # run_load_s: the group at p (shared memory), run `run`
    ("    if constexpr (TY == 18) {\n        const block_iq3_xxs* x = reinterpret_cast<const block_iq3_xxs*>(p);\n",
     "    if constexpr (TY == 12 || TY == 13) {   // q4x_patch.py: a group is one super-block\n        kq_words<TY>(p, run & 7, w);\n"
     "    } else if constexpr (TY == 7 || TY == 8) {   // four blocks\n        kq_words<TY>(p + (run & 3) * (TY == 7 ? 24 : 34), 0, w);\n"
     "    } else if constexpr (TY == 18) {\n        const block_iq3_xxs* x = reinterpret_cast<const block_iq3_xxs*>(p);\n", 1),
    # the tables: none
    ("template <> struct TabU<29> { static constexpr int n = 8; };     // IQ1_M: its 64 KB of grids stay in global memory\n",
     "template <> struct TabU<29> { static constexpr int n = 8; };     // IQ1_M: its 64 KB of grids stay in global memory\n"
     "// q4x_patch.py: no tables, and none of their room either: Q8_0's 256-lane job kernels (down's mode 5) need exactly Pascal's 48 KiB\n"
     "template <> struct TabU<12> { static constexpr int n = 0; };\n"
     "template <> struct TabU<13> { static constexpr int n = 0; };\n"
     "template <> struct TabU<7> { static constexpr int n = 0; };\n"
     "template <> struct TabU<8> { static constexpr int n = 0; };\n", 1),
    ("    if constexpr (TY == 18) {\n        for (int i = tid; i < 128; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.g3xxs) + i);\n",
     "    if constexpr (kQX<TY>) {   // q4x_patch.py: no tables\n"
     "    } else if constexpr (TY == 18) {\n        for (int i = tid; i < 128; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.g3xxs) + i);\n", 1),
    # finish_s
    ("    if constexpr (TY == 18) {\n        const uint2* grid = reinterpret_cast<const uint2*>(tab);\n        const uint4* sign = tab + 128;\n        const uint32_t aux = w[4];\n",
     "    if constexpr (kQX<TY>) {\n        kq_finish<TY>(w, out);   // q4x_patch.py\n"
     "    } else if constexpr (TY == 18) {\n        const uint2* grid = reinterpret_cast<const uint2*>(tab);\n        const uint4* sign = tab + 128;\n        const uint32_t aux = w[4];\n", 1),
])

ROWS_WORDS = r'''    if constexpr (TY == 12 || TY == 13) {   // q4x_patch.py, Q4_K / Q5_K: d, dmin, scales[12], (Q5_K: qh[32]), qs[128]; run IB = sub-block IB
        constexpr int C = IB >> 1, SH = 4 * (IB & 1), QS = TY == 12 ? 16 : 48;
#define P100_KQW(k) \
        w[k] = ((g16<QS + 32 * C + 4 * k>(W) | (g16<QS + 32 * C + 4 * k + 2>(W) << 16)) >> SH) & 0x0F0F0F0Fu; \
        if constexpr (TY == 13) w[k] |= (((g16<16 + 4 * k>(W) | (g16<18 + 4 * k>(W) << 16)) >> IB) & 0x01010101u) << 4;
        P100_KQW(0) P100_KQW(1) P100_KQW(2) P100_KQW(3) P100_KQW(4) P100_KQW(5) P100_KQW(6) P100_KQW(7)
#undef P100_KQW
        w[8] = g16<0>(W) | (g16<2>(W) << 16);
        if constexpr (IB < 4) w[9] = (g8<4 + IB>(W) & 63u) | ((g8<8 + IB>(W) & 63u) << 8);
        else w[9] = ((g8<8 + IB>(W) & 15u) | ((g8<IB>(W) >> 6) << 4)) | (((g8<8 + IB>(W) >> 4) | ((g8<4 + IB>(W) >> 6) << 4)) << 8);
    } else if constexpr (TY == 7) {   // Q5_1: blocks of d, m, qh, qs[16]
        constexpr int B = 24 * IB;
        const uint32_t qh = g16<B + 4>(W) | (g16<B + 6>(W) << 16);
#define P100_K51(k) { \
            const uint32_t q = g16<B + 8 + 4 * k>(W) | (g16<B + 10 + 4 * k>(W) << 16); \
            w[k] = (q & 0x0F0F0F0Fu) | kq_spread4((qh >> (4 * k)) & 15u); \
            w[k + 4] = ((q >> 4) & 0x0F0F0F0Fu) | kq_spread4((qh >> (16 + 4 * k)) & 15u); }
        P100_K51(0) P100_K51(1) P100_K51(2) P100_K51(3)
#undef P100_K51
        w[8] = g16<B>(W) | (g16<B + 2>(W) << 16);
    } else if constexpr (TY == 8) {   // Q8_0: blocks of d, qs[32]
        constexpr int B = 34 * IB;
#define P100_K80(k) w[k] = g16<B + 2 + 4 * k>(W) | (g16<B + 4 + 4 * k>(W) << 16);
        P100_K80(0) P100_K80(1) P100_K80(2) P100_K80(3) P100_K80(4) P100_K80(5) P100_K80(6) P100_K80(7)
#undef P100_K80
        w[8] = g16<B>(W);
    } else if constexpr (TY == 18) {   // IQ3_XXS: d, qs[96] (codes, then the signs and scale word of each run)
'''

edit("p100_moe_rows.cuh", [
    ("    if constexpr (TY == 18) {   // IQ3_XXS: d, qs[96] (codes, then the signs and scale word of each run)\n", ROWS_WORDS, 1),
    ("template <int TY> inline constexpr int kRowG = TY == 20 || TY == 2 ? 1 : FNX_ROW_G;",
     "template <int TY> inline constexpr int kRowG = TY == 20 || TY == 2 || TY == 7 || TY == 8 ? 1 : FNX_ROW_G;", 1),
    ("        case T_IQ1_M: rows_launch<29>(a, T, njobs_cap, s); break;\n        default: throw std::runtime_error(\"p100 moe rows: expert type \"",
     "        case T_IQ1_M: rows_launch<29>(a, T, njobs_cap, s); break;\n"
     "        case T_Q4_K: rows_launch<12>(a, T, njobs_cap, s); break;   // q4x_patch.py\n"
     "        case T_Q5_K: rows_launch<13>(a, T, njobs_cap, s); break;\n"
     "        case T_Q5_1: rows_launch<7>(a, T, njobs_cap, s); break;\n"
     "        case T_Q8_0: rows_launch<8>(a, T, njobs_cap, s); break;\n"
     "        default: throw std::runtime_error(\"p100 moe rows: expert type \"", 1),
])

edit("p100_moe.cu", [
    ("        case T_IQ4_NL: P100_JOB(20); break;\n",
     "        case T_IQ4_NL: P100_JOB(20); break;\n"
     "        case T_Q4_K: P100_JOB(12); break;   // q4x_patch.py\n"
     "        case T_Q5_K: P100_JOB(13); break;\n"
     "        case T_Q5_1: P100_JOB(7); break;\n"
     "        case T_Q8_0: P100_JOB(8); break;\n", 1),
])

edit("engine.cu", [
    ("    const bool sh_in = L.gu_e.type == T_Q4_K && L.sh_gu.s.q && L.sh_down.s.q && ffs % 32 == 0 && ffs <= SH_MAXF;\n",
     "    // FNX_SH_IN=0 (q4x_patch.py): Q4_K layers take the row path like the others (the P100 row kernel knows Q4_K now)\n"
     "    static const bool sh_in_q4k = env_int(\"FNX_SH_IN\", 1) != 0;\n"
     "    const bool sh_in = sh_in_q4k && L.gu_e.type == T_Q4_K && L.sh_gu.s.q && L.sh_down.s.q && ffs % 32 == 0 && ffs <= SH_MAXF;\n", 1),
    ("lyr + 1 < NL && w.L[(size_t) lyr + 1].gu_e.type != T_Q4_K) {",
     "lyr + 1 < NL && (!sh_in_q4k || w.L[(size_t) lyr + 1].gu_e.type != T_Q4_K)) {", 1),
])

edit("model.cpp", [
    ("        for (const Tensor* t : {m.tok_embd, m.ple_table}) lock_in_ram(m.g, *t);\n",
     "        // FNX_PLE_DISK=1 (q4x_patch.py): the n-gram table stays its file's mapping, read on demand (the page cache keeps the hot rows)\n"
     "        static const bool ple_disk = env_int(\"FNX_PLE_DISK\", 0) != 0;\n"
     "        for (const Tensor* t : {m.tok_embd, m.ple_table})\n"
     "            if (!(ple_disk && t == m.ple_table)) lock_in_ram(m.g, *t);\n"
     "        if (ple_disk) std::fprintf(stderr, \"fnx: FNX_PLE_DISK: %s (%.1f GB) read from its file on demand\\n\", m.ple_table->name.c_str(), m.ple_table->nbytes / 1e9);\n", 1),
    ("    for (const Tensor* t : {m.tok_embd, m.ple_table}) {\n        if (!t || !t->data) continue;\n",
     "    for (const Tensor* t : {m.tok_embd, m.ple_table}) {\n        if (!t || !t->data) continue;\n"
     "        if (t == m.ple_table && env_int(\"FNX_PLE_DISK\", 0) != 0) continue;   // q4x_patch.py: not read in whole\n", 1),
])
print("q4x_patch: p100_iq_gemm.cuh, p100_moe_rows.cuh, p100_moe.cu, engine.cu, model.cpp")
