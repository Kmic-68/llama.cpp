#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "active-tokens.cuh"

// Flash attention for pre-Volta GPUs over a q4_0 K/V cache at head size 256, for the few-token
// batches of decode and speculative verify (one to five tokens, one GQA group of six heads).
//
// The tile and vec kernels are issue-bound here, not memory-bound: at 262144 context they read
// the 151 MB q4_0 cache at ~125 GB/s, a fifth of what the card sustains, and they spend their
// instructions on fp16 conversions and shared-memory round trips. This kernel is built around
// the two things that actually cost:
//
//   * Dequantization is 2 instructions per value, straight into fp32. PRMT places a nibble into
//     the low mantissa bits of 2^23 (0x4B0000nn), and one FADD of -(2^23 + 8) leaves the exact
//     value n - 8. Every K/V byte is read once, into registers.
//   * Every dequantized value is reused by the query rows of the block: the whole GQA group of
//     every token, so a 5-token verify amortises one dequant over up to 30 FMAs.
//
// Accuracy is at least that of the existing kernels, not traded: Q stays fp32 (the other kernels
// round it to fp16), and products and sums are fp32. Where the q4_0 block scale is folded into a
// dequantized value first, the product is exact: a 4-bit integer times an fp16 scale fits in the
// fp32 mantissa.
//
// Softmax uses a lazily updated running max: a block-wide max reduction per chunk would cost as
// much as a chunk's arithmetic for large R, so the max only moves when some score exceeds it by
// more than 2^8. P can then reach 256, which fp32 accumulation absorbs without loss.

#define FATTN_Q4P_D 256

// R = query rows per block. Rows are split into RG groups of RQ; a thread works on one group.
// Defaults from a sweep at kv=262144 (us per call, tile kernel -> this): R=6 1206 -> 1007,
// R=12 1763 -> 1536, R=18 3060 -> 2196, R=24 3060 -> 2800, R=30 4880 -> 4299. OPTLOG attempt 181.
// Per-width tuning knobs. Q4P_TUNE_R selects which R a Q4P_* override applies to; a sweep writes
// them into fattn-q4p-tune.h, which does not exist in a normal build.
#if __has_include("fattn-q4p-tune.h")
#include "fattn-q4p-tune.h"
#endif
#ifndef Q4P_TUNE_R
#define Q4P_TUNE_R 0
#endif
#define Q4P_KNOB(name, dflt) ((R == Q4P_TUNE_R && (Q4P_##name) > 0) ? (Q4P_##name) : (dflt))
#ifndef Q4P_RG
#define Q4P_RG 0
#endif
#ifndef Q4P_NSPLIT
#define Q4P_NSPLIT 0
#endif
#ifndef Q4P_PT
#define Q4P_PT 0
#endif
#ifndef Q4P_DPT
#define Q4P_DPT 0
#endif
#ifndef Q4P_PF
#define Q4P_PF 0
#endif
#ifndef Q4P_ROLL
#define Q4P_ROLL 0
#endif
#ifndef Q4P_KPF
#define Q4P_KPF 0
#endif
#ifndef Q4P_MINB
#define Q4P_MINB 1
#endif

// QK in fp16: products on the HFMA2 pipe (two KV positions per instruction), one chain per 32-dim
// q4_0 block folded into fp32 through the block scale. 1 = on, 2 = off.
#ifndef Q4P_H16QK
#define Q4P_H16QK 1
#endif
// PV in fp16: P published as half2 (p,p), V nibbles exact in fp16 then scaled by the block scale,
// half2 chains over dimension pairs folded into fp32 every Q4P_PVF positions. Forces DPT 4
// (60 fp32 + 30 half2 accumulators at 15 rows, where DPT 8 fp32 needed 120). 1 = on, 2 = off.
#ifndef Q4P_H16PV
#define Q4P_H16PV 1
#endif
#ifndef Q4P_PVF
#define Q4P_PVF 32
#endif
// PV at DPT 8: the 8 position groups of a dimension group sit in one warp, and each chunk's fp16
// partials are converted and reduce-scattered across them by shuffles, so a thread keeps RQ fp32
// accumulators (one output dim) instead of RQ*DPT. Q4P_RSON 2 turns it off everywhere; per width
// (Q4P_RS 1 on, 2 off) it is on at 15 and 18 rows (kv 262144: 2520 -> 2430 us, 2290 -> 1676), off at
// 12 (1071 -> 1087). At 6 rows it is on with two blocks per SM (805 -> 741; alone it lost, 805 -> 827).
#ifndef Q4P_RSON
#define Q4P_RSON 1
#endif
#ifndef Q4P_RS
#define Q4P_RS 0
#endif

// no barrier at the chunk end (1 on, 2 off)
#ifndef Q4P_NOEB
#define Q4P_NOEB 1
#endif
// P published by every split, one position each (1 on, 2 off)
#ifndef Q4P_PSPL
#define Q4P_PSPL 1
#endif
// fp16 -> fp32 of the fp16 chains: 1 = conversion instruction (exact, subnormals kept),
// 2 = integer conversion below
#ifndef Q4P_CVT
#define Q4P_CVT 1
#endif
// fp32 value * 2^-112 of the half in the high / low lane (integer conversion; subnormals flush)
static __device__ __forceinline__ float fattn_q4p_hi_f(const uint32_t x) {
    return __int_as_float((int32_t(x) >> 3) & 0x8FFFE000);
}
static __device__ __forceinline__ float fattn_q4p_lo_f(const uint32_t x) {
    return __int_as_float((int32_t(x << 16) >> 3) & 0x8FFFE000);
}

// OLD = the configuration before the PV reduce-scatter (GGML_CUDA_Q4P_OLD=1), kept for in-model A/B
// KQ8: K is q8_0 (34-byte blocks of 32 signed bytes) instead of q4_0; V stays q4_0. 4 splits per position keep the
// K words in registers. QK in fp32: an fp16 variant (bytes as exact q/16 halves) measured the same, since at depth
// this kernel's time follows the K/V bytes it loads (q8_0 K: +44% bytes, +43% time at 262144 keys, at ~230 GB/s:
// load latency per word, not DRAM bandwidth).
template <int R, bool OLD = false, bool KQ8 = false>
struct fattn_q4p_cfg {
    static constexpr int NT     = 256;                    // threads per block
    static constexpr int RG     = Q4P_KNOB(RG, 1);        // row groups
    static constexpr int RQ     = R / RG;                 // rows per group
    static constexpr int RQP    = (RQ + 3) & ~3;          // padded for float4 access
    // QK: NSPLIT adjacent threads share one KV position, each over DPS dimensions; PT positions each.
// q8_0-K shape (sweep on 2x P100, kv 262144: NSPLIT 2 / PT 4 exceed shared memory at 5 tokens, NSPLIT 2 PT 1 is
// slower at every width)
#ifndef Q4P_K8_NSPLIT
#define Q4P_K8_NSPLIT 4
#endif
#ifndef Q4P_K8_PT
#define Q4P_K8_PT 2
#endif
    static constexpr int NSPLIT = KQ8 ? Q4P_K8_NSPLIT : Q4P_KNOB(NSPLIT, R <= 18 ? 2 : 4);
    static constexpr int PT     = KQ8 ? Q4P_K8_PT : Q4P_KNOB(PT, R <= 24 ? 2 : 1);
    static constexpr int NQP    = NT/(NSPLIT*RG);         // position slots per pass
    static constexpr int C      = NQP*PT;                 // KV positions per chunk
    static constexpr int BPS    = 8/NSPLIT;               // K blocks per split
    static constexpr int KBB    = KQ8 ? 34 : 18;          // bytes per K block
    static constexpr int DPS    = FATTN_Q4P_D/NSPLIT;     // dimensions per split
    static constexpr int QREG   = DPS*RQP + 4;            // one (split, group) region of Q_s; +4 spreads banks
    // scale per 32-dim partial sum (fewer multiplies) or per value (fewer registers)
#ifndef Q4P_BLOCKT
#define Q4P_BLOCKT 0
#endif
    // also at 12 and 15 rows (kv 262144: 1367 -> 1285 us, 3055 -> 2973); 18 rows lose (1926 -> 1975). OPTLOG 219.
    static constexpr bool BLOCK_T = Q4P_KNOB(BLOCKT, (RQ <= 6 || RQ == 12 || RQ == 15) ? 1 : 2) == 1;
    // PV: a thread owns DPT output dimensions for the RQ rows of its group, over every NPG-th position.
    // 15 and 18 rows: 8 (15: 3403 -> 3074 us, 18: 1931 -> 1873 at kv 262144; 24 rows: 2488 -> 4939). OPTLOG 204.
    static constexpr int DPT    = Q4P_KNOB(DPT, R <= 18 ? 8 : 4);
    static constexpr int NDG    = FATTN_Q4P_D/DPT;
    static constexpr int NPG    = NT/(NDG*RG);
    // Fold the V block scale into P (RQ multiplies per position) or into V (DPT multiplies).
    static constexpr bool FOLD  = RQ <= DPT;
    // L2 prefetch of the chunk's V rows and the next chunk's K rows (1 = on, 2 = off)
    // (off at 15 rows since the PV reduce-scatter: 2476 -> 2356 us at kv 262144)
    static constexpr bool PREFETCH = Q4P_KNOB(PF, (!OLD && R == 15) ? 2 : 1) == 1;
    // QK over the BPS blocks of a split: rolled (1) keeps the kernel inside the instruction cache,
    // unrolled (2) holds the whole split's words in registers
    static constexpr bool ROLL = !KQ8 && Q4P_KNOB(ROLL, 2) == 1;
    // load the next chunk's K words during this chunk's PV phase (needs NW*PT spare registers;
    // R=24 has none)
    static constexpr bool KPF  = !ROLL && !KQ8 && Q4P_KNOB(KPF, R == 24 ? 2 : 1) == 1;
    static constexpr bool H16QK = !KQ8 && Q4P_H16QK == 1 && PT == 2 && BLOCK_T && !ROLL;
    // fp16 QK with two rows per Q_s word (HFMA2 takes each half broadcast): one LDS.128 per 8 rows
#ifndef Q4P_QPK
#define Q4P_QPK 1
#endif
    static constexpr bool QPK   = H16QK && Q4P_QPK == 1 && RQP % 8 == 0;
    static constexpr bool H16PV = Q4P_H16PV == 1 && (DPT == 4 || DPT == 8);
    static constexpr bool RS    = !OLD && Q4P_RSON == 1 && Q4P_KNOB(RS, (R == 6 || R == 15 || R == 18) ? 1 : 2) == 1 &&
                                  H16PV && DPT == 8 && RG == 1 && NPG == 8 && C/NPG <= Q4P_PVF;
    // blocks per SM: 6 rows with RS fit 128 registers without spilling (kv 262144: 816 -> 741 us)
    static constexpr int  MINB  = R == Q4P_TUNE_R ? Q4P_MINB : ((RS && R == 6) ? 2 : 1);
    // P_s floats per position; with RS a warp reads 8 positions at once, so PSTR/4 is kept odd
    // fp16 P with two rows per word (PV takes each half broadcast): half the P_s reads
#ifndef Q4P_PPK
#define Q4P_PPK 1
#endif
    static constexpr bool PPK   = Q4P_PPK == 1 && RS && RQP % 8 == 0;
    static constexpr int PW     = PPK ? RG*RQP/2 : RG*RQP; // P_s words per position
    static constexpr int PSTR   = PW + ((RS && (PW/4) % 2 == 0) ? 4 : 0);
    static constexpr int NACC   = RS ? 1 : DPT;           // fp32 accumulators per row

    static_assert(R % RG == 0, "bad RG");
    static_assert(NT % (NDG*RG) == 0 && NPG >= 1, "bad DPT");
    static_assert(NPG <= 8, "red_s holds 8 rows");
    static_assert(C % NPG == 0, "bad NPG");
    static_assert(NSPLIT*RG*QREG >= R*FATTN_Q4P_D, "Q_s is reused for the output reduction");
};

// exact (n - 8) for the nibble in byte e of nib (whose bytes are all < 16)
static __device__ __forceinline__ float fattn_q4p_nib(const uint32_t nib, const int e) {
    return __int_as_float(__byte_perm(nib, 0x4B000000u, 0x7440u | e)) - 8388616.0f;
}

static __device__ __forceinline__ uint32_t fattn_q4p_ld16(const char * p) {
    return __ldg((const unsigned short *) p);
}

static __device__ __forceinline__ void fattn_q4p_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

// dimension of PV element j for dim-group dg (see fattn_q4p_cfg::DPT)
template <int DPT>
static __device__ __forceinline__ int fattn_q4p_pv_dim(const int dg, const int j) {
    if constexpr (DPT == 8) {
        const int b = dg >> 2, k = dg & 3;
        return 32*b + 4*k + (j & 3) + 16*(j >> 2);
    } else if constexpr (DPT == 4) {
        const int wi = dg >> 1, hh = dg & 1;
        return 32*(wi >> 2) + 4*(wi & 3) + j + 16*hh;
    } else {
        static_assert(DPT == 2, "bad DPT");
        const int wi = dg >> 2, hh = dg & 1, ep = (dg >> 1) & 1;
        return 32*(wi >> 2) + 4*(wi & 3) + 2*ep + j + 16*hh;
    }
}

template <int ncols1, int ncols2, bool OLD, bool KQ8 = false>
__launch_bounds__(256, (fattn_q4p_cfg<ncols1*ncols2, OLD, KQ8>::MINB))
static __global__ void flash_attn_ext_q4p(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
    ggml_cuda_pdl_lc();
#if defined(FLASH_ATTN_AVAILABLE) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    constexpr int D = FATTN_Q4P_D;
    constexpr int R = ncols1*ncols2;
    using cfg = fattn_q4p_cfg<R, OLD, KQ8>;
    constexpr int KBB = cfg::KBB;
    constexpr int NT = cfg::NT, RG = cfg::RG, RQ = cfg::RQ, RQP = cfg::RQP, C = cfg::C, PT = cfg::PT;
    constexpr int NSPLIT = cfg::NSPLIT, NQP = cfg::NQP, BPS = cfg::BPS, DPS = cfg::DPS, QREG = cfg::QREG;
    constexpr int DPT = cfg::DPT, NDG = cfg::NDG, NPG = cfg::NPG, PSTR = cfg::PSTR, NACC = cfg::NACC;
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr float LN2   = 0.6931471805599453f;

    const int tid  = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;

    const int ic0      = blockIdx.x*ncols1;
    const int sequence = blockIdx.z / (ne02/ncols2);
    const int head0    = blockIdx.z*ncols2 - sequence*ne02;
    const int gqa      = ne02/ne12;
    const int n_tok    = int(ne01.z);

    const char * Kb = K + nb13*sequence + int64_t(nb12)*(head0/gqa);
    const char * Vb = V + nb23*sequence + int64_t(nb22)*(head0/gqa);
    const char * Mb = mask ? mask + nb33*(sequence % ne33) + int64_t(nb31)*ic0 : nullptr;

    __shared__ __align__(16) float Q_s[NSPLIT*RG*QREG];  // Q, then the output reduction
    __shared__ __align__(16) float P_s[C*PSTR];
    __shared__ float m_s[R];                              // running max, log2 units
    __shared__ float a_s[R];                              // rescale factor
    __shared__ float red_s[8][R];
    constexpr int LRP  = R <= 16 ? 16 : 32;   // rows (padded) of the parallel denominator sum
    constexpr int LPPS = C / (NT/LRP);        // positions per thread in it
    static_assert(C % (NT/LRP) == 0, "bad denominator split");
    __shared__ float lsum_s[NT/WARP_SIZE][LRP];

    ggml_cuda_pdl_sync();

    // Q, pre-scaled into log2 units so the softmax is a bare exp2. Layout: region (split, group),
    // then [dim][row] with rows padded to RQP.
    const float qscale = scale*LOG2E;
    for (int idx = tid; idx < RG*RQP*D; idx += NT) {
        const int rgi = idx / (RQP*D), rem = idx % (RQP*D);
        const int rl  = rem / D, dim = rem % D;
        const int r   = rgi*RQ + rl;
        const int t = r / ncols2, g = r % ncols2;
        float q = 0.0f;
        if (rl < RQ && ic0 + t < n_tok) {
            q = ((const float *) (Q + nb03*sequence + int64_t(nb02)*(head0 + g) + int64_t(nb01)*(ic0 + t)))[dim]*qscale;
        }
        if constexpr (cfg::QPK) {
            ((half *) (Q_s + ((dim / DPS)*RG + rgi)*QREG + (dim % DPS)*(RQP/2)))[rl] = __float2half_rn(q);
        } else if constexpr (cfg::H16QK) {
            const half2 qq = __float2half2_rn(q);
            Q_s[((dim / DPS)*RG + rgi)*QREG + (dim % DPS)*RQP + rl] = __int_as_float(*(const int *) &qq);
        } else {
            Q_s[((dim / DPS)*RG + rgi)*QREG + (dim % DPS)*RQP + rl] = q;
        }
    }
    if (tid < R) {
        m_s[tid] = -1e30f;
    }

    // QK role: split (fastest), row group, position slot
    const int qsplit = tid % NSPLIT;
    const int qgrp   = (tid / NSPLIT) % RG;
    const int qslot  = tid / (NSPLIT*RG);
    const float * Qt = Q_s + (qsplit*RG + qgrp)*QREG;

    // PV role: dim group (fastest), row group, position group
    // (RS: position group fastest, so a warp holds all 8 groups of 4 dim groups)
    const int dg   = cfg::RS ? tid / NPG : tid % NDG;
    const int vgrp = cfg::RS ? 0 : (tid / NDG) % RG;
    const int pg   = cfg::RS ? tid % NPG : tid / (NDG*RG);

    float acc[RQ][NACC];   // RS: dim fattn_q4p_pv_dim(dg, pg) of each row
#pragma unroll
    for (int r = 0; r < RQ; ++r) {
#pragma unroll
        for (int j = 0; j < NACC; ++j) {
            acc[r][j] = 0.0f;
        }
    }
    // RS: fold_rs converts (exact) and scales each of a chunk's fp16 PV partials, then halves the
    // set 3 times across the position groups (lane bits 2, 1, 0): lane pg ends with element j = pg
    // of every row, added to its fp32 accumulators
    auto fold_rs = [&](auto & acc2s) {
        if constexpr (cfg::RS) {
            constexpr int NH = DPT/2;
            float x[DPT][RQ];
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
#pragma unroll
                for (int h = 0; h < NH; ++h) {
                    const uint32_t ab = *(const uint32_t *) &acc2s[r][h];
                    if constexpr (Q4P_CVT == 1) {
                        x[2*h + 0][r] = __low2float(acc2s[r][h]);
                        x[2*h + 1][r] = __high2float(acc2s[r][h]);
                    } else {
                        x[2*h + 0][r] = 0x1p112f*fattn_q4p_lo_f(ab);
                        x[2*h + 1][r] = 0x1p112f*fattn_q4p_hi_f(ab);
                    }
                }
            }
            // x[i] is element i ^ pg (see rs_sh): the partner's x[i + 4] is the same element as x[i]
            float y4[4][RQ];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
                    y4[j][r] = x[j][r] + __shfl_xor_sync(0xFFFFFFFF, x[j + 4][r], 4, WARP_SIZE);
                }
            }
            float y2[2][RQ];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
                    y2[j][r] = y4[j][r] + __shfl_xor_sync(0xFFFFFFFF, y4[j + 2][r], 2, WARP_SIZE);
                }
            }
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
                acc[r][0] += y2[0][r] + __shfl_xor_sync(0xFFFFFFFF, y2[1][r], 1, WARP_SIZE);
            }
        }
    };
    // RS lane permutation: pg bit 2 swaps the low/high nibbles, bit 1 the byte pairs, bit 0 the
    // bytes within a pair, so this lane's PV element i is element i ^ pg of the plain order
    const int      rs_sh = 4*((pg >> 2) & 1);
    const uint32_t rs_e0 = ((pg >> 1) & 1)*2 + (pg & 1), rs_e1 = rs_e0 ^ 1;
    const uint32_t rs_sa = 0x4040u | rs_e0 | (rs_e1 << 8);
    const uint32_t rs_sb = 0x4040u | (rs_e0 ^ 2) | ((rs_e1 ^ 2) << 8);
    // softmax denominator of row tid, for tid < R: summed once per chunk from P_s, so the PV
    // loop neither spends registers on it nor diverges to add it
    float l_row = 0.0f;
    auto add_lsum = [&]() {
        if (tid < R) {
            float ls = 0.0f;
#pragma unroll
            for (int w = 0; w < NT/WARP_SIZE; ++w) {
                ls += lsum_s[w][tid];
            }
            l_row += ls;
        }
    };
    // No barrier at the chunk end: the next chunk's first barrier (the lazy-max vote) already orders
    // this chunk's P_s reads before the next publish, so warps run ahead into the next QK phase
    constexpr bool NOEB = Q4P_NOEB == 1;

    // this thread's V word: block b, word k of each row
    const int pv_d0 = fattn_q4p_pv_dim<DPT>(dg, 0);
    const int pv_b  = pv_d0 / 32;
    const int pv_k  = (pv_d0 % 16) / 4;
    const int pv_o  = 18*pv_b + 2 + 4*pv_k;           // byte offset of the qs word in a row
    // the word is 2-byte aligned for even blocks: read the two aligned words around it and PRMT,
    // with the selector in a register so lanes on even and odd blocks do not diverge
    const int      pv_oa  = pv_o & ~3;
    const uint32_t pv_sel = (pv_o & 2) ? 0x5432u : 0x3210u;

    __syncthreads();

    constexpr int NW = cfg::ROLL ? 1 : KBB*BPS/4; // aligned words per split of a row
    uint32_t un[PT][NW]; // with KPF: the K words of the chunk about to start
    // The mask halves of a chunk's positions, one per token (RG == 1: a thread's rows cover the
    // tokens in order). Loaded with the K words, ahead of the QK math: loaded where they are added,
    // after it, their latency sat in front of the chunk's first barrier.
    constexpr bool MPRE = RG == 1;
    uint32_t mraw[MPRE ? PT : 1][MPRE ? ncols1 : 1];
    auto load_mask = [&](const int kc) {
        if constexpr (MPRE) {
#pragma unroll
            for (int pt = 0; pt < PT; ++pt) {
                const int pos = kc + qslot + NQP*pt;
#pragma unroll
                for (int t = 0; t < ncols1; ++t) {
                    mraw[pt][t] = Mb && pos < ne11 && ic0 + t < n_tok ?
                        __ldg((const unsigned short *) (Mb + int64_t(nb31)*t) + pos) : 0;
                }
            }
        }
    };
    auto load_k = [&](const int kc) {
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
            const int pos = kc + qslot + NQP*pt;
            const uint32_t * sp = (const uint32_t *) (Kb + int64_t(pos < ne11 ? pos : 0)*nb11 + qsplit*(KBB*BPS));
#pragma unroll
            for (int j = 0; j < NW; ++j) {
                un[pt][j] = __ldg(sp + j);
            }
        }
        load_mask(kc);
    };
    if constexpr (cfg::KPF) {
        load_k(blockIdx.y*C);
    }

    for (int k0 = blockIdx.y*C; k0 < ne11; k0 += gridDim.y*C) {
        const int p_end = min(C, ne11 - k0);
        if constexpr (!cfg::KPF) {
            load_mask(k0);
        }

        // Start this chunk's V rows toward L2 now, so the PV phase does not wait on DRAM: one
        // prefetch per 128-byte line of the chunk's rows.
        if constexpr (cfg::PREFETCH) {
            for (int off = tid*128; off < p_end*nb21; off += NT*128) {
                fattn_q4p_prefetch_l2(Vb + int64_t(k0)*nb21 + off);
            }
        }

        // ---- S = Q K^T for this thread's PT positions and RQ rows, over its split of the dims ----
        uint32_t u[PT][NW];
        bool kvalid[PT];
        const uint32_t * segp[PT];
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
            const int pos = k0 + qslot + NQP*pt;
            kvalid[pt] = pos < ne11;
            segp[pt] = (const uint32_t *) (Kb + int64_t(kvalid[pt] ? pos : 0)*nb11 + qsplit*(KBB*BPS));
            if constexpr (cfg::KPF) {
#pragma unroll
                for (int j = 0; j < NW; ++j) {
                    u[pt][j] = un[pt][j];
                }
            } else if constexpr (!cfg::ROLL) {
#pragma unroll
                for (int j = 0; j < NW; ++j) {
                    u[pt][j] = __ldg(segp[pt] + j);
                }
            }
        }

        float S[PT][RQ];
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
                S[pt][r] = 0.0f;
            }
        }

#pragma unroll (cfg::ROLL ? 1 : BPS)
        for (int bl = 0; bl < BPS; ++bl) {
            if constexpr (KQ8) {
                // block bl starts at byte 34*bl: word-aligned when bl is even (scale in the low half of word
                // wb, qs from byte 2), 2 bytes past when odd (scale in the high half of word wb + 8, qs
                // word-aligned from word wb + 9)
                const int wb = 17*(bl/2);
                float    d[PT];
                uint32_t w[PT][8];
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
                    if (bl % 2 == 0) {
                        d[pt] = __half2float(__ushort_as_half((unsigned short) (u[pt][wb] & 0xFFFF)));
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            w[pt][k] = __byte_perm(u[pt][wb + k], u[pt][wb + k + 1], 0x5432);
                        }
                    } else {
                        d[pt] = __half2float(__ushort_as_half((unsigned short) (u[pt][wb + 8] >> 16)));
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            w[pt][k] = u[pt][wb + 9 + k];
                        }
                    }
                }
                float t[PT][RQ];
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                    for (int r = 0; r < RQ; ++r) {
                        t[pt][r] = 0.0f;
                    }
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t x[PT];
#pragma unroll
                    for (int pt = 0; pt < PT; ++pt) {
                        x[pt] = w[pt][k] ^ 0x80808080u; // signed byte q -> unsigned q + 128
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float y[PT];
#pragma unroll
                        for (int pt = 0; pt < PT; ++pt) {
                            // exact q: 2^23 + (q + 128) in the mantissa, minus 2^23 + 128
                            y[pt] = __int_as_float(__byte_perm(x[pt], 0x4B000000u, 0x7440u | e)) - 8388736.0f;
                        }
                        const float4 * q4 = (const float4 *) (Qt + (32*bl + 4*k + e)*RQP);
#pragma unroll
                        for (int rq = 0; rq < RQP/4; ++rq) {
                            const float4 q = q4[rq];
                            const float qv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                            for (int c = 0; c < 4; ++c) {
                                if (4*rq + c < RQ) {
#pragma unroll
                                    for (int pt = 0; pt < PT; ++pt) {
                                        t[pt][4*rq + c] = fmaf(qv[c], y[pt], t[pt][4*rq + c]);
                                    }
                                }
                            }
                        }
                    }
                }
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                    for (int r = 0; r < RQ; ++r) {
                        S[pt][r] = fmaf(d[pt], t[pt][r], S[pt][r]);
                    }
                }
                continue;
            }
            // block bl starts at byte 18*bl: word-aligned when bl is even, 2 bytes past when odd
            const int wb = 9*(bl/2);
            float    d[PT];
            uint32_t w[PT][4];
#pragma unroll
            for (int pt = 0; pt < PT; ++pt) {
                if constexpr (cfg::ROLL) {
                    // the block's 5 aligned words; odd blocks start 2 bytes into the first
                    const uint32_t * bp = segp[pt] + (18*bl)/4;
                    uint32_t v[5];
#pragma unroll
                    for (int j = 0; j < 5; ++j) {
                        v[j] = __ldg(bp + j);
                    }
                    const int odd = bl & 1;
                    d[pt] = __half2float(__ushort_as_half((unsigned short) ((v[0] >> (16*odd)) & 0xFFFF)));
                    const uint32_t sel = odd ? 0x7654u : 0x5432u;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        w[pt][k] = __byte_perm(v[k], v[k + 1], sel);
                    }
                } else if (bl % 2 == 0) {
                    d[pt] = __half2float(__ushort_as_half((unsigned short) (u[pt][wb] & 0xFFFF)));
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        w[pt][k] = __byte_perm(u[pt][wb + k], u[pt][wb + k + 1], 0x5432);
                    }
                } else {
                    d[pt] = __half2float(__ushort_as_half((unsigned short) (u[pt][wb + 4] >> 16)));
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        w[pt][k] = u[pt][wb + 5 + k];
                    }
                }
            }

            if constexpr (cfg::H16QK) {
                // positions 0 and 1 in the two lanes; K nibbles are exact in fp16 as 1024 + n
                half2 t2[RQ];
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
                    t2[r] = make_half2(0.0f, 0.0f);
                }
                const half2 off = make_half2(1032.0f, 1032.0f);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const uint32_t n0 = (hh ? (w[0][k] >> 4) : w[0][k]) & 0x0F0F0F0Fu;
                        const uint32_t n1 = (hh ? (w[1][k] >> 4) : w[1][k]) & 0x0F0F0F0Fu;
                        // interleave the two positions' nibbles, then pair each with a 0x64 byte:
                        // half 0x64nn = 1024 + nn exactly
                        const uint32_t i01 = __byte_perm(n0, n1, 0x5140); // n0[0] n1[0] n0[1] n1[1]
                        const uint32_t i23 = __byte_perm(n0, n1, 0x7362); // n0[2] n1[2] n0[3] n1[3]
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const uint32_t yb = __byte_perm(e < 2 ? i01 : i23, 0x64646464u, (e & 1) ? 0x4342 : 0x4140);
                            const half2 y2 = __hsub2(*(const half2 *) &yb, off);
                            if constexpr (cfg::QPK) {
                                const uint4 * q4 = (const uint4 *) (Qt + (32*bl + 4*k + e + 16*hh)*(RQP/2));
#pragma unroll
                                for (int rq = 0; rq < RQP/8; ++rq) {
                                    const uint4 q = q4[rq];
                                    const uint32_t qv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                                    for (int c = 0; c < 8; ++c) {
                                        if (8*rq + c < RQ) {
                                            const half2 qh = *(const half2 *) &qv[c/2];
                                            const half2 qb = (c & 1) ? __high2half2(qh) : __low2half2(qh);
                                            t2[8*rq + c] = __hfma2(qb, y2, t2[8*rq + c]);
                                        }
                                    }
                                }
                                continue;
                            }
                            const uint4 * q4 = (const uint4 *) (Qt + (32*bl + 4*k + e + 16*hh)*RQP);
#pragma unroll
                            for (int rq = 0; rq < RQP/4; ++rq) {
                                const uint4 q = q4[rq];
                                const uint32_t qv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                                for (int c = 0; c < 4; ++c) {
                                    if (4*rq + c < RQ) {
                                        t2[4*rq + c] = __hfma2(*(const half2 *) &qv[c], y2, t2[4*rq + c]);
                                    }
                                }
                            }
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
                    const uint32_t tb = *(const uint32_t *) &t2[r];
                    if constexpr (Q4P_CVT == 1) {
                        S[0][r] = fmaf(d[0], __low2float(t2[r]),  S[0][r]);
                        S[1][r] = fmaf(d[1], __high2float(t2[r]), S[1][r]);
                    } else {
                        S[0][r] = fmaf(d[0], fattn_q4p_lo_f(tb), S[0][r]);
                        S[1][r] = fmaf(d[1], fattn_q4p_hi_f(tb), S[1][r]);
                    }
                }
                continue;
            }

            float t[cfg::BLOCK_T ? PT : 1][cfg::BLOCK_T ? RQ : 1];
            if constexpr (cfg::BLOCK_T) {
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                    for (int r = 0; r < RQ; ++r) {
                        t[pt][r] = 0.0f;
                    }
                }
            }

#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    uint32_t nib[PT];
#pragma unroll
                    for (int pt = 0; pt < PT; ++pt) {
                        nib[pt] = (hh ? (w[pt][k] >> 4) : w[pt][k]) & 0x0F0F0F0Fu;
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float y[PT];
#pragma unroll
                        for (int pt = 0; pt < PT; ++pt) {
                            y[pt] = fattn_q4p_nib(nib[pt], e);
                            if constexpr (!cfg::BLOCK_T) {
                                y[pt] *= d[pt]; // exact
                            }
                        }
                        const float4 * q4 = (const float4 *) (Qt + (32*bl + 4*k + e + 16*hh)*RQP);
#pragma unroll
                        for (int rq = 0; rq < RQP/4; ++rq) {
                            const float4 q = q4[rq];
                            const float qv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                            for (int c = 0; c < 4; ++c) {
                                if (4*rq + c < RQ) {
#pragma unroll
                                    for (int pt = 0; pt < PT; ++pt) {
                                        if constexpr (cfg::BLOCK_T) {
                                            t[pt][4*rq + c] = fmaf(qv[c], y[pt], t[pt][4*rq + c]);
                                        } else {
                                            S[pt][4*rq + c] = fmaf(qv[c], y[pt], S[pt][4*rq + c]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if constexpr (cfg::BLOCK_T) {
#pragma unroll
                for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                    for (int r = 0; r < RQ; ++r) {
                        S[pt][r] = fmaf(d[pt], t[pt][r], S[pt][r]);
                    }
                }
            }
        }

        if constexpr (cfg::H16QK && Q4P_CVT != 1) {
#pragma unroll
            for (int pt = 0; pt < PT; ++pt) {
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
                    S[pt][r] *= 0x1p112f; // undo the conversion's rebias
                }
            }
        }

        // combine the NSPLIT partial dot products, then mask
        float lmax[RQ];
#pragma unroll
        for (int r = 0; r < RQ; ++r) {
            lmax[r] = -INFINITY;
        }
#pragma unroll
        for (int pt = 0; pt < PT; ++pt) {
            const int pos = k0 + qslot + NQP*pt;
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
#pragma unroll
                for (int off = 1; off < NSPLIT; off <<= 1) {
                    S[pt][r] += __shfl_xor_sync(0xFFFFFFFF, S[pt][r], off, WARP_SIZE);
                }
            }
#pragma unroll
            for (int rl = 0; rl < RQ; ++rl) {
                const int r = qgrp*RQ + rl;
                const int t = r / ncols2;
                float mv = 0.0f;
                if (!kvalid[pt]) {
                    mv = -INFINITY;
                } else if constexpr (MPRE) {
                    mv = __half2float(__ushort_as_half((unsigned short) mraw[pt][t]))*LOG2E;
                } else if (Mb && ic0 + t < n_tok) {
                    mv = __half2float(((const half *) (Mb + int64_t(nb31)*t))[pos])*LOG2E;
                }
                S[pt][rl] += mv;
                lmax[rl] = fmaxf(lmax[rl], S[pt][rl]);
            }
        }

        // lazy max: only rescale when a score beats the running max by more than 2^8
        bool grow = false;
#pragma unroll
        for (int rl = 0; rl < RQ; ++rl) {
            grow |= lmax[rl] > m_s[qgrp*RQ + rl] + 8.0f;
        }
        const bool any_grow = __syncthreads_or(grow);
        if constexpr (NOEB) {
            // the previous chunk's denominators: every warp wrote them before this barrier, and
            // this chunk's are written only after the P barrier below
            if (k0 != blockIdx.y*C) {
                add_lsum();
            }
        }
        if (any_grow) {
            // per warp, reduce over the lanes that share a row group (the position-slot bits)
#pragma unroll
            for (int rl = 0; rl < RQ; ++rl) {
                float v = lmax[rl];
#pragma unroll
                for (int off = WARP_SIZE/2; off >= NSPLIT*RG; off >>= 1) {
                    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, off, WARP_SIZE));
                }
                if (lane < NSPLIT*RG && qsplit == 0) {
                    red_s[warp][qgrp*RQ + rl] = v;
                }
            }
            __syncthreads();
            if (tid < R) {
                float mx = m_s[tid];
#pragma unroll
                for (int w = 0; w < NT/WARP_SIZE; ++w) {
                    mx = fmaxf(mx, red_s[w][tid]);
                }
                a_s[tid] = exp2f(m_s[tid] - mx);
                m_s[tid] = mx;
                l_row *= a_s[tid];
            }
            __syncthreads();
#pragma unroll
            for (int rl = 0; rl < RQ; ++rl) {
                const float a = a_s[vgrp*RQ + rl];
#pragma unroll
                for (int j = 0; j < NACC; ++j) {
                    acc[rl][j] *= a;
                }
            }
        }

        // P for this chunk, published by the first thread of each position; with as many splits as
        // positions per thread, split j publishes position j (the reduction left S in both)
        constexpr bool PSPL = NSPLIT == PT && Q4P_PSPL == 1;
        if (PSPL || qsplit == 0) {
            if constexpr (PSPL) {
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
#pragma unroll
                    for (int pt = 1; pt < PT; ++pt) {
                        if (qsplit == pt) {
                            S[0][r] = S[pt][r];
                        }
                    }
                }
            }
#pragma unroll
            for (int pt = 0; pt < (PSPL ? 1 : PT); ++pt) {
                float * Pp = P_s + (qslot + NQP*(PSPL ? qsplit : pt))*PSTR + qgrp*(cfg::PPK ? RQP/2 : RQP);
#pragma unroll
                for (int rq = 0; rq < RQP/4; ++rq) {
                    float pv[4];
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        const int rl = 4*rq + c;
                        pv[c] = rl < RQ ? exp2f(S[pt][rl] - m_s[qgrp*RQ + rl]) : 0.0f;
                    }
                    if constexpr (cfg::PPK) {
                        const half2 h01 = __floats2half2_rn(pv[0], pv[1]);
                        const half2 h23 = __floats2half2_rn(pv[2], pv[3]);
                        ((uint2 *) Pp)[rq] = make_uint2(*(const uint32_t *) &h01, *(const uint32_t *) &h23);
                    } else if constexpr (cfg::H16PV) {
                        uint32_t hb[4];
#pragma unroll
                        for (int c = 0; c < 4; ++c) {
                            const half2 h = __float2half2_rn(pv[c]);
                            hb[c] = *(const uint32_t *) &h;
                        }
                        ((uint4 *) Pp)[rq] = make_uint4(hb[0], hb[1], hb[2], hb[3]);
                    } else {
                        ((float4 *) Pp)[rq] = make_float4(pv[0], pv[1], pv[2], pv[3]);
                    }
                }
            }
        }
        __syncthreads();

        // the next chunk's K rows toward L2, behind the PV work
        if constexpr (cfg::PREFETCH) {
            const int kn = k0 + gridDim.y*C;
            const int pn = min(C, ne11 - kn);
            for (int off = tid*128; off < pn*nb11; off += NT*128) {
                fattn_q4p_prefetch_l2(Kb + int64_t(kn)*nb11 + off);
            }
        }

        // This chunk's softmax denominators, summed by every thread over a segment of one row, not
        // by R threads walking all C positions: that serial loop sat on warp 0, and the whole block
        // waited for it at the chunk-end barrier. Thread tid owns row tid % LRP and positions
        // [seg*LPPS, (seg+1)*LPPS); per-warp partials go to lsum_s, and the R row owners add them
        // after the chunk-end barrier.
        {
            const int lr = tid % LRP, seg = tid / LRP;
            float ls = 0.0f;
            if (lr < R) {
                const int off = (lr / RQ)*RQP + lr % RQ;
#pragma unroll
                for (int j = 0; j < LPPS; ++j) {
                    const int p = seg*LPPS + j;
                    if (p < p_end) {
                        if constexpr (cfg::PPK) {
                            // the rounded p the numerator uses
                            ls += __half2float(((const half *) &P_s[p*PSTR])[off]);
                        } else if constexpr (cfg::H16PV) {
                            // the rounded p the numerator uses
                            ls += __low2float(*(const half2 *) &P_s[p*PSTR + off]);
                        } else {
                            ls += P_s[p*PSTR + off];
                        }
                    }
                }
            }
            if constexpr (LRP == 16) {
                ls += __shfl_xor_sync(0xFFFFFFFF, ls, 16, WARP_SIZE);
            }
            if (lane < LRP) {
                lsum_s[warp][lane] = ls;
            }
        }

        // the next chunk's K words, in flight during the PV phase
        if constexpr (cfg::KPF) {
            load_k(k0 + gridDim.y*C);
        }

        // ---- O += P V ----
        // V words are loaded one position ahead: at one block per SM there are too few warps to
        // hide an L2 round trip behind the FMAs of the others
        const char * vrow0 = Vb + int64_t(k0 + min(pg, p_end - 1))*nb21;
        uint32_t nd  = fattn_q4p_ld16(vrow0 + 18*pv_b);
        uint32_t nw0 = __ldg((const uint32_t *) (vrow0 + pv_oa));
        uint32_t nw1 = __ldg((const uint32_t *) (vrow0 + pv_oa + 4));
        if constexpr (cfg::H16PV) {
            constexpr int NH = DPT/2;   // half2 accumulators per row
            half2 acc2[RQ][NH];
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
#pragma unroll
                for (int h = 0; h < NH; ++h) {
                    acc2[r][h] = make_half2(0.0f, 0.0f);
                }
            }
            const half2 off = make_half2(1032.0f, 1032.0f);
            auto fold = [&]() {
                if constexpr (!cfg::RS) {
#pragma unroll
                    for (int r = 0; r < RQ; ++r) {
#pragma unroll
                        for (int h = 0; h < NH; ++h) {
                            const uint32_t ab = *(const uint32_t *) &acc2[r][h];
                            if constexpr (Q4P_CVT == 1) {
                                acc[r][2*h + 0] += __low2float(acc2[r][h]);
                                acc[r][2*h + 1] += __high2float(acc2[r][h]);
                            } else {
                                acc[r][2*h + 0] = fmaf(0x1p112f, fattn_q4p_lo_f(ab), acc[r][2*h + 0]);
                                acc[r][2*h + 1] = fmaf(0x1p112f, fattn_q4p_hi_f(ab), acc[r][2*h + 1]);
                            }
                            acc2[r][h] = make_half2(0.0f, 0.0f);
                        }
                    }
                }
            };
            int np = 0;
            // V words two positions ahead: a quarter of the fp32 path's work per position no
            // longer covers an L2 round trip
            uint32_t nd2, nw02, nw12;
            // row offsets within the chunk in 32 bits, advanced by one step and clamped to the last
            // row (the 64-bit row products cost ~20 instructions per position)
            const char * vcd = Vb + int64_t(k0)*nb21 + 18*pv_b;
            const char * vcw = Vb + int64_t(k0)*nb21 + pv_oa;
            const int vstep = NPG*nb21, vlast = (p_end - 1)*nb21;
            int vo = min(pg + NPG, p_end - 1)*nb21;
            {
                nd2  = fattn_q4p_ld16(vcd + vo);
                nw02 = __ldg((const uint32_t *) (vcw + vo));
                nw12 = __ldg((const uint32_t *) (vcw + vo + 4));
            }
#pragma unroll 2
            for (int p = pg; p < p_end; p += NPG) {
                const half2    dv2 = __half2half2(__ushort_as_half((unsigned short) nd));
                const uint32_t wv  = __byte_perm(nw0, nw1, pv_sel);
                nd = nd2; nw0 = nw02; nw1 = nw12;
                {
                    vo = min(vo + vstep, vlast); // row min(p + 2*NPG, p_end - 1)
                    nd2  = fattn_q4p_ld16(vcd + vo);
                    nw02 = __ldg((const uint32_t *) (vcw + vo));
                    nw12 = __ldg((const uint32_t *) (vcw + vo + 4));
                }
                // DPT 4: this thread's nibble half of the word; DPT 8: low nibbles (dims 0-3), then high
                half2 y[NH];
#pragma unroll
                for (int h2 = 0; h2 < NH/2; ++h2) {
                    uint32_t nb, ya, yb;
                    if constexpr (cfg::RS) {
                        // element i of this lane is element i ^ pg of the plain order, so that fold_rs
                        // keeps and sends fixed registers (no selects)
                        nb = (wv >> (h2 ? 4 - rs_sh : rs_sh)) & 0x0F0F0F0Fu;
                        ya = __byte_perm(nb, 0x64646464u, rs_sa);
                        yb = __byte_perm(nb, 0x64646464u, rs_sb);
                    } else {
                        nb = (DPT == 8 ? h2 : (dg & 1)) ? ((wv >> 4) & 0x0F0F0F0Fu) : (wv & 0x0F0F0F0Fu);
                        ya = __byte_perm(nb, 0x64646464u, 0x4140); // bytes 0, 1 as 1024 + n
                        yb = __byte_perm(nb, 0x64646464u, 0x4342); // bytes 2, 3
                    }
                    y[2*h2 + 0] = __hmul2(__hsub2(*(const half2 *) &ya, off), dv2);
                    y[2*h2 + 1] = __hmul2(__hsub2(*(const half2 *) &yb, off), dv2);
                }
                if constexpr (cfg::PPK) {
                    const uint4 * pq = (const uint4 *) (P_s + p*PSTR);
#pragma unroll
                    for (int rq = 0; rq < RQP/8; ++rq) {
                        const uint4 q = pq[rq];
                        const uint32_t pv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            if (8*rq + c < RQ) {
                                const half2 ph = *(const half2 *) &pv[c/2];
                                const half2 pb = (c & 1) ? __high2half2(ph) : __low2half2(ph);
#pragma unroll
                                for (int h = 0; h < NH; ++h) {
                                    acc2[8*rq + c][h] = __hfma2(pb, y[h], acc2[8*rq + c][h]);
                                }
                            }
                        }
                    }
                } else {
                const uint4 * pq = (const uint4 *) (P_s + p*PSTR + vgrp*RQP);
#pragma unroll
                for (int rq = 0; rq < RQP/4; ++rq) {
                    const uint4 q = pq[rq];
                    const uint32_t pv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
                    for (int c = 0; c < 4; ++c) {
                        if (4*rq + c < RQ) {
#pragma unroll
                            for (int h = 0; h < NH; ++h) {
                                acc2[4*rq + c][h] = __hfma2(*(const half2 *) &pv[c], y[h], acc2[4*rq + c][h]);
                            }
                        }
                    }
                }
                }
                if constexpr (!cfg::RS) {
                    if (++np == Q4P_PVF) {
                        np = 0;
                        fold();
                    }
                }
            }
            if constexpr (cfg::RS) {
                fold_rs(acc2);
            } else {
                fold();
            }
        } else
#pragma unroll 2
        for (int p = pg; p < p_end; p += NPG) {
            const float    dv = __half2float(__ushort_as_half((unsigned short) nd));
            const uint32_t wv = __byte_perm(nw0, nw1, pv_sel);
            {
                const char * vrow = Vb + int64_t(k0 + min(p + NPG, p_end - 1))*nb21;
                nd  = fattn_q4p_ld16(vrow + 18*pv_b);
                nw0 = __ldg((const uint32_t *) (vrow + pv_oa));
                nw1 = __ldg((const uint32_t *) (vrow + pv_oa + 4));
            }

            float y[DPT];
            if constexpr (DPT == 8) {
                const uint32_t lo = wv & 0x0F0F0F0Fu, hi = (wv >> 4) & 0x0F0F0F0Fu;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    y[e]     = fattn_q4p_nib(lo, e);
                    y[e + 4] = fattn_q4p_nib(hi, e);
                }
            } else if constexpr (DPT == 4) {
                const uint32_t nb = ((dg & 1) ? (wv >> 4) : wv) & 0x0F0F0F0Fu;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    y[e] = fattn_q4p_nib(nb, e);
                }
            } else {
                const uint32_t nb = ((dg & 1) ? (wv >> 4) : wv) & 0x0F0F0F0Fu;
                const int e0 = 2*((dg >> 1) & 1);
                y[0] = fattn_q4p_nib(nb, e0);
                y[1] = fattn_q4p_nib(nb, e0 + 1);
            }

            float P[RQP];
#pragma unroll
            for (int rq = 0; rq < RQP/4; ++rq) {
                const float4 pq = ((const float4 *) (P_s + p*PSTR + vgrp*RQP))[rq];
                P[4*rq + 0] = pq.x; P[4*rq + 1] = pq.y; P[4*rq + 2] = pq.z; P[4*rq + 3] = pq.w;
            }
            if constexpr (cfg::FOLD) {
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
                    const float pd = P[r]*dv;
#pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        acc[r][j] = fmaf(pd, y[j], acc[r][j]);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < DPT; ++j) {
                    y[j] *= dv; // exact
                }
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
#pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        acc[r][j] = fmaf(P[r], y[j], acc[r][j]);
                    }
                }
            }
        }
        if constexpr (!NOEB) {
            __syncthreads();
            add_lsum();
        }
    }
    if constexpr (NOEB) {
        __syncthreads();
        add_lsum();
    }

    // ---- reduce the NPG partial outputs and the partial denominators ----
    if (tid < R) {
        red_s[0][tid] = l_row;
#pragma unroll
        for (int w = 1; w < NPG; ++w) {
            red_s[w][tid] = 0.0f;
        }
    }
    float * O_s = Q_s;
    if constexpr (cfg::RS) {
#pragma unroll
        for (int r = 0; r < RQ; ++r) {
            O_s[r*D + fattn_q4p_pv_dim<DPT>(dg, pg)] = acc[r][0];
        }
        __syncthreads();
    } else {
#pragma unroll
        for (int g0 = 0; g0 < NPG; ++g0) {
            if (pg == g0) {
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
#pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        float & o = O_s[(vgrp*RQ + r)*D + fattn_q4p_pv_dim<DPT>(dg, j)];
                        o = g0 == 0 ? acc[r][j] : o + acc[r][j];
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < R*D; idx += NT) {
        const int r = idx / D, dim = idx % D;
        const int t = r / ncols2, g = r % ncols2;
        if (ic0 + t >= n_tok) {
            continue;
        }
        float lsum = 0.0f;
#pragma unroll
        for (int w = 0; w < NPG; ++w) {
            lsum += red_s[w][r];
        }
        float val = O_s[idx];
        if (gridDim.y == 1) {
            val /= lsum;
        }
        dst[(((int64_t(sequence)*n_tok + ic0 + t)*ne02 + head0 + g)*gridDim.y + blockIdx.y)*D + dim] = val;
    }
    if (gridDim.y != 1 && tid < R) {
        const int t = tid / ncols2, g = tid % ncols2;
        if (ic0 + t < n_tok) {
            float lsum = 0.0f;
#pragma unroll
            for (int w = 0; w < NPG; ++w) {
                lsum += red_s[w][tid];
            }
            dst_meta[((int64_t(sequence)*n_tok + ic0 + t)*ne02 + head0 + g)*gridDim.y + blockIdx.y] =
                make_float2(m_s[tid]*LN2, lsum);
        }
    }
    GGML_UNUSED_VARS(sinks, KV_max, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, ne10, ne12, ne13,
        ne31, ne32, nb32);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale, max_bias, m0, m1, n_head_log2,
        logit_softcap, ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

// Whether flash_attn_ext_q4p handles this op. Kept in one place because the alloc-size path must
// agree with the dispatch: it reserves no f16 staging when this kernel runs.
static bool ggml_cuda_fattn_q4p_supported(const ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * s = getenv("GGML_CUDA_FA_Q4P");
        return !s || atoi(s) != 0;
    }();
    if (!enabled) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc >= GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }
    // GGML_CUDA_FA_Q4P_Q8K=0: q8_0 K (with q4_0 V) back to the generic kernels
    static const bool q8k = [] { const char * s = getenv("GGML_CUDA_FA_Q4P_Q8K"); return !s || atoi(s) != 0; }();
    const bool k_ok = K->type == GGML_TYPE_Q4_0 || (q8k && K->type == GGML_TYPE_Q8_0);
    if (!k_ok || V->type != GGML_TYPE_Q4_0 || Q->type != GGML_TYPE_F32 || sinks) {
        return false;
    }
    if (Q->ne[0] != FATTN_Q4P_D || K->ne[0] != FATTN_Q4P_D || V->ne[0] != FATTN_Q4P_D) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > 5 || K->ne[2] == 0 || Q->ne[2] != 6*K->ne[2]) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || ggml_get_op_params_i32(dst, 4) > 0) {
        return false; // no ALiBi, softcap or sparse KV hint
    }
    if (mask && (mask->ne[2] != 1 || mask->ne[1] < Q->ne[1] || mask->ne[0] < K->ne[1])) {
        return false;
    }
    // 32-bit loads of the qs words need 4-aligned rows
    if (K->nb[1] % 4 != 0 || V->nb[1] % 4 != 0 || (uintptr_t) K->data % 4 != 0 || (uintptr_t) V->data % 4 != 0) {
        return false;
    }
    return true;
}

static bool q4p_old() {
    static const bool v = [] { const char * s = getenv("GGML_CUDA_Q4P_OLD"); return s && atoi(s) != 0; }();
    return v;
}

template <int ncols1, int ncols2 = 6>
static void ggml_cuda_flash_attn_ext_q4p_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[1]->type == GGML_TYPE_Q8_0) {
        launch_fattn<FATTN_Q4P_D, ncols1, ncols2>(ctx, dst, flash_attn_ext_q4p<ncols1, ncols2, false, true>, 256/WARP_SIZE, 0,
            fattn_q4p_cfg<ncols1*ncols2, false, true>::C, false, false, false, false);
        return;
    }
    if (q4p_old()) {
        launch_fattn<FATTN_Q4P_D, ncols1, ncols2>(ctx, dst, flash_attn_ext_q4p<ncols1, ncols2, true>, 256/WARP_SIZE, 0,
            fattn_q4p_cfg<ncols1*ncols2, true>::C, false, false, false, false);
    } else {
        launch_fattn<FATTN_Q4P_D, ncols1, ncols2>(ctx, dst, flash_attn_ext_q4p<ncols1, ncols2, false>, 256/WARP_SIZE, 0,
            fattn_q4p_cfg<ncols1*ncols2, false>::C, false, false, false, false);
    }
}

// GGML_CUDA_Q4P_NC2=6 runs the 5-token case over the whole GQA group per block (the old default)
static int q4p_half_group() {
    static const int v = [] { const char * s = getenv("GGML_CUDA_Q4P_NC2"); return s ? atoi(s) : 3; }();
    return v;
}

static void ggml_cuda_flash_attn_ext_q4p(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // a fixed-width verify graph may have only its first tokens real (active-tokens.cuh): run the
    // kernel on views of Q and the output that stop there. The mask keeps its rows.
    const int64_t n_act = ggml_cuda_active_cols(dst->src[0]->ne[1]);
    ggml_tensor q_act, dst_act;
    if (n_act < dst->src[0]->ne[1]) {
        q_act          = *dst->src[0];
        q_act.ne[1]    = n_act;
        dst_act        = *dst;
        dst_act.ne[2]  = n_act;
        dst_act.src[0] = &q_act;
        dst = &dst_act;
    }
    switch (dst->src[0]->ne[1]) {
        case 1: ggml_cuda_flash_attn_ext_q4p_case<1>(ctx, dst); break;
        case 2: ggml_cuda_flash_attn_ext_q4p_case<2>(ctx, dst); break;
        case 3: ggml_cuda_flash_attn_ext_q4p_case<3>(ctx, dst); break;
        case 4: ggml_cuda_flash_attn_ext_q4p_case<4>(ctx, dst); break;
        case 5:
            // 30 rows need ~120 accumulator registers per thread, so one block fills an SM and there
            // are too few warps to hide the loads. Half the GQA group per block (15 rows) runs two
            // blocks per SM and dequantizes each K/V value twice: 3.55 -> 3.27 ms per call at 260k in
            // the server. 4 tokens lose (2.48 -> 2.78). OPTLOG 204.
            if (q4p_half_group() == 6) { ggml_cuda_flash_attn_ext_q4p_case<5>(ctx, dst); }
            else                       { ggml_cuda_flash_attn_ext_q4p_case<5, 3>(ctx, dst); }
            break;
        default: GGML_ABORT("fatal error");
    }
}
