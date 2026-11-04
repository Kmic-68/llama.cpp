#include "tile-mm.cuh"
#include "mmid.cuh"

// rows of skipped slots (expert id < 0) get zeros: the grouped GEMM never writes them (from wispoffates' mmid.cu, local here)
static __global__ void tile_mm_zero_skipped(const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ne0, const int n_expert_used, const int si1, const int64_t s1, const int64_t s2) {
    const int it = blockIdx.x;
    for (int iex = 0; iex < n_expert_used; ++iex) {
        if (ids[it*si1 + iex] >= 0) {
            continue;
        }
        float * row = dst + it*s2 + iex*s1;
        for (int i = threadIdx.x; i < ne0; i += blockDim.x) {
            row[i] = 0.0f;
        }
    }
}

#include <atomic>
#include <cstdio>

// Quantized x f32 GEMM for 9..80 columns on Pascal (sm_60), fused: the weights are read once and decoded tile by tile in shared
// memory instead of being dequantized to a full fp16 copy for cuBLAS. Types: q6_K, q5_K, q4_K, q8_0, q4_0. Also a grouped MUL_MAT_ID mode for
// MoE prefill (one launch over all experts, token lists sorted on the device by mm_ids_helper), in place of MMQ's emulated-DP4A
// integer path on sm_60.
//
// Why: sm_60 has no DP4A, so there is no MMQ and every batch wider than the matvec kernels (> 8 columns, e.g. Swift's n-gram
// verify batches of 49..65 tokens) goes dequantize -> fp16 temp -> maxwell_hgemm_256x128 with fp16 accumulation over all of K.
// At 65 columns that tile is half empty (~6 TFLOPS) and the dequant pass writes 2 bytes per weight every call.
//
// Layout (tilegemm harness v3, ~/claude/bench-tools/tilegemm):
//   block = 64 rows x 16*TN columns, 256 threads; thread (rg, cg) owns rows rg + 16*i (i < 4), columns cg*TN + j.
//   The tile's raw weight bytes are staged one unit at a time (q6_K: a 256-value superblock, 210 bytes per row, copied from the
//   aligned-down address; q8_0: four 32-value blocks, 136 bytes per row, always 4-aligned) global -> registers -> shared, with
//   coalesced 32-bit loads. Re-reading the bytes every 64-value step capped the fetch at ~65-110 GB/s on the L1/tex path;
//   staging each byte once reaches ~400 GB/s. The next unit is prefetched into registers during the math.
//   Each 64-value step decodes the raw bytes to fp16 (x1024, away from subnormals) and multiplies with HFMA2; activations are
//   prescaled per 1024-value window by a power of 2 (|x| < 1, as in mmvq-f16.cu) and the fp16 chains are folded into fp32
//   every 32 k-pairs. The activation k-pairs are rotated by one inside each group of 4 so that the two HFMA2 sources never sit
//   in the same register bank.
//   Split-K over units fills the GPU; partial sums go to a pool buffer and a small kernel reduces them in a fixed order
//   (deterministic).
// q6_K: NMSE ~9e-7 against a double-precision reference on random weights (cuBLAS path ~1.1e-5). Measured (P100, 150 W, harness
// incl. activation prep and reduce), rows x K at 64 columns, against cuBLAS hgemm alone (production adds the dequant pass, ~320 us
// at 8704x5120): 8704x5120 1151 -> 844 us, 5120x8704 1152 -> 874, 6144x5120 898 -> 629, 5120x5120 775 -> 542, 3072x5120 404 -> 339.
// Swift serving (v5.1 + --spec-ngram-mod-n-max 63): edit +13.6 %, edit2 +19 %, short-prompt prefill +20 %.

static constexpr int TMM_WIN = 1024;   // activation prescale window (values)
static constexpr int TMM_KP  = 32;     // k-pairs per step (64 values)
static constexpr int TMM_LD  = 36;     // shared row stride in half2: 128-bit loads are conflict-free
static constexpr int TMM_BM  = 64;

template <ggml_type type> struct tile_mm_traits;
template <> struct tile_mm_traits<GGML_TYPE_Q6_K> {
    static constexpr int  UV      = QK_K;   // values per staged unit (one superblock)
    static constexpr int  RW      = 53;     // 32-bit words per staged row: the 210-byte block plus up to 2 bytes of alignment slack
    static constexpr int  UB      = sizeof(block_q6_K);
    static constexpr bool aligned = false;
    static constexpr bool flat    = false;  // per-warp row staging (rows are long)
};
template <> struct tile_mm_traits<GGML_TYPE_Q5_K> {
    static constexpr int  UV      = QK_K;   // one superblock, 176 bytes per row: always 4-aligned
    static constexpr int  RW      = 44;
    static constexpr int  UB      = sizeof(block_q5_K);
    static constexpr bool aligned = true;
    static constexpr bool flat    = false;
};
template <> struct tile_mm_traits<GGML_TYPE_Q4_K> {
    static constexpr int  UV      = QK_K;   // one superblock, 144 bytes per row: always 4-aligned
    static constexpr int  RW      = 36;
    static constexpr int  UB      = sizeof(block_q4_K);
    static constexpr bool aligned = true;
    static constexpr bool flat    = false;
};
template <> struct tile_mm_traits<GGML_TYPE_Q8_0> {
    static constexpr int  UV      = 4*QK8_0; // four blocks: 136 bytes per row, 4-aligned when the row holds an even number of blocks
    static constexpr int  RW      = 34;
    static constexpr int  UB      = 4*sizeof(block_q8_0);
    static constexpr bool aligned = true;
    static constexpr bool flat    = false;
};
template <> struct tile_mm_traits<GGML_TYPE_Q4_0> {
    static constexpr int  UV      = 2*QK4_0; // two blocks = one 64-value step: 36 bytes per row (K = 704 works: gemma4's ffn_down_exps)
    static constexpr int  RW      = 9;
    static constexpr int  UB      = 2*sizeof(block_q4_0);
    static constexpr bool aligned = true;
    static constexpr bool flat    = true;   // 9-word rows: stage the tile as one flat list (~3 rows per warp load, not 1)
};

// p100-series 10-07: expert banks of 2-bit qwen4exp builds (Sakura: IQ2_S / IQ2_XXS / IQ1_M gate+up, Q2_0 down), which otherwise
// prefill through MMQ's emulated DP4A. All flat (rows of 9..21 words); the 66 / 82-byte IQ2 blocks are 2-aligned and staged from the
// aligned-down address like q6_K.
template <> struct tile_mm_traits<GGML_TYPE_Q2_0> {
    static constexpr int  UV      = 2*QK2_0; // two blocks = two 64-value steps: 36 bytes per row, 4-aligned when K % 128 == 0
    static constexpr int  RW      = 9;
    static constexpr int  UB      = 2*sizeof(block_q2_0);
    static constexpr bool aligned = true;
    static constexpr bool flat    = true;
};
// one-block Q2_0 units for rows with an odd block count (K % 128 == 64), e.g. the per-device ffn_down slice of a -sm tensor split
// (Flash-Next K 640 -> 320 = 5 blocks, 90-byte rows, only 2-aligned): staged from the aligned-down address like the IQ2 types. LP4:
// these down projections fell back to MMQ's emulated DP4A (16 s of a 36K-token prefill).
static constexpr ggml_type TMM_Q2_0_B1 = (ggml_type) (GGML_TYPE_COUNT + 1);
template <> struct tile_mm_traits<TMM_Q2_0_B1> {
    static constexpr int  UV      = QK2_0;  // one block, 18 bytes: 5 words from the aligned-down address
    static constexpr int  RW      = 5;
    static constexpr int  UB      = sizeof(block_q2_0);
    static constexpr bool aligned = false;
    static constexpr bool flat    = true;
};
template <> struct tile_mm_traits<GGML_TYPE_IQ2_S> {
    static constexpr int  UV      = QK_K;   // one superblock, 82 bytes: 21 words from the aligned-down address
    static constexpr int  RW      = 21;
    static constexpr int  UB      = sizeof(block_iq2_s);
    static constexpr bool aligned = false;
    static constexpr bool flat    = true;
};
template <> struct tile_mm_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int  UV      = QK_K;   // one superblock, 66 bytes: 17 words from the aligned-down address
    static constexpr int  RW      = 17;
    static constexpr int  UB      = sizeof(block_iq2_xxs);
    static constexpr bool aligned = false;
    static constexpr bool flat    = true;
};
// Flash-Next UD-IQ4_XS expert banks (IQ3_S gate/up, IQ4_NL down) otherwise prefill through MMQ's emulated DP4A
template <> struct tile_mm_traits<GGML_TYPE_IQ3_S> {
    static constexpr int  UV      = QK_K;   // one superblock, 110 bytes: 28 words from the aligned-down address
    static constexpr int  RW      = 28;
    static constexpr int  UB      = sizeof(block_iq3_s);
    static constexpr bool aligned = false;
    static constexpr bool flat    = true;
};
template <> struct tile_mm_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int  UV      = 2*QK4_NL; // two blocks = one 64-value step: 36 bytes per row, as q4_0
    static constexpr int  RW      = 9;
    static constexpr int  UB      = 2*sizeof(block_iq4_nl);
    static constexpr bool aligned = true;
    static constexpr bool flat    = true;
};
template <> struct tile_mm_traits<GGML_TYPE_IQ1_M> {
    static constexpr int  UV      = QK_K;   // one superblock, 56 bytes: always 4-aligned
    static constexpr int  RW      = 14;
    static constexpr int  UB      = sizeof(block_iq1_m);
    static constexpr bool aligned = true;
    static constexpr bool flat    = true;
};

// byte address of staged unit u of row m
template <ggml_type type>
static __device__ __forceinline__ const unsigned char * tile_mm_unit(const void * W, const int64_t m, const int64_t row_bytes, const int u) {
    return (const unsigned char *) W + m*row_bytes + (int64_t) u*tile_mm_traits<type>::UB;
}

// decode step st (64 values) of a staged unit: lane j returns k-pair j of the step as fp16 x1024
template <ggml_type type>
static __device__ __forceinline__ half2 tile_mm_decode(const unsigned char * rb, const int st, const int lane) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        // g = 2sh + j/16, l = 2*(j%16), +1 (h = st/2, sh = st%2)
        const int h = st >> 1, sh = st & 1, g = 2*sh + (lane >> 4), lp = lane & 15;
        const uint32_t qlw = *(const unsigned short *) (rb + 64*h + 2*lane);
        const uint32_t qhw = *(const unsigned short *) (rb + 128 + 32*h + 2*lp);
        const int      sc  = (int) (signed char) rb[192 + 8*h + (lp >> 3) + 2*g];
        const float    d   = __half2float(*(const __half *) (rb + 208))*1024.0f*sc;
        const int nsh = 4*sh, hsh = 2*g;
        const int q0 = ((qlw >> nsh) & 0xF) | (((qhw >> hsh) & 3) << 4);
        const int q1 = (((qlw >> 8) >> nsh) & 0xF) | ((((qhw >> 8) >> hsh) & 3) << 4);
        return __floats2half2_rn(d*(q0 - 32), d*(q1 - 32));
    } else if constexpr (type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q4_K) {
        // sub-block s = 2st + j/16 (low nibbles for j < 16, high nibbles for j >= 16 of the step's 32 qs bytes), values 2*(j%16), +1;
        // x = d*sc*q - dmin*m with the 6-bit scale / min of s unpacked as in get_scale_min_k4; q5_K adds bit s of qh as 16
        constexpr int QS = type == GGML_TYPE_Q5_K ? 48 : 16;   // offset of qs (q5_K: qh at 16)
        const int g = lane >> 4, s = 2*st + g, q = lane & 15;
        const unsigned char * sc6 = rb + 4;
        int sc, m;
        if (s < 4) {
            sc = sc6[s] & 63;
            m  = sc6[s + 4] & 63;
        } else {
            sc = (sc6[s + 4] & 0xF) | ((sc6[s - 4] >> 6) << 4);
            m  = (sc6[s + 4] >>  4) | ((sc6[s]     >> 6) << 4);
        }
        const float d    = __half2float(*(const __half *) (rb + 0))*1024.0f*sc;
        const float dmin = __half2float(*(const __half *) (rb + 2))*1024.0f*m;
        const uint32_t qlw = *(const unsigned short *) (rb + QS + 32*st + 2*q);
        int q0 = (qlw >> (4*g)) & 0xF, q1 = (qlw >> (8 + 4*g)) & 0xF;
        if constexpr (type == GGML_TYPE_Q5_K) {
            const uint32_t qhw = *(const unsigned short *) (rb + 16 + 2*q);
            q0 |= ((qhw >> s) & 1) << 4;
            q1 |= ((qhw >> (8 + s)) & 1) << 4;
        }
        return __floats2half2_rn(d*q0 - dmin, d*q1 - dmin);
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        // block 2st + j/16 of the unit, quants 2*(j%16), +1
        const unsigned char * b = rb + (2*st + (lane >> 4))*sizeof(block_q8_0);
        const float    d  = __half2float(*(const __half *) b)*1024.0f;
        const uint32_t qw = *(const unsigned short *) (b + 2 + 2*(lane & 15));
        return __floats2half2_rn(d*(int) (signed char) (qw & 0xFF), d*(int) (signed char) (qw >> 8));
    } else if constexpr (type == GGML_TYPE_Q2_0 || type == TMM_Q2_0_B1) {
        // step st = block st of the unit; values 2j, 2j+1 = bits 4*(j%2) .. +3 of qs[j/2]; code c -> (c - 1)*d
        const unsigned char * b = rb + st*sizeof(block_q2_0);
        const float    d = __half2float(*(const __half *) b)*1024.0f;
        const uint32_t q = (b[2 + (lane >> 1)] >> (4*(lane & 1))) & 0xF;
        return __floats2half2_rn(d*((int) (q & 3) - 1), d*((int) (q >> 2) - 1));
    } else if constexpr (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ1_M) {
        // values v = 64st + 2j, +1: sub-block ib = 2st + j/16, grid group il = (j%16)/4, entries jj = 2*(j%4), +1 (as dequantize_row_*)
        const int ib = 2*st + (lane >> 4), il = (lane & 15) >> 2, jj = 2*(lane & 3);
        if constexpr (type == GGML_TYPE_IQ2_S) {
            const float    d   = __half2float(*(const __half *) rb);
            const uint32_t qh  = rb[2 + QK_K/4 + ib], sc = rb[2 + QK_K/4 + QK_K/32 + ib];
            const uint64_t g   = iq2s_grid[rb[2 + 4*ib + il] | ((qh << (8 - 2*il)) & 0x300)];
            const uint32_t sg  = rb[2 + QK_K/8 + 4*ib + il];
            const float    dl  = d*(0.5f + ((sc >> 4*(il/2)) & 0xF))*0.25f*1024.0f;
            const float w0 = (float) ((g >> 8*jj) & 0xFF), w1 = (float) ((g >> (8*jj + 8)) & 0xFF);
            return __floats2half2_rn(dl*((sg >> jj) & 1 ? -w0 : w0), dl*((sg >> (jj + 1)) & 1 ? -w1 : w1));
        } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
            const float    d    = __half2float(*(const __half *) rb);
            const unsigned char * q2 = rb + 2 + 8*ib;
            const uint32_t aux1 = q2[4] | (q2[5] << 8) | (q2[6] << 16) | ((uint32_t) q2[7] << 24);
            const uint64_t g    = iq2xxs_grid[q2[il]];
            const uint32_t sg   = ksigns_iq2xs[(aux1 >> 7*il) & 127];
            const float    dl   = d*(0.5f + (aux1 >> 28))*0.25f*1024.0f;
            const float w0 = (float) ((g >> 8*jj) & 0xFF), w1 = (float) ((g >> (8*jj + 8)) & 0xFF);
            return __floats2half2_rn(dl*((sg >> jj) & 1 ? -w0 : w0), dl*((sg >> (jj + 1)) & 1 ? -w1 : w1));
        } else {
            const uint16_t * sc = (const uint16_t *) (rb + QK_K/8 + QK_K/16);
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            const int      ib16  = 2*ib + il/2;
            const float    dl    = __half2float(scale.f16)*(2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1)*1024.0f;
            const uint32_t qhb   = rb[QK_K/8 + 2*ib + il/2];
            const float    delta = qhb & (0x08 << 4*(il%2)) ? -1.0f - IQ1M_DELTA : -1.0f + IQ1M_DELTA;
            const uint32_t g     = iq1s_grid_gpu[rb[4*ib + il] | (((qhb >> 4*(il%2)) & 7) << 8)];
            // entries 0..3 = low nibbles of the grid bytes, 4..7 = high nibbles
            const int sh = jj < 4 ? 8*jj : 8*(jj - 4) + 4;
            return __floats2half2_rn(dl*((float) ((g >> sh) & 0xF) + delta), dl*((float) ((g >> (sh + 8)) & 0xF) + delta));
        }
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        // sub-block ib = 2st + j/16 (32 values), k-pair p = j%16: group il = p/4 (8 values), grid gi = (p%4)/2, bytes 2*(p%2), +1
        // (as dequantize_block_iq3_s)
        const int ib = 2*st + (lane >> 4), p = lane & 15, il = p >> 2, o = 2*(p & 3), gi = o >> 2, jb = o & 3;
        const uint32_t qh  = rb[66 + ib];
        const uint32_t g   = iq3s_grid[rb[2 + 8*ib + 2*il + gi] | ((qh << (8 - 2*il - gi)) & 256)];
        const uint32_t sg  = rb[74 + 4*ib + il];
        const float    dl  = __half2float(*(const __half *) rb)*(1 + 2*((rb[106 + ib/2] >> 4*(ib%2)) & 0xF))*1024.0f;
        const float w0 = (float) ((g >> 8*jb) & 0xFF), w1 = (float) ((g >> (8*jb + 8)) & 0xFF);
        return __floats2half2_rn(dl*((sg >> o) & 1 ? -w0 : w0), dl*((sg >> (o + 1)) & 1 ? -w1 : w1));
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        // as q4_0 below, through the non-linear table
        const unsigned char * b = rb + (lane >> 4)*sizeof(block_iq4_nl);
        const int q = lane & 15, sh = q >= 8 ? 4 : 0;
        const float    d  = __half2float(*(const __half *) b)*1024.0f;
        const uint32_t qw = *(const unsigned short *) (b + 2 + 2*(q & 7));
        return __floats2half2_rn(d*kvalues_iq4nl[(qw >> sh) & 0xF], d*kvalues_iq4nl[((qw >> 8) >> sh) & 0xF]);
    } else {
        // q4_0 (one step per unit): block j/16, values p = 2*(j%16), p+1; values 0..15 are the low nibbles of qs[0..15], 16..31 the
        // high nibbles
        const unsigned char * b = rb + (lane >> 4)*sizeof(block_q4_0);
        const int q = lane & 15, sh = q >= 8 ? 4 : 0;
        const float    d  = __half2float(*(const __half *) b)*1024.0f;
        const uint32_t qw = *(const unsigned short *) (b + 2 + 2*(q & 7));
        return __floats2half2_rn(d*(int) (((qw >> sh) & 0xF) - 8), d*(int) ((((qw >> 8) >> sh) & 0xF) - 8));
    }
}

// launch parameters (dense: P = split-K partials [splits][N][M]; MUL_MAT_ID: one tile list entry (expert, first column) per block,
// output straight to dst rows through ids_dst)
struct tile_mm_params {
    const void   * W;
    const void   * W2;          // MUL_MAT_ID with a tail assembly: experts < split read from W2 (the pool), the rest from W
    int            split;
    int64_t        row_bytes;
    int64_t        expert_bytes;
    const __half * XS;          // [N or compact rows][K], prescaled
    const float  * S;           // [N or compact rows][nwin]
    float        * P;
    float        * dst;
    int64_t        s1;          // dst row stride (floats), MUL_MAT_ID
    const int    * ids_dst;     // compact row -> dst row
    int            tm;          // MUL_MAT_ID: rows per thread (4: 64-row blocks, 5: 80-row blocks)
    const int    * xrow;        // MUL_MAT_ID: compact row -> XS/S row (activations prepped once per token), or null (per compact row)
    const int    * bounds;      // expert -> first compact row, [n_experts + 1]
    const int    * tiles;       // [ntiles][2] = (expert, first column of the expert's compact rows)
    const int    * ntiles;
    int            M, K, N, nwin, u_per;
};

// X f32 [N][K] (row stride sx floats) -> XS half [N][K] prescaled by 2^-e per (column, window) so |x| < 1; S[n][w] = 2^e.
// With ids (MUL_MAT_ID) column c of XS is row ids[c] of X (the compact, expert-sorted order).
static __global__ void tile_mm_prep(const float * __restrict__ X, const int64_t sx, const int * __restrict__ ids, __half * __restrict__ XS,
                                    float * __restrict__ S, const int K, const int nwin) {
    const int w = blockIdx.x*blockDim.y + threadIdx.y, c = blockIdx.y;
    if (w >= nwin) {
        return;
    }
    const float * x = X + (int64_t) (ids ? ids[c] : c)*sx + (int64_t) w*TMM_WIN;
    const int len = min(TMM_WIN, K - w*TMM_WIN);
    float m = 0.0f;
    for (int i = threadIdx.x; i < len; i += WARP_SIZE) {
        m = fmaxf(m, fabsf(x[i]));
    }
#pragma unroll
    for (int o = WARP_SIZE/2; o; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, o));
    }
    int e = 0;
    if (m > 0.0f) {
        frexpf(m, &e);
    }
    const float inv = ldexpf(1.0f, -e);
    __half * xs = XS + (int64_t) c*K + (int64_t) w*TMM_WIN;
    for (int i = threadIdx.x; i < len; i += WARP_SIZE) {
        xs[i] = __float2half(x[i]*inv);
    }
    if (threadIdx.x == 0) {
        S[(int64_t) c*nwin + w] = ldexpf(1.0f, e);
    }
}

// TN >= 5 (80 / 96 columns) needs > 32 KB of shared memory, so only one block fits per SM anyway: let it use the registers
// (with the 2-block bound it spilled; 65-column verify batches -- ngram-mod's 64-token drafts + 1 -- land here)
template <ggml_type type, int TN, bool ID, int TM = 4>
static __global__ void __launch_bounds__(256, TN >= 5 ? 1 : 2) tile_mm(const tile_mm_params prm) {
    using tr = tile_mm_traits<type>;
    constexpr int BM = 16*TM, BN = 16*TN, RPW = BM/8, RW = tr::RW, UV = tr::UV, SPU = UV/64;
    constexpr int XI = (BN*8 + 255)/256;
    constexpr int NRAW = tr::flat ? (BM*RW + 255)/256 : 2*RPW;
    extern __shared__ __align__(16) unsigned char tmm_smem[];
    uint32_t * Rs = (uint32_t *) tmm_smem;              // [BM][RW] raw bytes of the current unit
    half2    * Ws = (half2 *) (Rs + BM*RW);             // [BM][TMM_LD] decoded weights of one step
    half2    * Xs = Ws + BM*TMM_LD;                     // [BN][TMM_LD] activations of one step (rotated k-pairs)
    __shared__ int Roff[BM];                            // byte offset of the unit inside its staged row (0 or 2; q6_K only)

    const int t = threadIdx.x, rg = t % 16, cg = t / 16, lane = t % WARP_SIZE, wid = t / WARP_SIZE;
    const int M = prm.M, K = prm.K, nwin = prm.nwin;
    const int m0 = blockIdx.x*BM;
    const int nu = K/UV;
    const void   * W  = prm.W;
    const __half * XS = prm.XS;
    const float  * S  = prm.S;
    int n0, N, u0, u1, base = 0;
    if constexpr (ID) {
        const int tile = blockIdx.y;
        if (tile >= *prm.ntiles) {
            return;                                    // whole block: the grid is sized for the worst case
        }
        const int e = prm.tiles[2*tile];
        n0   = prm.tiles[2*tile + 1];
        base = prm.bounds[e];
        N    = prm.bounds[e + 1] - base;
        W    = e < prm.split ? (const unsigned char *) prm.W2 + e*prm.expert_bytes
                             : (const unsigned char *) W + (e - prm.split)*prm.expert_bytes;
        if (!prm.xrow) {
            XS += (int64_t) base*K;
            S  += (int64_t) base*nwin;
        }
        u0 = 0;
        u1 = nu;
    } else {
        n0 = blockIdx.y*BN;
        N  = prm.N;
        u0 = blockIdx.z*prm.u_per;
        u1 = min(nu, u0 + prm.u_per);
    }
    const int64_t row_bytes = prm.row_bytes;

    // staging: per-warp rows (warp w: rows w + 8i, lane j holds words j and 32 + j of the row's aligned-down unit copy) or, for short
    // rows, one flat list (word it = r*RW + w of the tile)
    uint32_t raw[NRAW];
#define TMM_FETCH(uu) do {                                                                                            \
        if constexpr (tr::flat) {                                                                                     \
            _Pragma("unroll") for (int v = 0; v < NRAW; ++v) {                                                        \
                const int it = t + 256*v, r = it / RW, w = it % RW;                                                   \
                raw[v] = it < BM*RW ? __ldg((const uint32_t *) ((uintptr_t) tile_mm_unit<type>(W, min(m0 + r, M - 1), row_bytes, (uu)) & ~(uintptr_t) 3) + w) : 0u; \
            }                                                                                                         \
        } else {                                                                                                      \
            _Pragma("unroll") for (int i = 0; i < RPW; ++i) {                                                         \
                const int m = min(m0 + wid + 8*i, M - 1);                                                             \
                const uint32_t * w = (const uint32_t *) ((uintptr_t) tile_mm_unit<type>(W, m, row_bytes, (uu)) & ~(uintptr_t) 3); \
                raw[2*i]     = __ldg(w + lane);                                                                       \
                raw[2*i + 1] = lane < RW - 32 ? __ldg(w + 32 + lane) : 0u;                                            \
            }                                                                                                         \
        } } while (0)

    float accf[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            accf[i][j] = 0.0f;
        }
    }

    // this thread's activation rows (fixed for the block): XS offsets of its XI staged columns and S offsets of its TN columns
    const int * xrow = ID ? prm.xrow : nullptr;
    int64_t xoff[XI], soff[TN];
#pragma unroll
    for (int v = 0; v < XI; ++v) {
        const int it = t + 256*v, n = n0 + it % BN;
        xoff[v] = it < BN*8 && n < N ? (int64_t) (xrow ? __ldg(xrow + base + n) : n)*K + 8*(it / BN) : -1;
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int n = n0 + cg*TN + j;
        soff[j] = n < N ? (int64_t) (xrow ? __ldg(xrow + base + n) : n)*nwin : -1;
    }
    // the next step's activations and window scales, in registers
    uint4 xr[XI];
    float swr[TN];
#define TMM_XFETCH(kk) do {                                                                                           \
        _Pragma("unroll") for (int v = 0; v < XI; ++v) {                                                              \
            xr[v] = xoff[v] >= 0 ? __ldg((const uint4 *) (XS + xoff[v] + (kk))) : make_uint4(0, 0, 0, 0);            \
        }                                                                                                             \
        _Pragma("unroll") for (int j = 0; j < TN; ++j) {                                                              \
            swr[j] = soff[j] >= 0 ? S[soff[j] + (kk)/TMM_WIN]*(1.0f/1024.0f) : 0.0f;                                  \
        } } while (0)
    if (u0 < u1) {
        TMM_FETCH(u0);
        TMM_XFETCH(u0*UV);
    }
    for (int u = u0; u < u1; ++u) {
        __syncthreads();                               // the previous unit is done with Rs
        if constexpr (tr::flat) {
#pragma unroll
            for (int v = 0; v < NRAW; ++v) {
                const int it = t + 256*v;
                if (it < BM*RW) {
                    Rs[it] = raw[v];
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < RPW; ++i) {
                uint32_t * rr = Rs + (wid + 8*i)*RW;
                rr[lane] = raw[2*i];
                if (lane < RW - 32) {
                    rr[32 + lane] = raw[2*i + 1];
                }
            }
        }
        if (!tr::aligned && t < BM) {
            Roff[t] = (int) (((uintptr_t) tile_mm_unit<type>(W, min(m0 + t, M - 1), row_bytes, u)) & 3);
        }
        if (u + 1 < u1) {
            TMM_FETCH(u + 1);                          // in flight during this unit's steps
        }
#pragma unroll 1
        for (int st = 0; st < SPU; ++st) {
            __syncthreads();                           // Rs staged / the previous step is done with Ws, Xs
#pragma unroll 2
            for (int r = wid; r < BM; r += 8) {
                const unsigned char * rb = (const unsigned char *) (Rs + r*RW) + (tr::aligned ? 0 : Roff[r]);
                #ifdef TMM_STUB
                Ws[r*TMM_LD + lane] = __halves2half2(__ushort_as_half(rb[lane] | 0x3c00), __ushort_as_half(0x3c00));
#else
                Ws[r*TMM_LD + lane] = m0 + r < M ? tile_mm_decode<type>(rb, st, lane) : __float2half2_rn(0.0f);
#endif
            }
            const int k0 = u*UV + st*64;
            {   // activations: small and L2-resident, rotated by one k-pair inside each group of 4; fetched one step ahead
#pragma unroll
                for (int v = 0; v < XI; ++v) {
                    const int it = t + 256*v, c = it % BN, q = it / BN;
                    if (it < BN*8) {
                        const uint4 x = xr[v];
                        *(uint4 *) (Xs + c*TMM_LD + 4*q) = make_uint4(x.y, x.z, x.w, x.x);
                    }
                }
            }
            float sw[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                sw[j] = swr[j];
            }
            if (k0 + 64 < u1*UV) {
                TMM_XFETCH(k0 + 64);
            }
            __syncthreads();
            half2 acc[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = __float2half2_rn(0.0f);
                }
            }
#pragma unroll
            for (int kq = 0; kq < TMM_KP; kq += 4) {
                uint4 x4[TN];
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    x4[j] = *(const uint4 *) (Xs + (cg*TN + j)*TMM_LD + kq);
                }
#pragma unroll
                for (int i = 0; i < TM; ++i) {
                    const uint4 w4 = *(const uint4 *) (Ws + (rg + 16*i)*TMM_LD + kq);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
#pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            acc[i][j] = __hfma2(((const half2 *) &w4)[e], ((const half2 *) &x4[j])[(e + 3) & 3], acc[i][j]);
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const float2 f = __half22float2(acc[i][j]);
                    accf[i][j] = fmaf(f.x + f.y, sw[j], accf[i][j]);
                }
            }
        }
    }
#undef TMM_FETCH
#undef TMM_XFETCH
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int n = n0 + cg*TN + j;
        if (n >= N) {
            continue;
        }
        float * out;
        if constexpr (ID) {
            out = prm.dst + (int64_t) prm.ids_dst[base + n]*prm.s1;
        } else {
            out = prm.P + ((int64_t) blockIdx.z*N + n)*M;
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int m = m0 + rg + 16*i;
            if (m < M) {
                out[m] = accf[i][j];
            }
        }
    }
}

// MUL_MAT_ID: build the tile list (expert, first column) from the expert bounds; one thread, n_experts is small
static __global__ void tile_mm_id_tiles(const int * __restrict__ bounds, const int n_experts, const int bn, int * __restrict__ tiles,
                                        int * __restrict__ ntiles) {
    int k = 0;
    for (int e = 0; e < n_experts; ++e) {
        const int n = bounds[e + 1] - bounds[e];
        for (int c = 0; c < n; c += bn) {
            tiles[2*k]     = e;
            tiles[2*k + 1] = c;
            ++k;
        }
    }
    *ntiles = k;
}

// dst[n][m] (row stride sd floats) = sum over the splits, in a fixed order
static __global__ void tile_mm_reduce(const float * __restrict__ P, float * __restrict__ D, const int M, const int N,
                                        const int64_t sd, const int splits) {
    const int64_t NM = (int64_t) N*M;
    for (int64_t i = blockIdx.x*(int64_t) blockDim.x + threadIdx.x; i < NM; i += (int64_t) gridDim.x*blockDim.x) {
        float s = 0.0f;
        for (int z = 0; z < splits; ++z) {
            s += P[z*NM + i];
        }
        D[(i / M)*sd + i % M] = s;
    }
}

template <ggml_type type, int TN, bool ID, int TM>
static void tile_mm_launch_tm(const tile_mm_params & prm, const int splits_or_tiles, cudaStream_t stream) {
    constexpr int BN = 16*TN, BM = 16*TM;
    const size_t smem = (size_t) BM*tile_mm_traits<type>::RW*sizeof(uint32_t) + (size_t) (BM + BN)*TMM_LD*sizeof(half2);
    const dim3 grid((prm.M + BM - 1)/BM, ID ? splits_or_tiles : (prm.N + BN - 1)/BN, ID ? 1 : splits_or_tiles);
    tile_mm<type, TN, ID, TM><<<grid, 256, smem, stream>>>(prm);
}

template <ggml_type type, int TN, bool ID>
static void tile_mm_launch(const tile_mm_params & prm, const int splits_or_tiles, cudaStream_t stream) {
    if constexpr (ID && TN >= 3 && TN <= 4) {
        if (prm.tm == 5) {   // 80-row blocks: more outputs per staged activation (MoE prefill, M % 80 == 0)
            tile_mm_launch_tm<type, TN, ID, 5>(prm, splits_or_tiles, stream);
            return;
        }
    }
    constexpr int BN = 16*TN;
    const size_t smem = (size_t) TMM_BM*tile_mm_traits<type>::RW*sizeof(uint32_t) + (size_t) (TMM_BM + BN)*TMM_LD*sizeof(half2);
    const dim3 grid((prm.M + TMM_BM - 1)/TMM_BM, ID ? splits_or_tiles : (prm.N + BN - 1)/BN, ID ? 1 : splits_or_tiles);
    tile_mm<type, TN, ID><<<grid, 256, smem, stream>>>(prm);
}

template <ggml_type type, bool ID>
static void tile_mm_launch_tn(const int tn, const tile_mm_params & prm, const int splits_or_tiles, cudaStream_t stream) {
    switch (tn) {
        case 1:  tile_mm_launch<type, 1, ID>(prm, splits_or_tiles, stream); break;
        case 2:  tile_mm_launch<type, 2, ID>(prm, splits_or_tiles, stream); break;
        case 3:  tile_mm_launch<type, 3, ID>(prm, splits_or_tiles, stream); break;
        case 4:  tile_mm_launch<type, 4, ID>(prm, splits_or_tiles, stream); break;
        case 6:  tile_mm_launch<type, 6, ID>(prm, splits_or_tiles, stream); break;
        default: tile_mm_launch<type, 5, ID>(prm, splits_or_tiles, stream); break;
    }
}

template <bool ID>
static void tile_mm_dispatch(const ggml_type type, const int tn, const tile_mm_params & prm, const int splits_or_tiles, cudaStream_t stream) {
    if (type == GGML_TYPE_Q2_0 && prm.K % (2*QK2_0) != 0) {   // odd block count per row: one-block units
        tile_mm_launch_tn<TMM_Q2_0_B1, ID>(tn, prm, splits_or_tiles, stream);
        return;
    }
    switch (type) {
        case GGML_TYPE_Q6_K: tile_mm_launch_tn<GGML_TYPE_Q6_K, ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_Q5_K: tile_mm_launch_tn<GGML_TYPE_Q5_K, ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_Q4_K: tile_mm_launch_tn<GGML_TYPE_Q4_K, ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_Q8_0: tile_mm_launch_tn<GGML_TYPE_Q8_0, ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_Q4_0: tile_mm_launch_tn<GGML_TYPE_Q4_0, ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_Q2_0:    tile_mm_launch_tn<GGML_TYPE_Q2_0,    ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_IQ2_S:   tile_mm_launch_tn<GGML_TYPE_IQ2_S,   ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_IQ2_XXS: tile_mm_launch_tn<GGML_TYPE_IQ2_XXS, ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_IQ1_M:   tile_mm_launch_tn<GGML_TYPE_IQ1_M,   ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_IQ3_S:   tile_mm_launch_tn<GGML_TYPE_IQ3_S,   ID>(tn, prm, splits_or_tiles, stream); break;
        case GGML_TYPE_IQ4_NL:  tile_mm_launch_tn<GGML_TYPE_IQ4_NL,  ID>(tn, prm, splits_or_tiles, stream); break;
        default: GGML_ABORT("tile_mm: unsupported type");
    }
}

// GGML_CUDA_TILE_STATS=1 (or the older GGML_CUDA_TILE_Q6K_STATS=1): histogram of the column counts of every q6_K / q8_0 matmul
// offered to this hook (all widths > 8 on Pascal reach it), printed to stderr at exit. Diagnostics only.
static std::atomic<uint64_t> tmm_hist[5][2][9];   // [type: q6_K, q8_0, q4_0, q5_K, q4_K][taken][bucket]
static const char * tmm_bucket_name[9] = {"9-16", "17-32", "33-48", "49-64", "65-80", "81-128", "129-256", "257-512", ">512"};
static void tmm_stats_print() {
    const char * tname[5] = {"q6_K", "q8_0", "q4_0", "q5_K", "q4_K"};
    for (int ty = 0; ty < 5; ++ty) {
        fprintf(stderr, "tile_mm stats %s (matmul calls by column count): taken / cuBLAS\n", tname[ty]);
        for (int b = 0; b < 9; ++b) {
            fprintf(stderr, "  %-8s %10llu / %10llu\n", tmm_bucket_name[b], (unsigned long long) tmm_hist[ty][1][b].load(),
                    (unsigned long long) tmm_hist[ty][0][b].load());
        }
    }
}
static void tmm_stats_note(const ggml_type type, const int64_t n, const bool taken) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_TILE_STATS");
        if (!e) {
            e = getenv("GGML_CUDA_TILE_Q6K_STATS");
        }
        if (e && e[0] == '1') {
            atexit(tmm_stats_print);
            return true;
        }
        return false;
    }();
    if (!on) {
        return;
    }
    const int b = n <= 16 ? 0 : n <= 32 ? 1 : n <= 48 ? 2 : n <= 64 ? 3 : n <= 80 ? 4 : n <= 128 ? 5 : n <= 256 ? 6 : n <= 512 ? 7 : 8;
    tmm_hist[type == GGML_TYPE_Q8_0 ? 1 : type == GGML_TYPE_Q4_0 ? 2 : type == GGML_TYPE_Q5_K ? 3 : type == GGML_TYPE_Q4_K ? 4 : 0][taken][b]++;
}

static bool tmm_env_on(const char * name) {
    const char * e = getenv(name);
    return !e || e[0] != '0';
}

// the weight-layout conditions shared by both entry points (the staging reads whole 32-bit words)
static bool tmm_weights_ok(const ggml_tensor * src0) {
    static const bool q6k_on = tmm_env_on("GGML_CUDA_TILE_Q6K"), q80_on = tmm_env_on("GGML_CUDA_TILE_Q8_0"), q40_on = tmm_env_on("GGML_CUDA_TILE_Q4_0");
    static const bool q5k_on = tmm_env_on("GGML_CUDA_TILE_Q5K"), q4k_on = tmm_env_on("GGML_CUDA_TILE_Q4K");
    // GGML_CUDA_TILE_LB=0: Q2_0 / IQ2_S / IQ2_XXS / IQ1_M back to MMQ / cuBLAS
    static const bool lb_on = tmm_env_on("GGML_CUDA_TILE_LB");
    const int64_t K = src0->ne[0], M = src0->ne[1];
    switch (src0->type) {
        case GGML_TYPE_Q6_K: {
            // the aligned-down 53-word staging reads up to 2 bytes past a 4-aligned block but never past the matrix when M*nbk is even
            // (its last block then ends 4-aligned)
            const int64_t nbk = K/QK_K;
            return q6k_on && K % QK_K == 0 && (M*nbk) % 2 == 0 && src0->nb[1] == (size_t) nbk*sizeof(block_q6_K) &&
                   ((uintptr_t) src0->data % 2) == 0 && src0->nb[2] % 4 == 0;
        }
        case GGML_TYPE_Q5_K:
            return q5k_on && K % QK_K == 0 && src0->nb[1] == (size_t) (K/QK_K)*sizeof(block_q5_K) &&
                   ((uintptr_t) src0->data % 4) == 0 && src0->nb[2] % 4 == 0;
        case GGML_TYPE_Q4_K:
            return q4k_on && K % QK_K == 0 && src0->nb[1] == (size_t) (K/QK_K)*sizeof(block_q4_K) &&
                   ((uintptr_t) src0->data % 4) == 0 && src0->nb[2] % 4 == 0;
        case GGML_TYPE_Q8_0:
            // units of four blocks: K % 128 keeps every row 4-aligned (even block count) and the units whole
            return q80_on && K % (4*QK8_0) == 0 && src0->nb[1] == (size_t) (K/QK8_0)*sizeof(block_q8_0) &&
                   ((uintptr_t) src0->data % 4) == 0 && src0->nb[2] % 4 == 0;
        case GGML_TYPE_Q4_0:
            // units of two blocks: K % 64 keeps every row 4-aligned
            return q40_on && K % (2*QK4_0) == 0 && src0->nb[1] == (size_t) (K/QK4_0)*sizeof(block_q4_0) &&
                   ((uintptr_t) src0->data % 4) == 0 && src0->nb[2] % 4 == 0;
        case GGML_TYPE_Q2_0: {
            // units of two blocks (36 bytes) when K % 128 == 0 (rows 4-aligned); otherwise one-block units staged from the aligned-down
            // address, which read up to 2 bytes past a block but never past a 4-aligned matrix (or expert) when M*nbk is even
            if (!lb_on || K % QK2_0 != 0 || src0->nb[1] != (size_t) (K/QK2_0)*sizeof(block_q2_0) || src0->nb[2] % 4 != 0) {
                return false;
            }
            if (K % (2*QK2_0) == 0) {
                return ((uintptr_t) src0->data % 4) == 0;
            }
            static const bool b1_on = tmm_env_on("GGML_CUDA_TILE_Q2_0_B1");
            return b1_on && ((uintptr_t) src0->data % 4) == 0 && (M*(K/QK2_0)) % 2 == 0;
        }
        case GGML_TYPE_IQ4_NL:
            // units of two blocks: K % 64 keeps every row 4-aligned (as q4_0)
            return lb_on && K % (2*QK4_NL) == 0 && src0->nb[1] == (size_t) (K/QK4_NL)*sizeof(block_iq4_nl) &&
                   ((uintptr_t) src0->data % 4) == 0 && src0->nb[2] % 4 == 0;
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ2_XXS: {
            // 2-aligned blocks staged from the aligned-down address: never past the matrix when M*nbk is even (as q6_K)
            const int64_t nbk = K/QK_K;
            return lb_on && K % QK_K == 0 && (M*nbk) % 2 == 0 && src0->nb[1] == (size_t) nbk*ggml_type_size(src0->type) &&
                   ((uintptr_t) src0->data % 2) == 0 && src0->nb[2] % 4 == 0;
        }
        case GGML_TYPE_IQ1_M:
            return lb_on && K % QK_K == 0 && src0->nb[1] == (size_t) (K/QK_K)*sizeof(block_iq1_m) &&
                   ((uintptr_t) src0->data % 4) == 0 && src0->nb[2] % 4 == 0;
        default:
            return false;
    }
}

static int tmm_uv(const ggml_type type, const int64_t K) {
    switch (type) {
        case GGML_TYPE_Q8_0: return tile_mm_traits<GGML_TYPE_Q8_0>::UV;
        case GGML_TYPE_Q4_0: return tile_mm_traits<GGML_TYPE_Q4_0>::UV;
        case GGML_TYPE_IQ4_NL: return tile_mm_traits<GGML_TYPE_IQ4_NL>::UV;
        case GGML_TYPE_Q2_0: return K % (2*QK2_0) == 0 ? tile_mm_traits<GGML_TYPE_Q2_0>::UV : tile_mm_traits<TMM_Q2_0_B1>::UV;
        default:             return QK_K;    // q6_K, q5_K, q4_K, IQ2_S, IQ2_XXS, IQ1_M
    }
}

static bool tmm_type_ok(const ggml_type type) {
    return type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ1_M ||
           type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL;
}

bool ggml_cuda_tile_mm_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // GGML_CUDA_TILE_Q6K=0 / _Q5K=0 / _Q4K=0 / _Q8_0=0 / _Q4_0=0 turn a type off; GGML_CUDA_TILE_MIN / _MAX (or the older _Q6K_MIN / _MAX) set the column
    // range that takes this path (default 9..127, q6_K 9..80 -- see below). 81..127 used to go to dequantize -> fp16 cuBLAS; phase TWV
    // (llama-bench, 81..127): gemma4 +8..14 %, qwen36 +8..10 %, KLD qwen36 0.00157 -> 0.00072 at -ub 96. (The tilemm2 harness's cuBLAS
    // baseline was slower than the in-model cuBLAS path, so its q6_K numbers overstated the win.)
    static const int nmin = [] { const char * e = getenv("GGML_CUDA_TILE_MIN"); if (!e) { e = getenv("GGML_CUDA_TILE_Q6K_MIN"); } return e ? atoi(e) : 9; }();
    static const int nmax = [] { const char * e = getenv("GGML_CUDA_TILE_MAX"); if (!e) { e = getenv("GGML_CUDA_TILE_Q6K_MAX"); } return e ? std::min(atoi(e), 127) : 127; }();
    const ggml_type type = src0->type;
    if (!tmm_type_ok(type) || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // q6_K keeps 80: in the model (phase TWV, llama-bench) Swift / glimmer's dense q6_K weights lost 12-15 % at 104..127 columns with the
    // tile kernel (cuBLAS on them is faster than the harness baseline suggested), while gemma4 / qwen36 gained 8-14 % at 88..127.
    // GGML_CUDA_TILE_MAX_Q6K overrides (capped at nmax).
    static const int nmax_q6k = [] { const char * e = getenv("GGML_CUDA_TILE_MAX_Q6K"); return e ? atoi(e) : 80; }();
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    const int nmax_t = type == GGML_TYPE_Q6_K ? std::min(nmax, nmax_q6k) : nmax;
    const bool ok = src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && N >= nmin && N <= nmax_t &&
                    src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
                    src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
                    (src1->nb[1] % sizeof(float)) == 0 && (dst->nb[1] % sizeof(float)) == 0 && tmm_weights_ok(src0);
    tmm_stats_note(type, N, ok);
    if (!ok) {
        return false;
    }
    // 81..96: one 96-column tile; 97..127: 64-column tiles (two along N) beat both a 128-column tile and cuBLAS
    const int tn   = N <= 16 ? 1 : N <= 32 ? 2 : N <= 48 ? 3 : N <= 64 ? 4 : N <= 80 ? 5 : N <= 96 ? 6 : 4;
    const int nwin = (int) ((K + TMM_WIN - 1)/TMM_WIN);
    const int nu   = (int) (K/tmm_uv(type, K));
    const int tiles = (int) (((M + TMM_BM - 1)/TMM_BM)*((N + 16*tn - 1)/(16*tn)));
    const int nsm  = ggml_cuda_info().devices[ctx.device].nsm;
    // split-K: fill ~8 blocks per SM, at least 512 values per split
    int splits = std::max(1, std::min((int) (K/512), (8*nsm + tiles - 1)/tiles));
    const int u_per = (nu + splits - 1)/splits;
    splits = (nu + u_per - 1)/u_per;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<__half> xs(ctx.pool(), N*K);
    ggml_cuda_pool_alloc<float>  sc(ctx.pool(), N*nwin);
    ggml_cuda_pool_alloc<float>  part(ctx.pool(), (int64_t) splits*N*M);
    {
        const dim3 pb(WARP_SIZE, 4), pg((nwin + 3)/4, (unsigned) N);
        tile_mm_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, (int64_t) (src1->nb[1]/sizeof(float)), nullptr, xs.get(), sc.get(),
                                            (int) K, nwin);
    }
    tile_mm_params prm = {};
    prm.W = src0->data; prm.row_bytes = (int64_t) src0->nb[1]; prm.XS = xs.get(); prm.S = sc.get(); prm.P = part.get();
    prm.M = (int) M; prm.K = (int) K; prm.N = (int) N; prm.nwin = nwin; prm.u_per = u_per;
    tile_mm_dispatch<false>(type, tn, prm, splits, stream);
    tile_mm_reduce<<<std::min<int64_t>(1024, (N*M + 255)/256), 256, 0, stream>>>(part.get(), (float *) dst->data, (int) M, (int) N,
                                                                                (int64_t) (dst->nb[1]/sizeof(float)), splits);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_tile_mm_id_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                              ggml_tensor * dst) {
    // GGML_CUDA_TILE_MMID=0: leave MUL_MAT_ID to MMQ (A/B)
    static const bool on = tmm_env_on("GGML_CUDA_TILE_MMID");
    const ggml_type type = src0->type;
    if (!on || !tmm_type_ok(type) || ggml_cuda_info().devices[ctx.device].cc != GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // tail LRU assembly (ggml_lru_assemble_tail): src0 holds experts [n_slot, n_expert), the pool (src0->src[1]) the first n_slot
    const ggml_tensor * tpool = src0->op == GGML_OP_LRU_ASSEMBLE && ggml_get_op_params_i32(src0, 5) != 0 ? src0->src[1] : nullptr;
    if (tpool != nullptr && (tpool->type != src0->type || tpool->nb[1] != src0->nb[1] || tpool->nb[2] != src0->nb[2] ||
                             !tmm_weights_ok(tpool))) {
        return false;
    }
    // zero-copy tail (ggml_lru_assemble_zerocopy): the host experts are read in place from this device's region of the host pool
    const int n_zc = tpool ? ggml_get_op_params_i32(src0, 6) : 0;
    const void * zc_base = nullptr;
    if (n_zc > 0) {
        ggml_tensor * hsrc;
        memcpy(&hsrc, src0->op_params, sizeof(hsrc));
        const size_t es = tpool->nb[2];
        int xdev = 0;
        if (hsrc->nb[2] != es) {
            cudaGetDevice(&xdev);
        }
        zc_base = (const char *) hsrc->data + (size_t) xdev * (size_t) hsrc->ne[2] * es;
    }
    const int64_t K = src0->ne[0], M = src0->ne[1], n_experts = (n_zc > 0 ? n_zc : src0->ne[2]) + (tpool ? tpool->ne[2] : 0);
    const int64_t n_used = ids->ne[0], n_tokens = src1->ne[2], nch = src1->ne[1];
    // dst [M, n_used, n_tokens], rows addressed as token*n_used + slot; src1 [K, nch (1 = shared by the slots, or n_used), n_tokens]
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || !tmm_weights_ok(src0) ||
            src0->ne[3] != 1 || src1->ne[3] != 1 || dst->ne[1] != n_used || ids->ne[1] != n_tokens || (nch != 1 && nch != n_used) ||
            src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 || src1->nb[2] % src1->nb[1] != 0 ||
            dst->nb[1] % sizeof(float) != 0 || dst->nb[2] != dst->ne[1]*dst->nb[1] || ids->nb[0] != sizeof(int32_t) ||
            (size_t) n_tokens*sizeof(int32_t) > ggml_cuda_info().devices[ctx.device].smpbo || n_used > 32) {
        return false;
    }
    const int64_t total = n_tokens*n_used;
    const int64_t avg   = (total + n_experts - 1)/n_experts;
    const int tn   = avg <= 16 ? 1 : avg <= 32 ? 2 : avg <= 48 ? 3 : 4;
    const int bn   = 16*tn;
    const int nwin = (int) ((K + TMM_WIN - 1)/TMM_WIN);
    const int max_tiles = (int) ((total + bn - 1)/bn + n_experts);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), total);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), total);
    ggml_cuda_pool_alloc<int32_t> bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> tiles(ctx.pool(), 2*max_tiles + 1);
    const int si1  = (int) (ids->nb[1]/sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2]/src1->nb[1]);
    // one activation row shared by the slots (gate/up): prep each token once, the GEMM finds it through ids_src1
    static const bool per_token = [] { const char * e = getenv("GGML_CUDA_TMM_XTOKEN"); return !e || atoi(e) != 0; }();
    const bool xtok = per_token && nch == 1 && sis1 == 1;
    const int64_t nx = xtok ? n_tokens : total;
    ggml_cuda_pool_alloc<__half>  xs(ctx.pool(), nx*K);
    ggml_cuda_pool_alloc<float>   sc(ctx.pool(), nx*nwin);
    // skipped slots (id -1, [TAG_MUL_MAT_ID_SKIP]) get no column: the gather below still walks all `total` compact columns,
    // so the unused tail must hold valid indices, and their dst rows are zeroed (the GEMM never writes them)
    CUDA_CHECK(cudaMemsetAsync(ids_src1.get(), 0, total*sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(ids_dst.get(),  0, total*sizeof(int32_t), stream));
    tile_mm_zero_skipped<<<(int) n_tokens, 256, 0, stream>>>((const int32_t *) ids->data, (float *) dst->data, (int) M, (int) n_used, si1,
                                         (int64_t) (dst->nb[1]/sizeof(float)), (int64_t) (dst->nb[2]/sizeof(float)));
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(), (int) n_experts, (int) n_tokens,
                                   (int) n_used, (int) nch, si1, sis1, /*write_inverse =*/ false, stream);
    tile_mm_id_tiles<<<1, 1, 0, stream>>>(bounds.get(), (int) n_experts, bn, tiles.get() + 1, tiles.get());
    {
        const dim3 pb(WARP_SIZE, 4), pg((nwin + 3)/4, (unsigned) nx);
        tile_mm_prep<<<pg, pb, 0, stream>>>((const float *) src1->data, (int64_t) (src1->nb[1]/sizeof(float)), xtok ? nullptr : ids_src1.get(), xs.get(),
                                            sc.get(), (int) K, nwin);
    }
    tile_mm_params prm = {};
    prm.W = zc_base ? zc_base : src0->data; prm.row_bytes = (int64_t) src0->nb[1]; prm.expert_bytes = (int64_t) src0->nb[2];
    prm.W2 = tpool ? tpool->data : nullptr; prm.split = tpool ? (int) tpool->ne[2] : 0;
    prm.XS = xs.get(); prm.S = sc.get(); prm.dst = (float *) dst->data; prm.s1 = (int64_t) (dst->nb[1]/sizeof(float));
    prm.ids_dst = ids_dst.get(); prm.xrow = xtok ? ids_src1.get() : nullptr; prm.bounds = bounds.get(); prm.tiles = tiles.get() + 1; prm.ntiles = tiles.get();
    prm.M = (int) M; prm.K = (int) K; prm.N = 0; prm.nwin = nwin;
    static const int tm_env = [] { const char * e = getenv("GGML_CUDA_TMM_TM"); return e ? atoi(e) : 5; }();
    prm.tm = tm_env == 5 && M % 80 == 0 ? 5 : 4;
    tile_mm_dispatch<true>(type, tn, prm, max_tiles, stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
