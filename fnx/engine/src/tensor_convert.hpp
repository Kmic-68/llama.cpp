// strata-split: load-time conversion of the small tensors of a GGUF whose types the engine does not read, to the types it does
// (what Unsloth's UD-Q4_K_XL file carries: Q8_0 for every dense matrix, the shared experts, the embeddings, the hyper-connection
// and PLE projections; F32 for the routers, the injections and the SSM gates). A set with other types for them (the ISTA-DAS IQ3_S
// set keeps K-quants, IQ4_XS and BF16 there) is dequantized to float and quantized again, once, at load. A tensor that already has the
// target type is never touched, so UD-Q4_K_XL and Q8_0 files load exactly as before. Pure host code, no CUDA.
//
// The dequantizers are ggml's reference ones (ggml-quants.c, dequantize_row_*), written out here because the engine does not link
// ggml's CPU library; tests/split/ check them against gguf-py on real tensors.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.hpp"

namespace fnx {
namespace conv {

inline float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float f;
    if (e == 0) f = std::ldexp((float) m, -24);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = std::ldexp((float) (m + 1024), (int) e - 25);
    return s ? -f : f;
}
inline uint16_t f2h(float x) {   // round to nearest even, as ggml's GGML_FP32_TO_FP16
    uint32_t b;
    std::memcpy(&b, &x, 4);
    const uint32_t s = (b >> 16) & 0x8000, e = (b >> 23) & 0xFF, m = b & 0x7FFFFF;
    if (e == 0xFF) return (uint16_t) (s | 0x7C00 | (m ? 0x200 : 0));
    int ne = (int) e - 127 + 15;
    if (ne >= 31) return (uint16_t) (s | 0x7C00);
    if (ne <= 0) {
        if (ne < -10) return (uint16_t) s;
        const uint32_t mm = (m | 0x800000) >> (1 - ne);
        uint32_t r = mm >> 13;
        const uint32_t rem = mm & 0x1FFF;
        if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) ne << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) ++r;
    return (uint16_t) (s | r);
}
inline float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline const int8_t* kvalues_iq4nl() {
    static const int8_t k[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    return k;
}
// IQ3_S's grid (ggml-common.h iq3s_grid): 4 magnitudes a row, one byte each
inline const uint32_t* iq3s_grid() {
    static const uint32_t g[512] = {
        0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
        0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
        0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
        0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
        0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
        0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
        0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
        0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
        0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
        0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
        0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
        0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
        0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
        0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
        0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
        0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
        0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
        0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
        0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
        0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
        0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
        0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
        0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
        0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
        0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
        0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
        0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
        0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
        0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
        0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
        0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
        0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
        0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
        0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
        0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
        0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
        0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
        0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
        0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
        0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
        0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
        0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
        0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
        0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
        0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
        0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
        0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
        0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
        0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
        0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
        0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
        0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
        0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
        0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
        0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
        0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
        0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
        0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
        0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
        0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
        0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
        0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
        0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
        0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101};
    return g;
}
// the K-quants' 6-bit scale and min of sub-block j (ggml get_scale_min_k4)
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// n values (a whole number of blocks) of `type` at `src`, as floats; throws on a type it has no reader for
inline void dequantize(uint32_t type, const uint8_t* src, size_t n, float* y) {
    switch (type) {
    case T_F32: std::memcpy(y, src, n * 4); return;
    case T_F16:
        for (size_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); y[i] = h2f(h); }
        return;
    case T_BF16:
        for (size_t i = 0; i < n; ++i) { uint16_t h; std::memcpy(&h, src + 2 * i, 2); y[i] = bf2f(h); }
        return;
    case T_Q8_0:
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = src + b * 34;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 32; ++j) y[b * 32 + j] = d * (float) (int8_t) p[2 + j];
        }
        return;
    case T_Q4_0:
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = src + b * 18;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 16; ++j) {
                y[b * 32 + j] = d * (float) ((int) (p[2 + j] & 0xF) - 8);
                y[b * 32 + j + 16] = d * (float) ((int) (p[2 + j] >> 4) - 8);
            }
        }
        return;
    case T_Q2_0:
        for (size_t b = 0; b < n / 64; ++b) {
            const uint8_t* p = src + b * 18;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 64; ++j) y[b * 64 + j] = d * (float) ((int) ((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1);
        }
        return;
    case T_IQ4_NL:
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* p = src + b * 18;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            for (int j = 0; j < 16; ++j) {
                y[b * 32 + j] = d * (float) kvalues_iq4nl()[p[2 + j] & 0xF];
                y[b * 32 + j + 16] = d * (float) kvalues_iq4nl()[p[2 + j] >> 4];
            }
        }
        return;
    case T_IQ4_XS:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 136;
            uint16_t dh, sh;
            std::memcpy(&dh, p, 2);
            std::memcpy(&sh, p + 2, 2);
            const uint8_t* sl = p + 4;
            const uint8_t* qs = p + 8;
            const float d = h2f(dh);
            float* o = y + b * 256;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                const float dl = d * (float) (ls - 32);
                for (int j = 0; j < 16; ++j) {
                    o[j] = dl * (float) kvalues_iq4nl()[qs[j] & 0xF];
                    o[j + 16] = dl * (float) kvalues_iq4nl()[qs[j] >> 4];
                }
                o += 32;
                qs += 16;
            }
        }
        return;
    case T_Q4_K:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 144;
            uint16_t dh, mh;
            std::memcpy(&dh, p, 2);
            std::memcpy(&mh, p + 2, 2);
            const float d = h2f(dh), mn = h2f(mh);
            const uint8_t* sc = p + 4;
            const uint8_t* q = p + 16;
            float* o = y + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is, sc, s1, m1);
                scale_min_k4(is + 1, sc, s2, m2);
                const float d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
                for (int l = 0; l < 32; ++l) *o++ = d1 * (float) (q[l] & 0xF) - mm1;
                for (int l = 0; l < 32; ++l) *o++ = d2 * (float) (q[l] >> 4) - mm2;
                q += 32;
                is += 2;
            }
        }
        return;
    case T_Q5_K:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 176;
            uint16_t dh, mh;
            std::memcpy(&dh, p, 2);
            std::memcpy(&mh, p + 2, 2);
            const float d = h2f(dh), mn = h2f(mh);
            const uint8_t* sc = p + 4;
            const uint8_t* qh = p + 16;
            const uint8_t* ql = p + 48;
            float* o = y + b * 256;
            int is = 0;
            uint8_t u1 = 1, u2 = 2;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                scale_min_k4(is, sc, s1, m1);
                scale_min_k4(is + 1, sc, s2, m2);
                const float d1 = d * s1, mm1 = mn * m1, d2 = d * s2, mm2 = mn * m2;
                for (int l = 0; l < 32; ++l) *o++ = d1 * (float) ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - mm1;
                for (int l = 0; l < 32; ++l) *o++ = d2 * (float) ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - mm2;
                ql += 32;
                is += 2;
                u1 = (uint8_t) (u1 << 2);
                u2 = (uint8_t) (u2 << 2);
            }
        }
        return;
    case T_Q6_K:
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 210;
            const uint8_t* ql = p;
            const uint8_t* qh = p + 128;
            const int8_t* sc = (const int8_t*) (p + 192);
            uint16_t dh;
            std::memcpy(&dh, p + 208, 2);
            const float d = h2f(dh);
            float* o = y + b * 256;
            for (int h = 0; h < 256; h += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    o[l] = d * (float) sc[is] * (float) q1;
                    o[l + 32] = d * (float) sc[is + 2] * (float) q2;
                    o[l + 64] = d * (float) sc[is + 4] * (float) q3;
                    o[l + 96] = d * (float) sc[is + 6] * (float) q4;
                }
                o += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return;
    case T_IQ3_S:   // ggml dequantize_row_iq3_s: blocks of d, qs[64], qh[8], signs[32], scales[4]
        for (size_t b = 0; b < n / 256; ++b) {
            const uint8_t* p = src + b * 110;
            uint16_t dh;
            std::memcpy(&dh, p, 2);
            const float d = h2f(dh);
            const uint8_t *qs = p + 2, *qh = p + 66, *signs = p + 74, *scales = p + 106;
            float* o = y + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                const float db1 = d * (float) (1 + 2 * (scales[ib32 / 2] & 0xf)), db2 = d * (float) (1 + 2 * (scales[ib32 / 2] >> 4));
                for (int part = 0; part < 2; ++part) {
                    const float db = part ? db2 : db1;
                    for (int l = 0; l < 4; ++l) {
                        const uint32_t g1 = iq3s_grid()[qs[2 * l] | ((qh[part] << (8 - 2 * l)) & 256)];
                        const uint32_t g2 = iq3s_grid()[qs[2 * l + 1] | ((qh[part] << (7 - 2 * l)) & 256)];
                        for (int j = 0; j < 4; ++j) {
                            o[j] = db * (float) ((g1 >> (8 * j)) & 255) * (signs[l] & (1u << j) ? -1.f : 1.f);
                            o[j + 4] = db * (float) ((g2 >> (8 * j)) & 255) * (signs[l] & (1u << (j + 4)) ? -1.f : 1.f);
                        }
                        o += 8;
                    }
                    qs += 8;
                    signs += 4;
                }
                qh += 2;
            }
        }
        return;
    }
    throw std::runtime_error("tensor_convert: no reader for ggml type " + std::to_string(type));
}

// Q8_0 as ggml's quantize_row_q8_0_ref: per 32 values d = amax / 127 (stored as f16), q = round(x / d)
inline void quantize_q8_0(const float* x, size_t n, uint8_t* dst) {
    for (size_t b = 0; b < n / 32; ++b) {
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.f ? 1.0f / d : 0.f;
        const uint16_t dh = f2h(d);
        std::memcpy(dst + b * 34, &dh, 2);
        for (int j = 0; j < 32; ++j) dst[b * 34 + 2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
}

// the type the engine reads a tensor of this name as (what UD-Q4_K_XL carries), or 0xFFFFFFFF for a tensor that is left alone
inline uint32_t wanted_type(const std::string& name) {
    auto ends = [&](const char* s) { const size_t l = std::strlen(s); return name.size() >= l && name.compare(name.size() - l, l, s) == 0; };
    static const char* q8[] = {".attn_gate.weight", ".attn_k.weight", ".attn_output.weight", ".attn_q.weight", ".attn_qkv.weight", ".attn_v.weight",
                               ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight", ".hc_attn_down.weight",
                               ".hc_attn_up.weight", ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".ple_key.weight", ".ple_value.weight",
                               ".ssm_out.weight"};
    static const char* f32[] = {".ffn_gate_inp.weight", ".ffn_gate_inp_shexp.weight", ".hc_attn_inject.weight", ".hc_ffn_inject.weight",
                                ".ssm_alpha.weight", ".ssm_beta.weight", ".ple_conv1d.weight"};
    for (const char* s : q8)
        if (ends(s)) return T_Q8_0;
    for (const char* s : f32)
        if (ends(s)) return T_F32;
    if (name == "output.weight" || name == "output_hc_down.weight" || name == "output_hc_up.weight" || name == "token_embd.weight") return T_Q8_0;
    return 0xFFFFFFFFu;
}

// `t` converted to `target` (Q8_0 or F32) into `out`; the Tensor then points at `out`
inline void convert(Tensor& t, uint32_t target, std::vector<uint8_t>& out) {
    size_t n = 1;
    for (int d = 0; d < 4; ++d) n *= (size_t) t.ne[d];
    const TypeInfo from = type_info(t.type);
    if (n % (size_t) from.block) throw std::runtime_error(t.name + ": not a whole number of blocks");
    if (target == T_Q8_0 && t.ne[0] % 32) throw std::runtime_error(t.name + ": rows are not a whole number of Q8_0 blocks");
    std::vector<float> f(n);
    dequantize(t.type, t.data, n, f.data());
    if (target == T_F32) {
        out.resize(n * 4);
        std::memcpy(out.data(), f.data(), n * 4);
    } else if (target == T_Q8_0) {
        out.resize(n / 32 * 34);
        quantize_q8_0(f.data(), n, out.data());
    } else {
        throw std::runtime_error("tensor_convert: target type " + std::to_string(target));
    }
    t.type = target;
    t.data = out.data();
    t.nbytes = out.size();
    t.shard = -1;   // not in a GGUF file any more: readers that stage from the file by offset must take the bytes at t.data
    t.offset = 0;
}

// Strata's Q2_0 (64 values in 18 bytes: f16 d, two bits per value, value j in byte j / 4 at bit 2 (j % 4), meaning (code - 1) * d) as two
// Q4_0 blocks (32 values in 18 bytes: f16 d, nibble i of the low half = value i, of the high half = value i + 16, meaning (nibble - 8) * d).
// Lossless: nibble = code + 7 gives (code + 7 - 8) * d, the same product. The engine's Q4_0 kernels then read it (the draft layer's
// experts.bin goes the same way, model.cpp Mtp::experts). Twice the bytes.
inline void repack_q2_0_to_q4_0(Tensor& t, std::vector<uint8_t>& out) {
    if (t.ne[0] % 64) throw std::runtime_error(t.name + ": rows are not a whole number of Q2_0 blocks");
    const size_t nb = t.nbytes / 18;
    out.resize(nb * 36);
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* in = t.data + b * 18;
        auto code = [&](int j) { return (in[2 + j / 4] >> (2 * (j % 4))) & 3; };
        for (int h = 0; h < 2; ++h) {
            uint8_t* o = out.data() + (2 * b + (size_t) h) * 18;
            o[0] = in[0];
            o[1] = in[1];
            for (int i = 0; i < 16; ++i) o[2 + i] = (uint8_t) ((code(32 * h + i) + 7) | ((code(32 * h + 16 + i) + 7) << 4));
        }
    }
    t.type = T_Q4_0;
    t.data = out.data();
    t.nbytes = out.size();
    t.shard = -1;
    t.offset = 0;
}

}  // namespace conv
}  // namespace fnx
