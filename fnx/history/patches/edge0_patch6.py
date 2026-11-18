#!/usr/bin/env python3
# edge0_patch6.py (15. 11. 2026, step 2): FNX_LAT_SPLIT=1, Edge0's D and U by group across the two cards. Card h (its place by its ff
# range) holds groups 4h .. 4h + 3 only: D's rows [h LAT_HW, (h + 1) LAT_HW) and U's columns of those groups (U_h = [U_4h | .. | U_4h+3],
# [N][LAT_HW]), half of each card's 3.74 GB. A layer gets two exchanges more:
#   D: each card computes its 4 groups' rows of xl, the other 4 come from the other card (the experts stay ff-split: every card needs all 8)
#   z: each card's ff part of z for the other card's groups goes there; each card adds the other's part of its own groups and takes its U half
# then the ring sums the two U halves' outputs with the shared expert as before. Decode: k_ring's tagged words (two f16 values and the
# tag) in host mailboxes (lat_xg / lat_zs); the prompt path: pf_gather's way (the copy engine into the other card's host rows, flags).
# FNX_LAT_SPLIT=0 (default): as before (D and U whole on each card). Applies on top of edge0_patch.py .. edge0_patch4.py (edge0_patch5.py
# was rejected and reverted); originals to varne-edge0-20261114/patch6/.
import os, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '/srv/build/strata-hkrati/upstream/src/split/src'
SAFE = os.path.join(os.environ.get('EDGE0_SAFE', '/srv/build/strata-hkrati/varne-edge0-20261114'), 'patch6')
FILES = ['model.hpp', 'model.cpp', 'engine_impl.hpp', 'engine.cu', 'kernels.hpp', 'kernels.cu', 'prefill.cu']
texts = {f: open(os.path.join(SRC, f), encoding='utf-8').read() for f in FILES}
if 'edge0_patch6' in texts['model.hpp']:
    sys.exit('edge0_patch6.py: already applied')
if 'edge0_patch5' in texts['engine.cu']:
    sys.exit('edge0_patch6.py: edge0_patch5.py is applied (revert it first)')


def rep(f, old, new, count=1):
    n = texts[f].count(old)
    if n != count:
        sys.exit(f'edge0_patch6.py: {f}: anchor found {n} times, want {count}:\n{old}')
    texts[f] = texts[f].replace(old, new)


# ---- model.hpp / model.cpp: the flag and the halves on load
rep('model.hpp', 'constexpr int LAT_ELL = 1792, LAT_G = 8, LAT_W = LAT_ELL * LAT_G;\n',
    'constexpr int LAT_ELL = 1792, LAT_G = 8, LAT_W = LAT_ELL * LAT_G;\n'
    '// edge0_patch6.py, FNX_LAT_SPLIT=1: card h holds groups h LAT_GH .. h LAT_GH + 3 of D and U only (two exchanges a layer bring the rest)\n'
    'constexpr int LAT_GH = LAT_G / 2, LAT_HW = LAT_ELL * LAT_GH;\n'
    'extern bool g_lat_split;\n')
rep('model.cpp', '\nnamespace fnx {\n\nnamespace {\n',
    '\nnamespace fnx {\n\n'
    'bool g_lat_split = false;   // edge0_patch6.py (FNX_LAT_SPLIT; outside the file\'s anonymous namespace: model.hpp declares it)\n\n'
    'namespace {\n')
rep('model.cpp', '        g_lat_labels = (const int32_t*) lt.data;\n',
    '        g_lat_labels = (const int32_t*) lt.data;\n'
    '        if (const char* s = std::getenv("FNX_LAT_SPLIT")) g_lat_split = std::atoi(s) != 0;   // edge0_patch6.py\n'
    '        if (g_lat_split && NCARD != 2) throw std::runtime_error("FNX_LAT_SPLIT: two cards");\n'
    '        if (g_lat_split) std::fprintf(stderr, "fnx: FNX_LAT_SPLIT: D and U by group, four groups a card (two exchanges a layer more)\\n");\n')
rep('model.cpp',
    '''            w.lat_d = up.q8split(L(l, "lat_d"));
            const Tensor& tu = L(l, "lat_u");   // [G][N][LAT_ELL] -> [N][G * LAT_ELL]: z (all 8 groups a row) in one product
            const size_t rb = (size_t) LAT_ELL / 32 * 34;
            std::vector<uint8_t> uf(tu.nbytes);
            for (int i = 0; i < N; ++i)
                for (int gg = 0; gg < LAT_G; ++gg) std::memcpy(&uf[((size_t) i * LAT_G + gg) * rb], tu.data + ((size_t) gg * N + i) * rb, rb);
            Tensor tf = tu;
            tf.data = uf.data();
            tf.ne[0] = LAT_W;
            tf.ne[1] = N;
            w.lat_u = up.q8split(tf);
''',
    '''            // edge0_patch6.py, FNX_LAT_SPLIT: this card's half of the groups only (h: its place by its ff range, as in Engine::ffn_half)
            const int h = g_lat_split ? c.ff0 / (c.ff1 - c.ff0) : 0, ng = g_lat_split ? LAT_GH : LAT_G, g0 = h * ng;
            Tensor td = L(l, "lat_d");
            if (g_lat_split) {   // D's rows [h LAT_HW, (h + 1) LAT_HW)
                if (td.ne[0] != N || td.ne[1] != LAT_W) throw std::runtime_error("FNX_LAT_SPLIT: lat_d is not [LAT_W][N]");
                td.data += (size_t) h * LAT_HW * td.row_bytes();
                td.ne[1] = LAT_HW;
                td.nbytes = (size_t) LAT_HW * td.row_bytes();
            }
            w.lat_d = up.q8split(td);
            const Tensor& tu = L(l, "lat_u");   // [G][N][LAT_ELL] -> [N][G * LAT_ELL]: z (all 8 groups a row, or the card's 4) in one product
            const size_t rb = (size_t) LAT_ELL / 32 * 34;
            std::vector<uint8_t> uf((size_t) N * ng * rb);
            for (int i = 0; i < N; ++i)
                for (int gg = 0; gg < ng; ++gg) std::memcpy(&uf[((size_t) i * ng + gg) * rb], tu.data + ((size_t) (g0 + gg) * N + i) * rb, rb);
            Tensor tf = tu;
            tf.data = uf.data();
            tf.ne[0] = (int64_t) ng * LAT_ELL;
            tf.ne[1] = N;
            tf.ne[2] = tf.ne[3] = 1;
            tf.nbytes = uf.size();
            w.lat_u = up.q8split(tf);
''')

# ---- engine: the decode mailboxes and the two exchanges
rep('engine_impl.hpp', '    uint8_t* lzq = nullptr;\n',
    '    uint8_t* lzq = nullptr;\n'
    '    // edge0_patch6.py (FNX_LAT_SPLIT): host mailboxes of D\'s half (lxg) and of z\'s part (lzs), written by the other card; *_next: the other card\'s\n'
    '    unsigned long long *lxg_in = nullptr, *lxg_next = nullptr, *lzs_in = nullptr, *lzs_next = nullptr;\n')
rep('engine.cu', '            own([q = (const void*) b->lzq] { q8_untwin(q); });\n',
    '            own([q = (const void*) b->lzq] { q8_untwin(q); });\n'
    '            if (g_lat_split) {   // edge0_patch6.py\n'
    '                b->lxg_in = hxalloc<unsigned long long>(lat_x_words());\n'
    '                b->lzs_in = hxalloc<unsigned long long>(lat_x_words());\n'
    '            }\n')
rep('engine.cu', '    if (NCARD > 2 && env_int("FNX_ONEHOP", 1) != 0 && cb_[0]->ring.f32)\n',
    '    for (int c = 0; c < NCARD; ++c) {   // edge0_patch6.py (FNX_LAT_SPLIT): each card writes into the other card\'s mailboxes\n'
    '        cb_[c]->lxg_next = cb_[(c + 1) % NCARD]->lxg_in;\n'
    '        cb_[c]->lzs_next = cb_[(c + 1) % NCARD]->lzs_in;\n'
    '    }\n'
    '    if (NCARD > 2 && env_int("FNX_ONEHOP", 1) != 0 && cb_[0]->ring.f32)\n')
rep('engine.cu', '        ck(cudaMemset(b.att_ring.in, 0, att_ring_words() * sizeof(unsigned long long)), "ring clear");\n',
    '        ck(cudaMemset(b.att_ring.in, 0, att_ring_words() * sizeof(unsigned long long)), "ring clear");\n'
    '        if (b.lxg_in) {   // edge0_patch6.py\n'
    '            ck(cudaMemset(b.lxg_in, 0, lat_x_words() * sizeof(unsigned long long)), "ring clear");\n'
    '            ck(cudaMemset(b.lzs_in, 0, lat_x_words() * sizeof(unsigned long long)), "ring clear");\n'
    '        }\n')
rep('engine.cu',
    '''    const auto lat_in = [&] {   // D: 8 latent rows a token (rows t * LAT_G + g), f16 for the row kernel
        q8s_gemv(L.lat_d, LAT_W, N, b.xq, b.lxl, LAT_W, T, b.st);
''',
    '''    const int lh = w.ff0 / (w.ff1 - w.ff0);   // edge0_patch6.py: this card's half of the groups (FNX_LAT_SPLIT), as in model.cpp
    const auto lat_in = [&] {   // D: 8 latent rows a token (rows t * LAT_G + g), f16 for the row kernel
        if (g_lat_split) {   // edge0_patch6.py: this card's 4 groups' rows, the other 4 from the other card
            q8s_gemv(L.lat_d, LAT_HW, N, b.xq, b.lxl + (size_t) lh * LAT_HW, LAT_W, T, b.st);
            lat_xg(b.lxl, b.lxl16, b.lxg_in, b.lxg_next, T, lh, (const unsigned*) (ctl + 1), call, b.ring.err, b.st);
            return;
        }
        q8s_gemv(L.lat_d, LAT_W, N, b.xq, b.lxl, LAT_W, T, b.st);
''')
rep('engine.cu',
    '''        lat_z(b.part, b.topi, b.topw, L.lat_lab, L.lat_esc, T, b.lz, b.st);
        quantize_q8_1(b.lz, b.lzq, LAT_W, T, b.st);
        q8s_gemv(L.lat_u, N, LAT_W, b.lzq, b.latu, N, T, b.st);
''',
    '''        lat_z(b.part, b.topi, b.topw, L.lat_lab, L.lat_esc, T, b.lz, b.st);
        if (g_lat_split) {   // edge0_patch6.py: z of this card's groups, both cards' parts (into lxl, free since D's exchange), then its U half
            lat_zs(b.lz, b.lxl, b.lzs_in, b.lzs_next, T, lh, (const unsigned*) (ctl + 1), call, b.ring.err, b.st);
            quantize_q8_1(b.lxl, b.lzq, LAT_HW, T, b.st);
            q8s_gemv(L.lat_u, N, LAT_HW, b.lzq, b.latu, N, T, b.st);
        } else {
            quantize_q8_1(b.lz, b.lzq, LAT_W, T, b.st);
            q8s_gemv(L.lat_u, N, LAT_W, b.lzq, b.latu, N, T, b.st);
        }
''')

# ---- kernels: the decode exchanges
rep('kernels.hpp', 'void lat_z16(const float* part, const int* topi, const float* topw, const int* labels, const float* esc, int T, uint16_t* z, void* st);\n',
    'void lat_z16(const float* part, const int* topi, const float* topw, const int* labels, const float* esc, int T, uint16_t* z, void* st);\n'
    '// edge0_patch6.py (FNX_LAT_SPLIT): a window\'s two latent exchanges (card h: groups h LAT_GH ..), k_ring\'s tagged words in host mailboxes\n'
    '// (in: this card\'s, written by the other card; next: the other card\'s; lat_x_words() each, two sets by the tag; tag and call as ring_sum\'s).\n'
    '// lat_xg: x [T][LAT_W] f32 with this card\'s half at column h LAT_HW -> xl16 [T][LAT_W] f16, the other half from the other card.\n'
    '// lat_zs: z [T][LAT_W] f32 (this card\'s ff part) -> zh [T][LAT_HW] f32: both cards\' parts of this card\'s groups\n'
    'size_t lat_x_words();\n'
    'void lat_xg(const float* x, uint16_t* xl16, unsigned long long* in, unsigned long long* next, int T, int h, const unsigned* tag, int call,\n'
    '            unsigned* err, void* st);\n'
    'void lat_zs(const float* z, float* zh, unsigned long long* in, unsigned long long* next, int T, int h, const unsigned* tag, int call,\n'
    '            unsigned* err, void* st);\n')
rep('kernels.cu', '    ck_launch("lat_z16");\n}\n',
    r'''    ck_launch("lat_z16");
}
// ---- edge0_patch6.py (FNX_LAT_SPLIT): the latent halves' exchanges in a window, k_ring's words (two f16 values and the tag)
constexpr int LAT_HP = LAT_HW / 2;   // pairs of a half row
size_t lat_x_words() { return (size_t) 2 * MAXT * LAT_HP; }
__device__ __forceinline__ void lat_put(unsigned long long* p, float a, float b, unsigned tag) {
    const __half2 h = __floats2half2_rn(a, b);
    *(volatile unsigned long long*) p = (unsigned long long) *reinterpret_cast<const unsigned*>(&h) | ((unsigned long long) tag << 32);
}
__device__ __forceinline__ float2 lat_take(const unsigned long long* p, unsigned tag, unsigned* err) {
    const long long t0 = clock64();
    unsigned long long w;
    for (;;) {
        w = *(const volatile unsigned long long*) p;
        if ((unsigned) (w >> 32) == tag) break;
        if (clock64() - t0 > 4000000000LL) { atomicOr(err, 1u); break; }
    }
    const unsigned lo = (unsigned) w;
    return __half22float2(*reinterpret_cast<const __half2*>(&lo));
}
// ZS false: x's half h to f16 in out (xl16) and to the other card, the other half from the mailbox into out. ZS true: x's (z's) other half
// to the other card, then out (zh) = this card's half plus the other card's part. Every word out first: the other card waits for them
template <bool ZS>
__global__ void k_lat_x(const float* __restrict__ x, void* __restrict__ out, unsigned long long* in, unsigned long long* next, int T, int h,
                        const unsigned* tagp, int call, unsigned* err) {
    const unsigned tag = *tagp * 256u + (unsigned) call + 1u;
    const size_t set = (size_t) (tag & 1u) * MAXT * LAT_HP;
    in += set;
    next += set;
    const int n = gridDim.x * blockDim.x, tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int own = h * LAT_HW, oth = (1 - h) * LAT_HW;
    for (int e = tid; e < T * LAT_HP; e += n) {
        const int t = e / LAT_HP, i = e % LAT_HP;
        const float2 v = reinterpret_cast<const float2*>(x + (size_t) t * LAT_W + (ZS ? oth : own))[i];
        lat_put(next + e, v.x, v.y, tag);
        if constexpr (!ZS) reinterpret_cast<__half2*>((__half*) out + (size_t) t * LAT_W + own)[i] = __floats2half2_rn(v.x, v.y);
    }
    for (int e = tid; e < T * LAT_HP; e += n) {
        const int t = e / LAT_HP, i = e % LAT_HP;
        const float2 a = lat_take(in + e, tag, err);
        if constexpr (ZS) {
            const float2 o = reinterpret_cast<const float2*>(x + (size_t) t * LAT_W + own)[i];
            reinterpret_cast<float2*>((float*) out + (size_t) t * LAT_HW)[i] = make_float2(o.x + a.x, o.y + a.y);
        } else {
            reinterpret_cast<__half2*>((__half*) out + (size_t) t * LAT_W + oth)[i] = __floats2half2_rn(a.x, a.y);
        }
    }
}
void lat_xg(const float* x, uint16_t* xl16, unsigned long long* in, unsigned long long* next, int T, int h, const unsigned* tag, int call,
            unsigned* err, void* st) {
    ring_call_check(call, "lat_xg");
    if (T < 1 || T > MAXT || !in || !next) throw std::runtime_error("lat_xg: rows or mailboxes");
    k_lat_x<false><<<std::max(4, std::min(40, (T * LAT_HP + 255) / 256)), 256, 0, (cudaStream_t) st>>>(x, xl16, in, next, T, h, tag, call, err);
    ck_launch("lat_xg");
}
void lat_zs(const float* z, float* zh, unsigned long long* in, unsigned long long* next, int T, int h, const unsigned* tag, int call,
            unsigned* err, void* st) {
    ring_call_check(call, "lat_zs");
    if (T < 1 || T > MAXT || !in || !next) throw std::runtime_error("lat_zs: rows or mailboxes");
    k_lat_x<true><<<std::max(4, std::min(40, (T * LAT_HP + 255) / 256)), 256, 0, (cudaStream_t) st>>>(z, zh, in, next, T, h, tag, call, err);
    ck_launch("lat_zs");
}
''')

# ---- prefill: the chunk exchanges (pf_gather's way)
P = 'prefill.cu'
rep(P, '    float* latu = nullptr;            // U\'s output [C][N]\n',
    '    float* latu = nullptr;            // U\'s output [C][N]\n'
    '    // edge0_patch6.py (FNX_LAT_SPLIT): lrx: the other card\'s part of z for this card\'s groups, then the sum [C][LAT_HW] f16 (U\'s half\'s input);\n'
    '    // lhb: host rows the other card\'s halves land in (two sets), lnhb: the other card\'s; lflags [2 sets][rows in, rows read], lnflags: the other\'s\n'
    '    __half* lrx = nullptr;\n'
    '    __half *lhb[2] = {}, *lnhb[2] = {};\n'
    '    unsigned *lflags = nullptr, *lnflags = nullptr;\n'
    '    unsigned lseq = 0;\n')
rep(P, '                p->latu = a.take<float>((size_t) pf_c_ * N);\n',
    '                p->latu = a.take<float>((size_t) pf_c_ * N);\n'
    '                if (g_lat_split) p->lrx = a.take<__half>((size_t) pf_c_ * LAT_HW);   // edge0_patch6.py\n')
rep(P, '        // FNX_PF_SP (idea 58, sp_patch.py): a second mixed,',
    '        if (g_lat_split) {   // edge0_patch6.py (FNX_LAT_SPLIT): the latent halves\' host rows (a chunk of half rows, two sets) and flags\n'
    '            for (int k = 0; k < 2; ++k) p->lhb[k] = hpalloc<__half>((size_t) pf_c_ * LAT_HW);\n'
    '            p->lflags = hpalloc<unsigned>(4);\n'
    '        }\n'
    '        // FNX_PF_SP (idea 58, sp_patch.py): a second mixed,')
rep(P, '    for (int c = 0; c < NCARD; ++c) {   // FNX_PF_SP: each card\'s gathers land in the other card\'s host rows\n',
    '    if (g_lat_split)\n'
    '        for (int c = 0; c < NCARD; ++c) {   // edge0_patch6.py: each card\'s latent halves land in the other card\'s host rows\n'
    '            for (int k = 0; k < 2; ++k) pf_[c]->lnhb[k] = pf_[(c + 1) % NCARD]->lhb[k];\n'
    '            pf_[c]->lnflags = pf_[(c + 1) % NCARD]->lflags;\n'
    '        }\n'
    '    for (int c = 0; c < NCARD; ++c) {   // FNX_PF_SP: each card\'s gathers land in the other card\'s host rows\n')
rep(P, '''            ckc(cudaMemset(p.gflags, 0, 4 * sizeof(unsigned)), "gather flags");
            p.gseq = 0;
        }
''', '''            ckc(cudaMemset(p.gflags, 0, 4 * sizeof(unsigned)), "gather flags");
            p.gseq = 0;
        }
        if (p.lflags) {   // edge0_patch6.py: the latent exchanges count from 0 again
            ckc(cudaMemset(p.lflags, 0, 4 * sizeof(unsigned)), "latent flags");
            p.lseq = 0;
        }
''')
rep(P, '    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gnflags + set * 2 + 1), seq, 0), "gather read");\n}\n',
    r'''    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gnflags + set * 2 + 1), seq, 0), "gather read");
}

// edge0_patch6.py: r [C][LAT_HW] (the other card's part of z) += this card's, z's rows at stride LAT_W; f16 values added in f32
__global__ void k_lat_zadd(const __half* __restrict__ z, __half* __restrict__ r, int C) {
    const size_t n = (size_t) C * (LAT_HW / 2);
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < n; e += (size_t) gridDim.x * blockDim.x) {
        const size_t t = e / (LAT_HW / 2), i = e % (LAT_HW / 2);
        const float2 a = __half22float2(reinterpret_cast<const __half2*>(z + t * LAT_W)[i]);
        __half2* q = reinterpret_cast<__half2*>(r + t * LAT_HW) + i;
        const float2 b = __half22float2(*q);
        *q = __floats2half2_rn(a.x + b.x, a.y + b.y);
    }
}
// edge0_patch6.py (FNX_LAT_SPLIT): a latent exchange of a prompt chunk on stream s, pf_gather's way (the copy engine into the other card's host
// rows, flags [set][0]: the rows are in, [set][1]: the receiver has them). zs false: this card's half of D's rows (lat16's columns h LAT_HW) to
// the other card, its half into lat16's other columns. zs true: this card's part of z for the other card's groups to it, the other card's part
// of this card's groups into lrx (pf_lat_zadd adds this card's, a pass later). Every card makes the same calls in the same order (lseq), and
// the exchange is the last thing a card queues in its pass: the other card's send must be queued before this card queues more
void pf_lat_swap(Engine::Pf& p, int h, int C, bool zs, cudaStream_t s) {
    static_assert(NCARD == 2, "pf_lat_swap: two cards");
    if (!p.lrx || !p.lflags || !p.lnflags) throw std::runtime_error("FNX_LAT_SPLIT: the prompt path's latent exchange buffers");
    const unsigned seq = ++p.lseq;
    const int set = (int) (seq & 1u), own = h * LAT_HW, oth = (1 - h) * LAT_HW;   // lat16's columns: this card's groups, the other's
    const size_t row = (size_t) LAT_HW * sizeof(__half), pitch = (size_t) LAT_W * sizeof(__half);
    CUstream cs = (CUstream) s;
    if (seq > 2) cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.lflags + set * 2 + 1), seq - 2, CU_STREAM_WAIT_VALUE_GEQ), "latent consumed");
    ckc(cudaMemcpy2DAsync(p.lnhb[set], row, p.lat16 + (zs ? oth : own), pitch, row, C, cudaMemcpyDefault, s), "latent send");
    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.lnflags + set * 2), seq, 0), "latent flag");
    cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.lflags + set * 2), seq, CU_STREAM_WAIT_VALUE_GEQ), "latent wait");
    if (zs) ckc(cudaMemcpyAsync(p.lrx, p.lhb[set], (size_t) C * row, cudaMemcpyDefault, s), "latent in");
    else ckc(cudaMemcpy2DAsync(p.lat16 + oth, pitch, p.lhb[set], row, row, C, cudaMemcpyDefault, s), "latent in");
    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.lnflags + set * 2 + 1), seq, 0), "latent read");
    ckc(cudaGetLastError(), "latent exchange");
}
// edge0_patch6.py: lrx (the other card's part of z for this card's groups) += this card's: the U half's input. In the pass after the
// exchange's (a kernel's first launch may wait for the whole context, so nothing is queued behind an exchange's wait in its pass)
void pf_lat_zadd(Engine::Pf& p, int h, int C, cudaStream_t s) {
    k_lat_zadd<<<grid_for((size_t) C * (LAT_HW / 2)), 256, 0, s>>>(p.lat16 + (size_t) h * LAT_HW, p.lrx, C);
    ckc(cudaGetLastError(), "latent z sum");
}
''')
# both prompt variants' FFN half (the overlapped one in ffn_post, the plain one): with FNX_LAT_SPLIT three passes over the cards (lph),
# each ending at an exchange's wait. The prompt path queues the cards one after the other from one thread (each), and a kernel's first
# launch can wait for the whole context (CUDA's lazy loading): queued behind a wait for the other card's send, before that send is
# queued, it never returns (round edge0s hung in the prompt warm-up). The engine's own sums keep to the same rule (an exchange last)
head = ('each([&](int, CardBuf& b, const CardW& w, Pf& p) {\n', 'const LayerW& L = w.L[l];\n', 'cudaStream_t s = b.st;\n',
        'const int P = C * TOPK, ffs = w.ff1 - w.ff0;\n')
lines = texts[P].split('\n')
starts = [i for i in range(len(lines) - 3)
          if lines[i].strip() + '\n' == head[0] and all(lines[i + k].strip() + '\n' == head[k] for k in (1, 2, 3))
          and lines[i + 1].startswith(lines[i][:len(lines[i]) - len(lines[i].lstrip())] + '    ')]
if len(starts) != 2:
    sys.exit(f'edge0_patch6.py: the FFN half\'s lambda found {len(starts)} times, want 2')


def lat_block(I):
    b = [
        'if (L.lat_d.q) {   // edge0_patch.py: Edge0\'s latent MoE (D, the experts on rows t * LAT_G + label, z, U)',
        '    if (p.moe_q8 || !p.lat16) throw std::runtime_error("FNX_LATENT: the prompt path needs the f16 expert kernels (FNX_PF_MOE_Q8=0)");',
        '    static const bool lchk = std::getenv("FNX_LAT_CHECK") != nullptr;   // edge0_patch2.py',
        '    // edge0_patch6.py, FNX_LAT_SPLIT: pass 0 D (this card\'s 4 groups) and its exchange, pass 1 the experts, z and its exchange, pass 2',
        '    // U (this card\'s half); without it one pass does all',
        '    const int lh = w.ff0 / ffs, lp1 = g_lat_split ? 1 : 0;',
        '    if (lph == 0) {',
        '        if (lchk) lat_check(p.xh, (size_t) C * N, true, 1, (int) l, C, s);',
        '        if (g_lat_split) {   // this card\'s 4 groups of D, the other card\'s 4 by the copy engine',
        '            lat_dense(p, L.lat_d, LAT_HW, N, p.xh, C, p.lat16 + (size_t) lh * LAT_HW, true, LAT_W, s);',
        '            pf_lat_swap(p, lh, C, false, s);',
        '        } else lat_dense(p, L.lat_d, LAT_W, N, p.xh, C, p.lat16, true, LAT_W, s);   // xh: the f16 of mixed',
        '    }',
        '    if (lph == lp1) {',
        '        if (lchk) lat_check(p.lat16, (size_t) C * LAT_W, true, 2, (int) l, C, s);',
        '        lat_bigrows(p.jobs_gu, p.njobs, pf_jobs_, L.lat_lab, s);',
        '        moe_gate_up_mma(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, reinterpret_cast<const uint16_t*>(p.lat16), p.gu, s);',
        '        if (lchk) lat_check(p.gu, (size_t) P * 2 * ffs, false, 4, (int) l, C, s);',
        '        lat_swiglu_h16(p.gu, ffs, P, p.h16, p.topi, L.lat_esc, nullptr, s);',
        '        if (lchk) lat_check(p.h16, (size_t) P * ffs, true, 8, (int) l, C, s);',
        '        pf_prof_mark(b, PF_DOWN);',
        '        moe_down_mma(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.h16, p.part, s);',
        '        if (lchk) lat_check(p.part, (size_t) P * LAT_ELL, false, 16, (int) l, C, s);',
        '        lat_z16(p.part, p.topi, p.topw, L.lat_lab, L.lat_esc, C, reinterpret_cast<uint16_t*>(p.lat16), s);',
        '        if (lchk) lat_check(p.lat16, (size_t) C * LAT_W, true, 32, (int) l, C, s);',
        '        if (g_lat_split) pf_lat_swap(p, lh, C, true, s);   // both cards\' parts of z for this card\'s groups',
        '    }',
        '    if (lph == lph_n - 1) {',
        '        if (g_lat_split) {   // z of this card\'s groups, then its half of U',
        '            pf_lat_zadd(p, lh, C, s);',
        '            lat_dense(p, L.lat_u, N, LAT_HW, p.lrx, C, p.latu, false, N, s);',
        '        } else lat_dense(p, L.lat_u, N, LAT_W, p.lat16, C, p.latu, false, N, s);',
        '        if (lchk) lat_check(p.latu, (size_t) C * N, false, 64, (int) l, C, s);',
        '        lat_dump(p, (int) l, C, s);   // edge0_patch4.py (FNX_LAT_DUMP)',
        '    }',
    ]
    return [I + x for x in b]


for st in reversed(starts):   # from the end, so the earlier start's index holds
    I0 = lines[st][:len(lines[st]) - len(lines[st].lstrip())]
    I = I0 + '    '
    end = next(k for k in range(st, len(lines)) if lines[k] == I + 'ckc(cudaGetLastError(), "ffn");')
    if lines[end + 1] != I0 + '});':
        sys.exit('edge0_patch6.py: the FFN half\'s lambda does not end where expected')
    body = lines[st + 4:end + 1]
    # the staging block: its wait once (pass 0)
    sw = [k for k, x in enumerate(body) if x == I + '    if (j == 0) ckc(cudaStreamWaitEvent(s, p.ev_ready[l & 1], 0), "stage in");']
    if len(sw) != 1 or body[sw[0] - 2] != I + 'if (staging) {' or body[sw[0] + 1] != I + '}':
        sys.exit('edge0_patch6.py: the FFN half\'s staging block')
    body[sw[0]] = I + '    if (j == 0 && lph == 0) ckc(cudaStreamWaitEvent(s, p.ev_ready[l & 1], 0), "stage in");'
    pre0 = sw[0] + 2   # first line after the staging block
    gu = [k for k, x in enumerate(body) if x == I + 'pf_prof_mark(b, PF_GU);']
    la = [k for k, x in enumerate(body) if x.startswith(I + 'if (L.lat_d.q) {')]
    el = [k for k, x in enumerate(body) if x == I + '} else if (p.moe_q8) {']
    tl = [k for k, x in enumerate(body) if x == I + 'if (staging && j == nk - 1) ckc(cudaEventRecord(p.ev_done[l & 1], s), "stage done");']
    if not (len(gu) == len(la) == len(el) == len(tl) == 1 and pre0 < gu[0] < la[0] < el[0] < tl[0]):
        sys.exit('edge0_patch6.py: the FFN half\'s parts (router, latent branch, the rest)')
    ind4 = lambda xs: [('    ' + x) if x else x for x in xs]
    nb = (body[:pre0]
          + [I + 'if (lph == 0) {   // edge0_patch6.py: the router and the jobs in the first pass']
          + ind4(body[pre0:gu[0] + 1]) + [I + '}']
          + body[gu[0] + 1:la[0]]
          + lat_block(I)
          + body[el[0]:tl[0]]
          + [I + 'if (lph == lph_n - 1) {   // edge0_patch6.py: the rest in the last pass']
          + ind4(body[tl[0]:]) + [I + '}'])
    lines[st:end + 1] = ([I0 + 'const int lph_n = g_lat_split && m_.cards[0].L[l].lat_d.q ? 3 : 1;   // edge0_patch6.py: FNX_LAT_SPLIT\'s passes',
                          I0 + 'for (int lph = 0; lph < lph_n; ++lph) each([&](int, CardBuf& b, const CardW& w, Pf& p) {']
                         + lines[st + 1:st + 4] + nb)
texts[P] = '\n'.join(lines)
if texts[P].count('lph_n') < 6:
    sys.exit('edge0_patch6.py: the passes were not put in')

os.makedirs(SAFE, exist_ok=True)
for f in FILES:
    shutil.copy2(os.path.join(SRC, f), os.path.join(SAFE, f))
for f in FILES:
    open(os.path.join(SRC, f), 'w', encoding='utf-8', newline='\n').write(texts[f])
print(f'edge0_patch6.py: {len(FILES)} files patched')
