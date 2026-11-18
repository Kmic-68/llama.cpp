#!/usr/bin/env python3
# edge0_patch.py (14. 11. 2026; the user: "loti se gradnje edge0 145b ... build from ground up", NVFP4 for "max kakovost"):
# Edge0-145B's latent MoE on the split engine, with FNX_LATENT=<folder with experts.bin, latent_dense_q8.bin, labels.bin>
# beside --native <its model-skeleton.gguf> (VODNIK-edge0.md has the model and the design).
#   - gguf: types I8, I32, NVFP4; FNX_LATENT adds the n-gram table's shard (FNX_LATENT_PLE, default ISTA's shard 2: the same
#     table), experts.bin as blk.L.lat_exps blobs, D and U as blk.L.lat_d / lat_u (Q8_0), the labels; ffn_*_exps.weight as
#     shapes only (sizes for the placement code), and the skeleton's BF16 tensors converted as for ISTA's file
#   - load: each card's ff part of every expert from its blob (gate|up rows, down's input blocks of 64), the tail scales
#     (times NVFP4_PRE, see below), the labels, D and U = [U_0 | ... | U_7] as Q8S, every card the whole of both (step 1)
#   - kernels: NVFP4 (TY 40) in the P100 row kernel (decode) and job kernels (prompt), like IQ4_NL with a 16-value scale and
#     the e2m1 table; its UE4M3 scales come out divided by NVFP4_PRE = 16, so the FP16 sums stay in range (the weights before
#     the experts' tail scales are ~1000 x the model's); the tail scales take the 16 back
#   - decode: D's 8 latent rows a token (q8s_gemv, f16), the gate|up jobs on row t * 8 + label, swiglu with the gate and up
#     scales, down into [P][1792], z by group with the router weight and down scale, U (q8s_gemv) into latu, ring_x adds latu
#   - prompt: the same with cuBLAS for D (f16 out) and U, BigJobs on row t * 8 + label, z as f16 in D's buffer
# Without FNX_LATENT nothing changes (no tensor added, no branch taken). Usage: python3 edge0_patch.py [SRC_DIR];
# originals to varne-edge0-20261114/ (EDGE0_SAFE).
import os, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '/srv/build/strata-hkrati/upstream/src/split/src'
SAFE = os.environ.get('EDGE0_SAFE', '/srv/build/strata-hkrati/varne-edge0-20261114')
FILES = ['gguf.hpp', 'gguf.cpp', 'model.hpp', 'model.cpp', 'kernels.hpp', 'kernels.cu', 'engine_impl.hpp', 'engine.cu', 'prefill.cu',
         'p100_iq_gemm.cuh', 'p100_moe_rows.cuh', 'p100_moe.cu']
texts = {f: open(os.path.join(SRC, f), encoding='utf-8').read() for f in FILES}
if 'FNX_LATENT' in texts['model.hpp']:
    sys.exit('edge0_patch.py: already applied (model.hpp names FNX_LATENT)')
for f, t in texts.items():
    if 'FNX_LATENT' in t or 'NVFP4' in t:
        sys.exit(f'edge0_patch.py: {f} already names FNX_LATENT or NVFP4, model.hpp does not: a half-applied tree, restore from {SAFE}')


def rep(f, old, new, count=1):
    n = texts[f].count(old)
    if n != count:
        sys.exit(f'edge0_patch.py: {f}: anchor found {n} times, want {count}:\n{old}')
    texts[f] = texts[f].replace(old, new)


def after(f, anchor, add, count=1):
    rep(f, anchor, anchor + add, count)


def before(f, anchor, add, count=1):
    rep(f, anchor, add + anchor, count)


# ================================================================ gguf
rep('gguf.hpp', "    T_IQ4_XS = 23, T_IQ1_M = 29, T_BF16 = 30, T_Q2_0 = 42,   // 42: Strata's 64-value 2-bit blocks (not in llama.cpp)\n};",
    "    T_IQ4_XS = 23, T_IQ1_M = 29, T_BF16 = 30, T_Q2_0 = 42,   // 42: Strata's 64-value 2-bit blocks (not in llama.cpp)\n"
    "    T_I8 = 24, T_I32 = 26, T_NVFP4 = 40,   // edge0_patch.py: Edge0's expert blobs and labels (FNX_LATENT), its NVFP4 experts\n};")
after('gguf.hpp', "    int converted() const { return converted_; }",
      "\n    bool latent() const { return latent_; }   // FNX_LATENT (edge0_patch.py): Edge0's latent experts beside its skeleton GGUF")
rep('gguf.hpp', "    void load_shard(const std::string& path, bool meta);\n    void convert_small_tensors();\n};",
    "    void load_shard(const std::string& path, bool meta);\n    void convert_small_tensors();\n"
    "    // FNX_LATENT (edge0_patch.py): the files of Edge0's latent MoE as tensors of this file (see gguf.cpp)\n"
    "    bool latent_ = false;\n"
    "    int map_raw(const std::string& path, size_t want);\n"
    "    void add_latent(const std::string& dir);\n};")

after('gguf.cpp', "        case T_Q2_0: return {64, 18};\n",
      "        case T_I8: return {1, 1};\n        case T_I32: return {1, 4};\n"
      "        case T_NVFP4: return {64, 36};   // edge0_patch.py: 4 UE4M3 scales (one a 16 values), 32 bytes of e2m1 codes\n")
rep('gguf.cpp', "        load_shard(first, true);\n    }\n    convert_small_tensors();\n}",
    "        load_shard(first, true);\n    }\n    if (const char* d = std::getenv(\"FNX_LATENT\")) add_latent(d);   // edge0_patch.py\n    convert_small_tensors();\n}")
before('gguf.cpp', "    if (!mixed) return;\n",
       "    mixed |= latent_;   // edge0_patch.py: Edge0's skeleton keeps ISTA's BF16 tensors (the same bytes): converted as there\n")
before('gguf.cpp', "Gguf::~Gguf() {",
       r'''// ---- FNX_LATENT (edge0_patch.py): Edge0-145B's latent MoE (VODNIK-edge0.md). The skeleton GGUF holds every tensor but the routed
// experts and the n-gram table; these come from the folder:
//   experts.bin          layer l at l * 512 * 1935376, expert e at + e * 1935376: [gate 640 x 1008 B | up | down 1792 x 360 B |
//                        16 B: the gate, up and down output scales, 0], NVFP4 -> blk.l.lat_exps (I8 [1935376][512])
//   latent_dense_q8.bin  layer l: D [8 x 1792 rows][2560] then U [8 x 2560 rows][1792], Q8_0 -> blk.l.lat_d, blk.l.lat_u
//   labels.bin           int32 [48][512], each expert's group -> lat_labels
// and the n-gram table from FNX_LATENT_PLE (default ISTA's shard 2, the same table: VODNIK). blk.l.ffn_{gate,up,down}_exps.weight
// are added as shapes only (no file, no data): the placement code reads their sizes; the experts themselves come from lat_exps.
int Gguf::map_raw(const std::string& path, size_t want) {
    Map mp;
    mp.path = path;
    mp.fd = open(path.c_str(), O_RDONLY);
    if (mp.fd < 0) throw std::runtime_error("FNX_LATENT: cannot open " + path);
    struct stat st;
    fstat(mp.fd, &st);
    mp.size = (size_t) st.st_size;
    if (mp.size != want)
        throw std::runtime_error("FNX_LATENT: " + path + " is " + std::to_string(mp.size) + " bytes, not the " + std::to_string(want));
    mp.base = mmap(nullptr, mp.size, PROT_READ, MAP_SHARED, mp.fd, 0);
    if (mp.base == MAP_FAILED) throw std::runtime_error("FNX_LATENT: cannot map " + path);
    maps_.push_back(mp);
    return (int) maps_.size() - 1;
}

void Gguf::add_latent(const std::string& dir) {
    constexpr int NLY = 48, NE = 512, ELL = 1792, FFW = 640, H = 2560, G = 8;
    constexpr size_t BLOB = 1935376, LAYER = (size_t) NE * BLOB, GU_ROW = ELL / 64 * 36, DN_ROW = FFW / 64 * 36;
    constexpr size_t DB = (size_t) G * ELL * (H / 32) * 34, UB = (size_t) G * H * (ELL / 32) * 34;
    static_assert(2 * FFW * GU_ROW + ELL * DN_ROW + 16 == BLOB, "Edge0's NVFP4 expert blob");
    latent_ = true;
    if (!tensors_.count("per_layer_token_embd.weight")) {
        const char* ple = std::getenv("FNX_LATENT_PLE");
        load_shard(ple ? ple : "/srv/llm/strata/models/iq3_s/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf", false);
    }
    auto add = [&](const std::string& name, uint32_t type, std::vector<int64_t> ne, int shard, uint64_t off, size_t bytes) {
        Tensor t;
        t.name = name;
        t.type = type;
        t.n_dims = (int) ne.size();
        for (size_t d = 0; d < ne.size(); ++d) t.ne[d] = ne[d];
        t.shard = shard;
        t.offset = off;
        t.nbytes = bytes;
        t.data = shard >= 0 ? (const uint8_t*) maps_[(size_t) shard].base + off : nullptr;
        if (!tensors_.emplace(name, t).second) throw std::runtime_error("FNX_LATENT: " + name + " is in the skeleton already");
    };
    const int se = map_raw(dir + "/experts.bin", NLY * LAYER);
    const int sd = map_raw(dir + "/latent_dense_q8.bin", NLY * (DB + UB));
    const int sl = map_raw(dir + "/labels.bin", (size_t) NLY * NE * 4);
    for (int l = 0; l < NLY; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        add(b + "lat_exps", T_I8, {(int64_t) BLOB, NE}, se, l * LAYER, LAYER);
        add(b + "lat_d", T_Q8_0, {H, G * ELL}, sd, l * (DB + UB), DB);
        add(b + "lat_u", T_Q8_0, {ELL, G * H}, sd, l * (DB + UB) + DB, UB);
        add(b + "ffn_gate_exps.weight", T_NVFP4, {ELL, FFW, NE}, -1, 0, (size_t) NE * FFW * GU_ROW);
        add(b + "ffn_up_exps.weight", T_NVFP4, {ELL, FFW, NE}, -1, 0, (size_t) NE * FFW * GU_ROW);
        add(b + "ffn_down_exps.weight", T_NVFP4, {FFW, ELL, NE}, -1, 0, (size_t) NE * ELL * DN_ROW);
    }
    add("lat_labels", T_I32, {NE, NLY}, sl, 0, (size_t) NLY * NE * 4);
    std::fprintf(stderr, "gguf: FNX_LATENT %s: Edge0's latent experts (NVFP4), D/U (Q8_0) and labels beside the skeleton\n", dir.c_str());
}

''')
if '#include <cstdlib>' not in texts['gguf.cpp']:
    rep('gguf.cpp', '#include <stdexcept>\n', '#include <cstdlib>\n#include <stdexcept>\n')

# ================================================================ model
after('model.hpp', '#include "card_split.hpp"\n\nnamespace fnx {\n',
      "\n// Edge0's latent MoE (edge0_patch.py, FNX_LATENT): 8 groups of 1792 latent values a token; each expert's blob in experts.bin, its\n"
      "// tail at LAT_TAIL; NVFP4_PRE: the factor the NVFP4 decoders' scales are divided by (the tail scales take it back)\n"
      "constexpr int LAT_ELL = 1792, LAT_G = 8, LAT_W = LAT_ELL * LAT_G;\n"
      "constexpr size_t LAT_BLOB = 1935376, LAT_TAIL = 1935360;\n"
      "constexpr float NVFP4_PRE = 16.0f;\n")
after('model.hpp', "    DW sh_gu, sh_down;                 // shared expert ff range, gate and up stacked the same way\n",
      "    // Edge0's latent MoE (edge0_patch.py, FNX_LATENT): D [LAT_W][N] and U = [U_0 | ... | U_7] [N][LAT_W] (every card the whole\n"
      "    // of both), the layer's group labels [NEXP] and each expert's output scales [NEXP][gate, up, down, 0] (times NVFP4_PRE)\n"
      "    Q8S lat_d, lat_u;\n"
      "    const int* lat_lab = nullptr;\n"
      "    const float* lat_esc = nullptr;\n")

rep('model.cpp', "    const uint32_t ok[] = {T_Q4_K, T_Q5_K, T_Q5_1, T_Q8_0, T_Q4_0, T_IQ4_NL, T_IQ4_XS, T_IQ3_S, T_IQ3_XXS, T_IQ2_S, T_IQ2_XXS, T_IQ1_M};",
    "    const uint32_t ok[] = {T_Q4_K, T_Q5_K, T_Q5_1, T_Q8_0, T_Q4_0, T_IQ4_NL, T_IQ4_XS, T_IQ3_S, T_IQ3_XXS, T_IQ2_S, T_IQ2_XXS, T_IQ1_M,\n"
    "                           T_NVFP4};   // NVFP4: Edge0's latent experts (edge0_patch.py)")
rep('model.cpp', "        return w;\n    }\n};\n\nstd::vector<int> range(int a, int b) {",
    r'''        return w;
    }
    // edge0_patch.py: Edge0's latent experts from their blobs (blk.l.lat_exps): gate rows [r0, r1) then up rows, 1792 values a row
    DW lat_rows2(const Tensor& b, int r0, int r1, size_t& stride, const char** host_of, const char* perm, const char** perm_at) {
        constexpr size_t ROW = LAT_ELL / 64 * 36, UP = (size_t) FF * ROW;
        if (r0 < 0 || r1 <= r0 || r1 > FF) throw std::runtime_error(b.name + ": rows [" + std::to_string(r0) + ", " + std::to_string(r1) + ")");
        DW w;
        w.type = T_NVFP4;
        w.n_in = LAT_ELL;
        w.n_out = 2 * (r1 - r0);
        w.row_bytes = ROW;
        stride = (size_t) w.n_out * ROW;
        const size_t half = (size_t) (r1 - r0) * ROW;
        experts(w, stride, NEXP, host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {
            const uint8_t* src = b.data + (size_t) e * LAT_BLOB;
            std::memcpy(dst, src + (size_t) r0 * ROW, half);
            std::memcpy(dst + half, src + UP + (size_t) r0 * ROW, half);
        });
        return w;
    }
    // down: every one of its 1792 rows, input blocks [b0, b1) of 64 values
    DW lat_cols(const Tensor& b, int b0, int b1, size_t& stride, const char** host_of, const char* perm, const char** perm_at) {
        constexpr size_t SRC = FF / 64 * 36, DOWN = (size_t) 2 * FF * (LAT_ELL / 64 * 36);
        if (b0 < 0 || b1 <= b0 || b1 > FF / 64) throw std::runtime_error(b.name + ": down blocks [" + std::to_string(b0) + ", " + std::to_string(b1) + ")");
        DW w;
        w.type = T_NVFP4;
        w.n_in = (b1 - b0) * 64;
        w.n_out = LAT_ELL;
        w.row_bytes = (size_t) (b1 - b0) * 36;
        stride = (size_t) w.n_out * w.row_bytes;
        experts(w, stride, NEXP, host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {
            const uint8_t* src = b.data + (size_t) e * LAT_BLOB + DOWN;
            for (int r = 0; r < w.n_out; ++r) std::memcpy(dst + (size_t) r * w.row_bytes, src + (size_t) r * SRC + (size_t) b0 * 36, w.row_bytes);
        });
        return w;
    }
};

std::vector<int> range(int a, int b) {''')
before('model.cpp', "void load_card(Stager& st, CardW& c, const std::vector<std::vector<char>>& resident, const std::vector<double>* arena_score,",
       "// FNX_LATENT (edge0_patch.py): Edge0's group labels [NL][NEXP], in the mapping (Model::Model sets it)\n"
       "const int32_t* g_lat_labels = nullptr;\n\n")
rep('model.cpp', '''        w.gu_e = up.expert_rows2(L(l, "ffn_gate_exps.weight"), L(l, "ffn_up_exps.weight"), c.ff0, c.ff1, w.gu_e_stride, hg, pr,
                                 pr ? c.perm_gu.data() + (size_t) l * NEXP : nullptr);
        w.down_e = up.expert_cols(L(l, "ffn_down_exps.weight"), c.ff0 / 32, c.ff1 / 32, w.down_e_stride, hd, pr,
                                  pr ? c.perm_dn.data() + (size_t) l * NEXP : nullptr);
''', '''        if (g_lat_labels) {   // edge0_patch.py: Edge0's latent experts, D and U, labels and tail scales
            if (c.ff0 % 64 || c.ff1 % 64) throw std::runtime_error("FNX_LATENT: a card's ff range is not in NVFP4's 64-value blocks");
            const Tensor& lb = L(l, "lat_exps");
            w.gu_e = up.lat_rows2(lb, c.ff0, c.ff1, w.gu_e_stride, hg, pr, pr ? c.perm_gu.data() + (size_t) l * NEXP : nullptr);
            w.down_e = up.lat_cols(lb, c.ff0 / 64, c.ff1 / 64, w.down_e_stride, hd, pr, pr ? c.perm_dn.data() + (size_t) l * NEXP : nullptr);
            std::vector<float> esc((size_t) NEXP * 4);
            for (int e = 0; e < NEXP; ++e) {
                std::memcpy(&esc[(size_t) e * 4], lb.data + (size_t) e * LAT_BLOB + LAT_TAIL, 16);
                for (int k = 0; k < 3; ++k) esc[(size_t) e * 4 + k] *= NVFP4_PRE;
            }
            w.lat_esc = up.put<float>(esc.data(), esc.size() * 4);
            w.lat_lab = up.put<int>(g_lat_labels + (size_t) l * NEXP, (size_t) NEXP * 4);
            w.lat_d = up.q8split(L(l, "lat_d"));
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
        } else {
        w.gu_e = up.expert_rows2(L(l, "ffn_gate_exps.weight"), L(l, "ffn_up_exps.weight"), c.ff0, c.ff1, w.gu_e_stride, hg, pr,
                                 pr ? c.perm_gu.data() + (size_t) l * NEXP : nullptr);
        w.down_e = up.expert_cols(L(l, "ffn_down_exps.weight"), c.ff0 / 32, c.ff1 / 32, w.down_e_stride, hd, pr,
                                  pr ? c.perm_dn.data() + (size_t) l * NEXP : nullptr);
        }
''')
rep('model.cpp', "        dw(w.gu_e), dw(w.down_e), dw(w.sh_gu), dw(w.sh_down);\n    }\n",
    "        dw(w.gu_e), dw(w.down_e), dw(w.sh_gu), dw(w.sh_down);\n        q8(w.lat_d), q8(w.lat_u), f(w.lat_lab), f(w.lat_esc);   // edge0_patch.py\n    }\n")
rep('model.cpp', 'static_assert(sizeof(LayerW) == 952, "LayerW changed', 'static_assert(sizeof(LayerW) == 1000, "LayerW changed')
before('model.cpp', '        if (std::sscanf(kv.first.c_str(), "blk.%d.%n", &l, &n) == 1 && n > 0 && l >= 0 && l < NL) gs[(size_t) l].names.push_back(kv.first);\n',
       "        if (kv.second.shard < 0 && !kv.second.data) continue;   // edge0_patch.py: FNX_LATENT shape-only tensors, never read\n")
after('model.cpp', "    check_supported(g);\n",
      "    if (g.latent()) {   // edge0_patch.py\n"
      "        const Tensor& lt = g.t(\"lat_labels\");\n"
      "        for (size_t i = 0; i < (size_t) NL * NEXP; ++i)\n"
      "            if (((const int32_t*) lt.data)[i] < 0 || ((const int32_t*) lt.data)[i] >= LAT_G) throw std::runtime_error(\"FNX_LATENT: a group label out of 0..7\");\n"
      "        g_lat_labels = (const int32_t*) lt.data;\n"
      "    }\n")

# ================================================================ kernels.hpp / kernels.cu: the latent MoE's small kernels, ring_x
after('kernels.hpp', "void expert_mask_logits(float* logits, int rows, int stride, const unsigned* keep, void* st);\n",
      "// Edge0's latent MoE (edge0_patch.py): the gate|up jobs' rows to t * LAT_G + label[expert]; swiglu with each expert's gate and\n"
      "// up scales (list: only when the list holds an expert, as p100_swiglu_after); the routed rows' sum by group, z [T][LAT_W]\n"
      "// (router weight and down scale folded in; each card its ff part); z as f16 for the prompt path\n"
      "void lat_rows(Job* jobs_gu, const int* njobs, const int* labels, void* st);\n"
      "void lat_bigrows(BigJob* jobs_gu, const int* njobs, int njobs_cap, const int* labels, void* st);\n"
      "void lat_swiglu_h16(const float* gu, int ffs, int P, uint16_t* h16, const int* topi, const float* esc, const int* list, void* st);\n"
      "void lat_z(const float* part, const int* topi, const float* topw, const int* labels, const float* esc, int T, float* z, void* st);\n"
      "void lat_z16(const float* part, const int* topi, const float* topw, const int* labels, const float* esc, int T, uint16_t* z, void* st);\n")
rep('kernels.hpp', "struct RingMoe {\n    const float* part = nullptr;\n    const float* topw = nullptr;\n    const float* logits = nullptr;\n};",
    "struct RingMoe {\n    const float* part = nullptr;\n    const float* topw = nullptr;\n    const float* logits = nullptr;\n"
    "    const float* lat = nullptr;   // edge0_patch.py: the latent MoE's U output [T][N], the routed sum already (part's rows t * TOPK unused)\n};")
rep('kernels.cu', "    float s = 0.f;\n#pragma unroll\n    for (int k = 0; k < TOPK; ++k) s = fmaf(m.topw[t * TOPK + k], m.part[((size_t) t * TOPK + k) * N + i], s);\n"
    "    return fmaf(sigm(m.logits[(size_t) t * (NEXP + 1) + NEXP]), m.part[((size_t) T * TOPK + t) * N + i], s);",
    "    float s = 0.f;\n    if (m.lat) s = m.lat[(size_t) t * N + i];   // edge0_patch.py\n    else {\n#pragma unroll\n"
    "    for (int k = 0; k < TOPK; ++k) s = fmaf(m.topw[t * TOPK + k], m.part[((size_t) t * TOPK + k) * N + i], s);\n    }\n"
    "    return fmaf(sigm(m.logits[(size_t) t * (NEXP + 1) + NEXP]), m.part[((size_t) T * TOPK + t) * N + i], s);")
after('kernels.cu', '''    k_expert_mask<<<(rows * NEXP + 255) / 256, 256, 0, (cudaStream_t) st>>>(logits, rows, stride, keep);
    ck_launch("expert_mask_logits");
}
''', r'''// ---- edge0_patch.py: Edge0's latent MoE (FNX_LATENT). A marked job (w = ~expert, the cache's phase 2) keeps its expert in ~w.
template <typename J>
__global__ void k_lat_rows(J* jobs, const int* njobs, const int* lab) {
    for (int j = blockIdx.x; j < njobs[0]; j += gridDim.x) {
        J& b = jobs[j];
        const int g = lab[b.w >= 0 ? b.w : ~b.w];
        for (int k = threadIdx.x; k < b.n; k += blockDim.x) b.y[k] = b.y[k] * LAT_G + g;
    }
}
void lat_rows(Job* jobs_gu, const int* njobs, const int* labels, void* st) {
    k_lat_rows<Job><<<MAXT * TOPK, 32, 0, (cudaStream_t) st>>>(jobs_gu, njobs, labels);
    ck_launch("lat_rows");
}
void lat_bigrows(BigJob* jobs_gu, const int* njobs, int njobs_cap, const int* labels, void* st) {
    k_lat_rows<BigJob><<<std::max(1, std::min(njobs_cap, 1024)), 64, 0, (cudaStream_t) st>>>(jobs_gu, njobs, labels);
    ck_launch("lat_bigrows");
}
__global__ void k_lat_swiglu(const float* __restrict__ gu, int ffs, int P, __half* __restrict__ h, const int* __restrict__ topi,
                             const float* __restrict__ esc, const int* __restrict__ list) {
    if (list && list[0] == 0) return;
    for (size_t e = blockIdx.x * (size_t) blockDim.x + threadIdx.x; e < (size_t) P * ffs; e += (size_t) gridDim.x * blockDim.x) {
        const size_t r = e / ffs, i = e % ffs;
        const float* sc = esc + (size_t) topi[r] * 4;
        const float g = sc[0] * gu[r * 2 * ffs + i];
        h[e] = __float2half(g / (1.0f + __expf(-g)) * (sc[1] * gu[r * 2 * ffs + ffs + i]));
    }
}
void lat_swiglu_h16(const float* gu, int ffs, int P, uint16_t* h16, const int* topi, const float* esc, const int* list, void* st) {
    k_lat_swiglu<<<(unsigned) std::min<size_t>(4096, ((size_t) P * ffs + 255) / 256), 256, 0, (cudaStream_t) st>>>(
        gu, ffs, P, reinterpret_cast<__half*>(h16), topi, esc, list);
    ck_launch("lat_swiglu_h16");
}
// z[t][g][c] = sum over row t's picks k in group g of topw[t][k] * s_down[e] * part[t * TOPK + k][c] (down's rows of LAT_ELL)
template <typename O>
__global__ void k_lat_z(const float* __restrict__ part, const int* __restrict__ topi, const float* __restrict__ topw,
                        const int* __restrict__ lab, const float* __restrict__ esc, O* __restrict__ z) {
    const int t = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= LAT_W) return;
    const int g = j / LAT_ELL, c = j % LAT_ELL;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        const int e = topi[t * TOPK + k];
        if (lab[e] == g) s = fmaf(topw[t * TOPK + k] * esc[(size_t) e * 4 + 2], part[((size_t) t * TOPK + k) * LAT_ELL + c], s);
    }
    if constexpr (sizeof(O) == 2) z[(size_t) t * LAT_W + j] = __float2half(s);
    else z[(size_t) t * LAT_W + j] = s;
}
void lat_z(const float* part, const int* topi, const float* topw, const int* labels, const float* esc, int T, float* z, void* st) {
    k_lat_z<float><<<dim3(LAT_W / 256, T), 256, 0, (cudaStream_t) st>>>(part, topi, topw, labels, esc, z);
    ck_launch("lat_z");
}
void lat_z16(const float* part, const int* topi, const float* topw, const int* labels, const float* esc, int T, uint16_t* z, void* st) {
    k_lat_z<__half><<<dim3(LAT_W / 256, T), 256, 0, (cudaStream_t) st>>>(part, topi, topw, labels, esc, reinterpret_cast<__half*>(z));
    ck_launch("lat_z16");
}
''')

# ================================================================ decode (engine)
after('engine_impl.hpp', "    uint16_t *xh16 = nullptr, *h16 = nullptr;   // FNX_P100_ROWS: mixed and h as FP16, for the row kernel\n",
      "    // edge0_patch.py (FNX_LATENT): D's output, its f16 rows (t * LAT_G + group), z and its q8_1, U's output [T][N]\n"
      "    float *lxl = nullptr, *lz = nullptr, *latu = nullptr;\n"
      "    uint16_t* lxl16 = nullptr;\n"
      "    uint8_t* lzq = nullptr;\n")
after('engine.cu', "            b->h16 = dalloc<uint16_t>((size_t) (P + MAXT) * ffs);\n        }\n",
      "        if (std::getenv(\"FNX_LATENT\")) {   // edge0_patch.py\n"
      "            b->lxl = dalloc<float>((size_t) MAXT * LAT_W);\n"
      "            b->lxl16 = dalloc<uint16_t>((size_t) MAXT * LAT_W);\n"
      "            b->lz = dalloc<float>((size_t) MAXT * LAT_W);\n"
      "            b->latu = dalloc<float>((size_t) MAXT * N);\n"
      "            b->lzq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(LAT_W));\n"
      "            q8_twin(b->lzq, (size_t) MAXT * q8_1_bytes(LAT_W), dalloc<int8_t>((size_t) MAXT * LAT_W), dalloc<float>((size_t) MAXT * LAT_W / 32));\n"
      "            own([q = (const void*) b->lzq] { q8_untwin(q); });\n"
      "        }\n")
after('engine.cu', "    const int P = T * TOPK, ffs = w.ff1 - w.ff0;\n    prof_mark(b, PK_HC);\n",
      "    const bool lat = L.lat_d.q != nullptr;   // edge0_patch.py: Edge0's latent MoE\n"
      "    if (lat && !b.xh16) throw std::runtime_error(\"FNX_LATENT: the latent MoE runs on the P100 row kernels only\");\n"
      "    const uint16_t* gx = lat ? b.lxl16 : b.xh16;   // the gate|up kernels' rows: D's latent rows, or mixed as f16\n"
      "    const auto lat_in = [&] {   // D: 8 latent rows a token (rows t * LAT_G + g), f16 for the row kernel\n"
      "        q8s_gemv(L.lat_d, LAT_W, N, b.xq, b.lxl, LAT_W, T, b.st);\n"
      "        p100_f32_f16(b.lxl, (size_t) T * LAT_W, b.lxl16, b.st);\n"
      "    };\n",
      1)
rep('engine.cu', "    if (over && (fold & 2)) rt.xh = b.xh16;\n", "    if (over && (fold & 2) && !lat) rt.xh = b.xh16;\n")
after('engine.cu', "                 lyr, b.cache_list, sh_in ? &L.sh_gu.s : nullptr, 2 * ffs, b.xq, b.sh_gu, &rt);\n",
      "    if (lat) lat_rows(b.jobs_gu, b.njobs, L.lat_lab, b.st);   // edge0_patch.py: gate|up reads row t * LAT_G + label\n")
rep('engine.cu', "        if (!rt.xh) p100_f32_f16(b.mixed, (size_t) T * N, b.xh16, b.st);\n",
    "        if (lat) lat_in();\n        else if (!rt.xh) p100_f32_f16(b.mixed, (size_t) T * N, b.xh16, b.st);\n")
rep('engine.cu', "P, T, b.xh16, b.gu,", "P, T, gx, b.gu,", 4)   # phase 1, both phase 2s, the path without the cache's overlap
rep('engine.cu', "    else if (b.xh16) {\n        p100_f32_f16(b.mixed, (size_t) T * N, b.xh16, b.st);\n",
    "    else if (b.xh16) {\n        if (lat) lat_in();\n        else p100_f32_f16(b.mixed, (size_t) T * N, b.xh16, b.st);\n")
rep('engine.cu', "        swiglu_h16(b.gu, ffs, P, b.h16, b.st);\n",
    "        if (lat) lat_swiglu_h16(b.gu, ffs, P, b.h16, b.topi, L.lat_esc, nullptr, b.st);\n        else swiglu_h16(b.gu, ffs, P, b.h16, b.st);\n", 2)
rep('engine.cu', "        p100_swiglu_after(b.gu, ffs, P, b.h16, b.cache_list, b.st);\n",
    "        if (lat) lat_swiglu_h16(b.gu, ffs, P, b.h16, b.topi, L.lat_esc, b.cache_list, b.st);\n"
    "        else p100_swiglu_after(b.gu, ffs, P, b.h16, b.cache_list, b.st);\n")
rep('engine.cu', "    prof_mark(b, PK_JOIN);\n    if (!sh_in) ck(cudaStreamWaitEvent(b.st, b.join, 0), \"join wait\");\n",
    "    if (lat) {   // edge0_patch.py: the routed rows by group into z, then U (this card's ff part: the ring sums the cards)\n"
    "        lat_z(b.part, b.topi, b.topw, L.lat_lab, L.lat_esc, T, b.lz, b.st);\n"
    "        quantize_q8_1(b.lz, b.lzq, LAT_W, T, b.st);\n"
    "        q8s_gemv(L.lat_u, N, LAT_W, b.lzq, b.latu, N, T, b.st);\n"
    "    }\n"
    "    prof_mark(b, PK_JOIN);\n    if (!sh_in) ck(cudaStreamWaitEvent(b.st, b.join, 0), \"join wait\");\n")
rep('engine.cu', "    RingMoe m;\n    m.part = b.part;\n    m.topw = b.topw;\n    m.logits = b.logits;\n    ring_sum(b.ring, b.bo_f, T, (const unsigned*) (ctl + 1), call, b.st, m);",
    "    RingMoe m;\n    m.part = b.part;\n    m.topw = b.topw;\n    m.logits = b.logits;\n    if (lat) m.lat = b.latu;   // edge0_patch.py\n"
    "    ring_sum(b.ring, b.bo_f, T, (const unsigned*) (ctl + 1), call, b.st, m);")

# ================================================================ prompt path (prefill)
after('prefill.cu', "    __half* shh;\n",
      "    __half* lat16 = nullptr;          // edge0_patch.py (FNX_LATENT): D's f16 rows (t * LAT_G + group), then z as f16\n"
      "    float* latu = nullptr;            // U's output [C][N]\n")
after('prefill.cu', "            p->h16 = a.take<uint16_t>((size_t) pf_c_ * TOPK * ffs);\n",
      "            if (std::getenv(\"FNX_LATENT\")) {   // edge0_patch.py\n"
      "                p->lat16 = a.take<__half>((size_t) pf_c_ * LAT_W);\n"
      "                p->latu = a.take<float>((size_t) pf_c_ * N);\n"
      "            }\n")
before('prefill.cu', "// out = W . xh for a split-layout Q8_0 matrix (dequantized to f16), xh: C rows of cols as f16 already in p.xh\n",
       r'''// edge0_patch.py: out = W . X for a split-layout Q8_0 matrix (dequantized to f16, in slices of rows as pf_dense_xh), X f16 [C][cols];
// out f16 (D's latent rows) or f32 (U's), accumulated in f32
void lat_dense(Engine::Pf& p, const Q8S& w, int rows, int cols, const __half* X, int C, void* out, bool f16_out, int ldo, cudaStream_t s) {
    if (!w.q) throw std::runtime_error("prefill: a latent matrix without the split Q8_0 layout");
    if (cols % 32 != 0 || (size_t) cols > PF_WH) throw std::runtime_error("prefill: a latent matrix's rows don't fit the f16 staging");
    const int slice = (int) std::min<size_t>(rows, PF_WH / cols / 128 * 128);
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < rows; r0 += slice) {
        const int nr = std::min(slice, rows - r0);
        k_q8s_f16<<<grid_for((size_t) nr * cols), 256, 0, s>>>(w.q + (size_t) r0 * cols, w.d + (size_t) r0 * cols / 32, (size_t) nr * cols, p.wh);
        void* o = f16_out ? (void*) ((__half*) out + r0) : (void*) ((float*) out + r0);
        ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, nr, C, cols, &one, p.wh, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero, o,
                         f16_out ? CUDA_R_16F : CUDA_R_32F, ldo, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "latent gemm");
    }
}
''')
rep('prefill.cu', "__global__ void k_pfr_addend(float* __restrict__ x, const float* __restrict__ part, const float* __restrict__ topw,\n"
    "                             const float* __restrict__ logits, int C) {",
    "__global__ void k_pfr_addend(float* __restrict__ x, const float* __restrict__ part, const float* __restrict__ topw,\n"
    "                             const float* __restrict__ logits, int C, const float* __restrict__ lat = nullptr) {")
rep('prefill.cu', "        float s = 0.f;\n#pragma unroll\n        for (int k = 0; k < TOPK; ++k) s = fmaf(topw[t * TOPK + k], part[((size_t) t * TOPK + k) * N + i], s);\n",
    "        float s = 0.f;\n        if (lat) s = lat[(size_t) t * N + i];   // edge0_patch.py: U's output\n        else {\n#pragma unroll\n"
    "        for (int k = 0; k < TOPK; ++k) s = fmaf(topw[t * TOPK + k], part[((size_t) t * TOPK + k) * N + i], s);\n        }\n")
rep('prefill.cu', "(x, m->part, m->topw, m->logits, C);", "(x, m->part, m->topw, m->logits, C, m->lat);")
rep('prefill.cu', "(x, m->part, m->topw, m->logits, rows);", "(x, m->part, m->topw, m->logits, rows, m->lat);")
rep('prefill.cu', '''                pf_prof_mark(b, PF_GU);
                if (p.moe_q8) {
                    moe_gate_up_big(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, p.xq, p.gu, s);
                    moe_down_big(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.gu, p.part, s);
                } else {''', '''                pf_prof_mark(b, PF_GU);
                if (L.lat_d.q) {   // edge0_patch.py: Edge0's latent MoE (D, the experts on rows t * LAT_G + label, z, U)
                    if (p.moe_q8 || !p.lat16) throw std::runtime_error("FNX_LATENT: the prompt path needs the f16 expert kernels (FNX_PF_MOE_Q8=0)");
                    lat_dense(p, L.lat_d, LAT_W, N, p.xh, C, p.lat16, true, LAT_W, s);   // xh: the f16 of mixed
                    lat_bigrows(p.jobs_gu, p.njobs, pf_jobs_, L.lat_lab, s);
                    moe_gate_up_mma(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, reinterpret_cast<const uint16_t*>(p.lat16), p.gu, s);
                    lat_swiglu_h16(p.gu, ffs, P, p.h16, p.topi, L.lat_esc, nullptr, s);
                    pf_prof_mark(b, PF_DOWN);
                    moe_down_mma(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.h16, p.part, s);
                    lat_z16(p.part, p.topi, p.topw, L.lat_lab, L.lat_esc, C, reinterpret_cast<uint16_t*>(p.lat16), s);
                    lat_dense(p, L.lat_u, N, LAT_W, p.lat16, C, p.latu, false, N, s);
                } else if (p.moe_q8) {
                    moe_gate_up_big(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, p.xq, p.gu, s);
                    moe_down_big(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.gu, p.part, s);
                } else {''')
rep('prefill.cu', '''            each([&](int c, CardBuf& b, const CardW&, Pf& p) {
                RingMoe m;
                m.part = p.part;
                m.topw = p.topw;
                m.logits = p.logits;
''', '''            each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                RingMoe m;
                m.part = p.part;
                m.topw = p.topw;
                m.logits = p.logits;
                if (w.L[l].lat_d.q) m.lat = p.latu;   // edge0_patch.py
''')

# ================================================================ NVFP4 in the P100 kernels (TY 40)
P = 'p100_iq_gemm.cuh'
rep(P, "TY == 20 || TY == 2 ? 6 :", "TY == 20 || TY == 2 || TY == 40 ? 6 :")
rep(P, "kRunsGr = TY == 20 || TY == 2 || TY == 42 || TY == 7 || TY == 8 ? 4 : 8;", "kRunsGr = TY == 20 || TY == 2 || TY == 42 || TY == 7 || TY == 8 || TY == 40 ? 4 : 8;")
rep(P, "TY == 20 || TY == 2 ? 72 :", "TY == 20 || TY == 2 || TY == 40 ? 72 :")
after(P, "template <> struct RunWords<20> { static constexpr int n = 5; };\n",
      "template <> struct RunWords<40> { static constexpr int n = 5; };   // NVFP4 (edge0_patch.py): codes of sub-blocks A, B; their scales\n")
rep(P, "    uint4* g1m;      // 2048: the same - 0.125\n};", "    uint4* g1m;      // 2048: the same - 0.125\n"
    "    uint32_t* fp4;   // 256: half2(e2m1[low nibble], e2m1[high nibble]) doubled, NVFP4 (edge0_patch.py)\n};")
after(P, "        t.nib4[i] = as_u32(__floats2half2_rn(float(int(i & 15) - 8), float(int(i >> 4) - 8)));\n",
      "        {   // edge0_patch.py: NVFP4's e2m1 values, doubled as ggml's kvalues_fp4\n"
      "            const float fv[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f, 0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};\n"
      "            t.fp4[i] = as_u32(__floats2half2_rn(fv[i & 15], fv[i >> 4]));\n"
      "        }\n")
rep(P, "cudaMalloc(&t.g2xxs, 256 * 16); cudaMalloc(&t.g1p, 2048 * 16); cudaMalloc(&t.g1m, 2048 * 16);",
    "cudaMalloc(&t.g2xxs, 256 * 16); cudaMalloc(&t.g1p, 2048 * 16); cudaMalloc(&t.g1m, 2048 * 16); cudaMalloc(&t.fp4, 256 * 4);")
NV_WORDS = r'''
// NVFP4 (edge0_patch.py): a block of 64 is d[4] (UE4M3, a scale a 16 values) and qs[32] (sub-block s: bytes 8 s .. 8 s + 7, value j in
// the low nibble of byte j, j + 8 in its high one); run r of a row = half r & 1 of block r >> 1: sub-blocks A = 2 h, B = 2 h + 1.
// The scale's value is ggml's ue4m3 (with its 1/2, the codes being doubled) divided by NVFP4_PRE = 16 (the experts' tails take it back).
__device__ __forceinline__ float nvfp4_scale(uint32_t x) {
    x &= 255u;
    if (x == 0u || x == 0x7Fu) return 0.f;
    const int e = int(x >> 3) & 15, m = int(x & 7u);
    return e == 0 ? ldexpf(float(m), -14) : ldexpf(1.0f + float(m) * 0.125f, e - 12);
}
template <typename P8>
__device__ __forceinline__ void nvfp4_words(const P8* base, const int h, uint32_t* w) {
    const int q = 4 + 16 * h;
    w[0] = P100_U16(base, q) | (P100_U16(base, q + 2) << 16);
    w[1] = P100_U16(base, q + 4) | (P100_U16(base, q + 6) << 16);
    w[2] = P100_U16(base, q + 8) | (P100_U16(base, q + 10) << 16);
    w[3] = P100_U16(base, q + 12) | (P100_U16(base, q + 14) << 16);
    w[4] = P100_U16(base, 2 * h);
}
'''
before(P, "template <int TY> struct RunWords { static constexpr int n = 6; };\n", NV_WORDS)
# run_load (global, the job kernels' gate|up mode 7): before IQ4_NL's branch of run_load
rep(P, "    } else if constexpr (TY == 20) {   // IQ4_NL\n        const block_iq4_nl* x = reinterpret_cast<const block_iq4_nl*>(row) + run;\n",
    "    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py)\n"
    "        nvfp4_words(reinterpret_cast<const uint8_t*>(row) + 36 * (run >> 1), run & 1, w);\n"
    "    } else if constexpr (TY == 20) {   // IQ4_NL\n        const block_iq4_nl* x = reinterpret_cast<const block_iq4_nl*>(row) + run;\n")
NV_FIN_G = '''    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py): out[0] values 0-7 (A's low nibbles), out[1] 8-15 (A's high), out[2], out[3] B's
        const half2 da = dup_f(nvfp4_scale(w[4])), db = dup_f(nvfp4_scale(w[4] >> 8));
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t a = w[2 * g], b = w[2 * g + 1];
            const half2 dd = g ? db : da;
            out[2 * g] = make_uint4(FP4P(a & 0xffff), FP4P(a >> 16), FP4P(b & 0xffff), FP4P(b >> 16));
            out[2 * g + 1] = make_uint4(FP4P((a >> 4) & 0x0fff), FP4P(a >> 20), FP4P((b >> 4) & 0x0fff), FP4P(b >> 20));
        }
'''
# run_finish (global tables): before its IQ4_XS / IQ4_NL branch
rep(P, "    } else if constexpr (TY == 23 || TY == 20) {\n        half2 dd;\n",
    "#define FP4P(v) hmul(__ldg(t.fp4 + (((v) & 15) | (((v) >> 4) & 0xf0))), dd)\n" + NV_FIN_G + "#undef FP4P\n"
    "    } else if constexpr (TY == 23 || TY == 20) {\n        half2 dd;\n")
# run_load_s (a group in shared memory / registers): before IQ4_NL's branch
rep(P, "    } else if constexpr (TY == 20 || TY == 2) {\n        const block_iq4_nl* x = reinterpret_cast<const block_iq4_nl*>(p) + (run & 3);\n",
    "    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py): a group is two blocks, four runs\n"
    "        nvfp4_words(p + 36 * ((run & 3) >> 1), run & 1, w);\n"
    "    } else if constexpr (TY == 20 || TY == 2) {\n        const block_iq4_nl* x = reinterpret_cast<const block_iq4_nl*>(p) + (run & 3);\n")
# stage_tabs: NVFP4's table
rep(P, "    } else if constexpr (TY == 23 || TY == 20) {\n        for (int i = tid; i < 64; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.nib) + i);\n",
    "    } else if constexpr (TY == 40) {   // edge0_patch.py\n"
    "        for (int i = tid; i < 64; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.fp4) + i);\n"
    "    } else if constexpr (TY == 23 || TY == 20) {\n        for (int i = tid; i < 64; i += nt) tab[i] = __ldg(reinterpret_cast<const uint4*>(t.nib) + i);\n")
# finish_s (tables in shared memory): before its IQ4 branch
rep(P, "    } else if constexpr (TY == 23 || TY == 20 || TY == 2) {\n        const uint32_t* nib = reinterpret_cast<const uint32_t*>(tab);\n",
    "#define FP4P(v) nib_pair_s(reinterpret_cast<const uint32_t*>(tab), (v), dd)\n" + NV_FIN_G + "#undef FP4P\n"
    "    } else if constexpr (TY == 23 || TY == 20 || TY == 2) {\n        const uint32_t* nib = reinterpret_cast<const uint32_t*>(tab);\n")

R = 'p100_moe_rows.cuh'
rep(R, "kRowG = TY == 20 || TY == 2 || TY == 7 || TY == 8 ? 1 : FNX_ROW_G;", "kRowG = TY == 20 || TY == 2 || TY == 7 || TY == 8 || TY == 40 ? 1 : FNX_ROW_G;")
rep(R, "    } else {   // IQ4_NL, Q4_0: blocks of d, qs[16]\n        constexpr int B = 18 * IB;\n",
    "    } else if constexpr (TY == 40) {   // NVFP4 (edge0_patch.py): block IB >> 1, half IB & 1 (p100_iq_gemm.cuh's nvfp4_words)\n"
    "        constexpr int B = 36 * (IB >> 1), Q = B + 4 + 16 * (IB & 1);\n"
    "        w[0] = g16<Q>(W) | (g16<Q + 2>(W) << 16);\n"
    "        w[1] = g16<Q + 4>(W) | (g16<Q + 6>(W) << 16);\n"
    "        w[2] = g16<Q + 8>(W) | (g16<Q + 10>(W) << 16);\n"
    "        w[3] = g16<Q + 12>(W) | (g16<Q + 14>(W) << 16);\n"
    "        w[4] = g16<B + 2 * (IB & 1)>(W);\n"
    "    } else {   // IQ4_NL, Q4_0: blocks of d, qs[16]\n        constexpr int B = 18 * IB;\n")
after(R, "        case T_IQ4_NL: rows_launch<20>(a, T, njobs_cap, s); break;\n",
      "        case T_NVFP4: rows_launch<40>(a, T, njobs_cap, s); break;   // edge0_patch.py\n")
after('p100_moe.cu', "        case T_IQ4_NL: P100_JOB(20); break;\n", "        case T_NVFP4: P100_JOB(40); break;   // edge0_patch.py\n")

# ================================================================ write: originals first
os.makedirs(SAFE, exist_ok=True)
for f in FILES:
    dst = os.path.join(SAFE, f)
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(SRC, f), dst)
for f in FILES:
    with open(os.path.join(SRC, f), 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(texts[f])
print(f'edge0_patch.py: {len(FILES)} files patched in {SRC}, originals in {SAFE}')
