// The Engine (engine.hpp) for Qwen3.8-Flash-Next: its layers, its embeddings and its expert arena; what both models'
// engines share is engine_common.cu's (engine_parts.hpp).
#include <cstdlib>
#include "engine.hpp"
#include "engine_parts.hpp"
#include "env.hpp"
#include "peer_probe.hpp"
#include "p100_over.hpp"

#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <condition_variable>
#include <map>
#include <numeric>
#include <mutex>
#include <stdexcept>
#include <thread>
#if FNX_HIP
#include "hip/peer.h"
#include "hip/rccl.h"
#endif

namespace fnx {


// FNX_CVEC=file[:scale][,file[:scale]...] (the old engine's --control-vector-scaled), FNX_CVEC_LAYERS=first,last (its
// --control-vector-layer-range, inclusive, within 1..NL-2: the last layer's write goes into the head's read), FNX_CVEC_MODE=
// project|add (--cvec-mode, project by default), FNX_CVEC_DIR=per-layer|single:L (--cvec-dir, per-layer by default; single
// for project only). As the old engine's loader: d_l = the sum over the files of direction.<src> times the file's scale,
// src = l (per-layer) or L (single); project keeps v_l = d_l / |d_l| and s_l = |d_l|, add keeps d_l and s_l = 1; a layer
// without a direction (layer 0 has none) is not steered. It changes the model's answers: the uncensored profiles' vector
void Engine::load_cvec() {
    const char* spec = std::getenv("FNX_CVEC");
    if (!spec || !*spec) return;
    const char* md = std::getenv("FNX_CVEC_MODE");
    const std::string mode = md && *md ? md : "project";
    if (mode != "project" && mode != "add") throw std::runtime_error("FNX_CVEC_MODE: project or add, got " + mode);
    const char* dr = std::getenv("FNX_CVEC_DIR");
    const std::string dir = dr && *dr ? dr : "per-layer";
    int single = -1;
    if (dir.rfind("single:", 0) == 0) {
        char* end = nullptr;
        const long v = std::strtol(dir.c_str() + 7, &end, 10);
        if (end == dir.c_str() + 7 || *end || v < 1 || v >= NL) throw std::runtime_error("FNX_CVEC_DIR: single:L wants a layer 1.." + std::to_string(NL - 1) + ", got " + dir);
        if (mode != "project") throw std::runtime_error("FNX_CVEC_DIR: single:L is for FNX_CVEC_MODE=project only");
        single = (int) v;
    } else if (dir != "per-layer") {
        throw std::runtime_error("FNX_CVEC_DIR: per-layer or single:L, got " + dir);
    }
    int first = 1, last = NL - 2;
    if (const char* ly = std::getenv("FNX_CVEC_LAYERS"); ly && *ly) {
        char tail = 0;
        if (std::sscanf(ly, "%d%*[ ,]%d %c", &first, &last, &tail) != 2)
            throw std::runtime_error(std::string("FNX_CVEC_LAYERS: want first,last, got ") + ly);
    }
    if (first < 1 || last > NL - 2 || first > last)
        throw std::runtime_error("FNX_CVEC_LAYERS: want 1 <= first <= last <= " + std::to_string(NL - 2) + ", got " + std::to_string(first) + "," +
                                 std::to_string(last));
    std::vector<double> d((size_t) NL * N, 0.0);
    std::vector<char> has((size_t) NL, 0);
    const std::string list = spec;
    int files = 0;
    for (size_t at = 0; at <= list.size();) {
        const size_t comma = list.find(',', at);
        std::string one = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? list.size() + 1 : comma + 1;
        if (one.empty()) continue;
        double scale = 1.0;
        if (const size_t colon = one.rfind(':'); colon != std::string::npos) {
            char* end = nullptr;
            const double v = std::strtod(one.c_str() + colon + 1, &end);
            if (end != one.c_str() + colon + 1 && !*end) scale = v, one.resize(colon);
        }
        const Gguf g(one);
        for (int l = first; l <= last; ++l) {
            const Tensor* t = g.find("direction." + std::to_string(single >= 0 ? single : l));
            if (!t) continue;
            if (t->type != T_F32 || t->ne[0] != N || t->nrows() != 1)
                throw std::runtime_error("FNX_CVEC " + one + ": " + t->name + " is not " + std::to_string(N) + " F32 values");
            const float* x = (const float*) t->data;
            for (int i = 0; i < N; ++i) d[(size_t) l * N + (size_t) i] += scale * (double) x[i];
            has[(size_t) l] = 1;
        }
        ++files;
    }
    if (!files) throw std::runtime_error(std::string("FNX_CVEC: no file in '") + spec + "'");
    cvec_.dir.assign((size_t) NL * N, 0.f);
    cvec_.s.assign((size_t) NL, 0.f);
    int steered = 0;
    for (int l = first; l <= last; ++l) {
        if (!has[(size_t) l]) continue;
        double nn = 0.0;
        for (int i = 0; i < N; ++i) nn += d[(size_t) l * N + (size_t) i] * d[(size_t) l * N + (size_t) i];
        nn = std::sqrt(nn);
        if (nn == 0.0) continue;
        for (int i = 0; i < N; ++i)
            cvec_.dir[(size_t) l * N + (size_t) i] = (float) (mode == "add" ? d[(size_t) l * N + (size_t) i] : d[(size_t) l * N + (size_t) i] / nn);
        cvec_.s[(size_t) l] = mode == "add" ? 1.f : (float) nn;
        ++steered;
    }
    cvec_.first = first, cvec_.last = last, cvec_.add = mode == "add";
    std::fprintf(stderr, "fnx: FNX_CVEC %s: mode %s, dir %s, layers %d..%d (%d steered): the model's answers change\n", spec, mode.c_str(),
                 dir.c_str(), first, last, steered);
}

Engine::Engine(Model& m, int max_ctx, int lanes) : m_(m), max_ctx_(max_ctx) {
    // the 27B build's settings: this Flash-Next engine has none of what they set, so one given is a mistake, not a
    // run of the default
    for (const char* k : {"FNX_PF_SKIP", "FNX_PF_SKIP_CONFIRM", "FNX_PF_SCORE_MIB", "FNX_PF_WINDOWS", "FNX_PF_SCORES",
                          "FNX_KV_PAGES", "FNX_KV_RESERVE_MIB"})
        if (std::getenv(k)) throw std::runtime_error(std::string(k) + " is the 27B's: Flash-Next has nothing it sets");
    // the rotary's scaling (choose_rope, src/rope.hpp): the model's own YaRN, FNX_ROPE_SCALE / FNX_YARN_ORIG_CTX, or
    // both when they agree; set once here, before any graph capture. Positions past what the rotary covers have angles
    // the model never saw, so a context past that is refused here rather than served
    rope_.base = m_.rope_base;
    long long env_orig = 0;
    if (std::getenv("FNX_YARN_ORIG_CTX")) {
        env_orig = env_int("FNX_YARN_ORIG_CTX", 0);
        if (env_orig <= 0 || env_orig > INT32_MAX)
            throw std::runtime_error("FNX_YARN_ORIG_CTX: want a trained context length from 1 to " + std::to_string(INT32_MAX) + ", got " +
                                     std::to_string(env_orig));
    }
    const RopeChoice rc = choose_rope(m_.yarn_factor, m_.yarn_orig, m_.ctx_train, std::getenv("FNX_ROPE_SCALE") ? rope_scale() : -1.0, env_orig);
    if (max_ctx > rc.limit)
        throw std::runtime_error("context " + std::to_string(max_ctx) + " is past the " + std::to_string(rc.limit) + " the rotary covers (" +
                                 (rc.S ? "trained " + std::to_string(rc.orig) + " x YaRN " + std::to_string(rc.S)
                                       : "the model's trained length; FNX_ROPE_SCALE=S (--rope-scale S) covers trained x S") +
                                 ")");
    if (rc.S) {
        rope_ = yarn_rope(m_.rope_base, rc.orig, rc.S, NROT);
        std::fprintf(stderr, "fnx: rope scaling YaRN x%.4f%s: n_ctx_orig %lld, pairs %.0f..%.0f of %d blended, mscale %.4f\n", rc.S,
                     rc.from_model ? " (the model's own)" : "", rc.orig, rope_.c_low, rope_.c_high, NROT / 2, rope_.mscale);
        if (!rc.from_model && max_ctx <= m_.ctx_train)
            std::fprintf(stderr, "fnx: note: context %d is within the trained %d: the scaling changes every position's rotary "
                         "and no length needs it\n", max_ctx, m_.ctx_train);
    }
    const long long pf_rows = env_int("FNX_PF_ROWS", pf_c_);
    if (pf_rows < MAXT || pf_rows > 8192 || pf_rows % MAXT)
        throw std::runtime_error("FNX_PF_ROWS: a multiple of " + std::to_string(MAXT) + " from " + std::to_string(MAXT) + " to 8192, got " +
                                 std::to_string(pf_rows));
    pf_c_ = (int) pf_rows;
    hc1_rows();   // FNX_HC1 checked now, not at the first hc read (inside a graph capture)
    pf_jobs_ = pf_c_ * TOPK / PF_JOB + NEXP;   // jobs of up to PF_JOB pairs per expert, at most
    set_prompt_rows(pf_c_);
    const int extra_pages = (std::max(lanes, 1) - 1) * lane_block_pages();
    // the paged K/V pool: pages for the whole context (one more for the block a position past the last one needs)
    // and the extra lanes' blocks. Int8 rows: the context's pages as many as fit in their f16 bytes (the smallest ratio
    // of the cards': every card has the same pages)
    int ctx_pages = max_ctx / KVP_POS + 2;
    if (kv_q8()) {
        double ratio = 1e9;
        for (int cc = 0; cc < NCARD; ++cc) ratio = std::min(ratio, (double) kvp_page(cc, false) / kvp_page(cc, true));
        ctx_pages = (int) (ctx_pages * ratio);
    }
    int n_pages = ctx_pages + extra_pages;
    // FNX_PEER (AMD only; hip/peer_policy.hpp has the rules: 0 host exchange, 1 try the peer push, unset the build's default): every directed pair is probed at start
    // (kernel stores into the peer, a full read-back compare on both sides, a flag round trip); any failure leaves the host exchange for the whole run with one line.
    // FNX_PEER_PARTS: 1 the window sums, 2 the attention merge, 3 both (the default)
    bool peer_on = false;
    int peer_parts = 0;
#if FNX_HIP
    {
        std::vector<int> devs;
        for (int c = 0; c < NCARD; ++c) devs.push_back(m.cards[c].dev);
        const PeerDecision pd = peer_decide(std::getenv("FNX_PEER"), PEER_DEFAULT_ON, [&](std::string& why) { return peer_enable_and_probe(devs, why); });
        if (!pd.line.empty()) std::fprintf(stderr, "fnx: %s\n", pd.line.c_str());
        peer_on = pd.use;
        peer_parts = (int) env_int("FNX_PEER_PARTS", 3);
        if (peer_on) std::fprintf(stderr, "fnx: peer push: %s%s pushed into the peers' memory\n", peer_parts & 1 ? "the window sums " : "", peer_parts & 2 ? "and the attention merge" : "");
        {   // RCCL for the prompt sums: constant RCCL_DEFAULT_ON (off until the screens say otherwise), FNX_PF_RCCL=0/1 overrides
            const RcclDecision rd = rccl_decide(std::getenv("FNX_PF_RCCL"), peer_on, RCCL_DEFAULT_ON, [&](std::string& why) {
                std::shared_ptr<RcclState> st;
                if (!rccl_open(devs, st, why)) return false;
                rccl_ = st;
                return true;
            });
            if (!rd.line.empty()) std::fprintf(stderr, "fnx: %s\n", rd.line.c_str());
            rccl_on_ = rd.use;
        }
    }
    auto pxalloc = [&](size_t bytes) -> void* {
        void* p = nullptr;
        if (hipExtMallocWithFlags(&p, bytes, hipDeviceMallocFinegrained) != hipSuccess) throw std::runtime_error("FNX_PEER: fine-grained device allocation of " + std::to_string(bytes >> 10) + " KiB failed");
        ck(FNX_ZERO_DEV(p, bytes), "peer mailbox clear");
        own_dev(p);
        return p;
    };
#endif
    load_cvec();
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m.cards[c];
        CardBuf* b = new CardBuf();
        b->dev = w.dev;
        ck(cudaSetDevice(w.dev), "cudaSetDevice");
        // FNX_HOSTRING: no peer access is ever enabled; the cards exchange through host memory
        // FNX_PRIO=1 (prio_patch.py): the card's own stream at its highest priority, so the router's and the experts' blocks take
        // free SMs before the second stream's (the shared expert, the next layer's guess); scheduling only
        static const bool prio = env_flag("FNX_PRIO");
        int prio_least = 0, prio_most = 0;
        if (prio) ck(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_most), "stream priorities");
        ck(cudaStreamCreateWithPriority(&b->st, cudaStreamNonBlocking, prio ? prio_most : 0), "stream");
        own_stream(b->st);
        ck(cudaStreamCreateWithFlags(&b->st2, cudaStreamNonBlocking), "stream");
        own_stream(b->st2);
        ck(cudaEventCreateWithFlags(&b->fork, cudaEventDisableTiming), "event");
        own_event(b->fork);
        ck(cudaEventCreateWithFlags(&b->join, cudaEventDisableTiming), "event");
        own_event(b->join);
        if (p100_overlap_on()) {
            ck(cudaStreamCreateWithFlags(&b->st3, cudaStreamNonBlocking), "stream");
            own_stream(b->st3);
            if (p100_dma_on()) b->dma = p100_dma_start(b->dev);
            ck(cudaEventCreateWithFlags(&b->fork2, cudaEventDisableTiming), "event");
            own_event(b->fork2);
            ck(cudaEventCreateWithFlags(&b->join2, cudaEventDisableTiming), "event");
            own_event(b->join2);
        }
        ck(cudaEventCreate(&b->ev0), "event");
        own_event(b->ev0);
        ck(cudaEventCreate(&b->ev1), "event");
        own_event(b->ev1);
        const int P = MAXT * TOPK, ffs = w.ff1 - w.ff0;
        b->R = dalloc<float>((size_t) MAXT * D);
        if (cvec_.last >= 0) {   // FNX_CVEC: this card's copy of the directions
            b->cvec = dalloc<float>((size_t) NL * N);
            ck(cudaMemcpy(b->cvec, cvec_.dir.data(), (size_t) NL * N * sizeof(float), cudaMemcpyHostToDevice), "cvec");
        }
        b->mixed = dalloc<float>((size_t) MAXT * N);
        b->bo_a = dalloc<float>((size_t) MAXT * N);
        b->bo_f = dalloc<float>((size_t) MAXT * N);
        b->inj_a = dalloc<float>(MAXT * HC);
        b->inj_f = dalloc<float>(MAXT * HC);
        b->hcs = dalloc<float>(hc_scratch_floats(MAXT));
        if (const int g = hc_prepare(b->hcs); g < hc_blocks())
            std::fprintf(stderr, "fnx: card %d: %d SMs: the hc read runs its %d blocks of work on %d\n", c, g, hc_blocks(), g);
        if (std::getenv("FNX_HC_BAR_WRAP")) {
            const long long n = env_int("FNX_HC_BAR_WRAP", 0);
            if (n <= 0 || n > 0xffffffffLL) throw std::runtime_error("FNX_HC_BAR_WRAP: want a barrier count > 0, got " + std::to_string(n));
            hc_preset_barrier(b->hcs, (unsigned) n);
        }
        b->xq = dalloc<uint8_t>((size_t) P * q8_1_bytes(N));
        b->qkvz = dalloc<float>((size_t) MAXT * (w.conv_ch() + w.nv() * HS));
        b->ab = dalloc<float>((size_t) MAXT * 2 * w.nv());
        b->gdn_scratch = dalloc<float>(gdn_scratch_floats(w.nv()));
        b->qg = dalloc<float>((size_t) MAXT * NQH * 2 * HD);
        b->k = dalloc<float>((size_t) MAXT * NKVH * HD);
        b->v = dalloc<float>((size_t) MAXT * NKVH * HD);
        b->att = dalloc<float>((size_t) MAXT * w.nq * HD);
        b->att_part = dalloc<float>(attn_scratch_floats(w.nq));
        b->logits = dalloc<float>((size_t) MAXT * (NEXP + 1));
        if (!m_.keep.empty()) {   // FNX_EXPERT_MASK (maska_patch.py): the kept experts as bits, a layer's row NEXP / 32 words
            std::vector<unsigned> bits((size_t) NL * NEXP / 32, 0u);
            for (int i = 0; i < NL * NEXP; ++i)
                if (m_.kept(i)) bits[(size_t) (i >> 5)] |= 1u << (i & 31);
            b->ekeep = dalloc<unsigned>(bits.size());
            ck(cudaMemcpy(b->ekeep, bits.data(), bits.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "expert mask");
        }
        b->topw = dalloc<float>(P);
        b->topi = dalloc<int>(P);
        b->njobs = dalloc<int>(2);
        b->jobs_gu = dalloc<Job>(P + 1);
        b->jobs_dn = dalloc<Job>(P + 1);
        for (int T = 1; T <= MAXT; ++T) {
            Job j{};
            j.w = 0;
            j.n = T;
            for (int t = 0; t < T; ++t) { j.y[t] = t; j.out[t] = t; }
            b->dense[T] = dalloc<Job>(1);
            ck(cudaMemcpy(b->dense[T], &j, sizeof j, cudaMemcpyHostToDevice), "dense job");
        }
        if (w.ec.n_slots) b->cache_list = dalloc<int>(1 + P);
        b->gu = dalloc<float>((size_t) (P + MAXT) * 2 * ffs);
        b->part = dalloc<float>((size_t) (P + MAXT) * N);
        if (p100_rows_on()) {
            b->xh16 = dalloc<uint16_t>((size_t) MAXT * N);
            b->h16 = dalloc<uint16_t>((size_t) (P + MAXT) * ffs);
        }
        if (std::getenv("FNX_LATENT")) {   // edge0_patch.py
            b->lxl = dalloc<float>((size_t) MAXT * LAT_W);
            b->lxl16 = dalloc<uint16_t>((size_t) MAXT * LAT_W);
            b->lz = dalloc<float>((size_t) MAXT * LAT_W);
            b->latu = dalloc<float>((size_t) MAXT * N);
            b->lzq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(LAT_W));
            q8_twin(b->lzq, (size_t) MAXT * q8_1_bytes(LAT_W), dalloc<int8_t>((size_t) MAXT * LAT_W), dalloc<float>((size_t) MAXT * LAT_W / 32));
            own([q = (const void*) b->lzq] { q8_untwin(q); });
            if (g_lat_split) {   // edge0_patch6.py
                b->lxg_in = hxalloc<unsigned long long>(lat_x_words());
                b->lzs_in = hxalloc<unsigned long long>(lat_x_words());
            }
        }
        b->sh_gu = dalloc<float>((size_t) MAXT * 2 * ffs);
        b->sh_hq = dalloc<uint8_t>((size_t) MAXT * q8_1_bytes(ffs));
        if (!std::getenv("FNX_NO_TWIN")) {
            // the split twins: xq's rows of n <= N columns sit at r q8_1_bytes(n), so r n < P N
            q8_twin(b->xq, (size_t) P * q8_1_bytes(N), dalloc<int8_t>((size_t) P * N), dalloc<float>((size_t) P * N / 32));
            q8_twin(b->sh_hq, (size_t) MAXT * q8_1_bytes(ffs), dalloc<int8_t>((size_t) MAXT * ffs), dalloc<float>((size_t) MAXT * ffs / 32));
            for (const void* q : {(const void*) b->xq, (const void*) b->sh_hq}) own([q] { q8_untwin(q); });
        }
        b->ple_emb = dalloc<float>((size_t) MAXT * N);
        b->ple_key = dalloc<float>((size_t) 3 * MAXT * D);   // key, then room for gated and normalized rows
        b->ple_val = dalloc<float>((size_t) MAXT * N);
        b->head_logits = dalloc<float>((size_t) MAXT * (w.v1 - w.v0));
        b->best_v = dalloc<float>(MAXT);
        b->best_i = dalloc<int>(MAXT);
        b->am_keys = dalloc<unsigned long long>(MAXT);
        if (c == 0 && env_flag("FNX_PREFETCH_PROBE")) {
            b->probe_logits = dalloc<float>((size_t) MAXT * NEXP);
            b->probe_guess = dalloc<int>((size_t) NL * MAXT * TOPK);
            b->probe_cnt = dalloc<unsigned long long>(7);
        }
        if (env_int("FNX_PREFETCH", 0) != 0 && b->dma && !env_flag("FNX_PREFETCH_PROBE")) {   // FNX_PREFETCH: every card
            b->pf_logits = dalloc<float>((size_t) MAXT * NEXP);
            b->pf_guess = dalloc<int>((size_t) MAXT * TOPK);
            b->pf_list = dalloc<int>((size_t) 1 + MAXT * TOPK);
            b->pf_want = dalloc<int>(1);
            if (env_int("FNX_MEMO", 0) != 0 || std::getenv("FNX_NOC_FILE")) {   // memo_patch.py
                b->memo_guess = dalloc<int>((size_t) NL * MAXT * TOPK + 1);
                b->memo_probe = dalloc<int>((size_t) MAXT * TOPK);
                ck(cudaMemset(b->memo_guess, 0xff, ((size_t) NL * MAXT * TOPK + 1) * sizeof(int)), "memo");
                ck(cudaMemset(b->memo_guess + (size_t) NL * MAXT * TOPK, 0, sizeof(int)), "memo");
                ck(cudaMemset(b->memo_probe, 0xff, (size_t) MAXT * TOPK * sizeof(int)), "memo");
            }
        }
        if (c == 0 && env_flag("FNX_ROUTE_COUNT")) {
            b->route_cnt = dalloc<unsigned>((size_t) 2 * NL * NEXP);
            ck(cudaMemset(b->route_cnt, 0, (size_t) 2 * NL * NEXP * sizeof(unsigned)), "route counts");
        }
        if (const long long tr = env_int("FNX_ROUTE_TRACE", 0); c == 0 && tr > 0) {
            b->route_trace_cap = (unsigned long long) tr * 1000000ull;
            void* hp = nullptr;   // in mapped host memory: no VRAM for it
            ck(cudaHostAlloc(&hp, b->route_trace_cap * sizeof(unsigned), cudaHostAllocMapped), "route trace");
            own_pinned(hp);
            b->route_trace = (unsigned*) hp;
            b->route_trace_len = dalloc<unsigned long long>(1);
            ck(cudaMemset(b->route_trace_len, 0, sizeof(unsigned long long)), "route trace");
        }
        b->S.assign(NL, nullptr);
        b->chist.assign(NL, nullptr);
        b->snap.assign(NL, nullptr);
        b->hsnap.assign(NL, nullptr);
        b->kc.assign(NL, nullptr);
        b->vc.assign(NL, nullptr);
        b->qsa_ring.assign(NL, nullptr);
        b->qsa_pooled.assign(NL, nullptr);
        b->qsa_y = dalloc<float>((size_t) MAXT * qsa_rows());
        b->qsa_q = dalloc<float>((size_t) MAXT * 512);
        b->qsa_scores = dalloc<float>((size_t) MAXT * (max_ctx / 4));
        b->qsa_cells = dalloc<int>((size_t) MAXT * qsa_maxsel());
        b->qsa_ncells = dalloc<int>(MAXT);
        const size_t snap_floats = (size_t) (MAXT - 1) * w.nv() * HS * HS;
        int n_rec = 0;
        for (int l = 0; l < NL; ++l) n_rec += is_attn(l) ? 0 : 1;
        b->snap_block = dalloc<float>(snap_floats * n_rec);
        b->snap_block_bytes = snap_floats * n_rec * sizeof(float);
        for (int l = 0, k = 0; l < NL; ++l) {
            if (is_attn(l)) {
                b->qsa_ring[l] = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);
            } else {
                b->S[l] = dalloc<float>((size_t) w.nv() * HS * HS);
                b->chist[l] = dalloc<float>((size_t) 2 * 3 * w.conv_ch());
                b->snap[l] = b->snap_block + snap_floats * k++;
                b->hsnap[l] = dalloc<float>((size_t) MAXT * 3 * w.conv_ch());
            }
        }
        b->ple_hist = dalloc<float>((size_t) 9 * D);
        b->ple_hsnap = dalloc<float>((size_t) MAXT * 9 * D);
        b->ctl = dalloc<int>(2);
        b->ring.card = c;
        // sized for fp32 words (two per pair); fp16 pairs use the first half of each slot row's room
        b->ring.in = hxalloc<unsigned long long>(ring_in_words(MAXT));
        b->ring.ag = hxalloc<unsigned long long>(ring_ag_words(MAXT));
        b->ring.f32 = ring_f32();
        b->ring.err = dalloc<unsigned>(1);
        b->att_ring.card = c;
#if FNX_HIP
        if (peer_on && (peer_parts & 1)) b->ring.pk = pxalloc((size_t) 2 * NCARD * MAXT * (N / 2) * sizeof(unsigned long long));
        b->att_ring.in = peer_on && (peer_parts & 2) ? (unsigned long long*) pxalloc(att_ring_words() * sizeof(unsigned long long))
                                                     : hxalloc<unsigned long long>(att_ring_words());
#else
        b->att_ring.in = hxalloc<unsigned long long>(att_ring_words());
#endif
        b->att_ring.err = b->ring.err;
        cb_.push_back(b);
    }
    // every link carries a probe's stores before the ring's (src/peer_probe.hpp): one that drops them stops the engine
    // here, named, instead of in the first window's ring; a slow one is said and the engine goes on
    std::fprintf(stderr, "fnx: HOSTRING build: the cards exchange through page-locked host memory; peer access is never enabled, "
                         "no peer probe, IOMMU and BAR1 checks skipped\n");
    for (int c = 0; c < NCARD; ++c) {
        cb_[c]->ring.next_in = cb_[(c + 1) % NCARD]->ring.in;
        cb_[c]->ring.next_ag = cb_[(c + 1) % NCARD]->ring.ag;
        cb_[c]->att_ring.next_in = cb_[(c + 1) % NCARD]->att_ring.in;
        for (int d = 0; d < NCARD; ++d) {
            cb_[c]->ring.peer_in[d] = cb_[d]->ring.in;
            cb_[c]->att_ring.peer_in[d] = cb_[d]->att_ring.in;
        }
        // one hop for the window sums and the attention merge on 3 or more cards (FNX_ONEHOP=0: the rings); the mailboxes live in the ring's buffers
        // FNX_ONEHOP=2 (upstream's screen, 8969119a): the one-hop sum on two cards too (each card adds the two fp16-rounded addends in fp32: one host
        // round trip instead of two; the sums differ in the last bits)
        const bool onehop = (NCARD > 2 || peer_on || env_int("FNX_ONEHOP", 1) == 2) && env_int("FNX_ONEHOP", 1) != 0 && !cb_[c]->ring.f32;
        cb_[c]->ring.onehop = cb_[c]->att_ring.onehop = onehop;
        for (int d = 0; d < NCARD; ++d) cb_[c]->ring.peer_pk[d] = cb_[d]->ring.pk;
        cb_[c]->ring.peer = onehop && cb_[c]->ring.pk != nullptr;
    }
    for (int c = 0; c < NCARD; ++c) {   // edge0_patch6.py (FNX_LAT_SPLIT): each card writes into the other card's mailboxes
        cb_[c]->lxg_next = cb_[(c + 1) % NCARD]->lxg_in;
        cb_[c]->lzs_next = cb_[(c + 1) % NCARD]->lzs_in;
    }
    if (NCARD > 2 && env_int("FNX_ONEHOP", 1) != 0 && cb_[0]->ring.f32)
        std::fprintf(stderr, "fnx: FNX_RING_F32: the one-hop sum carries fp16 words only, the rings stay\n");
    // the prompt path's buffers now, so the arena below is sized from what is left after them (FNX_PF_LEND: after the arena, in the pools' last slots)
    pf_lend_ = m.arena && env_int("FNX_PF_LEND", 0) != 0;
    pf_refill_ok_ = pf_lend_ && m.cards[0].ec.fill && !m.cards[0].ec.arena_full;
    pf_refill_env_ = (int) env_int("FNX_PF_REFILL", 0);
    prefetch_ = !cb_.empty() && cb_[0]->pf_want && cb_[0]->dma;
    if (prefetch_ && cb_[0]->memo_guess) {   // memo_patch.py: the memo in mapped host memory, card 0 writes it
        memo_cap_ = (int) std::max<long long>(1024, env_int("FNX_MEMO_CAP", 131072));
        const size_t n = (size_t) memo_cap_ * NL * TOPK;
        void* p = nullptr;
        ck(cudaHostAlloc(&p, n * sizeof(uint16_t), cudaHostAllocMapped | cudaHostAllocPortable), "memo");
        memo_ = (uint16_t*) p;
        memo_pos_ = halloc<int>((size_t) memo_cap_);
        memo_tok_ = halloc<int>((size_t) memo_cap_);
        for (int i = 0; i < memo_cap_; ++i) memo_pos_[i] = -1, memo_tok_[i] = -1;
        memo_h_ = halloc<int>((size_t) NL * MAXT * TOPK + 1);
        ck(cudaHostAlloc(&p, 8 * sizeof(unsigned long long), cudaHostAllocMapped | cudaHostAllocPortable), "memo stat");
        memo_stat_ = (unsigned long long*) p;
        std::memset(memo_stat_, 0, 8 * sizeof(unsigned long long));
        std::fprintf(stderr, "fnx: memo: %d positions (%.0f MiB), FNX_NOC_FILE / FNX_MEMO decides\n", memo_cap_, n * 2.0 / (1 << 20));
    }
    if (prefetch_) {
        prefetch_want_ = (int) std::max<long long>(0, std::min<long long>(TOPK, env_int("FNX_PREFETCH_WANT", TOPK)));
        prefetch_min_t_ = (int) std::max<long long>(1, std::min<long long>(MAXT, env_int("FNX_PREFETCH_MIN_T", 1)));
        for (CardBuf* b : cb_) {
            ck(cudaSetDevice(b->dev), "set");
            ck(cudaMemcpy(b->pf_want, &prefetch_want_, sizeof prefetch_want_, cudaMemcpyHostToDevice), "prefetch");
        }
        std::fprintf(stderr, "fnx: FNX_PREFETCH: the next layer's guessed experts copied ahead, %d a row, in windows of %d rows and more%s\n", prefetch_want_, prefetch_min_t_,
                     std::getenv("FNX_PREFETCH_FILE") ? " (FNX_PREFETCH_FILE decides)" : "");
    }
    if (pf_refill_ok_ && (pf_refill_env_ || std::getenv("FNX_PF_REFILL_FILE")))
        std::fprintf(stderr, "fnx: FNX_PF_REFILL=%d%s: the prompt path's slots come back full after a read\n", pf_refill_env_,
                     std::getenv("FNX_PF_REFILL_FILE") ? " (FNX_PF_REFILL_FILE decides at each read)" : "");
    prior_ = (float) std::max(0.0, env_num("FNX_ROUTE_PRIOR", 0.0));
    for (int c = 0; c < NCARD; ++c)
        if (m_.cards[c].ec.prior) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            ck(cudaMemcpy(m_.cards[c].ec.prior, &prior_, sizeof prior_, cudaMemcpyHostToDevice), "route prior");
        }
    if (prior_ > 0.f || std::getenv("FNX_ROUTE_PRIOR_FILE"))
        std::fprintf(stderr, "fnx: FNX_ROUTE_PRIOR=%g%s: experts in a slot preferred past each row's top 2 (lossy)\n", prior_,
                     std::getenv("FNX_ROUTE_PRIOR_FILE") ? " (FNX_ROUTE_PRIOR_FILE decides)" : "");
    if (!pf_lend_) prefill_alloc();
    // FNX_ARENA: the pool's own pages are those, or as many as every card's free VRAM holds beside the arena pool's
    // slots and FNX_ARENA_RESERVE_MIB, or FNX_ARENA_BASE_PAGES; the slots follow the own pages in the same allocation,
    // and the pool grows into them (lent coldest first) up to where every card keeps 2 MAXT TOPK slots. The buffers
    // above and the draft layer (an Mtp is made before its Engine) are already allocated, so the reserve is only what
    // comes later: the window graphs captured at first use (with batched drafts a verify graph per row total), cuBLAS's
    // workspaces and the lanes' page tables. Default 512: they took ~270 MiB a card on three cards and ~340 on two,
    // through one to six lanes, batched drafts and a 114k prompt
    if (m.arena) {
        const long long reserve_mib = env_int("FNX_ARENA_RESERVE_MIB", 512);
        if (reserve_mib < 0) throw std::runtime_error("FNX_ARENA_RESERVE_MIB: want MiB >= 0, got " + std::to_string(reserve_mib));
        const size_t reserve = (size_t) reserve_mib << 20;
        // a card with less free VRAM than the slots, the reserve and the smallest own pool (two cards, another process
        // on it, a leak): the coldest experts that many slots' worth go to RAM, placed again, and the engine goes on.
        // Too few slots left for a window is still an error (Model::place_experts says so)
        for (int round = 0;; ++round) {
            int more = 0;
            size_t short_mib = 0;
            for (int c = 0; c < NCARD; ++c) {
                const ExpertCache& ec = m.cards[c].ec;
                if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                const size_t sb = ec.slot_bytes[ec.arena_pool],
                             need = (size_t) ec.n[ec.arena_pool] * sb + reserve + (size_t) (extra_pages + 1) * kvp_page(c) * sizeof(uint16_t) +
                                    (pf_lend_ ? 0 : 2 * pf_stage_bytes(c));
                if (fr < need) {   // 256 MiB more than short, so the next engine's few MiB of drift place nothing again
                    more = std::max(more, (int) ((need - fr + (256u << 20) + sb - 1) / sb));
                    short_mib = std::max(short_mib, (need - fr) >> 20);
                }
            }
            if (!more) break;
            if (round == 8) throw std::runtime_error("FNX_ARENA: VRAM still short after placing the experts again 8 times");
            if (!m.perm.empty())   // the permanent experts have no host copy: they cannot be placed again
                throw std::runtime_error("FNX_EXPERT_DYN_GIB: VRAM short by " + std::to_string(short_mib) + " MiB on a card (" + std::to_string(more) +
                                         " slots): raise FNX_EXPERT_RAM_GIB by about " + std::to_string((short_mib * NCARD + 1023) / 1024) + " GiB");
            // two 32 GB cards are short by design; on three or more the experts fit unless something else holds VRAM
            std::fprintf(stderr, "fnx: VRAM short by %zu MiB on a card%s: %d more of the coldest experts to RAM\n", short_mib,
                         NCARD >= 3 ? " (another process on it?)" : "", more);
            m.free_experts();
            m.ram_extra += more;
            m.place_experts();
        }
        if (!pf_lend_) prefill_stages();
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            if (ec.arena_pool < 0) throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has no arena pool");
            ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            const size_t slots = (size_t) ec.n[ec.arena_pool] * ec.slot_bytes[ec.arena_pool], page = kvp_page(c) * sizeof(uint16_t);
            const int fit = fr > slots + reserve ? (int) ((fr - slots - reserve) / page) : 0;
            if (fit <= extra_pages)
                throw std::runtime_error("FNX_ARENA: card " + std::to_string(c) + " has " + std::to_string(fr >> 20) + " MiB free: the arena's " +
                                         std::to_string(slots >> 20) + " MiB of slots, " + std::to_string(reserve >> 20) +
                                         " MiB in reserve (FNX_ARENA_RESERVE_MIB) and the lanes' blocks do not fit");
            n_pages = std::min(n_pages, fit);
        }
        if (std::getenv("FNX_ARENA_BASE_PAGES")) {
            const long long want = env_int("FNX_ARENA_BASE_PAGES", 0);
            if (want <= extra_pages || want > n_pages)
                throw std::runtime_error("FNX_ARENA_BASE_PAGES: want " + std::to_string(extra_pages + 1) + " to " + std::to_string(n_pages) +
                                         " (the lanes' blocks take " + std::to_string(extra_pages) + ")");
            n_pages = (int) want;
        }
        if (pf_lend_) {
            // each pool gives its share of what the prompt path holds, by bytes (FNX_PF_TAIL_MIB: its buffers and stages at FNX_PF_ROWS 4096);
            // the arena pool no more than K/V can spare at the largest context, the other pools the rest
            const size_t want = (size_t) env_int("FNX_PF_TAIL_MIB", 2450) << 20;
            const int need_pages = max_ctx / KVP_POS + 2 + extra_pages;
            for (int c = 0; c < NCARD; ++c) {
                const ExpertCache& ec = m.cards[c].ec;
                const int ap = ec.arena_pool, least = 2 * MAXT * TOPK;
                const size_t page = kvp_page(c) * sizeof(uint16_t);
                size_t total = 0;
                for (int p = 0; p < ec.n_pools; ++p) total += (size_t) ec.n[p] * ec.slot_bytes[p];
                const size_t own = ((size_t) n_pages * page + 255) & ~(size_t) 255, need = (size_t) need_pages * page;
                const long long for_kv = need > own ? (long long) ((need - own + ec.slot_bytes[ap] - 1) / ec.slot_bytes[ap]) : 0;
                const int spare = (int) std::max(0LL, (long long) ec.n[ap] - least - for_kv);
                pf_tail_[c][ap] = std::min(spare, (int) ((double) want * (double) ((size_t) ec.n[ap] * ec.slot_bytes[ap]) / (double) total / (double) ec.slot_bytes[ap]));
                const size_t rest = want - std::min(want, (size_t) pf_tail_[c][ap] * ec.slot_bytes[ap]);
                const size_t others = total - (size_t) ec.n[ap] * ec.slot_bytes[ap];
                for (int p = 0; p < ec.n_pools; ++p)
                    if (p != ap) {
                        const double share = (double) rest * (double) ((size_t) ec.n[p] * ec.slot_bytes[p]) / (double) std::max<size_t>(others, 1);
                        pf_tail_[c][p] = std::max(0, std::min((int) (share / (double) ec.slot_bytes[p]), ec.n[p] - least));
                    }
                if (c == 0) {
                    std::string line;
                    for (int p = 0; p < ec.n_pools; ++p) line += " " + std::to_string(pf_tail_[c][p]) + (p == ap ? " (arena)" : "") + " of " + std::to_string(ec.n[p]);
                    std::fprintf(stderr, "fnx: slots lent to the prompt path, by pool:%s\n", line.c_str());
                }
            }
        }
        arena_base_ = arena_w_ = n_pages;
        arena_cap_ = INT32_MAX;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool, keep = 2 * MAXT * TOPK + pf_tail_[c][p];
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            const size_t room = (size_t) (ec.n[p] - keep) * ec.slot_bytes[p] + (arena_slot_off(c) - (size_t) n_pages * page);
            arena_cap_ = std::min(arena_cap_, n_pages + (int) (room / page));
        }
        // FNX_EXPERT_DYN_GIB: the K/V pool grows only into the arena pool's dynamic slots; short of the context, a long request
        // would find it full half-way, so the load says so now
        if (const int need = max_ctx / KVP_POS + 2 + extra_pages; !m.perm.empty() && arena_cap_ < need)
            throw std::runtime_error("FNX_EXPERT_DYN_GIB: the K/V pool reaches " + std::to_string(arena_cap_ * KVP_POS) + " positions, --max-context " +
                                     std::to_string(max_ctx) + " needs " + std::to_string(need - arena_cap_) + " more pages (" +
                                     std::to_string((size_t) (need - arena_cap_) * kvp_page(0) * sizeof(uint16_t) >> 20) +
                                     " MiB a card) of the arena pool's dynamic slots: raise FNX_EXPERT_DYN_GIB or lower --max-context");
    } else {
        prefill_stages();
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf* b = cb_[c];
        ck(cudaSetDevice(b->dev), "cudaSetDevice");
        // the K/V pool and lane 0's page table (pages taken as it grows); with an arena the pool is its start and
        // n_pages counts the pages it can grow to
        if (m.arena) {
            ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            b->n_pages = arena_cap_;
            char* region = (char*) dalloc<uint8_t>(arena_slot_off(c) + (size_t) ec.n[p] * ec.slot_bytes[p]);
            b->pool = (uint16_t*) region;
            ec.pool[p] = region + arena_slot_off(c);
            m.cards[c].bytes += (size_t) ec.n[p] * ec.slot_bytes[p];
            arena_fill(c, b->st);
            size_t fr = 0, tot = 0;
            ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            std::fprintf(stderr, "fnx: arena: card %d: %zu MiB free after its arena\n", c, fr >> 20);
        } else {
            b->n_pages = n_pages;
            b->pool = dalloc<uint16_t>((size_t) b->n_pages * kvp_page(c));
        }
        b->pt = dalloc<int>((size_t) b->n_pages);
        b->pg.pt = b->pt;
        b->pg.page = kvp_page(c);
        b->pg.rows = kvp_rows(c);
        b->pg.q8 = kv_q8();
        const size_t kv_halves = kvp_rows_halves(c, kv_q8());
        for (int l = 0, slot = 0; l < NL; ++l)
            if (is_attn(l)) {
                b->kc[l] = b->pool + (size_t) slot * kvp_slot(c);
                b->vc[l] = b->kc[l] + kv_halves;
                b->qsa_pooled[l] = b->kc[l] + 2 * kv_halves;
                ++slot;
            }
        {   // lane 0: the buffers above
            LaneDev L;
            L.pt = b->pt;
            L.S = b->S;
            L.chist = b->chist;
            L.ple_hist = b->ple_hist;
            L.ring = b->qsa_ring;
            L.ctl = dalloc<int>(2);
            b->lanes.push_back(L);
        }
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "init sync"); }
    lanes_.emplace_back();
    if (pf_lend_) {
        pf_ranges_.resize(NCARD);
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            for (int p = 0; p < ec.n_pools; ++p)
                if (pf_tail_[c][p] > 0)
                    pf_ranges_[(size_t) c].emplace_back(ec.pool[p] + (size_t) (ec.n[p] - pf_tail_[c][p]) * ec.slot_bytes[p], (size_t) pf_tail_[c][p] * ec.slot_bytes[p]);
            std::sort(pf_ranges_[(size_t) c].begin(), pf_ranges_[(size_t) c].end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        }
        pf_tails(true);   // the buffers are zeroed as they are carved: their slots first
        // the stages first: the largest buffers, and they come last
        pf_stage_ranges_.resize(NCARD);
        for (int c = 0; c < NCARD; ++c) {
            auto& rs = pf_ranges_[(size_t) c];
            const size_t sb = pf_stage_bytes(c) + 256;
            for (int k = 0; k < 2 && sb > 256; ++k) {
                auto it = std::max_element(rs.begin(), rs.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
                if (it == rs.end() || it->second < sb) break;
                pf_stage_ranges_[(size_t) c].emplace_back(it->first, sb);
                it->first += sb;
                it->second -= sb;
            }
            std::sort(rs.begin(), rs.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        }
        try {
            prefill_alloc();
            prefill_stages();
        } catch (const std::exception& e) {
            if (m.perm.empty()) throw;
            throw std::runtime_error(std::string(e.what()) + " (FNX_EXPERT_DYN_GIB: the prompt path's buffers go in the pools' last dynamic slots, "
                                                             "too few of them: raise FNX_EXPERT_DYN_GIB)");
        }
        size_t left = 0, lent = 0;
        for (const auto& r : pf_ranges_[0]) left += r.second;
        for (int p = 0; p < m.cards[0].ec.n_pools; ++p) lent += (size_t) pf_tail_[0][p] * m.cards[0].ec.slot_bytes[p];
        std::fprintf(stderr, "fnx: prompt buffers in the last slots of the expert pools: %zu MiB lent while reading, %zu MiB of it unused (card 0)\n", lent >> 20, left >> 20);
    }
    for (int i = m.arena ? arena_base_ : cb_[0]->n_pages; i-- > 0;) free_pages_.push_back(i);   // taken from the back: low pages first
    if (m.arena) {
        std::string per;
        for (int c = 0; c < NCARD; ++c) {
            const ExpertCache& ec = m.cards[c].ec;
            const int p = ec.arena_pool;
            char buf[96];
            std::snprintf(buf, sizeof buf, "%s%d x %.2f MiB (%.1f GiB)", c ? ", " : "", ec.n[p], ec.slot_bytes[p] / 1048576.0,
                          (double) ec.n[p] * ec.slot_bytes[p] / (1ull << 30));
            per += buf;
        }
        std::fprintf(stderr, "fnx: arena: the K/V pool's own %d pages (%d positions), up to %d (%d positions) by lending the slots of %s\n",
                     arena_base_, arena_base_ * KVP_POS, arena_cap_, arena_cap_ * KVP_POS, per.c_str());
    }
    if (std::getenv("FNX_RING_TAG_START")) {
        const long long v = env_int("FNX_RING_TAG_START", 0);
        if (v < 1 || v >= (long long) TAG_END)
            throw std::runtime_error("FNX_RING_TAG_START: want 1.." + std::to_string(TAG_END - 1) + ", got " + std::to_string(v));
        tag_max_ = (unsigned) v;
    }
}

void Engine::clear_rings() {
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaMemset(b.ring.in, 0, ring_in_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.ring.ag, 0, ring_ag_words(MAXT) * sizeof(unsigned long long)), "ring clear");
        ck(cudaMemset(b.att_ring.in, 0, att_ring_words() * sizeof(unsigned long long)), "ring clear");
        if (b.lxg_in) {   // edge0_patch6.py
            ck(cudaMemset(b.lxg_in, 0, lat_x_words() * sizeof(unsigned long long)), "ring clear");
            ck(cudaMemset(b.lzs_in, 0, lat_x_words() * sizeof(unsigned long long)), "ring clear");
        }
        if (!pf_.empty()) pf_clear_ring(c);
    }
    for (CardBuf* b : cb_) { ck(cudaSetDevice(b->dev), "set"); ck(cudaDeviceSynchronize(), "ring clear sync"); }
    std::fprintf(stderr, "fnx: ring tags reached %u: the rings' buffers cleared, tags start again\n", tag_max_);
    tag_max_ = 1;
}

void Engine::tags_room(unsigned n) {
    if (tag_max_ + n + 1u >= TAG_END) clear_rings();
}

// an arena's K/V pool grows by at least this many pages a lend (64: ~12k positions f16), so lends stay rare
constexpr int ARENA_STEP = 64;

size_t Engine::arena_slot_off(int c) const {
    return ((size_t) arena_base_ * kvp_page(c) * sizeof(uint16_t) + 255) & ~(size_t) 255;
}

// the arena pool as loaded: the slot tables back to the model's first assignment (an earlier engine in this process
// lent, reclaimed and replaced slots), every expert of the pool's layers at its host copy, then each slot's expert
// copied in from it
void Engine::arena_fill(int c, cudaStream_t st) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool;
    // only the arena pool's part: the other pools' slots keep what an earlier engine copied into them, and their
    // tables say so
    const int s0 = ec.s0[p], n = ec.n[p];
    ck(cudaMemcpy(ec.owner + s0, w.ec_owner0.data() + s0, (size_t) n * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) slot_of[(size_t) l * NEXP + e] = w.ec_slot0[(size_t) l * NEXP + e];
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemset(ec.ref + s0, 0, (size_t) n), "arena: refs");
    ck(cudaMemcpy(ec.hand + p, &ec.s0[p], sizeof(int), cudaMemcpyHostToDevice), "arena: hand");
    ck(cudaMemset(ec.lent + p, 0, sizeof(int)), "arena: lent");
    const std::vector<int> owner(w.ec_owner0.begin() + s0, w.ec_owner0.begin() + s0 + n);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e) {
                const size_t i = (size_t) l * NEXP + e;
                // FNX_EXPERT_DYN_GIB: a permanent expert has no host copy and never leaves its own VRAM block
                const bool perm = !w.perm_gu.empty() && w.perm_gu[i];
                epg[i] = perm ? w.perm_gu[i] : w.ec_host_gu[i];
                epd[i] = perm ? w.perm_dn[i] : w.ec_host_dn[i];
            }
    // the slots' experts in one kernel reading their host copies (a copy call each ran the cards one after another)
    std::vector<int> list{n};
    for (int s = 0; s < ec.n[p]; ++s) {
        const int i = owner[(size_t) s];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        list.push_back(i);
    }
    int* dlist = dalloc<int>(list.size());
    ck(cudaMemcpy(dlist, list.data(), list.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: fill list");
    cache_copy(ec, p, dlist, st);
    ck(cudaMemcpyAsync(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
    // no wait here: the cards fill at once, and the constructor syncs every card before the engine is used (the tables
    // come from pageable vectors, copied out before cudaMemcpyAsync returns)
    ck(cudaMemcpyAsync(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice, st), "arena: tables");
}

void Engine::pf_tails(bool lend) {
    if (!pf_lend_ || pf_tails_lent_ == lend) return;
    pf_refill_stop();
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        ck(cudaDeviceSynchronize(), "prompt slots: cards idle");
    }
    // FNX_PF_REFILL: given back full, each still lent until its expert has landed (pf_refill_start)
    const int refill = lend ? 0 : pf_refill_mode();
    if (!refill || !pf_refill_start(refill))
        for (int c = 0; c < NCARD; ++c) {
            CardBuf& b = *cb_[c];
            const ExpertCache& ec = m_.cards[c].ec;
            ck(cudaSetDevice(b.dev), "set");
            for (int p = 0; p < ec.n_pools; ++p)
                if (pf_tail_[c][p] > 0) p100_tail_slots(ec, p, pf_tail_[c][p], lend, b.st);
            ck(cudaStreamSynchronize(b.st), "prompt slots");
        }
    pf_tails_lent_ = lend;
}

// FNX_PF_REFILL: 0 off, 1 by the card's copier while the windows run, 2 all at once before the first window. FNX_PF_REFILL_FILE: a file
// whose number decides at each read (tests: every mode in one load)
int Engine::pf_refill_mode() const {
    if (!pf_refill_ok_) return 0;
    static const char* file = std::getenv("FNX_PF_REFILL_FILE");
    int mode = pf_refill_env_;
    if (file)
        if (FILE* f = std::fopen(file, "r")) {
            int v = 0;
            if (std::fscanf(f, "%d", &v) == 1) mode = v;
            std::fclose(f);
        }
    mode = std::max(0, std::min(2, mode));
    return mode == 1 && !cb_[0]->dma ? 2 : mode;
}

// FNX_PF_REFILL: the prompt path's slots back from a read, the cards idle. Each takes again the expert it held before (k_tail_slots kept
// it), or else the hottest of its pool's experts out of a slot (the profile's score); the hottest are copied first. 1: by the card's copier
// while the windows run, each slot lent until its expert has landed on every card (pf_refill_tick); 2: copied here. Every card takes the
// same experts into the same slots in the same window, so their slots stay alike. false: the cards' slots differ (they come back empty)
bool Engine::pf_refill_start(int mode) {
    auto heat = [&](int g) { return (size_t) g < m_.score.size() ? m_.score[(size_t) g] : 0.0; };
    auto hotter = [&](int a, int b) { return heat(a) > heat(b); };
    std::vector<std::vector<int>> lists(NCARD), picks(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        const ExpertCache& ec = m_.cards[c].ec;
        ck(cudaSetDevice(cb_[c]->dev), "set");
        std::vector<int> slot_of((size_t) NL * NEXP);
        std::vector<int>& list = lists[(size_t) c];
        list.resize((size_t) ec.n_slots);
        ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "refill: slots");
        ck(cudaMemcpy(list.data(), ec.fill_list, list.size() * sizeof(int), cudaMemcpyDeviceToHost), "refill: held");
        std::vector<char> taken(slot_of.size(), 0);
        for (int p = 0; p < ec.n_pools; ++p) {
            const int cnt = pf_tail_[c][p];
            if (cnt <= 0) continue;
            const int t0 = ec.s0[p] + ec.n[p] - cnt;
            std::vector<int> pick;
            for (int i = 0; i < cnt; ++i)   // what they held, still out of a slot
                if (const int g = list[(size_t) (t0 + i)]; g >= 0 && slot_of[(size_t) g] < 0 && !taken[(size_t) g]) {
                    taken[(size_t) g] = 1;
                    pick.push_back(g);
                }
            if ((int) pick.size() < cnt) {   // the rest: the pool's hottest out of a slot
                std::vector<int> out;
                for (int l = 0; l < NL; ++l)
                    if (ec.pool_of[l] == p)
                        for (int e = 0; e < NEXP; ++e)
                            if (const int g = l * NEXP + e; slot_of[(size_t) g] < 0 && !taken[(size_t) g] && m_.kept(g)) out.push_back(g);   // FNX_EXPERT_MASK
                const size_t k = std::min(out.size(), (size_t) cnt - pick.size());
                std::partial_sort(out.begin(), out.begin() + (long) k, out.end(), hotter);
                pick.insert(pick.end(), out.begin(), out.begin() + (long) k);
            }
            std::stable_sort(pick.begin(), pick.end(), hotter);
            pick.resize((size_t) cnt, -1);
            std::copy(pick.begin(), pick.end(), list.begin() + t0);
            picks[(size_t) c].insert(picks[(size_t) c].end(), pick.begin(), pick.end());
        }
    }
    for (int c = 1; c < NCARD; ++c)
        if (picks[(size_t) c] != picks[0] || std::memcmp(pf_tail_[c], pf_tail_[0], sizeof pf_tail_[0]) != 0) {
            static bool said = false;
            if (!said) std::fprintf(stderr, "fnx: FNX_PF_REFILL: the cards' slots differ, the prompt path's come back empty\n");
            said = true;
            return false;
        }
    const bool copier = mode == 1;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const ExpertCache& ec = w.ec;
        ck(cudaSetDevice(b.dev), "set");
        std::vector<DmaFill> jobs;
        std::vector<double> jheat;
        int state[3 * EC_POOLS] = {}, landed[EC_POOLS] = {};
        for (int p = 0; p < ec.n_pools; ++p) {
            const int cnt = pf_tail_[c][p];
            if (cnt <= 0) continue;
            const int t0 = ec.s0[p] + ec.n[p] - cnt;
            const size_t first = jobs.size();
            for (int i = 0; i < cnt; ++i)
                if (const int g = lists[(size_t) c][(size_t) (t0 + i)]; g >= 0) {
                    const LayerW& L = w.L[(size_t) (g / NEXP)];
                    char* at = ec.pool[p] + (size_t) (t0 + i - ec.s0[p]) * ec.slot_bytes[p];
                    jobs.push_back({w.ec_host_gu[(size_t) g], at, w.ec_host_dn[(size_t) g], at + ec.dn_off[p], L.gu_e_stride, L.down_e_stride, p, i + 1});
                    jheat.push_back(heat(g));
                }
            if (jobs.size() == first) landed[p] = cnt;   // nothing to copy: they join empty at once
            else jobs.back().done = cnt;                 // the empty ones after the last copy join with it
            state[2 * EC_POOLS + p] = cnt;
            if (!copier) state[p] = cnt;
        }
        ck(cudaMemcpy(ec.fill_list, lists[(size_t) c].data(), lists[(size_t) c].size() * sizeof(int), cudaMemcpyHostToDevice), "refill: list");
        ck(cudaMemcpy(ec.fill, state, sizeof state, cudaMemcpyHostToDevice), "refill: state");
        if (copier) {   // the pools interleaved, the hottest first (each pool's own order stays, and with it its counts)
            std::vector<size_t> order(jobs.size());
            std::iota(order.begin(), order.end(), (size_t) 0);
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b2) { return jheat[a] > jheat[b2]; });
            std::vector<DmaFill> sorted;
            sorted.reserve(jobs.size());
            for (size_t k : order) sorted.push_back(jobs[k]);
            p100_dma_fill(b.dma, std::move(sorted), landed);
        } else
            for (const DmaFill& f : jobs) {
                ck(cudaMemcpyAsync(f.dst_gu, f.src_gu, f.gu_bytes, cudaMemcpyHostToDevice, b.st), "refill");
                ck(cudaMemcpyAsync(f.dst_dn, f.src_dn, f.dn_bytes, cudaMemcpyHostToDevice, b.st), "refill");
            }
    }
    if (!copier)
        for (int c = 0; c < NCARD; ++c) {   // 2: landed before the first window, which takes them all
            ck(cudaSetDevice(cb_[c]->dev), "set");
            ck(cudaStreamSynchronize(cb_[c]->st), "refill");
        }
    for (int p = 0; p < EC_POOLS; ++p) {
        pf_fill_cnt_[p] = pf_tail_[0][p];
        pf_fill_to_[p] = copier ? 0 : pf_fill_cnt_[p];
    }
    pf_refill_live_ = copier;
    pf_fill_dirty_ = true;
    return true;
}

// FNX_PF_REFILL: the copiers stopped (what they issued has landed) and the cards' figures cleared, before the slots are lent again
void Engine::pf_refill_stop() {
    if (!pf_fill_dirty_) return;
    for (int c = 0; c < NCARD; ++c) {
        ck(cudaSetDevice(cb_[c]->dev), "set");
        if (cb_[c]->dma) p100_dma_fill_stop(cb_[c]->dma);
        ck(cudaDeviceSynchronize(), "refill: cards idle");
        ck(cudaMemset(m_.cards[c].ec.fill, 0, 3 * EC_POOLS * sizeof(int)), "refill: state");
    }
    pf_refill_live_ = pf_fill_dirty_ = false;
}

// before a window (FNX_PF_REFILL=1): of each pool's slots, as many as have landed on every card, one figure for all the cards. true: it
// grew, and the window's upload takes it to the cards
bool Engine::pf_refill_tick() {
    if (!pf_refill_live_) return false;
    int to[EC_POOLS];
    std::fill(to, to + EC_POOLS, 1 << 30);
    for (int c = 0; c < NCARD; ++c) {
        int l[EC_POOLS];
        p100_dma_landed(cb_[c]->dma, l);
        for (int p = 0; p < EC_POOLS; ++p) to[p] = std::min(to[p], l[p]);
    }
    bool grew = false, all = true;
    for (int p = 0; p < EC_POOLS; ++p) {
        if (to[p] > pf_fill_to_[p]) {
            pf_fill_to_[p] = to[p];
            grew = true;
        }
        all = all && pf_fill_to_[p] >= pf_fill_cnt_[p];
    }
    if (all) pf_refill_live_ = false;   // the copiers are done; this window's upload takes the last figure
    return grew;
}

// FNX_ROUTE_PRIOR_FILE: its number (lambda, 0 off) at most every 20 ms; true when it changed (the window's upload takes it to the cards)
bool Engine::prior_tick() {
    static const char* file = std::getenv("FNX_ROUTE_PRIOR_FILE");
    if (!file) return false;
    static std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds(20)) return false;
    last = now;
    float v = prior_;
    if (FILE* f = std::fopen(file, "r")) {
        if (std::fscanf(f, "%f", &v) != 1) v = prior_;
        std::fclose(f);
    }
    v = std::max(0.f, std::min(4.f, v));
    if (v == prior_) return false;
    prior_ = v;
    return true;
}

// FNX_PREFETCH_FILE: its number (guesses a row, 0 off) at most every 20 ms; true when it changed (the window's upload takes it to the cards)
bool Engine::prefetch_tick() {
    static const char* file = std::getenv("FNX_PREFETCH_FILE");
    if (!prefetch_ || !file) return false;
    static std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds(20)) return false;
    last = now;
    int v = prefetch_want_;
    if (FILE* f = std::fopen(file, "r")) {
        if (std::fscanf(f, "%d", &v) != 1) v = prefetch_want_;
        std::fclose(f);
    }
    v = std::max(0, std::min(TOPK, v));
    if (v == prefetch_want_) return false;
    prefetch_want_ = v;
    return true;
}

int Engine::arena_slots_for(int c, int w) const {
    const ExpertCache& ec = m_.cards[c].ec;
    const long need = (long) w * (long) (kvp_page(c) * sizeof(uint16_t)) - (long) arena_slot_off(c);
    if (need <= 0) return 0;
    const long sb = (long) ec.slot_bytes[ec.arena_pool];
    return (int) ((need + sb - 1) / sb);
}

std::vector<int> Engine::arena_lent() const {
    std::vector<int> v;
    if (arena_cap_)
        for (int c = 0; c < NCARD; ++c) v.push_back(arena_slots_for(c, arena_w_));
    return v;
}

// slots [to, from) of card c's arena pool, just given back empty: each takes one of the pool's experts not in a slot,
// the hottest first (the profile's score), copied in from its host copy (the card idle). Nothing lent then means every
// expert is in a slot, which the plan's early return relies on (cache_plan, moe.cuh)
void Engine::arena_refill(int c, int from, int to) {
    ExpertCache& ec = m_.cards[c].ec;
    const CardW& w = m_.cards[c];
    const int p = ec.arena_pool, s0 = ec.s0[p];
    std::vector<int> slot_of((size_t) NL * NEXP);
    ck(cudaMemcpy(slot_of.data(), ec.slot_of, slot_of.size() * sizeof(int), cudaMemcpyDeviceToHost), "arena: slots");
    std::vector<int> out;   // the pool's experts not in a slot
    for (int l = 0; l < NL; ++l)
        if (ec.pool_of[l] == p)
            for (int e = 0; e < NEXP; ++e)
                if (slot_of[(size_t) l * NEXP + e] < 0 && m_.kept(l * NEXP + e)) out.push_back(l * NEXP + e);   // FNX_EXPERT_MASK: never a dropped one
    std::stable_sort(out.begin(), out.end(), [&](int a, int b) { return m_.score[(size_t) a] > m_.score[(size_t) b]; });
    if (ec.arena_full && to == 0 && (int) out.size() > from)
        throw std::runtime_error("arena: " + std::to_string(out.size()) + " experts out of a slot, " + std::to_string(from) + " slots to take them");
    const int n = std::min((int) out.size(), from - to);
    std::vector<int> owner((size_t) (from - to), -1);
    std::vector<const char*> epg((size_t) NL * NEXP), epd(epg.size());
    ck(cudaMemcpy(epg.data(), ec.ep_gu, epg.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    ck(cudaMemcpy(epd.data(), ec.ep_dn, epd.size() * sizeof(char*), cudaMemcpyDeviceToHost), "arena: tables");
    for (int k = 0; k < n; ++k) {
        const int i = out[(size_t) k], s = to + k;
        const LayerW& L = w.L[(size_t) (i / NEXP)];
        char* at = ec.pool[p] + (size_t) s * ec.slot_bytes[p];
        ck(cudaMemcpy(at, w.ec_host_gu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), "arena: refill");
        ck(cudaMemcpy(at + ec.dn_off[p], w.ec_host_dn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), "arena: refill");
        epg[(size_t) i] = at;
        epd[(size_t) i] = at + ec.dn_off[p];
        slot_of[(size_t) i] = s0 + s;
        owner[(size_t) k] = i;
    }
    ck(cudaMemcpy(ec.ep_gu, epg.data(), epg.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.ep_dn, epd.data(), epd.size() * sizeof(char*), cudaMemcpyHostToDevice), "arena: tables");
    ck(cudaMemcpy(ec.slot_of, slot_of.data(), slot_of.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: slots");
    ck(cudaMemcpy(ec.owner + s0 + to, owner.data(), owner.size() * sizeof(int), cudaMemcpyHostToDevice), "arena: owners");
}

// lending: the slots' experts to RAM (their tables first), then K/V may write there; giving back: the pages are free,
// the slots come back and are filled again at once (arena_refill). Every card idle first: a window or a prompt pass in
// flight may read a slot being lent, or a page being given back
void Engine::arena_set(int w) {
    if (w == arena_w_) return;
    if (w < arena_base_ || w > arena_cap_) throw std::runtime_error("arena: top " + std::to_string(w) + " outside its range");
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const ExpertCache& ec = m_.cards[c].ec;
        ck(cudaSetDevice(b.dev), "set");
        ck(cudaDeviceSynchronize(), "arena: cards idle");
        const int from = arena_slots_for(c, arena_w_), to = arena_slots_for(c, w);
        if (to > ec.n[ec.arena_pool] - 2 * MAXT * TOPK - pf_tail_[c][ec.arena_pool]) throw std::runtime_error("arena: card " + std::to_string(c) + " would keep too few slots");
        arena_slots(ec, from, to, b.st);
        ck(cudaStreamSynchronize(b.st), "arena: slots");
        if (to < from) arena_refill(c, from, to);
    }
    if (w > arena_w_) {
        for (int pg = arena_w_; pg < w; ++pg) free_pages_.push_back(pg);
        // zeroed: they held expert weights, and the draft layer reads K/V it never wrote (batched windows skip it), which
        // as int8 rows and scales could be inf or NaN (the own pages only ever held finite K/V or zeros)
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            const size_t page = kvp_page(c) * sizeof(uint16_t);
            ck(cudaMemset((char*) cb_[c]->pool + (size_t) arena_w_ * page, 0, (size_t) (w - arena_w_) * page), "arena: zero");
        }
        arena_last_lend_ = std::chrono::steady_clock::now();
    } else {
        free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int pg) { return pg >= w; }), free_pages_.end());
    }
    std::sort(free_pages_.begin(), free_pages_.end(), std::greater<int>());
    const int old = arena_w_;
    arena_w_ = w;
    const std::vector<int> now = arena_lent();
    std::string per;
    double gib = 0;
    for (int c = 0; c < NCARD; ++c) {
        const ExpertCache& ec = m_.cards[c].ec;
        per += (c ? " / " : "") + std::to_string(now[(size_t) c]);
        gib = std::max(gib, (double) now[(size_t) c] * ec.slot_bytes[ec.arena_pool] / (1ull << 30));
    }
    std::fprintf(stderr, "fnx: arena: K/V pool %d -> %d pages (own %d, at most %d); slots lent %s (up to %.2f GiB of a card's experts in RAM), in %.1f ms\n",
                 old, w, arena_base_, arena_cap_, per.c_str(), gib,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Engine::arena_reclaim(double hold_s) {
    if (!arena_cap_ || arena_w_ == arena_base_) return;
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - arena_last_lend_).count() < hold_s) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    if (top < arena_w_) arena_set(top);
}

// FNX_ARENA_CHURN=1 (a test): before every window, ARENA_STEP more pages lent and the pool's top then given back to the
// highest page in use (those slots refilled), so lending and giving back run beside every kind of window
void Engine::arena_churn() {
    static const bool on = env_flag("FNX_ARENA_CHURN");
    if (!on || !arena_cap_) return;
    int top = arena_base_;
    for (const LaneHost& L : lanes_)
        for (int pg : L.pages) top = std::max(top, pg + 1);
    arena_set(std::min(arena_cap_, std::max(arena_w_, top) + ARENA_STEP));
    arena_set(top);
}

void Engine::ensure_pages(int n_pos) { ensure_pages_of(cur_, n_pos); }

void Engine::ensure_pages_of(int lane, int n_pos) {
    LaneHost& L = lanes_[(size_t) lane];
    const int need = n_pos / KVP_POS + 1;   // the page of position n_pos too: a block's pooled key can sit one page on
    if (need > cb_[0]->n_pages) throw Refused("context full: " + std::to_string(n_pos) + " positions");
    const size_t had = L.pages.size();
    // with an arena, slots lent for what the free pages lack: at least ARENA_STEP pages at a time
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && arena_w_ < arena_cap_)
        arena_set(std::min(arena_cap_, arena_w_ + std::max(short_by, ARENA_STEP)));
    if (const int short_by = need - (int) had - (int) free_pages_.size(); short_by > 0 && on_pool_short && !in_pool_short_ && pending_.empty()) {
        const int was = cur_;
        in_pool_short_ = true;
        try {
            on_pool_short(lane, short_by);
        } catch (...) {
            in_pool_short_ = false;
            throw;
        }
        in_pool_short_ = false;
        if (cur_ != was) throw std::runtime_error("on_pool_short left lane " + std::to_string(cur_) + " current, not " + std::to_string(was));
    }
    // all the pages or none: a lane holding pages its device page table lacks would write K/V through stale entries
    if (need - (int) had > (int) free_pages_.size())
        throw Refused("the K/V pool is full (" + std::to_string(n_pos) + " positions on this lane, " + std::to_string(need - (int) had) +
                      " more pages wanted, " + std::to_string(free_pages_.size()) + " free)");
    while ((int) L.pages.size() < need) {
        L.pages.push_back(free_pages_.back());
        free_pages_.pop_back();
    }
    if (L.pages.size() == had) return;
    for (CardBuf* b : cb_) {
        ck(cudaSetDevice(b->dev), "set");
        ck(cudaMemcpy(b->lanes[(size_t) lane].pt + had, L.pages.data() + had, (L.pages.size() - had) * sizeof(int), cudaMemcpyHostToDevice),
           "page table");
    }
}

std::vector<unsigned long long> Engine::probe_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.probe_cnt) return {};
    std::vector<unsigned long long> v(7);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.probe_cnt, 7 * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "probe counts");
    return v;
}

std::vector<unsigned> Engine::route_counts() const {
    const CardBuf& b = *cb_[0];
    if (!b.route_cnt) return {};
    std::vector<unsigned> v((size_t) 2 * NL * NEXP);
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(v.data(), b.route_cnt, v.size() * sizeof(unsigned), cudaMemcpyDeviceToHost), "route counts");
    return v;
}

std::vector<unsigned> Engine::route_trace(size_t from) const {
    const CardBuf& b = *cb_[0];
    if (!b.route_trace) return {};
    unsigned long long n = 0;
    ck(cudaSetDevice(b.dev), "set");
    ck(cudaMemcpy(&n, b.route_trace_len, sizeof n, cudaMemcpyDeviceToHost), "route trace length");
    if (n > b.route_trace_cap) throw std::runtime_error("FNX_ROUTE_TRACE: " + std::to_string(n) + " picks, room for " +
                                                         std::to_string(b.route_trace_cap) + ": raise it");
    if (from >= n) return {};
    std::vector<unsigned> v((size_t) (n - from));
    std::memcpy(v.data(), b.route_trace + from, v.size() * sizeof(unsigned));
    return v;
}

std::vector<int> Engine::graph_key(const std::vector<int>& lanes) const {
    std::vector<int> k(lanes);
    for (int l : lanes) k.push_back(cb_[0]->lanes[(size_t) l].mrope != nullptr);
    k.push_back(cache_live());
    k.push_back(noc_gen());   // FNX_NOC_FILE: a new mode, new graphs
    return k;
}

// a lane's states on each card, carved from a run of pool pages: recurrent states, conv histories (two halves), the
// PLE history, the QSA raw-key rings (every attention layer's and the draft layer's)
int Engine::lane_block_pages() const {
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    int n = 0;   // the most any card needs (pages differ in size between cards)
    for (int c = 0; c < NCARD; ++c) {
        const CardW& w = m_.cards[c];
        size_t bytes = (size_t) 9 * D * 4 + ring_bytes + 256 * 64;   // the PLE history, the draft layer's ring, alignment
        for (int l = 0; l < NL; ++l)
            bytes += is_attn(l) ? ring_bytes : (size_t) w.nv() * HS * HS * 4 + (size_t) 2 * 3 * w.conv_ch() * 4;
        const size_t page = kvp_page(c) * sizeof(uint16_t);
        n = std::max(n, (int) ((bytes + page - 1) / page));
    }
    return n;
}

int Engine::add_lane() {
    if (!pending_.empty() || !rows_pending_.lanes.empty()) throw std::runtime_error("add_lane: a window is not committed");
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    auto page_bytes = [](int c) { return kvp_page(c) * sizeof(uint16_t); };
    const int n = lane_block_pages();
    // a run of n free pages, highest first
    std::vector<int> fr(free_pages_);
    std::sort(fr.begin(), fr.end());
    int start = -1;
    for (int i = (int) fr.size() - n; i >= 0; --i)
        if (fr[(size_t) i + n - 1] - fr[(size_t) i] == n - 1) {
            start = fr[(size_t) i];
            break;
        }
    // an arena lends the run at its top (a lane's block stays as long as the engine: those slots stay lent)
    if (start < 0 && arena_w_ + n <= arena_cap_) {
        start = arena_w_;
        arena_set(arena_w_ + n);
        std::fprintf(stderr, "fnx: arena: lane %zu's block at pages %d..%d, lent for good (no run of %d free pages below)\n", lanes_.size(),
                     start, start + n - 1, n);
    }
    if (start < 0) throw std::runtime_error("add_lane: no run of " + std::to_string(n) + " free pages in the K/V pool");
    free_pages_.erase(std::remove_if(free_pages_.begin(), free_pages_.end(), [&](int p) { return p >= start && p < start + n; }),
                      free_pages_.end());
    LaneHost h;
    h.block0 = start;
    h.block_n = n;
    lanes_.push_back(h);
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        char* at = (char*) (b.pool + (size_t) start * kvp_page(c));
        ck(cudaMemset(at, 0, (size_t) n * page_bytes(c)), "lane block");
        auto take = [&](size_t bytes) {
            char* p = at;
            at += (bytes + 255) & ~(size_t) 255;
            return p;
        };
        LaneDev L;
        L.pt = dalloc<int>((size_t) b.n_pages);
        L.S.assign(NL, nullptr);
        L.chist.assign(NL, nullptr);
        L.ring.assign(NL, nullptr);
        for (int l = 0; l < NL; ++l) {
            if (is_attn(l)) {
                L.ring[l] = (uint16_t*) take(ring_bytes);
            } else {
                L.S[l] = (float*) take((size_t) w.nv() * HS * HS * 4);
                L.chist[l] = (float*) take((size_t) 2 * 3 * w.conv_ch() * 4);
            }
        }
        L.ple_hist = (float*) take((size_t) 9 * D * 4);
        L.mtp_ring = (uint16_t*) take(ring_bytes);
        L.ctl = dalloc<int>(2);
        b.lanes.push_back(L);
    }
    return (int) lanes_.size() - 1;
}

// the token rows into each of the HC streams, and the PLE rows
void Engine::embed_into(int r, int k, int p0, const std::function<int(int)>& tok_at, const char*) {
    if (!h_.ple) h_.ple = halloc<float>((size_t) MAXT * N);
    std::vector<float> emb((size_t) k * N);
    embed_rows(p0, k, tok_at, emb.data(), h_.ple + (size_t) r * N);
    for (int j = 0; j < k; ++j)
        for (int c = 0; c < HC; ++c) std::memcpy(h_.R0 + (size_t) (r + j) * D + c * N, emb.data() + (size_t) j * N, N * sizeof(float));
}

void Engine::rows_upload(CardBuf& b, int T) {
    ck(cudaMemcpyAsync(b.R, h_.R0, (size_t) T * D * 4, cudaMemcpyHostToDevice, b.st), "R0");
    ck(cudaMemcpyAsync(b.ple_emb, h_.ple, (size_t) T * N * 4, cudaMemcpyHostToDevice, b.st), "ple");
}

void Engine::embed_rows(int p0, int T, const std::function<int(int)>& tok_at_cell, float* emb, float* ple) const {
    std::memset(ple, 0, (size_t) T * N * sizeof(float));
    static const int8_t kv4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const Tensor& pt = *m_.ple_table;
    const size_t prb = pt.row_bytes();
    // the PLE n-grams read an image cell as the model's image token
    const auto tok_at = [&](int q) -> int64_t {
        const int t = tok_at_cell(q);
        return t >= IMAGE_TOK ? m_.ple_image : t;
    };
    // FNX_PLE_RAM=0: the table is the file's mapping (model_loaded), so a row not in the page cache is a disk read; asked for all
    // at once the reads overlap (MADV_WILLNEED starts them and returns), where the loop below alone would wait for one at a time.
    // The same rows as the loop's, each page asked for once
    static const bool ple_file = env_int("FNX_PLE_RAM", 1) == 0;
    if (ple_file) {
        constexpr uintptr_t PG = 4096;
        std::vector<uintptr_t> pages;
        pages.reserve((size_t) T * (PLE_ORDERS - 1) * 8 * 2);
        for (int t = 0; t < T; ++t) {
            const int p = p0 + t;
            int64_t ctx[PLE_ORDERS];
            ctx[0] = tok_at(p);
            bool cut = false;
            for (int s = 1; s < PLE_ORDERS; ++s) {
                const int q = p - s;
                const int64_t tk = q >= 0 ? tok_at(q) : -1;
                cut = cut || tk < 0 || tk == m_.ple_eos;
                ctx[s] = cut ? m_.ple_eos : tk;
            }
            for (int n = 2; n <= PLE_ORDERS; ++n) {
                uint64_t mixed = (uint64_t) ctx[0] * m_.ple_mult[0];
                for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * m_.ple_mult[j];
                for (int gi = 0; gi < 8; ++gi) {
                    const int h = (n - 2) * 8 + gi;
                    const uintptr_t a = (uintptr_t) (pt.data + (mixed % m_.ple_vocab[h] + m_.ple_off[h]) * prb);
                    for (uintptr_t q = a / PG * PG; q < a + prb; q += PG) pages.push_back(q);
                }
            }
        }
        std::sort(pages.begin(), pages.end());
        for (size_t i = 0; i < pages.size();) {   // runs of whole pages, one call each
            size_t j = i + 1;
            while (j < pages.size() && pages[j] <= pages[j - 1] + PG) ++j;
            madvise((void*) pages[i], pages[j - 1] + PG - pages[i], MADV_WILLNEED);
            i = j;
        }
    }
    for (int t = 0; t < T; ++t) {
        const int p = p0 + t;
        embed_token(tok_at_cell(p), p, emb + (size_t) t * N);
        // PLE rows: n-grams of this token and the PLE_ORDERS - 1 before it (EOS / sequence start cut the window)
        int64_t ctx[PLE_ORDERS];
        ctx[0] = tok_at(p);
        bool cut = false;
        for (int s = 1; s < PLE_ORDERS; ++s) {
            const int q = p - s;
            const int64_t tk = q >= 0 ? tok_at(q) : -1;
            cut = cut || tk < 0 || tk == m_.ple_eos;
            ctx[s] = cut ? m_.ple_eos : tk;
        }
        for (int n = 2; n <= PLE_ORDERS; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * m_.ple_mult[0];
            for (int j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * m_.ple_mult[j];
            for (int gi = 0; gi < 8; ++gi) {
                const int h = (n - 2) * 8 + gi;
                const uint64_t r = mixed % m_.ple_vocab[h] + m_.ple_off[h];
                const uint8_t* src = pt.data + r * prb;
                for (int b = 0; b < PLE_DIM / 32; ++b) {
                    float* dst = &ple[(size_t) t * N + h * PLE_DIM + b * 32];
                    uint16_t dh;
                    if (pt.type == T_Q8_0) {
                        std::memcpy(&dh, src + b * 34, 2);
                        const float d = h2f(dh);
                        for (int j = 0; j < 32; ++j) dst[j] = d * (float) (int8_t) src[b * 34 + 2 + j];
                        continue;
                    }
                    std::memcpy(&dh, src + b * 18, 2);   // IQ4_NL
                    const float d = h2f(dh);
                    const uint8_t* qs = src + b * 18 + 2;
                    for (int j = 0; j < 16; ++j) {
                        dst[j] = d * kv4[qs[j] & 15];
                        dst[j + 16] = d * kv4[qs[j] >> 4];
                    }
                }
            }
        }
    }
}

// a sequence's device state on one card, in a fixed order: (pointer, bytes) pieces
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, bool mtp, int pos, bool kv, unsigned hist_half,
                                                   const std::vector<int>& pages) {
    std::vector<std::pair<char*, size_t>> v;
    const size_t ring_bytes = (size_t) qsa_ring_rows() * 128 * sizeof(uint16_t);
    for (int l = 0; l < NL; ++l) {
        if (is_attn(l)) {
            v.emplace_back((char*) b.qsa_ring[l], ring_bytes);
        } else {
            v.emplace_back((char*) b.S[l], (size_t) w.nv() * HS * HS * sizeof(float));
            v.emplace_back((char*) (b.chist[l] + (size_t) hist_half * 3 * w.conv_ch()), (size_t) 3 * w.conv_ch() * sizeof(float));
        }
    }
    v.emplace_back((char*) b.ple_hist, (size_t) 9 * D * sizeof(float));
    if (mtp) v.emplace_back((char*) b.mtp_qsa_ring, ring_bytes);
    // the pages holding positions 0 .. pos (every layer's K/V rows and pooled keys, the draft layer's too)
    if (kv)
        for (int i = 0; i <= pos / KVP_POS && i < (int) pages.size(); ++i)
            v.emplace_back((char*) (b.pool + (size_t) pages[(size_t) i] * b.pg.page), b.pg.page * sizeof(uint16_t));
    return v;
}


namespace {
__global__ void k_prof_stamp(unsigned long long* p) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *p = t;
}
constexpr int PROF_CAP = 4096;
struct ProfState {
    bool on = env_flag("FNX_PROF");
    unsigned long long* dev = nullptr;
    int dev0 = -1, capturing = 0;
    std::vector<int> kinds[MAXT + 1];
    double sum[MAXT + 1][PK_N] = {};
    long windows[MAXT + 1] = {};
    long since = 0;
};
ProfState& prof_state() {
    static ProfState p;
    return p;
}
}  // namespace

namespace {
std::mutex g_dalloc_mu;
std::map<std::pair<std::string, int>, std::pair<size_t, long>> g_dalloc;
}  // namespace
void dalloc_note(const char* file, int line, size_t bytes) {
    static const bool on = env_flag("FNX_PROF");
    if (!on) return;
    std::lock_guard<std::mutex> g(g_dalloc_mu);
    std::string f(file);
    const size_t sl = f.rfind('/');
    if (sl != std::string::npos) f = f.substr(sl + 1);
    auto& e = g_dalloc[{f, line}];
    e.first += bytes;
    ++e.second;
}
void dalloc_report() {
    std::lock_guard<std::mutex> g(g_dalloc_mu);
    std::vector<std::pair<size_t, std::string>> rows;
    size_t all = 0;
    for (const auto& kv : g_dalloc) {
        all += kv.second.first;
        char c[128];
        std::snprintf(c, sizeof c, "%s:%d x%ld", kv.first.first.c_str(), kv.first.second, kv.second.second);
        rows.emplace_back(kv.second.first, c);
    }
    std::sort(rows.rbegin(), rows.rend());
    std::fprintf(stderr, "fnx vram: dalloc on both cards %.2f GiB\n", all / 1073741824.0);
    for (size_t i = 0; i < rows.size() && i < 24; ++i) std::fprintf(stderr, "fnx vram:   %8.1f MiB %s\n", rows[i].first / 1048576.0, rows[i].second.c_str());
}

namespace {
constexpr int PF_PROF_CAP = 200000;
struct PfProf {
    unsigned long long* dev = nullptr;
    int dev0 = -1;
    std::vector<int> kinds;
    bool on = false;
};
PfProf& pf_prof() {
    static PfProf p;
    return p;
}
}  // namespace
void pf_prof_begin() {
    PfProf& p = pf_prof();
    p.on = env_flag("FNX_PROF");
    p.kinds.clear();
}
void pf_prof_mark(CardBuf& b, int kind) {
    PfProf& p = pf_prof();
    if (!p.on) return;
    if (p.dev0 < 0) p.dev0 = b.dev;
    if (b.dev != p.dev0 || (int) p.kinds.size() >= PF_PROF_CAP) return;
    if (!p.dev) ck(cudaMalloc(&p.dev, sizeof(unsigned long long) * PF_PROF_CAP), "pf prof");
    k_prof_stamp<<<1, 1, 0, b.st>>>(p.dev + p.kinds.size());
    p.kinds.push_back(kind);
}
void pf_prof_report(int rows) {
    PfProf& p = pf_prof();
    if (!p.on || p.kinds.size() < 2) return;
    std::vector<unsigned long long> t(p.kinds.size());
    ck(cudaSetDevice(p.dev0), "set");
    ck(cudaMemcpy(t.data(), p.dev, t.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "pf prof read");
    double sum[PF_N] = {}, all = 0;
    for (size_t i = 0; i + 1 < t.size(); ++i) {
        const double ms = (double) (t[i + 1] - t[i]) / 1e6;
        sum[p.kinds[i]] += ms;
        all += ms;
    }
    static const char* const names[PF_N] = {"hc", "qkvz", "ab", "gdn", "out", "qkv", "idx", "sel", "att", "wo", "sum", "router", "gu", "down", "shared", "draft", "end"};
    std::string line;
    char c[64];
    for (int q = 0; q < PF_N; ++q) {
        std::snprintf(c, sizeof c, " %s %.0f", names[q], sum[q]);
        line += c;
    }
    std::fprintf(stderr, "fnx prof branje: %d vrstic, %.0f ms, %.2f ms/vrstico:%s\n", rows, all, all / std::max(rows, 1), line.c_str());
    p.kinds.clear();
}

void prof_begin(int T) {
    ProfState& p = prof_state();
    if (!p.on) return;
    p.capturing = T;
    if (T > 0) p.kinds[T].clear();
}

void prof_mark(CardBuf& b, int kind) {
    ProfState& p = prof_state();
    if (!p.on || p.capturing <= 0) return;
    if (p.dev0 < 0) p.dev0 = b.dev;
    if (b.dev != p.dev0) return;
    if (!p.dev) ck(cudaMalloc(&p.dev, sizeof(unsigned long long) * PROF_CAP * (MAXT + 1)), "prof");
    std::vector<int>& k = p.kinds[p.capturing];
    if ((int) k.size() >= PROF_CAP) return;
    k_prof_stamp<<<1, 1, 0, b.st>>>(p.dev + (size_t) p.capturing * PROF_CAP + k.size());
    k.push_back(kind);
}

void prof_collect(int T) {
    ProfState& p = prof_state();
    if (!p.on || p.kinds[T].size() < 2) return;
    const std::vector<int>& k = p.kinds[T];
    std::vector<unsigned long long> t(k.size());
    ck(cudaSetDevice(p.dev0), "set");
    ck(cudaMemcpy(t.data(), p.dev + (size_t) T * PROF_CAP, t.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "prof read");
    for (size_t i = 0; i + 1 < k.size(); ++i) p.sum[T][k[i]] += (double) (t[i + 1] - t[i]) / 1e6;
    ++p.windows[T];
    if (++p.since < 100) return;
    p.since = 0;
    static const char* const names[PK_N] = {"hc", "qkvz+ab", "gdn", "out", "sum", "idx", "qkv", "att", "wo", "router", "cache", "gu", "down", "join", "head", "ple"};
    for (int w = 1; w <= MAXT; ++w) {
        if (!p.windows[w]) continue;
        double all = 0;
        std::string line;
        char c[64];
        for (int q = 0; q < PK_N; ++q) {
            const double ms = p.sum[w][q] / (double) p.windows[w];
            all += ms;
            std::snprintf(c, sizeof c, " %s %.2f", names[q], ms);
            line += c;
            p.sum[w][q] = 0;
        }
        std::fprintf(stderr, "fnx prof T=%d n=%ld total %.2f ms:%s\n", w, p.windows[w], all, line.c_str());
        p.windows[w] = 0;
    }
}

// out[t][r] = W[r] . y[t]: the split Q8_0 kernel, or mmvq for other types
void Engine::dense(const DW& W, const void* y, float* out, CardBuf& b, int T) {
    if (W.s.q) q8s_gemv(W.s, W.n_out, W.n_in, y, out, W.n_out, T, b.st);
    else mmvq(W, 0, b.dense[T], nullptr, 1, y, out, W.n_out, b.st);
}

// an attention layer's mixer after its hc read (mixed in b.xq): q/k/v, the window's K/V into kc/vc at ctl[0], the
// attention, the output projection into b.bo_a
void Engine::attention(int T, CardBuf& b, const CardW& w, const LayerW& L, uint16_t* kc, uint16_t* vc, const int* ctl,
                       uint16_t* qsa_ring, uint16_t* qsa_pooled, int call) {
    prof_mark(b, PK_IDX);
    // the indexer first (it reads mixed / its q8 copy, which the projections below leave as they are)
    if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
    qsa(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, L.idx_qn, L.idx_kn, qsa_ring, qsa_pooled, b.pg, b.qsa_q, b.qsa_scores, b.qsa_cells,
        b.qsa_ncells, max_ctx_ / 4, ctl, T, m_.eps, rope_of(b.mrope), b.att_ring.card, b.st);
    prof_mark(b, PK_QKV);
    if (L.wq.s.q && L.wk.s.q && L.wv.s.q && L.wq.n_in == L.wk.n_in && L.wq.n_in == L.wv.n_in &&
        L.wq.n_out + L.wk.n_out + L.wv.n_out >= 8192) {   // q, k and v in one launch (compiled for 8192 rows and more)
        const Q8S ws[3] = {L.wq.s, L.wk.s, L.wv.s};
        const int rows[3] = {L.wq.n_out, L.wk.n_out, L.wv.n_out};
        float* const outs[3] = {b.qg, b.k, b.v};
        q8s_gemv_segs(ws, rows, outs, rows, 3, L.wq.n_in, b.xq, T, b.st);
    } else {
        dense(L.wq, b.xq, b.qg, b, T);
        dense(L.wk, b.xq, b.k, b, T);
        dense(L.wv, b.xq, b.v, b, T);
    }
    prof_mark(b, PK_ATT);
    attn_prep_append(b.qg, b.k, b.v, L.q_norm, L.k_norm, kc, vc, b.pg, b.att_ring.card, T, ctl, m_.eps, rope_of(b.mrope), b.st);
    attn(b.qg, kc, vc, b.pg, b.att, b.att_ring, (const unsigned*) (ctl + 1), call, ctl, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells);
    prof_mark(b, PK_WO);
    quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
    dense(L.wo, b.xq, b.bo_a, b, T);
}

// the FFN half: the hc read with the attention half's write folded in, the shared expert (in the router's and the
// routed gate|up's launches on Q4_K layers, else on the second stream beside the router), the routing and the routed
// experts, the weighted sum across the cards into b.bo_f
void Engine::ffn_half(int T, int call, CardBuf& b, const CardW& w, const LayerW& L, float* R, const int* ctl) {
    const int P = T * TOPK, ffs = w.ff1 - w.ff0;
    prof_mark(b, PK_HC);
    const bool lat = L.lat_d.q != nullptr;   // edge0_patch.py: Edge0's latent MoE
    if (lat && !b.xh16) throw std::runtime_error("FNX_LATENT: the latent MoE runs on the P100 row kernels only");
    const uint16_t* gx = lat ? b.lxl16 : b.xh16;   // the gate|up kernels' rows: D's latent rows, or mixed as f16
    const int lh = w.ff0 / (w.ff1 - w.ff0);   // edge0_patch6.py: this card's half of the groups (FNX_LAT_SPLIT), as in model.cpp
    const auto lat_in = [&] {   // D: 8 latent rows a token (rows t * LAT_G + g), f16 for the row kernel
        if (g_lat_split) {   // edge0_patch6.py: this card's 4 groups' rows, the other 4 from the other card
            q8s_gemv(L.lat_d, LAT_HW, N, b.xq, b.lxl + (size_t) lh * LAT_HW, LAT_W, T, b.st);
            lat_xg(b.lxl, b.lxl16, b.lxg_in, b.lxg_next, T, lh, (const unsigned*) (ctl + 1), call, b.ring.err, b.st);
            return;
        }
        q8s_gemv(L.lat_d, LAT_W, N, b.xq, b.lxl, LAT_W, T, b.st);
        p100_f32_f16(b.lxl, (size_t) T * LAT_W, b.lxl16, b.st);
    };
    hc_read(R, b.bo_a, b.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], m_.eps, b.mixed, b.inj_f, b.hcs, b.ring.err, T, b.st, b.xq);
    // the shared expert (rows P + t of part): its gate|up in the router's launch and its swiglu and down in the routed
    // gate|up's (Q4_K experts), else on the second stream (a fork and a join cost ~5 us more than the launches' blocks)
    // FNX_SH_IN=0 (q4x_patch.py): Q4_K layers take the row path like the others (the P100 row kernel knows Q4_K now)
    static const bool sh_in_q4k = env_int("FNX_SH_IN", 1) != 0;
    const bool sh_in = sh_in_q4k && L.gu_e.type == T_Q4_K && L.sh_gu.s.q && L.sh_down.s.q && ffs % 32 == 0 && ffs <= SH_MAXF;
    // FNX_RR_ORDER (idea 60): 0 the branch launched before the router; 1 forked before it but launched after it (the router's
    // blocks get the SMs first); 2 forked after the router's end
    static const int rr_order = [] {
        const int o = (int) env_int("FNX_RR_ORDER", 0);
        if (o < 0 || o > 2) throw std::runtime_error("FNX_RR_ORDER: 0, 1 or 2");
        if (o) std::fprintf(stderr, "fnx: FNX_RR_ORDER %d: the shared expert's branch after the router%s\n", o, o == 1 ? "'s launch" : "");
        return o;
    }();
    if (!sh_in && rr_order < 2) ck(cudaEventRecord(b.fork, b.st), "fork");
    const auto shared_branch = [&] {
        ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
        q8s_gemv(L.sh_gu.s, 2 * ffs, N, b.xq, b.sh_gu, 2 * ffs, T, b.st2);
        swiglu_stacked_q8_1(b.sh_gu, b.sh_hq, ffs, T, b.st2);
        q8s_gemv(L.sh_down.s, N, ffs, b.sh_hq, b.part + (size_t) P * N, N, T, b.st2);
        // FNX_PREFETCH: the next layer's router on this FFN input and its top 10 a row, beside the shared expert
        if (const int nl = (int) (&L - w.L.data()) + 1; prefetch_ && T >= prefetch_min_t_ && nl > 0 && nl < NL) {
            gemv_bf16(w.L[(size_t) nl].router, b.mixed, b.pf_logits, NEXP, N, T, b.st2);
            expert_mask_logits(b.pf_logits, T, NEXP, b.ekeep ? b.ekeep + (size_t) nl * (NEXP / 32) : nullptr, b.st2);   // FNX_EXPERT_MASK
            probe_guess(b.pf_logits, T, b.pf_guess, b.st2);
            if (b.memo_guess && noc_mode().memo == 1) memo_merge(b.memo_guess, nl, T, b.pf_guess, b.st2);   // memo_patch.py: lookup rows from the memo
        }
        ck(cudaEventRecord(b.join, b.st2), "join");
    };
    if (!sh_in && rr_order == 0) shared_branch();
    const int lyr = (int) (&L - w.L.data());
    // experts in RAM: the router's last block plans this window's slots (the prefetch probe scores the picks against
    // the slots as they were, so with it the plan runs after, in moe_cache)
    const bool live = cache_live() && lyr >= 0 && lyr < NL;
    const bool plan_in_router = live && !b.probe_cnt;
    // FNX_OVER_MIN_T: a two-token window has few misses and loses more than it gains; a wide one (an edit) has long copies
    static const int over_min_t = (int) env_int("FNX_OVER_MIN_T", 3);
    // the copy engine (FNX_P100_DMA) takes no SM from the rows: overlap at every window size
    const bool dma = b.dma != nullptr;
    const bool over = live && plan_in_router && b.st3 && b.xh16 && !sh_in && (T >= over_min_t || dma);
    // FNX_FOLD (IDEJE 39): a window's small launches in their neighbours, a launch less a layer each. 1: the plan's jobs marked and its
    // list sent to the copy engine in the router's last block; 2: the rows to f16 in the router's blocks; 4: the guessed experts' plan
    // and publish in one launch
    static const int fold = [] {
        const int f = (int) env_int("FNX_FOLD", 0);
        if (f < 0 || f > 7) throw std::runtime_error("FNX_FOLD: a sum of 1 (split and publish in the router), 2 (f16 rows in the router), 4 (the guesses' plan and publish)");
        if (f) std::fprintf(stderr, "fnx: FNX_FOLD %d: a window's small launches folded into their neighbours\n", f);
        return f;
    }();
    RouteTail rt;
    if (b.ekeep && lyr >= 0 && lyr < NL) rt.keep = b.ekeep + (size_t) lyr * (NEXP / 32);   // FNX_EXPERT_MASK (maska_patch.py)
    if (over && dma && (fold & 1)) rt.box = p100_dma_box(b.dma), rt.dev = p100_dma_dev(b.dma), rt.gu_bytes = L.gu_e_stride, rt.dn_bytes = L.down_e_stride;
    if (over && (fold & 2) && !lat) rt.xh = b.xh16;
    prof_mark(b, PK_ROUTER);
    router_route(L.router, b.mixed, b.logits, T, b.topi, b.topw, b.jobs_gu, b.jobs_dn, b.njobs, b.st, plan_in_router ? &w.ec : nullptr,
                 lyr, b.cache_list, sh_in ? &L.sh_gu.s : nullptr, 2 * ffs, b.xq, b.sh_gu, &rt);
    if (lat) lat_rows(b.jobs_gu, b.njobs, L.lat_lab, b.st);   // edge0_patch.py: gate|up reads row t * LAT_G + label
    if (!sh_in && rr_order == 2) ck(cudaEventRecord(b.fork, b.st), "fork");
    if (!sh_in && rr_order > 0) shared_branch();
    if (const int l = lyr; l >= 0 && l < NL) {
        if (b.route_cnt) route_count(b.topi, P, b.route_cnt + (size_t) l * NEXP, b.st);
        if (b.route_trace) fnx::route_trace(b.topi, P, l, b.route_trace, b.route_trace_len, b.route_trace_cap, b.st);
        if (memo_ && &b == cb_[0]) memo_put_win(b.topi, T, ctl, l, memo_, memo_cap_, b.st);   // memo_patch.py
        if (b.probe_cnt) {
            if (l > 0)
                probe_score(b.topi, b.probe_guess + (size_t) l * MAXT * TOPK, T, w.ec.n_slots ? w.ec.slot_of + (size_t) l * NEXP : nullptr,
                            b.probe_cnt, b.st);
            if (l + 1 < NL) {
                gemv_bf16(w.L[(size_t) l + 1].router, b.mixed, b.probe_logits, NEXP, N, T, b.st);
                expert_mask_logits(b.probe_logits, T, NEXP, b.ekeep ? b.ekeep + (size_t) (l + 1) * (NEXP / 32) : nullptr, b.st);   // FNX_EXPERT_MASK
                probe_guess(b.probe_logits, T, b.probe_guess + (size_t) (l + 1) * MAXT * TOPK, b.st);
            }
        }
        prof_mark(b, PK_CACHE);
        // the planned experts copied into their slots before the MoE kernels read the layer's tables
        if (live && over && rt.box) {   // FNX_FOLD 1: the router's last block did both
        } else if (live && over) {
            p100_jobs_split(b.jobs_gu, b.jobs_dn, b.njobs, l, b.cache_list, b.st);
            if (dma) p100_dma_publish(b.dma, l, w.ec, b.cache_list, L.gu_e_stride, L.down_e_stride, b.st);
            else {
                ck(cudaEventRecord(b.fork2, b.st), "fork2");
                ck(cudaStreamWaitEvent(b.st3, b.fork2, 0), "fork2 wait");
                p100_cache_copy(l, L.gu_e_stride, L.down_e_stride, w.ec, b.cache_list, b.st3);
                ck(cudaEventRecord(b.join2, b.st3), "join2");
            }
        } else if (live) moe_cache(l, L.gu_e_stride, L.down_e_stride, w.ec, b.cache_list, b.st, plan_in_router ? nullptr : b.jobs_gu, b.njobs);
    }
    prof_mark(b, PK_GU);
    if (over) {
        // first the experts that stayed in VRAM, while the copy runs; then wait for it, and the copied ones
        if (lat) lat_in();
        else if (!rt.xh) p100_f32_f16(b.mixed, (size_t) T * N, b.xh16, b.st);
        // FNX_NOC_FILE split (IDEJE 31): the copier sends gate|up parts first, so gate|up waits for those only and down for the rest
        const bool split = dma && noc_mode().split;
        const bool split2 = dma && noc_mode().split == 2;   // perexp_patch.py: phase 2's blocks wait for their own expert's part
        const bool pf_wait = prefetch_ && dma && T >= prefetch_min_t_;   // FNX_PREFETCH: this layer's guessed experts, copied since the layer before
        if (pf_wait) (split ? p100_dma_wait2_gu : p100_dma_wait2)(b.dma, b.st);
        p100_moe_rows_phase(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, T, gx, b.gu, 1, b.st);
        prof_mark(b, PK_DOWN);
        if (lat) lat_swiglu_h16(b.gu, ffs, P, b.h16, b.topi, L.lat_esc, nullptr, b.st);
        else swiglu_h16(b.gu, ffs, P, b.h16, b.st);
        if (pf_wait && split) p100_dma_wait2(b.dma, b.st);
        p100_moe_rows_phase(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, T, b.h16, b.part, 1, b.st);
        prof_mark(b, PK_CACHE);
        if (split2) {
        } else if (dma) (split ? p100_dma_wait_gu : p100_dma_wait)(b.dma, b.st);
        else ck(cudaStreamWaitEvent(b.st, b.join2, 0), "join2 wait");
        prof_mark(b, PK_GU);
        if (split2)
            p100_moe_rows_phase_w(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, T, gx, b.gu, 2, b.st, p100_dma_done(b.dma, 0),
                                  p100_dma_want_ptr(b.dma), b.cache_list, lyr, p100_dma_err_ptr(b.dma));
        else p100_moe_rows_phase(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, T, gx, b.gu, 2, b.st);
        prof_mark(b, PK_DOWN);
        if (lat) lat_swiglu_h16(b.gu, ffs, P, b.h16, b.topi, L.lat_esc, b.cache_list, b.st);
        else p100_swiglu_after(b.gu, ffs, P, b.h16, b.cache_list, b.st);
        if (split && !split2) {
            prof_mark(b, PK_CACHE);
            p100_dma_wait(b.dma, b.st);
            prof_mark(b, PK_DOWN);
        }
        if (split2)
            p100_moe_rows_phase_w(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, T, b.h16, b.part, 2, b.st, p100_dma_done(b.dma, 1),
                                  p100_dma_want_ptr(b.dma), b.cache_list, lyr, p100_dma_err_ptr(b.dma));
        else p100_moe_rows_phase(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, T, b.h16, b.part, 2, b.st);
    } else if (sh_in) moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st, &L.sh_down.s, ffs, b.sh_gu, b.part + (size_t) P * N);
    else if (b.xh16) {
        if (lat) lat_in();
        else p100_f32_f16(b.mixed, (size_t) T * N, b.xh16, b.st);
        p100_moe_rows(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, T, gx, b.gu, b.st);
    } else moe_gate_up(L.gu_e, L.gu_e_stride, b.jobs_gu, b.njobs, P, b.xq, b.gu, b.st);
    if (!over) prof_mark(b, PK_DOWN);
    if (over) {
    } else if (b.xh16 && !sh_in) {
        if (lat) lat_swiglu_h16(b.gu, ffs, P, b.h16, b.topi, L.lat_esc, nullptr, b.st);
        else swiglu_h16(b.gu, ffs, P, b.h16, b.st);
        p100_moe_rows(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, T, b.h16, b.part, b.st);
    } else moe_down(L.down_e, L.down_e_stride, b.jobs_dn, b.njobs, P, b.gu, b.part, b.st);
    if (lat) {   // edge0_patch.py: the routed rows by group into z, then U (this card's ff part: the ring sums the cards)
        lat_z(b.part, b.topi, b.topw, L.lat_lab, L.lat_esc, T, b.lz, b.st);
        if (g_lat_split) {   // edge0_patch6.py: z of this card's groups, both cards' parts (into lxl, free since D's exchange), then its U half
            lat_zs(b.lz, b.lxl, b.lzs_in, b.lzs_next, T, lh, (const unsigned*) (ctl + 1), call, b.ring.err, b.st);
            quantize_q8_1(b.lxl, b.lzq, LAT_HW, T, b.st);
            q8s_gemv(L.lat_u, N, LAT_HW, b.lzq, b.latu, N, T, b.st);
        } else {
            quantize_q8_1(b.lz, b.lzq, LAT_W, T, b.st);
            q8s_gemv(L.lat_u, N, LAT_W, b.lzq, b.latu, N, T, b.st);
        }
    }
    prof_mark(b, PK_JOIN);
    if (!sh_in) ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
    // FNX_PREFETCH: this layer's experts done with, the next layer's guessed ones take slots and the copier copies them while its
    // attention half runs (a layer that takes the same path: its window waits for them before its first phase)
    if (prefetch_ && T >= prefetch_min_t_ && over && dma && !sh_in && lyr + 1 < NL && (!sh_in_q4k || w.L[(size_t) lyr + 1].gu_e.type != T_Q4_K)) {
        const LayerW& Ln = w.L[(size_t) lyr + 1];
        prof_mark(b, PK_CACHE);
        if (fold & 4) p100_pf_plan_publish(b.dma, b.pf_guess, T, lyr + 1, w.ec, b.pf_want, b.pf_list, Ln.gu_e_stride, Ln.down_e_stride, b.st);
        else {
            p100_pf_plan(b.pf_guess, T, lyr + 1, w.ec, b.pf_want, b.pf_list, b.st);
            p100_dma_publish2(b.dma, lyr + 1, w.ec, b.pf_list, Ln.gu_e_stride, Ln.down_e_stride, b.st);
        }
        if (b.memo_probe) ck(cudaMemcpyAsync(b.memo_probe, b.pf_guess, (size_t) MAXT * TOPK * sizeof(int), cudaMemcpyDeviceToDevice, b.st), "memo probe");
    }
    if (memo_stat_ && &b == cb_[0] && lyr >= 0 && lyr < NL && cache_live() && b.memo_guess)   // memo_patch.py: the counts of fnx: noc
        memo_stat(b.topi, b.memo_guess, b.memo_probe, b.cache_list, lyr, T,
                  prefetch_ && T >= prefetch_min_t_ && over && dma && !sh_in && lyr + 1 < NL, b.pf_list, memo_stat_, b.st);
    prof_mark(b, PK_SUM);
    RingMoe m;
    m.part = b.part;
    m.topw = b.topw;
    m.logits = b.logits;
    if (lat) m.lat = b.latu;   // edge0_patch.py
    ring_sum(b.ring, b.bo_f, T, (const unsigned*) (ctl + 1), call, b.st, m);
}

// every card's work for one window of T rows, in stream order (captured into a graph unless dumping)
void Engine::enqueue(int T, bool dumps) {
    prof_begin(dumps ? 0 : T);
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    auto dd = [&](const std::string& name, int layer, const float* p, int n) {
        if (dumps) dump_dev(name, layer, p, T, n);
    };
    for (int l = 0; l < NL; ++l) {
        const bool pending = l > 0;
        // ---- attention half
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = pending;
            // memo_patch.py (memo 1): layer 0's guesses (the lookup rows' memo, nothing else) planned and copied while its mixer runs
            if (l == 0 && b.memo_guess && noc_mode().memo == 1 && prefetch_ && b.dma && T >= prefetch_min_t_ && cache_live() && !dumps) {
                p100_pf_plan(b.memo_guess, T, 0, w.ec, b.pf_want, b.pf_list, b.st);
                p100_dma_publish2(b.dma, 0, w.ec, b.pf_list, L.gu_e_stride, L.down_e_stride, b.st);
            }
            if (l == PLE_LAYER) {
                prof_mark(b, PK_PLE);
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                ple_post(b.R, b.ple_key, b.ple_val, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv, w.ple_conv,
                         b.ple_hist, b.ple_hsnap, T, eps, b.st, b.ple_key + (size_t) MAXT * D);
            }
            prof_mark(b, PK_HC);
            if (apply && cvec_.covers(l - 1)) {   // FNX_CVEC: layer l - 1's FFN write on its own, then its vector
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                cvec_apply(b.R, b.cvec + (size_t) (l - 1) * N, cvec_.s[(size_t) (l - 1)], cvec_.add, T, b.st);
                apply = false;
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                prof_mark(b, PK_QKVZ);
                const int nv = w.nv(), ch = w.conv_ch();
                // ab beside qkvz in one launch (a fork and a join: ~5 us more; compiled for 4096..8191 rows)
                if (L.qkvz.s.q && 2 * nv <= 160 && L.qkvz.n_out >= 4096 && L.qkvz.n_out < 8192) {
                    q8s_gemv_f32(L.qkvz.s, L.qkvz.n_out, L.qkvz.n_in, b.xq, b.qkvz, L.ab, b.mixed, b.ab, 2 * nv, T, b.st);
                } else {
                    ck(cudaEventRecord(b.fork, b.st), "fork");
                    ck(cudaStreamWaitEvent(b.st2, b.fork, 0), "fork wait");
                    gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st2);
                    ck(cudaEventRecord(b.join, b.st2), "join");
                    dense(L.qkvz, b.xq, b.qkvz, b, T);
                    ck(cudaStreamWaitEvent(b.st, b.join, 0), "join wait");
                }
                prof_mark(b, PK_GDN);
                gdn_core(b.qkvz, b.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], b.xq, b.ctl, b.gdn_scratch, b.snap[l],
                         b.hsnap[l], w.nk, nv, w.vmap, T, eps, b.st);
                prof_mark(b, PK_OUT);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                attention(T, b, w, L, b.kc[l], b.vc[l], b.ctl, b.qsa_ring[l], b.qsa_pooled[l], 128 + l);
            }
            prof_mark(b, PK_SUM);
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        dd("hc_mixed", l, cb_[0]->mixed, N);
        dd("mixer_out", l, cb_[0]->bo_a, N);
        // ---- FFN half
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
        dd("ffn_mixed", l, cb_[0]->mixed, N);
        dd("ffn_out", l, cb_[0]->bo_f, N);
        dd("R_pending", l, cb_[0]->R, D);
        dd("inj_ffn", l, cb_[0]->inj_f, HC);
    }
    // ---- head: the final mixer (no injection) with the last write folded in, then this card's vocabulary
    each([&](int c, CardBuf& b, const CardW& w) {
        prof_mark(b, PK_HEAD);
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
        prof_mark(b, PK_HEAD);
    });
    prof_begin(0);
    dd("result_norm", -1, cb_[0]->mixed, N);
}

// enqueue()'s layers for a window of one row per lane (row r: lanes[r]): the shared work over all rows, the
// sequence's own work (the PLE block, the recurrent core, QSA, the rotation, the K/V append, the attention partials)
// per row with its lane's state and ctl
void Engine::enqueue_batch(const std::vector<int>& lanes) {
    const int T = (int) lanes.size();
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int r = 0; r < T; ++r)   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    ple_post(b.R + (size_t) r * D, b.ple_key + (size_t) r * D, b.ple_val + (size_t) r * N, w.ple_norm_key, w.ple_norm_query,
                             w.ple_norm_conv, w.ple_conv, b.lanes[(size_t) lanes[r]].ple_hist, b.ple_hsnap, 1, eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
            }
            if (apply && cvec_.covers(l - 1)) {   // FNX_CVEC: layer l - 1's FFN write on its own, then its vector
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                cvec_apply(b.R, b.cvec + (size_t) (l - 1) * N, cvec_.s[(size_t) (l - 1)], cvec_.add, T, b.st);
                apply = false;
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv();
                gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st);
                dense(L.qkvz, b.xq, b.qkvz, b, T);
                float** rec = b.lane_gdn[(size_t) T];
                const GdnLanes gl{rec + (size_t) l * MAXT, rec + (size_t) (NL + l) * MAXT, b.lane_rows[(size_t) T]};
                gdn_core_lanes(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, T, eps, b.st);
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense(L.wq, b.xq, b.qg, b, T);
                dense(L.wk, b.xq, b.k, b, T);
                dense(L.wv, b.xq, b.v, b, T);
                const int mp = max_ctx_ / 4;
                for (int r = 0; r < T; ++r) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[r]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    qsa_index(b.qsa_y + (size_t) r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + (size_t) r * 512,
                              LD.ctl, 1, eps, rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + (size_t) r * NQH * 2 * HD, b.k + (size_t) r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, LD.ctl, 1, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + (size_t) r * NKVH * HD, b.v + (size_t) r * NKVH * HD, b.kc[l], b.vc[l], pg, b.att_ring.card, 1, LD.ctl,
                              b.st);
                }
                // every row's selection and attention partials at once, each on its lane's position and pages
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st);
                quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
}

// enqueue_batch()'s layers for a window of several rows a lane (verify_rows): the shared work over all rows as there;
// each lane's own work over its rows as a window of its own does it (the rows chained through its recurrent state, the
// state and histories after each row kept for commit_rows), and every row's selection and attention partials at once,
// each at its own position (row_pos) on its lane's pages
void Engine::enqueue_rows(const std::vector<int>& lanes, const std::vector<int>& counts) {
    const RowSegs sg = row_segs(counts);
    const int T = sg.T, nl = (int) lanes.size();
    const float eps = m_.eps;
    int call = 0;
    auto each = [&](auto&& f) {
        for (int c = 0; c < NCARD; ++c) {
            ck(cudaSetDevice(cb_[c]->dev), "set");
            f(c, *cb_[c], m_.cards[c]);
        }
    };
    for (int l = 0; l < NL; ++l) {
        each([&](int c, CardBuf& b, const CardW& w) {
            const LayerW& L = w.L[l];
            bool apply = l > 0;
            if (l == PLE_LAYER) {
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                apply = false;
                quantize_q8_1(b.ple_emb, b.xq, N, T, b.st);
                dense(w.ple_key, b.xq, b.ple_key, b, T);
                dense(w.ple_value, b.xq, b.ple_val, b, T);
                for (int i = 0; i < nl; ++i) {   // gated and normed rows past every key row (ple_key holds 3 MAXT rows)
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    ple_post(b.R + r * D, b.ple_key + r * D, b.ple_val + r * N, w.ple_norm_key, w.ple_norm_query, w.ple_norm_conv,
                             w.ple_conv, b.lanes[(size_t) lanes[i]].ple_hist, b.ple_hsnap + r * 9 * D, counts[(size_t) i], eps, b.st,
                             b.ple_key + (size_t) MAXT * D);
                }
            }
            if (apply && cvec_.covers(l - 1)) {   // FNX_CVEC: layer l - 1's FFN write on its own, then its vector
                hc_write(b.R, b.bo_f, b.inj_f, T, b.st);
                cvec_apply(b.R, b.cvec + (size_t) (l - 1) * N, cvec_.s[(size_t) (l - 1)], cvec_.add, T, b.st);
                apply = false;
            }
            hc_read(b.R, apply ? b.bo_f : nullptr, b.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed,
                    b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
            if (!is_attn(l)) {
                const int nv = w.nv();
                gemv_f32(L.ab, b.mixed, b.ab, 2 * nv, N, T, b.st);
                dense(L.qkvz, b.xq, b.qkvz, b, T);
                float** rec = b.lane_gdn[(size_t) T];
                for (const RowGroup& g : sg.groups) {
                    const GdnLanes gl{rec + (size_t) l * MAXT + g.s0, rec + (size_t) (NL + l) * MAXT + g.s0, b.seg_ctl[(size_t) T] + g.s0,
                                      b.seg_row0[(size_t) T] + g.s0, b.snap[l], b.hsnap[l]};
                    gdn_core_segs(b.qkvz, b.ab, L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.xq, b.gdn_scratch, w.nk, nv, w.vmap, gl, g.n,
                                  g.count, eps, b.st);
                }
                dense(L.ssm_out, b.xq, b.bo_a, b, T);
            } else {
                if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
                else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
                dense(L.wq, b.xq, b.qg, b, T);
                dense(L.wk, b.xq, b.k, b, T);
                dense(L.wv, b.xq, b.v, b, T);
                const int mp = max_ctx_ / 4;
                for (int i = 0; i < nl; ++i) {
                    const LaneDev& LD = b.lanes[(size_t) lanes[i]];
                    KvPages pg = b.pg;   // the card's page size and rows, the lane's table
                    pg.pt = LD.pt;
                    const size_t r = (size_t) sg.r0[(size_t) i];
                    const int k = counts[(size_t) i];
                    qsa_index(b.qsa_y + r * qsa_rows(), L.idx_qn, L.idx_kn, LD.ring[l], b.qsa_pooled[l], pg, b.qsa_q + r * 512, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    attn_prep(b.qg + r * NQH * 2 * HD, b.k + r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, LD.ctl, k, eps,
                              rope_of(LD.mrope), b.st);
                    kv_append(b.k + r * NKVH * HD, b.v + r * NKVH * HD, b.kc[l], b.vc[l], pg, b.att_ring.card, k, LD.ctl, b.st);
                }
                const LaneRows lr{b.lane_rows[(size_t) T], b.lane_rows[(size_t) T] + MAXT};
                qsa_select_lanes(b.qsa_q, b.qsa_pooled[l], b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
                attn_part_lanes(b.qg, b.kc[l], b.vc[l], b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
                attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.ctl + 1), 128 + l, T, b.att_part, b.st);
                quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
                dense(L.wo, b.xq, b.bo_a, b, T);
            }
            ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.ctl + 1), call, b.st);
        });
        ++call;
        each([&](int, CardBuf& b, const CardW& w) { ffn_half(T, call, b, w, w.L[l], b.R, b.ctl); });
        ++call;
    }
    each([&](int, CardBuf& b, const CardW& w) {
        hc_read(b.R, b.bo_f, b.inj_f, w.head_norm, w.head_down, w.head_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(w.out, b.xq, b.head_logits, b, T);
        argmax(b.head_logits, w.v1 - w.v0, T, w.v0, b.best_v, b.best_i, b.am_keys, b.ring.err, b.st);
    });
}

}  // namespace fnx

namespace fnx {

void Engine::mtp_attach(const Mtp& mtp) {
    if (mtp.cards.size() != (size_t) NCARD) throw std::runtime_error("mtp_attach: the draft layer is not on every card");
    mtp_ = &mtp;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp.cards[c];
        ck(cudaSetDevice(b.dev), "set");
        b.mtp_emb = dalloc<float>((size_t) MAXT * N);
        b.mtp_en = dalloc<float>((size_t) MAXT * N);
        b.mtp_e2 = dalloc<float>((size_t) MAXT * N);
        if (c == 0 && env_flag("FNX_MTP_NORM_STREAMS")) std::fprintf(stderr, "sfnx: the draft layer's source normalised per stream\n");
        b.mtp_src = dalloc<float>((size_t) MAXT * D);
        b.mtp_hn = dalloc<float>((size_t) MAXT * D);
        b.mtp_h2 = dalloc<float>((size_t) MAXT * D);
        b.mtp_R = dalloc<float>((size_t) MAXT * D);
        b.mtp_kc = b.pool + (size_t) (KVP_SLOTS - 1) * kvp_slot(c);   // the draft layer: the pool's last slot
        b.mtp_vc = b.mtp_kc + kvp_rows_halves(c, kv_q8());
        b.lanes[0].mtp_ring = dalloc<uint16_t>((size_t) qsa_ring_rows() * 128);   // lanes after 0 carry theirs in their blocks
        b.mtp_qsa_ring = b.lanes[(size_t) cur_].mtp_ring;
        b.mtp_qsa_pooled = b.mtp_kc + 2 * kvp_rows_halves(c, kv_q8());
        b.mtp_logits = dalloc<float>((size_t) MAXT * std::max(1, mc.dv1 - mc.dv0));
        b.mtp_bv = dalloc<float>(MAXT);
        b.mtp_bi = dalloc<int>(MAXT);
        b.mtp_ctl = dalloc<int>(2);
        b.mtp_lctl = dalloc<int>(2 * MAXT);
        b.mtp_row_pos = dalloc<int>(MAXT);
        b.mtp_lane_rows = dalloc<const int*>((size_t) LANE_ROW_COLS * MAXT);
        b.mtp_pen_l = dalloc<int>((size_t) MAXT * (1 + 2 * PEN_CAP));
        b.mtp_pen_par_l = dalloc<float>((size_t) MAXT * 3);
        b.mtp_tk_v_l = dalloc<float>((size_t) MAXT * TOPK_N);
        b.mtp_tk_i_l = dalloc<int>((size_t) MAXT * TOPK_N);
        b.mtp_tk_err_l = dalloc<unsigned>(MAXT);
        b.mtp_pen = dalloc<int>(1 + 2 * PEN_CAP);
        b.mtp_pen_par = dalloc<float>(3);
        b.mtp_tk_v = dalloc<float>(TOPK_N);
        b.mtp_tk_i = dalloc<int>(TOPK_N);
        b.mtp_tk_err = dalloc<unsigned>(1);
        b.mtp_zero = dalloc<int>(1);
        ck(cudaMemset(b.mtp_zero, 0, 4), "mtp zero");
    }
    if (!h_.mtp_emb) {
        h_.mtp_emb = halloc<float>((size_t) MAXT * N);
        h_.mtp_ctl = halloc<int>(2);
        h_.mtp_lctl = halloc<int>(2 * MAXT);
        h_.mtp_row_pos = halloc<int>(MAXT);
        h_.mtp_pen_l = halloc<int>((size_t) MAXT * NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par_l = halloc<float>((size_t) MAXT * 3);
        h_.mtp_tk_v_l = halloc<float>((size_t) MAXT * NCARD * TOPK_N);
        h_.mtp_tk_i_l = halloc<int>((size_t) MAXT * NCARD * TOPK_N);
        h_.mtp_tk_err_l = halloc<unsigned>((size_t) MAXT * NCARD);
        h_.mtp_pen = halloc<int>((size_t) NCARD * (1 + 2 * PEN_CAP));
        h_.mtp_pen_par = halloc<float>(3);
        h_.mtp_tk_v = halloc<float>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_i = halloc<int>((size_t) NCARD * TOPK_N);
        h_.mtp_tk_err = halloc<unsigned>(NCARD);
    }
    draft_index_.assign((size_t) m_.vocab, -1);
    for (size_t i = 0; i < mtp.vocab.size(); ++i) draft_index_[(size_t) mtp.vocab[i]] = (int) i;
    if (!h_.bv) {
        h_.bv = halloc<float>(NCARD * MAXT);
        h_.bi = halloc<int>(NCARD * MAXT);
        h_.err = halloc<unsigned>(NCARD);
    }
}

// one draft-layer window of T rows from b.mtp_src (rows of D) and b.mtp_emb (the tokens' embeddings), cells at
// mtp_ctl[0]: R = fc_hidden(rms(src) w) per stream + fc_embedding(rms(emb) w), then the layer as a main attention
// layer, then the final mixer and this card's draft head rows
void Engine::mtp_enqueue(int T) {
    const float eps = m_.eps;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        ck(cudaSetDevice(b.dev), "set");
        rms_norm_rows(b.mtp_emb, mc.norm_emb, b.mtp_en, T, N, eps, b.st);
        quantize_q8_1(b.mtp_en, b.xq, N, T, b.st);
        dense(mc.fc_emb, b.xq, b.mtp_e2, b, T);
        static const bool norm_streams = env_flag("FNX_MTP_NORM_STREAMS");   // mtp_norm_patch.py: each stream on its own
        if (norm_streams) rms_norm_streams(b.mtp_src, mc.norm_hid, b.mtp_hn, T, HC, N, eps, b.st);
        else rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);
        quantize_q8_1(b.mtp_hn, b.xq, N, HC * T, b.st);
        for (int r0 = 0; r0 < HC * T; r0 += MAXT)   // every stream of every row through fc_hidden
            q8s_gemv(mc.fc_hid.s, N, N, (const uint8_t*) b.xq + (size_t) r0 * q8_1_bytes(N), b.mtp_h2 + (size_t) r0 * N, N,
                     std::min(MAXT, HC * T - r0), b.st);
        add_streams(b.mtp_h2, b.mtp_e2, b.mtp_R, T, b.st);
        hc_read(b.mtp_R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T,
                b.st, b.xq);
        attention(T, b, w, L, b.mtp_kc, b.mtp_vc, b.mtp_ctl, b.mtp_qsa_ring, b.mtp_qsa_pooled, 128);
        ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.mtp_ctl + 1), 0, b.st);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn_half(T, 1, b, m_.cards[c], mtp_->cards[c].L, b.mtp_R, b.mtp_ctl);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        ck(cudaSetDevice(b.dev), "set");
        hc_read(b.mtp_R, b.bo_f, b.inj_f, mc.mix_norm, mc.mix_down, mc.mix_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(mc.head, b.xq, b.mtp_logits, b, T);
        penalize_row(b.mtp_logits + (size_t) (T - 1) * (mc.dv1 - mc.dv0), mc.dv1 - mc.dv0, b.mtp_pen, b.mtp_pen_par, b.st);
        argmax(b.mtp_logits, mc.dv1 - mc.dv0, T, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.ring.err, b.st);
    }
}


// The draft layer for several lanes in ONE window (Engine::draft_lanes): mtp_enqueue's layer, with the attention of enqueue_rows' batched windows for the draft
// layer's own K/V slot (each lane's rows at its cells on its pages). Greedy: no penalties, argmax of every row.
void Engine::mtp_enqueue_lanes(const std::vector<DraftReq>& reqs, int T) {
    const float eps = m_.eps;
    const int nl = (int) reqs.size(), mp = max_ctx_ / 4;
    std::vector<int> off((size_t) nl);
    for (int i = 0, o = 0; i < nl; ++i) off[(size_t) i] = o, o += reqs[(size_t) i].T;
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const CardW& w = m_.cards[c];
        const MtpCard& mc = mtp_->cards[c];
        const LayerW& L = mc.L;
        ck(cudaSetDevice(b.dev), "set");
        rms_norm_rows(b.mtp_emb, mc.norm_emb, b.mtp_en, T, N, eps, b.st);
        quantize_q8_1(b.mtp_en, b.xq, N, T, b.st);
        dense(mc.fc_emb, b.xq, b.mtp_e2, b, T);
        static const bool norm_streams = env_flag("FNX_MTP_NORM_STREAMS");   // mtp_norm_patch.py: each stream on its own
        if (norm_streams) rms_norm_streams(b.mtp_src, mc.norm_hid, b.mtp_hn, T, HC, N, eps, b.st);
        else rms_norm_rows(b.mtp_src, mc.norm_hid, b.mtp_hn, T, D, eps, b.st);
        quantize_q8_1(b.mtp_hn, b.xq, N, HC * T, b.st);
        for (int r0 = 0; r0 < HC * T; r0 += MAXT)
            q8s_gemv(mc.fc_hid.s, N, N, (const uint8_t*) b.xq + (size_t) r0 * q8_1_bytes(N), b.mtp_h2 + (size_t) r0 * N, N, std::min(MAXT, HC * T - r0), b.st);
        add_streams(b.mtp_h2, b.mtp_e2, b.mtp_R, T, b.st);
        hc_read(b.mtp_R, nullptr, nullptr, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        // the attention as a batched window does it, on the draft layer's slot of the pool
        if (!L.idx_wq) dense(L.idx_qk, b.xq, b.qsa_y, b, T);
        else qsa_proj(L.idx_wq, L.idx_wk, b.mixed, b.qsa_y, T, b.st);
        dense(L.wq, b.xq, b.qg, b, T);
        dense(L.wk, b.xq, b.k, b, T);
        dense(L.wv, b.xq, b.v, b, T);
        for (int i = 0; i < nl; ++i) {
            const LaneDev& LD = b.lanes[(size_t) reqs[(size_t) i].lane];
            KvPages pg = b.pg;
            pg.pt = LD.pt;
            const size_t r = (size_t) off[(size_t) i];
            const int k = reqs[(size_t) i].T;
            const int* lc = b.mtp_lctl + 2 * i;
            qsa_index(b.qsa_y + r * qsa_rows(), L.idx_qn, L.idx_kn, LD.mtp_ring, b.mtp_qsa_pooled, pg, b.qsa_q + r * 512, lc, k, eps, rope_of(LD.mrope), b.st);
            attn_prep(b.qg + r * NQH * 2 * HD, b.k + r * NKVH * HD, L.q_norm, L.k_norm, NQH, NKVH, lc, k, eps, rope_of(LD.mrope), b.st);
            kv_append(b.k + r * NKVH * HD, b.v + r * NKVH * HD, b.mtp_kc, b.mtp_vc, pg, b.att_ring.card, k, lc, b.st);
        }
        const LaneRows lr{b.mtp_lane_rows, b.mtp_lane_rows + MAXT};
        qsa_select_lanes(b.qsa_q, b.mtp_qsa_pooled, b.pg, b.qsa_scores, b.qsa_cells, b.qsa_ncells, mp, lr, T, b.att_ring.card, b.st);
        attn_part_lanes(b.qg, b.mtp_kc, b.mtp_vc, b.pg, lr, T, b.att_part, b.st, b.qsa_cells, b.qsa_ncells, b.att_ring.card);
        attn_finish(b.qg, b.att, b.att_ring, (const unsigned*) (b.mtp_ctl + 1), 128, T, b.att_part, b.st);
        quantize_q8_1(b.att, b.xq, w.nq * HD, T, b.st);
        dense(L.wo, b.xq, b.bo_a, b, T);
        ring_sum(b.ring, b.bo_a, T, (const unsigned*) (b.mtp_ctl + 1), 0, b.st);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        ck(cudaSetDevice(b.dev), "set");
        ffn_half(T, 1, b, m_.cards[c], mtp_->cards[c].L, b.mtp_R, b.mtp_ctl);
    }
    for (int c = 0; c < NCARD; ++c) {
        CardBuf& b = *cb_[c];
        const MtpCard& mc = mtp_->cards[c];
        ck(cudaSetDevice(b.dev), "set");
        hc_read(b.mtp_R, b.bo_f, b.inj_f, mc.mix_norm, mc.mix_down, mc.mix_up, nullptr, eps, b.mixed, b.inj_a, b.hcs, b.ring.err, T, b.st, b.xq);
        dense(mc.head, b.xq, b.mtp_logits, b, T);
        const int nv = mc.dv1 - mc.dv0;
        for (int i = 0; i < nl; ++i)   // each lane's last row, penalized as draft() does before the argmax
            if (reqs[(size_t) i].pen)
                penalize_row(b.mtp_logits + (size_t) (off[(size_t) i] + reqs[(size_t) i].T - 1) * nv, nv, b.mtp_pen_l + (size_t) i * (1 + 2 * PEN_CAP),
                             b.mtp_pen_par_l + 3 * i, b.st);
        argmax(b.mtp_logits, mc.dv1 - mc.dv0, T, mc.dv0, b.mtp_bv, b.mtp_bi, b.am_keys, b.ring.err, b.st);
        for (int i = 0; i < nl; ++i)   // and the candidates of the lanes that sample their drafts
            if (reqs[(size_t) i].cands && nv > 0) {
                ck(cudaMemsetAsync(b.mtp_tk_err_l + i, 0, 4, b.st), "draft top err");
                fnx::top_tokens(b.mtp_logits + (size_t) (off[(size_t) i] + reqs[(size_t) i].T - 1) * nv, nv, 1, mc.dv0, b.mtp_zero, b.mtp_zero, b.mtp_zero, 0,
                                b.mtp_pen_par_l + 3 * i, nullptr, nullptr, b.mtp_zero, b.mtp_zero, b.mtp_tk_v_l + (size_t) i * TOPK_N, b.mtp_tk_i_l + (size_t) i * TOPK_N, b.mtp_tk_err_l + i, b.st);
            }
    }
}

}  // namespace fnx

namespace fnx {
// memo (memo_patch.py): the host's record of which position and token each memo slot holds
void Engine::memo_tags(int pos, const int* tokens, int n) {
    if (!memo_) return;
    for (int i = 0; i < n; ++i) {
        const int p = pos + i, k = p % memo_cap_;
        memo_pos_[k] = p;
        memo_tok_[k] = tokens[i];
    }
}

void Engine::noc_report() {
    if (!memo_stat_) return;
    unsigned long long s[8];
    std::memcpy(s, (const void*) memo_stat_, sizeof s);
    if (!s[0]) return;
    std::memset((void*) memo_stat_, 0, sizeof s);
    const NocMode& m = noc_mode();
    std::fprintf(stderr, "fnx: noc g2 %d split %d memo %d: %llu layer-windows, missing %.2f a layer, copied ahead %.2f a layer; memo rows %llu: "
                 "memo hits %.1f %%, probe hits %.1f %% (of 6 each)\n", m.g2, m.split, m.memo, s[0], (double) s[1] / s[0], (double) s[2] / s[0],
                 s[3], s[3] ? 100.0 * s[4] / (6.0 * s[3]) : 0.0, s[3] ? 100.0 * s[5] / (6.0 * s[3]) : 0.0);
}
}  // namespace fnx
