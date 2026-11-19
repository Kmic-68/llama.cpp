// lgr-dump: llama.cpp's logits in fnx_run's .lgr format (top 64 of each step), the reference side of fnx's math checks.
//
//   lgr-dump MODEL tokenize TEXT.txt OUT.bin [MAX]   tokenize a text file (no BOS) into a prompt file (fnx's ref format)
//   lgr-dump MODEL gen PROMPT.bin N OUT.lgr          the prompt, then N greedy steps: OUT holds each step's top 64
//   lgr-dump MODEL force PROMPT.bin CONT.lgr OUT.lgr the prompt, then CONT.lgr's tokens teacher-forced
//   lgr-dump MODEL pieces ID...                     each token id's text, one per line (to read a .lgr by eye)
//
// Step i is the distribution after the prompt and the continuation's first i tokens, and records token i of the
// continuation (fnx_run logits' convention). Env: NGL (default 0 = CPU backend), NCTX (default prompt + steps + 16),
// NBATCH (default 512), THREADS (default all), LGR_ONE=1 feeds the continuation one token per decode.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static int envi(const char * k, int d) { const char * v = getenv(k); return v ? atoi(v) : d; }

static std::vector<int> read_prompt(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string(path) + ": cannot open");
    uint32_t nl = 0, nd = 0; int64_t ne = 0;
    std::string name;
    bool ok = fread(&nl, 4, 1, f) == 1 && nl == 6;
    if (ok) { name.resize(nl); ok = fread(name.data(), 1, nl, f) == nl && name == "tokens"; }
    ok = ok && fread(&nd, 4, 1, f) == 1 && nd == 1 && fread(&ne, 8, 1, f) == 1 && ne > 0;
    std::vector<float> v(ok ? (size_t) ne : 0);
    ok = ok && fread(v.data(), 4, v.size(), f) == v.size();
    fclose(f);
    if (!ok) throw std::runtime_error(std::string(path) + ": not a prompt file");
    return std::vector<int>(v.begin(), v.end());
}

static void write_prompt(const char * path, const std::vector<int> & t) {
    FILE * f = fopen(path, "wb");
    if (!f) throw std::runtime_error(std::string(path) + ": cannot write");
    const uint32_t nl = 6, nd = 1; const int64_t ne = (int64_t) t.size();
    fwrite(&nl, 4, 1, f); fwrite("tokens", 1, 6, f); fwrite(&nd, 4, 1, f); fwrite(&ne, 8, 1, f);
    for (int x : t) { const float v = (float) x; fwrite(&v, 4, 1, f); }
    if (fclose(f) != 0) throw std::runtime_error(std::string(path) + ": write failed");
}

static std::vector<int> read_cont(const char * path, const std::vector<int> & prompt) {
    FILE * f = fopen(path, "rb");
    char m[4]; uint32_t k = 0; int32_t np = 0, ns = 0;
    if (!f || fread(m, 1, 4, f) != 4 || memcmp(m, "LGR1", 4) || fread(&k, 4, 1, f) != 1 || fread(&np, 4, 1, f) != 1)
        throw std::runtime_error(std::string(path) + ": not an .lgr file");
    std::vector<int32_t> pr((size_t) np);
    if (fread(pr.data(), 4, pr.size(), f) != pr.size() || fread(&ns, 4, 1, f) != 1) throw std::runtime_error("short .lgr");
    if (std::vector<int>(pr.begin(), pr.end()) != prompt) throw std::runtime_error("the .lgr file has another prompt");
    std::vector<int> c;
    for (int i = 0; i < ns; ++i) {
        int32_t t; if (fread(&t, 4, 1, f) != 1) throw std::runtime_error("short .lgr");
        c.push_back(t); fseek(f, 4 + (long) k * 8, SEEK_CUR);
    }
    fclose(f);
    return c;
}

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: see the header of lgr-dump.cpp\n"); return 2; }
    const std::string mode = argv[2];
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = envi("NGL", 0);
    if (mode == "tokenize" || mode == "pieces") mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    if (mode == "tokenize") {
        std::ifstream in(argv[3]); std::stringstream ss; ss << in.rdbuf(); const std::string text = ss.str();
        std::vector<llama_token> t(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), t.data(), (int32_t) t.size(), false, true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        t.resize(n);
        if (argc > 5) t.resize(std::min<size_t>(t.size(), (size_t) atoi(argv[5])));
        write_prompt(argv[4], std::vector<int>(t.begin(), t.end()));
        fprintf(stderr, "%zu tokens -> %s\n", t.size(), argv[4]);
        return 0;
    }

    if (mode == "pieces") {
        for (int i = 3; i < argc; ++i) {
            char buf[256];
            const int n = llama_token_to_piece(vocab, atoi(argv[i]), buf, sizeof buf, 0, true);
            printf("%s\t%.*s\n", argv[i], n > 0 ? n : 0, buf);
        }
        return 0;
    }

    const std::vector<int> prompt = read_prompt(argv[3]);
    const bool gen = mode == "gen";
    std::vector<int> cont;
    int ns = 0;
    const char * out_path;
    if (gen) { ns = atoi(argv[4]); out_path = argv[5]; }
    else if (mode == "force") { cont = read_cont(argv[4], prompt); ns = (int) cont.size(); out_path = argv[5]; }
    else { fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2; }
    if (argc < 6) { fprintf(stderr, "missing OUT.lgr\n"); return 2; }

    auto cp = llama_context_default_params();
    cp.n_ctx = envi("NCTX", (int) prompt.size() + ns + 16);
    cp.n_batch = envi("NBATCH", 512);
    cp.n_ubatch = std::min<int>(cp.n_batch, envi("NUBATCH", cp.n_batch));
    cp.n_seq_max = 1;
    const int th = envi("THREADS", (int) std::thread::hardware_concurrency());
    cp.n_threads = th; cp.n_threads_batch = th;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context failed\n"); return 1; }

    constexpr uint32_t K = 64;
    FILE * out = fopen(out_path, "wb");
    if (!out) { perror(out_path); return 1; }
    const int32_t np = (int32_t) prompt.size(), ns32 = ns;
    fwrite("LGR1", 1, 4, out); fwrite(&K, 4, 1, out); fwrite(&np, 4, 1, out);
    for (int t : prompt) { const int32_t v = t; fwrite(&v, 4, 1, out); }
    fwrite(&ns32, 4, 1, out);
    int written = 0;
    std::vector<int> idx(n_vocab);
    auto put = [&](const float * row) -> int {
        float mx = -INFINITY;
        for (int i = 0; i < n_vocab; ++i) mx = std::max(mx, row[i]);
        double s = 0;
        for (int i = 0; i < n_vocab; ++i) s += std::exp((double) row[i] - mx);
        const float lse = (float) (mx + std::log(s));
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
        const int32_t taken = gen ? idx[0] : cont[(size_t) written];
        fwrite(&taken, 4, 1, out); fwrite(&lse, 4, 1, out);
        for (uint32_t k = 0; k < K; ++k) { const int32_t id = idx[k]; fwrite(&id, 4, 1, out); fwrite(&row[id], 4, 1, out); }
        ++written;
        return taken;
    };

    // feed tokens [0, n) of seq at positions pos0..; logits wanted for the last `nlog` of them
    llama_batch b = llama_batch_init(cp.n_batch, 0, 1);
    int pos = 0;
    auto feed = [&](const std::vector<int> & toks, int nlog) {
        const int n = (int) toks.size();
        for (int i0 = 0; i0 < n; i0 += cp.n_batch) {
            const int i1 = std::min(n, i0 + (int) cp.n_batch);
            b.n_tokens = 0;
            for (int i = i0; i < i1; ++i) {
                b.token[b.n_tokens] = toks[i]; b.pos[b.n_tokens] = pos++; b.n_seq_id[b.n_tokens] = 1;
                b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = i >= n - nlog; b.n_tokens++;
            }
            if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode failed");
            for (int i = i0; i < i1; ++i)
                if (i >= n - nlog) put(llama_get_logits_ith(ctx, i - i0));
        }
    };

    if (gen) {
        // prompt, then one token per decode (the greedy choice of the previous step)
        b.n_tokens = 0;
        std::vector<int> p(prompt.begin(), prompt.end());
        // feed the prompt but take only its last row, then generate
        const int n = (int) p.size();
        int last = -1;
        for (int i0 = 0; i0 < n; i0 += cp.n_batch) {
            const int i1 = std::min(n, i0 + (int) cp.n_batch);
            b.n_tokens = 0;
            for (int i = i0; i < i1; ++i) {
                b.token[b.n_tokens] = p[i]; b.pos[b.n_tokens] = pos++; b.n_seq_id[b.n_tokens] = 1;
                b.seq_id[b.n_tokens][0] = 0; b.logits[b.n_tokens] = i == n - 1; b.n_tokens++;
            }
            if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode failed");
            if (i1 == n) last = put(llama_get_logits_ith(ctx, i1 - 1 - i0));
        }
        while (written < ns) {
            b.n_tokens = 1; b.token[0] = last; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
            if (llama_decode(ctx, b) != 0) throw std::runtime_error("decode failed");
            last = put(llama_get_logits_ith(ctx, 0));
        }
    } else {
        // the prompt with its last row (step 0), then cont[0..ns-2] with every row (steps 1..ns-1)
        feed(prompt, 1);
        std::vector<int> c(cont.begin(), cont.end() - 1);
        if (envi("LGR_ONE", 0)) {
            for (int t : c) feed({t}, 1);
        } else {
            feed(c, (int) c.size());
        }
    }
    if (fclose(out) != 0) { fprintf(stderr, "write failed\n"); return 1; }
    fprintf(stderr, "lgr-dump: %d prompt tokens, %d steps -> %s\n", np, written, out_path);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
