// Per-position KL divergence between two llama-perplexity "--kl-divergence-base" files (out of tree).
// Both files hold, for every scored token, the full next-token distribution as 16-bit quantized log-probs
// (record = float scale, float min_log_prob, then n_vocab uint16). llama-perplexity scores positions
// n_ctx/2 .. n_ctx-2 of each chunk, so earlier positions are not in the files.
//
//   kldpos <base.bin> <test.bin> [bucket=1024]
//
// Prints: overall mean / median / 99.9% / max KLD(base || test) and top-1 agreement, then one CSV line per
// position bucket (aggregated over chunks): pos_start,pos_end,n,mean_kld,max_kld,top1_agree_pct
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

struct Hdr { int n_ctx, n_vocab, n_chunk, first, per_chunk; off_t data; };   // first = first scored position

static Hdr open_file(const char * p, int & fd) {
    fd = open(p, O_RDONLY);
    if (fd < 0) { perror(p); exit(1); }
    char magic[8]; Hdr h;
    if (read(fd, magic, 8) != 8 || (strncmp(magic, "_logits_", 8) && strncmp(magic, "_logitsA", 8))) { fprintf(stderr, "%s: bad magic\n", p); exit(1); }
    const bool all = !strncmp(magic, "_logitsA", 8);     // logitdump: every position 0 .. n_ctx-2
    if (read(fd, &h.n_ctx, 4) != 4 || read(fd, &h.n_vocab, 4) != 4 || read(fd, &h.n_chunk, 4) != 4) { fprintf(stderr, "%s: short header\n", p); exit(1); }
    h.data = 20 + (off_t) h.n_chunk * h.n_ctx * 4;
    h.first = all ? 0 : h.n_ctx / 2; h.per_chunk = h.n_ctx - 1 - h.first;
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: kldpos base.bin test.bin [bucket]\n"); return 2; }
    const int bucket = argc > 3 ? atoi(argv[3]) : 1024;
    int fa, fb;
    Hdr a = open_file(argv[1], fa), b = open_file(argv[2], fb);
    if (a.n_ctx != b.n_ctx || a.n_vocab != b.n_vocab) {
        fprintf(stderr, "header mismatch: n_ctx %d/%d n_vocab %d/%d n_chunk %d/%d\n", a.n_ctx, b.n_ctx, a.n_vocab, b.n_vocab, a.n_chunk, b.n_chunk);
        return 1;
    }
    { // the token streams must be identical
        std::vector<int> ta((size_t) std::min(a.n_chunk, b.n_chunk) * a.n_ctx), tb(ta.size());
        if (pread(fa, ta.data(), ta.size() * 4, 20) != (ssize_t) (ta.size() * 4) || pread(fb, tb.data(), tb.size() * 4, 20) != (ssize_t) (tb.size() * 4)) { fprintf(stderr, "token read failed\n"); return 1; }
        if (ta != tb) { fprintf(stderr, "token streams differ\n"); return 1; }
    }
    const int nv = 2 * ((a.n_vocab + 1) / 2) + 4;
    const int first = std::max(a.first, b.first);          // compare the positions both files hold
    const int per_chunk = a.n_ctx - 1 - first;
    const int n_chunk = std::min(a.n_chunk, b.n_chunk);
    const long n_tok = (long) n_chunk * per_chunk;
    const size_t rec = (size_t) nv * 2;
    std::vector<double> kld(n_tok); std::vector<char> same(n_tok); std::vector<char> bad(n_tok, 0);
    std::atomic<long> next(0), ndiff(0);
    const int nth = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t) th.emplace_back([&] {
        std::vector<uint16_t> ra(nv), rb(nv);
        for (;;) {
            const long i = next.fetch_add(1);
            if (i >= n_tok) break;
            const long c = i / per_chunk, p = first + i % per_chunk;
            const off_t offa = (off_t) (c * a.per_chunk + (p - a.first)) * rec, offb = (off_t) (c * b.per_chunk + (p - b.first)) * rec;
            if (pread(fa, ra.data(), rec, a.data + offa) != (ssize_t) rec || pread(fb, rb.data(), rec, b.data + offb) != (ssize_t) rec) { bad[i] = 1; continue; }
            float sa, ma, sb, mb;
            memcpy(&sa, ra.data(), 4); memcpy(&ma, ra.data() + 2, 4); memcpy(&sb, rb.data(), 4); memcpy(&mb, rb.data() + 2, 4);
            const uint16_t * qa = ra.data() + 4; const uint16_t * qb = rb.data() + 4;
            double s = 0, pa_sum = 0; int ia = 0, ib = 0;
            for (int v = 0; v < a.n_vocab; ++v) {
                const double la = (double) sa * qa[v] + ma, lb = (double) sb * qb[v] + mb;
                const double pa = exp(la);
                s += pa * (la - lb); pa_sum += pa;
                if (qa[v] > qa[ia]) ia = v;
                if (qb[v] > qb[ib]) ib = v;
            }
            if (!std::isfinite(s)) bad[i] = 1;
            kld[i] = s; same[i] = ia == ib; if (memcmp(ra.data(), rb.data(), rec)) ndiff.fetch_add(1);
        }
    });
    for (auto & t : th) t.join();
    long nbad = 0; for (long i = 0; i < n_tok; ++i) nbad += bad[i];
    std::vector<double> v; v.reserve(n_tok); double sum = 0, mx = -1e30; long ns = 0;
    for (long i = 0; i < n_tok; ++i) if (!bad[i]) { v.push_back(kld[i]); sum += kld[i]; mx = std::max(mx, kld[i]); ns += same[i]; }
    std::sort(v.begin(), v.end());
    const long n = (long) v.size();
    printf("files: n_ctx=%d n_vocab=%d n_chunk=%d scored_tokens=%ld (positions %d..%d per chunk) bad_or_nonfinite=%ld\n", a.n_ctx, a.n_vocab, n_chunk, n_tok, first, a.n_ctx - 2, nbad);
    printf("records that differ bytewise: %ld of %ld\n", ndiff.load(), n_tok);
    if (n == 0) { printf("no valid tokens\n"); return 1; }
    printf("overall: mean_kld=%.6f median=%.6f p99.9=%.6f max=%.6f top1_agree=%.3f%%\n", sum / n, v[n / 2], v[std::min(n - 1, (long) (0.999 * n))], mx, 100.0 * ns / n);
    printf("pos_start,pos_end,n,mean_kld,max_kld,top1_agree_pct\n");
    for (int p0 = first; p0 < a.n_ctx - 1; p0 += bucket) {
        const int p1 = std::min(a.n_ctx - 1, p0 + bucket);
        double bs = 0, bm = -1e30; long bn = 0, bsame = 0;
        for (int c = 0; c < n_chunk; ++c) for (int p = p0; p < p1; ++p) {
            const long i = (long) c * per_chunk + (p - first);
            if (bad[i]) continue;
            bs += kld[i]; bm = std::max(bm, kld[i]); ++bn; bsame += same[i];
        }
        if (bn) printf("%d,%d,%ld,%.6f,%.6f,%.3f\n", p0, p1 - 1, bn, bs / bn, bm, 100.0 * bsame / bn);
    }
    return 0;
}
