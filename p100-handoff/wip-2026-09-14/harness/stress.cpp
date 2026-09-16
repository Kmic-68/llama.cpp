// Does the load/store race that hit the GEMM softmax generalize to upstream kernels with the same
// shape -- a load and a store of the same device address inside a multi-iteration thread loop?
// Each case recomputes one big op many times on identical input and compares every output to the
// first one, bit for bit. Usage: stress <case> <iterations>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s softmax_oop|softmax_ip|groupnorm_oop|rmsnorm_ip|norm_ip <iters>\n", argv[0]); return 2; }
    const std::string which = argv[1];
    const int iters = atoi(argv[2]);

    ggml_backend_t be = ggml_backend_cuda_init(0);
    if (!be) { fprintf(stderr, "no CUDA backend\n"); return 1; }

    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x = nullptr, * y = nullptr;
    bool inplace = false;
    if (which == "softmax_oop" || which == "softmax_ip") {
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32768, 4096);          // row > shared memory: vals = dst
        inplace = which == "softmax_ip";
        y = inplace ? ggml_soft_max_inplace(ctx, x) : ggml_soft_max(ctx, x);
    } else if (which == "groupnorm_oop") {
        x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 32768);          // 32 groups of 4.2M: block 1024
        y = ggml_group_norm(ctx, x, 32, 1e-6f);
    } else if (which == "rmsnorm_ip") {
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16384, 8192);          // > 8192 cols: two-pass reload path
        inplace = true;
        y = ggml_rms_norm_inplace(ctx, x, 1e-6f);
    } else if (which == "norm_ip") {
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16384, 8192);
        inplace = true;
        y = ggml_norm_inplace(ctx, x, 1e-6f);
    } else { fprintf(stderr, "unknown case\n"); return 2; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    const size_t nbytes = ggml_nbytes(x);
    const size_t n = nbytes / sizeof(float);
    std::vector<float> in(n), ref(n), out(n);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < n; i++) {                          // logits-like: N(0, 3) plus a few spikes
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const double u = ((s >> 11) + 0.5) * (1.0/9007199254740992.0);
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        const double v = ((s >> 11) + 0.5) * (1.0/9007199254740992.0);
        in[i] = (float) (3.0*sqrt(-2*log(u))*cos(2*M_PI*v)) + ((s & 0xfff) == 7 ? 12.0f : 0.0f);
    }
    printf("case %s: %zu elements, in place %d, y->data %s x->data\n", which.c_str(), n, (int) inplace, y->data == x->data ? "==" : "!=");

    ggml_backend_tensor_set(x, in.data(), 0, nbytes);
    long bad_iters = 0, bad_elems = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) {
        if (inplace && it > 0) {
            ggml_backend_tensor_set(x, in.data(), 0, nbytes);
        }
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        ggml_backend_tensor_get(y, it == 0 ? ref.data() : out.data(), 0, nbytes);
        if (it > 0 && memcmp(ref.data(), out.data(), nbytes) != 0) {
            long e = 0; size_t first = n;
            for (size_t i = 0; i < n; i++) { if (memcmp(&ref[i], &out[i], 4) != 0) { e++; if (first == n) first = i; } }
            bad_iters++; bad_elems += e;
            printf("  iter %d: %ld elements differ, first at %zu: %.9g vs %.9g\n", it, e, first, ref[first], out[first]);
            fflush(stdout);
        }
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("case %s: %d iterations (%.3g element-computations) in %.0f s: %ld differing iterations, %ld differing elements\n",
           which.c_str(), iters, (double) iters * n, dt, bad_iters, bad_elems);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
