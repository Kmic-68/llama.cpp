// Alternative gated delta net paths for prefill (scalar gate, S_v = 128), kept out of
// gated_delta_net.cu so a change here rebuilds one small translation unit.
//   GGML_CUDA_GDN_CHUNKED=1  chunked form (gdn-chunked.cuh), all fp32 (default; 0 = shipped recurrence)
//   GGML_CUDA_GDN_CHUNKED=2  chunked form, K K^T / Q K^T and the state update in fp64
//   GGML_CUDA_GDN_REF=1      sequential recurrence with an fp64 state: an accuracy reference only
#include "common.cuh"

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace gdn_f32 {
#include "gdn-chunked.cuh"
}
#undef TG
#undef TA
#undef TB
#undef TC
#undef TF
#define TG double
#define TC double
namespace gdn_f64 {
#include "gdn-chunked.cuh"
}

struct ggml_cuda_gdn_alt_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta;
    const float * curr_state; float * dst; float * state;
    int64_t H, n_tokens, n_seqs, neqk1, rq3;
    int64_t sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    float scale; int64_t state_slot_stride; int K; const int32_t * s_idx; int64_t s_row;
};

// one thread per (column, head, seq); state column in fp64 local memory
static __global__ void gdn_ref_f64(const ggml_cuda_gdn_alt_args p) {
    const int     j   = threadIdx.x;
    const int64_t h   = blockIdx.x;
    const int64_t seq = blockIdx.y;
    const int64_t iq1 = h % p.neqk1;
    const int64_t iq3 = seq / p.rq3;
    const bool keep = p.K > 1;

    const float * cs = p.s_idx ? p.curr_state + (int64_t) p.s_idx[seq] * p.s_row + h * 16384
                               : p.curr_state + (seq * p.H + h) * 16384;
    double s[128];
    for (int i = 0; i < 128; i++) {
        s[i] = cs[j * 128 + i];
    }
    float * st = p.state + (seq * p.H + h) * 16384 + j * 128;
    for (int64_t t = 0; t < p.n_tokens; t++) {
        const float * kt = p.k + iq3 * p.sq3 + iq1 * p.sq1 + t * p.sq2;
        const float * qt = p.q + iq3 * p.sq3 + iq1 * p.sq1 + t * p.sq2;
        const int64_t gbo = seq * p.sb3 + t * p.sb2 + h * p.sb1;
        const double eg = exp((double) p.g[gbo]);
        const double bt = p.beta[gbo];
        const double vj = p.v[seq * p.sv3 + t * p.sv2 + h * p.sv1 + j];
        double kv = 0.0;
        for (int i = 0; i < 128; i++) {
            kv = fma(s[i], (double) kt[i], kv);
        }
        const double delta = (vj - eg * kv) * bt;
        double o = 0.0;
        for (int i = 0; i < 128; i++) {
            s[i] = fma(eg, s[i], (double) kt[i] * delta);
            o = fma(s[i], (double) qt[i], o);
        }
        p.dst[((seq * p.n_tokens + t) * p.H + h) * 128 + j] = (float) (o * p.scale);
        const int64_t slot = p.n_tokens - 1 - t;
        if ((keep && slot < p.K) || (!keep && slot == 0)) {
            for (int i = 0; i < 128; i++) {
                st[slot * p.state_slot_stride + i] = (float) s[i];
            }
        }
    }
}

// returns false when the shipped kernel should run
bool ggml_cuda_gdn_alt(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_alt_args & a) {
    static const int mode = getenv("GGML_CUDA_GDN_CHUNKED") ? atoi(getenv("GGML_CUDA_GDN_CHUNKED")) : 1;
    static const bool ref = getenv("GGML_CUDA_GDN_REF") && atoi(getenv("GGML_CUDA_GDN_REF")) != 0;
    cudaStream_t stream = ctx.stream();
    if (ref) {
        gdn_ref_f64<<<dim3(a.H, a.n_seqs), 128, 0, stream>>>(a);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    const bool keep = a.K > 1;
    const int64_t L = keep ? a.n_tokens - (a.K - 1) : a.n_tokens;
    if (mode == 0 || L < GCH) {
        return false;
    }
    // q, k, v are read as float4: bases and strides must be 16-byte aligned (odd V strides take the recurrence)
    if ((((uintptr_t) a.q | (uintptr_t) a.k | (uintptr_t) a.v) & 15) != 0 ||
        ((a.sq1 | a.sq2 | a.sq3 | a.sv1 | a.sv2 | a.sv3) & 3) != 0) {
        return false;
    }
    const size_t n = gdn_f32::gdn_ch_scratch_floats(a.H, a.neqk1, a.n_seqs, L);
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), n);
#define GDN_CH_RUN(NS) do { \
        NS::gdn_ch_params p = {}; \
        p.q = a.q; p.k = a.k; p.v = a.v; p.g = a.g; p.beta = a.beta; \
        p.curr_state = a.curr_state; p.dst = a.dst; p.state = a.state; \
        p.H = a.H; p.n_tokens = a.n_tokens; p.n_seqs = a.n_seqs; p.neqk1 = a.neqk1; p.rq3 = a.rq3; p.K = a.K; \
        p.sq1 = a.sq1; p.sq2 = a.sq2; p.sq3 = a.sq3; p.sv1 = a.sv1; p.sv2 = a.sv2; p.sv3 = a.sv3; \
        p.sb1 = a.sb1; p.sb2 = a.sb2; p.sb3 = a.sb3; \
        p.scale = a.scale; p.state_slot_stride = a.state_slot_stride; p.s_idx = a.s_idx; p.s_row = a.s_row; \
        p.keep = keep; \
        NS::gdn_ch_launch(p, scratch.get(), stream); \
    } while (0)
    if (mode == 2) {
        GDN_CH_RUN(gdn_f64);
    } else {
        GDN_CH_RUN(gdn_f32);
    }
#undef GDN_CH_RUN
    CUDA_CHECK(cudaGetLastError());
    return true;
}
