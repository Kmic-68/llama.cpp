#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    // p100-series: a 2D grid (embd blocks, token) instead of 64-bit div/mod per element
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t it = blockIdx.y;

    if (i0 >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// gated: post = gate_scale_out*sigmoid(gate_scale_in*post), the fused SCALE -> SIGMOID -> SCALE chain of qwen4exp
template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float gate_scale_in,
        float gate_scale_out) {
    ggml_cuda_pdl_lc();
    // p100-series: a 3D grid (embd blocks, stream, token) instead of 64-bit div/mod per element
    const int64_t i0   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t idst = blockIdx.y;
    const int64_t it   = blockIdx.z;

    if (i0 >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    float p = post[idst*sp0 + it*sp1];
    if constexpr (gated) {
        p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
    }

    float sum = x[i0*sx0 + it*sx1] * p;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// p100-series: every output stream of an (element, token) in one thread, so x and the residual streams are read once instead of
// once per output stream (the (embd, stream, token) grid read them hc times). Each output's arithmetic is dsv4_hc_post_f32's in the
// same order: the same bits. GGML_CUDA_HC_POST_ALL=0: the per-stream kernel
// one (element, token): every output stream of dsv4_hc_post_f32, in its order (shared by the fused hc_post + rms_norm kernel)
template <bool has_comb, bool gated, int HC>
static __device__ __forceinline__ void dsv4_hc_post_elem(
        const float * x, const float * residual, const float * post, const float * comb, const int64_t i0, const int64_t it,
        int64_t sx0, int64_t sx1, int64_t sr0, int64_t sr1, int64_t sr2, int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2, float gate_scale_in, float gate_scale_out, float (&out)[HC]) {
    const float xv = x[i0*sx0 + it*sx1];
    float r[HC];
#pragma unroll
    for (int isrc = 0; isrc < HC; ++isrc) {
        r[isrc] = residual[i0*sr0 + isrc*sr1 + it*sr2];
    }
#pragma unroll
    for (int idst = 0; idst < HC; ++idst) {
        float p = post[idst*sp0 + it*sp1];
        if constexpr (gated) {
            p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
        }
        // explicit, as the per-stream kernel compiles: with comb a rounded product, then one fma a stream (unrolled, the compiler
        // could fuse x*p into the first term instead); without comb one fma, x*p + r
        float sum;
        if constexpr (has_comb) {
            sum = __fmul_rn(xv, p);
#pragma unroll
            for (int isrc = 0; isrc < HC; ++isrc) {
                sum = __fmaf_rn(r[isrc], comb[idst*sc0 + isrc*sc1 + it*sc2], sum);
            }
        } else {
            sum = __fmaf_rn(xv, p, r[idst]);
        }
        out[idst] = sum;
    }
}

// p100-series: every output stream of an (element, token) in one thread, so x and the residual streams are read once instead of
// once per output stream (the (embd, stream, token) grid read them hc times). Each output's arithmetic is dsv4_hc_post_f32's in the
// same order: the same bits. GGML_CUDA_HC_POST_ALL=0: the per-stream kernel
template <bool has_comb, bool gated, int HC>
static __global__ void dsv4_hc_post_f32_all(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        int64_t n_embd, int64_t sx0, int64_t sx1, int64_t sr0, int64_t sr1, int64_t sr2, int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2, int64_t sd0, int64_t sd1, int64_t sd2, float gate_scale_in, float gate_scale_out) {
    const int64_t i0 = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t it = blockIdx.z;
    if (i0 >= n_embd) {
        return;
    }
    float o[HC];
    dsv4_hc_post_elem<has_comb, gated, HC>(x, residual, post, comb, i0, it, sx0, sx1, sr0, sr1, sr2, sp0, sp1, sc0, sc1, sc2,
                                           gate_scale_in, gate_scale_out, o);
#pragma unroll
    for (int idst = 0; idst < HC; ++idst) {
        dst[i0*sd0 + idst*sd1 + it*sd2] = o[idst];
    }
}

// p100-series: hc_post fused with the next hyper-connection RMS norm (+ weight) of each stream: a 1024-thread block per token
// computes the token's hc_post outputs (dsv4_hc_post_elem, the same bits), stores them, and normalizes each stream row exactly as
// rms_norm_f32<1024, true> does (thread t sums columns t, t + 1024, t + 2048 with fmas in that order, warp butterflies, a 32-lane
// butterfly over the warps), so neither output changes; the norm no longer re-reads the 4 streams. Rows of up to 8192 values.
template <bool has_comb, bool gated, int NU>
static __global__ void __launch_bounds__(1024) dsv4_hc_post_rms_f32(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        int n_embd, int64_t sx0, int64_t sx1, int64_t sr0, int64_t sr1, int64_t sr2, int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2, int64_t sd0, int64_t sd1, int64_t sd2, float gate_scale_in, float gate_scale_out,
        const float * w, int64_t w_s1, const float eps, float * y) {
    constexpr int HC = 4;   // NU: columns a thread (the 1024-thread rms kernel's u < 8 terms past the row are zeros: skipped, exact)
    __shared__ float s_sum[HC][32];
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE;
    const int64_t it = blockIdx.x;
    float v[HC][NU];
    float tmp[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int u = 0; u < NU; ++u) {
        const int i0 = tid + u*1024;
        float o[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (i0 < n_embd) {
            dsv4_hc_post_elem<has_comb, gated, HC>(x, residual, post, comb, i0, it, sx0, sx1, sr0, sr1, sr2, sp0, sp1, sc0, sc1, sc2,
                                                   gate_scale_in, gate_scale_out, o);
#pragma unroll
            for (int s = 0; s < HC; ++s) {
                dst[i0*sd0 + s*sd1 + it*sd2] = o[s];
            }
        }
#pragma unroll
        for (int s = 0; s < HC; ++s) {
            v[s][u] = o[s];
            tmp[s] = __fmaf_rn(o[s], o[s], tmp[s]);
        }
    }
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        tmp[s] = warp_reduce_sum(tmp[s]);
        if (lane == 0) {
            s_sum[s][warp] = tmp[s];
        }
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < HC; ++s) {
        const float tot   = warp_reduce_sum(s_sum[s][lane]);
        const float mean  = tot / n_embd;
        const float scale = rsqrtf(mean + eps);
        float * yr = y + (it*HC + s)*n_embd;
        const float * wr = w + s*w_s1;
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            const int i0 = tid + u*1024;
            if (i0 < n_embd) {
                yr[i0] = scale * v[s][u] * wr[i0];
            }
        }
    }
}

// the fused hc_post (gated or not, as ggml_cuda_op_dsv4_hc_post) + RMS_NORM + MUL(w); false: not this shape (nothing launched)
bool ggml_cuda_op_dsv4_hc_post_rms(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * gate_scale_in,
                                   const ggml_tensor * gate_scale_out, const ggml_tensor * rms, ggml_tensor * mul) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];
    const ggml_tensor * w     = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t n_embd = x->ne[0], n_tokens = x->ne[1], hc = residual->ne[1];
    if (hc != 4 || n_embd > 8192 || n_embd < 1024 || n_tokens > 65535 || x->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 ||
            post->type != GGML_TYPE_F32 || (comb && comb->type != GGML_TYPE_F32) || dst->type != GGML_TYPE_F32 ||
            w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || !ggml_is_contiguous(w) ||
            w->ne[0] != n_embd || (w->ne[1] != 1 && w->ne[1] != hc) || w->ne[2] != 1 || w->ne[3] != 1 ||
            !ggml_are_same_shape(mul, dst) || dst->ne[0] != n_embd || dst->ne[1] != hc || dst->ne[2] != n_tokens) {
        return false;
    }
    // the norm output is written while other blocks still read the hc_post inputs: the unfused graph may have placed it on them
    auto ovl = [&](const ggml_tensor * t) {
        if (t == nullptr) {
            return false;
        }
        const char * a0 = (const char *) mul->data, * t0 = (const char *) t->data;
        return a0 < t0 + ggml_nbytes(t) && t0 < a0 + ggml_nbytes(mul);
    };
    if (ovl(x) || ovl(residual) || ovl(post) || ovl(comb) || ovl(dst) || ovl(w)) {
        return false;
    }
    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);
    const size_t nbc0 = comb ? comb->nb[0] : 0, nbc1 = comb ? comb->nb[1] : 0, nbc2 = comb ? comb->nb[2] : 0;
    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    const bool nu3 = n_embd <= 3072;
    auto k = comb ? (gated ? (nu3 ? dsv4_hc_post_rms_f32<true, true, 3> : dsv4_hc_post_rms_f32<true, true, 8>)
                           : (nu3 ? dsv4_hc_post_rms_f32<true, false, 3> : dsv4_hc_post_rms_f32<true, false, 8>))
                  : (gated ? (nu3 ? dsv4_hc_post_rms_f32<false, true, 3> : dsv4_hc_post_rms_f32<false, true, 8>)
                           : (nu3 ? dsv4_hc_post_rms_f32<false, false, 3> : dsv4_hc_post_rms_f32<false, false, 8>));
    k<<<(unsigned) n_tokens, 1024, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) residual->data, (const float *) post->data, comb ? (const float *) comb->data : nullptr,
        (float *) dst->data, (int) n_embd, nbx0 / sizeof(float), nbx1 / sizeof(float), nbr0 / sizeof(float), nbr1 / sizeof(float),
        nbr2 / sizeof(float), nbp0 / sizeof(float), nbp1 / sizeof(float), nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
        nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float), gate_in, gate_out,
        (const float *) w->data, w->ne[1] == 1 ? 0 : (int64_t) (w->nb[1] / sizeof(float)), eps, (float *) mul->data);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    GGML_ASSERT(n_tokens <= 65535);
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, n_tokens, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];

    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    GGML_ASSERT(hc <= 65535 && n_tokens <= 65535);
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, hc, n_tokens);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    static const bool all_on = [] { const char * e = getenv("GGML_CUDA_HC_POST_ALL"); return !e || atoi(e) != 0; }();
    if (all_on && hc == 4) {
        const dim3 grid_all((n_embd + block_size - 1) / block_size, 1, n_tokens);
        auto ka = comb ? (gated ? dsv4_hc_post_f32_all<true,  true, 4> : dsv4_hc_post_f32_all<true,  false, 4>)
                       : (gated ? dsv4_hc_post_f32_all<false, true, 4> : dsv4_hc_post_f32_all<false, false, 4>);
        ka<<<grid_all, block_dims, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
        return;
    }
    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true,  true> : dsv4_hc_post_f32<true,  false>)
                       : (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}
