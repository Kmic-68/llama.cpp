#include "common.cuh"
#include "ssm-conv.cuh"
#include "gated_delta_net.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// Single-token decode of the short conv in recurrent layers: CONCAT(conv state, new column) -> CPY of the
// window's last d_conv - 1 columns back into the state -> SSM_CONV -> SILU, as one kernel with one thread
// per channel. Each thread reads its channel's state before writing it, so the in-place state update
// (state read straight from the cache) is race-free. The convolution is ssm_conv_f32's first-token
// expression (same products in the same order, the same + bias of 0, the same silu), so the result is
// bit-identical; the concat output is written too, for any other reader.
template <int d_conv>
static __global__ void ssm_conv_decode_fused_f32(
        const char * st, const int64_t st_nb0, const int64_t st_nb1, const int32_t * g_idx, const int64_t g_row_bytes,
        const char * __restrict__ xn, const int64_t xn_nb1,
        const float * __restrict__ w, const int64_t w_nb1,
        float * cat, float * st_out, float * __restrict__ y, const int64_t C) {
    const int64_t c = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    // the state may be the cache row itself (st_out aliases it): no __restrict__ on st / st_out / cat
    if (g_idx != nullptr) {
        st += (int64_t) g_idx[0]*g_row_bytes;   // gathered in place: the sequence's cache row
    }
    float x[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        x[j] = *(const float *) (st + c*st_nb1 + j*st_nb0);
    }
    x[d_conv - 1] = *(const float *) (xn + c*xn_nb1);
    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = w[c*(w_nb1/sizeof(float)) + j];
    }
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        cat[c*d_conv + j] = x[j];
    }
#pragma unroll
    for (int j = 0; j < d_conv - 1; ++j) {
        st_out[c*(d_conv - 1) + j] = x[j + 1];
    }
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        sumf += x[j] * wr[j];
    }
    sumf += 0.0f; // ssm_conv_f32's bias of 0 (turns a -0 into +0, as there)
    y[c] = ggml_cuda_op_silu_single(sumf);
}

bool ggml_cuda_ssm_conv_decode_fused_ok(const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * conv,
                                        const ggml_tensor * silu) {
    const ggml_tensor * st = concat->src[0];   // [d_conv - 1, C, 1] conv state (may be read in place from the cache)
    const ggml_tensor * xn = concat->src[1];   // [1, C, 1] new column
    const ggml_tensor * w  = conv->src[1];     // [d_conv, C]
    const ggml_tensor * tail = cpy->src[0];
    const ggml_tensor * sd   = cpy->src[1];
    const int64_t C = concat->ne[1];
    if (ggml_get_op_params_i32(concat, 0) != 0 || concat->ne[0] != 4 || st->ne[0] != 3 || xn->ne[0] != 1 ||
            concat->ne[2] != 1 || concat->ne[3] != 1 || st->ne[1] != C || xn->ne[1] != C ||
            concat->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 || xn->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(concat) || tail->view_src != concat || tail->view_offs != sizeof(float) || tail->ne[0] != 3 ||
            tail->ne[1] != C || tail->ne[2] != 1 || sd->type != GGML_TYPE_F32 || !ggml_is_contiguous(sd) || ggml_nelements(sd) != 3*C ||
            conv->src[0] != concat || w->ne[0] != 4 || w->ne[1] != C || w->nb[0] != sizeof(float) ||
            conv->ne[0] != C || conv->ne[1] != 1 || conv->ne[2] != 1 || silu->src[0] != conv ||
            silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(silu) || ggml_nelements(silu) != C) {
        return false;
    }
    return true;
}

bool ggml_cuda_op_ssm_conv_decode_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * cpy,
                                        const ggml_tensor * conv, const ggml_tensor * silu) {
    if (!ggml_cuda_ssm_conv_decode_fused_ok(concat, cpy, conv, silu)) {
        return false;
    }
    const ggml_tensor * st = concat->src[0];
    const ggml_tensor * xn = concat->src[1];
    const ggml_tensor * w  = conv->src[1];
    const ggml_tensor * sd = cpy->src[1];
    const int64_t C = concat->ne[1];
    const int nt = 256;
    // the conv state's GET_ROWS may have been skipped, the concat reading the cache row in place
    // (gated_delta_net.cu, ggml_cuda_gdn_gather_*): read it from there too
    const float *   g_base = nullptr;
    const int32_t * g_idx  = nullptr;
    int64_t         g_row  = 0;
    const bool gathered = ggml_cuda_gdn_gather_lookup(concat, &g_base, &g_idx, &g_row);
    ssm_conv_decode_fused_f32<4><<<(unsigned) ((C + nt - 1)/nt), nt, 0, ctx.stream()>>>(
        gathered ? (const char *) g_base : (const char *) st->data, st->nb[0], st->nb[1],
        gathered ? g_idx : nullptr, g_row*(int64_t) sizeof(float), (const char *) xn->data, xn->nb[1],
        (const float *) w->data, w->nb[1], (float *) concat->data, (float *) sd->data, (float *) silu->data, C);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
