#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

// p100-series: the unfused graph's rounding (the MUL's rounded products, then the ADD chain in slot order): for the prefill graphs where
// the fused path above is refused (its output aliases the in-place MUL's buffer), so the fusion leaves the model's outputs unchanged
static __global__ void moe_weighted_reduction_seq_f32(const float * __restrict__ experts, const float * __restrict__ expert_scale,
                                                      const float * __restrict__ weights, float * __restrict__ dst,
                                                      const int64_t n_embd, const int n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    float sum = 0.0f;
    for (int expert = 0; expert < n_expert_used; ++expert) {
        const uint64_t row = first_row + expert;
        float p = experts[row * n_embd + col];
        if (expert_scale != nullptr) {
            p = __fmul_rn(p, expert_scale[row]);
        }
        p = __fmul_rn(p, weights[row]);
        sum = expert == 0 ? p : __fadd_rn(sum, p);
    }
    dst[token * n_embd + col] = sum;
}

void ggml_cuda_op_moe_weighted_reduction_seq(ggml_backend_cuda_context & ctx, const ggml_tensor * experts, const ggml_tensor * expert_scale,
                                             const ggml_tensor * weights, ggml_tensor * dst, const bool alias) {
    const int64_t n_embd = experts->ne[0], n_expert_used = experts->ne[1], n_tokens = experts->ne[2] * experts->ne[3];
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> tmp(ctx.pool());
    float * out = (float *) dst->data;
    if (alias) {   // the output overlaps the experts' buffer: another block may still read the rows it would write
        out = tmp.alloc(n_embd * n_tokens);
    }
    const dim3 blocks(n_tokens, (n_embd + 255) / 256, 1);
    moe_weighted_reduction_seq_f32<<<blocks, 256, 0, stream>>>((const float *) experts->data,
        expert_scale ? (const float *) expert_scale->data : nullptr, (const float *) weights->data, out, n_embd, (int) n_expert_used);
    CUDA_CHECK(cudaGetLastError());
    if (alias) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, out, n_embd * n_tokens * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
