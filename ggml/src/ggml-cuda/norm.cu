#include "norm.cuh"
#include "unary.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    extern __shared__ float2 s_sum2[];

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// do_scale: the output is then out_scale*y + out_bias, exactly the SCALE op's arithmetic on the
// norm's value (the l2 norm of the gated delta net is RMS_NORM -> SCALE)
// do_pair: the upper half of grid z normalises a second tensor (x2 -> dst2) of the same shape and strides
// do_gate (with do_multiply): the output is then silu(gate) * (norm * mul), the fused SILU -> MUL's
// arithmetic on the stored value (qwen35's gated norm); gate has dst's layout
template <int block_size, bool do_multiply = false, bool do_add = false, bool do_scale = false, bool do_pair = false, bool do_gate = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   out_scale            = 1.0f,
                                    const float   out_bias             = 0.0f,
                                    const float * x2                   = nullptr,
                                    float *       dst2                 = nullptr,
                                    const float * gate                 = nullptr) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    int       sample    = blockIdx.z;
    const int tid       = threadIdx.x;
    if constexpr (do_pair) {
        if (sample >= (int) gridDim.z/2) {
            sample -= gridDim.z/2;
            x   = x2;
            dst = dst2;
        }
    }

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
    if constexpr (do_gate) {
        gate += ((sample*nchannels + channel)*nrows + row)*ncols;
    }

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    // A row is normalised by a single block, so no other block covers this one's memory latency
    // and the row is read twice: once to accumulate the sum of squares and again to scale it.
    // When the row fits in a fixed number of registers per thread, hold it there instead: the
    // loads all issue up front, and the second pass costs nothing.
    constexpr int max_regs = 8;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    if (ncols <= block_size*max_regs) {
        float xv[max_regs];
        float tmp = 0.0f;
#pragma unroll
        for (int u = 0; u < max_regs; ++u) {
            const int col = tid + u*block_size;
            xv[u] = col < ncols ? x[col] : 0.0f;
            tmp += xv[u] * xv[u];
        }

        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

        const float mean  = tmp / ncols;
        const float scale = rsqrtf(mean + eps);

#pragma unroll
        for (int u = 0; u < max_regs; ++u) {
            const int col = tid + u*block_size;
            if (col < ncols) {
                if constexpr (do_multiply && do_add) {
                    dst[col] = scale * xv[u] * mul[fastmodulo(col, mul_ncols_packed)] + add[fastmodulo(col, add_ncols_packed)];
                } else if constexpr (do_multiply && do_gate) {
                    const float v = scale * xv[u] * mul[fastmodulo(col, mul_ncols_packed)];
                    dst[col] = ggml_cuda_op_silu_single(gate[col]) * v;
                } else if constexpr (do_multiply) {
                    dst[col] = scale * xv[u] * mul[fastmodulo(col, mul_ncols_packed)];
                } else if constexpr (do_scale) {
                    const float y = scale * xv[u];
                    dst[col] = out_scale * y + out_bias;
                } else {
                    dst[col] = scale * xv[u];
                }
            }
        }
        return;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col] + add[add_col];
        } else if constexpr (do_multiply && do_gate) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const float v     = scale * x[col] * mul[mul_col];
            dst[col]          = ggml_cuda_op_silu_single(gate[col]) * v;
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = scale * x[col] * mul[mul_col];
        } else if constexpr (do_scale) {
            const float y = scale * x[col];
            dst[col] = out_scale * y + out_bias;
        } else {
            dst[col] = scale * x[col];
        }
    }
}

// rms_norm_f32<256, ...> for rows of 32..256 floats, one warp per row (RPB rows per block). The 256-thread
// block left most of its threads idle on such rows and ran at ~50 GB/s. Bit-identical: lane l holds
// x[32w + l]^2 for each "virtual warp" w, sums each with the same butterfly the block's warp w ran, and
// combines the NW sums with the same butterfly the block ran over its 8 warp sums (the empty warps
// contributing exact zeros); every output uses the same expression.
template <int NW, bool do_multiply, bool do_add, bool do_scale, bool do_pair, bool do_gate>
static __global__ void __launch_bounds__(256) rms_norm_f32_warp(
        const float * x, float * dst, const int nrows, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps,
        const float * mul, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const float * add, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3 add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        const float out_scale, const float out_bias, const float * x2, float * dst2, const float * gate) {
    constexpr int ncols = 32*NW;
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int nchannels = gridDim.y;
    const int channel   = blockIdx.y;
    int       sample    = blockIdx.z;
    const int lane      = threadIdx.x;
    if constexpr (do_pair) {
        if (sample >= (int) gridDim.z/2) {
            sample -= gridDim.z/2;
            x   = x2;
            dst = dst2;
        }
    }
    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;
    if constexpr (do_gate) {
        gate += ((sample*nchannels + channel)*nrows + row)*ncols;
    }
    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }
    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float xv[NW];
    float ws[NW];
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        xv[w] = x[32*w + lane];
        float t = 0.0f;
        t += xv[w] * xv[w];
        ws[w] = warp_reduce_sum(t);
    }
    float v2 = 0.0f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        v2 = lane == w ? ws[w] : v2;
    }
    const float tmp   = warp_reduce_sum(v2);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
        const int col = 32*w + lane;
        if constexpr (do_multiply && do_add) {
            dst[col] = scale * xv[w] * mul[fastmodulo(col, mul_ncols_packed)] + add[fastmodulo(col, add_ncols_packed)];
        } else if constexpr (do_multiply && do_gate) {
            const float v = scale * xv[w] * mul[fastmodulo(col, mul_ncols_packed)];
            dst[col] = ggml_cuda_op_silu_single(gate[col]) * v;
        } else if constexpr (do_multiply) {
            dst[col] = scale * xv[w] * mul[fastmodulo(col, mul_ncols_packed)];
        } else if constexpr (do_scale) {
            const float y = scale * xv[w];
            dst[col] = out_scale * y + out_bias;
        } else {
            dst[col] = scale * xv[w];
        }
    }
}

// launches rms_norm_f32_warp in place of rms_norm_f32<256, ...> when the row fits; false otherwise
template <bool do_multiply, bool do_add, bool do_scale = false, bool do_pair = false, bool do_gate = false>
static bool rms_norm_f32_try_warp(const dim3 blocks_num, const int ncols, cudaStream_t stream,
        const float * x, float * dst, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const float * mul, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const float * add, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3 add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        const float out_scale, const float out_bias, const float * x2, float * dst2, const float * gate) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_NORM_WARP"); return !e || atoi(e) != 0; }();
    if (!on || ncols % 32 != 0 || ncols > 256 || ncols == 0) {
        return false;
    }
    constexpr int RPB = 8;
    const int nrows = (int) blocks_num.x;
    const dim3 grid((nrows + RPB - 1)/RPB, blocks_num.y, blocks_num.z);
    const dim3 block(32, RPB, 1);
#define RMS_WARP_CASE(NW) case NW: rms_norm_f32_warp<NW, do_multiply, do_add, do_scale, do_pair, do_gate><<<grid, block, 0, stream>>>( \
        x, dst, nrows, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
        mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add, add_stride_row, add_stride_channel, \
        add_stride_sample, add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, out_scale, out_bias, x2, dst2, gate); break;
    switch (ncols/32) {
        RMS_WARP_CASE(1) RMS_WARP_CASE(2) RMS_WARP_CASE(3) RMS_WARP_CASE(4)
        RMS_WARP_CASE(5) RMS_WARP_CASE(6) RMS_WARP_CASE(7) RMS_WARP_CASE(8)
    }
#undef RMS_WARP_CASE
    CUDA_CHECK(cudaGetLastError());
    return true;
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * x[col];
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        if (rms_norm_f32_try_warp<false, false>(blocks_num, ncols, stream, x, dst, stride_row, stride_channel, stride_sample, eps,
                nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f, nullptr, nullptr, nullptr)) {
            return;
        }
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f, nullptr, nullptr, nullptr);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f, nullptr, nullptr, nullptr);
    }
}

static void rms_norm_mul_f32_cuda(const float *  x,
                                  const float *  mul,
                                  const float *  add,
                                  float *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ncols < 1024) {
            if (rms_norm_f32_try_warp<true, false>(blocks_num, ncols, stream, x, dst, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed,
                    mul_nsamples_packed, nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f, nullptr, nullptr, nullptr)) {
                return;
            }
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f, nullptr, nullptr, nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f, nullptr, nullptr, nullptr);
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (ncols < 1024) {
            if (rms_norm_f32_try_warp<true, true>(blocks_num, ncols, stream, x, dst, stride_row, stride_channel, stride_sample, eps,
                    mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed,
                    mul_nsamples_packed, add, add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                    add_nchannels_packed, add_nsamples_packed, 1.0f, 0.0f, nullptr, nullptr, nullptr)) {
                return;
            }
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, 1.0f, 0.0f, nullptr, nullptr, nullptr);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, 1.0f, 0.0f, nullptr, nullptr, nullptr);
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// Two RMS_NORM -> SCALE of the same shape, strides and parameters (the delta net's q and k l2 norms)
// in one launch; each is bit-identical to ggml_cuda_op_rms_norm_scale's.
bool ggml_cuda_op_rms_norm_scale2(ggml_backend_cuda_context & ctx, ggml_tensor * n1, ggml_tensor * s1, ggml_tensor * n2, ggml_tensor * s2) {
    const ggml_tensor * a = n1->src[0], * b = n2->src[0];
    auto ok = [](const ggml_tensor * n, const ggml_tensor * s) {
        return n->src[0]->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 &&
            n->src[0]->nb[0] == sizeof(float) && ggml_is_contiguous(n) && ggml_is_contiguous(s) &&
            ggml_are_same_shape(n, s) && n->src[0]->ne[0] < 1024;
    };
    if (!ok(n1, s1) || !ok(n2, s2) || !ggml_are_same_shape(a, b) || a->nb[1] != b->nb[1] || a->nb[2] != b->nb[2] ||
            a->nb[3] != b->nb[3] || memcmp(n1->op_params, n2->op_params, sizeof(float)) != 0 ||
            memcmp(s1->op_params, s2->op_params, 2*sizeof(float)) != 0 || a->ne[3] > 32768) {
        return false;
    }
    float eps, sc, bias;
    memcpy(&eps, n1->op_params, sizeof(float));
    memcpy(&sc,   (const float *) s1->op_params + 0, sizeof(float));
    memcpy(&bias, (const float *) s1->op_params + 1, sizeof(float));
    const int64_t ts0 = sizeof(float);
    const dim3 blocks_num(a->ne[1], a->ne[2], 2*a->ne[3]);
    if (rms_norm_f32_try_warp<false, false, true, true>(blocks_num, (int) a->ne[0], ctx.stream(),
            (const float *) a->data, (float *) s1->data, a->nb[1]/ts0, a->nb[2]/ts0, a->nb[3]/ts0, eps,
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), sc, bias, (const float *) b->data, (float *) s2->data, nullptr)) {
        return true;
    }
    rms_norm_f32<256, false, false, true, true><<<blocks_num, 256, 32*sizeof(float), ctx.stream()>>>(
        (const float *) a->data, (float *) s1->data, (int) a->ne[0], a->nb[1]/ts0, a->nb[2]/ts0, a->nb[3]/ts0, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), sc, bias,
        (const float *) b->data, (float *) s2->data, nullptr);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// RMS_NORM -> SCALE in one launch, bit-identical to the two (same kernel, same block size)
bool ggml_cuda_op_rms_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0 = dst->src[0];
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || scale_tensor->type != GGML_TYPE_F32 ||
            src0->nb[0] != sizeof(float) || !ggml_is_contiguous(dst) || !ggml_is_contiguous(scale_tensor) ||
            !ggml_are_same_shape(dst, scale_tensor) || src0->ne[0] >= 1024) {
        return false;
    }
    float eps, sc, bias;
    memcpy(&eps, dst->op_params, sizeof(float));
    memcpy(&sc,   (const float *) scale_tensor->op_params + 0, sizeof(float));
    memcpy(&bias, (const float *) scale_tensor->op_params + 1, sizeof(float));
    const int64_t ts0 = sizeof(float);
    const dim3 blocks_num(src0->ne[1], src0->ne[2], src0->ne[3]);
    if (rms_norm_f32_try_warp<false, false, true>(blocks_num, (int) src0->ne[0], ctx.stream(),
            (const float *) src0->data, (float *) scale_tensor->data, src0->nb[1]/ts0, src0->nb[2]/ts0, src0->nb[3]/ts0, eps,
            nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), sc, bias, nullptr, nullptr, nullptr)) {
        return true;
    }
    rms_norm_f32<256, false, false, true><<<blocks_num, 256, 32*sizeof(float), ctx.stream()>>>(
        (const float *) src0->data, (float *) scale_tensor->data, (int) src0->ne[0], src0->nb[1]/ts0, src0->nb[2]/ts0, src0->nb[3]/ts0, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), sc, bias);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src0_d = (const float *) rms_norm_src->data;
    const float * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_d  = (const float *) rms_norm_src->data;
    const float *       mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = (float *) mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = (float *) mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const float *       add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = (float *) add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = (float *) add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(src0_d, mul_d,add_d,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// ADD -> RMS_NORM -> MUL (the residual add and the next pre-norm), one block per row. The sum is
// written out (it is the next residual) and kept in registers for the norm. Every value is computed
// by exactly the operations of the unfused ADD kernel (a + b) and the fused rms_norm_f32<1024, true>
// register path (same accumulation order, reduction, scale and multiply), so the results are
// bit-identical; it saves the ADD launch and one read of the row.
template <int block_size>
static __global__ void add_rms_norm_mul_f32(const float * a, const float * b, float * sum_out, const float * mul,
                                            float * dst, const int ncols, const int64_t sa, const int64_t sb,
                                            const int64_t ss, const int64_t sd, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    a += row*sa; b += row*sb; sum_out += row*ss; dst += row*sd;

    constexpr int max_regs = 8;
    extern __shared__ float s_sum[];

    float xv[max_regs];
    float tmp = 0.0f;
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        xv[u] = col < ncols ? a[col] + b[col] : 0.0f;
        tmp += xv[u] * xv[u];
    }
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if (col < ncols) {
            sum_out[col] = xv[u];
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if (col < ncols) {
            dst[col] = scale * xv[u] * mul[col];
        }
    }
}

bool ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms, ggml_tensor * mul) {
    const ggml_tensor * a = add->src[0];
    const ggml_tensor * b = add->src[1];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t ncols = add->ne[0], nrows = add->ne[1];
    auto f32_rows = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[0] == ncols && t->ne[1] == nrows &&
            t->ne[2] == 1 && t->ne[3] == 1 && t->nb[1] % sizeof(float) == 0;
    };
    if (!f32_rows(a) || !f32_rows(b) || !f32_rows(add) || !f32_rows(rms) || !f32_rows(mul) ||
            w->type != GGML_TYPE_F32 || w->ne[0] != ncols || ggml_nelements(w) != ncols || w->nb[0] != sizeof(float) ||
            ncols < 1024 || ncols > 1024*8 || rms->src[0] != add) {
        return false;
    }
    // the fused kernel reads a row of a and b, writes it to add, and later the norm to mul: the
    // norm's output must not overlap anything the kernel reads or writes, except row for row
    auto range = [](const ggml_tensor * t) { return std::make_pair((const char *) t->data, (const char *) t->data + ggml_nbytes(t)); };
    auto overlap = [&](const ggml_tensor * x, const ggml_tensor * y) {
        const auto rx = range(x), ry = range(y);
        return rx.first < ry.second && ry.first < rx.second;
    };
    // exact aliases are fine: a thread reads a, b at its own columns of its block's row before it
    // writes add or mul there (the block reduction orders the reads of a row before any write of it)
    auto same = [](const ggml_tensor * x, const ggml_tensor * y) { return x->data == y->data && x->nb[1] == y->nb[1]; };
    if ((overlap(mul, a) && !same(mul, a)) || (overlap(mul, b) && !same(mul, b)) || overlap(mul, add) ||
        overlap(mul, w) || overlap(add, w) || (overlap(add, a) && !same(add, a)) || (overlap(add, b) && !same(add, b))) {
        return false;
    }
    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int64_t fs = sizeof(float);
    add_rms_norm_mul_f32<1024><<<(int) nrows, 1024, 32*sizeof(float), ctx.stream()>>>(
        (const float *) a->data, (const float *) b->data, (float *) add->data, (const float *) w->data, (float *) mul->data,
        (int) ncols, a->nb[1]/fs, b->nb[1]/fs, add->nb[1]/fs, mul->nb[1]/fs, eps);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// RMS_NORM -> MUL(weight) -> MUL(silu(gate)) (qwen35's gated norm) in one launch, bit-identical to the
// fused RMS_NORM+MUL and fused SILU+MUL kernels: the same register path at the same block size.
// check_only: only report whether it applies.
bool ggml_cuda_op_rms_norm_mul_silu_gate(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * mul,
                                         const ggml_tensor * gate, ggml_tensor * out, const bool check_only) {
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    const int64_t ncols = x->ne[0];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            x->nb[0] != sizeof(float) || !ggml_is_contiguous(w) || ggml_nelements(w) != ncols || w->ne[0] != ncols ||
            !ggml_is_contiguous(gate) || !ggml_is_contiguous(out) || !ggml_are_same_shape(gate, rms) ||
            !ggml_are_same_shape(out, rms) || ncols >= 1024) {
        return false;
    }
    if (check_only) {
        return true;
    }
    float eps;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int64_t ts0 = sizeof(float);
    const uint3 one = init_fastdiv_values(1), nc = init_fastdiv_values((uint32_t) ncols);
    const dim3 blocks_num(x->ne[1], x->ne[2], x->ne[3]);
    if (rms_norm_f32_try_warp<true, false, false, false, true>(blocks_num, (int) ncols, ctx.stream(),
            (const float *) x->data, (float *) out->data, x->nb[1]/ts0, x->nb[2]/ts0, x->nb[3]/ts0, eps,
            (const float *) w->data, 0, 0, 0, nc, one, one, one, nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f,
            nullptr, nullptr, (const float *) gate->data)) {
        return true;
    }
    rms_norm_f32<256, true, false, false, false, true><<<blocks_num, 256, 32*sizeof(float), ctx.stream()>>>(
        (const float *) x->data, (float *) out->data, (int) ncols, x->nb[1]/ts0, x->nb[2]/ts0, x->nb[3]/ts0, eps,
        (const float *) w->data, 0, 0, 0, nc, one, one, one,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), 1.0f, 0.0f,
        nullptr, nullptr, (const float *) gate->data);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// p100-series: rms_norm_f32<block_size, true>'s register path (rows of 1024..8192 values, a [ncols] or per-row weight), which
// also writes the q8_1 blocks of its output (flat over the rows) with quantize_q8_1's operations on the stored values: the
// next one-column quantized MUL_MAT needs no quantize launch. Bit-identical to the unfused pair.
template <int block_size>
static __global__ void rms_norm_mul_q8_f32(const float * x, float * dst, const int ncols, const int64_t stride_row, const float eps,
        const float * mul, const int64_t mul_stride_row, block_q8_1 * q8) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    x   += row*stride_row;
    mul += row*mul_stride_row;
    dst += (int64_t) row*ncols;
    constexpr int max_regs = 8;
    extern __shared__ float s_sum[];
    float xv[max_regs];
    float tmp = 0.0f;
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        xv[u] = col < ncols ? x[col] : 0.0f;
        tmp += xv[u] * xv[u];
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int u = 0; u < max_regs; ++u) {
        const int col = tid + u*block_size;
        if (col < ncols) {   // ncols % 32 == 0: whole warps
            const float v = scale * xv[u] * mul[col];
            dst[col] = v;
            float amax = fabsf(v);
            float sum  = v;
            amax = warp_reduce_max<QK8_1>(amax);
            sum  = warp_reduce_sum<QK8_1>(sum);
            const float d = amax / 127.0f;
            const int64_t ic = (int64_t) row*ncols + col;
            q8[ic / QK8_1].qs[ic % QK8_1] = amax == 0.0f ? 0 : roundf(v / d);
            if (ic % QK8_1 == 0) {
                q8[ic / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
}

bool ggml_cuda_op_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * mul_tensor, block_q8_1 * q8) {
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul_tensor->src[0] == rms ? mul_tensor->src[1] : mul_tensor->src[0];
    const int64_t ncols = x->ne[0];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || mul_tensor->type != GGML_TYPE_F32 || ncols < 1024 || ncols > 8192 ||
            ncols % QK8_1 != 0 || x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float) || w->ne[0] != ncols ||
            (ggml_nelements(w) != ncols && ggml_nelements(w) != ncols*x->ne[1]) ||
            !ggml_is_contiguous(w) || !ggml_is_contiguous(mul_tensor) || !ggml_are_same_shape(mul_tensor, rms)) {
        return false;
    }
    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    rms_norm_mul_q8_f32<1024><<<(unsigned) x->ne[1], 1024, 32*sizeof(float), ctx.stream()>>>((const float *) x->data,
        (float *) mul_tensor->data, (int) ncols, (int64_t) (x->nb[1]/sizeof(float)), eps, (const float *) w->data,
        ggml_nelements(w) == ncols ? (int64_t) 0 : (int64_t) ncols, q8);
    return true;
}
