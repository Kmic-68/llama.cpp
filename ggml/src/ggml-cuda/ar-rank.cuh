#pragma once

#include "common.cuh"

// p100-series: per-device two-GPU AllReduce for the meta backend's threaded dispatch (see ar-rank.cu)
struct ggml_cuda_ar_rank_ctx;

ggml_cuda_ar_rank_ctx * ggml_cuda_ar_rank_init(int dev0, int dev1, size_t max_bytes);
// GGML_CUDA_AR_WIRE: 16-bit wire for reductions above 1 MiB (2 = scaled fp16, 1 = bf16, 0 = off: default, exact)
int ggml_cuda_ar_rank_wire();
void                    ggml_cuda_ar_rank_free(ggml_cuda_ar_rank_ctx * c);
bool                    ggml_cuda_ar_rank_eligible(const ggml_cuda_ar_rank_ctx * c, const ggml_tensor * t);
void                    ggml_cuda_ar_rank_reduce(ggml_cuda_ar_rank_ctx * c, int rank, cudaStream_t stream, ggml_tensor * t);

// AllReduce + the RMS norm (and residual ADDs) that open the next subgraph; bit-identical to the separate kernels
struct ggml_cuda_ar_rank_norm_args {
    const float * res   = nullptr; // ADD before the norm: x = AR + res, stored to x_out
    float *       x_out = nullptr;
    const float * w     = nullptr; // MUL weight [ncols]
    const float * add2  = nullptr; // ADD after the MUL
    float *       y     = nullptr; // output of the last fused node
    float         eps   = 0.0f;
};
bool ggml_cuda_ar_rank_norm_eligible(const ggml_cuda_ar_rank_ctx * c, const ggml_tensor * t);
void ggml_cuda_ar_rank_reduce_norm(ggml_cuda_ar_rank_ctx * c, int rank, cudaStream_t stream, ggml_tensor * t, const ggml_cuda_ar_rank_norm_args & a);
