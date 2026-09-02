#pragma once

#include "common.cuh"

// cuBLAS-GEMM flash attention for pre-Volta NVIDIA. On by default there; GGML_CUDA_FA_GEMM=0 disables.
// See fattn-gemm.cu for why this path exists.
bool ggml_cuda_fa_gemm_enabled();
bool ggml_cuda_flash_attn_ext_gemm_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
