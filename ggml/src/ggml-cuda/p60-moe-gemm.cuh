#pragma once

#include "common.cuh"

// MUL_MAT_ID of q4_K / q5_K experts for prompt processing with fp16 math (p60-moe-gemm.cu). Returns false if not
// applicable. Needs GGML_CUDA_P60_MEGA=1; GGML_CUDA_P60_MOE_GEMM=0 disables it.
bool ggml_cuda_p60_moe_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
    const ggml_tensor * ids, ggml_tensor * dst);
