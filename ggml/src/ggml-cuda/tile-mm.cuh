#pragma once

#include "common.cuh"

// Fused quantized x f32 tile GEMM for Pascal at 9..80 columns (n-gram / draft verify batches, short prompts), in place of
// dequantize -> fp16 -> cuBLAS hgemm. Handles q6_K, q5_K, q4_K, q8_0 and q4_0. Returns false if it does not handle the shape. See tile-mm.cu.
bool ggml_cuda_tile_mm_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Grouped MUL_MAT_ID version (MoE prefill on Pascal, in place of MMQ's emulated-DP4A path): one launch over all experts, token lists
// sorted on the device. Returns false if it does not handle the shape.
bool ggml_cuda_tile_mm_id_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                              ggml_tensor * dst);
