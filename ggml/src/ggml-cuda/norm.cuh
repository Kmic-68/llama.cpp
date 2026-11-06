#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

bool ggml_cuda_op_rms_norm_scale2(ggml_backend_cuda_context & ctx, ggml_tensor * n1, ggml_tensor * s1, ggml_tensor * n2, ggml_tensor * s2);

bool ggml_cuda_op_rms_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// ADD -> RMS_NORM(ADD) -> MUL in one kernel (bit-identical); false if the tensors don't qualify
bool ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * rms, ggml_tensor * mul);

bool ggml_cuda_op_rms_norm_mul_silu_gate(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * mul,
                                         const ggml_tensor * gate, ggml_tensor * out, bool check_only);

bool ggml_cuda_op_rms_norm_mul_q8(ggml_backend_cuda_context & ctx, ggml_tensor * rms, ggml_tensor * mul_tensor, block_q8_1 * q8);
