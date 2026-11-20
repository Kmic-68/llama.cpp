#include "common.cuh"

bool ggml_cuda_hc_mix_supported(const ggml_tensor * op);
void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
