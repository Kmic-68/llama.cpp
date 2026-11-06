#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Pascal: 17..GGML_CUDA_MMVQ_CHUNK_MAX (default 28) columns of q6_K/q5_K weights as fp16 mat-vec column chunks
bool ggml_cuda_mmvq_chunked_ok(int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mul_mat_vec_q_chunked(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// strided (view) quantized src0 without a non-contiguous converter: mat-vec column chunks instead of cuBLAS
bool ggml_cuda_mul_mat_vec_q_chunked_strided(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cuda_q8_smallk_scale_silu(ggml_backend_cuda_context & ctx, const ggml_tensor * sc, const ggml_tensor * silu, ggml_tensor * mm);

// p100-series: hand a SWIGLU (dst = the next MUL_MAT_ID's src1) to mmvq, which applies it in the IQ4_NL short-row kernel or
// computes it first; take_glu_consumed() is false when mmvq never ran (the caller must run the SWIGLU and redo the matmul)
void ggml_cuda_mmvq_set_glu(ggml_tensor * glu);
bool ggml_cuda_mmvq_take_glu_consumed();

block_q8_1 * ggml_cuda_mmvq_q8_1_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * src1);
void ggml_cuda_mmvq_q8_1_mark(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_type type_src0);
