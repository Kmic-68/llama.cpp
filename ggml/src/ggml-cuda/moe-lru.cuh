#pragma once
// p100-series MoE expert cache (moe-lru), CUDA side: the LRU_DECIDE / LRU_FILL / LRU_ASSEMBLE / LRU_HOST_MERGE ops, copy-engine
// fills, assembly prefetch and the victim stash. Ported from wispoffates/llama.cpp (p100-series) with the 10-31 decode/prefill
// work; how it works: p100-docs/MOE.md.
#include "common.cuh"

struct lru_fill_desc {
    char * src; char * pool;
    size_t src_estride, estride, seg_bytes, n_rows, src_row_stride, src_off;
    int vec;
    int swap; // exclusive cache: the pool slot's victim goes back into the host slot (each device writes its own slice bytes)
};
struct lru_fill_descs { lru_fill_desc d[3]; };

// LRU cleanup of a backend context (its destructor)
void ggml_cuda_lru_ctx_free(ggml_backend_cuda_context & ctx);
void ggml_cuda_lru_stash_flush_ce(ggml_backend_cuda_context & ctx);
void ggml_cuda_lru_stash_worker_stop(ggml_backend_cuda_context & ctx);
void ggml_cuda_lru_ovl_join(ggml_backend_cuda_context & ctx);
bool ggml_cuda_lru_stash_worker_post(ggml_backend_cuda_context & ctx, uint32_t b);
void ggml_cuda_lru_stash_pass(ggml_backend_cuda_context & ctx, int k);
void ggml_cuda_lru_ovl_guard(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_lru_host_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_lru_fill_n(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes, int n);
lru_fill_desc ggml_cuda_lru_fill_desc(const ggml_tensor * dst);
void ggml_cuda_lru_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_lru_drain(ggml_backend_cuda_context & ctx);
bool ggml_cuda_lru_dma_enabled();
void ggml_cuda_lru_decide(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_lru_ce_sync_host(ggml_backend_cuda_context & ctx);
bool ggml_cuda_lru_ce_enabled();
bool ggml_cuda_lru_async_enabled();
void ggml_cuda_lru_assemble_dma(ggml_backend_cuda_context & ctx, ggml_tensor * dst, cudaStream_t copy_stream);
void ggml_cuda_lru_assemble_async(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_lru_assemble(ggml_backend_cuda_context & ctx, ggml_tensor * dst, cudaStream_t stream = nullptr, int gx = 32, int pooled_only = 0);
void ggml_cuda_lru_wait_srcs(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// wait on the main stream for a pool still being filled by the copy engines (data == nullptr: all of them)
void ggml_cuda_lru_ce_pending_wait(ggml_backend_cuda_context & ctx, const void * data);
