#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data        = nullptr; // rollback slot 0 (nullptr: state goes to the dst tail)
    int64_t slot_stride = 0;       // between rollback slots (0 when K==1)

    // optional input: read the state of sequence s from row state_ids[s] of state_src (nullptr: use src[5])
    const float * state_src        = nullptr;
    const int *   state_ids        = nullptr;
    int64_t       state_row_stride = 0; // in floats

    // optional pre-op done in the kernel (S_v = 128): q, k before l2_scale * rms_norm(x, l2_eps)
    const ggml_tensor * q        = nullptr;
    const ggml_tensor * k        = nullptr;
    float               l2_eps   = 0.0f;
    float               l2_scale = 0.0f;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Recurrent-state gather fused into the kernel. build_rs gathers each sequence's state from its cache
// slot with a GET_ROWS (a ~3 MB copy per layer for this model) whose only consumer is the gated delta
// net. The graph executor skips such a GET_ROWS and registers it here; the kernel then reads the state
// in place through the row index (on the device, no sync). See ggml_cuda_try_gdn_state_gather.
void ggml_cuda_gdn_gather_reset();
void ggml_cuda_gdn_gather_register(const ggml_tensor * gdn, const float * base, const int32_t * idx, int64_t row);
bool ggml_cuda_gdn_gather_lookup(const ggml_tensor * consumer, const float ** base, const int32_t ** idx, int64_t * row);
