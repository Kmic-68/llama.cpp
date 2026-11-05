#pragma once

// p100-series: fused QSA gather selection (qwen4exp). A mode of GGML_OP_TOP_K_MASK (op_params[0] = GGML_QSA_SEL_MAGIC), declared
// here so that adding it does not touch ggml.h. One op builds what the gather path's ~20 small ops built per layer:
//   sorted = the picked pools of each token (top_k, [n_top, nt], any order) in ascending order
//   cells  = pool_idxs[:, sorted] (kpool cells per pool), then the tail cells tail[:, t]  -> n_sel = kpool*n_top + n_tail
//   live   = clamp(score[pool] + 1, 0, 1) for a pool cell, clamp(n_kv - cell, 0, 1) for a tail cell, 0 for the padding to n_gpad
//   mode 0: I32 [n_gpad, nt]           = (int) ((float) cell * live)
//   mode 1: mtype [n_gpad, 1, 1, nt]   = (mtype) logf(live)
// The same float operations in the same order as the unfused graph, so the results are identical.

#include "ggml.h"

#define GGML_QSA_SEL_MAGIC 0x47415351  // "QSAG"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API struct ggml_tensor * ggml_qsa_gather_sel(
        struct ggml_context * ctx,
        struct ggml_tensor  * top_k,      // I32 [n_top, nt]
        struct ggml_tensor  * score,      // F32 [n_pool, nt]
        struct ggml_tensor  * pool_idxs,  // I32 [kpool, n_pool]
        struct ggml_tensor  * tail,       // I32 [n_tail, nt]
        int                   n_kv,
        int                   n_gpad,
        int                   mode,       // 0: gather indices (I32), 1: mask (mtype)
        enum ggml_type        mtype);

#ifdef __cplusplus
}
#endif
