#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    // p100-series: the bin walk in one warp (a serial walk of 256 bins in thread 0 cost ~14 us per pass). Same result as
    //   bin = NBINS - 1; while (bin > 0 && histogram[bin] < rank) rank -= histogram[bin--];
    // lane l holds bins [l*BPL, l*BPL + BPL); the walk stops in the highest lane whose bins and all above reach rank
    // (lane 0 when none does: the walk then ends at bin 0)
    static_assert(NBINS % WARP_SIZE == 0, "bins per lane");
    constexpr int BPL = NBINS / WARP_SIZE;
    if (tid < WARP_SIZE) {
        const int lane = tid;
        top_k_radix_state state = states[row];
        int c[BPL];
        int sum = 0;
#pragma unroll
        for (int j = 0; j < BPL; ++j) {
            c[j] = histogram[lane*BPL + j];
            sum += c[j];
        }
        int suf = sum;   // bins >= lane*BPL
#pragma unroll
        for (int off = 1; off < WARP_SIZE; off *= 2) {
            const int v = __shfl_down_sync(0xffffffffu, suf, off);
            if (lane + off < WARP_SIZE) {
                suf += v;
            }
        }
        const unsigned reach = __ballot_sync(0xffffffffu, suf >= state.rank);
        const int target = reach ? 31 - __clz((int) reach) : 0;
        if (lane == target) {
            int rank = state.rank - (suf - sum);   // minus the bins above this lane
            int bin = lane*BPL + BPL - 1;
#pragma unroll
            for (int j = BPL - 1; j >= 0; --j) {
                bin = lane*BPL + j;
                if (bin == 0 || c[j] >= rank) {
                    break;
                }
                rank -= c[j];
            }
            state.rank = rank;
            state.prefix |= (uint32_t) bin << shift;
            state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
            states[row] = state;
        }
    }
}

// p100-series: histogram + select of one pass in one launch (the last block of a row to finish does the select), the state
// initialised by the first pass and the gather counters cleared by the last: 6 launches per top-k instead of 12. Same bins,
// same walk, same state, so the same result. done[] holds one counter per row, zero between launches (the last block resets it).
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_hist_select(
        const float * __restrict__ src, top_k_radix_state * __restrict__ states, int * __restrict__ block_histograms,
        int * __restrict__ done, int ncols, int blocks_per_row, int shift, int k, bool first, bool last_pass) {
    constexpr int NBINS = 1 << RADIX_BITS;
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];
    __shared__ bool is_last;

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = first ? top_k_radix_state{0, 0, k, 0, 0} : states[row];
    for (int64_t col = row_block * BLOCK_SIZE + tid; col < ncols; col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();
    block_histograms[((size_t) row * blocks_per_row + row_block) * NBINS + tid] = histogram[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        is_last = atomicAdd(&done[row], 1) == blocks_per_row - 1;
    }
    __syncthreads();
    if (!is_last) {
        return;
    }
    __threadfence();
    int count = 0;
    for (int rb = 0; rb < blocks_per_row; ++rb) {
        count += ((volatile int *) block_histograms)[((size_t) row * blocks_per_row + rb) * NBINS + tid];
    }
    histogram[tid] = count;
    __syncthreads();
    static_assert(NBINS % WARP_SIZE == 0, "bins per lane");
    constexpr int BPL = NBINS / WARP_SIZE;
    if (tid < WARP_SIZE) {
        const int lane = tid;
        top_k_radix_state st = state;
        int c[BPL];
        int sum = 0;
#pragma unroll
        for (int j = 0; j < BPL; ++j) {
            c[j] = histogram[lane*BPL + j];
            sum += c[j];
        }
        int suf = sum;
#pragma unroll
        for (int off = 1; off < WARP_SIZE; off *= 2) {
            const int v = __shfl_down_sync(0xffffffffu, suf, off);
            if (lane + off < WARP_SIZE) {
                suf += v;
            }
        }
        const unsigned reach = __ballot_sync(0xffffffffu, suf >= st.rank);
        const int target = reach ? 31 - __clz((int) reach) : 0;
        if (lane == target) {
            int rank = st.rank - (suf - sum);
            int bin = lane*BPL + BPL - 1;
#pragma unroll
            for (int j = BPL - 1; j >= 0; --j) {
                bin = lane*BPL + j;
                if (bin == 0 || c[j] >= rank) {
                    break;
                }
                rank -= c[j];
            }
            st.rank = rank;
            st.prefix |= (uint32_t) bin << shift;
            st.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
            if (last_pass) {
                st.greater_count = 0;
                st.equal_count = 0;
            }
            states[row] = st;
            done[row] = 0;
        }
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int64_t col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        }
        // ties at the threshold: top_k_radix_ties, in index order (an atomic slot here picked a timing-dependent subset of them;
        // qwen4exp's QSA indexer scores are rectified sums with many exact zeros, so long-context prefill was not reproducible)
    }
}

// p100-series: the threshold ties of each row, lowest column first, one block per row (deterministic)
template <int BLOCK_SIZE>
static __global__ void top_k_radix_ties(const float * __restrict__ src, int * __restrict__ dst, const top_k_radix_state * __restrict__ states,
                                        int ncols, int k) {
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid % WARP_SIZE, wid = tid / WARP_SIZE;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const top_k_radix_state st = states[row];
    __shared__ int wsum[BLOCK_SIZE / WARP_SIZE];
    int found = 0;
    for (int base = 0; base < ncols && found < st.rank; base += BLOCK_SIZE) {
        const int col = base + tid;
        const bool eq = col < ncols && top_k_float_to_ordered(row_src[col]) == st.prefix;
        const unsigned m = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) {
            wsum[wid] = __popc(m);
        }
        __syncthreads();
        int before = 0, total = 0;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; w++) {
            before += w < wid ? wsum[w] : 0;
            total  += wsum[w];
        }
        const int pos = found + before + __popc(m & ((1u << lane) - 1u));
        if (eq && pos < st.rank) {
            row_dst[k - st.rank + pos] = col;
        }
        found += total;
        __syncthreads();
    }
}

bool ggml_cuda_ab_alt();
#define TOPK_FUSED_MAX_ROWS 64
static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    // the fused passes' per-row done counters: zeroed once per device, every launch leaves them zero (GGML_CUDA_TOPK_FUSED=0: off)
    static const bool fused_env = [] { const char * e = getenv("GGML_CUDA_TOPK_FUSED"); return !e || atoi(e) != 0; }();
    static int * done_dev[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = ggml_cuda_get_device();
    if (fused_env && done_dev[dev] == nullptr) {
        CUDA_CHECK(cudaMalloc(&done_dev[dev], TOPK_FUSED_MAX_ROWS*sizeof(int)));
        CUDA_CHECK(cudaMemset(done_dev[dev], 0, TOPK_FUSED_MAX_ROWS*sizeof(int)));
    }
    const bool fused = fused_env && !ggml_cuda_ab_alt();
    int * done = done_dev[dev];

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        if (fused && iter_nrows <= TOPK_FUSED_MAX_ROWS) {
            for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
                top_k_radix_hist_select<BLOCK_SIZE, RADIX_BITS><<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, done, ncols, blocks_per_row, shift, k, shift == 32 - RADIX_BITS, shift == 0);
            }
        } else {
        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_reset_counters
            <<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows);
        }
        top_k_radix_gather<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, ncols, k, blocks_per_row);
        top_k_radix_ties<BLOCK_SIZE><<<iter_nrows, BLOCK_SIZE, 0, stream>>>(src, dst, states, ncols, k);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
