#include "common.cuh"
#include "mmid.cuh"

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
// Every matching (token, slot) gets its own compact row, so a token may list the same expert more than once.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    // Calls f(it, iex, expert_used) on all lanes together for every slot, in token then slot order.
    // The specialized path packs whole tokens into a warp, padding the slot count to a power of 2.
    auto for_each_slot = [&](auto && f) {
        if constexpr (n_expert_used_template == 0) {
            for (int it = 0; it < n_tokens; ++it) {
                for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                    const int iex = iex0 + threadIdx.x;
                    f(it, iex, iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX);
                }
            }
        } else {
            constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
            static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
            for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
                const int it  = it0 + threadIdx.x / neu_padded;
                const int iex = threadIdx.x % neu_padded;
                f(it, iex, (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ? ids[it*si1 + iex] : INT_MAX);
            }
        }
    };

    // Pass 1: number of columns for experts with a lower index.
    int nex_prev = 0;
    for_each_slot([&](int, int, int expert_used) {
        nex_prev += expert_used < expert;
    });
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // Pass 2: write this expert's compact rows.
    int it_compact = 0; // Running index for the compact slice of this expert.
    for_each_slot([&](int it, int iex, int expert_used) {
        const int used = expert_used == expert;
        const auto mask = __ballot_sync(0xFFFFFFFF, used);
        const int  pos  = __popcll(mask & ((uint64_t(1) << threadIdx.x) - 1)); // matches in lower lanes
        if (used) {
            const int itc = nex_prev + it_compact + pos;
            ids_dst[itc] = it*n_expert_used + iex;
            // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = itc;
            } else {
                ids_src1[itc] = it*sis1 + iex % nchannels_y;
            }
        }
        it_compact += __popcll(mask);
    });

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, 0, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

// p100-series: the same compact order by a counting sort, for large batches (prefill: 4096 tokens x 10 slots, 512 experts). The
// warp-per-expert kernel scans every slot twice per expert (~42M reads, ~600 us); here (1) per-chunk counts of each bucket (bucket 0:
// negative ids, which count as below every expert as there; b = e + 1: expert e; ids >= n_experts are ignored as there), (2) a scan
// over the chunks per bucket and over the buckets, (3) each chunk places its slots in order, a warp's 32 slots ranked by comparing
// lanes: a slot's row is bounds[e] + its rank among earlier slots of expert e, exactly the warp-per-expert result.
// GGML_CUDA_MMID_SORT=0: the warp-per-expert kernel
static constexpr int MMIDS_CHUNK = 256;

static __device__ __forceinline__ int mmids_bucket(const int v, const int n_experts) {
    return v < 0 ? 0 : v < n_experts ? v + 1 : -1;
}

static __global__ void mm_ids_count(const int32_t * __restrict__ ids, int32_t * __restrict__ counts, const int n_tokens,
        const int n_used, const int si1, const int n_experts) {
    extern __shared__ int hist_sh[];
    const int nb = n_experts + 1, chunk = blockIdx.x;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) hist_sh[b] = 0;
    __syncthreads();
    const int s0 = chunk*MMIDS_CHUNK, s1 = min(n_tokens*n_used, s0 + MMIDS_CHUNK);
    for (int sl = s0 + threadIdx.x; sl < s1; sl += blockDim.x) {
        const int b = mmids_bucket(ids[(sl / n_used)*si1 + sl % n_used], n_experts);
        if (b >= 0) atomicAdd(&hist_sh[b], 1);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nb; b += blockDim.x) counts[(int64_t) chunk*nb + b] = hist_sh[b];
}

// one block: per bucket, the exclusive prefix over chunks; then the buckets' totals, prefix-summed into each chunk's offsets and the
// expert bounds
static __global__ void mm_ids_scan(int32_t * __restrict__ counts, int32_t * __restrict__ expert_bounds, const int n_chunks,
        const int n_experts) {
    extern __shared__ int tot_sh[];
    const int nb = n_experts + 1;
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        int run = 0;
        for (int c = 0; c < n_chunks; ++c) {
            const int v = counts[(int64_t) c*nb + b];
            counts[(int64_t) c*nb + b] = run;
            run += v;
        }
        tot_sh[b] = run;
    }
    __syncthreads();
    if (threadIdx.x == 0) {   // exclusive scan of the bucket totals (nb <= 1025: serial is fine)
        int run = 0;
        for (int b = 0; b < nb; ++b) {
            const int v = tot_sh[b];
            tot_sh[b] = run;
            run += v;
        }
        for (int e = 0; e < n_experts; ++e) expert_bounds[e] = tot_sh[e + 1];
        expert_bounds[n_experts] = run;
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        const int base = tot_sh[b];
        for (int c = 0; c < n_chunks; ++c) counts[(int64_t) c*nb + b] += base;
    }
}

// a warp per chunk, its slots in order, 32 at a time
static __global__ void mm_ids_place(const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ offs, const int n_tokens, const int n_used, const int nchannels_y, const int si1, const int sis1,
        const int n_experts, const bool write_inverse) {
    const int chunk = blockIdx.x, lane = threadIdx.x, nb = n_experts + 1;
    int32_t * off = offs + (int64_t) chunk*nb;   // this chunk's running position per bucket (scanned offsets, advanced here)
    const int s0 = chunk*MMIDS_CHUNK, s1 = min(n_tokens*n_used, s0 + MMIDS_CHUNK);
    for (int p0 = s0; p0 < s1; p0 += 32) {
        const int sl = p0 + lane;
        const int it = sl / n_used, iex = sl % n_used;
        const int b = sl < s1 ? mmids_bucket(ids[it*si1 + iex], n_experts) : -1;
        int rank = 0, total = 0;
        for (int j = 0; j < 32; ++j) {
            const int bj = __shfl_sync(0xffffffff, b, j);
            rank  += (bj == b && j < lane) ? 1 : 0;
            total += (bj == b) ? 1 : 0;
        }
        const int base = b >= 0 ? off[b] : 0;
        __syncwarp();
        if (b >= 0 && rank == 0) off[b] = base + total;   // the first lane of each bucket advances it
        __syncwarp();
        if (b >= 1) {   // an expert (bucket 0, negative ids, only takes rows)
            const int itc = base + rank;
            ids_dst[itc] = it*n_used + iex;
            if (write_inverse) {
                ids_src1[it*n_used + iex] = itc;
            } else {
                ids_src1[itc] = it*sis1 + iex % nchannels_y;
            }
        }
    }
}

static bool mm_ids_sort_launch(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, const int n_experts,
        const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse,
        cudaStream_t stream) {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_MMID_SORT"); return !e || atoi(e) != 0; }();
    const int64_t n_slots = (int64_t) n_tokens*n_expert_used;
    if (!on || n_slots < 4096 || n_experts > 1024 || n_slots > INT32_MAX/2) {
        return false;
    }
    const int n_chunks = (int) ((n_slots + MMIDS_CHUNK - 1)/MMIDS_CHUNK), nb = n_experts + 1;
    // per-device scratch for the chunk counts, grown on demand (never freed)
    static int32_t * scratch[GGML_CUDA_MAX_DEVICES] = {};
    static size_t    scratch_n[GGML_CUDA_MAX_DEVICES] = {};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    const size_t need = (size_t) n_chunks*nb;
    if (scratch_n[dev] < need) {
        if (scratch[dev]) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(scratch[dev]));
        }
        CUDA_CHECK(cudaMalloc(&scratch[dev], need*sizeof(int32_t)));
        scratch_n[dev] = need;
    }
    int32_t * counts = scratch[dev];
    mm_ids_count<<<n_chunks, 256, nb*sizeof(int), stream>>>(ids, counts, n_tokens, n_expert_used, si1, n_experts);
    mm_ids_scan<<<1, 1024, nb*sizeof(int), stream>>>(counts, expert_bounds, n_chunks, n_experts);
    mm_ids_place<<<n_chunks, 32, 0, stream>>>(ids, ids_src1, ids_dst, counts, n_tokens, n_expert_used, nchannels_y, si1, sis1,
                                              n_experts, write_inverse);
    return true;
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    if (mm_ids_sort_launch(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream)) {
        return;
    }
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
