#include "concat.cuh"
#include "gated_delta_net.cuh"

#include <stdint.h>
#include <limits>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2,
                                                                             uint3     split_fd) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    // split_fd divides by ne0 (dim 0) or by the destination plane (dim 1) with a multiply-shift:
    // sm_60 has no integer divider, so the 64-bit division this used to do per element dominated.
    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = fastdiv((uint32_t) i, split_fd);
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = fastdiv((uint32_t) i, split_fd);
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    // dim 2 does not use fastdiv at all; dims 0 and 1 divide the linear index n
    // by ne0 / (ne0*ne1) respectively, so both numerator and divisor must be in
    // the fastdiv domain. See GGML_CUDA_FASTDIV_MAX in common.cuh.
    if (dim != 2) {
        GGML_ASSERT(n <= int64_t(GGML_CUDA_FASTDIV_MAX));
        GGML_ASSERT((dim == 0 ? ne0 : ne0*ne1) <= int64_t(GGML_CUDA_FASTDIV_MAX));
    }

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2,
            init_fastdiv_values((uint32_t) ne0));
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2,
            init_fastdiv_values((uint32_t) (ne0*ne1)));
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2,
        init_fastdiv_values(1u));
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// Narrow rows along dim 0 (e.g. a conv state of 3 columns joined with a few new tokens, one row
// per channel): the row-per-block kernel above would leave most of each block idle across thousands
// of blocks, so give every thread one output element instead. Same copy, same result.
template <typename T>
static __global__ void concat_non_cont_dim0_flat(
        const char * src0, const char * src1, char * dst, const int64_t n,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const uint64_t nb10, const uint64_t nb11, const uint64_t nb12, const uint64_t nb13,
        const int64_t ne0, const uint64_t nb0, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3,
        const int32_t * src0_rows, const uint64_t src0_row_bytes,
        char * tail, const int64_t tail_i0, const int64_t tail_ne0, const uint64_t tail_nb2) {
    const int64_t idx = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int64_t i0 = idx % ne0;
    int64_t r = idx / ne0;
    const int64_t i1 = r % ne01; r /= ne01;
    const int64_t i2 = r % ne02;
    const int64_t i3 = r / ne02;
    // src0_rows: src0 was a gather of cache rows (one per i2), read in place (ggml_cuda_try_gdn_state_gather)
    const char * s0 = src0_rows ? src0 + (int64_t) src0_rows[i2]*src0_row_bytes : src0 + i3*nb03 + i2*nb02;
    const T * x = i0 < ne00 ? (const T *)(s0 + i1*nb01 + i0*nb00)
                            : (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
    *(T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = *x;
    // tail: the fused CPY of columns [tail_i0, tail_i0 + tail_ne0) (all from src1) to a contiguous
    // [tail_ne0, ne01] block per i2. Thread k writes column k's slot after it read src0 column k, the
    // only element that slot may alias when src0 is the state the tail updates in place.
    if (tail && i0 < tail_ne0) {
        const int64_t j0 = tail_i0 + i0 - ne00;
        *(T *)(tail + i2*tail_nb2 + (i1*tail_ne0 + i0)*sizeof(T)) = *(const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + j0*nb10);
    }
}

// dim-0 concat whose src1 is a transposed view (src1 contiguous along dim 1, e.g. the delta net's conv
// input: 3 state columns + the transposed new tokens). The row-per-block kernel reads src1 with a
// stride of a whole row per thread; a 32x32 shared tile reads src1 along dim 1 and writes dst along
// dim 0, both coalesced. Same copy, same result.
template <typename T>
static __global__ void concat_dim0_tiled(
        const char * src0, const char * src1, char * dst, const int64_t ne00, const int64_t ne0, const int64_t ne1,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb10, const uint64_t nb12,
        const uint64_t nb1, const uint64_t nb2) {
    __shared__ T tile[32][33];
    const int64_t i00 = (int64_t) blockIdx.x*32;
    const int64_t i10 = (int64_t) blockIdx.y*32;
    const int64_t i2  = blockIdx.z;
    // read: threadIdx.x runs along dim 1 (contiguous in src1)
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int64_t i0 = i00 + r;
        const int64_t i1 = i10 + threadIdx.x;
        if (i0 < ne0 && i1 < ne1) {
            tile[r][threadIdx.x] = i0 < ne00 ? *(const T *) (src0 + i2*nb02 + i1*nb01 + i0*nb00)
                                             : *(const T *) (src1 + i2*nb12 + i1*sizeof(T) + (i0 - ne00)*nb10);
        }
    }
    __syncthreads();
    // write: threadIdx.x runs along dim 0 (contiguous in dst)
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const int64_t i0 = i00 + threadIdx.x;
        const int64_t i1 = i10 + r;
        if (i0 < ne0 && i1 < ne1) {
            *(T *) (dst + i2*nb2 + i1*nb1 + i0*sizeof(T)) = tile[threadIdx.x][r];
        }
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream,
                        const ggml_tensor * tail = nullptr) {
    // src0 gathered in place (its GET_ROWS was skipped): only the flat kernel below handles that
    const float *   g_base = nullptr;
    const int32_t * g_idx  = nullptr;
    int64_t         g_row  = 0;
    const bool gathered = ggml_cuda_gdn_gather_lookup(dst, &g_base, &g_idx, &g_row);
    GGML_ASSERT(!gathered || (dim == 0 && dst->ne[0] <= 64));
    GGML_ASSERT(!tail || (dim == 0 && dst->ne[0] <= 64));
    static const bool tiled_ok = [] { const char * e = getenv("GGML_CUDA_CONCAT_TILED"); return !e || atoi(e) != 0; }();
    if (!tail && !gathered && dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (!tail && !gathered && dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else if (tiled_ok && !tail && !gathered && dim == 0 && dst->ne[0] > 64 && !ggml_is_quantized(src0->type) &&
               src1->nb[1] == sizeof(T) && src1->nb[0] > src1->nb[1] && dst->nb[0] == sizeof(T) &&
               src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1) {
        const dim3 grid((dst->ne[0] + 31)/32, (dst->ne[1] + 31)/32, dst->ne[2]);
        concat_dim0_tiled<T><<<grid, dim3(32, 8, 1), 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (char *) dst->data, src0->ne[0], dst->ne[0], dst->ne[1],
            src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[0], src1->nb[2], dst->nb[1], dst->nb[2]);
    } else if (dim == 0 && dst->ne[0] <= 64 && !ggml_is_quantized(src0->type)) {
        const int64_t n = ggml_nelements(dst);
        const int nblk = (int) ((n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE);
        concat_non_cont_dim0_flat<T><<<nblk, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
            gathered ? (const char *) g_base : (const char *) src0->data, (const char *) src1->data, (char *) dst->data, n,
            src0->ne[0], dst->ne[1], dst->ne[2],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            dst->ne[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
            gathered ? g_idx : nullptr, (uint64_t) g_row*sizeof(float),
            tail ? (char *) tail->data : nullptr, tail ? tail->src[0]->view_offs/dst->nb[0] : 0, tail ? tail->src[0]->ne[0] : 0,
            tail ? tail->nb[1] : 0);
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
        }
    }
}

// CONCAT (dim 0, narrow) -> CPY of its last columns (the conv state update): one launch, the same copies
bool ggml_cuda_op_concat_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * cpy) {
    const ggml_tensor * src0 = dst->src[0], * src1 = dst->src[1], * v = cpy->src[0], * cd = cpy->src[1];
    const int32_t dim = ((int32_t *) dst->op_params)[0];
    if (dim != 0 || dst->ne[0] > 64 || dst->type != GGML_TYPE_F32 || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
            v->view_src != dst || v->type != dst->type || cpy->type != dst->type || cd->type != dst->type ||
            dst->ne[3] != 1 || v->nb[1] != dst->nb[1] || v->nb[2] != dst->nb[2] || v->nb[0] != dst->nb[0] ||
            dst->nb[0] != sizeof(float) || v->ne[1] != dst->ne[1] || v->ne[2] != dst->ne[2] || v->ne[3] != 1 ||
            v->view_offs % dst->nb[1] + v->ne[0]*dst->nb[0] != (size_t) dst->ne[0]*dst->nb[0] || v->view_offs >= dst->nb[1] ||
            (int64_t) (v->view_offs/dst->nb[0]) < src0->ne[0] || v->ne[0] > src0->ne[0] ||
            cpy->ne[0] != v->ne[0]*v->ne[1] || cpy->ne[1] != v->ne[2] || cpy->nb[0] != sizeof(float) || cpy->ne[2] != 1 || cpy->ne[3] != 1) {
        return false;
    }
    concat_cuda<uint32_t>(src0, src1, dst, 0, ctx.stream(), cpy);
    return true;
}
