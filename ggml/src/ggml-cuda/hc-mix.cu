// p100-series: the hyper-connection read (Qwen3.8-Flash-Next build_hc_mix, GGML_OP_HC_MIX) in three launches, after
// Megla123's Flash-Next engine (fnx/engine/src/hc.cuh), reimplemented for ggml's q8_0 layout:
//   k_hc_mix_down: block (column slice s, row group g): x = R * w_norm over the slice, its square sums, x as int8 per 32;
//                  the group's down rows (q8_0) and, in the last group, the inject rows (f32) against the slice -> partials.
//   k_hc_mix_up:   block = HM_UPC columns of every stream: the up rows (q8_0) . lo, then mean_c xn_c * sigmoid(gate_c).
//   k_hc_mix_reduce: block t: rs per stream, lo = silu(sum_c rs_c * partial_c / hc) as q8_1, the inject rows.
// The rms factor is applied after the dots (rms_norm is a per-stream scalar), so x is quantized before it; the sums run in
// fp32 in a fixed order (deterministic; the order differs from the unfused graph's).

#include "hc-mix.cuh"
#include "unary.cuh"

static constexpr int HM_HC    = 4;     // streams
static constexpr int HM_COLS  = 256;   // columns a down slice (8 q8 blocks)
static constexpr int HM_NG    = 4;     // down row groups
static constexpr int HM_MAXLR = 320;   // low rank: at most 10 q8 blocks (K of the up rows)
static constexpr int HM_UPC   = 24;    // columns a up block (x 4 streams = 96 rows)
static constexpr int HM_MAXT  = 8;
static constexpr int HM_SPS_MAX = 16;  // down slices a stream: n <= 4096

// a q8_0 block's scale and its 8 int words (34-byte blocks: 2-byte aligned)
static __device__ __forceinline__ float hm_ld_q8_0(const char * p, int (&q)[8]) {
    const size_t o = (size_t) p;
    const uint32_t * w = (const uint32_t *) (o & ~(size_t) 3);
    uint32_t v[9];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        v[i] = __ldg(w + i);
    }
    if ((o & 3) == 0) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            q[j] = (int) __funnelshift_r(v[j], v[j + 1], 16);
        }
        return __half2float(__ushort_as_half((uint16_t) (v[0] & 0xffff)));
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        q[j] = (int) v[j + 1];
    }
    return __half2float(__ushort_as_half((uint16_t) (v[0] >> 16)));
}

template <int T>
static __global__ void __launch_bounds__(256) k_hc_mix_down(
        const float * __restrict__ R, const float * __restrict__ wn, const char * __restrict__ wdown, const size_t nb_down,
        const float * __restrict__ winj, const int n, const int lr, float * __restrict__ part, float * __restrict__ ssq) {
    const int d = n*HM_HC;
    const int nrows = lr + (winj ? HM_HC : 0), rpg = (nrows + HM_NG - 1)/HM_NG;
    const int s = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ float  xs[T][HM_COLS];
    __shared__ int8_t xq[T][HM_COLS];
    __shared__ float  xd[T][HM_COLS/32];
    __shared__ float  red[T][8];
    const int col = s*HM_COLS + tid;
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float r = R[(size_t) t*d + col];
        const float x = r*wn[col];
        xs[t][tid] = x;
        const float amax = warp_reduce_max(fabsf(x));
        const float dq = amax/127.0f;
        xq[t][tid] = amax == 0.0f ? 0 : (int8_t) roundf(x/dq);
        if (lane == 0) {
            xd[t][warp] = dq;
        }
        if (g == 0) {
            const float v = warp_reduce_sum(r*r);
            if (lane == 0) {
                red[t][warp] = v;
            }
        }
    }
    __syncthreads();
    if (g == 0 && tid < T) {
        float v = 0.0f;
        for (int w = 0; w < 8; ++w) {
            v += red[tid][w];
        }
        ssq[s*T + tid] = v;
    }
    // 8 lanes a row (lane j: the slice's q8 block j), 4 rows a warp, 32 rows a pass
    const int j = lane & 7, rw = lane >> 3;
    const int r0 = g*rpg, r1 = min(nrows, r0 + rpg);
    for (int rb = r0 + warp*4; rb < r1; rb += 32) {
        const int row = rb + rw;
        const bool live = row < r1;
        float acc[T];
        if (live && row < lr) {
            int q[8];
            const float dw = hm_ld_q8_0(wdown + (size_t) row*nb_down + (size_t) (s*(HM_COLS/QK8_0) + j)*sizeof(block_q8_0), q);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int * xi = (const int *) &xq[t][j*32];
                int si = 0;
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    si = ggml_cuda_dp4a(q[u], xi[u], si);
                }
                acc[t] = dw*xd[t][j]*(float) si;
            }
        } else if (live) {
            const float4 * w4 = (const float4 *) (winj + (size_t) (row - lr)*d + s*HM_COLS + j*32);
            float4 w[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                w[u] = __ldg(w4 + u);
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                float a = 0.0f;
                const float * x = &xs[t][j*32];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    a += w[u].x*x[4*u] + w[u].y*x[4*u + 1] + w[u].z*x[4*u + 2] + w[u].w*x[4*u + 3];
                }
                acc[t] = a;
            }
        } else {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[t] = 0.0f;
            }
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float v = acc[t];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            if (live && j == 0) {
                part[((size_t) s*T + t)*nrows + row] = v;
            }
        }
    }
}

// block t (a window row), a thread per row of partials (each 32-thread warp = one q8_1 block of lo): rs per stream, lo, inject
template <int T, int SPS>
static __global__ void __launch_bounds__(HM_MAXLR + 32) k_hc_mix_reduce(
        const float * __restrict__ part, const float * __restrict__ ssq, const int n, const int lr, const bool has_inj,
        const float eps, block_q8_1 * __restrict__ loq, float * __restrict__ rs_out, float * __restrict__ dst) {
    const int ns = n*HM_HC/HM_COLS, sps = SPS ? SPS : ns/HM_HC, nrows = lr + (has_inj ? HM_HC : 0);
    const int t = blockIdx.x, r = threadIdx.x, lane = r & 31;
    __shared__ float rs[HM_HC];
    if (r < HM_HC) {
        float v = 0.0f;
#pragma unroll
        for (int q = 0; q < HM_SPS_MAX; ++q) {
            if (q < sps) {
                v += ssq[(r*sps + q)*T + t];
            }
        }
        rs[r] = rsqrtf(v/(float) n + eps);
        rs_out[t*HM_HC + r] = rs[r];
    }
    __syncthreads();
    float sum = 0.0f;
    if (r < nrows) {
#pragma unroll
        for (int c = 0; c < HM_HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int q = 0; q < HM_SPS_MAX; ++q) {
                if (q < sps) {
                    p += part[((size_t) (c*sps + q)*T + t)*nrows + r];
                }
            }
            sum = fmaf(rs[c], p, sum);
        }
    }
    if (r - lane < lr) {   // warp-uniform: lr is a multiple of 32
        const float xi   = ggml_cuda_op_silu_single(sum/(float) HM_HC);
        const float amax = warp_reduce_max(fabsf(xi));
        const float sm   = warp_reduce_sum(xi);
        const float dq   = amax/127.0f;
        block_q8_1 & b = loq[t*(lr/32) + r/32];
        b.qs[lane] = amax == 0.0f ? 0 : (int8_t) roundf(xi/dq);
        if (lane == 0) {
            b.ds = make_half2(dq, sm);
        }
    } else if (r >= lr && r < lr + HM_HC) {
        dst[(size_t) T*n + t*HM_HC + (r - lr)] = has_inj ? sum : 0.0f;
    }
}

template <int T>
static __global__ void __launch_bounds__(256) k_hc_mix_up(
        const float * __restrict__ R, const float * __restrict__ wn, const char * __restrict__ wup, const size_t nb_up,
        const block_q8_1 * __restrict__ loq, const float * __restrict__ rsv, const int n, const int lr, float * __restrict__ dst) {
    const int d = n*HM_HC, nb = lr/QK8_0, rpw = 32/nb, nrow = HM_UPC*HM_HC;
    const int i0 = blockIdx.x*HM_UPC, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    __shared__ block_q8_1 ys[T][HM_MAXLR/QK8_0];
    __shared__ float gs[T][HM_HC][HM_UPC];
    for (int e = tid; e < T*nb*(int) (sizeof(block_q8_1)/4); e += blockDim.x) {
        ((uint32_t *) ys)[(e/(nb*9))*(HM_MAXLR/QK8_0)*9 + e%(nb*9)] = ((const uint32_t *) loq)[e];
    }
    __syncthreads();
    const int rr = lane/nb, b = lane - rr*nb;
    for (int p = warp*rpw; p < nrow; p += 8*rpw) {
        const int lr_ = p + rr;   // stream c = lr_ / HM_UPC, column i0 + lr_ % HM_UPC
        const int c = lr_/HM_UPC, ci = lr_%HM_UPC;
        const bool act = rr < rpw && lr_ < nrow && i0 + ci < n;
        float dw = 0.0f;
        int q[8] = {};
        if (act) {
            dw = hm_ld_q8_0(wup + ((size_t) c*n + i0 + ci)*nb_up + (size_t) b*sizeof(block_q8_0), q);
        }
        const int src0 = (rr < rpw ? rr : 0)*nb;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int * yi = (const int *) ys[t][act ? b : 0].qs;
            int si = 0;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                si = ggml_cuda_dp4a(q[u], yi[u], si);
            }
            const float v = act ? dw*__low2float(ys[t][b].ds)*(float) si : 0.0f;
            float acc = 0.0f;   // the row's nb lanes in block order
            for (int bb = 0; bb < nb; ++bb) {
                acc += __shfl_sync(0xffffffffu, v, src0 + bb);
            }
            if (act && b == 0) {
                gs[t][c][ci] = acc;
            }
        }
    }
    __syncthreads();
    for (int e = tid; e < T*HM_UPC; e += blockDim.x) {
        const int t = e/HM_UPC, ci = e%HM_UPC, i = i0 + ci;
        if (i >= n) {
            continue;
        }
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HM_HC; ++c) {
            const float xn = R[(size_t) t*d + c*n + i]*rsv[t*HM_HC + c]*wn[c*n + i];
            s += xn*(1.0f/(1.0f + expf(-gs[t][c][ci])));
        }
        dst[(size_t) t*n + i] = s*(1.0f/(float) HM_HC);
    }
}

bool ggml_cuda_hc_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0], * wn = op->src[1], * wd = op->src[2], * wu = op->src[3], * wi = op->src[4];
    const int64_t n = x->ne[0], lr = wd->ne[1];
    return x->type == GGML_TYPE_F32 && wn->type == GGML_TYPE_F32 && wd->type == GGML_TYPE_Q8_0 && wu->type == GGML_TYPE_Q8_0 &&
        (wi == nullptr || (wi->type == GGML_TYPE_F32 && ggml_is_contiguous(wi))) && op->type == GGML_TYPE_F32 &&
        x->ne[1] == HM_HC && n % HM_COLS == 0 && n/HM_COLS <= HM_SPS_MAX && x->ne[2] >= 1 && x->ne[2] <= HM_MAXT && lr % 32 == 0 && lr <= HM_MAXLR &&
        ggml_is_contiguous(x) && ggml_is_contiguous(wn) && ggml_is_contiguous(op) &&
        wd->nb[0] == sizeof(block_q8_0) && wu->nb[0] == sizeof(block_q8_0) && wd->ne[2] == 1 && wu->ne[2] == 1;
}

template <int T>
static void hc_mix_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0], * wn = dst->src[1], * wd = dst->src[2], * wu = dst->src[3], * wi = dst->src[4];
    const int n = (int) x->ne[0], lr = (int) wd->ne[1], d = n*HM_HC, ns = d/HM_COLS;
    const int nrows = lr + (wi ? HM_HC : 0);
    const float eps = ggml_get_op_params_f32(dst, 0);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float>      part(ctx.pool(), (size_t) ns*T*nrows);
    ggml_cuda_pool_alloc<float>      ssq(ctx.pool(), (size_t) ns*T);
    ggml_cuda_pool_alloc<block_q8_1> loq(ctx.pool(), (size_t) T*(lr/32));
    ggml_cuda_pool_alloc<float>      rs(ctx.pool(), (size_t) T*HM_HC);
    float * out = (float *) dst->data;
    k_hc_mix_down<T><<<dim3(ns, HM_NG), 256, 0, stream>>>((const float *) x->data, (const float *) wn->data, (const char *) wd->data,
        wd->nb[1], wi ? (const float *) wi->data : nullptr, n, lr, part.get(), ssq.get());
    auto * reduce = ns/HM_HC == 10 ? k_hc_mix_reduce<T, 10> : k_hc_mix_reduce<T, 0>;
    reduce<<<T, (lr + HM_HC + 31)/32*32, 0, stream>>>(   // the inject rows are written (0 without w_inject) either way
        part.get(), ssq.get(), n, lr, wi != nullptr, eps, loq.get(), rs.get(), out);
    k_hc_mix_up<T><<<(n + HM_UPC - 1)/HM_UPC, 256, 0, stream>>>((const float *) x->data, (const float *) wn->data, (const char *) wu->data,
        wu->nb[1], loq.get(), rs.get(), n, lr, out);
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[0]->ne[2]) {
        case 1: hc_mix_launch<1>(ctx, dst); break;
        case 2: hc_mix_launch<2>(ctx, dst); break;
        case 3: hc_mix_launch<3>(ctx, dst); break;
        case 4: hc_mix_launch<4>(ctx, dst); break;
        case 5: hc_mix_launch<5>(ctx, dst); break;
        case 6: hc_mix_launch<6>(ctx, dst); break;
        case 7: hc_mix_launch<7>(ctx, dst); break;
        case 8: hc_mix_launch<8>(ctx, dst); break;
        default: GGML_ABORT("hc_mix: 1 to %d tokens", HM_MAXT);
    }
}
