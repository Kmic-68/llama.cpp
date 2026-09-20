// Does the half2 KQ accumulation (7c77a2b80) round more than the code it replaced?
//
// UPSTREAM  ggml_cuda_mad(float&, half2, half2) under FAST_FP16_AVAILABLE:
//               half2 p = v*u;                      // each product ROUNDED to fp16
//               acc += float(p.x) + float(p.y);     // fp32 accumulate
// NEW       s = __hfma2(K,Q,s) over the cpy_ne group, then acc += float(s.x)+float(s.y):
//               products FUSED (not rounded), partial sum of the group kept in fp16.
//
// On sm_60 ggml_cuda_get_max_cpy_bytes()==8 so cpy_ne==2: the fp16 group is TWO terms per
// lane, not eight. Per lane the two schemes are
//     upstream: fl16(a) + fl16(b)        summed in fp32   -> 2 fp16 roundings
//     new     : fl16(b + fl16(a))                         -> 2 fp16 roundings
// i.e. the same NUMBER of fp16 roundings, differently placed: the new form trades rounding
// the second product for rounding the pair's sum, and fuses the second product exactly.
//
// This measures the full DKQ=256 dot product both ways against a double-precision reference
// over the identical fp16 inputs, so the only thing being compared is the accumulation order.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <curand_kernel.h>

#define DKQ     256
#define CPY_NE  2          // sm_60
#define NTRIAL  (1<<20)

struct Acc { double se_up, se_new, max_up, max_new; unsigned long long new_better, up_better, tie, of; double max_abs_s; };

__device__ double d_se_up, d_se_new, d_max_up, d_max_new, d_max_abs_s;
__device__ unsigned long long d_new_better, d_up_better, d_tie, d_overflow;

__device__ __forceinline__ double dot_ref(const half2 *K, const half2 *Q, int n2) {
    double s = 0.0;
    for (int i = 0; i < n2; ++i) {
        s += (double) __low2float(K[i])  * (double) __low2float(Q[i]);
        s += (double) __high2float(K[i]) * (double) __high2float(Q[i]);
    }
    return s;
}

// upstream: product rounded to fp16, fp32 accumulate
__device__ __forceinline__ float dot_upstream(const half2 *K, const half2 *Q, int n2) {
    float acc = 0.0f;
    for (int g = 0; g < n2; g += CPY_NE) {
        for (int k = 0; k < CPY_NE; ++k) {
            const half2  p   = __hmul2(K[g+k], Q[g+k]);
            const float2 tmp = __half22float2(p);
            acc += tmp.x + tmp.y;
        }
    }
    return acc;
}

// new: fused products, fp16 partial sum within the cpy_ne group, widen once
__device__ __forceinline__ float dot_new(const half2 *K, const half2 *Q, int n2, float *max_abs_s) {
    float acc = 0.0f;
    for (int g = 0; g < n2; g += CPY_NE) {
        half2 s = make_half2(0.0f, 0.0f);
        for (int k = 0; k < CPY_NE; ++k) {
            s = __hfma2(K[g+k], Q[g+k], s);
        }
        const float sx = __low2float(s), sy = __high2float(s);
        const float m = fmaxf(fabsf(sx), fabsf(sy));
        if (m > *max_abs_s) *max_abs_s = m;
        acc += sx + sy;
    }
    return acc;
}

// dist: 0 = iid gaussian, 1 = gaussian with rare large outliers (attention sinks),
//       2 = heavy cancellation (K and Q nearly antiparallel in pairs)
__global__ void trial(int dist, float sigK, float sigQ, unsigned seed) {
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= NTRIAL) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, tid, 0, &st);

    half2 K[DKQ/2], Q[DKQ/2];
    for (int i = 0; i < DKQ/2; ++i) {
        float k0 = curand_normal(&st)*sigK, k1 = curand_normal(&st)*sigK;
        float q0 = curand_normal(&st)*sigQ, q1 = curand_normal(&st)*sigQ;
        if (dist == 1) {
            if (curand_uniform(&st) < 0.02f) { k0 *= 16.0f; q0 *= 16.0f; }
            if (curand_uniform(&st) < 0.02f) { k1 *= 16.0f; q1 *= 16.0f; }
        } else if (dist == 2) {
            // make the two terms of each group nearly cancel: b ~ -a
            q1 = -q0 * (k0/ (k1 == 0.0f ? 1e-6f : k1)) * (1.0f + 1e-3f*curand_normal(&st));
        }
        K[i] = make_half2(k0, k1);
        Q[i] = make_half2(q0, q1);
    }

    float max_abs_s = 0.0f;
    const double ref = dot_ref(K, Q, DKQ/2);
    const double up  = (double) dot_upstream(K, Q, DKQ/2);
    const double nw  = (double) dot_new(K, Q, DKQ/2, &max_abs_s);

    if (!isfinite(up) || !isfinite(nw)) atomicAdd(&d_overflow, 1ull);

    const double eu = fabs(up - ref), en = fabs(nw - ref);
    atomicAdd(&d_se_up,  eu*eu);
    atomicAdd(&d_se_new, en*en);
    if (en < eu) atomicAdd(&d_new_better, 1ull);
    else if (eu < en) atomicAdd(&d_up_better, 1ull);
    else atomicAdd(&d_tie, 1ull);

    // atomicMax on double via CAS-free trick: use atomicMax on the bit pattern of a positive double
    unsigned long long *pu = (unsigned long long *)&d_max_up;
    unsigned long long *pn = (unsigned long long *)&d_max_new;
    unsigned long long *ps = (unsigned long long *)&d_max_abs_s;
    unsigned long long bu, bn, bs;
    memcpy(&bu, &eu, 8); memcpy(&bn, &en, 8);
    const double sd = (double) max_abs_s; memcpy(&bs, &sd, 8);
    atomicMax(pu, bu); atomicMax(pn, bn); atomicMax(ps, bs);
}

static void reset() {
    double z = 0.0; unsigned long long zz = 0;
    cudaMemcpyToSymbol(d_se_up, &z, 8);   cudaMemcpyToSymbol(d_se_new, &z, 8);
    cudaMemcpyToSymbol(d_max_up, &z, 8);  cudaMemcpyToSymbol(d_max_new, &z, 8);
    cudaMemcpyToSymbol(d_max_abs_s, &z, 8);
    cudaMemcpyToSymbol(d_new_better, &zz, 8); cudaMemcpyToSymbol(d_up_better, &zz, 8);
    cudaMemcpyToSymbol(d_tie, &zz, 8); cudaMemcpyToSymbol(d_overflow, &zz, 8);
}

static void report(const char *name, float sigK, float sigQ) {
    double seu, sen, mu, mn, ms; unsigned long long nb, ub, tie, of;
    cudaMemcpyFromSymbol(&seu, d_se_up, 8);  cudaMemcpyFromSymbol(&sen, d_se_new, 8);
    cudaMemcpyFromSymbol(&mu, d_max_up, 8);  cudaMemcpyFromSymbol(&mn, d_max_new, 8);
    cudaMemcpyFromSymbol(&ms, d_max_abs_s, 8);
    cudaMemcpyFromSymbol(&nb, d_new_better, 8); cudaMemcpyFromSymbol(&ub, d_up_better, 8);
    cudaMemcpyFromSymbol(&tie, d_tie, 8); cudaMemcpyFromSymbol(&of, d_overflow, 8);
    const double ru = sqrt(seu/NTRIAL), rn = sqrt(sen/NTRIAL);
    printf("%-26s sK=%-6g sQ=%-6g\n", name, sigK, sigQ);
    printf("   RMS abs err   upstream %.6g   new %.6g   ratio new/up = %.4f\n", ru, rn, rn/ru);
    printf("   max abs err   upstream %.6g   new %.6g   ratio new/up = %.4f\n", mu, mn, mn/mu);
    printf("   new closer %llu   upstream closer %llu   tie %llu   (%.1f%% new wins)\n",
           nb, ub, tie, 100.0*nb/(double)NTRIAL);
    printf("   max |fp16 partial sum| %.6g  (fp16 max 65504)   non-finite results: %llu\n\n", ms, of);
}

int main() {
    const int threads = 256, blocks = (NTRIAL + threads - 1)/threads;
    struct { const char *n; int d; float sk, sq; } cases[] = {
        {"iid gaussian",            0, 1.0f,  0.05f},
        {"iid gaussian (larger)",   0, 4.0f,  0.25f},
        {"gaussian + 2% outliers",  1, 1.0f,  0.05f},
        {"near-total cancellation", 2, 1.0f,  0.05f},
        {"stress: near fp16 range", 0, 40.0f, 4.0f},
    };
    for (auto &c : cases) {
        reset();
        trial<<<blocks, threads>>>(c.d, c.sk, c.sq, 1234u + c.d*7 + (unsigned)(c.sk*13));
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
        report(c.n, c.sk, c.sq);
    }
    return 0;
}
