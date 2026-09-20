// Classify every (d, q) pair where NEW differs from upstream REF, over the complete
// 65536 x 16 domain. Reports unique pairs, not slot occurrences.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

struct Counts { unsigned long long zero_sign, numeric, d_inf, d_nan, d_zero, q_eq_8, underflow, other; };
__device__ Counts C = {0,0,0,0,0,0,0,0};
__device__ unsigned long long n_pairs = 0;
__device__ unsigned int ex_num[8];   // example encodings of numeric diffs
__device__ unsigned int n_ex = 0;

__device__ __forceinline__ uint16_t bits(half h) { uint16_t u; memcpy(&u, &h, 2); return u; }
__device__ __forceinline__ bool nan_h (uint16_t u) { return ((u & 0x7C00) == 0x7C00) && (u & 0x03FF); }
__device__ __forceinline__ bool inf_h (uint16_t u) { return ((u & 0x7C00) == 0x7C00) && !(u & 0x03FF); }
__device__ __forceinline__ bool zero_h(uint16_t u) { return (u & 0x7FFF) == 0; }

__global__ void classify() {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 65536) return;
    const uint16_t dbits = (uint16_t) idx;
    half d; memcpy(&d, &dbits, 2);
    const half2 dh = __half2half2(d), k1032 = __float2half2_rn(1032.0f);

    for (int q = 0; q < 16; ++q) {
        const uint32_t w    = 0x64006400u | (uint32_t)(q | (q << 16));
        half2 wh; memcpy(&wh, &w, 4);
        const half nv = __low2half(__hmul2(__hsub2(wh, k1032), dh));

        const float df = __half2float(d);
        const half  rv = __float2half(df * (float) q + (-8.0f * df));

        const uint16_t nb = bits(nv), rb = bits(rv);
        atomicAdd(&n_pairs, 1ull);
        if (nb == rb) continue;

        if (zero_h(nb) && zero_h(rb)) {
            atomicAdd(&C.zero_sign, 1ull);
            if (zero_h(dbits))                  atomicAdd(&C.d_zero,    1ull);
            else if (q == 8)                    atomicAdd(&C.q_eq_8,    1ull);
            else                                atomicAdd(&C.underflow, 1ull);
        } else if (nan_h(nb) && nan_h(rb)) {
            // both NaN: numerically the same outcome
        } else {
            atomicAdd(&C.numeric, 1ull);
            if      (inf_h(dbits)) atomicAdd(&C.d_inf, 1ull);
            else if (nan_h(dbits)) atomicAdd(&C.d_nan, 1ull);
            else {
                atomicAdd(&C.other, 1ull);
                unsigned slot = atomicAdd(&n_ex, 1u);
                if (slot < 8) ex_num[slot] = (unsigned)((dbits << 8) | q);
            }
        }
    }
}

int main() {
    classify<<<256,256>>>();
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("cuda err\n"); return 1; }
    Counts c; unsigned long long n; unsigned ex[8], ne;
    cudaMemcpyFromSymbol(&c, C, sizeof(c));
    cudaMemcpyFromSymbol(&n, n_pairs, sizeof(n));
    cudaMemcpyFromSymbol(ex, ex_num, sizeof(ex));
    cudaMemcpyFromSymbol(&ne, n_ex, sizeof(ne));
    printf("unique (d,q) pairs checked          : %llu\n", n);
    printf("zero-sign-only differences          : %llu\n", c.zero_sign);
    printf("    of which d == +-0               : %llu\n", c.d_zero);
    printf("    of which q == 8  (exact zero)   : %llu\n", c.q_eq_8);
    printf("    of which underflow to zero      : %llu\n", c.underflow);
    printf("NUMERIC differences                 : %llu\n", c.numeric);
    printf("    of which d == +-inf             : %llu\n", c.d_inf);
    printf("    of which d is NaN               : %llu\n", c.d_nan);
    printf("    of which d FINITE and nonzero   : %llu   %s\n", c.other,
           c.other ? "<-- would be a real defect" : "(none)");
    for (unsigned i = 0; i < ne && i < 8; ++i) printf("    example: dbits=0x%04x q=%u\n", ex[i]>>8, ex[i]&0xFF);
    return c.other ? 1 : 0;
}
