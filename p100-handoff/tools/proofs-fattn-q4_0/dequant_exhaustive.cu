// Exhaustive equivalence check for the q4_0 tile-loader dequant.
//
// Three expressions for the same quantity, d*(q-8):
//   REF : upstream dequantize_block_q4_0 -> ggml_cuda_cast<half>(d*q + (-8*d)) in fp32
//   OLD : last public push (0dacb39a8)   -> __hmul2(__halves2half2(__int2half_rn(q-8),..), dh)
//   NEW : HEAD (c6f5211f4)               -> byte_perm + 0x6400 magic, __hmul2(__hsub2(.,1032), dh)
//
// Domain is the COMPLETE input space: every one of the 65536 fp16 bit patterns for the block
// scale d (normals, subnormals, +-0, +-inf, every NaN payload) crossed with every one of the
// 256 byte values, which covers all 16 nibbles in both positions. The two bytes packed into
// the uint16 are made different so the byte_perm lane routing is exercised, not just the mask.
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

__device__ unsigned long long n_new_vs_old_bit   = 0;  // NEW != OLD, bitwise
__device__ unsigned long long n_new_vs_ref_bit   = 0;  // NEW != REF, bitwise
__device__ unsigned long long n_new_vs_ref_val   = 0;  // NEW != REF, numerically (NaN==NaN, +0==-0)
__device__ unsigned long long n_zero_sign_only   = 0;  // differ ONLY in the sign of a zero
__device__ unsigned long long n_checked          = 0;
__device__ unsigned int        first_bad         = 0xffffffffu;

__device__ __forceinline__ uint16_t bits(half h) { uint16_t u; memcpy(&u, &h, 2); return u; }
__device__ __forceinline__ bool is_nan_h(uint16_t u) { return ((u & 0x7C00) == 0x7C00) && (u & 0x03FF); }
__device__ __forceinline__ bool is_zero_h(uint16_t u) { return (u & 0x7FFF) == 0; }

__global__ void check(void) {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= 65536) return;

    const uint16_t dbits = (uint16_t) idx;
    half d; memcpy(&d, &dbits, 2);
    const half2 dh    = __half2half2(d);
    const half2 k1032 = __float2half2_rn(1032.0f);

    unsigned long long loc_ob = 0, loc_rb = 0, loc_rv = 0, loc_z = 0, loc_n = 0;

    for (int b0 = 0; b0 < 256; ++b0) {
        const int b1 = 255 - b0;                       // distinct second byte: exercises lane routing
        const uint16_t qs16 = (uint16_t)(b0 | (b1 << 8));   // little-endian: byte0=b0, byte1=b1

        // ---- NEW: exactly the expression in fattn-tile.cuh -------------------------------
        const uint32_t s    = __byte_perm((uint32_t) qs16, 0, 0x4140);
        const uint32_t lo_b = ((s >> 0) & 0x000F000F) | 0x64006400;
        const uint32_t hi_b = ((s >> 4) & 0x000F000F) | 0x64006400;
        half2 lo_h, hi_h;
        memcpy(&lo_h, &lo_b, sizeof(lo_h));
        memcpy(&hi_h, &hi_b, sizeof(hi_h));
        const half2 new_lo = __hmul2(__hsub2(lo_h, k1032), dh);
        const half2 new_hi = __hmul2(__hsub2(hi_h, k1032), dh);

        // four (nibble, lane) slots produced from this one uint16 read
        const int  q  [4] = { b0 & 0x0F, b1 & 0x0F, (b0 >> 4) & 0x0F, (b1 >> 4) & 0x0F };
        const half nv [4] = { __low2half(new_lo), __high2half(new_lo),
                              __low2half(new_hi), __high2half(new_hi) };

        for (int t = 0; t < 4; ++t) {
            // ---- OLD: last public push ---------------------------------------------------
            const half ov = __hmul(__int2half_rn(q[t] - 8), d);

            // ---- REF: upstream to_fp16, fp32 math then cast ------------------------------
            const float df = __half2float(d);
            const half  rv = __float2half(df * (float) q[t] + (-8.0f * df));

            const uint16_t nb = bits(nv[t]), ob = bits(ov), rb = bits(rv);

            loc_n++;
            if (nb != ob) { loc_ob++; if (first_bad == 0xffffffffu) atomicCAS(&first_bad, 0xffffffffu, (unsigned)((dbits<<8)|(q[t]<<4)|t)); }
            if (nb != rb) {
                loc_rb++;
                const bool both_nan  = is_nan_h(nb) && is_nan_h(rb);
                const bool both_zero = is_zero_h(nb) && is_zero_h(rb);
                if (!both_nan && !both_zero) loc_rv++;
                else if (both_zero)          loc_z++;
            }
        }
    }
    atomicAdd(&n_new_vs_old_bit, loc_ob);
    atomicAdd(&n_new_vs_ref_bit, loc_rb);
    atomicAdd(&n_new_vs_ref_val, loc_rv);
    atomicAdd(&n_zero_sign_only, loc_z);
    atomicAdd(&n_checked,        loc_n);
}

int main() {
    check<<<256, 256>>>();
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }

    unsigned long long ob, rb, rv, z, n; unsigned fb;
    cudaMemcpyFromSymbol(&ob, n_new_vs_old_bit, sizeof(ob));
    cudaMemcpyFromSymbol(&rb, n_new_vs_ref_bit, sizeof(rb));
    cudaMemcpyFromSymbol(&rv, n_new_vs_ref_val, sizeof(rv));
    cudaMemcpyFromSymbol(&z,  n_zero_sign_only, sizeof(z));
    cudaMemcpyFromSymbol(&n,  n_checked,        sizeof(n));
    cudaMemcpyFromSymbol(&fb, first_bad,        sizeof(fb));

    printf("cases checked                        : %llu\n", n);
    printf("NEW vs OLD  bitwise differences      : %llu   %s\n", ob, ob ? "<-- FAIL" : "(bit-identical)");
    printf("NEW vs REF  bitwise differences      : %llu\n", rb);
    printf("NEW vs REF  differ only in zero sign : %llu\n", z);
    printf("NEW vs REF  NUMERIC differences      : %llu   %s\n", rv, rv ? "<-- FAIL" : "(numerically identical)");
    if (fb != 0xffffffffu) printf("first NEW!=OLD case: dbits=0x%04x q=%d slot=%d\n", fb>>8, (fb>>4)&0xF, fb&0xF);
    return (ob || rv) ? 1 : 0;
}
