// The model's prefill matmuls on Pascal run cublasGemmEx COMPUTE_16F with CUBLAS_GEMM_DEFAULT_TENSOR_OP
// (upstream, ggml-cuda.cu ggml_cuda_mul_mat_cublas_impl). Which accuracy family does that pick at the
// real shapes, and what do the blocked algorithms cost? Shapes per GPU under -sm tensor, ub=2048:
// C[m x n] = A^T[m x k] B[k x n], k = input width (the accumulation length).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint16_t f2h(float f){ if (f == 0.0f) return 0; uint16_t s = f < 0 ? 0x8000 : 0; double a = fabs(f); int e; double m = frexp(a, &e); int E = e - 1 + 15;
    if (E >= 31) return s | 0x7c00; double q; if (E <= 0) { q = nearbyint(a / ldexp(1.0, -24)); if (q >= 1024) return s | (1<<10); return s | (uint16_t) q; }
    q = nearbyint((m*2 - 1) * 1024); if (q == 1024) { q = 0; E++; if (E >= 31) return s | 0x7c00; } return s | (uint16_t)(E << 10) | (uint16_t) q; }
static double h2d(uint16_t h){ int s=h>>15, e=(h>>10)&0x1f, m=h&0x3ff; double v = e ? ldexp(1.0+m/1024.0, e-15) : ldexp(m/1024.0, -14); return s ? -v : v; }
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double urand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0/9007199254740992.0); }
static double grand(void){ double u = urand(), v = urand(); return sqrt(-2*log(u + 1e-300)) * cos(2*M_PI*v); }
static int cmpd(const void*a,const void*b){ double x=*(double*)a,y=*(double*)b; return x<y?-1:x>y; }
int main(void){
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH); void * ws; cudaMalloc(&ws, 4u<<20); cublasSetWorkspace(h, ws, 4u<<20);
    struct { const char * name; int m, k; } shp[] = { {"ffn_up/gate 5120->8704", 8704, 5120}, {"ffn_down 8704->5120", 5120, 8704}, {"attn_q 5120->6144", 6144, 5120}, {"attn_out 3072->5120", 5120, 3072} };
    const int n = 2048;
    struct { const char * name; int algo; } al[] = { {"DEF_TENSOR_OP", -2}, {"DEFAULT", -1}, {"ALGO1", 1}, {"ALGO3", 3}, {"ALGO4", 4}, {"ALGO5", 5}, {"ALGO6", 6} };
    for (int si = 0; si < 4; si++) {
        int m = shp[si].m, k = shp[si].k;
        size_t nA = (size_t)k*m, nB = (size_t)k*n, nC = (size_t)m*n;
        uint16_t * hA = malloc(nA*2), * hB = malloc(nB*2), * hC = malloc(nC*2); double * dA = malloc(nA*8), * dB = malloc(nB*8), * ref = malloc(nC*8);
        // weights ~ N(0, 0.02); activations ~ N(0,1) with 8 massive-activation channels (x40)
        for (size_t i = 0; i < nA; i++) { hA[i] = f2h((float)(0.02*grand())); dA[i] = h2d(hA[i]); }
        for (int c = 0; c < n; c++) for (int d = 0; d < k; d++) { double v = grand(); if (d % (k/8) == 3) v *= 40; size_t i = (size_t)c*k + d; hB[i] = f2h((float) v); dB[i] = h2d(hB[i]); }
        void * gA, * gB, * gC, * gA64, * gB64, * gC64;
        cudaMalloc(&gA, nA*2); cudaMalloc(&gB, nB*2); cudaMalloc(&gC, nC*2); cudaMalloc(&gA64, nA*8); cudaMalloc(&gB64, nB*8); cudaMalloc(&gC64, nC*8);
        cudaMemcpy(gA, hA, nA*2, 1); cudaMemcpy(gB, hB, nB*2, 1); cudaMemcpy(gA64, dA, nA*8, 1); cudaMemcpy(gB64, dB, nB*8, 1);
        const double one = 1, zero = 0; cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, gA64, k, gB64, k, &zero, gC64, m);
        cudaDeviceSynchronize(); cudaMemcpy(ref, gC64, nC*8, 2);
        double rss = 0; for (size_t i = 0; i < nC; i++) rss += ref[i]*ref[i];
        printf("=== %s (m=%d n=%d k=%d)\n", shp[si].name, m, n, k);
        double t_def = 0;
        for (int a = 0; a < 7; a++) {
            cublasGemmAlgo_t algo = al[a].algo == -2 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : al[a].algo < 0 ? CUBLAS_GEMM_DEFAULT : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0 + al[a].algo);
            const uint16_t a16 = 0x3c00, b16 = 0; double t[7]; int ok = 1;
            for (int r = 0; r < 9; r++) { cudaDeviceSynchronize(); double t0 = now();
                if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a16, gA, CUDA_R_16F, k, gB, CUDA_R_16F, k, &b16, gC, CUDA_R_16F, m, CUBLAS_COMPUTE_16F, algo)) { ok = 0; break; }
                cudaDeviceSynchronize(); if (r >= 2) t[r-2] = now() - t0; }
            if (!ok) { printf("  %-14s n/a\n", al[a].name); continue; }
            qsort(t, 7, sizeof(double), cmpd);
            cudaMemcpy(hC, gC, nC*2, 2); double es = 0; for (size_t i = 0; i < nC; i++) { double e = h2d(hC[i]) - ref[i]; es += e*e; }
            if (a == 0) t_def = t[3];
            printf("  %-14s NMSE %9.2e   %7.2f ms  (%.3fx DEF_TENSOR_OP time)\n", al[a].name, es/rss, t[3]*1e3, t[3]/t_def); fflush(stdout);
        }
        cudaFree(gA); cudaFree(gB); cudaFree(gC); cudaFree(gA64); cudaFree(gB64); cudaFree(gC64); free(hA); free(hB); free(hC); free(dA); free(dB); free(ref);
    }
    return 0;
}
