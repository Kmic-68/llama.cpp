// Bitwise reproducibility of the GEMM attention path's cuBLAS calls. Same inputs, called repeatedly
// in one process: compare each output with the first; print a hash for cross-process comparison.
// Arg "streams": also create a second active stream (ggml has several: compute, peer copy, per-thread).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double urand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0/9007199254740992.0); }
static uint64_t fnv(const unsigned char * p, size_t n){ uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h; }
int main(int argc, char ** argv){
    int streams = argc > 1 && !strcmp(argv[1], "streams");
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH); void * ws; cudaMalloc(&ws, 4u<<20); cublasSetWorkspace(h, ws, 4u<<20);
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking); cublasSetStream(h, st);
    cudaStream_t st2 = NULL; void * junk = NULL;
    if (streams) { cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking); cudaMalloc(&junk, 64u<<20); }
    const int D = 256, k = 2048, n = 2048*6;
    size_t nK = (size_t)D*k, nQ = (size_t)D*n, nS = (size_t)k*n, nO = (size_t)D*n;
    uint16_t * hK = malloc(nK*2), * hQ = malloc(nQ*2); float * fV = malloc(nK*4), * fP = malloc(nS*4);
    for (size_t i = 0; i < nK; i++) hK[i] = 0x3000 | ((uint16_t)(urand()*0x3ff)) | (urand() < 0.5 ? 0x8000 : 0);
    for (size_t i = 0; i < nQ; i++) hQ[i] = 0x2c00 | ((uint16_t)(urand()*0x3ff)) | (urand() < 0.5 ? 0x8000 : 0);
    for (size_t i = 0; i < nK; i++) fV[i] = (float)(2*urand()-1);
    for (size_t i = 0; i < nS; i++) fP[i] = (float)(0.125*urand());
    void * gK, * gQ, * gV, * gP, * gS, * gO; cudaMalloc(&gK, nK*2); cudaMalloc(&gQ, nQ*2); cudaMalloc(&gV, nK*4); cudaMalloc(&gP, nS*4); cudaMalloc(&gS, nS*4); cudaMalloc(&gO, nO*4);
    cudaMemcpy(gK, hK, nK*2, 1); cudaMemcpy(gQ, hQ, nQ*2, 1); cudaMemcpy(gV, fV, nK*4, 1); cudaMemcpy(gP, fP, nS*4, 1);
    float * first = malloc(nS*4), * cur = malloc(nS*4);
    const float a = 1.0f, b = 0.0f;
    for (int which = 0; which < 2; which++) {
        size_t nC = which ? nO : nS; uint64_t h0 = 0; int diffs = 0;
        for (int r = 0; r < 6; r++) {
            if (streams) { cudaMemsetAsync(junk, r, 64u<<20, st2); }   // concurrent work on another stream
            cublasStatus_t s;
            if (which == 0) s = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, k, n, D, &a, gK, CUDA_R_16F, D, gQ, CUDA_R_16F, D, &b, gS, CUDA_R_32F, k, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            else            s = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, n, k, &a, gV, CUDA_R_32F, D, gP, CUDA_R_32F, k, &b, gO, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaStreamSynchronize(st); if (streams) cudaStreamSynchronize(st2);
            if (s) { printf("status %d\n", (int) s); return 1; }
            cudaMemcpy(cur, which ? gO : gS, nC*4, 2);
            if (r == 0) { memcpy(first, cur, nC*4); h0 = fnv((unsigned char *) cur, nC*4); }
            else if (memcmp(first, cur, nC*4)) { diffs++; size_t nd = 0; double mx = 0; for (size_t i = 0; i < nC; i++) if (first[i] != cur[i]) { nd++; double rel = fabs(first[i]-cur[i])/(fabs(first[i])+1e-30); if (rel > mx) mx = rel; } printf("   %s call %d differs from call 0 in %zu of %zu elements (max rel %.2e)\n", which ? "PV 32F" : "QK^T 32F(16F in)", r, nd, nC, mx); }
        }
        printf("%s%s: %d of 5 repeat calls differ; hash of call 0 = %016llx\n", which ? "PV 32F" : "QK^T 32F(16F in)", streams ? " [2 streams]" : "", diffs, (unsigned long long) h0);
    }
    return 0;
}
