// fp16 overflow headroom per algorithm: V = +A everywhere, P = 1/8 everywhere, so the exact sum is
// k*A/8 (fits fp16 when < 65504). Also mixed sign: V alternates +A / -A by key (exact sum ~0, but
// a sequential partial never exceeds A/8). Count non-finite outputs.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static uint16_t f2h(float f){ if (f == 0.0f) return 0; uint16_t s = f < 0 ? 0x8000 : 0; double a = fabs(f); int e; double m = frexp(a, &e); int E = e - 1 + 15;
    if (E >= 31) return s | 0x7c00; if (E <= 0) return s; int q = (int) nearbyint((m*2 - 1) * 1024); if (q == 1024) { q = 0; E++; } if (E >= 31) return s | 0x7c00; return s | (uint16_t)(E << 10) | (uint16_t) q; }
int main(void){
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH); void * ws; cudaMalloc(&ws, 4u<<20); cublasSetWorkspace(h, ws, 4u<<20);
    const int DV = 256, k = 2048; const int nts[] = {512, 2048};
    const double As[] = {16, 64, 128, 200, 250};
    const int algos[] = {-1, 1, 4, 5, 6};
    for (int ti = 0; ti < 2; ti++) for (int pat = 0; pat < 3; pat++) for (int ai = 0; ai < 5; ai++) {
        int n = nts[ti]*6; double A = As[ai];
        size_t nV = (size_t)DV*k, nP = (size_t)k*n, nO = (size_t)DV*n;
        uint16_t * hV = malloc(nV*2), * hP = malloc(nP*2), * hO = malloc(nO*2);
        for (int j = 0; j < k; j++) for (int d = 0; d < DV; d++) {
            double v = A; if (pat == 1) v = (j % 2) ? -A : A; if (pat == 2) v = (j < k/2) ? A : -A;   // pat 2: first half +, second half -
            hV[(size_t)j*DV + d] = f2h((float) v); }
        for (size_t i = 0; i < nP; i++) hP[i] = 0x3000;  // 0.125
        void * gV, * gP, * gO; cudaMalloc(&gV, nV*2); cudaMalloc(&gP, nP*2); cudaMalloc(&gO, nO*2);
        cudaMemcpy(gV, hV, nV*2, 1); cudaMemcpy(gP, hP, nP*2, 1);
        printf("nt=%4d %-10s A=%5.0f (exact %8.0f) |", nts[ti], pat == 0 ? "all +A" : pat == 1 ? "alternate" : "+A then -A", A, pat == 0 ? k*A/8 : 0.0);
        for (int a = 0; a < 5; a++) {
            const uint16_t a16 = 0x3c00, b16 = 0; cublasGemmAlgo_t al = algos[a] < 0 ? CUBLAS_GEMM_DEFAULT : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0 + algos[a]);
            memset(hO, 0, nO*2); cudaMemcpy(gO, hO, nO*2, 1);
            cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &a16, gV, CUDA_R_16F, DV, gP, CUDA_R_16F, k, &b16, gO, CUDA_R_16F, DV, CUBLAS_COMPUTE_16F, al);
            cudaDeviceSynchronize(); if (st) { printf(" n/a"); continue; }
            cudaMemcpy(hO, gO, nO*2, 2); size_t bad = 0; for (size_t i = 0; i < nO; i++) if ((hO[i] & 0x7c00) == 0x7c00) bad++;
            printf(" %s:%zu", algos[a] < 0 ? "DEF" : (char[6]){'A','0'+algos[a],0}, bad);
        }
        printf("\n"); fflush(stdout);
        cudaFree(gV); cudaFree(gP); cudaFree(gO); free(hV); free(hP); free(hO);
    }
    return 0;
}
