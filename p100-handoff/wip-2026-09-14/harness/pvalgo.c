// PV GEMM of the GEMM attention path, O[DV x n] = V[DV x k] P[k x n], fp16 in/out, COMPUTE_16F.
// Round-robin timing: 30 rounds, every algorithm once per round with the order rotated, median per
// algorithm -- so drift and warm-up land on all algorithms alike. ggml's handle settings.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void*a,const void*b){ double x=*(double*)a,y=*(double*)b; return x<y?-1:x>y; }
int main(void){
    cublasHandle_t h; cublasCreate(&h); cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    const int DV = 256, k = 2048, nts[] = {128, 512, 1024, 2048}, gqa = 6;
    const int al[] = {-1, 4, 5, 6}; const char * nm[] = {"DEFAULT", "ALGO4", "ALGO5", "ALGO6"};
    size_t maxn = 2048*gqa;
    void * V, * P, * O; cudaMalloc(&V, (size_t)DV*k*2); cudaMalloc(&P, (size_t)k*maxn*2); cudaMalloc(&O, (size_t)DV*maxn*2);
    uint16_t * buf = malloc((size_t)k*maxn*2); uint64_t s = 88172645463325252ull;
    for (size_t i = 0; i < (size_t)k*maxn; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; buf[i] = (uint16_t)(0x2000 + (s & 0x0fff)); }
    cudaMemcpy(V, buf, (size_t)DV*k*2, 1); cudaMemcpy(P, buf, (size_t)k*maxn*2, 1);
    const uint16_t a16 = 0x3c00, b16 = 0;
    printf("PV  DV=%d k=%d   median ms of 30 round-robin rounds (ratio vs ALGO4)\n", DV, k);
    for (int ti = 0; ti < 4; ti++) {
        int n = nts[ti]*gqa; double t[4][30];
        for (int w = 0; w < 3; w++) for (int a = 0; a < 4; a++) {   // warm-up
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &a16, V, CUDA_R_16F, DV, P, CUDA_R_16F, k, &b16, O, CUDA_R_16F, DV, CUBLAS_COMPUTE_16F,
                al[a] < 0 ? CUBLAS_GEMM_DEFAULT : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0 + al[a])); }
        for (int r = 0; r < 30; r++) for (int j = 0; j < 4; j++) {
            int a = (j + r) % 4; cudaDeviceSynchronize(); double t0 = now();
            cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &a16, V, CUDA_R_16F, DV, P, CUDA_R_16F, k, &b16, O, CUDA_R_16F, DV, CUBLAS_COMPUTE_16F,
                al[a] < 0 ? CUBLAS_GEMM_DEFAULT : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0 + al[a]));
            cudaDeviceSynchronize(); t[a][r] = st == CUBLAS_STATUS_SUCCESS ? now() - t0 : 1e9; }
        double med[4]; for (int a = 0; a < 4; a++) { qsort(t[a], 30, sizeof(double), cmpd); med[a] = (t[a][14] + t[a][15]) / 2; }
        printf("  nt=%4d n=%5d:", nts[ti], n);
        for (int a = 0; a < 4; a++) printf("  %s %7.3f (%.3fx)", nm[a], med[a]*1e3, med[a]/med[1]);
        printf("\n"); fflush(stdout);
    }
    return 0;
}
