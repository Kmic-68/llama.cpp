// Does a COMPUTE_16F GEMM read C when beta == 0? Fill C with NaN (and separately +inf), call with
// beta = 0, count non-finite outputs per algorithm, at the GEMM path's PV shapes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
int main(void){
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH); void * ws; cudaMalloc(&ws, 4u<<20); cublasSetWorkspace(h, ws, 4u<<20);
    const int DV = 256; const int nts[] = {128, 512, 2048}; const int ks[] = {2048, 1024, 256};
    const int algos[] = {-1, 1, 4, 5, 6};
    for (int ti = 0; ti < 3; ti++) for (int ki = 0; ki < 3; ki++) {
        int n = nts[ti]*6, k = ks[ki];
        size_t nV = (size_t)DV*k, nP = (size_t)k*n, nO = (size_t)DV*n;
        uint16_t * hV = malloc(nV*2), * hP = malloc(nP*2), * hO = malloc(nO*2);
        for (size_t i = 0; i < nV; i++) hV[i] = 0x3c00;          // 1.0
        for (size_t i = 0; i < nP; i++) hP[i] = 0x2800;          // 1/32
        void * gV, * gP, * gO; cudaMalloc(&gV, nV*2); cudaMalloc(&gP, nP*2); cudaMalloc(&gO, nO*2);
        cudaMemcpy(gV, hV, nV*2, 1); cudaMemcpy(gP, hP, nP*2, 1);
        printf("nt=%4d k=%4d |", nts[ti], k);
        for (int fill = 0; fill < 2; fill++) {
            uint16_t fv = fill ? 0x7c00 /* +inf */ : 0x7e00 /* NaN */;
            for (int a = 0; a < 5; a++) {
                for (size_t i = 0; i < nO; i++) hO[i] = fv;
                cudaMemcpy(gO, hO, nO*2, 1);
                const uint16_t a16 = 0x3c00, b16 = 0x0000;
                cublasGemmAlgo_t al = algos[a] < 0 ? CUBLAS_GEMM_DEFAULT : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0 + algos[a]);
                cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &a16, gV, CUDA_R_16F, DV, gP, CUDA_R_16F, k, &b16, gO, CUDA_R_16F, DV, CUBLAS_COMPUTE_16F, al);
                cudaDeviceSynchronize();
                if (st != CUBLAS_STATUS_SUCCESS) { printf(" %s%s:n/a", fill ? "inf-" : "nan-", algos[a] < 0 ? "DEF" : (char[6]){'A','0'+algos[a],0}); continue; }
                cudaMemcpy(hO, gO, nO*2, 2);
                size_t bad = 0; for (size_t i = 0; i < nO; i++) if ((hO[i] & 0x7c00) == 0x7c00) bad++;
                printf(" %s%s:%zu", fill ? "inf-" : "nan-", algos[a] < 0 ? "DEF" : (char[6]){'A','0'+algos[a],0}, bad);
            }
        }
        printf("   (of %zu)\n", nO); fflush(stdout);
        cudaFree(gV); cudaFree(gP); cudaFree(gO); free(hV); free(hP); free(hO);
    }
    return 0;
}
