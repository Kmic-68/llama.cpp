#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc, char ** argv){
    int n = argc > 1 ? atoi(argv[1]) : 800;
    cublasHandle_t h; cublasCreate(&h);
    void * A, * C; cudaMalloc(&A, (size_t)n*n*4); cudaMalloc(&C, (size_t)n*n*4);
    cudaMemset(A, 1, (size_t)n*n*4);
    const float a = 1e-3f, b = 0.0f;
    for (int k = 0; k < 4; k++) { double t0 = now(); int st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &a, A, n, A, n, &b, C, n); int e = cudaDeviceSynchronize(); printf("sgemm n=%d: %.1f ms st=%d sync=%d\n", n, (now()-t0)*1e3, st, e); }
    return 0;
}
