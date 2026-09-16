#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(void){
    size_t nb = (size_t)256 << 20; void * p; cudaMalloc(&p, nb);
    cudaMemset(p, 0, nb); cudaDeviceSynchronize();
    for (int k = 0; k < 3; k++) { double t0 = now(); for (int r = 0; r < 4; r++) cudaMemset(p, r, nb); cudaDeviceSynchronize(); printf("4 x 256 MB memset: %.1f ms\n", (now()-t0)*1e3); }
    return 0;
}
