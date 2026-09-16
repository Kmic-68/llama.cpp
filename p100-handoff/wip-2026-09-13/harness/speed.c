// Idle-GPU speed of the fp16 GEMM algorithms at the shapes the GEMM attention path issues.
// Median of 9 timed calls after 2 warmups; ratio vs DEFAULT at the same shape.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int cmpd(const void*a,const void*b){ double x=*(double*)a,y=*(double*)b; return x<y?-1:x>y; }
static cublasHandle_t h;
static double tm(cublasOperation_t oa, int m, int n, int k, void*A, int lda, void*B, int ldb, void*C, int algo){
    const uint16_t a16=0x3c00,b16=0; cublasGemmAlgo_t al = algo<0?CUBLAS_GEMM_DEFAULT:(cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0+algo);
    double t[9];
    for (int r=0;r<11;r++){ cudaDeviceSynchronize(); double t0=now();
        if (cublasGemmEx(h,oa,CUBLAS_OP_N,m,n,k,&a16,A,CUDA_R_16F,lda,B,CUDA_R_16F,ldb,&b16,C,CUDA_R_16F,m,CUBLAS_COMPUTE_16F,al)!=CUBLAS_STATUS_SUCCESS) return -1;
        cudaDeviceSynchronize(); if (r>=2) t[r-2]=now()-t0; }
    qsort(t,9,sizeof(double),cmpd); return t[4];
}
int main(void){
    cublasCreate(&h);
    const int D=256; const int nts[]={128,512,1024,2048}; const int ks[]={2048,1024,256};
    size_t maxn=2048*6, maxk=2048;
    void*A,*B,*C; cudaMalloc(&A,(size_t)D*maxn*2); cudaMalloc(&B,maxk*maxn*2); cudaMalloc(&C,maxk*maxn*2);
    uint16_t *buf=malloc(maxk*maxn*2); for(size_t i=0;i<maxk*maxn;i++) buf[i]=0x3000|(i*2654435761u>>22 & 0x3ff); // ~0.25..0.5 halves
    cudaMemcpy(A,buf,(size_t)D*maxn*2,1); cudaMemcpy(B,buf,maxk*maxn*2,1);
    const int al[]={-1,4,5,6};
    printf("%-5s %5s %5s | %9s | %8s %8s %8s   (time relative to DEFAULT; >1 = slower)\n","GEMM","nt","k","DEF ms","ALGO4","ALGO5","ALGO6");
    for(int g=0; g<2; g++) for(int ti=0;ti<4;ti++) for(int ki=0;ki<3;ki++){
        int n=nts[ti]*6, k=ks[ki]; double r[4];
        for(int a=0;a<4;a++){
            if(g==0) r[a]=tm(CUBLAS_OP_N,D,n,k,A,D,B,k,C,al[a]);     // PV:   O[D x n] = V[D x k] P[k x n]
            else     r[a]=tm(CUBLAS_OP_T,k,n,D,A,D,B,D,C,al[a]);     // QK^T: S[k x n] = K^T[k x D] Q[D x n]
        }
        printf("%-5s %5d %5d | %9.3f | %8.3f %8.3f %8.3f\n", g?"QK^T":"PV", nts[ti], k, r[0]*1e3, r[1]/r[0], r[2]/r[0], r[3]/r[0]); fflush(stdout);
    }
    return 0;
}
