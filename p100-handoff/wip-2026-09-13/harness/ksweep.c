// (1) PV: does each 16F algorithm stay in its accuracy family across every shape the GEMM path
//     issues? n = nt*6 for nt in 128..2048, k = keys in a chunk (2048, or a ragged last chunk).
// (2) QK^T: accuracy of every 16F algorithm at S = K^T Q, D = 256, k = 2048, n = nt*6.
// Reference: the same fp16 inputs multiplied in fp64. Timings are only meaningful on an idle GPU.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint16_t f2h(float f){
    if (f == 0.0f) return 0; uint16_t s = f < 0 ? 0x8000 : 0; double a = fabs(f);
    int e; double m = frexp(a, &e); int E = e - 1 + 15;
    if (E >= 31) return s | 0x7c00;
    double q; if (E <= 0) { q = nearbyint(a / ldexp(1.0, -24)); if (q >= 1024) return s | (1<<10); return s | (uint16_t) q; }
    q = nearbyint((m*2 - 1) * 1024); if (q == 1024) { q = 0; E++; if (E >= 31) return s | 0x7c00; }
    return s | (uint16_t)(E << 10) | (uint16_t) q; }
static double h2d(uint16_t h){ int s=h>>15, e=(h>>10)&0x1f, m=h&0x3ff; double v = e ? ldexp(1.0+m/1024.0, e-15) : ldexp(m/1024.0, -14); return s ? -v : v; }
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double urand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0/9007199254740992.0); }
static double grand(void){ double u = urand(), v = urand(); return sqrt(-2*log(u + 1e-300)) * cos(2*M_PI*v); }
static cublasHandle_t h;
// run one 16F GEMM: C[m x n] = op(A)[m x k] * B[k x n]; returns NMSE vs fp64, ms
static int run(cublasOperation_t oa, int m, int n, int k, void * A16, int lda, void * B16, int ldb, void * C16, const double * ref, int algo, double * nmse, double * ms, double * rmsabs) {
    const uint16_t a16 = 0x3c00, b16 = 0; cublasGemmAlgo_t al = algo < 0 ? CUBLAS_GEMM_DEFAULT : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0 + algo);
    double best = 1e9; cublasStatus_t st = 0;
    for (int r = 0; r < 4; r++) {
        cudaDeviceSynchronize(); double t0 = now();
        st = cublasGemmEx(h, oa, CUBLAS_OP_N, m, n, k, &a16, A16, CUDA_R_16F, lda, B16, CUDA_R_16F, ldb, &b16, C16, CUDA_R_16F, m, CUBLAS_COMPUTE_16F, al);
        cudaDeviceSynchronize(); double dt = now() - t0;
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        if (r > 0 && dt < best) best = dt;
    }
    size_t nC = (size_t)m*n; uint16_t * o = malloc(nC*2); cudaMemcpy(o, C16, nC*2, 2);
    double es = 0, rsum = 0; for (size_t i = 0; i < nC; i++) { double e = h2d(o[i]) - ref[i]; es += e*e; rsum += ref[i]*ref[i]; }
    free(o); *nmse = es/rsum; *ms = best*1e3; *rmsabs = sqrt(es/nC); return 1;
}
int main(void){
    cublasCreate(&h);
    const int DV = 256;
    const int algos[] = {-1, 0, 1, 2, 3, 4, 5, 6};
    printf("=== (1) PV  O = V P: NMSE per algorithm (DEF = CUBLAS_GEMM_DEFAULT) ===\n");
    printf("  %6s %6s |", "nt", "k"); for (int a = 0; a < 8; a++) printf(" %9s%s", algos[a] < 0 ? "DEF" : (char[8]){'A','L','G','O','0'+algos[a],0}, ""); printf("\n");
    const int nts[] = {512, 2048}; const int ks[] = {2048, 1024, 512, 256, 128};
    for (int ti = 0; ti < 2; ti++) for (int ki = 0; ki < 5; ki++) {
        const int n = nts[ti]*6, k = ks[ki];
        size_t nV = (size_t)DV*k, nP = (size_t)k*n, nO = (size_t)DV*n;
        uint16_t * hV = malloc(nV*2), * hP = malloc(nP*2); double * dV = malloc(nV*8), * dP = malloc(nP*8), * ref = malloc(nO*8);
        for (int j = 0; j < k; j++) for (int b = 0; b < DV; b += 32) { float xs[32]; float amax = 0, mx = 0; for (int x = 0; x < 32; x++) { xs[x] = (float)(2*urand()-1); if (fabs(xs[x]) > amax) { amax = fabs(xs[x]); mx = xs[x]; } } double d = h2d(f2h(mx/-8.0f)); for (int x = 0; x < 32; x++) { int q = (int) floor(xs[x]/d + 8.5); if (q < 0) q = 0; if (q > 15) q = 15; size_t i = (size_t)j*DV + b + x; hV[i] = f2h((float)(d*(q-8))); dV[i] = h2d(hV[i]); } }
        double * lg = malloc(k*8);
        for (int c = 0; c < n; c++) { double mx = -1e300; int kind = c % 3; int spike = (int)(urand()*k);
            for (int j = 0; j < k; j++) { lg[j] = grand() * 0.33 + (2*urand()-1); if (lg[j] > mx) mx = lg[j]; }
            for (int j = 0; j < k; j++) { size_t i = (size_t)c*k + j; hP[i] = f2h((float)(0.125*exp(lg[j] - mx))); dP[i] = h2d(hP[i]); } }
        free(lg);
        void * gV16, * gP16, * gO16, * gV64, * gP64, * gO64;
        cudaMalloc(&gV16, nV*2); cudaMalloc(&gP16, nP*2); cudaMalloc(&gO16, nO*2); cudaMalloc(&gV64, nV*8); cudaMalloc(&gP64, nP*8); cudaMalloc(&gO64, nO*8);
        cudaMemcpy(gV16, hV, nV*2, 1); cudaMemcpy(gP16, hP, nP*2, 1); cudaMemcpy(gV64, dV, nV*8, 1); cudaMemcpy(gP64, dP, nP*8, 1);
        const double one = 1.0, zero = 0.0; cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &one, gV64, DV, gP64, k, &zero, gO64, DV);
        cudaDeviceSynchronize(); cudaMemcpy(ref, gO64, nO*8, 2);
        printf("  %6d %6d |", nts[ti], k);
        for (int a = 0; a < 8; a++) { double e, ms, ra; if (run(CUBLAS_OP_N, DV, n, k, gV16, DV, gP16, k, gO16, ref, algos[a], &e, &ms, &ra)) printf(" %9.2e", e); else printf(" %9s", "n/a"); }
        printf("\n"); fflush(stdout);
        cudaFree(gV16); cudaFree(gP16); cudaFree(gO16); cudaFree(gV64); cudaFree(gP64); cudaFree(gO64); free(hV); free(hP); free(dV); free(dP); free(ref);
    }
    return 0; printf("\n=== (2) QK^T  S = K^T Q (D=256): NMSE and RMS absolute logit error x4 (after the softmax's *4) ===\n");
    for (int ti = 0; ti < 2; ti++) {
        const int nt = ti ? 2048 : 512, n = nt*6, k = 2048, D = 256;
        size_t nK = (size_t)D*k, nQ = (size_t)D*n, nS = (size_t)k*n;
        uint16_t * hK = malloc(nK*2), * hQ = malloc(nQ*2); double * dK = malloc(nK*8), * dQ = malloc(nQ*8), * ref = malloc(nS*8);
        // outlier channels: per-dimension scale log-uniform in [0.1, 10]; Q carries scale*0.25 = 1/64
        double sc[256]; for (int d = 0; d < D; d++) sc[d] = exp(log(0.1) + urand()*log(100.0));
        for (int j = 0; j < k; j++) for (int d = 0; d < D; d++) { size_t i = (size_t)j*D + d; hK[i] = f2h((float)(grand()*sc[d])); dK[i] = h2d(hK[i]); }
        for (int c = 0; c < n; c++) for (int d = 0; d < D; d++) { size_t i = (size_t)c*D + d; hQ[i] = f2h((float)(grand()*sc[d]/64.0)); dQ[i] = h2d(hQ[i]); }
        void * gK16, * gQ16, * gS16, * gK64, * gQ64, * gS64;
        cudaMalloc(&gK16, nK*2); cudaMalloc(&gQ16, nQ*2); cudaMalloc(&gS16, nS*2); cudaMalloc(&gK64, nK*8); cudaMalloc(&gQ64, nQ*8); cudaMalloc(&gS64, nS*8);
        cudaMemcpy(gK16, hK, nK*2, 1); cudaMemcpy(gQ16, hQ, nQ*2, 1); cudaMemcpy(gK64, dK, nK*8, 1); cudaMemcpy(gQ64, dQ, nQ*8, 1);
        const double one = 1.0, zero = 0.0; cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, k, n, D, &one, gK64, D, gQ64, D, &zero, gS64, k);
        cudaDeviceSynchronize(); cudaMemcpy(ref, gS64, nS*8, 2);
        printf("  nt=%d:\n", nt);
        for (int a = 0; a < 8; a++) { double e, ms, ra; if (run(CUBLAS_OP_T, k, n, D, gK16, D, gQ16, D, gS16, ref, algos[a], &e, &ms, &ra))
            printf("    %-8s NMSE %9.2e  rms|dlogit| %9.2e nats  %7.2f ms  %6.2f TFLOPS\n", algos[a] < 0 ? "DEFAULT" : (char[8]){'A','L','G','O','0'+algos[a],0}, e, 4*ra, ms, 2.0*D*(double)n*k/(ms*1e-3)/1e12); }
        cudaFree(gK16); cudaFree(gQ16); cudaFree(gS16); cudaFree(gK64); cudaFree(gQ64); cudaFree(gS64); free(hK); free(hQ); free(dK); free(dQ); free(ref);
    }
    return 0;
}
