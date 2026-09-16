// Accuracy AND speed of every cuBLAS COMPUTE_16F GEMM algorithm at the GEMM attention path's PV
// shape: O[DV x n] = V[DV x k] * P[k x n], k = 2048 keys, n = nt*gqa query columns.
// Reference: the same fp16-rounded V and P multiplied in fp64 (cublasDgemm), so the reported
// error is ONLY the fp16 accumulation. P mimics attention after the max offset: every column's
// max is <= 1/8, with diffuse, medium and peaked columns mixed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint16_t f2h(float f){ // round-to-nearest-even float->half (normal + subnormal range)
    if (f == 0.0f) return 0; uint16_t s = f < 0 ? 0x8000 : 0; double a = fabs(f);
    int e; double m = frexp(a, &e); // a = m*2^e, m in [0.5,1)
    int E = e - 1 + 15;             // biased exponent for mantissa in [1,2)
    if (E >= 31) return s | 0x7c00;
    double q; if (E <= 0) { q = nearbyint(a / ldexp(1.0, -24)); if (q >= 1024) return s | (1<<10); return s | (uint16_t) q; }
    q = nearbyint((m*2 - 1) * 1024); if (q == 1024) { q = 0; E++; if (E >= 31) return s | 0x7c00; }
    return s | (uint16_t)(E << 10) | (uint16_t) q; }
static double h2d(uint16_t h){ int s=h>>15, e=(h>>10)&0x1f, m=h&0x3ff; double v = e ? ldexp(1.0+m/1024.0, e-15) : ldexp(m/1024.0, -14); return s ? -v : v; }
static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double urand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0/9007199254740992.0); }
static double grand(void){ double u = urand(), v = urand(); return sqrt(-2*log(u + 1e-300)) * cos(2*M_PI*v); }
int main(int argc, char ** argv){
    const int DV = 256, k = 2048;
    const int nts[2] = {512, 2048};
    cublasHandle_t h; cublasCreate(&h);
    for (int si = 0; si < 2; si++) {
        const int n = nts[si]*6;
        size_t nV = (size_t)DV*k, nP = (size_t)k*n, nO = (size_t)DV*n;
        uint16_t * hV = malloc(nV*2), * hP = malloc(nP*2);
        double * dV = malloc(nV*8), * dP = malloc(nP*8), * ref = malloc(nO*8);
        // V: q4_0-like: per 32-block scale d, values d*(q-8), q in 0..15
        for (size_t i = 0; i < nV; i++) { if (i % 32 == 0) { } }
        for (int j = 0; j < k; j++) for (int b = 0; b < DV; b += 32) { double d = 0.02 + 0.2*urand(); for (int x = 0; x < 32; x++) { int q = (int)(urand()*16); float v = (float)(d*(q-8)); size_t i = (size_t)j*DV + b + x; hV[i] = f2h(v); dV[i] = h2d(hV[i]); } }
        // P columns: 1/3 diffuse (logits ~ N(0,1)), 1/3 medium (N(0,3)), 1/3 peaked (N(0,1) plus one +12 spike)
        for (int c = 0; c < n; c++) {
            double * lg = malloc(k*8); double mx = -1e300; int kind = c % 3; int spike = (int)(urand()*k);
            for (int j = 0; j < k; j++) { lg[j] = grand() * (kind == 1 ? 3.0 : 1.0); if (kind == 2 && j == spike) lg[j] += 12.0; if (lg[j] > mx) mx = lg[j]; }
            for (int j = 0; j < k; j++) { float p = (float)(0.125*exp(lg[j] - mx)); size_t i = (size_t)c*k + j; hP[i] = f2h(p); dP[i] = h2d(hP[i]); }
            free(lg);
        }
        void * gV16, * gP16, * gO16, * gV64, * gP64, * gO64, * gO32;
        cudaMalloc(&gV16, nV*2); cudaMalloc(&gP16, nP*2); cudaMalloc(&gO16, nO*2);
        cudaMalloc(&gV64, nV*8); cudaMalloc(&gP64, nP*8); cudaMalloc(&gO64, nO*8); cudaMalloc(&gO32, nO*4);
        cudaMemcpy(gV16, hV, nV*2, 1); cudaMemcpy(gP16, hP, nP*2, 1); cudaMemcpy(gV64, dV, nV*8, 1); cudaMemcpy(gP64, dP, nP*8, 1);
        const double one = 1.0, zero = 0.0;
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &one, gV64, DV, gP64, k, &zero, gO64, DV);
        cudaDeviceSynchronize(); cudaMemcpy(ref, gO64, nO*8, 2);
        double refss[3] = {0,0,0}; for (size_t i = 0; i < nO; i++) refss[(i/DV) % 3] += ref[i]*ref[i];
        printf("=== PV shape: DV=%d k=%d n=%d (nt=%d x gqa 6)\n", DV, k, n, nts[si]);
        printf("  %-22s %9s %8s | %10s %10s %10s %10s\n", "variant", "ms", "TFLOPS", "NMSE all", "diffuse", "medium", "peaked");
        uint16_t * o16 = malloc(nO*2); float * o32 = malloc(nO*4);
        for (int kind = 0; kind < 2; kind++)
        for (int a = -1; a <= 23; a++) {
            if (kind == 1 && a > -1) break;
            cublasGemmAlgo_t al = a < 0 ? CUBLAS_GEMM_DEFAULT : (cublasGemmAlgo_t)(CUBLAS_GEMM_ALGO0 + a);
            const uint16_t a16 = 0x3c00, b16 = 0; const float a32 = 1.0f, b32 = 0.0f;
            double best = 1e9; cublasStatus_t st = 0;
            for (int r = 0; r < 5; r++) {
                cudaDeviceSynchronize(); double t0 = now();
                if (kind == 0) st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &a16, gV16, CUDA_R_16F, DV, gP16, CUDA_R_16F, k, &b16, gO16, CUDA_R_16F, DV, CUBLAS_COMPUTE_16F, al);
                else           st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, DV, n, k, &a32, gV16, CUDA_R_16F, DV, gP16, CUDA_R_16F, k, &b32, gO32, CUDA_R_32F, DV, CUBLAS_COMPUTE_32F, al);
                cudaDeviceSynchronize(); double dt = now() - t0;
                if (st != CUBLAS_STATUS_SUCCESS) break;
                if (r > 0 && dt < best) best = dt;
            }
            if (st != CUBLAS_STATUS_SUCCESS) continue;
            double ess[3] = {0,0,0};
            if (kind == 0) { cudaMemcpy(o16, gO16, nO*2, 2); for (size_t i = 0; i < nO; i++) { double e = h2d(o16[i]) - ref[i]; ess[(i/DV) % 3] += e*e; } }
            else           { cudaMemcpy(o32, gO32, nO*4, 2); for (size_t i = 0; i < nO; i++) { double e = (double) o32[i] - ref[i]; ess[(i/DV) % 3] += e*e; } }
            char name[64]; snprintf(name, sizeof name, "%s %s", kind == 0 ? "16F" : "32F(16F in)", a < 0 ? "DEFAULT" : (snprintf((char[8]){0}, 8, "x"), ""));
            if (a >= 0) snprintf(name, sizeof name, "16F ALGO%d", a);
            double fl = 2.0*DV*(double)n*k;
            printf("  %-22s %9.2f %8.2f | %10.3e %10.3e %10.3e %10.3e\n", name, best*1e3, fl/best/1e12,
                   (ess[0]+ess[1]+ess[2])/(refss[0]+refss[1]+refss[2]), ess[0]/refss[0], ess[1]/refss[1], ess[2]/refss[2]);
            fflush(stdout);
        }
        cudaFree(gV16); cudaFree(gP16); cudaFree(gO16); cudaFree(gV64); cudaFree(gP64); cudaFree(gO64); cudaFree(gO32);
        free(hV); free(hP); free(dV); free(dP); free(ref); free(o16); free(o32);
    }
    return 0;
}
