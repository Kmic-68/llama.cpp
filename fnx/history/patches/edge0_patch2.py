#!/usr/bin/env python3
# edge0_patch2.py (14. 11. 2026, after round edge0i; the user: "ok ko bos lahko se kar loti dodelav za branje"): the prompt path's
# D and U accumulate as the engine's other prompt products do (FNX_PF_F16ACC=1: f16 accumulation, about twice as fast on GP100);
# FNX_LAT_F32=1 keeps them in f32 (the A/B for the first-token KL). D's output is f16 either way (the expert kernels read f16).
# Applies to the dev tree that has edge0_patch.py; originals to varne-edge0-20261114/patch2/.
import os, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '/srv/build/strata-hkrati/upstream/src/split/src'
SAFE = os.path.join(os.environ.get('EDGE0_SAFE', '/srv/build/strata-hkrati/varne-edge0-20261114'), 'patch2')
f = 'prefill.cu'
t = open(os.path.join(SRC, f), encoding='utf-8').read()
if 'FNX_LAT_F32' in t:
    sys.exit('edge0_patch2.py: already applied')
old = '''    const int slice = (int) std::min<size_t>(rows, PF_WH / cols / 128 * 128);
    const float one = 1.0f, zero = 0.0f;
    for (int r0 = 0; r0 < rows; r0 += slice) {
        const int nr = std::min(slice, rows - r0);
        k_q8s_f16<<<grid_for((size_t) nr * cols), 256, 0, s>>>(w.q + (size_t) r0 * cols, w.d + (size_t) r0 * cols / 32, (size_t) nr * cols, p.wh);
        void* o = f16_out ? (void*) ((__half*) out + r0) : (void*) ((float*) out + r0);
        ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, nr, C, cols, &one, p.wh, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero, o,
                         f16_out ? CUDA_R_16F : CUDA_R_32F, ldo, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "latent gemm");
    }
}
'''
new = '''    const int slice = (int) std::min<size_t>(rows, PF_WH / cols / 128 * 128);
    const float one = 1.0f, zero = 0.0f;
    const __half one_h = __float2half(1.0f), zero_h = __float2half(0.0f);
    // edge0_patch2.py: f16 accumulation as the other prompt products (FNX_PF_F16ACC), unless FNX_LAT_F32=1
    static const bool f32 = std::getenv("FNX_LAT_F32") != nullptr;
    const bool h = p.f16acc && !f32;
    for (int r0 = 0; r0 < rows; r0 += slice) {
        const int nr = std::min(slice, rows - r0);
        k_q8s_f16<<<grid_for((size_t) nr * cols), 256, 0, s>>>(w.q + (size_t) r0 * cols, w.d + (size_t) r0 * cols / 32, (size_t) nr * cols, p.wh);
        if (h && !f16_out) {   // f32 out through pf_gemm's f16 path (its f16 product, then to f32)
            pf_gemm(p, p.wh, nr, cols, X, C, (float*) out + r0, ldo, s);
            continue;
        }
        void* o = f16_out ? (void*) ((__half*) out + r0) : (void*) ((float*) out + r0);
        if (h)
            ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, nr, C, cols, &one_h, p.wh, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero_h, o,
                             CUDA_R_16F, ldo, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "latent gemm f16");
        else
            ckb(cublasGemmEx(p.blas, CUBLAS_OP_T, CUBLAS_OP_N, nr, C, cols, &one, p.wh, CUDA_R_16F, cols, X, CUDA_R_16F, cols, &zero, o,
                             f16_out ? CUDA_R_16F : CUDA_R_32F, ldo, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), "latent gemm");
    }
}
'''
n = t.count(old)
if n != 1:
    sys.exit(f'edge0_patch2.py: lat_dense found {n} times')
t = t.replace(old, new)
# pf_gemm is defined after lat_dense: declare it before
decl = 'void lat_dense(Engine::Pf& p, const Q8S& w, int rows, int cols, const __half* X, int C, void* out, bool f16_out, int ldo, cudaStream_t s) {\n'
assert t.count(decl) == 1
fwd = ('void pf_gemm(Engine::Pf& p, const __half* W, int rows, int cols, const __half* X, int C, float* out, int ldo, cudaStream_t s,\n'
       '             cublasGemmAlgo_t algo);\n')
if t.index('void pf_gemm(Engine::Pf& p') > t.index(decl):
    t = t.replace(decl, fwd + decl)
# FNX_LAT_CHECK=1 (a test switch): after each step of the prompt path's latent MoE, every value finite? The first layer, chunk and step
# with a NaN or inf is printed and the read stops (round edge0i/edge0j: "prefill: ring error 2" at a 10.8k-token prompt).
chk = r'''// edge0_patch2.py, FNX_LAT_CHECK=1: are n values at p (f16 or f32) all finite? A bit `code` in *flag if not
__global__ void k_lat_finite(const void* __restrict__ p, size_t n, int half, unsigned* flag, unsigned code) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const float v = half ? __half2float(((const __half*) p)[i]) : ((const float*) p)[i];
        if (!isfinite(v)) { atomicOr(flag, code); return; }
    }
}
// the codes: 1 xh (D's input), 2 D's rows, 4 gate|up, 8 swiglu, 16 down, 32 z, 64 U
void lat_check(const void* p, size_t n, bool half, unsigned code, int l, int C, cudaStream_t s) {
    static unsigned* flag[16] = {};   // a flag on each card
    int dev = 0;
    cudaGetDevice(&dev);
    if (!flag[dev & 15]) ckc(cudaMalloc(&flag[dev & 15], 4), "lat check");
    unsigned* f = flag[dev & 15];
    ckc(cudaMemsetAsync(f, 0, 4, s), "lat check");
    k_lat_finite<<<grid_for(n), 256, 0, s>>>(p, n, half ? 1 : 0, f, code);
    unsigned h = 0;
    ckc(cudaMemcpyAsync(&h, f, 4, cudaMemcpyDeviceToHost, s), "lat check");
    ckc(cudaStreamSynchronize(s), "lat check");
    if (h) {
        std::fprintf(stderr, "fnx: FNX_LAT_CHECK: card %d layer %d chunk of %d rows: not finite after step %u (1 xh, 2 D, 4 gate|up, 8 swiglu, 16 down, 32 z, 64 U)\n",
                     dev, l, C, h);
        throw std::runtime_error("FNX_LAT_CHECK: a NaN or inf in the latent MoE");
    }
}
'''
anchor = 'void lat_dense(Engine::Pf& p, const Q8S& w, int rows, int cols, const __half* X, int C, void* out, bool f16_out, int ldo, cudaStream_t s) {\n'
t = t.replace(anchor, chk + anchor)
old_br = '''                    lat_dense(p, L.lat_d, LAT_W, N, p.xh, C, p.lat16, true, LAT_W, s);   // xh: the f16 of mixed
                    lat_bigrows(p.jobs_gu, p.njobs, pf_jobs_, L.lat_lab, s);
                    moe_gate_up_mma(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, reinterpret_cast<const uint16_t*>(p.lat16), p.gu, s);
                    lat_swiglu_h16(p.gu, ffs, P, p.h16, p.topi, L.lat_esc, nullptr, s);
                    pf_prof_mark(b, PF_DOWN);
                    moe_down_mma(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.h16, p.part, s);
                    lat_z16(p.part, p.topi, p.topw, L.lat_lab, L.lat_esc, C, reinterpret_cast<uint16_t*>(p.lat16), s);
                    lat_dense(p, L.lat_u, N, LAT_W, p.lat16, C, p.latu, false, N, s);
'''
new_br = '''                    static const bool lchk = std::getenv("FNX_LAT_CHECK") != nullptr;   // edge0_patch2.py
                    if (lchk) lat_check(p.xh, (size_t) C * N, true, 1, (int) l, C, s);
                    lat_dense(p, L.lat_d, LAT_W, N, p.xh, C, p.lat16, true, LAT_W, s);   // xh: the f16 of mixed
                    if (lchk) lat_check(p.lat16, (size_t) C * LAT_W, true, 2, (int) l, C, s);
                    lat_bigrows(p.jobs_gu, p.njobs, pf_jobs_, L.lat_lab, s);
                    moe_gate_up_mma(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, reinterpret_cast<const uint16_t*>(p.lat16), p.gu, s);
                    if (lchk) lat_check(p.gu, (size_t) P * 2 * ffs, false, 4, (int) l, C, s);
                    lat_swiglu_h16(p.gu, ffs, P, p.h16, p.topi, L.lat_esc, nullptr, s);
                    if (lchk) lat_check(p.h16, (size_t) P * ffs, true, 8, (int) l, C, s);
                    pf_prof_mark(b, PF_DOWN);
                    moe_down_mma(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.h16, p.part, s);
                    if (lchk) lat_check(p.part, (size_t) P * LAT_ELL, false, 16, (int) l, C, s);
                    lat_z16(p.part, p.topi, p.topw, L.lat_lab, L.lat_esc, C, reinterpret_cast<uint16_t*>(p.lat16), s);
                    if (lchk) lat_check(p.lat16, (size_t) C * LAT_W, true, 32, (int) l, C, s);
                    lat_dense(p, L.lat_u, N, LAT_W, p.lat16, C, p.latu, false, N, s);
                    if (lchk) lat_check(p.latu, (size_t) C * N, false, 64, (int) l, C, s);
'''
n = t.count(old_br)
if n != 1:
    sys.exit(f'edge0_patch2.py: the latent branch found {n} times')
t = t.replace(old_br, new_br)
os.makedirs(SAFE, exist_ok=True)
shutil.copy2(os.path.join(SRC, f), os.path.join(SAFE, f))
open(os.path.join(SRC, f), 'w', encoding='utf-8', newline='\n').write(t)
print('edge0_patch2.py: prefill.cu patched')
