#!/usr/bin/env python3
# prep_patch.py (2026-11-11): FNX_PF_PREP=1 in prefill.cu (the dev tree): k_hcb_prep loads a row's four streams before the first
# block sum (k_hcb_prep4) instead of a stream at a time, so the row's loads are in flight together (the trace: 1.39 ms a call at
# about 320 GB/s, one block a row). Every sum and product in the same order: the same bits. Unset: as before. Saves the original as
# mikro/nvp/prefill.cu.pred-prep once.
import shutil, sys
from pathlib import Path
src = Path("/srv/build/strata-hkrati/upstream/src/split/src/prefill.cu")
bak = Path("/srv/build/strata-hkrati/mikro/nvp/prefill.cu.pred-prep")
code = src.read_text(encoding="utf-8")
if "k_hcb_prep4" in code:
    sys.exit("already patched")
if not bak.exists():
    shutil.copy2(src, bak)
anchor = "// lo = silu(down / HC) as f16\n"
assert code.count(anchor) == 1
code = code.replace(anchor, r"""// FNX_PF_PREP=1 (prep_patch.py): k_hcb_prep with the row's four streams loaded (and written back) before the first block sum
__global__ void __launch_bounds__(HCB_THREADS) k_hcb_prep4(float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj,
                                                           const float* __restrict__ wn, float* __restrict__ rs, __half* __restrict__ xh,
                                                           float eps, const uint16_t* __restrict__ w_inj, float* __restrict__ inject_out) {
    const int t = blockIdx.x, i0 = threadIdx.x;
    __shared__ float sh[32];
    __shared__ float shi[HC][HCB_THREADS / 32];
    float acc[HC] = {};
    float v[HC][HCB_PER];
    float ss[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        ss[c] = 0.f;
        const float g = bo ? 2.0f * pf_sigm(inj[t * HC + c] / (float) HC) : 0.f;
#pragma unroll
        for (int k = 0; k < HCB_PER; ++k) {
            const int i = i0 + k * HCB_THREADS;
            float r = R[(size_t) t * D + c * N + i];
            if (bo) {
                r = fmaf(bo[(size_t) t * N + i], g, r);
                R[(size_t) t * D + c * N + i] = r;
            }
            v[c][k] = r;
            ss[c] += r * r;
        }
    }
#pragma unroll
    for (int c = 0; c < HC; ++c) {
        const float r_s = rsqrtf(pf_block_sum(ss[c], sh) / (float) N + eps);
        if (i0 == 0) rs[t * HC + c] = r_s;
#pragma unroll
        for (int k = 0; k < HCB_PER; ++k) {
            const int i = i0 + k * HCB_THREADS;
            const float xn = v[c][k] * wn[c * N + i] * r_s;
            xh[(size_t) t * D + c * N + i] = __float2half(xn);
            if (w_inj)
#pragma unroll
                for (int r = 0; r < HC; ++r) acc[r] = fmaf(__uint_as_float((unsigned) w_inj[(size_t) r * D + c * N + i] << 16), xn, acc[r]);
        }
    }
    if (!w_inj) return;
    const int lane = i0 & 31, warp = i0 >> 5;
#pragma unroll
    for (int r = 0; r < HC; ++r) {
        const float s = pf_wsum(acc[r]);
        if (lane == 0) shi[r][warp] = s;
    }
    __syncthreads();
    if (i0 < HC) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < HCB_THREADS / 32; ++w) s += shi[i0][w];
        inject_out[t * HC + i0] = s;
    }
}
""" + anchor, 1)
old = "    k_hcb_prep<<<C, HCB_THREADS, 0, s>>>(R, bo, inj_prev, w_norm, p.rs, p.xh, eps, w_inj, inject);   // the inject rows here"
assert code.count(old) == 1
code = code.replace(old, r"""    static const bool prep4 = env_flag("FNX_PF_PREP");   // prep_patch.py
    if (prep4) k_hcb_prep4<<<C, HCB_THREADS, 0, s>>>(R, bo, inj_prev, w_norm, p.rs, p.xh, eps, w_inj, inject);
    else k_hcb_prep<<<C, HCB_THREADS, 0, s>>>(R, bo, inj_prev, w_norm, p.rs, p.xh, eps, w_inj, inject);   // the inject rows here""", 1)
src.write_text(code, encoding="utf-8")
print("patched")
