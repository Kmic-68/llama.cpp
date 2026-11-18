#!/usr/bin/env python3
# edge0_patch3.py (14. 11. 2026, the cause of "prefill: ring error 2" in rounds edge0i-edge0l): the prompt path's overlapped variant
# (passes of several chunks, ring_async) had no latent branch: Edge0's experts ran as plain ones on mixed's 2560 columns. The same
# branch as edge0_patch.py's in the plain variant, and its ring's m.lat. Originals to varne-edge0-20261114/patch3/.
import os, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '/srv/build/strata-hkrati/upstream/src/split/src'
SAFE = os.path.join(os.environ.get('EDGE0_SAFE', '/srv/build/strata-hkrati/varne-edge0-20261114'), 'patch3')
f = 'prefill.cu'
t = open(os.path.join(SRC, f), encoding='utf-8').read()
old = '''                    pf_prof_mark(b, PF_GU);
                    if (p.moe_q8) {
                        moe_gate_up_big(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, p.xq, p.gu, s);
                        moe_down_big(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.gu, p.part, s);
                    } else {'''
new = '''                    pf_prof_mark(b, PF_GU);
                    if (L.lat_d.q) {   // edge0_patch3.py: Edge0's latent MoE, as the plain variant's
                        if (p.moe_q8 || !p.lat16) throw std::runtime_error("FNX_LATENT: the prompt path needs the f16 expert kernels (FNX_PF_MOE_Q8=0)");
                        static const bool lchk = std::getenv("FNX_LAT_CHECK") != nullptr;
                        if (lchk) lat_check(p.xh, (size_t) C * N, true, 1, (int) l, C, s);
                        lat_dense(p, L.lat_d, LAT_W, N, p.xh, C, p.lat16, true, LAT_W, s);
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
                    } else if (p.moe_q8) {
                        moe_gate_up_big(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, p.xq, p.gu, s);
                        moe_down_big(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.gu, p.part, s);
                    } else {'''
n = t.count(old)
if n != 1:
    sys.exit(f'edge0_patch3.py: the overlapped variant found {n} times')
t = t.replace(old, new)
old2 = '''                each([&](int c, CardBuf& b, const CardW&, Pf& p) {
                    RingMoe m;
                    m.part = p.part;
                    m.topw = p.topw;
                    m.logits = p.logits;
                    ring_async(c, b, p, p.bo_f, &m, j, p.ev_rf[j], C);'''
new2 = '''                each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                    RingMoe m;
                    m.part = p.part;
                    m.topw = p.topw;
                    m.logits = p.logits;
                    if (w.L[l].lat_d.q) m.lat = p.latu;   // edge0_patch3.py
                    ring_async(c, b, p, p.bo_f, &m, j, p.ev_rf[j], C);'''
n = t.count(old2)
if n != 1:
    sys.exit(f'edge0_patch3.py: the overlapped ring found {n} times')
t = t.replace(old2, new2)
os.makedirs(SAFE, exist_ok=True)
shutil.copy2(os.path.join(SRC, f), os.path.join(SAFE, f))
open(os.path.join(SRC, f), 'w', encoding='utf-8', newline='\n').write(t)
print('edge0_patch3.py: prefill.cu patched')
