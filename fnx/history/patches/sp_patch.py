#!/usr/bin/env python3
# sp_patch.py SRC_DIR: idea 58 (2026-11-12, NACRT-ideja58-hc-po-vrsticah.md). FNX_PF_SP=1: in a prompt pass of two chunks and more,
# from layer PLE_LAYER on each card reads hc only for the rows of the ring part it finishes (pf_ring_plan: card g finishes part g + 1),
# the two cards trade their mixed rows through host memory on a stream of their own (pf_gather), and each layer's sums stop after their
# reduce-scatter (a card needs its own rows' sums only; the ring's second flag still tells the sender its slot was read). The next
# chunk's hc read and gather go before a chunk's half, so the gather crosses the link while that half computes; mixed alternates between
# two buffers and each half makes xh again from its chunk's mixed. Layer 0 and PLE_LAYER's write run on every row as before; the draft
# layer's source R is gathered whole after the last layer. The same sums in the same order: a part's two hc products take the cuBLAS
# algorithm that gives the whole chunk's default's bits (sp_algo; mikro/k58/gemm_pol3: 0 differences at every C from 2 to 4096 with
# production's math mode). Off (FNX_PF_SP unset or 0): the pass as before, every line of it. Each edit must match as counted.
import sys

src = sys.argv[1]


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1)
    open(p, "w").write(s)


# ---- pf_common.cuh: the ring's reduce-scatter alone
edit("pf_common.cuh", [(
    "template <typename Ring>\n"
    "inline void pf_ring_run(const Ring& r, int dev, int g, float* x, int C, unsigned seq, unsigned* err, cudaStream_t s) {\n"
    "    if (r.plan.empty() || r.plan.front().part != g) throw std::runtime_error(\"prefill: the ring's plan is not card \" + std::to_string(g) + \"'s\");\n",
    "// rs_only (FNX_PF_SP, sp_patch.py): the reduce-scatter alone: this card's finished part back into x (the other parts of x keep\n"
    "// what they held), no finished part sent on; the step flags still go, so a card's next sum into a set waits until the other\n"
    "// card has read its slot. The ring plan only (not the direct gather)\n"
    "template <typename Ring>\n"
    "inline void pf_ring_run(const Ring& r, int dev, int g, float* x, int C, unsigned seq, unsigned* err, cudaStream_t s, bool rs_only = false) {\n"
    "    if (r.plan.empty() || r.plan.front().part != g) throw std::runtime_error(\"prefill: the ring's plan is not card \" + std::to_string(g) + \"'s\");\n"
    "    if (rs_only)\n"
    "        for (const PfRingOp& o : r.plan)\n"
    "            if (o.k == PfRingOp::GADD) throw std::runtime_error(\"prefill: the reduce-scatter alone takes the ring plan, not the direct gather\");\n",
    1,
), (
    "            case PfRingOp::PASS:\n"
    "                if (rows(o.part) > 0) {\n",
    "            case PfRingOp::PASS:\n"
    "                if (rs_only) break;\n"
    "                if (rows(o.part) > 0) {\n",
    1,
), (
    "            case PfRingOp::OUT:\n"
    "                // the parts that arrived (everything but the one finished here) go to device memory by DMA, then one kernel reads it all\n",
    "            case PfRingOp::OUT:\n"
    "                if (rs_only) {\n"
    "                    if (rows(fin_part) > 0)\n"
    "                        k_pfr_out<<<grid_for((size_t) rows(fin_part) * N / 2), 256, 0, s>>>((const __half2*) (own_ag + (size_t) r0(fin_part) * N),\n"
    "                                                                                        (float2*) (x + (size_t) r0(fin_part) * N), rows(fin_part) * N / 2);\n"
    "                    break;\n"
    "                }\n"
    "                // the parts that arrived (everything but the one finished here) go to device memory by DMA, then one kernel reads it all\n",
    1,
)])

SP_LOOP = r'''    // FNX_PF_SP=1 (idea 58, sp_patch.py; NACRT-ideja58-hc-po-vrsticah.md): a pass of two chunks and more reads hc, from layer PLE_LAYER
    // on, on each card for the rows of the ring part it finishes only; the cards trade those mixed rows on ag_st (pf_gather) and the
    // layer's sums stop after their reduce-scatter (ring_rs). A half's next chunk's hc read and gather go first, so the gather crosses
    // the link while this chunk's half computes; mixed alternates between mixk[0] and mixk[1] by the chunk, and each half makes xh again
    // from its chunk's mixed. Layer 0 and PLE_LAYER's write run on every row as before; the draft layer's source R is gathered whole.
    bool sp_pass = false;
    {
        static const bool draft8_sp = env_flag("FNX_PF_DRAFT8");
        const Pf& p0 = *pf_[0];
        sp_pass = p0.sp && nk >= 2 && p0.f16acc && !draft8_sp && !p0.ring_ce.direct && !p0.rccl;
    }
    if (sp_pass) {
        for (int l = 0; l < NL; ++l) {
            if (staging && l + 1 < NL - 1) stage_in(l + 1);   // the last layer runs no experts
            const bool spl = l >= PLE_LAYER;   // this layer's hc reads on the card's own rows, its sums a reduce-scatter
            ring_rs = spl;
            // ---- attention half: a chunk's hc read on its own rows, its mixed gathered (issued a chunk ahead of the half)
            auto att_pre = [&](int j) {
                use(j);
                each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                    const LayerW& L = w.L[l];
                    cudaStream_t s = b.st;
                    const bool pending = l > 0;
                    if (pending) ckc(cudaStreamWaitEvent(s, p.ev_rf[j], 0), "ffn sum");
                    pf_prof_mark(b, PF_HC);
                    bool apply = pending;
                    const int own = (c + 1) % NCARD, r0 = own * C / NCARD, nr = (own + 1) * C / NCARD - r0;
                    if (l == PLE_LAYER) {   // every row: the sum before it was whole
                        ckc(cudaMemcpyAsync(p.ple_emb, pf_ple_[set], (size_t) C * N * 4, cudaMemcpyHostToDevice, s), "ple");
                        hc_write(p.R, p.bo_f, p.inj_f, C, s);
                        apply = false;
                        pf_dense(p, w.ple_key.s, w.ple_key.n_out, w.ple_key.n_in, p.ple_emb, C, p.ple_key, w.ple_key.n_out, s);
                        pf_dense(p, w.ple_value.s, w.ple_value.n_out, w.ple_value.n_in, p.ple_emb, C, p.ple_val, w.ple_value.n_out, s);
                        for (int g = 0; g < ng; ++g) {
                            const int q0 = g * MAXT, Tg = std::min(MAXT, C - q0);
                            ple_post(p.R + (size_t) q0 * D, p.ple_key + (size_t) q0 * D, p.ple_val + (size_t) q0 * N, w.ple_norm_key,
                                     w.ple_norm_query, w.ple_norm_conv, w.ple_conv, b.ple_hist, b.ple_hsnap, Tg, eps, s, p.ple_tmp);
                        }
                    }
                    if (apply && cvec_.covers(l - 1)) {   // FNX_CVEC: on the rows this card holds
                        if (spl) {
                            if (nr > 0) {
                                hc_write(p.R + (size_t) r0 * D, p.bo_f + (size_t) r0 * N, p.inj_f + (size_t) r0 * HC, nr, s);
                                cvec_apply(p.R + (size_t) r0 * D, b.cvec + (size_t) (l - 1) * N, cvec_.s[(size_t) (l - 1)], cvec_.add, nr, s);
                            }
                        } else {
                            hc_write(p.R, p.bo_f, p.inj_f, C, s);
                            cvec_apply(p.R, b.cvec + (size_t) (l - 1) * N, cvec_.s[(size_t) (l - 1)], cvec_.add, C, s);
                        }
                        apply = false;
                    }
                    float* const mx = p.mixk[j & 1];
                    if (spl) {
                        if (nr > 0)
                            pf_hc_read(p, p.R + (size_t) r0 * D, apply ? p.bo_f + (size_t) r0 * N : nullptr, p.inj_f + (size_t) r0 * HC,
                                       L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, mx + (size_t) r0 * N, p.inj_a + (size_t) r0 * HC,
                                       nr, s, C);
                        ckc(cudaEventRecord(p.ev_hcd[j], s), "hc read done");
                        ckc(cudaStreamWaitEvent(p.ag_st, p.ev_hcd[j], 0), "gather after the hc read");
                        pf_gather(p, c, mx, N, C, p.ag_st);
                        ckc(cudaEventRecord(p.ev_agd[j], p.ag_st), "gather done");
                    } else {
                        pf_hc_read(p, p.R, apply ? p.bo_f : nullptr, p.inj_f, L.hc_norm[0], L.hc_down[0], L.hc_up[0], L.hc_inj[0], eps, mx,
                                   p.inj_a, C, s);
                    }
                });
            };
            auto att_post = [&](int j) {
                use(j);
                each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                    const LayerW& L = w.L[l];
                    cudaStream_t s = b.st;
                    if (spl) ckc(cudaStreamWaitEvent(s, p.ev_agd[j], 0), "mixed gathered");
                    p.mixed = p.mixk[j & 1];
                    k_f32_f16<<<grid_for((size_t) C * N), 256, 0, s>>>(p.mixed, (size_t) C * N, p.xh);   // the f16 of mixed (the next chunk's hc read used xh)
                    if (!is_attn(l)) {
                        const int nv = w.nv(), ch = w.conv_ch(), rs_qkvz = ch + nv * HS;
                        pf_prof_mark(b, PF_QKVZ);
                        pf_dense_xh(p, L.qkvz.s, rs_qkvz, N, C, p.qkvz, rs_qkvz, s);
                        pf_prof_mark(b, PF_AB);
                        pf_sgemm(p, L.ab, 2 * nv, N, p.mixed, C, p.ab, 2 * nv);
                        pf_prof_mark(b, PF_GDN);
                        gdn_prompt(p.qkvz, p.ab, b.chist[l], L.conv, L.dt_bias, L.ssm_a, L.ssm_norm, b.S[l], p.ctl, ng, w.nk, nv, w.vmap, C, eps,
                                   p.gdn_xc, p.gdn_sq, p.y32, p.gdn_ch, s);
                        pf_prof_mark(b, PF_OUT);
                        pf_dense(p, L.ssm_out.s, N, nv * HS, p.y32, C, p.bo_a, N, s);
                    } else {
                        pf_prof_mark(b, PF_WO);
                        finish_att(c, b, w, p, l);   // the previous chunk's, before qg is this chunk's
                        pf_prof_mark(b, PF_QKV);
                        pf_dense(p, L.wq.s, NQH * 2 * HD, N, p.mixed, C, p.qg, NQH * 2 * HD, s);
                        pf_dense_xh(p, L.wk.s, NKVH * HD, N, C, p.k, NKVH * HD, s);
                        pf_dense_xh(p, L.wv.s, NKVH * HD, N, C, p.v, NKVH * HD, s);
                        attn_prep(p.qg, p.k, L.q_norm, L.k_norm, NQH, NKVH, p.ctl, C, eps, rope_of(b.mrope), s);
                        kv_append(p.k, p.v, b.kc[l], b.vc[l], b.pg, b.att_ring.card, C, p.ctl, s);
                        pf_prof_mark(b, PF_IDX);
                        k_bf16_f32<<<grid_for((size_t) 512 * N), 256, 0, s>>>(L.idx_wq, (size_t) 512 * N, p.widx);
                        k_bf16_f32<<<grid_for((size_t) 128 * N), 256, 0, s>>>(L.idx_wk, (size_t) 128 * N, p.widx + (size_t) 512 * N);
                        pf_sgemm(p, p.widx, qsa_rows(), N, p.mixed, C, p.yidx, qsa_rows());
                        qsa_index(p.yidx, L.idx_qn, L.idx_kn, b.qsa_ring[l], b.qsa_pooled[l], b.pg, p.qidx, p.ctl, C, eps, rope_of(b.mrope), s);
                        pf_prof_mark(b, PF_SEL);
                        const int AR = p.attn_dec ? attn_rows() : p.qsa_pf_rows;
                        for (int g = 0; g * AR < C; ++g) {
                            const int q0 = g * AR, Tg = std::min(AR, C - q0);
                            const int* gctl = p.ctl + 2 + 2 * (q0 / MAXT);
                            qsa_select(p.qidx + (size_t) q0 * 512, b.qsa_pooled[l], b.pg, p.scores, p.cells + (size_t) q0 * qsa_maxsel(),
                                       p.ncells + q0, max_ctx_ / 4, gctl, Tg, b.att_ring.card, s);
                            if (!p.attn_dec) continue;
                            attn(p.qg + (size_t) q0 * NQH * 2 * HD, b.kc[l], b.vc[l], b.pg, p.att + (size_t) q0 * w.nq * HD, b.att_ring,
                                 (const unsigned*) (gctl + 1), 128 + l, gctl, Tg, b.att_part, s, p.cells + (size_t) q0 * qsa_maxsel(),
                                 p.ncells + q0);
                        }
                        pf_prof_mark(b, PF_ATT);
                        if (!p.attn_dec) {
                            pf_attn_start(p, b, b.kc[l], b.vc[l], c, C, j & 1, s);   // merged by finish_att
                            p.att_pend = j, p.att_pend_rows = C;
                        } else {
                            pf_prof_mark(b, PF_WO);
                            pf_dense(p, L.wo.s, N, w.nq * HD, p.att, C, p.bo_a, N, s);
                        }
                    }
                });
                if (l == NL - 1 && !mtp_) return;   // the last layer's FFN half changes no state
                each([&](int c, CardBuf& b, const CardW&, Pf& p) {
                    pf_prof_mark(b, PF_SUM);
                    if (!is_attn(l) || p.attn_dec) ring_async(c, b, p, p.bo_a, nullptr, j, p.ev_ra[j], C);   // else finish_att's
                });
            };
            att_pre(0);
            for (int j = 0; j < nk; ++j) {
                if (j + 1 < nk) att_pre(j + 1);
                att_post(j);
            }
            if (is_attn(l)) each([&](int c, CardBuf& b, const CardW& w, Pf& p) { finish_att(c, b, w, p, l); });   // the last chunk
            if (l == NL - 1 && !mtp_) continue;
            // ---- FFN half
            auto ffn_pre = [&](int j) {
                use(j);
                each([&](int c, CardBuf& b, const CardW& w, Pf& p) {
                    const LayerW& L = w.L[l];
                    cudaStream_t s = b.st;
                    ckc(cudaStreamWaitEvent(s, p.ev_ra[j], 0), "attention sum");
                    pf_prof_mark(b, PF_HC);
                    const int own = (c + 1) % NCARD, r0 = own * C / NCARD, nr = (own + 1) * C / NCARD - r0;
                    float* const mx = p.mixk[j & 1];
                    if (!spl) {
                        pf_hc_read(p, p.R, p.bo_a, p.inj_a, L.hc_norm[1], L.hc_down[1], L.hc_up[1], L.hc_inj[1], eps, mx, p.inj_f, C, s);
                        return;
                    }
                    if (nr > 0)
                        pf_hc_read(p, p.R + (size_t) r0 * D, p.bo_a + (size_t) r0 * N, p.inj_a + (size_t) r0 * HC, L.hc_norm[1], L.hc_down[1],
                                   L.hc_up[1], L.hc_inj[1], eps, mx + (size_t) r0 * N, p.inj_f + (size_t) r0 * HC, nr, s, C);
                    if (l == NL - 1) {   // the draft layer's source: R with the last attention write folded in, whole on both cards
                        pf_gather(p, c, p.R, D, C, s);
                        return;
                    }
                    ckc(cudaEventRecord(p.ev_hcd[j], s), "hc read done");
                    ckc(cudaStreamWaitEvent(p.ag_st, p.ev_hcd[j], 0), "gather after the hc read");
                    pf_gather(p, c, mx, N, C, p.ag_st);
                    ckc(cudaEventRecord(p.ev_agd[j], p.ag_st), "gather done");
                });
            };
            auto ffn_post = [&](int j) {
                use(j);
                each([&](int, CardBuf& b, const CardW& w, Pf& p) {
                    const LayerW& L = w.L[l];
                    cudaStream_t s = b.st;
                    const int P = C * TOPK, ffs = w.ff1 - w.ff0;
                    DW gu_e = L.gu_e, down_e = L.down_e;
                    if (staging) {
                        gu_e.ep = p.ep_gu + (size_t) l * NEXP, down_e.ep = p.ep_dn + (size_t) l * NEXP;
                        if (j == 0) ckc(cudaStreamWaitEvent(s, p.ev_ready[l & 1], 0), "stage in");
                    }
                    if (spl) ckc(cudaStreamWaitEvent(s, p.ev_agd[j], 0), "mixed gathered");
                    p.mixed = p.mixk[j & 1];
                    k_f32_f16<<<grid_for((size_t) C * N), 256, 0, s>>>(p.mixed, (size_t) C * N, p.xh);   // the f16 of mixed
                    if (p.moe_q8) quantize_q8_1(p.mixed, p.xq, N, C, s);
                    pf_prof_mark(b, PF_ROUTER);
                    k_bf16_f32<<<grid_for((size_t) (NEXP + 1) * N), 256, 0, s>>>(L.router, (size_t) (NEXP + 1) * N, p.router32);
                    pf_sgemm(p, p.router32, NEXP + 1, N, p.mixed, C, p.logits, NEXP + 1);
                    k_pf_top10<<<(C + 7) / 8, 256, 0, s>>>(p.logits, C, p.topi, p.topw);
                    if (memo_ && &b == cb_[0]) memo_put_rows(p.topi, C, pos0[(size_t) j], l, memo_, memo_cap_, s);   // memo_patch.py
                    ckc(cudaMemsetAsync(p.cnt, 0, NEXP * 4, s), "cnt");
                    k_pf_count<<<(P + 255) / 256, 256, 0, s>>>(p.topi, P, p.cnt);
                    if (b.route_cnt) route_count(p.topi, P, b.route_cnt + (size_t) (NL + l) * NEXP, s);
                    k_pf_jobbase<<<1, NEXP, 0, s>>>(p.cnt, p.jbase, p.njobs);
                    k_pf_fill<<<NEXP, PF_FILL_THREADS, 0, s>>>(p.topi, P, p.cnt, p.jbase, p.jobs_gu, p.jobs_dn);
                    pf_prof_mark(b, PF_GU);
                    if (p.moe_q8) {
                        moe_gate_up_big(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, p.xq, p.gu, s);
                        moe_down_big(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.gu, p.part, s);
                    } else {
                        moe_gate_up_mma(gu_e, L.gu_e_stride, p.jobs_gu, p.njobs, pf_jobs_, reinterpret_cast<const uint16_t*>(p.xh), p.gu, s);
                        swiglu_h16(p.gu, ffs, P, p.h16, s);
                        pf_prof_mark(b, PF_DOWN);
                        moe_down_mma(down_e, L.down_e_stride, p.jobs_dn, p.njobs, pf_jobs_, p.h16, p.part, s);
                    }
                    if (staging && j == nk - 1) ckc(cudaEventRecord(p.ev_done[l & 1], s), "stage done");
                    pf_prof_mark(b, PF_SHARED);
                    pf_dense_xh(p, L.sh_gu.s, 2 * ffs, N, C, p.shg, 2 * ffs, s);   // xh: still the f16 of mixed
                    k_pf_swiglu<<<grid_for((size_t) C * ffs), 256, 0, s>>>(p.shg, ffs, C, p.shh);
                    k_q8s_f16<<<grid_for((size_t) N * ffs), 256, 0, s>>>(L.sh_down.s.q, L.sh_down.s.d, (size_t) N * ffs, p.wh);
                    pf_gemm(p, p.wh, N, ffs, p.shh, C, p.part + (size_t) P * N, N, s);
                    ckc(cudaGetLastError(), "ffn");
                });
                each([&](int c, CardBuf& b, const CardW&, Pf& p) {
                    RingMoe m;
                    m.part = p.part;
                    m.topw = p.topw;
                    m.logits = p.logits;
                    ring_async(c, b, p, p.bo_f, &m, j, p.ev_rf[j], C);
                });
            };
            if (l == NL - 1) {
                for (int j = 0; j < nk; ++j) ffn_pre(j);
                continue;
            }
            ffn_pre(0);
            for (int j = 0; j < nk; ++j) {
                if (j + 1 < nk) ffn_pre(j + 1);
                ffn_post(j);
            }
        }
        each([&](int, CardBuf&, const CardW&, Pf& p) { p.mixed = p.mixk[0]; });
    } else
'''

GATHER = r'''
// FNX_PF_SP (sp_patch.py): rows of base (row_floats floats a row) for a chunk of C rows on stream s: this card's part (the ring part it
// finishes, pf_ring_plan) goes to the other card's host rows by the copy engine, the other card's part comes from this card's host rows
// into base. Flags [set][0]: the rows are in; [set][1]: the receiver has them (the sender's next use of the set waits for that). Every
// card makes the same calls in the same order (seq)
void pf_gather(Engine::Pf& p, int g, float* base, int row_floats, int C, cudaStream_t s) {
    static_assert(NCARD == 2, "pf_gather: two cards");
    const unsigned seq = ++p.gseq;
    const int set = (int) (seq & 1u), own = (g + 1) % NCARD, oth = g;
    auto r0 = [&](int k) { return k * C / NCARD; };
    auto rows = [&](int k) { return r0(k + 1) - r0(k); };
    CUstream cs = (CUstream) s;
    if (seq > 2) cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.gflags + set * 2 + 1), seq - 2, CU_STREAM_WAIT_VALUE_GEQ), "gather consumed");
    if (rows(own) > 0)
        ckc(cudaMemcpyAsync(p.gnhb[set], base + (size_t) r0(own) * row_floats, (size_t) rows(own) * row_floats * sizeof(float), cudaMemcpyDefault, s),
            "gather send");
    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gnflags + set * 2), seq, 0), "gather flag");
    cud(cuStreamWaitValue32(cs, (CUdeviceptr) (p.gflags + set * 2), seq, CU_STREAM_WAIT_VALUE_GEQ), "gather wait");
    if (rows(oth) > 0)
        ckc(cudaMemcpyAsync(base + (size_t) r0(oth) * row_floats, p.ghb[set], (size_t) rows(oth) * row_floats * sizeof(float), cudaMemcpyDefault, s),
            "gather in");
    cud(cuStreamWriteValue32(cs, (CUdeviceptr) (p.gnflags + set * 2 + 1), seq, 0), "gather read");
}
'''

edit("prefill.cu", [
    # the Pf members
    (
        "    cudaEvent_t ev_go[Engine::PF_PASS], ev_ra[Engine::PF_PASS], ev_rf[Engine::PF_PASS];\n",
        "    cudaEvent_t ev_go[Engine::PF_PASS], ev_ra[Engine::PF_PASS], ev_rf[Engine::PF_PASS];\n"
        "    // FNX_PF_SP=1 (idea 58, sp_patch.py): see the pass's note. mixk: two chunks' mixed; ghb: host rows the other card's gather lands in\n"
        "    // (rows of D, for the draft source's R too), gnhb: the other card's; gflags [2 sets][rows in, rows read], gnflags: the other card's;\n"
        "    // ag_st: the gathers' stream; ev_hcd / ev_agd: a chunk's hc read / its gather done\n"
        "    bool sp = false;\n"
        "    float* mixk[2] = {};\n"
        "    float *ghb[2] = {}, *gnhb[2] = {};\n"
        "    unsigned *gflags = nullptr, *gnflags = nullptr;\n"
        "    unsigned gseq = 0;\n"
        "    cudaStream_t ag_st = nullptr;\n"
        "    cudaEvent_t ev_hcd[Engine::PF_PASS], ev_agd[Engine::PF_PASS];\n",
        1,
    ),
    # pf_gemm's algorithm
    (
        "void pf_gemm(Engine::Pf& p, const __half* W, int rows, int cols, const __half* X, int C, float* out, int ldo, cudaStream_t s) {\n",
        "// algo: CUBLAS_GEMM_DEFAULT_TENSOR_OP, or sp_algo's for an hc read on a part of a chunk (FNX_PF_SP)\n"
        "void pf_gemm(Engine::Pf& p, const __half* W, int rows, int cols, const __half* X, int C, float* out, int ldo, cudaStream_t s,\n"
        "             cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {\n",
        1,
    ),
    (
        "CUDA_R_16F, rows, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), \"gemm f16\");\n",
        "CUDA_R_16F, rows, CUBLAS_COMPUTE_16F, algo), \"gemm f16\");\n",
        1,
    ),
    (
        "CUDA_R_32F, ldo, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), \"gemm\");\n",
        "CUDA_R_32F, ldo, CUBLAS_COMPUTE_32F, algo), \"gemm\");\n",
        1,
    ),
    # pf_hc_read on a part
    (
        "// the hyper-connection read for C rows (see hc.cuh for what it computes)\n"
        "void pf_hc_read(Engine::Pf& p, float* R, const float* bo, const float* inj_prev, const float* w_norm, const Q8S& w_down, const Q8S& w_up,\n"
        "                const uint16_t* w_inj, float eps, float* mixed, float* inject, int C, cudaStream_t s) {\n",
        "// FNX_PF_SP: for a part of a chunk of C rows, the algorithm of the hc read's down (or up) product whose bits for those rows are the whole\n"
        "// chunk's default's: cuBLAS's default picks another kernel for half the rows at C 17..33 and 3073..4096 (down), 17..33 (up), and\n"
        "// these were equal at every C (mikro/k58/gemm_pol2, gemm_pol3 with production's math mode: 0 differences from 2 to 4096 rows)\n"
        "cublasGemmAlgo_t sp_algo(bool down, int C) {\n"
        "    if (C >= 17 && C <= 33) return down ? CUBLAS_GEMM_ALGO4 : CUBLAS_GEMM_ALGO1;\n"
        "    if (down && C >= 3073) return CUBLAS_GEMM_ALGO1;\n"
        "    return CUBLAS_GEMM_DEFAULT_TENSOR_OP;\n"
        "}\n"
        "// the hyper-connection read for C rows (see hc.cuh for what it computes); sp_c: these rows are a part of a chunk of sp_c rows (FNX_PF_SP)\n"
        "void pf_hc_read(Engine::Pf& p, float* R, const float* bo, const float* inj_prev, const float* w_norm, const Q8S& w_down, const Q8S& w_up,\n"
        "                const uint16_t* w_inj, float eps, float* mixed, float* inject, int C, cudaStream_t s, int sp_c = 0) {\n"
        "    const cublasGemmAlgo_t a_down = sp_c ? sp_algo(true, sp_c) : CUBLAS_GEMM_DEFAULT_TENSOR_OP,\n"
        "                           a_up = sp_c ? sp_algo(false, sp_c) : CUBLAS_GEMM_DEFAULT_TENSOR_OP;\n",
        1,
    ),
    (
        "    pf_gemm(p, p.wh, LR, D, p.xh, C, p.lo32, LR, s);\n",
        "    pf_gemm(p, p.wh, LR, D, p.xh, C, p.lo32, LR, s, a_down);\n",
        1,
    ),
    (
        "                         reinterpret_cast<__half*>(p.up32), CUDA_R_16F, D, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), \"gemm f16 (hc up)\");\n",
        "                         reinterpret_cast<__half*>(p.up32), CUDA_R_16F, D, CUBLAS_COMPUTE_16F, a_up), \"gemm f16 (hc up)\");\n",
        1,
    ),
    (
        "                         CUDA_R_16F, D, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), \"gemm f16 (hc up)\");\n",
        "                         CUDA_R_16F, D, CUBLAS_COMPUTE_16F, a_up), \"gemm f16 (hc up)\");\n",
        1,
    ),
    (
        "                         CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP), \"gemm (hc up)\");\n",
        "                         CUDA_R_32F, D, CUBLAS_COMPUTE_32F, a_up), \"gemm (hc up)\");\n",
        1,
    ),
    # pf_ring's reduce-scatter alone, and pf_gather after pf_ring
    (
        "void pf_ring(Engine::Pf& p, int g, float* x, int C, const RingMoe* m, unsigned* err, cudaStream_t s) {\n",
        "void pf_ring(Engine::Pf& p, int g, float* x, int C, const RingMoe* m, unsigned* err, cudaStream_t s, bool rs_only = false) {\n",
        1,
    ),
    (
        "    pf_ring_run(p.ring_ce, p.dev, g, x, C, seq, err, s);\n"
        "    ckc(cudaGetLastError(), \"ring\");\n"
        "}\n",
        "    pf_ring_run(p.ring_ce, p.dev, g, x, C, seq, err, s, rs_only);\n"
        "    ckc(cudaGetLastError(), \"ring\");\n"
        "}\n" + GATHER,
        1,
    ),
    # the sequence reset clears the gathers' flags too
    (
        "        p.seq = p.att_seq[0] = p.att_seq[1] = start;\n",
        "        p.seq = p.att_seq[0] = p.att_seq[1] = start;\n"
        "        if (p.sp) {   // FNX_PF_SP: the gathers count from 0 again\n"
        "            ckc(cudaMemset(p.gflags, 0, 4 * sizeof(unsigned)), \"gather flags\");\n"
        "            p.gseq = 0;\n"
        "        }\n",
        1,
    ),
    # the buffers
    (
        "        if (w.ec.n_slots) {   // the stages themselves: prefill_stages, once the experts' places are final\n",
        "        // FNX_PF_SP (idea 58, sp_patch.py): a second mixed, the gathers' host rows (half a chunk of rows of D, two sets), flags, stream\n"
        "        if (env_int(\"FNX_PF_SP\", 0) != 0 && NCARD == 2 && p->overlap && !p->ring_sm) {\n"
        "            p->sp = true;\n"
        "            p->mixk[0] = p->mixed;\n"
        "            p->mixk[1] = palloc<float>((size_t) pf_c_ * N);\n"
        "            for (int k = 0; k < 2; ++k) p->ghb[k] = hpalloc<float>((size_t) (pf_c_ + 1) / 2 * D);\n"
        "            p->gflags = hpalloc<unsigned>(4);\n"
        "            ckc(cudaStreamCreateWithFlags(&p->ag_st, cudaStreamNonBlocking), \"gather stream\");\n"
        "            own_stream(p->ag_st);\n"
        "            for (int k = 0; k < PF_PASS; ++k)\n"
        "                for (cudaEvent_t* e : {&p->ev_hcd[k], &p->ev_agd[k]}) {\n"
        "                    ckc(cudaEventCreateWithFlags(e, cudaEventDisableTiming), \"event\");\n"
        "                    own_event(*e);\n"
        "                }\n"
        "            std::fprintf(stderr, \"fnx: FNX_PF_SP: card %d reads hc on its ring part of a chunk's rows (passes of 2 chunks and more)\\n\", c);\n"
        "        }\n"
        "        if (w.ec.n_slots) {   // the stages themselves: prefill_stages, once the experts' places are final\n",
        1,
    ),
    (
        "    for (int c = 0; c < NCARD; ++c) {\n"
        "        Pf& p = *pf_[c];\n"
        "        const Pf& n = *pf_[(c + 1) % NCARD];\n"
        "        p.ring.next_in = n.ring.in;\n",
        "    for (int c = 0; c < NCARD; ++c) {   // FNX_PF_SP: each card's gathers land in the other card's host rows\n"
        "        Pf& p = *pf_[c];\n"
        "        const Pf& n = *pf_[(c + 1) % NCARD];\n"
        "        if (p.sp != n.sp) throw std::runtime_error(\"FNX_PF_SP: on every card or on none\");\n"
        "        if (!p.sp) continue;\n"
        "        for (int k = 0; k < 2; ++k) p.gnhb[k] = n.ghb[k];\n"
        "        p.gnflags = n.gflags;\n"
        "    }\n"
        "    for (int c = 0; c < NCARD; ++c) {\n"
        "        Pf& p = *pf_[c];\n"
        "        const Pf& n = *pf_[(c + 1) % NCARD];\n"
        "        p.ring.next_in = n.ring.in;\n",
        1,
    ),
    # the sums' mode
    (
        "    auto ring_async = [&](int c, CardBuf& b, Pf& p, float* x, const RingMoe* m, int j, cudaEvent_t done, int rows) {\n",
        "    bool ring_rs = false;   // FNX_PF_SP: the split layers' sums stop after their reduce-scatter\n"
        "    auto ring_async = [&](int c, CardBuf& b, Pf& p, float* x, const RingMoe* m, int j, cudaEvent_t done, int rows) {\n",
        1,
    ),
    (
        "        pf_ring(p, c, x, rows, nullptr, b.ring.err, p.ring_st);\n",
        "        pf_ring(p, c, x, rows, nullptr, b.ring.err, p.ring_st, ring_rs);\n",
        1,
    ),
    # the pass
    (
        "    for (int l = 0; l < NL; ++l) {\n"
        "        if (staging && l + 1 < NL - 1) stage_in(l + 1);   // the last layer runs no experts\n",
        SP_LOOP +
        "    for (int l = 0; l < NL; ++l) {\n"
        "        if (staging && l + 1 < NL - 1) stage_in(l + 1);   // the last layer runs no experts\n",
        1,
    ),
])
print("sp_patch: pf_common.cuh, prefill.cu")
