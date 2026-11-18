#!/usr/bin/env python3
# maska_patch.py (13. 11. 2026): FNX_EXPERT_MASK=file, the experts a pruned release keeps (ISTA-DASLab's Qwen3.8-Flash-Next
# GSQ-RCO Coder: 256 of 512 a layer, its tensor-allocation list), applied to a full model's weights (the authors: "the mask
# can be reapplied to the original weights").
#   - the router: a dropped expert's logit is -inf in every place that picks from a layer's logits (the windows' router,
#     the prompt path's, the next layer's guesses), so no top 10 or guess takes it; the weights of the ten are those of the
#     pruned model (softmax over the picks), the draft layer is untouched
#   - placement: a dropped expert ranks below every kept one (score -1), so FNX_EXPERT_RAM_GIB moves the dropped ones to RAM
#     first; the prompt path never stages them, the arena and FNX_PF_REFILL never refill a slot with one
# Without FNX_EXPERT_MASK every change is inert (no launch, no branch taken): the same bits as before.
# Usage: python3 maska_patch.py [SRC_DIR]   (default: the dev tree's upstream/src/split/src); originals to varne-maska-20261113/
import os, re, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '/srv/build/strata-hkrati/upstream/src/split/src'
SAFE = os.environ.get('MASKA_SAFE', '/srv/build/strata-hkrati/varne-maska-20261113')
FILES = ['kernels.hpp', 'moe.cuh', 'kernels.cu', 'engine_impl.hpp', 'engine.cu', 'prefill.cu', 'model.hpp', 'model.cpp']

texts = {f: open(os.path.join(SRC, f), encoding='utf-8').read() for f in FILES}
if 'FNX_EXPERT_MASK' in texts['model.hpp']:
    sys.exit('maska_patch.py: already applied (model.hpp names FNX_EXPERT_MASK)')
for f, t in texts.items():
    if 'FNX_EXPERT_MASK' in t:
        sys.exit(f'maska_patch.py: {f} already names FNX_EXPERT_MASK, model.hpp does not: a half-applied tree, restore from {SAFE}')


def rep(f, old, new, count=1):
    n = texts[f].count(old)
    if n != count:
        sys.exit(f'maska_patch.py: {f}: anchor found {n} times, want {count}:\n{old}')
    texts[f] = texts[f].replace(old, new)


def after(f, anchor, add, count=1):
    rep(f, anchor, anchor + add, count)


# ---- kernels.hpp: the mask's row in RouteTail, the launcher for the other pickers
after('kernels.hpp', '    size_t gu_bytes = 0, dn_bytes = 0;\n',
      '    const unsigned* keep = nullptr;   // FNX_EXPERT_MASK (maska_patch.py): the layer\'s kept experts, a bit each (null: every one)\n')
after('kernels.hpp', 'float* sh_out = nullptr, const RouteTail* rt = nullptr);\n',
      '// FNX_EXPERT_MASK (maska_patch.py): rows of `stride` floats whose first NEXP are a layer\'s router logits; a dropped\n'
      '// expert\'s (its bit in keep clear) to -inf, so no top 10 or guess ever picks it. keep null: nothing to do\n'
      'void expert_mask_logits(float* logits, int rows, int stride, const unsigned* keep, void* st);\n')

# ---- moe.cuh: the windows' router writes a dropped expert's logit as -inf (the routing in the same launch reads them back)
rep('moe.cuh', '        if (lrow < L) logits[(size_t) t * L + lrow] = rsum[t][r][0] + rsum[t][r][1];\n',
    '        if (lrow < L)   // FNX_EXPERT_MASK (maska_patch.py): a dropped expert is never picked\n'
    '            logits[(size_t) t * L + lrow] = lrow < NEXP && rt.keep && !(rt.keep[lrow >> 5] >> (lrow & 31) & 1u) ? -INFINITY : rsum[t][r][0] + rsum[t][r][1];\n')

# ---- kernels.cu: the launcher, in the router's part
after('kernels.cu', '#if FNX_KPART == 8\nvoid router_route(', '', 1)   # the anchor must exist once
rep('kernels.cu', '#if FNX_KPART == 8\nvoid router_route(',
    '#if FNX_KPART == 8\n'
    '// FNX_EXPERT_MASK (maska_patch.py): a dropped expert\'s logit to -inf in rows of `stride` floats (the first NEXP the router\'s)\n'
    '__global__ void k_expert_mask(float* logits, int rows, int stride, const unsigned* keep) {\n'
    '    const int i = blockIdx.x * blockDim.x + threadIdx.x;\n'
    '    if (i >= rows * NEXP) return;\n'
    '    const int e = i % NEXP;\n'
    '    if (!(keep[e >> 5] >> (e & 31) & 1u)) logits[(size_t) (i / NEXP) * stride + e] = -INFINITY;\n'
    '}\n'
    'void expert_mask_logits(float* logits, int rows, int stride, const unsigned* keep, void* st) {\n'
    '    if (!keep || rows <= 0) return;\n'
    '    k_expert_mask<<<(rows * NEXP + 255) / 256, 256, 0, (cudaStream_t) st>>>(logits, rows, stride, keep);\n'
    '    ck_launch("expert_mask_logits");\n'
    '}\n'
    'void router_route(')

# ---- engine_impl.hpp: each card's copy of the mask
after('engine_impl.hpp', '    unsigned* route_cnt = nullptr;   // FNX_ROUTE_COUNT, card 0: [decode | prompt][NL][NEXP] picks\n',
      '    unsigned* ekeep = nullptr;       // FNX_EXPERT_MASK (maska_patch.py): [NL][NEXP / 32] bits, set: the expert is kept\n')

# ---- engine.cu
after('engine.cu', '        b->logits = dalloc<float>((size_t) MAXT * (NEXP + 1));\n',
      '        if (!m_.keep.empty()) {   // FNX_EXPERT_MASK (maska_patch.py): the kept experts as bits, a layer\'s row NEXP / 32 words\n'
      '            std::vector<unsigned> bits((size_t) NL * NEXP / 32, 0u);\n'
      '            for (int i = 0; i < NL * NEXP; ++i)\n'
      '                if (m_.kept(i)) bits[(size_t) (i >> 5)] |= 1u << (i & 31);\n'
      '            b->ekeep = dalloc<unsigned>(bits.size());\n'
      '            ck(cudaMemcpy(b->ekeep, bits.data(), bits.size() * sizeof(unsigned), cudaMemcpyHostToDevice), "expert mask");\n'
      '        }\n')
after('engine.cu', '    RouteTail rt;\n',
      '    if (b.ekeep && lyr >= 0 && lyr < NL) rt.keep = b.ekeep + (size_t) lyr * (NEXP / 32);   // FNX_EXPERT_MASK (maska_patch.py)\n')
after('engine.cu', '            gemv_bf16(w.L[(size_t) nl].router, b.mixed, b.pf_logits, NEXP, N, T, b.st2);\n',
      '            expert_mask_logits(b.pf_logits, T, NEXP, b.ekeep ? b.ekeep + (size_t) nl * (NEXP / 32) : nullptr, b.st2);   // FNX_EXPERT_MASK\n')
after('engine.cu', '                gemv_bf16(w.L[(size_t) l + 1].router, b.mixed, b.probe_logits, NEXP, N, T, b.st);\n',
      '                expert_mask_logits(b.probe_logits, T, NEXP, b.ekeep ? b.ekeep + (size_t) (l + 1) * (NEXP / 32) : nullptr, b.st);   // FNX_EXPERT_MASK\n')
rep('engine.cu', '                if (slot_of[(size_t) l * NEXP + e] < 0) out.push_back(l * NEXP + e);\n',
    '                if (slot_of[(size_t) l * NEXP + e] < 0 && m_.kept(l * NEXP + e)) out.push_back(l * NEXP + e);   // FNX_EXPERT_MASK: never a dropped one\n')
rep('engine.cu', 'if (const int g = l * NEXP + e; slot_of[(size_t) g] < 0 && !taken[(size_t) g]) out.push_back(g);\n',
    'if (const int g = l * NEXP + e; slot_of[(size_t) g] < 0 && !taken[(size_t) g] && m_.kept(g)) out.push_back(g);   // FNX_EXPERT_MASK\n')

# ---- prefill.cu: the prompt path's router, its stages
t = texts['prefill.cu']
pat = re.compile(r'^([ \t]*)pf_sgemm\(p, p\.router32, NEXP \+ 1, N, p\.mixed, C, p\.logits, NEXP \+ 1\);\n', re.M)
if len(pat.findall(t)) != 2:
    sys.exit(f'maska_patch.py: prefill.cu: the prompt router\'s sgemm found {len(pat.findall(t))} times, want 2')
texts['prefill.cu'] = pat.sub(lambda m: m.group(0) + m.group(1) + 'expert_mask_logits(p.logits, C, NEXP + 1, b.ekeep ? b.ekeep + (size_t) l * (NEXP / 32) : nullptr, s);   // FNX_EXPERT_MASK (maska_patch.py)\n', t)
rep('prefill.cu', '                for (int e = 0; e < NEXP; ++e) n_ram += p.slot_h[l * NEXP + e] < 0 ? 1 : 0;\n',
    '                for (int e = 0; e < NEXP; ++e) n_ram += p.slot_h[l * NEXP + e] < 0 && m_.kept(l * NEXP + e) ? 1 : 0;   // FNX_EXPERT_MASK: a dropped one is never staged\n')
rep('prefill.cu', '                    } else if (k < fit) {\n',
    '                    } else if (!m_.kept((int) i)) {   // FNX_EXPERT_MASK (maska_patch.py): never picked, so never copied or read\n'
    '                        p.ep_hgu[i] = w.ec_host_gu[i], p.ep_hdn[i] = w.ec_host_dn[i];\n'
    '                    } else if (k < fit) {\n')
rep('prefill.cu', '        for (int e = 0; e < NEXP; ++e) n += !all && m_.resident[(size_t) l][(size_t) e] ? 0 : 1;\n',
    '        for (int e = 0; e < NEXP; ++e) n += (!all && m_.resident[(size_t) l][(size_t) e]) || !m_.kept(l * NEXP + e) ? 0 : 1;   // FNX_EXPERT_MASK: dropped ones never staged\n')

# ---- model.hpp / model.cpp: the mask, read with the residency; a dropped expert ranks below every kept one
after('model.hpp', '    std::vector<double> score;           // [NL * NEXP]\n',
      '    // FNX_EXPERT_MASK=file (maska_patch.py): [NL * NEXP], 1 = kept (a pruned release\'s experts, e.g. ISTA-DASLab\'s Coder: 256\n'
      '    // of 512 a layer); empty: every expert. The router never picks a dropped one, and it ranks below every kept one for VRAM\n'
      '    std::vector<char> keep;\n'
      '    bool kept(int g) const { return keep.empty() || keep[(size_t) g] != 0; }\n'
      '    void load_expert_mask();   // reads FNX_EXPERT_MASK once (unset: keep stays empty)\n')
rep('model.cpp', 'void Model::choose_residency() {\n',
    '// FNX_EXPERT_MASK=file (maska_patch.py): one line a layer, "layer e e ...", the experts kept (indices of the full model);\n'
    '// \'#\' starts a comment line. Every layer needs a line and at least TOPK experts\n'
    'void Model::load_expert_mask() {\n'
    '    const char* mf = std::getenv("FNX_EXPERT_MASK");\n'
    '    if (!mf || !*mf || !keep.empty()) return;\n'
    '    std::ifstream in(mf);\n'
    '    if (!in) throw std::runtime_error(std::string("fnx: FNX_EXPERT_MASK: cannot open ") + mf);\n'
    '    std::vector<char> k((size_t) NL * NEXP, 0);\n'
    '    std::vector<int> per(NL, 0);\n'
    '    std::vector<char> seen(NL, 0);\n'
    '    std::string line;\n'
    '    while (std::getline(in, line)) {\n'
    '        if (line.empty() || line[0] == \'#\') continue;\n'
    '        std::istringstream is(line);\n'
    '        int l = -1, e = -1;\n'
    '        if (!(is >> l) || l < 0 || l >= NL || seen[(size_t) l]) throw std::runtime_error("fnx: FNX_EXPERT_MASK: not a new layer\'s line: " + line.substr(0, 80));\n'
    '        seen[(size_t) l] = 1;\n'
    '        while (is >> e) {\n'
    '            if (e < 0 || e >= NEXP) throw std::runtime_error("fnx: FNX_EXPERT_MASK: layer " + std::to_string(l) + ": expert " + std::to_string(e) + " out of range");\n'
    '            if (!k[(size_t) l * NEXP + e]) k[(size_t) l * NEXP + e] = 1, ++per[(size_t) l];\n'
    '        }\n'
    '        if (!is.eof()) throw std::runtime_error("fnx: FNX_EXPERT_MASK: layer " + std::to_string(l) + ": not a number in its line");\n'
    '    }\n'
    '    int total = 0;\n'
    '    for (int l = 0; l < NL; ++l) {\n'
    '        if (per[(size_t) l] < TOPK)\n'
    '            throw std::runtime_error("fnx: FNX_EXPERT_MASK: layer " + std::to_string(l) + " keeps " + std::to_string(per[(size_t) l]) +\n'
    '                                     " experts, fewer than the " + std::to_string(TOPK) + " a token takes");\n'
    '        total += per[(size_t) l];\n'
    '    }\n'
    '    keep.swap(k);\n'
    '    std::fprintf(stderr, "fnx: FNX_EXPERT_MASK %s: %d of %d routed experts kept\\n", mf, total, NL * NEXP);\n'
    '}\n'
    '\n'
    'void Model::choose_residency() {\n'
    '    load_expert_mask();   // FNX_EXPERT_MASK (maska_patch.py)\n')
after('model.cpp', '                score[i] = (cnt[i] / std::max(1.0, td) + cnt[(size_t) NL * NEXP + i] / std::max(1.0, tp)) / bytes[(size_t) l];\n',
      '                if (!kept((int) i)) score[i] = -1.0;   // FNX_EXPERT_MASK: never picked, the first to RAM\n')

os.makedirs(SAFE, exist_ok=True)
for f in FILES:
    dst = os.path.join(SAFE, f)
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(SRC, f), dst)
for f, t in texts.items():
    with open(os.path.join(SRC, f), 'w', encoding='utf-8', newline='\n') as h:
        h.write(t)
print('maska_patch.py: applied to', SRC, '- originals in', SAFE)
