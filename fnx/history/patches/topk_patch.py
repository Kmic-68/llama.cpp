#!/usr/bin/env python3
# topk_patch.py (2026-11-11): FNX_RR_FAST in k_router_route (the dev tree; after rrt_patch.py). The tail picked each token's top 10 in
# ten rounds of a warp maximum (twice with the route prior): 21 us of a 66 us launch (FNX_RR_TIMES). The fast path finds the same picks
# in the same order (by score, ties to the lower index) from a threshold (the K-th largest of the lanes' maxima, a bitonic sort) and
# the ranks of the scores at or above it, then the weights as the loop makes them (top0, z summed in pick order, __expf). More than
# RRF_CAP candidates (equal scores) or fewer than K finite: the loop as before. 1: the fast path; 2: both, the loop's results used and
# each token whose picks or weights differ counted (rr_bad, printed with FNX_RR_TIMES). Unset or 0: as before.
# Saves the originals in mikro/nvp/*.pred-topk once.
import shutil, sys
from pathlib import Path
S = Path("/srv/build/strata-hkrati/upstream/src/split/src")
B = Path("/srv/build/strata-hkrati/mikro/nvp")
BS = chr(92)


def patch(name, pairs):
    p = S / name
    c = p.read_text(encoding="utf-8")
    if "rr_fast_pick" in c or "FNX_RR_FAST" in c:
        sys.exit(name + ": already patched")
    b = B / (name + ".pred-topk")
    if not b.exists():
        shutil.copy2(p, b)
    for old, new in pairs:
        assert c.count(old) == 1, (name, old[:70])
        c = c.replace(old, new, 1)
    p.write_text(c, encoding="utf-8")


FAST = """// FNX_RR_FAST (topk_patch.py): the top K of a warp's NEXP scores (lane + 32 i in s[i] of lane `lane`; those in `skip` left out) by score,
// ties to the lower index, as k_router_route's loop of warp maxima picks them: every top K score is at or above the K-th largest of the
// lanes' maxima, so those are the candidates, ranked against each other. out[r] = the index of rank r < K. False (warp-uniform):
// over RRF_CAP candidates or fewer than K finite scores
constexpr int RRF_CAP = 64;
__device__ unsigned rr_bad;   // FNX_RR_FAST=2: tokens whose fast picks or weights differed from the loop's
__device__ __forceinline__ float rr_sel16(const float (&a)[NEXP / 32], const int i) {
    float r = a[0];
#pragma unroll
    for (int q = 1; q < NEXP / 32; ++q) r = i == q ? a[q] : r;
    return r;
}
__device__ __forceinline__ bool rr_topk_fast(const float (&s)[NEXP / 32], const unsigned skip, const int K, const int lane, float* cs,
                                             int* ci, int* out) {
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NEXP / 32; ++i)
        if (!(skip >> i & 1u) && s[i] > m) m = s[i];
    float x = m;   // the lanes' maxima sorted in descending order across the warp (bitonic)
#pragma unroll
    for (int size = 2; size <= 32; size <<= 1)
#pragma unroll
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            const float y = __shfl_xor_sync(0xffffffffu, x, stride);
            const bool desc = (lane & size) == 0, low = (lane & stride) == 0;
            x = low == desc ? fmaxf(x, y) : fminf(x, y);
        }
    const float th = __shfl_sync(0xffffffffu, x, K - 1);
    if (!(th > -INFINITY)) return false;
    int n = 0;
#pragma unroll
    for (int i = 0; i < NEXP / 32; ++i) {
        const bool c = !(skip >> i & 1u) && s[i] >= th;
        const unsigned b = __ballot_sync(0xffffffffu, c);
        const int pos = n + __popc(b & ((1u << lane) - 1u));
        if (c && pos < RRF_CAP) {
            cs[pos] = s[i];
            ci[pos] = lane + 32 * i;
        }
        n += __popc(b);
    }
    if (n > RRF_CAP) return false;
    __syncwarp();
    for (int j = lane; j < n; j += 32) {
        const float v = cs[j];
        const int id = ci[j];
        int r = 0;
        for (int q = 0; q < n; ++q) {
            const float w = cs[q];
            r += w > v || (w == v && ci[q] < id);
        }
        if (r < K) out[r] = id;
    }
    __syncwarp();
    return true;
}
// a token's picks into o[0..TOPK) and their weights' numerators into e[0..TOPK) (the warp's shared memory), z summed in pick order
__device__ __forceinline__ bool rr_fast_pick(const float (&v)[NEXP / 32], const float lam, const ExpertCache& ec, const int layer,
                                             const int lane, float* cs, int* ci, int* o, float* e, float& z) {
    if (!rr_topk_fast(v, 0u, TOPK, lane, cs, ci, o)) return false;
    if (lam > 0.f) {   // picks 0 and 1 stay the unraised top 2; 2..9 by the raised scores, as the loop's second pass
        const int a0 = o[0], a1 = o[1], a9 = o[TOPK - 1];
        const float b0 = __shfl_sync(0xffffffffu, rr_sel16(v, a0 >> 5), a0 & 31);
        const float b9 = __shfl_sync(0xffffffffu, rr_sel16(v, a9 >> 5), a9 & 31);
        const float up = lam * (b0 - b9);
        float vb[NEXP / 32];
#pragma unroll
        for (int i = 0; i < NEXP / 32; ++i) vb[i] = v[i] + (ec.slot_of[(size_t) layer * NEXP + lane + 32 * i] >= 0 ? up : 0.f);
        unsigned skip = 0;
        if ((a0 & 31) == lane) skip |= 1u << (a0 >> 5);
        if ((a1 & 31) == lane) skip |= 1u << (a1 >> 5);
        __syncwarp();
        if (!rr_topk_fast(vb, skip, TOPK - 2, lane, cs, ci, o + 2)) return false;
    }
    float top0 = 0.f;
    z = 0.f;
    for (int k = 0; k < TOPK; ++k) {
        const int id = o[k];
        const float bo = __shfl_sync(0xffffffffu, rr_sel16(v, id >> 5), id & 31);
        if (k == 0) top0 = bo;
        const float ek = __expf(bo - top0);
        z += ek;
        if (lane == 0) e[k] = ek;
    }
    __syncwarp();
    return true;
}

"""

patch("moe.cuh", [
    ("// FNX_RR_TIMES (rrt_patch.py): the launch's stamps", FAST + "// FNX_RR_TIMES (rrt_patch.py): the launch's stamps"),
    ("""                                                            const ExpertCache ec, int layer, int* list, const ShexpUp su,
                                                            const RouteTail rt) {""",
     """                                                            const ExpertCache ec, int layer, int* list, const ShexpUp su,
                                                            const RouteTail rt, const int fast) {"""),
    ("""    // top 10 per token: warp w takes tokens w, w + 8, ...; lane l holds experts l + 32 i
    __shared__ int ex[MAXT * TOPK];""", """    // top 10 per token: warp w takes tokens w, w + 8, ...; lane l holds experts l + 32 i
    __shared__ int ex[MAXT * TOPK];
    __shared__ float rf_s[RR_THREADS / 32][RRF_CAP], rf_e[RR_THREADS / 32][TOPK];   // FNX_RR_FAST: a warp's candidates, weights
    __shared__ int rf_i[RR_THREADS / 32][RRF_CAP], rf_o[RR_THREADS / 32][TOPK];"""),
    ("""        const float lam = ec.n_slots && ec.prior ? *ec.prior : 0.f;
        float vb[NEXP / 32];""", """        const float lam = ec.n_slots && ec.prior ? *ec.prior : 0.f;
        float fz = 0.f;
        const bool fok = fast && rr_fast_pick(v, lam, ec, layer, lane, rf_s[warp], rf_i[warp], rf_o[warp], rf_e[warp], fz);
        if (fok && fast == 1) {
            if (lane < TOPK) {
                const int id = rf_o[warp][lane];
                topi[t * TOPK + lane] = id;
                ex[t * TOPK + lane] = id;
                topw[t * TOPK + lane] = rf_e[warp][lane] / fz;
            }
            continue;
        }
        float vb[NEXP / 32];"""),
    ("""        if (lane < TOPK) topw[t * TOPK + lane] = wk / z;
    }""", """        if (lane < TOPK) topw[t * TOPK + lane] = wk / z;
        if (fok) {   // FNX_RR_FAST=2: the fast path against the loop
            const bool bad = lane < TOPK && (rf_o[warp][lane] != ex[t * TOPK + lane] ||
                                             __float_as_uint(rf_e[warp][lane] / fz) != __float_as_uint(wk / z));
            if (__any_sync(0xffffffffu, bad) && lane == 0) atomicAdd(&rr_bad, 1u);
        }
    }"""),
])
patch("kernels.cu", [
    ("""#define FNX_RR(t) k_router_route<t><<<(NEXP + 1 + rr_rows<t>() - 1) / rr_rows<t>() + su.blocks, RR_THREADS, 0, s>>>(W, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su, tail)""",
     """    static const int rr_fast = (int) env_int("FNX_RR_FAST", 0);   // topk_patch.py
#define FNX_RR(t) k_router_route<t><<<(NEXP + 1 + rr_rows<t>() - 1) / rr_rows<t>() + su.blocks, RR_THREADS, 0, s>>>(W, x, logits, topi, topw, jobs_gu, jobs_dn, njobs, counter, ec ? *ec : none, layer, list, su, tail, rr_fast)"""),
    ("""    last[(size_t) dev] = n;
    if (c)""", """    last[(size_t) dev] = n;
    unsigned bad = 0;   // topk_patch.py: FNX_RR_FAST=2's count of tokens where the fast path differed
    cudaMemcpyFromSymbol(&bad, rr_bad, sizeof bad);
    if (bad) std::fprintf(stderr, "rr fast: %u tokens differ from the loop""" + BS + """n", bad);
    if (c)"""),
])
print("patched")
