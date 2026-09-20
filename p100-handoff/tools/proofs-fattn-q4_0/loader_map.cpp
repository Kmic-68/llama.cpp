// Does the restructured q4_0 tile loader move the SAME values to the SAME shared-memory slots
// as the version in the last public push?
//
// The restructure (961e63c18 + c6f5211f4) changed which thread reads which byte. The dequant
// arithmetic is proven bit-identical separately; what is unproven is the data movement. Here
// both index schemes are simulated exactly as written, for every thread, and compared on:
//   1. coverage   -- every tile slot written exactly once by each scheme
//   2. agreement  -- both schemes put the same (block, byte, nibble) in the same slot
//   3. canonical  -- that source is the one the q4_0 layout says holds that head dimension
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>

static const int QK4_0 = 32;
enum { LOW = 0, HIGH = 1 };

struct Src { int blk, byte, nib; bool set; };

static inline bool same(const Src &a, const Src &b) {
    return a.set == b.set && a.blk == b.blk && a.byte == b.byte && a.nib == b.nib;
}

// canonical q4_0: value v of a block is qs[v] low nibble for v<16, qs[v-16] high for v>=16
static Src canonical(int a) {
    Src s; s.set = true; s.blk = a / QK4_0;
    const int v = a % QK4_0;
    s.byte = (v < QK4_0/2) ? v : v - QK4_0/2;
    s.nib  = (v < QK4_0/2) ? LOW : HIGH;
    return s;
}

struct Res { bool ok; const char *why; };

static Res run(int warp_size, int nwarps, int I, int J, int cpy_ne, int k_dim_0, bool verbose) {
    const int NS = J/2;                       // half2 slots per row
    std::vector<Src> mo(I*NS), mn(I*NS);      // lane0 source per slot, old / new
    std::vector<Src> mo1(I*NS), mn1(I*NS);    // lane1 source per slot
    std::vector<int> co(I*NS, 0), cn(I*NS, 0);// write counts
    for (auto &s : mo) s.set = false;  for (auto &s : mn) s.set = false;
    for (auto &s : mo1) s.set = false; for (auto &s : mn1) s.set = false;

    // ---------- OLD: last public push (0dacb39a8) ------------------------------------------
    // ggml_cuda_unroll<7>{}(load) calls load(6), load(5), ... load(0)
    for (int n = 6; n >= 0; --n) {
        const int stride_j = warp_size >> n;
        if (stride_j == 0) continue;
        const int jj = (J/2)/cpy_ne;
        const int j0_start = (stride_j == warp_size) ? 0 : jj - jj % (2*stride_j);
        const int j0_stop  =                               jj - jj % (1*stride_j);
        const int stride_i = warp_size / stride_j;
        if (j0_start == j0_stop) continue;

        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            for (int ty = 0; ty < nwarps; ++ty) for (int tx = 0; tx < warp_size; ++tx) {
                const int i = i0 + ty*stride_i + ((stride_j == warp_size) ? 0 : tx/stride_j);
                if (!(i0 + nwarps*stride_i <= I || i < I)) continue;
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*cpy_ne + ((stride_j == warp_size) ? tx : tx % stride_j)*cpy_ne;
                    const int a     = k_dim_0 + 2*j;
                    const int iqs   = a % QK4_0;
                    const int base  = (iqs < QK4_0/2) ? iqs : iqs - QK4_0/2;
                    const int shift = (iqs < QK4_0/2) ? LOW : HIGH;
                    const int blk   = a / QK4_0;
                    for (int l = 0; l < cpy_ne; ++l) {
                        const int slot = j + l;
                        if (i >= I || slot >= NS) return {false, "OLD wrote out of range"};
                        co[i*NS + slot]++;
                        mo [i*NS + slot] = Src{blk, base + 2*l + 0, shift, true};
                        mo1[i*NS + slot] = Src{blk, base + 2*l + 1, shift, true};
                    }
                }
            }
        }
    }

    // ---------- NEW: HEAD ------------------------------------------------------------------
    const int VPS   = 4*cpy_ne;
    const int NSLOT = J / VPS;
    const int SPB   = (QK4_0/2) / (2*cpy_ne);
    const int nthr  = nwarps*warp_size;
    for (int w0 = 0; w0 < I*NSLOT; w0 += nthr) {
        for (int tid = 0; tid < nthr; ++tid) {
            const int w = w0 + tid;
            if (!(w0 + nthr <= I*NSLOT || w < I*NSLOT)) continue;
            const int i    = w / NSLOT;
            const int s    = w % NSLOT;
            const int m    = (s % SPB) * (2*cpy_ne);
            const int a_lo = k_dim_0 + (s / SPB)*QK4_0 + m;
            const int blk  = a_lo / QK4_0;
            const int j_lo = (a_lo - k_dim_0) / 2;
            for (int l = 0; l < cpy_ne; ++l) {
                const int sl = j_lo + l, sh = j_lo + QK4_0/4 + l;
                if (i >= I || sl >= NS || sh >= NS) return {false, "NEW wrote out of range"};
                cn[i*NS + sl]++;  mn [i*NS + sl] = Src{blk, m + 2*l + 0, LOW,  true};
                                  mn1[i*NS + sl] = Src{blk, m + 2*l + 1, LOW,  true};
                cn[i*NS + sh]++;  mn [i*NS + sh] = Src{blk, m + 2*l + 0, HIGH, true};
                                  mn1[i*NS + sh] = Src{blk, m + 2*l + 1, HIGH, true};
            }
        }
    }

    for (int i = 0; i < I; ++i) for (int j = 0; j < NS; ++j) {
        const int k = i*NS + j;
        if (co[k] != 1) { if (verbose) printf("   old coverage slot(%d,%d)=%d\n", i, j, co[k]); return {false, "OLD coverage != 1"}; }
        if (cn[k] != 1) { if (verbose) printf("   new coverage slot(%d,%d)=%d\n", i, j, cn[k]); return {false, "NEW coverage != 1"}; }
        if (!same(mo[k], mn[k]) || !same(mo1[k], mn1[k])) {
            if (verbose) printf("   slot(%d,%d) old=(b%d,B%d,n%d) new=(b%d,B%d,n%d)\n", i, j,
                   mo[k].blk, mo[k].byte, mo[k].nib, mn[k].blk, mn[k].byte, mn[k].nib);
            return {false, "OLD/NEW source mismatch"};
        }
        const Src c0 = canonical(k_dim_0 + 2*j), c1 = canonical(k_dim_0 + 2*j + 1);
        if (!same(mn[k], c0) || !same(mn1[k], c1)) {
            if (verbose) printf("   slot(%d,%d) new=(b%d,B%d,n%d) canonical=(b%d,B%d,n%d)\n", i, j,
                   mn[k].blk, mn[k].byte, mn[k].nib, c0.blk, c0.byte, c0.nib);
            return {false, "NEW disagrees with q4_0 layout"};
        }
    }
    return {true, "ok"};
}

int main() {
    const int warp_size = 32;
    const int cpy_ne    = 2;   // sm_60: ggml_cuda_get_max_cpy_bytes()==8
    int pass = 0, fail = 0;

    printf("%-6s %-7s %-4s %-4s %-9s %s\n", "nwarps", "I", "J", "k0", "result", "note");
    for (int nwarps : {2, 4, 6, 8}) {
        for (int I : {32, 64, 128, 256}) {
            for (int J : {64, 128, 192, 256, 288}) {
                if (J % (4*cpy_ne) != 0) continue;              // static_assert(J % VPS == 0)
                for (int k0 = 0; k0 < 512; k0 += J) {
                    Res r = run(warp_size, nwarps, I, J, cpy_ne, k0, false);
                    if (r.ok) { pass++; }
                    else {
                        fail++;
                        printf("%-6d %-7d %-4d %-4d %-9s %s\n", nwarps, I, J, k0, "FAIL", r.why);
                        run(warp_size, nwarps, I, J, cpy_ne, k0, true);
                    }
                }
            }
        }
    }
    printf("\nconfigurations passing : %d\nconfigurations failing : %d\n", pass, fail);

    // The restructure assumes a tile's head-dim origin lands on a q4_0 block boundary.
    // Probe that assumption directly with an nbatch_K that is not a multiple of QK4_0.
    printf("\n-- probe: nbatch_K not a multiple of QK4_0 (k_dim_0 %% 32 != 0) --\n");
    for (int J : {40, 72, 88, 120}) {
        if (J % (4*cpy_ne) != 0) { printf("J=%-4d skipped (fails static_assert J %% VPS == 0)\n", J); continue; }
        Res r = run(warp_size, 4, 64, J, cpy_ne, J, false);   // k_dim_0 = J, the second tile
        printf("J=%-4d k_dim_0=%-4d %s  %s\n", J, J, r.ok ? "ok" : "MISMATCH", r.why);
    }
    return fail ? 1 : 0;
}
