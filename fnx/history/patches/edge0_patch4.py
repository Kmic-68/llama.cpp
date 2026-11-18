#!/usr/bin/env python3
# edge0_patch4.py (14. 11. 2026, round edge0n: Slovenian 18/30 wrong against the base's 5/30): FNX_LAT_DUMP=<dir> (a test switch).
# The first prompt chunk that reaches layer FNX_LAT_DUMP_L (default 20) in the prompt path's latent MoE writes, on each card,
#   <dir>/c<dev>_xh.f16 (C x N, D's input), c<dev>_topi.i32, c<dev>_topw.f32 (C x TOPK), c<dev>_latu.f32 (C x N, U's output: the
#   card's ff part), c<dev>_C.txt (C, layer)
# so mikro/edge0/lat_ref.py can recompute the layer from the files in numpy (f64) and compare the two cards' sum.
import os, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else '/srv/build/strata-hkrati/upstream/src/split/src'
SAFE = os.path.join(os.environ.get('EDGE0_SAFE', '/srv/build/strata-hkrati/varne-edge0-20261114'), 'patch4')
f = 'prefill.cu'
t = open(os.path.join(SRC, f), encoding='utf-8').read()
if 'FNX_LAT_DUMP' in t:
    sys.exit('edge0_patch4.py: already applied')
fn = r'''// edge0_patch4.py, FNX_LAT_DUMP=<dir>: the layer FNX_LAT_DUMP_L's inputs and U output of the first chunk, each card's to files
void lat_dump(const Engine::Pf& p, int l, int C, cudaStream_t s) {
    static const char* dir = std::getenv("FNX_LAT_DUMP");
    static const int want = std::getenv("FNX_LAT_DUMP_L") ? std::atoi(std::getenv("FNX_LAT_DUMP_L")) : 20;
    static bool done[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!dir || l != want || done[dev & 15]) return;
    done[dev & 15] = true;
    ckc(cudaStreamSynchronize(s), "lat dump");
    auto put = [&](const char* name, const void* d, size_t bytes) {
        std::vector<char> h(bytes);
        ckc(cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost), "lat dump");
        const std::string path = std::string(dir) + "/c" + std::to_string(dev) + "_" + name;
        FILE* fp = std::fopen(path.c_str(), "wb");
        if (!fp) throw std::runtime_error("FNX_LAT_DUMP: cannot write " + path);
        std::fwrite(h.data(), 1, bytes, fp);
        std::fclose(fp);
    };
    put("xh.f16", p.xh, (size_t) C * N * 2);
    put("topi.i32", p.topi, (size_t) C * TOPK * 4);
    put("topw.f32", p.topw, (size_t) C * TOPK * 4);
    put("latu.f32", p.latu, (size_t) C * N * 4);
    const std::string meta = std::to_string(C) + " " + std::to_string(l) + "\n";
    FILE* fp = std::fopen((std::string(dir) + "/c" + std::to_string(dev) + "_C.txt").c_str(), "w");
    if (fp) { std::fputs(meta.c_str(), fp); std::fclose(fp); }
    std::fprintf(stderr, "fnx: FNX_LAT_DUMP: card %d layer %d, %d rows written to %s\n", dev, l, C, dir);
}
'''
anchor = 'void lat_dense(Engine::Pf& p, const Q8S& w, int rows, int cols, const __half* X, int C, void* out, bool f16_out, int ldo, cudaStream_t s) {\n'
assert t.count(anchor) == 1
t = t.replace(anchor, fn + anchor)
# after U in both variants (plain: 20 spaces; overlapped: 24)
for ind in (' ' * 20, ' ' * 24):
    a = '\n' + ind + 'if (lchk) lat_check(p.latu, (size_t) C * N, false, 64, (int) l, C, s);\n'
    n = t.count(a)
    if n != 1:
        sys.exit(f'edge0_patch4.py: the U check at indent {len(ind)} found {n} times')
    t = t.replace(a, a + ind + 'lat_dump(p, (int) l, C, s);   // edge0_patch4.py (FNX_LAT_DUMP)\n')
os.makedirs(SAFE, exist_ok=True)
shutil.copy2(os.path.join(SRC, f), os.path.join(SAFE, f))
open(os.path.join(SRC, f), 'w', encoding='utf-8', newline='\n').write(t)
print('edge0_patch4.py: prefill.cu patched')
