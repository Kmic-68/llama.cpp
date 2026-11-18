#include "model.hpp"
#include "card_check.hpp"
#include "env.hpp"
#include "rope_gguf.hpp"

#include "store.hpp"
#if FNX_HIP
#include "hip/selftest_policy.hpp"
#include "selftest.hpp"
#endif

#include <cuda_runtime.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <malloc.h>
#include <sys/syscall.h>
#include <map>
#include <mutex>

namespace fnx {

bool g_lat_split = false;   // edge0_patch6.py (FNX_LAT_SPLIT; outside the file's anonymous namespace: model.hpp declares it)

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// a card's PCI address in sysfs's spelling
std::string card_bdf(int dev) {
    char bus[32];
    ck(cudaDeviceGetPCIBusId(bus, sizeof bus, dev), "cudaDeviceGetPCIBusId");
    try {
        return sysfs_bdf(bus);
    } catch (const std::exception& e) {
        throw std::runtime_error("card " + std::to_string(dev) + ": " + e.what());
    }
}
// FNX_NUMA_EXPERTS (numa_patch.py): the NUMA node of a card's PCI slot, -1 when sysfs names none
int card_numa_node(int dev) {
    std::ifstream f("/sys/bus/pci/devices/" + card_bdf(dev) + "/numa_node");
    int n = -1;
    if (!(f >> n)) n = -1;
    return n;
}
// a node's room for page-locked memory: its free and file-backed (reclaimable) bytes from its meminfo, less its CMA pageblocks
// (/proc/zoneinfo's cma, pages): pinned pages never come from those (ai-streznik: 8.6 GiB of node 1, 2.5 of node 0), so without
// them MPOL_BIND could reclaim the node and still come up short (Strata PR #5436 met the same)
size_t numa_node_room(int node) {
    std::ifstream f("/sys/devices/system/node/node" + std::to_string(node) + "/meminfo");
    std::string line;
    size_t kb = 0;
    while (std::getline(f, line)) {
        char key[64];
        unsigned long long v = 0;
        if (std::sscanf(line.c_str(), "Node %*d %63[^:]: %llu", key, &v) == 2 &&
            (!std::strcmp(key, "MemFree") || !std::strcmp(key, "Active(file)") || !std::strcmp(key, "Inactive(file)")))
            kb += v;
    }
    std::ifstream z("/proc/zoneinfo");
    size_t cma_pages = 0;
    bool mine = false;
    while (std::getline(z, line)) {
        int n = -1;
        unsigned long long v = 0;
        if (std::sscanf(line.c_str(), "Node %d, zone", &n) == 1) mine = n == node;
        else if (mine && std::sscanf(line.c_str(), " cma %llu", &v) == 1) cma_pages += v;
    }
    const size_t bytes = kb << 10, cma = cma_pages * 4096;
    return bytes > cma ? bytes - cma : 0;
}
size_t g_numa_bound[2], g_numa_pref[2];   // per card: bytes allocated under MPOL_BIND, under MPOL_PREFERRED
// the ring stores straight into the next card's memory: with translated IOMMU domains every such write faults
// (dmesg: "DMAR: ... PTE Write access is not set") and the first window ends in a ring timeout that names nothing.
// Each card's IOMMU group must be identity (boot with iommu=pt); a card whose iommu_group link is absent has no IOMMU
// in the way. Only absent: a link that cannot be read or points nowhere (iommu_groups hidden from this process) is
// refused, as a translated domain behind it would look the same
void require_identity_iommu(int dev, const std::string& dir) {
    if (access(dir.c_str(), F_OK) != 0)
        throw std::runtime_error("card " + std::to_string(dev) + ": no " + dir + " to read its IOMMU group from");
    const std::string link = dir + "/iommu_group", path = link + "/type";
    struct stat st;
    if (lstat(link.c_str(), &st) != 0) {
        if (errno == ENOENT) return;
        throw std::runtime_error("card " + std::to_string(dev) + ": cannot read " + link + ": " + std::strerror(errno));
    }
    std::FILE* f = std::fopen(path.c_str(), "r");
    char type[64] = {0};
    if (!f || !std::fgets(type, sizeof type, f)) {
        if (f) std::fclose(f);
        throw std::runtime_error("card " + std::to_string(dev) + ": cannot read " + path);
    }
    std::fclose(f);
    std::string t(type);
    while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
    if (t != "identity")
        throw std::runtime_error("card " + std::to_string(dev) + " is in an IOMMU group of type '" + t + "' (" + path +
                                 "): the ring's writes between cards need identity domains; boot with iommu=pt");
}
// the card's BAR1 from sysfs, held to its memory (require_bar1_covers)
void check_bar1(int dev, const std::string& bdf, const std::string& dir, const cudaDeviceProp& prop) {
    const std::string path = dir + "/resource";
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("card " + std::to_string(dev) + ": cannot open " + path + " to read its BAR1 from");
    std::string text;
    char buf[4096];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
    const bool bad = std::ferror(f);
    std::fclose(f);
    if (bad) throw std::runtime_error("card " + std::to_string(dev) + ": cannot read " + path);
    require_bar1_covers(dev, bdf, prop.name, bar1_bytes(text, path), prop.totalGlobalMem, NCARD);
}
// the Makefile passes the arch the kernels were compiled for (ARCH): a default here would let a card of another one
// run them through the driver's PTX compiler unchecked
#ifndef FNX_ARCH
#error "FNX_ARCH is unset: build through the Makefile (make ARCH=70 or 86)"
#endif
// The cards are checked before anything is placed on them (src/card_check.hpp has the rules): a card that fails one
// would otherwise take the whole load first and then fail far from the cause, an sm_70 build on 24 GB cards out of
// memory, a short BAR1 as a ring that never completes. The IOMMU check comes before BAR1's, as it is the one that
// refuses a card with no sysfs directory: BAR1's would only say it cannot open the file
void check_cards() {
    const bool skip_bar1 = env_flag("FNX_SKIP_BAR1_CHECK");
    if (skip_bar1) std::fprintf(stderr, "fnx: FNX_SKIP_BAR1_CHECK=1: the cards' BAR1 is not checked against their memory\n");
    for (int dev = 0; dev < NCARD; ++dev) {
        cudaDeviceProp prop;
        ck(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
#if FNX_HIP
        // the code objects in the binary: gfx1201 (R9700, RX 9070; run), gfx1200 (RX 9060) and gfx1100..1103 (RDNA3; not run on a multi-card PC): the card's gfx name, not a
        // compute capability. Any other card would find no kernels
        {
            bool built = false;
            for (const char* a : {"gfx1201", "gfx1200", "gfx1100", "gfx1101", "gfx1102", "gfx1103"})
                if (std::strncmp(prop.gcnArchName, a, 7) == 0) built = true;
            if (!built)
                throw std::runtime_error("card " + std::to_string(dev) + " (" + prop.name + ") is " + prop.gcnArchName +
                                         " and this build has kernels for gfx1201, gfx1200 and gfx1100..gfx1103 only");
        }
        (void) FNX_ARCH;
#else
        require_card_arch(dev, prop.name, prop.major, prop.minor, FNX_ARCH);
#endif
#if FNX_HOSTRING
        (void) skip_bar1;   // no card writes into another: the identity-domain and BAR1 requirements do not apply
#else
        const std::string bdf = card_bdf(dev), dir = "/sys/bus/pci/devices/" + bdf;
        require_identity_iommu(dev, dir);
        if (!skip_bar1) check_bar1(dev, bdf, dir, prop);
#endif
    }
#if FNX_HIP
    // a card Strata has not run the split on (anything but gfx1201): say so once, then check its kernels against a scalar reference before the model is loaded (hip/selftest_policy.hpp)
    {
        cudaDeviceProp prop;
        ck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        const SelftestDecision d = selftest_decide(prop.gcnArchName, std::getenv("FNX_SELFTEST"), std::getenv("FNX_SELFTEST_FAIL"), [&](std::string& covered) {
            for (int dev = 0; dev < NCARD; ++dev) {
                std::string c;
                const std::string bad = selftest_card(dev, c);
                if (!bad.empty()) return "card " + std::to_string(dev) + ": " + bad;
                covered += (covered.empty() ? "" : "; ") + c;
            }
            return std::string();
        });
        for (const std::string& line : d.lines) std::fprintf(stderr, "fnx: %s\n", line.c_str());
        if (!d.ok) throw std::runtime_error("the kernel self-test failed on " + std::string(prop.gcnArchName).substr(0, std::string(prop.gcnArchName).find(':')) + ", the split engine does not run on this card");
    }
#endif
}

float f16_to_f32(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = s << 31;
        else {
            int ee = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) { mm <<= 1; --ee; }
            u = (s << 31) | ((uint32_t) (ee + 127 - 14) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) u = (s << 31) | 0x7f800000u | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32_to_f16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t s = (u >> 16) & 0x8000u;
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (e <= 0) {   // subnormal or zero in f16
        if (e < -10) return (uint16_t) s;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t h = m >> shift;
        if ((m >> (shift - 1)) & 1u && ((m & ((1u << (shift - 1)) - 1)) || (h & 1u))) ++h;
        return (uint16_t) (s | h);
    }
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    if ((m >> 12) & 1u && ((m & 0xfffu) || (h & 1u))) ++h;   // nearest, ties to even
    return (uint16_t) (s | h);
}

uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// an F32 tensor's first n values, read on the host
const float* f32_data(const Tensor& t, size_t n) {
    if (t.type != T_F32) throw std::runtime_error(t.name + " is not F32");
    if (t.nbytes < n * sizeof(float)) throw std::runtime_error(t.name + " holds fewer than " + std::to_string(n) + " values");
    return (const float*) t.data;
}

// a tensor's range of the mapping locked, LOCKERS threads each faulting in a part: one thread faults a cold file in
// readahead-sized pieces (1.1 GB/s from the NVMe), 16 side by side reached 2.3 (tools/stage_test.cpp)
void lock_in_ram(const Gguf& g, const Tensor& t) {
    // Without CAP_IPC_LOCK a mapped table stays page cache, and the weights read after it push it out, so every row of a window could
    // wait for the disk (10-24 ms a round on this machine). A private anonymous copy stays in RAM and needs no limit raised.
    if (t.shard < 0) return;   // converted at load: its bytes are in RAM already, and a second copy only takes room
    constexpr int WORKERS = 16;
    constexpr size_t STEP = size_t(64) << 20;
    void* m = mmap(nullptr, t.nbytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) throw std::runtime_error("fnx: RAM for " + t.name + ": " + std::strerror(errno));
    madvise(m, t.nbytes, MADV_NOHUGEPAGE);   // loaded last, into RAM the pinned experts have fragmented: huge pages took 309 s of compaction instead of 55 s
    // the file's pages go as soon as they are copied: the copy and its page cache together filled RAM, and reclaim then sent the new copy to swap until swap was full
    const int fd = ::open(g.shard_path(t.shard).c_str(), O_RDONLY);
    const bool at_offset = g.shard_base(t.shard) + t.offset == t.data;
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int i = 0; i < WORKERS; ++i)
        th.emplace_back([&] {
            const uintptr_t PG = 4096;
            for (size_t lo; (lo = next.fetch_add(STEP)) < t.nbytes;) {
                const size_t n = std::min(STEP, t.nbytes - lo);
                std::memcpy((char*) m + lo, t.data + lo, n);
                const uintptr_t a0 = ((uintptr_t) t.data + lo + PG - 1) / PG * PG, e0 = ((uintptr_t) t.data + lo + n) / PG * PG;
                if (e0 <= a0) continue;
                madvise((void*) a0, e0 - a0, MADV_DONTNEED);
                if (fd >= 0 && at_offset) posix_fadvise(fd, (off_t) (t.offset + (a0 - (uintptr_t) t.data)), (off_t) (e0 - a0), POSIX_FADV_DONTNEED);
            }
        });
    for (auto& x : th) x.join();
    if (fd >= 0) ::close(fd);
    const_cast<Tensor&>(t).data = (const uint8_t*) m;
    std::fprintf(stderr, "fnx: %s (%zu MiB) copied into RAM\n", t.name.c_str(), (size_t) (t.nbytes >> 20));
}

// the stager's groups for load_card: group l every tensor of layer l, group NL the output head's. The head is read
// through the page cache: the draft layer (Mtp) takes its rows of output.weight from the mapping after the load
std::vector<Stager::Group> stage_groups(const Gguf& g) {
#if FNX_27B
    // and group NL + 1 the draft layer, blk.NL (Flash-Next's comes from separate files, Mtp)
    std::vector<Stager::Group> gs(NL + 2);
    for (const auto& kv : g.tensors()) {
        int l = -1, n = 0;
        if (std::sscanf(kv.first.c_str(), "blk.%d.%n", &l, &n) == 1 && n > 0 && l >= 0 && l <= NL)
            gs[(size_t) (l < NL ? l : NL + 1)].names.push_back(kv.first);
    }
    gs[NL].names = {"output_norm.weight", "output.weight"};
#else
    std::vector<Stager::Group> gs(NL + 1);
    for (const auto& kv : g.tensors()) {
        int l = -1, n = 0;
        if (kv.second.shard < 0 && !kv.second.data) continue;   // edge0_patch.py: FNX_LATENT shape-only tensors, never read
        if (std::sscanf(kv.first.c_str(), "blk.%d.%n", &l, &n) == 1 && n > 0 && l >= 0 && l < NL) gs[(size_t) l].names.push_back(kv.first);
    }
    gs[NL].names = {"output_hc_norm.weight", "output_hc_down.weight", "output_hc_up.weight", "output.weight"};
#endif
    gs[NL].cached = true;
    return gs;
}

// the column slicers take block indices that the callers count in 32-value blocks (ff0 / 32, HS / 32, HD / 32)
TypeInfo block32(const Tensor& t) {
    const TypeInfo ti = type_info(t.type);
    if (ti.block != 32) throw std::runtime_error(t.name + ": sliced by columns, which needs 32-value blocks, not " + std::to_string(ti.block));
    return ti;
}

// what each line of the load uploads, over both cards (FNX_PROF)
std::map<int, std::pair<size_t, long>> g_put_note;
std::mutex g_put_mu;
struct Up {
    CardW& c;
    Store* store = nullptr;               // the lab's holder: the shared memory instead of cudaMalloc / cudaHostAlloc
    std::vector<void*>* track = nullptr;  // a worker placing its experts: what it allocates (Model::free_experts)
    template <typename T> T* put(const void* host, size_t bytes, int line = __builtin_LINE()) {
        {
            std::lock_guard<std::mutex> g(g_put_mu);
            auto& e = g_put_note[line];
            e.first += bytes;
            ++e.second;
        }
        void* d = nullptr;
        if (store) d = store->dev(c.dev, bytes);
        else ck(cudaMalloc(&d, bytes), "cudaMalloc");
        if (track) track->push_back(d);
        ck(cudaMemcpy(d, host, bytes, cudaMemcpyDefault), "upload");
        c.bytes += bytes;
        return (T*) d;
    }
    // a norm as the window reads it: exactly n floats. A tensor of another width would feed garbage rows silently;
    // the whole tensor must match, so extra rows fail too (the window never reads them)
    float* f32(const Tensor& t, size_t n) {
        if (t.type != T_F32) throw std::runtime_error(t.name + " is not F32");
        if (t.nbytes != n * 4)
            throw std::runtime_error(t.name + " holds " + std::to_string(t.nbytes / 4) + " floats, not the " + std::to_string(n) +
                                     " the window reads it at");
        return put<float>(t.data, t.nbytes);
    }
    // where a tensor's routed experts are assembled: with an ExpertCache (host_of) their mapped page-locked host copy,
    // else tmp
    uint8_t* expert_buf(size_t bytes, const char** host_of, std::vector<uint8_t>& tmp) {
        if (!host_of) {
            tmp.resize(bytes);
            return tmp.data();
        }
        void* hp = nullptr;
        // FNX_NUMA_EXPERTS=1 (numa_patch.py): on the card's own socket (MPOL_BIND while the node has room, else MPOL_PREFERRED)
        static const bool numa = env_int("FNX_NUMA_EXPERTS", 0) != 0;
        const int node = numa && !store ? card_numa_node(c.dev) : -1;
        if (node >= 0 && node < 64) {
            const unsigned long mask = 1ul << node;
            const bool bind = numa_node_room(node) > bytes + ((size_t) 4 << 30);
            if (syscall(SYS_set_mempolicy, bind ? 2 /* MPOL_BIND */ : 1 /* MPOL_PREFERRED */, &mask, 8 * sizeof mask) != 0)
                throw std::runtime_error(std::string("FNX_NUMA_EXPERTS: set_mempolicy: ") + std::strerror(errno));
            (bind ? g_numa_bound : g_numa_pref)[c.dev & 1] += bytes;
        }
        const cudaError_t err = store ? (hp = store->host(c.dev, bytes), cudaSuccess) : cudaHostAlloc(&hp, bytes, cudaHostAllocMapped);
        if (node >= 0 && node < 64) syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, 0);
        if (err != cudaSuccess)
            throw std::runtime_error("place: " + std::to_string(bytes >> 20) + " MiB of page-locked host memory for experts: " +
                                     cudaGetErrorString(err));
        c.host_bytes += bytes;
        return (uint8_t*) hp;
    }
    // n routed experts of stride bytes each, copy(e, dst) writing expert e: to VRAM at w.p, or with an ExpertCache each one's
    // address in host_of (the layer's row of the card's host addresses). perm (the layer's row of Model::perm, FNX_EXPERT_DYN_GIB):
    // those go to a VRAM block of their own (perm_at[e], host_of[e] null) and only the others to page-locked host memory
    template <typename F>
    void experts(DW& w, size_t stride, int n, const char** host_of, const char* perm, const char** perm_at, F&& copy) {
        std::vector<uint8_t> tmp;
        if (!host_of) {
            uint8_t* h = expert_buf((size_t) n * stride, nullptr, tmp);
            for (int e = 0; e < n; ++e) copy(e, h + (size_t) e * stride);
            w.p = put<uint8_t>(h, (size_t) n * stride);
            return;
        }
        int np = 0;
        if (perm)
            for (int e = 0; e < n; ++e) np += perm[e] ? 1 : 0;
        if (np && store) throw std::runtime_error("experts: the holder keeps every expert's host copy (no FNX_EXPERT_DYN_GIB)");
        std::vector<uint8_t> pt((size_t) np * stride);
        uint8_t* h = np < n ? expert_buf((size_t) (n - np) * stride, host_of, tmp) : nullptr;
        for (int e = 0, kh = 0, kp = 0; e < n; ++e) copy(e, np && perm[e] ? &pt[(size_t) kp++ * stride] : h + (size_t) kh++ * stride);
        char* dp = nullptr;
        if (h) {
            if (store) dp = (char*) h;   // the shared host memory has one address for the CPU and the cards
            else ck(cudaHostGetDevicePointer((void**) &dp, (void*) h, 0), "place: mapped pointer");
        }
        char* vp = nullptr;
        if (np) {
            ck(cudaMalloc((void**) &vp, pt.size()), "experts: the permanent ones' block");
            ck(cudaMemcpy(vp, pt.data(), pt.size(), cudaMemcpyHostToDevice), "experts: the permanent ones' upload");
            c.perm_mem.push_back(vp);
            c.perm_bytes += pt.size();
            c.bytes += pt.size();
        }
        for (int e = 0, kh = 0, kp = 0; e < n; ++e)
            if (np && perm[e]) {
                host_of[e] = nullptr;
                perm_at[e] = vp + (size_t) kp++ * stride;
            } else {
                host_of[e] = dp + (size_t) kh++ * stride;
            }
    }
    // the card's ExpertCache: a slot for each resident expert (res) in its size's pool, filled from its host copy; the
    // layers' tables. With arena_score (FNX_ARENA) the largest pool's slots go coldest first and are neither allocated
    // nor filled here: the engine puts them in its arena (Engine::arena_fill); until then its experts point at RAM.
    // perm (FNX_EXPERT_DYN_GIB): those experts are in VRAM already (c.perm_gu / perm_dn) and take no pool slot: their slot
    // numbers follow the pools' (n_dyn and on), so a hit finds them and no CLOCK hand, lend or copy ever reaches them
    void expert_cache(const std::vector<std::vector<char>>& res, const std::vector<const char*>& hgu, const std::vector<const char*>& hdn,
                      const std::vector<double>* arena_score, const std::vector<std::vector<char>>& perm = {}) {
        ExpertCache& ec = c.ec;
        std::vector<std::pair<size_t, size_t>> sizes;   // each pool's (gate|up, down) bytes
        for (int l = 0; l < NL; ++l) {
            const LayerW& L = c.L[(size_t) l];
            if (L.gu_e_stride % 16 || L.down_e_stride % 16) throw std::runtime_error("expert cache: expert strides not in 16-byte units");
            const std::pair<size_t, size_t> s{L.gu_e_stride, L.down_e_stride};
            auto it = std::find(sizes.begin(), sizes.end(), s);
            if (it == sizes.end()) {
                if (sizes.size() == EC_POOLS) throw std::runtime_error("expert cache: more than " + std::to_string(EC_POOLS) + " expert sizes");
                sizes.push_back(s);
                it = sizes.end() - 1;
            }
            ec.pool_of[l] = (int) (it - sizes.begin());
        }
        ec.n_pools = (int) sizes.size();
        if (arena_score) {   // the largest pool by bytes
            size_t most = 0;
            for (int p = 0; p < ec.n_pools; ++p) {
                size_t b = 0;
                for (int l = 0; l < NL; ++l) b += ec.pool_of[l] == p ? (size_t) NEXP * (sizes[(size_t) p].first + sizes[(size_t) p].second) : 0;
                if (b > most) most = b, ec.arena_pool = p;
            }
        }
        std::vector<int> owner, slot_of((size_t) NL * NEXP, -1);
        auto permanent = [&](int l, int e) { return !perm.empty() && perm[(size_t) l][(size_t) e]; };
        for (int p = 0; p < ec.n_pools; ++p) {
            ec.s0[p] = (int) owner.size();
            for (int l = 0; l < NL; ++l)
                if (ec.pool_of[l] == p)
                    for (int e = 0; e < NEXP; ++e)
                        if (res[(size_t) l][(size_t) e] && !permanent(l, e)) owner.push_back(l * NEXP + e);
            if (p == ec.arena_pool)
                std::stable_sort(owner.begin() + ec.s0[p], owner.end(),
                                 [&](int a, int b) { return (*arena_score)[(size_t) a] < (*arena_score)[(size_t) b]; });
            for (int s = ec.s0[p]; s < (int) owner.size(); ++s) slot_of[(size_t) owner[(size_t) s]] = s;
            ec.n[p] = (int) owner.size() - ec.s0[p];
            if (ec.n[p] < 2 * MAXT * TOPK)
                throw std::runtime_error("expert cache: pool " + std::to_string(p) + " has " + std::to_string(ec.n[p]) + " slots, a window needs " +
                                         std::to_string(2 * MAXT * TOPK) + " (fewer experts in RAM)");
            ec.dn_off[p] = sizes[(size_t) p].first;
            ec.slot_bytes[p] = sizes[(size_t) p].first + sizes[(size_t) p].second;
            if (p == ec.arena_pool) continue;
            ck(cudaMalloc(&ec.pool[p], (size_t) ec.n[p] * ec.slot_bytes[p]), "expert cache: slots");
            if (track) track->push_back(ec.pool[p]);
            c.bytes += (size_t) ec.n[p] * ec.slot_bytes[p];
        }
        ec.n_dyn = (int) owner.size();
        std::vector<const char*> epg(hgu), epd(hdn);
        if (!perm.empty()) {   // the permanent experts: slot numbers after the pools', their VRAM blocks' addresses
            if (c.perm_gu.size() != (size_t) NL * NEXP || c.perm_dn.size() != (size_t) NL * NEXP)
                throw std::runtime_error("expert cache: the permanent experts were not placed");
            for (int l = 0; l < NL; ++l)
                for (int e = 0; e < NEXP; ++e)
                    if (permanent(l, e)) {
                        const size_t i = (size_t) l * NEXP + e;
                        if (!res[(size_t) l][(size_t) e] || !c.perm_gu[i] || !c.perm_dn[i] || hgu[i] || hdn[i])
                            throw std::runtime_error("expert cache: a permanent expert with a host copy or without its VRAM block");
                        slot_of[i] = (int) owner.size();
                        owner.push_back((int) i);
                        epg[i] = c.perm_gu[i];
                        epd[i] = c.perm_dn[i];
                    }
        }
        ec.n_slots = (int) owner.size();
        ec.arena_full = arena_score && ec.n_slots == NL * NEXP;
        for (int s = 0; s < ec.n_dyn; ++s) {
            const int i = owner[(size_t) s], p = ec.pool_of[i / NEXP];
            if (p == ec.arena_pool) continue;
            const LayerW& L = c.L[(size_t) (i / NEXP)];
            char* at = ec.pool[p] + (size_t) (s - ec.s0[p]) * ec.slot_bytes[p];
            ck(cudaMemcpy(at, hgu[(size_t) i], L.gu_e_stride, cudaMemcpyDefault), "expert cache: fill");
            ck(cudaMemcpy(at + ec.dn_off[p], hdn[(size_t) i], L.down_e_stride, cudaMemcpyDefault), "expert cache: fill");
            epg[(size_t) i] = at;
            epd[(size_t) i] = at + ec.dn_off[p];
        }
        ec.ep_gu = put<const char*>(epg.data(), epg.size() * sizeof(char*));
        ec.ep_dn = put<const char*>(epd.data(), epd.size() * sizeof(char*));
        ec.host_gu = put<const char*>(hgu.data(), hgu.size() * sizeof(char*));
        ec.host_dn = put<const char*>(hdn.data(), hdn.size() * sizeof(char*));
        c.ec_host_gu = hgu;
        c.ec_host_dn = hdn;
        ec.slot_of = put<int>(slot_of.data(), slot_of.size() * sizeof(int));
        c.ec_owner0 = owner;
        c.ec_slot0 = slot_of;
        ec.owner = put<int>(owner.data(), owner.size() * sizeof(int));
        const std::vector<unsigned char> ref((size_t) ec.n_slots, 0);
        ec.ref = put<unsigned char>(ref.data(), ref.size());
        ec.hand = put<int>(ec.s0, sizeof ec.s0);   // each pool's hand at its first slot
        const int none[2 * (EC_POOLS)] = {};
        ec.lent = put<int>(none, sizeof none);
        const int unfilled[3 * EC_POOLS] = {};
        ec.fill = put<int>(unfilled, sizeof unfilled);
        const std::vector<int> unheld((size_t) ec.n_slots, -1);
        ec.fill_list = put<int>(unheld.data(), unheld.size() * sizeof(int));
        const float noprior = 0.f;
        ec.prior = put<float>(&noprior, sizeof noprior);
        for (int l = 0; l < NL; ++l) {
            c.L[(size_t) l].gu_e.ep = ec.ep_gu + (size_t) l * NEXP;
            c.L[(size_t) l].down_e.ep = ec.ep_dn + (size_t) l * NEXP;
        }
    }
    // row r of t (counting every matrix's rows, in order), which must be one of its rows: a row list made from the model's
    // geometry would otherwise read the next tensor's bytes from the mapping when a GGUF's tensor is shorter than it
    static const uint8_t* row_of(const Tensor& t, int r, size_t row_bytes) {
        if (r < 0 || r >= t.nrows())
            throw std::runtime_error(t.name + ": row " + std::to_string(r) + " asked for, it has " + std::to_string(t.nrows()));
        return t.data + (size_t) r * row_bytes;
    }
    // rows `rows` of a 2-D F32 tensor (ne1 indexes rows)
    float* f32_rows(const Tensor& t, const std::vector<int>& rows, size_t row_floats) {
        if (t.type != T_F32) throw std::runtime_error(t.name + " is not F32");
        if ((size_t) t.ne[0] != row_floats)
            throw std::runtime_error(t.name + " is " + std::to_string(t.ne[0]) + " floats a row, not the " + std::to_string(row_floats) +
                                     " its reader takes");
        std::vector<float> h(rows.size() * (size_t) t.ne[0]);
        for (size_t i = 0; i < rows.size(); ++i)
            std::memcpy(&h[i * t.ne[0]], row_of(t, rows[i], (size_t) t.ne[0] * 4), (size_t) t.ne[0] * 4);
        return put<float>(h.data(), h.size() * 4);
    }
    // Q8_0 -> bf16, the whole tensor; F32 -> bf16 likewise
    uint16_t* bf16(const Tensor& t) {
        const size_t n = (size_t) (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
        std::vector<uint16_t> h(n);
        if (t.type == T_F32) {
            const float* f = (const float*) t.data;
            for (size_t i = 0; i < n; ++i) h[i] = f32_to_bf16(f[i]);
        } else if (t.type == T_Q8_0) {
            for (size_t b = 0; b < n / 32; ++b) {
                const uint8_t* blk = t.data + b * 34;
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                const float d = f16_to_f32(dh);
                for (int j = 0; j < 32; ++j) h[b * 32 + j] = f32_to_bf16(d * (float) (int8_t) blk[2 + j]);
            }
        } else if (t.type == T_BF16) {
            std::memcpy(h.data(), t.data, n * 2);
        } else {
            throw std::runtime_error(t.name + ": bf16 conversion supports F32, Q8_0 and BF16");
        }
        return put<uint16_t>(h.data(), h.size() * 2);
    }
    // a dense matrix for q8s_gemv: Q8_0 slices move to the split layout (other types stay as they are)
    DW dense(DW w) {
        if (w.type != T_Q8_0) return w;
        const size_t bytes = (size_t) w.n_out * w.row_bytes, nblk = bytes / 34;
        std::vector<uint8_t> h(bytes);
        ck(cudaMemcpy(h.data(), w.p, bytes, cudaMemcpyDeviceToHost), "dense: download");
        if (store) store->free_dev(c.dev, w.p, bytes);   // the upload just before: its room is used again
        else ck(cudaFree(w.p), "dense: free");
        if (track) track->erase(std::remove(track->begin(), track->end(), (void*) w.p), track->end());   // not freed twice
        c.bytes -= bytes;
        std::vector<int8_t> q(nblk * 32);
        std::vector<uint16_t> d(nblk);
        for (size_t b = 0; b < nblk; ++b) {
            std::memcpy(&d[b], &h[b * 34], 2);
            std::memcpy(&q[b * 32], &h[b * 34 + 2], 32);
        }
        w.p = nullptr;
        w.s.q = put<int8_t>(q.data(), q.size());
        w.s.d = put<uint16_t>(d.data(), d.size() * 2);
        return w;
    }
    // bf16 values [n] as the split int8 layout (per 32 values: d = amax / 127, ggml's rounding)
    Q8S q8split_bf16(const uint16_t* v, size_t n) {
        std::vector<int8_t> q(n);
        std::vector<uint16_t> d(n / 32);
        for (size_t b = 0; b < n / 32; ++b) {
            float x[32], amax = 0.f;
            for (int j = 0; j < 32; ++j) {
                const uint32_t u = (uint32_t) v[b * 32 + j] << 16;
                std::memcpy(&x[j], &u, 4);
                amax = std::max(amax, std::fabs(x[j]));
            }
            const float dd = amax / 127.0f, id = dd > 0.f ? 1.0f / dd : 0.f;
            for (int j = 0; j < 32; ++j) q[b * 32 + j] = (int8_t) std::lround(x[j] * id);
            d[b] = f32_to_f16(dd);
        }
        Q8S s;
        s.q = put<int8_t>(q.data(), q.size());
        s.d = put<uint16_t>(d.data(), d.size() * 2);
        return s;
    }
    // a whole Q8_0 tensor as int8 values and f16 scales
    Q8S q8split(const Tensor& t) {
        if (t.type != T_Q8_0) throw std::runtime_error(t.name + ": q8split needs Q8_0");
        const size_t n = (size_t) (t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3]);
        std::vector<int8_t> q(n);
        std::vector<uint16_t> d(n / 32);
        for (size_t b = 0; b < n / 32; ++b) {
            const uint8_t* blk = t.data + b * 34;
            std::memcpy(&d[b], blk, 2);
            std::memcpy(&q[b * 32], blk + 2, 32);
        }
        Q8S s;
        s.q = put<int8_t>(q.data(), q.size());
        s.d = put<uint16_t>(d.data(), d.size() * 2);
        return s;
    }
    // whole rows of a quantized 2-D tensor (or of matrix `mat` of a 3-D one)
    DW rows(const Tensor& t, const std::vector<int>& rows) {
        DW w;
        w.type = t.type;
        w.n_in = (int) t.ne[0];
        w.n_out = (int) rows.size();
        w.row_bytes = t.row_bytes();
        std::vector<uint8_t> h(rows.size() * w.row_bytes);
        for (size_t i = 0; i < rows.size(); ++i) std::memcpy(&h[i * w.row_bytes], row_of(t, rows[i], w.row_bytes), w.row_bytes);
        w.p = put<uint8_t>(h.data(), h.size());
        return w;
    }
    // rows r1 of t1 then rows r2 of t2 (same type and row width), one matrix
    DW rows2(const Tensor& t1, const std::vector<int>& r1, const Tensor& t2, const std::vector<int>& r2) {
        if (t1.type != t2.type || t1.ne[0] != t2.ne[0]) throw std::runtime_error(t1.name + " and " + t2.name + " do not stack");
        DW w;
        w.type = t1.type;
        w.n_in = (int) t1.ne[0];
        w.n_out = (int) (r1.size() + r2.size());
        w.row_bytes = t1.row_bytes();
        std::vector<uint8_t> h((size_t) w.n_out * w.row_bytes);
        size_t o = 0;
        for (int r : r1) { std::memcpy(&h[o], row_of(t1, r, w.row_bytes), w.row_bytes); o += w.row_bytes; }
        for (int r : r2) { std::memcpy(&h[o], row_of(t2, r, w.row_bytes), w.row_bytes); o += w.row_bytes; }
        w.p = put<uint8_t>(h.data(), h.size());
        return w;
    }
#if FNX_27B
    // rows r1 of t1, r2 of t2, then r3 of t3, one matrix
    DW rows3(const Tensor& t1, const std::vector<int>& r1, const Tensor& t2, const std::vector<int>& r2, const Tensor& t3,
             const std::vector<int>& r3) {
        for (const Tensor* t : {&t2, &t3})
            if (t1.type != t->type || t1.ne[0] != t->ne[0]) throw std::runtime_error(t1.name + " and " + t->name + " do not stack");
        DW w;
        w.type = t1.type;
        w.n_in = (int) t1.ne[0];
        w.n_out = (int) (r1.size() + r2.size() + r3.size());
        w.row_bytes = t1.row_bytes();
        std::vector<uint8_t> h((size_t) w.n_out * w.row_bytes);
        size_t o = 0;
        for (const auto& [t, rs] : {std::pair{&t1, &r1}, std::pair{&t2, &r2}, std::pair{&t3, &r3}})
            for (int r : *rs) { std::memcpy(&h[o], row_of(*t, r, w.row_bytes), w.row_bytes); o += w.row_bytes; }
        w.p = put<uint8_t>(h.data(), h.size());
        return w;
    }
#endif
    // rows `rows` of t1 then of t2 as f32 (F32 rows as they are, Q8_0 rows dequantized)
    float* f32_rows2(const Tensor& t1, const Tensor& t2, const std::vector<int>& rows, size_t row_floats) {
        std::vector<float> h;
        for (const Tensor* t : {&t1, &t2}) {
            const size_t n = (size_t) t->ne[0];
            if (n != row_floats)
                throw std::runtime_error(t->name + " is " + std::to_string(n) + " floats a row, not the " + std::to_string(row_floats) +
                                         " its reader takes");
            for (int r : rows) {
                if (t->type == T_F32) {
                    const float* row = (const float*) row_of(*t, r, n * 4);
                    h.insert(h.end(), row, row + n);
                } else if (t->type == T_Q8_0 && n % 32 == 0) {
                    const uint8_t* row = row_of(*t, r, t->row_bytes());
                    for (size_t b = 0; b < n / 32; ++b) {
                        uint16_t dh;
                        std::memcpy(&dh, row + b * 34, 2);
                        const float d = f16_to_f32(dh);
                        for (int j = 0; j < 32; ++j) h.push_back(d * (float) (int8_t) row[b * 34 + 2 + j]);
                    }
                } else {
                    throw std::runtime_error(t->name + " is neither F32 nor Q8_0");
                }
            }
        }
        return put<float>(h.data(), h.size() * 4);
    }
    // every row, keeping input blocks `blocks` (block indices along ne0) of each: a column slice
    DW cols(const Tensor& t, const std::vector<int>& blocks) {
        const TypeInfo ti = block32(t);
        DW w;
        w.type = t.type;
        w.n_in = (int) blocks.size() * ti.block;
        w.n_out = (int) t.ne[1];
        w.row_bytes = blocks.size() * ti.bytes;
        const size_t src_row = t.row_bytes();
        for (int b : blocks)
            if (b < 0 || (size_t) (b + 1) * ti.bytes > src_row)
                throw std::runtime_error(t.name + ": block " + std::to_string(b) + " asked for, a row has " + std::to_string(src_row / ti.bytes));
        std::vector<uint8_t> h((size_t) w.n_out * w.row_bytes);
        for (int r = 0; r < w.n_out; ++r)
            for (size_t b = 0; b < blocks.size(); ++b)
                std::memcpy(&h[(size_t) r * w.row_bytes + b * ti.bytes], t.data + (size_t) r * src_row + (size_t) blocks[b] * ti.bytes, ti.bytes);
        w.p = put<uint8_t>(h.data(), h.size());
        return w;
    }
    // per expert: rows [r0, r1) (gate / up), expert-major
    static void row_range(const Tensor& t, int r0, int r1) {
        if (r0 < 0 || r1 <= r0 || r1 > t.ne[1])
            throw std::runtime_error(t.name + ": rows [" + std::to_string(r0) + ", " + std::to_string(r1) + ") asked for, it has " +
                                     std::to_string(t.ne[1]) + " a matrix");
    }
    DW expert_rows(const Tensor& t, int r0, int r1, size_t& stride) {
        row_range(t, r0, r1);
        DW w;
        w.type = t.type;
        w.n_in = (int) t.ne[0];
        w.n_out = r1 - r0;
        w.row_bytes = t.row_bytes();
        stride = (size_t) w.n_out * w.row_bytes;
        const size_t src_mat = (size_t) t.ne[1] * w.row_bytes;
        std::vector<uint8_t> h((size_t) t.ne[2] * stride);
        for (int64_t e = 0; e < t.ne[2]; ++e) std::memcpy(&h[e * stride], t.data + e * src_mat + (size_t) r0 * w.row_bytes, stride);
        w.p = put<uint8_t>(h.data(), h.size());
        return w;
    }
    // per expert: rows [r0, r1) of t1 then of t2 (gate; up), expert-major
    DW expert_rows2(const Tensor& t1, const Tensor& t2, int r0, int r1, size_t& stride, const char** host_of = nullptr,
                    const char* perm = nullptr, const char** perm_at = nullptr) {
        if (t1.type != t2.type || t1.ne[0] != t2.ne[0] || t1.ne[2] != t2.ne[2]) throw std::runtime_error(t1.name + " and " + t2.name + " do not stack");
        row_range(t1, r0, r1);
        row_range(t2, r0, r1);
        DW w;
        w.type = t1.type;
        w.n_in = (int) t1.ne[0];
        w.n_out = 2 * (r1 - r0);
        w.row_bytes = t1.row_bytes();
        stride = (size_t) w.n_out * w.row_bytes;
        const size_t half = (size_t) (r1 - r0) * w.row_bytes, src_mat = (size_t) t1.ne[1] * w.row_bytes;
        experts(w, stride, (int) t1.ne[2], host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {
            std::memcpy(dst, t1.data + e * src_mat + (size_t) r0 * w.row_bytes, half);
            std::memcpy(dst + half, t2.data + e * src_mat + (size_t) r0 * w.row_bytes, half);
        });
        return w;
    }
    // f32 tensors' rows one after another
    float* f32_cat(const Tensor& t1, const Tensor& t2, size_t row_floats) {
        if (t1.type != T_F32 || t2.type != T_F32 || t1.ne[0] != t2.ne[0]) throw std::runtime_error(t1.name + " and " + t2.name + " do not stack");
        if ((size_t) t1.ne[0] != row_floats)
            throw std::runtime_error(t1.name + " is " + std::to_string(t1.ne[0]) + " floats a row, not the " + std::to_string(row_floats) +
                                     " its reader takes");
        std::vector<float> h((const float*) t1.data, (const float*) t1.data + t1.ne[0] * t1.ne[1]);
        h.insert(h.end(), (const float*) t2.data, (const float*) t2.data + t2.ne[0] * t2.ne[1]);
        return put<float>(h.data(), h.size() * 4);
    }
    // the same rows as bf16, which they were in the file (the load made them f32): exact, or it throws
    uint16_t* bf16_cat(const Tensor& t1, const Tensor& t2, size_t row_floats) {
        if (t1.type != T_F32 || t2.type != T_F32 || t1.ne[0] != t2.ne[0]) throw std::runtime_error(t1.name + " and " + t2.name + " do not stack");
        if ((size_t) t1.ne[0] != row_floats)
            throw std::runtime_error(t1.name + " is " + std::to_string(t1.ne[0]) + " floats a row, not the " + std::to_string(row_floats) +
                                     " its reader takes");
        std::vector<uint16_t> h;
        h.reserve((size_t) (t1.ne[1] + t2.ne[1]) * row_floats);
        for (const Tensor* t : {&t1, &t2}) {
            const uint32_t* u = (const uint32_t*) t->data;
            for (size_t i = 0; i < (size_t) (t->ne[0] * t->ne[1]); ++i) {
                if (u[i] & 0xffffu) throw std::runtime_error(t->name + ": not bf16 values (the router is kept as bf16)");
                h.push_back((uint16_t) (u[i] >> 16));
            }
        }
        return put<uint16_t>(h.data(), h.size() * 2);
    }
    // per expert: every row, input blocks [b0, b1) (down), expert-major
    DW expert_cols(const Tensor& t, int b0, int b1, size_t& stride, const char** host_of = nullptr, const char* perm = nullptr,
                   const char** perm_at = nullptr) {
        const TypeInfo ti = block32(t);
        DW w;
        w.type = t.type;
        w.n_in = (b1 - b0) * ti.block;
        w.n_out = (int) t.ne[1];
        w.row_bytes = (size_t) (b1 - b0) * ti.bytes;
        stride = (size_t) w.n_out * w.row_bytes;
        const size_t src_row = t.row_bytes(), src_mat = (size_t) t.ne[1] * src_row;
        if (b0 < 0 || b1 <= b0 || (size_t) b1 * ti.bytes > src_row)
            throw std::runtime_error(t.name + ": blocks [" + std::to_string(b0) + ", " + std::to_string(b1) + ") asked for, a row has " +
                                     std::to_string(src_row / ti.bytes));
        experts(w, stride, (int) t.ne[2], host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {
            for (int r = 0; r < w.n_out; ++r)
                std::memcpy(dst + (size_t) r * w.row_bytes, t.data + e * src_mat + (size_t) r * src_row + (size_t) b0 * ti.bytes, w.row_bytes);
        });
        return w;
    }
    // edge0_patch.py: Edge0's latent experts from their blobs (blk.l.lat_exps): gate rows [r0, r1) then up rows, 1792 values a row
    DW lat_rows2(const Tensor& b, int r0, int r1, size_t& stride, const char** host_of, const char* perm, const char** perm_at) {
        constexpr size_t ROW = LAT_ELL / 64 * 36, UP = (size_t) FF * ROW;
        if (r0 < 0 || r1 <= r0 || r1 > FF) throw std::runtime_error(b.name + ": rows [" + std::to_string(r0) + ", " + std::to_string(r1) + ")");
        DW w;
        w.type = T_NVFP4;
        w.n_in = LAT_ELL;
        w.n_out = 2 * (r1 - r0);
        w.row_bytes = ROW;
        stride = (size_t) w.n_out * ROW;
        const size_t half = (size_t) (r1 - r0) * ROW;
        experts(w, stride, NEXP, host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {
            const uint8_t* src = b.data + (size_t) e * LAT_BLOB;
            std::memcpy(dst, src + (size_t) r0 * ROW, half);
            std::memcpy(dst + half, src + UP + (size_t) r0 * ROW, half);
        });
        return w;
    }
    // down: every one of its 1792 rows, input blocks [b0, b1) of 64 values
    DW lat_cols(const Tensor& b, int b0, int b1, size_t& stride, const char** host_of, const char* perm, const char** perm_at) {
        constexpr size_t SRC = FF / 64 * 36, DOWN = (size_t) 2 * FF * (LAT_ELL / 64 * 36);
        if (b0 < 0 || b1 <= b0 || b1 > FF / 64) throw std::runtime_error(b.name + ": down blocks [" + std::to_string(b0) + ", " + std::to_string(b1) + ")");
        DW w;
        w.type = T_NVFP4;
        w.n_in = (b1 - b0) * 64;
        w.n_out = LAT_ELL;
        w.row_bytes = (size_t) (b1 - b0) * 36;
        stride = (size_t) w.n_out * w.row_bytes;
        experts(w, stride, NEXP, host_of, perm, perm_at, [&](int64_t e, uint8_t* dst) {
            const uint8_t* src = b.data + (size_t) e * LAT_BLOB + DOWN;
            for (int r = 0; r < w.n_out; ++r) std::memcpy(dst + (size_t) r * w.row_bytes, src + (size_t) r * SRC + (size_t) b0 * 36, w.row_bytes);
        });
        return w;
    }
};

std::vector<int> range(int a, int b) {
    std::vector<int> v;
    for (int i = a; i < b; ++i) v.push_back(i);
    return v;
}

#if FNX_27B
// The 27B's layers on a card, from load_card's channel lists: the recurrent layers as Flash-Next's; attention split by
// heads (K/V too: each card keeps its own heads' cache, src/s27_attn.cuh), q | gate, K and V stacked in one matrix;
// the dense FFN by FF width; then the head, and the draft layer (blk.NL, stage group NL + 1). Every matrix q8s reads
// is Q8_0: the 27B's path has no other matvec
void load_layers_27b(Stager& st, CardW& c, Up& up, const std::vector<int>& conv_rows, const std::vector<int>& zrows,
                     const std::vector<int>& vblocks, const std::vector<int>& qrows, const std::vector<int>& kvrows,
                     const std::vector<int>& woblocks) {
    auto L = [&](int l, const char* s) -> const Tensor& { return st.t(l, "blk." + std::to_string(l) + "." + s); };
    const std::vector<int>& vheads = c.vheads;
    auto q8 = [](const DW& d, const std::string& what) {
        if (!d.s.q) throw std::runtime_error("27b: " + what + " is not Q8_0 (the 27B's matvecs read Q8_0 only)");
    };
    // the shape the window writes and reads it at (rows are the output stride; a wrong one would land at another)
    auto shape = [](const DW& d, int out, int in, const std::string& what) {
        if (d.n_out != out || d.n_in != in)
            throw std::runtime_error("27b: " + what + " is " + std::to_string(d.n_out) + " x " + std::to_string(d.n_in) + ", not " +
                                     std::to_string(out) + " x " + std::to_string(in));
    };
    const int ffs = c.fd1 - c.fd0, qkv = c.nq * 2 * HD + 2 * c.nkv * HD;
    auto attn = [&](LayerW& w, auto&& T) {
        w.wq = up.dense(up.rows3(T("attn_q.weight"), qrows, T("attn_k.weight"), kvrows, T("attn_v.weight"), kvrows));
        w.wo = up.dense(up.cols(T("attn_output.weight"), woblocks));
        w.q_norm = up.f32(T("attn_q_norm.weight"), HD);
        w.k_norm = up.f32(T("attn_k_norm.weight"), HD);
    };
    auto ffn = [&](LayerW& w, auto&& T) {
        w.attn_norm = up.f32(T("attn_norm.weight"), N);
        w.ffn_norm = up.f32(T("post_attention_norm.weight"), N);
        w.ffn_gu = up.dense(up.rows2(T("ffn_gate.weight"), range(c.fd0, c.fd1), T("ffn_up.weight"), range(c.fd0, c.fd1)));
        w.ffn_down = up.dense(up.cols(T("ffn_down.weight"), range(c.fd0 / 32, c.fd1 / 32)));
    };
    c.L.resize(NL);
    for (int l = 0; l < NL; ++l) {
        LayerW& w = c.L[l];
        auto T = [&](const char* s) -> const Tensor& { return L(l, s); };
        if (!is_attn(l)) {
            w.qkvz = up.dense(up.rows2(T("attn_qkv.weight"), conv_rows, T("attn_gate.weight"), zrows));
            w.ab = up.f32_rows2(T("ssm_alpha.weight"), T("ssm_beta.weight"), vheads, N);
            w.ssm_out = up.dense(up.cols(T("ssm_out.weight"), vblocks));
            w.conv = up.f32_rows(T("ssm_conv1d.weight"), conv_rows, 4);
            std::vector<float> a, dt;
            const size_t nh = (size_t) *std::max_element(vheads.begin(), vheads.end()) + 1;
            const float* af = f32_data(T("ssm_a"), nh);
            const float* df = f32_data(T("ssm_dt.bias"), nh);
            for (int h : vheads) { a.push_back(af[h]); dt.push_back(df[h]); }
            w.ssm_a = up.put<float>(a.data(), a.size() * 4);
            w.dt_bias = up.put<float>(dt.data(), dt.size() * 4);
            w.ssm_norm = up.f32(T("ssm_norm.weight"), HS);
            q8(w.qkvz, "layer " + std::to_string(l) + "'s qkv | gate"), q8(w.ssm_out, "layer " + std::to_string(l) + "'s ssm_out");
            shape(w.qkvz, c.conv_ch() + c.nv() * HS, N, "layer " + std::to_string(l) + "'s qkv | gate");
            shape(w.ssm_out, N, c.nv() * HS, "layer " + std::to_string(l) + "'s ssm_out");
        } else {
            attn(w, T);
            q8(w.wq, "layer " + std::to_string(l) + "'s q | k | v"), q8(w.wo, "layer " + std::to_string(l) + "'s attn_output");
            shape(w.wq, qkv, N, "layer " + std::to_string(l) + "'s q | k | v");
            shape(w.wo, N, c.nq * HD, "layer " + std::to_string(l) + "'s attn_output");
        }
        ffn(w, T);
        q8(w.ffn_gu, "layer " + std::to_string(l) + "'s gate | up"), q8(w.ffn_down, "layer " + std::to_string(l) + "'s down");
        shape(w.ffn_gu, 2 * ffs, N, "layer " + std::to_string(l) + "'s gate | up");
        shape(w.ffn_down, N, ffs, "layer " + std::to_string(l) + "'s down");
        st.done(l);
    }
    c.out_norm = up.f32(st.t(NL, "output_norm.weight"), N);
    c.out = up.dense(up.rows(st.t(NL, "output.weight"), range(c.v0, c.v1)));
    q8(c.out, "output.weight");
    shape(c.out, c.v1 - c.v0, N, "output.weight's rows");
    st.done(NL);
    {
        auto T = [&](const char* s) -> const Tensor& { return st.t(NL + 1, "blk." + std::to_string(NL) + "." + s); };
        attn(c.mtp, T);
        ffn(c.mtp, T);
        c.mtp_eh = up.dense(up.rows(T("nextn.eh_proj.weight"), range(0, N)));   // every card the whole of it
        c.mtp_enorm = up.f32(T("nextn.enorm.weight"), N);
        c.mtp_hnorm = up.f32(T("nextn.hnorm.weight"), N);
        c.mtp_head_norm = up.f32(T("nextn.shared_head_norm.weight"), N);
        for (const DW* d : {&c.mtp.wq, &c.mtp.wo, &c.mtp.ffn_gu, &c.mtp.ffn_down, &c.mtp_eh}) q8(*d, "a draft layer matrix");
        shape(c.mtp.wq, qkv, N, "the draft layer's q | k | v");
        shape(c.mtp.wo, N, c.nq * HD, "the draft layer's attn_output");
        shape(c.mtp.ffn_gu, 2 * ffs, N, "the draft layer's gate | up");
        shape(c.mtp.ffn_down, N, ffs, "the draft layer's down");
        shape(c.mtp_eh, N, 2 * N, "the draft layer's eh_proj");
        st.done(NL + 1);
    }
}
#endif

// The tensors come from st: group l is layer l's, group NL the output head's (stage_groups); each group is given back
// once this card is through with it.
// host_experts: every routed expert to a host copy (an ExpertCache, or the lab's holder); store: the holder's shared memory.
// perm (FNX_EXPERT_DYN_GIB): those experts to VRAM for good as their layer loads, with no host copy
// FNX_LATENT (edge0_patch.py): Edge0's group labels [NL][NEXP], in the mapping (Model::Model sets it)
const int32_t* g_lat_labels = nullptr;

void load_card(Stager& st, CardW& c, const std::vector<std::vector<char>>& resident, const std::vector<double>* arena_score,
               Store* store, bool host_experts, const std::vector<std::vector<char>>& perm = {}) {
    ck(cudaSetDevice(c.dev), "cudaSetDevice");
    Up up{c, store};
    if (!perm.empty()) {
        c.perm_gu.assign((size_t) NL * NEXP, nullptr);
        c.perm_dn.assign((size_t) NL * NEXP, nullptr);
    }
    auto L = [&](int l, const char* s) -> const Tensor& { return st.t(l, "blk." + std::to_string(l) + "." + s); };
    // recurrent channel lists: q, k for this card's key heads; v for its value heads (s * NKH + k)
    std::vector<int> qch, kch, vch, vblocks;
    const std::vector<int>& vheads = c.vheads;
    for (int k = c.k0; k < c.k0 + c.nk; ++k)
        for (int i = 0; i < HS; ++i) { qch.push_back(k * HS + i); kch.push_back(NKH * HS + k * HS + i); }
    {
        std::vector<int> vmap;
        std::vector<bool> seen(c.nk, false);
        for (int h : vheads) {
            const int kl = h % NKH - c.k0;
            if (kl < 0 || kl >= c.nk) throw std::runtime_error("fnx: a value head whose key head is not on its card");
            vmap.push_back(kl | (seen[kl] ? 0 : 1 << 8));
            seen[kl] = true;
            for (int i = 0; i < HS; ++i) vch.push_back(2 * NKH * HS + h * HS + i);
            for (int b = 0; b < HS / 32; ++b) vblocks.push_back(h * HS / 32 + b);
        }
        for (int kl = 0; kl < c.nk; ++kl)
            if (!seen[kl]) throw std::runtime_error("fnx: a key head with no value head on its card");
        c.vmap = up.put<int>(vmap.data(), vmap.size() * 4);
    }
    std::vector<int> conv_rows = qch;
    conv_rows.insert(conv_rows.end(), kch.begin(), kch.end());
    conv_rows.insert(conv_rows.end(), vch.begin(), vch.end());
    std::vector<int> zrows;
    for (int ch : vch) zrows.push_back(ch - 2 * NKH * HS);
    // attention rows: q heads (q + gate, 2 * HD rows each), KV heads; wo input blocks of the q heads
    std::vector<int> qrows, kvrows, woblocks;
    for (int h = c.q0; h < c.q0 + c.nq; ++h) {
        for (int i = 0; i < 2 * HD; ++i) qrows.push_back(h * 2 * HD + i);
        for (int b = 0; b < HD / 32; ++b) woblocks.push_back(h * HD / 32 + b);
    }
    for (int h = c.kv0; h < c.kv0 + c.nkv; ++h)
        for (int i = 0; i < HD; ++i) kvrows.push_back(h * HD + i);
#if FNX_27B
    (void) resident, (void) arena_score, (void) host_experts;
    load_layers_27b(st, c, up, conv_rows, zrows, vblocks, qrows, kvrows, woblocks);
    return;
#endif
    // the K/V cache is split by position: every card projects q for every head and k/v for both KV heads
    std::vector<int> qrows_all, kvrows_all;
    for (int i = 0; i < NQH * 2 * HD; ++i) qrows_all.push_back(i);
    for (int i = 0; i < NKVH * HD; ++i) kvrows_all.push_back(i);
    c.L.resize(NL);
    std::vector<const char*> hgu(host_experts ? (size_t) NL * NEXP : 0), hdn(hgu.size());   // the experts' host copies
    for (int l = 0; l < NL; ++l) {
        LayerW& w = c.L[l];
        const char** hg = host_experts ? hgu.data() + (size_t) l * NEXP : nullptr;
        const char** hd = host_experts ? hdn.data() + (size_t) l * NEXP : nullptr;
        const char* hp[2] = {"hc_attn_", "hc_ffn_"};
        for (int h = 0; h < 2; ++h) {
            w.hc_norm[h] = up.f32(L(l, (std::string(hp[h]) + "norm.weight").c_str()), D);
            w.hc_down[h] = up.q8split(L(l, (std::string(hp[h]) + "down.weight").c_str()));
            w.hc_up[h] = up.q8split(L(l, (std::string(hp[h]) + "up.weight").c_str()));
            w.hc_inj[h] = up.bf16(L(l, (std::string(hp[h]) + "inject.weight").c_str()));
        }
        const Tensor& gi = L(l, "ffn_gate_inp.weight");
        const Tensor& gs = L(l, "ffn_gate_inp_shexp.weight");
        if (gi.ne[1] != NEXP || gs.ne[1] != 1)
            throw std::runtime_error("fnx: " + gi.name + " holds " + std::to_string(gi.ne[1]) + " rows and " + gs.name + " " +
                                     std::to_string(gs.ne[1]) + ", not the " + std::to_string(NEXP) +
                                     " gate rows and 1 shared row k_router_route reads");
        w.router = up.bf16_cat(gi, gs, N);
        if (!is_attn(l)) {
            w.qkvz = up.dense(up.rows2(L(l, "attn_qkv.weight"), conv_rows, L(l, "attn_gate.weight"), zrows));
            w.ab = up.f32_rows2(L(l, "ssm_alpha.weight"), L(l, "ssm_beta.weight"), vheads, N);
            w.ssm_out = up.dense(up.cols(L(l, "ssm_out.weight"), vblocks));
            w.conv = up.f32_rows(L(l, "ssm_conv1d.weight"), conv_rows, 4);
            std::vector<float> a, dt;
            const size_t nh = (size_t) *std::max_element(vheads.begin(), vheads.end()) + 1;
            const float* af = f32_data(L(l, "ssm_a"), nh);
            const float* df = f32_data(L(l, "ssm_dt.bias"), nh);
            for (int h : vheads) { a.push_back(af[h]); dt.push_back(df[h]); }
            w.ssm_a = up.put<float>(a.data(), a.size() * 4);
            w.dt_bias = up.put<float>(dt.data(), dt.size() * 4);
            w.ssm_norm = up.f32(L(l, "ssm_norm.weight"), HS);
        } else {
            w.wq = up.dense(up.rows(L(l, "attn_q.weight"), qrows_all));
            w.wk = up.dense(up.rows(L(l, "attn_k.weight"), kvrows_all));
            w.wv = up.dense(up.rows(L(l, "attn_v.weight"), kvrows_all));
            w.wo = up.dense(up.cols(L(l, "attn_output.weight"), woblocks));
            w.q_norm = up.f32(L(l, "attn_q_norm.weight"), HD);
            w.k_norm = up.f32(L(l, "attn_k_norm.weight"), HD);
            const Tensor& iq = L(l, "indexer.q_proj.weight");
            const Tensor& ik = L(l, "indexer.k_proj.weight");
            if (iq.ne[1] != QSA_HEADS * QSA_KEY || iq.ne[0] != N || ik.ne[1] != QSA_KEY || ik.ne[0] != N)
                throw std::runtime_error("fnx: " + iq.name + " is " + std::to_string(iq.ne[0]) + " by " + std::to_string(iq.ne[1]) +
                                         " and " + ik.name + " is " + std::to_string(ik.ne[0]) + " by " + std::to_string(ik.ne[1]) +
                                         ", not the " + std::to_string(N) + " by " + std::to_string(QSA_HEADS * QSA_KEY) + " and " +
                                         std::to_string(N) + " by " + std::to_string(QSA_KEY) + " k_idx_proj reads");
            w.idx_wq = up.bf16(iq);
            w.idx_wk = up.bf16(ik);
            w.idx_qn = up.f32(L(l, "indexer.q_norm.weight"), QSA_KEY);
            w.idx_kn = up.f32(L(l, "indexer.k_norm.weight"), QSA_KEY);
        }
        const char* pr = perm.empty() ? nullptr : perm[(size_t) l].data();
        if (g_lat_labels) {   // edge0_patch.py: Edge0's latent experts, D and U, labels and tail scales
            if (c.ff0 % 64 || c.ff1 % 64) throw std::runtime_error("FNX_LATENT: a card's ff range is not in NVFP4's 64-value blocks");
            const Tensor& lb = L(l, "lat_exps");
            w.gu_e = up.lat_rows2(lb, c.ff0, c.ff1, w.gu_e_stride, hg, pr, pr ? c.perm_gu.data() + (size_t) l * NEXP : nullptr);
            w.down_e = up.lat_cols(lb, c.ff0 / 64, c.ff1 / 64, w.down_e_stride, hd, pr, pr ? c.perm_dn.data() + (size_t) l * NEXP : nullptr);
            std::vector<float> esc((size_t) NEXP * 4);
            for (int e = 0; e < NEXP; ++e) {
                std::memcpy(&esc[(size_t) e * 4], lb.data + (size_t) e * LAT_BLOB + LAT_TAIL, 16);
                for (int k = 0; k < 3; ++k) esc[(size_t) e * 4 + k] *= NVFP4_PRE;
            }
            w.lat_esc = up.put<float>(esc.data(), esc.size() * 4);
            w.lat_lab = up.put<int>(g_lat_labels + (size_t) l * NEXP, (size_t) NEXP * 4);
            // edge0_patch6.py, FNX_LAT_SPLIT: this card's half of the groups only (h: its place by its ff range, as in Engine::ffn_half)
            const int h = g_lat_split ? c.ff0 / (c.ff1 - c.ff0) : 0, ng = g_lat_split ? LAT_GH : LAT_G, g0 = h * ng;
            Tensor td = L(l, "lat_d");
            if (g_lat_split) {   // D's rows [h LAT_HW, (h + 1) LAT_HW)
                if (td.ne[0] != N || td.ne[1] != LAT_W) throw std::runtime_error("FNX_LAT_SPLIT: lat_d is not [LAT_W][N]");
                td.data += (size_t) h * LAT_HW * td.row_bytes();
                td.ne[1] = LAT_HW;
                td.nbytes = (size_t) LAT_HW * td.row_bytes();
            }
            w.lat_d = up.q8split(td);
            const Tensor& tu = L(l, "lat_u");   // [G][N][LAT_ELL] -> [N][G * LAT_ELL]: z (all 8 groups a row, or the card's 4) in one product
            const size_t rb = (size_t) LAT_ELL / 32 * 34;
            std::vector<uint8_t> uf((size_t) N * ng * rb);
            for (int i = 0; i < N; ++i)
                for (int gg = 0; gg < ng; ++gg) std::memcpy(&uf[((size_t) i * ng + gg) * rb], tu.data + ((size_t) (g0 + gg) * N + i) * rb, rb);
            Tensor tf = tu;
            tf.data = uf.data();
            tf.ne[0] = (int64_t) ng * LAT_ELL;
            tf.ne[1] = N;
            tf.ne[2] = tf.ne[3] = 1;
            tf.nbytes = uf.size();
            w.lat_u = up.q8split(tf);
        } else {
        w.gu_e = up.expert_rows2(L(l, "ffn_gate_exps.weight"), L(l, "ffn_up_exps.weight"), c.ff0, c.ff1, w.gu_e_stride, hg, pr,
                                 pr ? c.perm_gu.data() + (size_t) l * NEXP : nullptr);
        w.down_e = up.expert_cols(L(l, "ffn_down_exps.weight"), c.ff0 / 32, c.ff1 / 32, w.down_e_stride, hd, pr,
                                  pr ? c.perm_dn.data() + (size_t) l * NEXP : nullptr);
        }
        w.sh_gu = up.dense(up.rows2(L(l, "ffn_gate_shexp.weight"), range(c.ff0, c.ff1), L(l, "ffn_up_shexp.weight"), range(c.ff0, c.ff1)));
        w.sh_down = up.dense(up.cols(L(l, "ffn_down_shexp.weight"), range(c.ff0 / 32, c.ff1 / 32)));
        if (!w.sh_gu.s.q || !w.sh_down.s.q) throw std::runtime_error("fnx: the shared expert is not Q8_0");
        if (l == PLE_LAYER) {
            const Tensor& pk = L(l, "ple_key.weight");
            const Tensor& pv = L(l, "ple_value.weight");
            // dense() reads N-wide xq rows and writes D-float rows into b->ple_key, N-float into b->ple_val
            if (pk.ne[0] != N || pk.ne[1] != D)
                throw std::runtime_error("fnx: " + pk.name + " is " + std::to_string(pk.ne[0]) + " by " + std::to_string(pk.ne[1]) +
                                         ", not the " + std::to_string(N) + " by " + std::to_string(D) + " its dense into the ple_key rows reads");
            if (pv.ne[0] != N || pv.ne[1] != N)
                throw std::runtime_error("fnx: " + pv.name + " is " + std::to_string(pv.ne[0]) + " by " + std::to_string(pv.ne[1]) +
                                         ", not the " + std::to_string(N) + " by " + std::to_string(N) + " its dense into the ple_val rows reads");
            c.ple_key = up.dense(up.rows(pk, range(0, (int) pk.ne[1])));
            c.ple_value = up.dense(up.rows(pv, range(0, (int) pv.ne[1])));
            c.ple_norm_key = up.f32(L(l, "ple_norm_key.weight"), D);
            c.ple_norm_query = up.f32(L(l, "ple_norm_query.weight"), D);
            c.ple_norm_conv = up.f32(L(l, "ple_norm_conv.weight"), D);
            const Tensor& pc = L(l, "ple_conv1d.weight");   // k_ple_conv reads w[ch * 4 + k]: 4 a row, D of them
            if (pc.ne[0] != 4 || pc.ne[1] != D)
                throw std::runtime_error("fnx: " + pc.name + " is " + std::to_string(pc.ne[0]) + " by " + std::to_string(pc.ne[1]) +
                                         ", not the 4 by " + std::to_string(D) + " k_ple_conv reads");
            c.ple_conv = up.f32(pc, 4 * D);
        }
        st.done(l);
    }
    c.head_norm = up.f32(st.t(NL, "output_hc_norm.weight"), D);
    c.head_down = up.q8split(st.t(NL, "output_hc_down.weight"));
    c.head_up = up.q8split(st.t(NL, "output_hc_up.weight"));
    c.out = up.dense(up.rows(st.t(NL, "output.weight"), range(c.v0, c.v1)));
    st.done(NL);
    if (!resident.empty()) {   // tracked, so the engine can place them again with more in RAM (Engine: VRAM short)
        const size_t before = c.bytes;
        Up ue{c, store, &c.expert_mem};
        ue.expert_cache(resident, hgu, hdn, arena_score, perm);
        c.expert_bytes = c.bytes - before;
    } else if (host_experts) c.ec_host_gu = hgu, c.ec_host_dn = hdn;   // the holder: placed by a worker
}

// ---------------------------------------------------------------- the draft layer
// Strata's runtime files: dense.txt lines "name kind rows cols offset bytes" into dense.bin (kind q8_0 | bf16 | f32)
struct MtpFiles {
    std::map<std::string, Tensor> t;
    void* base = nullptr;
    size_t size = 0;
    std::vector<uint8_t> gate_b, up_b, down_b;   // the experts' blocks: [512][640][2560], [512][640][2560], [512][2560][640]
    uint32_t etype = T_Q4_0;
    Tensor gate, up, down;
    explicit MtpFiles(const std::string& dir) {
        const int fd = ::open((dir + "/dense.bin").c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("mtp: cannot open " + dir + "/dense.bin");
        size = (size_t) lseek(fd, 0, SEEK_END);
        base = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) throw std::runtime_error("mtp: cannot map dense.bin");
        std::ifstream idx(dir + "/dense.txt");
        if (!idx) throw std::runtime_error("mtp: cannot open " + dir + "/dense.txt");
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            std::string name, kind;
            int64_t rows, cols;
            uint64_t off, bytes;
            if (!(is >> name >> kind >> rows >> cols >> off >> bytes)) throw std::runtime_error("mtp: dense.txt: " + line);
            Tensor x;
            x.name = name;
            x.type = kind == "q8_0" ? T_Q8_0 : kind == "bf16" ? T_BF16 : kind == "f32" ? T_F32 : 0xffffffffu;
            if (x.type == 0xffffffffu) throw std::runtime_error("mtp: dense.txt: kind " + kind);
            x.ne[0] = cols;
            x.ne[1] = rows;
            x.n_dims = 2;
            x.data = (const uint8_t*) base + off;
            x.nbytes = bytes;
            if (off + bytes > size || x.row_bytes() * rows != bytes) throw std::runtime_error("mtp: dense.txt: " + name + " does not fit");
            t[name] = x;
        }
        // FNX_MTP_EXPERTS: q2_0 is Strata's experts.bin (unpacked to Q4_0 blocks), q4_0 is tools/mtp_experts.py's
        // experts-q4_0.bin, made from the BF16 checkpoint; the same VRAM either way. Q8_0 drafted exactly as Q4_0 did
        // for ~420 MiB more a card, which the arena's reserve does not count: not offered.
        const char* ex_s = std::getenv("FNX_MTP_EXPERTS");
        const std::string ex = ex_s ? ex_s : "q2_0";   // strata-split: Strata's draft-layer files (mtp/rt), no download; FNX_MTP_EXPERTS=q4_0 for sFNX's
        if (ex == "q2_0") experts(dir + "/experts.bin");
        else if (ex == "q4_0") experts_blocks(dir + "/experts-q4_0.bin", T_Q4_0, 18);
        else throw std::runtime_error("mtp: FNX_MTP_EXPERTS=" + ex + ": q2_0 or q4_0");
        std::fprintf(stderr, "fnx: draft layer experts %s\n", ex.c_str());
        views();
    }
    ~MtpFiles() { if (base) munmap(base, size); }
    const Tensor& operator[](const std::string& n) const {
        const auto it = t.find(n);
        if (it == t.end()) throw std::runtime_error("mtp: " + n + " is missing");
        return it->second;
    }
    // experts.bin: per expert codes (gate/up rows interleaved, 640 B each; then 2560 down rows of 160 B), then the
    // fp16 scales (gate/up rows, 40 each; down rows, 10 each). Q2_0: value j of a 64-value block is code j (byte j / 4,
    // bits 2 (j % 4)) minus one, times the scale. Q4_0 keeps 32 values per block: nibble = code + 7 (value i low,
    // i + 16 high), the same scale for both halves of a Q2_0 block.
    void experts(const std::string& path) {
        constexpr int H = N, F = FF, NE = NEXP;
        constexpr size_t ROW_GU = H / 4, ROW_D = F / 4, SC_GU = H / 64, SC_D = F / 64;
        constexpr size_t O_D = 2ull * F * ROW_GU, O_GS = O_D + (size_t) H * ROW_D, O_DS = O_GS + 2ull * F * SC_GU * 2;
        constexpr size_t BLOB = O_DS + (size_t) H * SC_D * 2;
#if !FNX_27B
        static_assert(BLOB == 1382400, "Strata's Q2_0 expert blob");
#endif
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("mtp: cannot open " + path);
        const size_t qrow_gu = (size_t) H / 32 * 18, qrow_d = (size_t) F / 32 * 18;
        gate_b.resize((size_t) NE * F * qrow_gu);
        up_b.resize((size_t) NE * F * qrow_gu);
        down_b.resize((size_t) NE * H * qrow_d);
        etype = T_Q4_0;
        std::vector<uint8_t> blob(BLOB);
        // one row of n values: codes (n / 4 bytes) and n / 64 scales -> n / 32 Q4_0 blocks
        auto row = [](const uint8_t* codes, const uint8_t* scales, int n, uint8_t* out) {
            for (int b = 0; b < n / 64; ++b) {
                uint16_t d;
                std::memcpy(&d, scales + b * 2, 2);
                const uint8_t* c = codes + b * 16;
                auto code = [&](int j) { return (c[j / 4] >> (2 * (j % 4))) & 3; };
                for (int h = 0; h < 2; ++h) {
                    uint8_t* o = out + (size_t) (2 * b + h) * 18;
                    std::memcpy(o, &d, 2);
                    for (int i = 0; i < 16; ++i)
                        o[2 + i] = (uint8_t) ((code(32 * h + i) + 7) | ((code(32 * h + 16 + i) + 7) << 4));
                }
            }
        };
        for (int e = 0; e < NE; ++e) {
            if (std::fread(blob.data(), 1, BLOB, f) != BLOB) {
                std::fclose(f);
                throw std::runtime_error("mtp: experts.bin is truncated");
            }
            for (int r = 0; r < F; ++r) {
                row(&blob[(2 * r) * ROW_GU], &blob[O_GS + (2 * r) * SC_GU * 2], H, &gate_b[((size_t) e * F + r) * qrow_gu]);
                row(&blob[(2 * r + 1) * ROW_GU], &blob[O_GS + (2 * r + 1) * SC_GU * 2], H, &up_b[((size_t) e * F + r) * qrow_gu]);
            }
            for (int r = 0; r < H; ++r)
                row(&blob[O_D + r * ROW_D], &blob[O_DS + r * SC_D * 2], F, &down_b[((size_t) e * H + r) * qrow_d]);
        }
        std::fclose(f);
    }
    // experts-q4_0.bin / experts-q8_0.bin: gate, up, down, each [512][rows] rows of 32-value blocks of `bb` bytes
    void experts_blocks(const std::string& path, uint32_t type, size_t bb) {
        const size_t qrow_gu = (size_t) N / 32 * bb, qrow_d = (size_t) FF / 32 * bb;
        gate_b.resize((size_t) NEXP * FF * qrow_gu);
        up_b.resize((size_t) NEXP * FF * qrow_gu);
        down_b.resize((size_t) NEXP * N * qrow_d);
        etype = type;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("mtp: cannot open " + path + " (tools/mtp_experts.py makes it)");
        const bool ok = std::fread(gate_b.data(), 1, gate_b.size(), f) == gate_b.size() &&
                        std::fread(up_b.data(), 1, up_b.size(), f) == up_b.size() &&
                        std::fread(down_b.data(), 1, down_b.size(), f) == down_b.size() && std::fgetc(f) == EOF;
        std::fclose(f);
        if (!ok) throw std::runtime_error("mtp: " + path + " is not " + std::to_string(gate_b.size() * 2 + down_b.size()) + " bytes");
    }
    void views() {
        constexpr int H = N, F = FF;
        auto view = [this](const char* name, std::vector<uint8_t>& v, int64_t cols, int64_t rows) {
            Tensor x;
            x.name = name;
            x.type = etype;
            x.ne[0] = cols;
            x.ne[1] = rows;
            x.ne[2] = NEXP;
            x.n_dims = 3;
            x.data = v.data();
            x.nbytes = v.size();
            return x;
        };
        gate = view("mtp gate", gate_b, H, F);
        up = view("mtp up", up_b, H, F);
        down = view("mtp down", down_b, F, H);
    }
};

void free_mtp_cards(std::vector<MtpCard>& cards) {
    for (MtpCard& mc : cards) {
        if (mc.allocs.empty()) continue;
        if (cudaSetDevice(mc.dev) != cudaSuccess) std::fprintf(stderr, "mtp: cudaSetDevice(%d) failed while freeing\n", mc.dev);
        for (size_t i = 0; i < mc.allocs.size(); ++i) {
            const cudaError_t e = cudaFree(mc.allocs[i]);
            if (e == cudaSuccess) continue;
            cudaGetLastError();
            cudaPointerAttributes at{};
            const cudaError_t ea = cudaPointerGetAttributes(&at, mc.allocs[i]);
            cudaGetLastError();
            std::fprintf(stderr, "mtp: card %d: cudaFree of allocation %zu of %zu (%p) failed: %s (%s); the pointer: %s, type %d, device %d\n",
                         mc.dev, i, mc.allocs.size(), mc.allocs[i], cudaGetErrorName(e), cudaGetErrorString(e),
                         ea == cudaSuccess ? "known" : cudaGetErrorName(ea), (int) at.type, at.device);
        }
        mc.allocs.clear();
    }
}

void load_mtp_card(const Model& m, const MtpFiles& f, const CardW& c, const std::vector<int>& vocab, MtpCard& mc) {
    ck(cudaSetDevice(c.dev), "cudaSetDevice");
    CardW sink;   // Up counts bytes into a CardW
    sink.dev = c.dev;
    mc.dev = c.dev;
    Up up{sink, nullptr, &mc.allocs};
    auto bf16_of = [&](const Tensor& t) {
        if (t.type != T_BF16) throw std::runtime_error("mtp: " + t.name + " is not bf16");
        return (const uint16_t*) t.data;
    };
    auto q8 = [&](const char* name) { return up.q8split_bf16(bf16_of(f[name]), (size_t) f[name].ne[0] * f[name].ne[1]); };
    std::vector<int> qrows, kvrows, woblocks;
    for (int h = c.q0; h < c.q0 + c.nq; ++h) {
        for (int i = 0; i < 2 * HD; ++i) qrows.push_back(h * 2 * HD + i);
        for (int b = 0; b < HD / 32; ++b) woblocks.push_back(h * HD / 32 + b);
    }
    for (int h = c.kv0; h < c.kv0 + c.nkv; ++h)
        for (int i = 0; i < HD; ++i) kvrows.push_back(h * HD + i);
    // the K/V cache is split by position: every card projects q for every head and k/v for both KV heads
    std::vector<int> qrows_all, kvrows_all;
    for (int i = 0; i < NQH * 2 * HD; ++i) qrows_all.push_back(i);
    for (int i = 0; i < NKVH * HD; ++i) kvrows_all.push_back(i);
    LayerW& w = mc.L;
    const char* hp[2] = {"attn_hyper_connection.", "mlp_hyper_connection."};
    for (int h = 0; h < 2; ++h) {
        const std::string p = hp[h];
        w.hc_norm[h] = up.f32(f[p + "hc_norm.weight"], D);
        w.hc_down[h] = q8((p + "input_mix_weight_down.weight").c_str());
        w.hc_up[h] = q8((p + "input_mix_weight_up.weight").c_str());
        const Tensor& inj = f[p + "block_inject_weight.weight"];
        w.hc_inj[h] = up.put<uint16_t>(bf16_of(inj), inj.nbytes);
    }
    {   // the router's rows, then the shared expert's gate, as f32
        const Tensor &r = f["mlp.gate.weight"], &sg = f["mlp.shared_expert_gate.weight"];
        if (r.ne[1] != NEXP || r.ne[0] != N)
            throw std::runtime_error("mtp: mlp.gate.weight is " + std::to_string(r.ne[0]) + " by " + std::to_string(r.ne[1]) +
                                     ", not the " + std::to_string(N) + " by " + std::to_string(NEXP) + " the router reads");
        if (sg.ne[0] != N || sg.ne[1] != 1)
            throw std::runtime_error("mtp: mlp.shared_expert_gate.weight is " + std::to_string(sg.ne[0]) + " by " +
                                     std::to_string(sg.ne[1]) + ", not the " + std::to_string(N) + " by 1 it reads");
        std::vector<uint16_t> h((size_t) (NEXP + 1) * N);
        std::memcpy(h.data(), bf16_of(r), (size_t) NEXP * N * 2);
        std::memcpy(h.data() + (size_t) NEXP * N, bf16_of(sg), (size_t) N * 2);
        w.router = up.put<uint16_t>(h.data(), h.size() * 2);
    }
    w.wq = up.dense(up.rows(f["self_attn.q_proj.weight"], qrows_all));
    w.wk = up.dense(up.rows(f["self_attn.k_proj.weight"], kvrows_all));
    w.wv = up.dense(up.rows(f["self_attn.v_proj.weight"], kvrows_all));
    w.wo = up.dense(up.cols(f["self_attn.o_proj.weight"], woblocks));
    w.q_norm = up.f32(f["self_attn.q_norm.weight"], HD);
    w.k_norm = up.f32(f["self_attn.k_norm.weight"], HD);
    const Tensor& iqk = f["self_attn.indexer.index_qk_proj.weight"];
    if (iqk.ne[1] != (QSA_HEADS + 1) * QSA_KEY || iqk.ne[0] != N)
        throw std::runtime_error("mtp: " + iqk.name + " is " + std::to_string(iqk.ne[0]) + " by " + std::to_string(iqk.ne[1]) + ", not the " +
                                 std::to_string(N) + " by " + std::to_string((QSA_HEADS + 1) * QSA_KEY) + " (queries then key) it reads");
    w.idx_qk = up.dense(up.rows(iqk, range(0, (QSA_HEADS + 1) * QSA_KEY)));
    w.idx_qn = up.f32(f["self_attn.indexer.q_layernorm.weight"], QSA_KEY);
    w.idx_kn = up.f32(f["self_attn.indexer.k_layernorm.weight"], QSA_KEY);
    w.gu_e = up.expert_rows2(f.gate, f.up, c.ff0, c.ff1, w.gu_e_stride);
    w.down_e = up.expert_cols(f.down, c.ff0 / 32, c.ff1 / 32, w.down_e_stride);
    w.sh_gu = up.dense(up.rows2(f["mlp.shared_expert.gate_proj.weight"], range(c.ff0, c.ff1), f["mlp.shared_expert.up_proj.weight"],
                                range(c.ff0, c.ff1)));
    w.sh_down = up.dense(up.cols(f["mlp.shared_expert.down_proj.weight"], range(c.ff0 / 32, c.ff1 / 32)));
    mc.fc_emb = up.dense(up.rows(f["fc_embedding.weight"], range(0, N)));
    mc.fc_hid = up.dense(up.rows(f["fc_hidden.weight"], range(0, N)));
    mc.norm_emb = up.f32(f["pre_fc_norm_embedding.weight"], N);
    mc.norm_hid = up.f32(f["pre_fc_norm_hidden.weight"], D);
    mc.mix_norm = up.f32(f["hyper_connection_mixer.hc_norm.weight"], D);
    mc.mix_down = q8("hyper_connection_mixer.input_mix_weight_down.weight");
    mc.mix_up = q8("hyper_connection_mixer.input_mix_weight_up.weight");
    std::vector<int> rows(vocab.begin() + mc.dv0, vocab.begin() + mc.dv1);
    mc.head = up.dense(up.rows(*m.g.find("output.weight"), rows));
    for (const DW* d : {&mc.head, &w.wq, &w.wk, &w.wv, &w.wo, &w.idx_qk, &w.sh_gu, &w.sh_down, &mc.fc_emb, &mc.fc_hid})
        if (!d->s.q) throw std::runtime_error("mtp: a projection is not Q8_0");
}

}  // namespace

#if FNX_27B
// the 27B's draft head: this card's rows of output.weight for its share of the draft vocabulary (the layer itself came
// with the model). dir names where the layer comes from, so a Flash-Next runtime directory in a 27B config fails here
Mtp::Mtp(const Model& m, const std::string& dir, const std::string& vocab_file) {
    if (dir != "nextn") throw std::runtime_error("mtp: the 27B's draft layer is the GGUF's own: its directory is \"nextn\", not \"" + dir + "\"");
    if (vocab_file.empty()) throw std::runtime_error("mtp: the 27B needs a draft vocabulary file");
    {
        std::FILE* f = std::fopen(vocab_file.c_str(), "rb");
        if (!f) throw std::runtime_error("mtp: cannot open " + vocab_file);
        struct stat stb{};
        if (fstat(fileno(f), &stb) != 0 || stb.st_size % 4 != 0) {
            std::fclose(f);
            throw std::runtime_error("mtp: " + vocab_file + " is not whole int32 token ids (" + std::to_string((long long) stb.st_size) + " bytes)");
        }
        int32_t id;
        while (std::fread(&id, 4, 1, f) == 1) vocab.push_back(id);
        const bool io_bad = std::ferror(f) != 0;
        std::fclose(f);
        if (io_bad || vocab.size() * 4 != (size_t) stb.st_size)
            throw std::runtime_error("mtp: " + vocab_file + " stopped at " + std::to_string(vocab.size() * 4) +
                                     " of its " + std::to_string((long long) stb.st_size) + " bytes");
        if (vocab.empty()) throw std::runtime_error("mtp: " + vocab_file + " is empty");
        for (int id2 : vocab)
            if (id2 < 0 || id2 >= m.vocab) throw std::runtime_error("mtp: " + vocab_file + " holds a token outside the vocabulary");
    }
    cards.resize(NCARD);
    const int step = ((int) vocab.size() + NCARD - 1) / NCARD;
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        cards[c].dv0 = std::min((int) vocab.size(), c * step);
        cards[c].dv1 = std::min((int) vocab.size(), (c + 1) * step);
        th.emplace_back([&, c] {
            try {
                MtpCard& mc = cards[c];
                ck(cudaSetDevice(m.cards[c].dev), "cudaSetDevice");
                CardW sink;   // Up counts bytes into a CardW
                sink.dev = mc.dev = m.cards[c].dev;
                Up up{sink, nullptr, &mc.allocs};
                mc.head = up.dense(up.rows(*m.g.find("output.weight"), std::vector<int>(vocab.begin() + mc.dv0, vocab.begin() + mc.dv1)));
                if (!mc.head.s.q) throw std::runtime_error("mtp: output.weight is not Q8_0");
            } catch (const std::exception& e) { errs[c] = e.what(); }
        });
    }
    for (auto& t : th) t.join();
    for (int c = 0; c < NCARD; ++c)
        if (!errs[c].empty()) {
            free_mtp_cards(cards);
            throw std::runtime_error("mtp card " + std::to_string(c) + ": " + errs[c]);
        }
}
#else
Mtp::Mtp(const Model& m, const std::string& dir, const std::string& vocab_file) {
    {
        const std::string vf = vocab_file.empty() ? dir + "/draft_vocab.bin" : vocab_file;
        std::FILE* f = std::fopen(vf.c_str(), "rb");
        if (!f) throw std::runtime_error("mtp: cannot open " + vf);
        struct stat stb{};
        if (fstat(fileno(f), &stb) != 0 || stb.st_size % 4 != 0) {
            std::fclose(f);
            throw std::runtime_error("mtp: " + vf + " is not whole int32 token ids (" + std::to_string((long long) stb.st_size) + " bytes)");
        }
        int32_t id;
        while (std::fread(&id, 4, 1, f) == 1) vocab.push_back(id);
        const bool io_bad = std::ferror(f) != 0;
        std::fclose(f);
        if (io_bad || vocab.size() * 4 != (size_t) stb.st_size)
            throw std::runtime_error("mtp: " + vf + " stopped at " + std::to_string(vocab.size() * 4) +
                                     " of its " + std::to_string((long long) stb.st_size) + " bytes");
        if (vocab.empty()) throw std::runtime_error("mtp: " + vf + " is empty");
        for (int id2 : vocab)
            if (id2 < 0 || id2 >= m.vocab) throw std::runtime_error("mtp: " + vf + " holds a token outside the vocabulary");
    }
    const MtpFiles files(dir);
    cards.resize(NCARD);
    const int step = ((int) vocab.size() + NCARD - 1) / NCARD;
    for (int c = 0; c < NCARD; ++c) {
        cards[c].dv0 = std::min((int) vocab.size(), c * step);
        cards[c].dv1 = std::min((int) vocab.size(), (c + 1) * step);
    }
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c)
        th.emplace_back([&, c] {
            try { load_mtp_card(m, files, m.cards[c], vocab, cards[c]); } catch (const std::exception& e) { errs[c] = e.what(); }
        });
    for (auto& t : th) t.join();
    for (int c = 0; c < NCARD; ++c)
        if (!errs[c].empty()) {
            free_mtp_cards(cards);   // no destructor runs for a constructor that throws
            throw std::runtime_error("mtp card " + std::to_string(c) + ": " + errs[c]);
        }
}
#endif

Mtp::~Mtp() { free_mtp_cards(cards); }

// A file whose routed experts the engine has no kernel for is refused here, in one plain line, before anything is read into RAM or placed on
// the cards (the app's rule, strata_control/split_engine.py, refuses it earlier; this is the same check for a start that skipped the rule). The
// server then starts the normal layer split. Q2_0 is not listed: the Gguf has already repacked it to Q4_0.
static void check_supported(const Gguf& g) {
    const uint32_t ok[] = {T_Q4_K, T_Q5_K, T_Q5_1, T_Q8_0, T_Q4_0, T_IQ4_NL, T_IQ4_XS, T_IQ3_S, T_IQ3_XXS, T_IQ2_S, T_IQ2_XXS, T_IQ1_M,
                           T_NVFP4};   // NVFP4: Edge0's latent experts (edge0_patch.py)
    for (const auto& kv : g.tensors()) {
        const std::string& n = kv.first;
        if (n.find("_exps.weight") == std::string::npos || n.find("shexp") != std::string::npos) continue;
        if (std::find(std::begin(ok), std::end(ok), kv.second.type) == std::end(ok))
            throw std::runtime_error("the all-cards split engine does not read this model's routed experts (" + n + " is ggml type " +
                                     std::to_string(kv.second.type) + "); use the normal layer split");
    }
}

Model::Model(const std::string& path, Load how, Store* store, const std::string* image) : g(path) {
    // the cards first: reading the embeddings into RAM below takes seconds on a cold cache
    {
        int ndev = 0;
        ck(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
        // a build for another count than the cards it is given is a config or build mistake (a 2-card binary on
        // prod's three would serve, slower): CUDA_VISIBLE_DEVICES (the config's "gpu") must name exactly NCARD
        if (ndev != NCARD)
            throw std::runtime_error("fnx: built for " + std::to_string(NCARD) + " cards (FNX_NCARD), " + std::to_string(ndev) +
                                     " visible: set CUDA_VISIBLE_DEVICES (a config's \"gpu\") to " + std::to_string(NCARD) +
                                     " of them, or build for " + std::to_string(ndev) + " (make FNX_NCARD=" + std::to_string(ndev) + ")");
        check_cards();
    }
    check_supported(g);
    if (g.latent()) {   // edge0_patch.py
        const Tensor& lt = g.t("lat_labels");
        for (size_t i = 0; i < (size_t) NL * NEXP; ++i)
            if (((const int32_t*) lt.data)[i] < 0 || ((const int32_t*) lt.data)[i] >= LAT_G) throw std::runtime_error("FNX_LATENT: a group label out of 0..7");
        g_lat_labels = (const int32_t*) lt.data;
        if (const char* s = std::getenv("FNX_LAT_SPLIT")) g_lat_split = std::atoi(s) != 0;   // edge0_patch6.py
        if (g_lat_split && NCARD != 2) throw std::runtime_error("FNX_LAT_SPLIT: two cards");
        if (g_lat_split) std::fprintf(stderr, "fnx: FNX_LAT_SPLIT: D and U by group, four groups a card (two exchanges a layer more)\n");
    }
#if FNX_27B
    const std::string p = "qwen35.";
    // block_count holds the draft (nextn) layer too
    if (g.i(p + "block_count") != NL + g.i(p + "nextn_predict_layers") || g.i(p + "nextn_predict_layers") != 1 ||
        g.i(p + "embedding_length") != N || g.i(p + "feed_forward_length") != FFD || g.i(p + "ssm.group_count") != NKH ||
        g.i(p + "ssm.time_step_rank") != NVH || g.i(p + "attention.head_count") != NQH ||
        g.i(p + "attention.head_count_kv") != NKVH || g.i(p + "full_attention_interval") != 4 ||
        g.i(p + "attention.key_length") != HD || g.i(p + "attention.value_length") != HD || g.i(p + "ssm.state_size") != HS ||
        g.i(p + "ssm.inner_size") != NVH * HS || g.i(p + "ssm.conv_kernel") != 4 || g.i(p + "rope.dimension_count") != NROT)
        throw std::runtime_error("27b: this GGUF's geometry is not Qwen3.8-27B's");
#else
    const std::string p = "qwen4exp.";
    if (g.i(p + "block_count") != NL || g.i(p + "embedding_length") != N || g.i(p + "expert_count") != NEXP ||
        g.i(p + "ssm.group_count") != NKH || g.i(p + "ssm.time_step_rank") != NVH || g.i(p + "attention.head_count") != NQH ||
        g.i(p + "attention.head_count_kv") != NKVH || g.i(p + "hyper_connection.count") != HC ||
        g.i(p + "hyper_connection.low_rank") != LR || g.i(p + "expert_feed_forward_length") != FF ||
        g.i(p + "attention.indexer.head_count") != QSA_HEADS || g.i(p + "attention.indexer.key_length") != QSA_KEY ||
        g.i(p + "attention.indexer.top_k") != QSA_TOP_K || g.i(p + "expert_used_count") != TOPK ||
        g.i(p + "rope.dimension_count") != NROT)
        throw std::runtime_error("fnx: this GGUF's geometry is not Flash-Next's");
    // the shared expert is split by the routed experts' ff ranges (CardW ff0, ff1)
    for (int l = 0; l < NL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        if (g.t(b + "ffn_gate_shexp.weight").ne[1] != FF || g.t(b + "ffn_up_shexp.weight").ne[1] != FF || g.t(b + "ffn_down_shexp.weight").ne[0] != FF)
            throw std::runtime_error("fnx: layer " + std::to_string(l) + "'s shared expert is not " + std::to_string(FF) + " wide");
    }
#endif
    eps = (float) g.f(p + "attention.layer_norm_rms_epsilon");
    rope_base = (float) g.f(p + "rope.freq_base");
    // the trained length bounds the context (Engine refuses past it, or past it x FNX_ROPE_SCALE), so a model must name it
    {
        const int64_t n = g.i(p + "context_length");
        if (n <= 0 || n > INT32_MAX) throw std::runtime_error("fnx: " + p + "context_length " + std::to_string(n) + " is not a context length");
        ctx_train = (int) n;
    }
    // the model's own rotary scaling, which Engine runs by default (choose_rope); what sFNX cannot follow throws here
    {
        const ModelRope mr = model_rope(g, p);
        yarn_factor = mr.yarn_factor;
        yarn_orig = mr.yarn_orig;
    }
    tok_embd = &g.t("token_embd.weight");
#if FNX_27B
    if (tok_embd->type != T_Q8_0) throw std::runtime_error("27b: token_embd.weight is not Q8_0 (the host's embedding reads Q8_0)");
    vocab = (int) tok_embd->ne[1];
    if (g.t("output.weight").ne[1] != vocab || g.t("output.weight").ne[0] != N)
        throw std::runtime_error("27b: output.weight is not [" + std::to_string(N) + "][" + std::to_string(vocab) + "] (token_embd's vocabulary)");
    {   // the host gathers its rows every window: a row read from the disk costs milliseconds, so keep it in RAM
        const auto t0 = std::chrono::steady_clock::now();
        lock_in_ram(g, *tok_embd);
        std::fprintf(stderr, "27b: %s (%.1f GB) locked in RAM in %.1f s\n", tok_embd->name.c_str(), tok_embd->nbytes / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
#else
    ple_table = &g.t("per_layer_token_embd.weight");
    if (tok_embd->type != T_Q8_0) throw std::runtime_error("fnx: token_embd.weight is not Q8_0 (the host's embedding reads Q8_0)");
    if (ple_table->type != T_IQ4_NL && ple_table->type != T_Q8_0)
        throw std::runtime_error("fnx: per_layer_token_embd.weight is neither IQ4_NL nor Q8_0 (the host's PLE rows read those)");
    vocab = (int) tok_embd->ne[1];
    // the host gathers rows of these two every window: a row read from the disk costs milliseconds, so they go to RAM, last (model_loaded)
    for (int64_t v : g.ia(p + "ple.layer_multipliers")) ple_mult.push_back((uint64_t) v);
    for (int64_t v : g.ia(p + "ple.head_offsets")) ple_off.push_back((uint64_t) v);
    for (int64_t v : g.ia(p + "ple.head_vocab_sizes")) ple_vocab.push_back((uint64_t) v);
    // the gather reads ple_mult[0..PLE_ORDERS) and a head h at mixed % ple_vocab[h] + ple_off[h]: the arrays must be that
    // long and every head's range must stay inside the table, or the gather reads rows (or zero-modulo) the metadata invented
    if (ple_mult.size() != PLE_ORDERS || ple_off.size() != PLE_HEADS || ple_vocab.size() != PLE_HEADS)
        throw std::runtime_error("fnx: the PLE metadata holds " + std::to_string(ple_mult.size()) + " multipliers and " +
                                 std::to_string(ple_off.size()) + " offsets and " + std::to_string(ple_vocab.size()) +
                                 " vocab sizes, want " + std::to_string(PLE_ORDERS) + " and " + std::to_string(PLE_HEADS) + " and " +
                                 std::to_string(PLE_HEADS));
    {
        // a zero multiplier drops its n-gram from the hash silently: the gather still runs, on fewer orders than the model trained with
        for (int j = 0; j < PLE_ORDERS; ++j)
            if (!ple_mult[j]) throw std::runtime_error("fnx: PLE multiplier " + std::to_string(j) + " is 0, which drops that n-gram from the hash");
        static_assert(PLE_HEADS * PLE_DIM == N, "the gather tiles PLE_HEADS rows of PLE_DIM into one embedding row");
        if (ple_table->ne[0] != PLE_DIM)
            throw std::runtime_error("fnx: " + ple_table->name + " is " + std::to_string(ple_table->ne[0]) + " a row, not the " +
                                     std::to_string(PLE_DIM) + " the gather tiles into the embedding");
        const uint64_t rows = (uint64_t) ple_table->ne[1];   // the gather reads rows of per_layer_token_embd, not the layer's key table
        for (int h = 0; h < PLE_HEADS; ++h) {
            // head h gathers rows [off, off+vocab): bound each side on its own so a negative offset (which wraps huge)
            // trips the check instead of slipping past a summed top
            if (!ple_vocab[h]) throw std::runtime_error("fnx: PLE head " + std::to_string(h) + " has an empty vocabulary");
            if (ple_off[h] >= rows || ple_vocab[h] > rows - ple_off[h])
                throw std::runtime_error("fnx: PLE head " + std::to_string(h) + " gathers from row " + std::to_string((int64_t) ple_off[h]) +
                                         " for " + std::to_string(ple_vocab[h]) + ", past " + ple_table->name + "'s " + std::to_string(rows));
        }
    }
    ple_eos = g.i(p + "ple.eos_token_id");
    ple_image = g.i(p + "ple.image_token_id");
#endif
    cards.resize(NCARD);
    // key heads per card (their q and k), value heads (numbered s * NKH + k, listed ascending: the order the card's
    // slices are laid out in) and FF slices, by card_split.hpp's rule
    int kparts[NCARD][2], ffparts[NCARD][2];
    std::vector<std::vector<int>> vparts(NCARD);
    for (int c = 0; c < NCARD; ++c) {
        kparts[c][0] = khead_first(NCARD, c);
        kparts[c][1] = khead_count(NCARD, c);
        ffparts[c][0] = ff_lo(NCARD, c);
        ffparts[c][1] = ff_hi(NCARD, c);
    }
    for (int v = 0; v < NVH; ++v) vparts[(size_t) vhead_card(NCARD, v)].push_back(v);
    for (int c = 0; c < NCARD; ++c) {
        const std::string slice = "card " + std::to_string(c) + "'s FF slice [" + std::to_string(ffparts[c][0]) + ", " +
                                  std::to_string(ffparts[c][1]) + ")";
        if (ffparts[c][0] % 32 || ffparts[c][1] % 32) throw std::runtime_error("fnx: " + slice + " is not whole 32-value blocks");
        if (ffparts[c][1] - ffparts[c][0] > FF_CARD_MAX)
            throw std::runtime_error("fnx: " + slice + " is wider than FF_CARD_MAX (" + std::to_string(FF_CARD_MAX) + ")");
        if (ffparts[c][0] != (c ? ffparts[c - 1][1] : 0)) throw std::runtime_error("fnx: " + slice + " does not start where the last ended");
    }
    if (ffparts[NCARD - 1][1] != FF) throw std::runtime_error("fnx: the cards' FF slices do not cover the FF width");
    {   // every value head on exactly one card, beside its key head (value head v belongs to key head v % NKH)
        std::vector<int> seen(NVH, 0);
        for (int c = 0; c < NCARD; ++c)
            for (const int v : vparts[(size_t) c]) {
                if (v < 0 || v >= NVH) throw std::runtime_error("fnx: card " + std::to_string(c) + " lists value head " + std::to_string(v));
                ++seen[(size_t) v];
                const int k = v % NKH;
                if (k < kparts[c][0] || k >= kparts[c][0] + kparts[c][1])
                    throw std::runtime_error("fnx: card " + std::to_string(c) + " has value head " + std::to_string(v) + " without its key head " +
                                             std::to_string(k));
            }
        for (int v = 0; v < NVH; ++v)
            if (seen[(size_t) v] != 1)
                throw std::runtime_error("fnx: value head " + std::to_string(v) + " is on " + std::to_string(seen[(size_t) v]) + " cards");
    }
    const int vstep = (vocab + NCARD - 1) / NCARD;
    for (int c = 0; c < NCARD; ++c) {
        CardW& cw = cards[c];
        cw.dev = c;
        cw.k0 = kparts[c][0];
        cw.nk = kparts[c][1];
        cw.vheads = vparts[c];
        cw.q0 = c * (NQH / NCARD);
        cw.nq = NQH / NCARD;
        cw.kv0 = cw.q0 / (NQH / NKVH);
        cw.nkv = (cw.q0 + cw.nq - 1) / (NQH / NKVH) - cw.kv0 + 1;
        cw.ff0 = ffparts[c][0];
        cw.ff1 = ffparts[c][1];
        cw.v0 = c * vstep;
        cw.v1 = std::min(vocab, (c + 1) * vstep);
    }
#if FNX_27B
    for (int c = 0; c < NCARD; ++c) cards[c].fd0 = ffd_lo(NCARD, c), cards[c].fd1 = ffd_lo(NCARD, c + 1);
    if (cards[NCARD - 1].fd1 != FFD) throw std::runtime_error("27b: the cards' dense FF slices do not cover its width");
    {   // the image-pad token, whose runs in a prompt are a picture's cells (tools/sfnx.cpp, GENI)
        const std::vector<std::string>& toks = g.tokens();
        const auto it = std::find(toks.begin(), toks.end(), "<|image_pad|>");
        if (it == toks.end()) throw std::runtime_error("27b: the tokenizer has no <|image_pad|> token");
        ple_image = it - toks.begin();
        // the ids the engine process stops and starts turns on (tools/sfnx.cpp's EOS and 248045, front/engine_limits.py's
        // EOS_IDS), which are Flash-Next's tokenizer's: a GGUF tokenized otherwise would run past its turns' ends
        for (const auto& [id, want] : {std::pair{248044, "<|endoftext|>"}, std::pair{248045, "<|im_start|>"}, std::pair{248046, "<|im_end|>"}})
            if ((size_t) id >= toks.size() || toks[(size_t) id] != want)
                throw std::runtime_error("27b: token " + std::to_string(id) + " is '" + ((size_t) id < toks.size() ? toks[(size_t) id] : "") +
                                         "', the engine stops on it as " + want);
    }
    if (how != Load::PLAIN) throw std::runtime_error("27b: plain loads only (the lab's holder does not hold the 27B)");
#endif
    if (how == Load::ATTACH) {   // a lab worker: the holder's weights are mapped already; take its tables, place the experts
        if (!image) throw std::runtime_error("fnx: attaching needs the holder's image");
        if (!store) throw std::runtime_error("fnx: attaching needs the store the holder's memory is mapped in");
        restore(*image, *store);
        place_experts();
        return;
    }
    if (how == Load::HOLD && !store) throw std::runtime_error("fnx: holding needs a store");
#if !FNX_27B
    if (how == Load::PLAIN) choose_residency();   // the holder leaves that to its workers: every expert to a host copy
#endif
    const auto t0 = std::chrono::steady_clock::now();
    Stager st(g, stage_groups(g), NCARD);
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c)
        th.emplace_back([&, c] {
            try {
                load_card(st, cards[c], resident, arena ? &score : nullptr, how == Load::HOLD ? store : nullptr,
                          how == Load::HOLD || !resident.empty(), perm);
            } catch (const std::exception& e) {
                errs[c] = e.what();
                st.abort("card " + std::to_string(c) + " failed");   // the other cards wait on groups no one gives back
            }
        });
    for (auto& t : th) t.join();
    for (int c = 0; c < NCARD; ++c)
        if (!errs[c].empty()) throw std::runtime_error("card " + std::to_string(c) + ": " + errs[c]);
    // the disk rate only when the disk gave nearly all of it (tools/lab.sh estimates its next cold loads from it)
    const double disk = st.disk_bytes() / 1e9, cache = st.cache_bytes() / 1e9;
    char rate[64] = "";
    if (cache < 0.05 * (disk + cache)) std::snprintf(rate, sizeof rate, " at %.2f GB/s", disk / st.read_seconds());
    std::fprintf(stderr, "fnx: weights read in %.1f s of reading (%.1f GB from the disk%s, %.1f GB from the page cache), placed in %.1f s\n",
                 st.read_seconds(), disk, rate, cache, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

// FNX_EXPERT_MASK=file (maska_patch.py): one line a layer, "layer e e ...", the experts kept (indices of the full model);
// '#' starts a comment line. Every layer needs a line and at least TOPK experts
void Model::load_expert_mask() {
    const char* mf = std::getenv("FNX_EXPERT_MASK");
    if (!mf || !*mf || !keep.empty()) return;
    std::ifstream in(mf);
    if (!in) throw std::runtime_error(std::string("fnx: FNX_EXPERT_MASK: cannot open ") + mf);
    std::vector<char> k((size_t) NL * NEXP, 0);
    std::vector<int> per(NL, 0);
    std::vector<char> seen(NL, 0);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        int l = -1, e = -1;
        if (!(is >> l) || l < 0 || l >= NL || seen[(size_t) l]) throw std::runtime_error("fnx: FNX_EXPERT_MASK: not a new layer's line: " + line.substr(0, 80));
        seen[(size_t) l] = 1;
        while (is >> e) {
            if (e < 0 || e >= NEXP) throw std::runtime_error("fnx: FNX_EXPERT_MASK: layer " + std::to_string(l) + ": expert " + std::to_string(e) + " out of range");
            if (!k[(size_t) l * NEXP + e]) k[(size_t) l * NEXP + e] = 1, ++per[(size_t) l];
        }
        if (!is.eof()) throw std::runtime_error("fnx: FNX_EXPERT_MASK: layer " + std::to_string(l) + ": not a number in its line");
    }
    int total = 0;
    for (int l = 0; l < NL; ++l) {
        if (per[(size_t) l] < TOPK)
            throw std::runtime_error("fnx: FNX_EXPERT_MASK: layer " + std::to_string(l) + " keeps " + std::to_string(per[(size_t) l]) +
                                     " experts, fewer than the " + std::to_string(TOPK) + " a token takes");
        total += per[(size_t) l];
    }
    keep.swap(k);
    std::fprintf(stderr, "fnx: FNX_EXPERT_MASK %s: %d of %d routed experts kept\n", mf, total, NL * NEXP);
}

void Model::choose_residency() {
    load_expert_mask();   // FNX_EXPERT_MASK (maska_patch.py)
    resident.clear();
    score.clear();
    perm.clear();
    // prod's settings are the defaults: the arena, the coldest 3 GiB of experts in RAM, ranked by
    // data/expert-routes.bin (expert use tallied over an agentic code-review panel).
    // FNX_ARENA=0 and FNX_EXPERT_RAM_GIB=off together: every expert in VRAM, no expert cache
    const char* ram_s = std::getenv("FNX_EXPERT_RAM_GIB");
    const char* arena_s = std::getenv("FNX_ARENA");
    if (arena_s && std::strcmp(arena_s, "0") && std::strcmp(arena_s, "1"))
        throw std::runtime_error(std::string("fnx: FNX_ARENA: 0 or 1, not '") + arena_s + "'");
    arena = !arena_s || !std::strcmp(arena_s, "1");
    const bool cache = !(ram_s && !std::strcmp(ram_s, "off"));
    double ram_gib = 3.0;
    if (ram_s && cache) {
        char* end = nullptr;
        ram_gib = std::strtod(ram_s, &end);
        if (end == ram_s || *end || ram_gib < 0)
            throw std::runtime_error(std::string("fnx: FNX_EXPERT_RAM_GIB: GiB (0 or more) or off, not '") + ram_s + "'");
    }
    if (arena && !cache) throw std::runtime_error("fnx: FNX_EXPERT_RAM_GIB=off needs FNX_ARENA=0 (the arena lends the cache's slots)");
    const char* dyn_s = std::getenv("FNX_EXPERT_DYN_GIB");
    if (dyn_s && !cache) throw std::runtime_error("fnx: FNX_EXPERT_DYN_GIB needs the expert cache (FNX_EXPERT_RAM_GIB not off)");
    if (cache) {
        const double want = ram_gib * (double) (1ull << 30);
        const char* prof_s = std::getenv("FNX_EXPERT_PROFILE");
        const std::string prof_d = std::string(FNX_DIR) + "/data/expert-routes.bin";
        const char* prof = prof_s ? prof_s : prof_d.c_str();
        std::vector<unsigned> cnt((size_t) 2 * NL * NEXP);
        {
            std::FILE* f = std::fopen(prof, "rb");
            if (!f || std::fread(cnt.data(), sizeof(unsigned), cnt.size(), f) != cnt.size() || std::fgetc(f) != EOF)
                throw std::runtime_error(std::string("fnx: FNX_EXPERT_PROFILE ") + prof + ": not " + std::to_string(cnt.size()) + " tallies");
            std::fclose(f);
        }
        // decode and prompt picks weighted equally (each kind's share of its total), per byte; the coldest go to RAM
        double td = 0, tp = 0;
        for (int i = 0; i < NL * NEXP; ++i) { td += cnt[(size_t) i]; tp += cnt[(size_t) (NL * NEXP + i)]; }
        std::vector<double> bytes(NL);
        score.assign((size_t) NL * NEXP, 0.0);
        for (int l = 0; l < NL; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            bytes[(size_t) l] = (double) (g.t(b + "ffn_gate_exps.weight").nbytes + g.t(b + "ffn_up_exps.weight").nbytes +
                                          g.t(b + "ffn_down_exps.weight").nbytes) / NEXP;
            for (int e = 0; e < NEXP; ++e) {
                const size_t i = (size_t) l * NEXP + e;
                score[i] = (cnt[i] / std::max(1.0, td) + cnt[(size_t) NL * NEXP + i] / std::max(1.0, tp)) / bytes[(size_t) l];
                if (!kept((int) i)) score[i] = -1.0;   // FNX_EXPERT_MASK: never picked, the first to RAM
            }
        }
        std::vector<int> order((size_t) NL * NEXP);
        for (int i = 0; i < NL * NEXP; ++i) order[(size_t) i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[(size_t) a] < score[(size_t) b]; });
        resident.assign(NL, std::vector<char>(NEXP, 1));
        double moved = 0;
        int n = 0, extra = 0;
        // every size keeps a window's picks in VRAM (Strata 20885f7a): a size is a layer's (gate|up, down) bytes an expert, the
        // pairs Up::expert_cache keys its pools by; an expert passed over for that is not one of the ram_extra, and a placement
        // the floor cannot meet stops the load
        std::vector<std::pair<std::pair<size_t, size_t>, int>> left;   // the experts each size still has in VRAM
        std::vector<int> size_of((size_t) NL);
        for (int l = 0; l < NL; ++l) {
            const std::string b = "blk." + std::to_string(l) + ".";
            const std::pair<size_t, size_t> sz{(g.t(b + "ffn_gate_exps.weight").nbytes + g.t(b + "ffn_up_exps.weight").nbytes) / NEXP,
                                               g.t(b + "ffn_down_exps.weight").nbytes / NEXP};
            auto it = std::find_if(left.begin(), left.end(), [&](const auto& x) { return x.first == sz; });
            if (it == left.end()) it = left.insert(left.end(), {sz, 0});
            it->second += NEXP;
            size_of[(size_t) l] = (int) (it - left.begin());
        }
        constexpr int floor = 2 * MAXT * TOPK + 64;   // a pool must keep a window's picks
        for (int i : order) {
            if (moved >= want && extra >= ram_extra) break;
            int& keep = left[(size_t) size_of[(size_t) (i / NEXP)]].second;
            if (keep <= floor) continue;
            if (moved >= want) ++extra;
            --keep;
            resident[(size_t) (i / NEXP)][(size_t) (i % NEXP)] = 0;
            moved += bytes[(size_t) (i / NEXP)];
            ++n;
        }
        if (moved < want || extra < ram_extra)
            throw std::runtime_error("fnx: FNX_EXPERT_RAM_GIB: the experts asked for in RAM would leave an expert size fewer than a window's " +
                                     std::to_string(floor) + " slots");
        std::fprintf(stderr, "fnx: %d of %d routed experts (%.1f GiB) in RAM, ranked by %s%s\n", n, NL * NEXP, moved / (1ull << 30), prof,
                     ram_extra ? (" (" + std::to_string(ram_extra) + " of them because VRAM was short)").c_str() : "");
        if (dyn_s) {
            // FNX_EXPERT_DYN_GIB=G (64 GB of RAM; the M40 port's): of the experts in VRAM the coldest G GiB (at least `keep` a
            // pool) stay dynamic, with a host copy; the rest are permanent: placed once, no host copy, so RAM holds G plus
            // FNX_EXPERT_RAM_GIB of experts. What the pools lend (K/V with FNX_ARENA, the prompt path with FNX_PF_LEND) comes
            // from the dynamic slots only: the Engine says so when they are too few
            char* end = nullptr;
            const double dyn_gib = std::strtod(dyn_s, &end);
            if (end == dyn_s || *end || dyn_gib < 0) throw std::runtime_error(std::string("fnx: FNX_EXPERT_DYN_GIB: GiB (0 or more), not '") + dyn_s + "'");
            const int keep = (int) std::max<long long>(floor, env_int("FNX_EXPERT_DYN_MIN", floor));
            perm.assign(NL, std::vector<char>(NEXP, 0));
            std::vector<char> dyn((size_t) NL * NEXP, 0);
            std::vector<int> per(left.size(), 0);
            double dyn_bytes = 0;
            int nd = 0;
            auto take = [&](int i) {
                dyn[(size_t) i] = 1;
                ++per[(size_t) size_of[(size_t) (i / NEXP)]];
                dyn_bytes += bytes[(size_t) (i / NEXP)];
                ++nd;
            };
            for (int i : order)   // each pool's coldest `keep` first
                if (resident[(size_t) (i / NEXP)][(size_t) (i % NEXP)] && per[(size_t) size_of[(size_t) (i / NEXP)]] < keep) take(i);
            for (int i : order)
                if (dyn_bytes < dyn_gib * (double) (1ull << 30) && resident[(size_t) (i / NEXP)][(size_t) (i % NEXP)] && !dyn[(size_t) i]) take(i);
            int np = 0;
            double pb = 0;
            for (int l = 0; l < NL; ++l)
                for (int e = 0; e < NEXP; ++e)
                    if (resident[(size_t) l][(size_t) e] && !dyn[(size_t) l * NEXP + e]) perm[(size_t) l][(size_t) e] = 1, ++np, pb += bytes[(size_t) l];
            std::string pools;
            for (size_t p = 0; p < left.size(); ++p)
                pools += (p ? ", " : "") + std::to_string(per[p]) + " of " + std::to_string(left[p].second);
            std::fprintf(stderr, "fnx: FNX_EXPERT_DYN_GIB: %d experts (%.1f GiB) permanent in VRAM with no host copy, %d (%.1f GiB) dynamic "
                                 "(by pool: %s); RAM holds %.1f GiB of experts\n",
                         np, pb / (1ull << 30), nd, dyn_bytes / (1ull << 30), pools.c_str(), (dyn_bytes + moved) / (1ull << 30));
        }
    }
}

// What only the load needed, once the model, the draft layer and the engine stand: the converted tensors (every card's slices are placed; 9 layers' down experts are
// among them), the mappings' pages and the file's page cache. Then what the load's peak sent to swap comes back here, not a fault at a time in the rounds.
void model_loaded(Model& m) {
    if (g_numa_bound[0] + g_numa_bound[1] + g_numa_pref[0] + g_numa_pref[1])   // numa_patch.py
        std::fprintf(stderr, "fnx: FNX_NUMA_EXPERTS: card 0 %.1f GiB bound + %.1f preferred to its node, card 1 %.1f + %.1f\n",
                     g_numa_bound[0] / 1073741824.0, g_numa_pref[0] / 1073741824.0, g_numa_bound[1] / 1073741824.0, g_numa_pref[1] / 1073741824.0);
    const auto t0 = std::chrono::steady_clock::now();
    constexpr uintptr_t PG = 4096;
    size_t freed = 0;
    for (const auto& kv : m.g.tensors()) {
        const Tensor& t = kv.second;
        if (t.shard >= 0 || !t.data || &t == m.tok_embd || &t == m.ple_table) continue;
        const uintptr_t a0 = ((uintptr_t) t.data + PG - 1) / PG * PG, e0 = ((uintptr_t) t.data + t.nbytes) / PG * PG;
        if (e0 > a0 && madvise((void*) a0, e0 - a0, MADV_DONTNEED) == 0) freed += e0 - a0;
        const_cast<Tensor&>(t).data = nullptr;   // a reader past this point stops at once, it does not read zeros
    }
    malloc_trim(0);
    // FNX_PLE_RAM=0 (64 GB of RAM; the M40 port's): the PLE table (26.8 GiB) stays in the file, read through the page cache as Strata's
    // own engine does: random access (no read-around of 128 KiB a fault), and its file's cached pages are kept
    const bool ple_ram = env_int("FNX_PLE_RAM", 1) != 0;
    const int ple_shard = m.ple_table ? m.ple_table->shard : -1;
    {   // after everything else: with the converted tensors gone the load's peak stays about 9 GiB lower, and the kernel has no reason to swap
        const auto t1 = std::chrono::steady_clock::now();
        lock_in_ram(m.g, *m.tok_embd);
        if (ple_ram) lock_in_ram(m.g, *m.ple_table);
        std::fprintf(stderr, "fnx: %s%s%s (%.1f GB) locked in RAM in %.1f s\n", m.tok_embd->name.c_str(), ple_ram ? " and " : "",
                     ple_ram ? m.ple_table->name.c_str() : "", (m.tok_embd->nbytes + (ple_ram ? m.ple_table->nbytes : 0)) / 1e9,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
    }
    for (int s = 0; s < m.g.shards(); ++s) {
        if (!ple_ram && s == ple_shard) {
            // the table's own pages stay mapped and cached; only what the load touched around it is dropped
            const uintptr_t base = (uintptr_t) m.g.shard_base(s), end = base + m.g.shard_size(s),
                            a0 = (uintptr_t) m.ple_table->data / PG * PG, e0 = ((uintptr_t) m.ple_table->data + m.ple_table->nbytes + PG - 1) / PG * PG;
            if (a0 > base) madvise((void*) base, a0 - base, MADV_DONTNEED);
            if (end > e0) madvise((void*) e0, end - e0, MADV_DONTNEED);
            madvise((void*) a0, e0 - a0, MADV_RANDOM);
            continue;
        }
        madvise((void*) m.g.shard_base(s), m.g.shard_size(s), MADV_DONTNEED);
        const int fd = ::open(m.g.shard_path(s).c_str(), O_RDONLY);
        if (fd >= 0) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }
    if (!ple_ram)
        std::fprintf(stderr, "fnx: FNX_PLE_RAM=0: %s (%.1f GB) read from its file through the page cache\n", m.ple_table->name.c_str(),
                     m.ple_table->nbytes / 1e9);
    // FNX_PLE_WARM=1 (warm_patch.py): the table read once by a thread at idle I/O priority, so the page cache holds what it can of it
    // before the first requests ask for its rows (they would each wait for the disk)
    if (!ple_ram && env_int("FNX_PLE_WARM", 0) != 0) {
        const std::string path = m.g.shard_path(ple_shard);
        const uint64_t off = (uint64_t) ((uintptr_t) m.ple_table->data - (uintptr_t) m.g.shard_base(ple_shard)), len = m.ple_table->nbytes;
        std::thread([path, off, len] {
            syscall(SYS_ioprio_set, 1 /* IOPRIO_WHO_PROCESS: this thread */, 0, 3 << 13 /* IOPRIO_CLASS_IDLE */);
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) return;
            std::vector<char> buf((size_t) 16 << 20);
            const auto t0 = std::chrono::steady_clock::now();
            uint64_t done = 0;
            while (done < len) {
                const ssize_t r = ::pread(fd, buf.data(), (size_t) std::min<uint64_t>(buf.size(), len - done), (off_t) (off + done));
                if (r <= 0) break;
                done += (uint64_t) r;
            }
            ::close(fd);
            std::fprintf(stderr, "fnx: FNX_PLE_WARM: %.1f GB of %s read in %.0f s\n", done / 1e9, path.c_str(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }).detach();
    }
    auto swap_mib = [] {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("VmSwap:", 0) == 0) return std::atoll(line.c_str() + 7) / 1024;
        return 0ll;
    };
    const long long sw0 = swap_mib();
    std::vector<std::thread> th;
    for (const Tensor* t : {m.tok_embd, ple_ram ? m.ple_table : nullptr}) {
        if (!t || !t->data) continue;
        for (int i = 0; i < 4; ++i)
            th.emplace_back([t, i] {
                unsigned char sink = 0;
                const size_t part = t->nbytes / 4 + 1, lo = (size_t) i * part, hi = std::min(t->nbytes, lo + part);
                for (size_t k = lo; k < hi; k += PG) sink += *(const volatile unsigned char*) (t->data + k);
                (void) sink;
            });
    }
    for (auto& x : th) x.join();
    std::fprintf(stderr, "fnx: RAM after the load: %.2f GiB of converted tensors freed, swap %lld -> %lld MiB, in %.1f s\n", freed / (double) (1ull << 30), sw0, swap_mib(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}

namespace {
template <typename F>
void each_card(std::vector<CardW>& cards, F f) {
    std::vector<std::thread> th;
    std::vector<std::string> errs(NCARD);
    for (int c = 0; c < NCARD; ++c)
        th.emplace_back([&, c] {
            try {
                ck(cudaSetDevice(cards[c].dev), "cudaSetDevice");
                f(cards[c]);
            } catch (const std::exception& e) { errs[c] = e.what(); }
        });
    for (auto& t : th) t.join();
    for (int c = 0; c < NCARD; ++c)
        if (!errs[c].empty()) throw std::runtime_error("card " + std::to_string(c) + ": " + errs[c]);
}
}  // namespace

void put_report() {
    std::vector<std::pair<size_t, std::string>> rows;
    for (const auto& kv : g_put_note) rows.emplace_back(kv.second.first, "model.cpp:" + std::to_string(kv.first) + " x" + std::to_string(kv.second.second));
    std::sort(rows.rbegin(), rows.rend());
    for (size_t i = 0; i < rows.size() && i < 30; ++i) std::fprintf(stderr, "fnx vram: nalozeno %8.1f MiB %s\n", rows[i].first / 1048576.0, rows[i].second.c_str());
}
void vram_step(const char* what) {
    if (!std::getenv("FNX_PROF")) return;
    size_t fr = 0, tot = 0;
    ck(cudaSetDevice(0), "cudaSetDevice");
    ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
    std::fprintf(stderr, "fnx vram: prosto po \"%s\": %zu od %zu MiB\n", what, fr >> 20, tot >> 20);
}

void Model::place_experts() {
    choose_residency();
    if (!perm.empty())
        throw std::runtime_error("fnx: FNX_EXPERT_DYN_GIB: the permanent experts have no host copy, so they cannot be placed again "
                                 "(more in RAM: FNX_EXPERT_RAM_GIB higher)");
    each_card(cards, [&](CardW& c) {
        if (c.ec_host_gu.size() != (size_t) NL * NEXP) throw std::runtime_error("fnx: placing experts needs their host copies");
        const size_t before = c.bytes;
        Up up{c, nullptr, &c.expert_mem};
        if (resident.empty()) {   // every expert in VRAM: each layer's host copy is one block
            for (int l = 0; l < NL; ++l) {
                LayerW& L = c.L[(size_t) l];
                L.gu_e.p = up.put<uint8_t>(c.ec_host_gu[(size_t) l * NEXP], (size_t) NEXP * L.gu_e_stride);
                L.down_e.p = up.put<uint8_t>(c.ec_host_dn[(size_t) l * NEXP], (size_t) NEXP * L.down_e_stride);
            }
        } else {
            const std::vector<const char*> hgu = c.ec_host_gu, hdn = c.ec_host_dn;
            up.expert_cache(resident, hgu, hdn, arena ? &score : nullptr);
        }
        c.expert_bytes = c.bytes - before;
    });
}

void Model::free_experts() {
    each_card(cards, [&](CardW& c) {
        for (void* p : c.expert_mem) ck(cudaFree(p), "free experts");
        c.expert_mem.clear();
        c.bytes -= c.expert_bytes;
        c.expert_bytes = 0;
        c.ec = ExpertCache{};
        c.ec_owner0.clear();
        c.ec_slot0.clear();
        for (LayerW& L : c.L) {
            L.gu_e.p = L.down_e.p = nullptr;
            L.gu_e.ep = L.down_e.ep = nullptr;
        }
    });
}

// ---------------------------------------------------------------- the holder's tables for a worker
// Per card what load_card stored (the geometry the constructor sets again): its pointers, the layers, the experts'
// host copies. card_fields lists them once for both directions; a field added to CardW without it fails the size check.
namespace {
static_assert(std::is_trivially_copyable<LayerW>::value && std::is_trivially_copyable<DW>::value, "layers copied as bytes");
struct Img {
    std::string s;
    size_t at = 0;
    template <typename T> void put(const T& v) { s.append((const char*) &v, sizeof v); }
    template <typename T> void get(T& v) {
        if (at + sizeof v > s.size()) throw std::runtime_error("fnx: the holder's image is short");
        std::memcpy((void*) &v, s.data() + at, sizeof v);
        at += sizeof v;
    }
    template <typename T> void put_vec(const std::vector<T>& v) {
        put((uint64_t) v.size());
        s.append((const char*) v.data(), v.size() * sizeof(T));
    }
    template <typename T> void get_vec(std::vector<T>& v) {
        uint64_t n = 0;
        get(n);
        if (at + n * sizeof(T) > s.size()) throw std::runtime_error("fnx: the holder's image is short");
        v.resize(n);
        std::memcpy((void*) v.data(), s.data() + at, n * sizeof(T));
        at += n * sizeof(T);
    }
};
template <typename F>
void card_fields(CardW& c, F f) {
    f(c.vmap), f(c.head_norm), f(c.head_down), f(c.head_up), f(c.out), f(c.ple_key), f(c.ple_value), f(c.ple_norm_key),
        f(c.ple_norm_query), f(c.ple_norm_conv), f(c.ple_conv), f(c.bytes), f(c.host_bytes);
}
#if !FNX_27B   // the 27B has no holder (plain loads only)
// 1008: the permanent experts (perm_*, ExpertCache n_dyn; FNX_EXPERT_DYN_GIB) are not in card_fields: the holder never has them
static_assert(sizeof(CardW) == 1008, "CardW changed: put what load_card stores in card_fields, then update this size");
#endif
// every address load_card stores, in a card and in a layer: a worker moves them to where it mapped the holder's memory
template <typename F>
void card_ptrs(CardW& c, F f) {
    auto q8 = [&](Q8S& s) { f(s.q), f(s.d); };
    auto dw = [&](DW& d) { f(d.p), q8(d.s), f(d.ep); };
    f(c.vmap), f(c.head_norm), q8(c.head_down), q8(c.head_up), dw(c.out), dw(c.ple_key), dw(c.ple_value), f(c.ple_norm_key),
        f(c.ple_norm_query), f(c.ple_norm_conv), f(c.ple_conv);
    for (LayerW& w : c.L) {
        for (int h = 0; h < 2; ++h) f(w.hc_norm[h]), q8(w.hc_down[h]), q8(w.hc_up[h]), f(w.hc_inj[h]);
        f(w.router), dw(w.qkv), dw(w.z), dw(w.ssm_out), dw(w.qkvz), f(w.ab), f(w.conv), f(w.alpha), f(w.beta), f(w.ssm_a),
            f(w.dt_bias), f(w.ssm_norm);
        dw(w.wq), dw(w.wk), dw(w.wv), dw(w.wo), f(w.q_norm), f(w.k_norm), f(w.idx_wq), f(w.idx_wk), dw(w.idx_qk), f(w.idx_qn),
            f(w.idx_kn);
        dw(w.gu_e), dw(w.down_e), dw(w.sh_gu), dw(w.sh_down);
        q8(w.lat_d), q8(w.lat_u), f(w.lat_lab), f(w.lat_esc);   // edge0_patch.py
    }
    for (const char*& p : c.ec_host_gu) f(p);
    for (const char*& p : c.ec_host_dn) f(p);
}
#if !FNX_27B
static_assert(sizeof(LayerW) == 1000, "LayerW changed: put its new addresses in card_ptrs, then update this size");
#endif
}  // namespace

std::string Model::image() const {
    Img m;
    m.put((uint32_t) 0x31474d49);   // "IMG1"
    for (const CardW& cc : cards) {
        CardW& c = const_cast<CardW&>(cc);
        card_fields(c, [&](auto& v) { m.put(v); });
        m.put_vec(c.L);
        m.put_vec(c.ec_host_gu);
        m.put_vec(c.ec_host_dn);
    }
    return m.s;
}

void Model::restore(const std::string& image, const Store& store) {
    Img m{image};
    uint32_t magic = 0;
    m.get(magic);
    if (magic != 0x31474d49) throw std::runtime_error("fnx: not a holder's image");
    for (CardW& c : cards) {
        card_fields(c, [&](auto& v) { m.get(v); });
        m.get_vec(c.L);
        m.get_vec(c.ec_host_gu);
        m.get_vec(c.ec_host_dn);
        if (c.L.size() != (size_t) NL) throw std::runtime_error("fnx: the holder's image has another layer count");
        card_ptrs(c, [&](auto& p) { store.move_ptr(p); });
    }
    if (m.at != image.size()) throw std::runtime_error("fnx: the holder's image is longer than its cards");
}

}  // namespace fnx
