#!/usr/bin/env python3
# numa_patch.py SRC_DIR (2026-11-12 night): FNX_NUMA_EXPERTS=1 puts each card's page-locked copy of the routed experts (expert_buf's
# cudaHostAlloc, the halves that card copies into its slots) on the memory of the socket that card hangs off (sysfs numa_node of its
# PCI address: card 0 -> node 0, card 1 -> node 1 on ai-streznik). Without it the pages land wherever there is room as the load reads
# the file through the page cache (vm.zone_reclaim_mode 0 takes the other node's free pages before reclaiming this one's): production
# had 27.8 GiB of its 48.8 on node 0 and 22.4 on node 1, so part of card 1's copies crossed the link between the sockets.
# MPOL_BIND (the node's own page cache is reclaimed for it) when the node's free and file pages less its CMA pageblocks exceed the
# allocation by 4 GiB, else
# MPOL_PREFERRED (that node while it has free pages, then the other); the calling thread only, reset after. Placement only: the same bits.
# The load prints how much went each way.
import sys

src = sys.argv[1]


def edit(name, pairs):
    p = src + "/" + name
    s = open(p).read()
    for i, (a, b, want) in enumerate(pairs):
        assert s.count(a) == want, (name, i, s.count(a))
        s = s.replace(a, b, 1)
    open(p, "w").write(s)


edit("model.cpp", [
    ("#include <malloc.h>\n", "#include <malloc.h>\n#include <sys/syscall.h>\n", 1),
    ("// the ring stores straight into the next card's memory: with translated IOMMU domains every such write faults\n",
     "// FNX_NUMA_EXPERTS (numa_patch.py): the NUMA node of a card's PCI slot, -1 when sysfs names none\n"
     "int card_numa_node(int dev) {\n"
     "    std::ifstream f(\"/sys/bus/pci/devices/\" + card_bdf(dev) + \"/numa_node\");\n"
     "    int n = -1;\n"
     "    if (!(f >> n)) n = -1;\n"
     "    return n;\n"
     "}\n"
     "// a node's room for page-locked memory: its free and file-backed (reclaimable) bytes from its meminfo, less its CMA pageblocks\n"
     "// (/proc/zoneinfo's cma, pages): pinned pages never come from those (ai-streznik: 8.6 GiB of node 1, 2.5 of node 0), so without\n"
     "// them MPOL_BIND could reclaim the node and still come up short (Strata PR #5436 met the same)\n"
     "size_t numa_node_room(int node) {\n"
     "    std::ifstream f(\"/sys/devices/system/node/node\" + std::to_string(node) + \"/meminfo\");\n"
     "    std::string line;\n"
     "    size_t kb = 0;\n"
     "    while (std::getline(f, line)) {\n"
     "        char key[64];\n"
     "        unsigned long long v = 0;\n"
     "        if (std::sscanf(line.c_str(), \"Node %*d %63[^:]: %llu\", key, &v) == 2 &&\n"
     "            (!std::strcmp(key, \"MemFree\") || !std::strcmp(key, \"Active(file)\") || !std::strcmp(key, \"Inactive(file)\")))\n"
     "            kb += v;\n"
     "    }\n"
     "    std::ifstream z(\"/proc/zoneinfo\");\n"
     "    size_t cma_pages = 0;\n"
     "    bool mine = false;\n"
     "    while (std::getline(z, line)) {\n"
     "        int n = -1;\n"
     "        unsigned long long v = 0;\n"
     "        if (std::sscanf(line.c_str(), \"Node %d, zone\", &n) == 1) mine = n == node;\n"
     "        else if (mine && std::sscanf(line.c_str(), \" cma %llu\", &v) == 1) cma_pages += v;\n"
     "    }\n"
     "    const size_t bytes = kb << 10, cma = cma_pages * 4096;\n"
     "    return bytes > cma ? bytes - cma : 0;\n"
     "}\n"
     "size_t g_numa_bound[2], g_numa_pref[2];   // per card: bytes allocated under MPOL_BIND, under MPOL_PREFERRED\n"
     "// the ring stores straight into the next card's memory: with translated IOMMU domains every such write faults\n", 1),
    ("        void* hp = nullptr;\n        const cudaError_t err = store ? (hp = store->host(c.dev, bytes), cudaSuccess) : cudaHostAlloc(&hp, bytes, cudaHostAllocMapped);\n",
     "        void* hp = nullptr;\n"
     "        // FNX_NUMA_EXPERTS=1 (numa_patch.py): on the card's own socket (MPOL_BIND while the node has room, else MPOL_PREFERRED)\n"
     "        static const bool numa = env_int(\"FNX_NUMA_EXPERTS\", 0) != 0;\n"
     "        const int node = numa && !store ? card_numa_node(c.dev) : -1;\n"
     "        if (node >= 0 && node < 64) {\n"
     "            const unsigned long mask = 1ul << node;\n"
     "            const bool bind = numa_node_room(node) > bytes + ((size_t) 4 << 30);\n"
     "            if (syscall(SYS_set_mempolicy, bind ? 2 /* MPOL_BIND */ : 1 /* MPOL_PREFERRED */, &mask, 8 * sizeof mask) != 0)\n"
     "                throw std::runtime_error(std::string(\"FNX_NUMA_EXPERTS: set_mempolicy: \") + std::strerror(errno));\n"
     "            (bind ? g_numa_bound : g_numa_pref)[c.dev & 1] += bytes;\n"
     "        }\n"
     "        const cudaError_t err = store ? (hp = store->host(c.dev, bytes), cudaSuccess) : cudaHostAlloc(&hp, bytes, cudaHostAllocMapped);\n"
     "        if (node >= 0 && node < 64) syscall(SYS_set_mempolicy, 0 /* MPOL_DEFAULT */, nullptr, 0);\n", 1),
    # the report, beside model_loaded's RAM line
    ("void model_loaded(Model& m) {\n",
     "void model_loaded(Model& m) {\n"
     "    if (g_numa_bound[0] + g_numa_bound[1] + g_numa_pref[0] + g_numa_pref[1])   // numa_patch.py\n"
     "        std::fprintf(stderr, \"fnx: FNX_NUMA_EXPERTS: card 0 %.1f GiB bound + %.1f preferred to its node, card 1 %.1f + %.1f\\n\",\n"
     "                     g_numa_bound[0] / 1073741824.0, g_numa_pref[0] / 1073741824.0, g_numa_bound[1] / 1073741824.0, g_numa_pref[1] / 1073741824.0);\n", 1),
])
print("numa_patch: model.cpp")
