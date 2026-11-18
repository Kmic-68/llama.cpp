#!/usr/bin/env python3
# warm_patch.py SRC_DIR (2026-11-13 night): FNX_PLE_WARM=1 with FNX_PLE_RAM=0: after the load a detached thread at idle I/O priority
# reads the n-gram (PLE) table once from its file, so the page cache holds what it can of it before requests ask for rows (Q4 XL on
# ai-streznik, DYN 14: the first 36k read 722 -> 794 tok/s with the table read beforehand, the same tokens). Needs dyn_port.sh's
# model_loaded (ple_ram, ple_shard) and numa_patch.py's <sys/syscall.h>.
import sys

src = sys.argv[1]
p = src + "/model.cpp"
s = open(p).read()
a = ('        std::fprintf(stderr, "fnx: FNX_PLE_RAM=0: %s (%.1f GB) read from its file through the page cache\\n", m.ple_table->name.c_str(),\n'
     '                     m.ple_table->nbytes / 1e9);\n')
b = a + '''    // FNX_PLE_WARM=1 (warm_patch.py): the table read once by a thread at idle I/O priority, so the page cache holds what it can of it
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
            std::fprintf(stderr, "fnx: FNX_PLE_WARM: %.1f GB of %s read in %.0f s\\n", done / 1e9, path.c_str(),
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }).detach();
    }
'''
assert s.count(a) == 1, s.count(a)
open(p, "w").write(s.replace(a, b, 1))
print("warm_patch: model.cpp")
