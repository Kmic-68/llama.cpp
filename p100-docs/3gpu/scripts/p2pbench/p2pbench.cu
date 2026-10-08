// Out-of-tree P2P characterization (round 1, G0).
// Per directed pair (src -> dst):
//   ce_push : cudaMemcpyPeerAsync issued on the SOURCE device's stream (source copy engine writes)
//   ce_pull : cudaMemcpyPeerAsync issued on the DESTINATION device's stream (dest copy engine reads)
//   k_write : kernel on src stores 16 B/thread into dst memory (P2P posted writes)
//   k_read  : kernel on dst loads 16 B/thread from src memory (P2P reads)
//   lat     : one-way flag latency, src kernel stores a token into dst memory, dst kernel spins on its
//             local copy and answers into src memory; round trip / 2, median of many round trips
//   staged  : same copy with peer access disabled (driver stages through host memory)
// Usage: p2pbench [--no-peer]   (run once with peer access on, once with --no-peer)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void k_store(int4 * __restrict__ dst, size_t n, int v) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = make_int4(v, v, v, v);
    }
}

__global__ void k_load(const int4 * __restrict__ src, size_t n, int * __restrict__ sink) {
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int4 x = src[i];
        acc ^= x.x ^ x.y ^ x.z ^ x.w;
    }
    if (acc == 0x7fffffff) *sink = acc;  // keep the loads alive
}

// ping: writes token into remote 'out', waits for the same token in local 'in'
__global__ void k_ping(volatile int * out_remote, volatile int * in_local, int iters, long long * cycles) {
    long long t0 = clock64();
    for (int t = 1; t <= iters; ++t) {
        *out_remote = t;
        __threadfence_system();
        while (*in_local != t) { }
    }
    *cycles = clock64() - t0;
}

// pong: waits for token in local 'in', echoes it into remote 'out'
__global__ void k_pong(volatile int * in_local, volatile int * out_remote, int iters) {
    for (int t = 1; t <= iters; ++t) {
        while (*in_local != t) { }
        *out_remote = t;
        __threadfence_system();
    }
}

static double time_ms(cudaEvent_t a, cudaEvent_t b) { float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char ** argv) {
    const bool no_peer = argc > 1 && strcmp(argv[1], "--no-peer") == 0;
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    printf("# devices %d, peer access %s\n", ndev, no_peer ? "DISABLED (staged)" : "enabled");
    for (int i = 0; i < ndev; ++i) {
        cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, i));
        printf("# dev %d %s pci %04x:%02x:%02x cc %d.%d clock %d kHz\n", i, p.name, p.pciDomainID, p.pciBusID, p.pciDeviceID, p.major, p.minor, p.clockRate);
    }
    if (!no_peer) {
        for (int i = 0; i < ndev; ++i) for (int j = 0; j < ndev; ++j) if (i != j) {
            int can = 0; CK(cudaDeviceCanAccessPeer(&can, i, j));
            printf("# canAccessPeer %d->%d = %d\n", i, j, can);
            if (can) { CK(cudaSetDevice(i)); cudaError_t pe = cudaDeviceEnablePeerAccess(j, 0); if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) CK(pe); cudaGetLastError(); }
        }
    }

    const size_t sizes[] = { 20 * 1024, 100 * 1024, 256 * 1024, 1 << 20, 20 << 20, 64 << 20 };
    const size_t maxb = 64 << 20;
    std::vector<void *> buf(ndev), buf2(ndev);
    std::vector<cudaStream_t> st(ndev);
    std::vector<int *> sink(ndev), flag(ndev);
    std::vector<long long *> cyc(ndev);
    for (int i = 0; i < ndev; ++i) {
        CK(cudaSetDevice(i));
        CK(cudaMalloc(&buf[i], maxb)); CK(cudaMalloc(&buf2[i], maxb));
        CK(cudaMemset(buf[i], i + 1, maxb)); CK(cudaMemset(buf2[i], 0, maxb));
        CK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
        CK(cudaMalloc(&sink[i], sizeof(int))); CK(cudaMalloc(&flag[i], 64)); CK(cudaMalloc(&cyc[i], sizeof(long long)));
    }
    CK(cudaDeviceSynchronize());

    printf("\nmode,src,dst,bytes,median_us,GBps\n");
    const int reps = 30;
    for (int s = 0; s < ndev; ++s) for (int d = 0; d < ndev; ++d) {
        if (s == d) continue;
        for (size_t nb : sizes) {
            // copy engine: push (stream on src), pull (stream on dst)
            for (int mode = 0; mode < 2; ++mode) {
                const int sd = mode == 0 ? s : d;
                CK(cudaSetDevice(sd));
                cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                std::vector<double> t;
                for (int r = 0; r < reps + 3; ++r) {
                    CK(cudaEventRecord(a, st[sd]));
                    CK(cudaMemcpyPeerAsync(buf2[d], d, buf[s], s, nb, st[sd]));
                    CK(cudaEventRecord(b, st[sd]));
                    CK(cudaEventSynchronize(b));
                    if (r >= 3) t.push_back(time_ms(a, b) * 1000.0);
                }
                const double us = median(t);
                printf("%s,%d,%d,%zu,%.2f,%.3f\n", no_peer ? (mode == 0 ? "staged_srcstream" : "staged_dststream") : (mode == 0 ? "ce_push" : "ce_pull"), s, d, nb, us, nb / us / 1e3);
                CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            }
            if (no_peer) continue;
            // kernel remote write (kernel on s writes into d) and remote read (kernel on d reads s)
            for (int mode = 0; mode < 2; ++mode) {
                const int kd = mode == 0 ? s : d;
                CK(cudaSetDevice(kd));
                cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
                const size_t n4 = nb / sizeof(int4);
                const int blocks = (int) std::min<size_t>(56 * 4, (n4 + 255) / 256);
                std::vector<double> t;
                for (int r = 0; r < reps + 3; ++r) {
                    CK(cudaEventRecord(a, st[kd]));
                    if (mode == 0) k_store<<<blocks, 256, 0, st[kd]>>>((int4 *) buf2[d], n4, r);
                    else           k_load <<<blocks, 256, 0, st[kd]>>>((const int4 *) buf[s], n4, sink[kd]);
                    CK(cudaGetLastError());
                    CK(cudaEventRecord(b, st[kd]));
                    CK(cudaEventSynchronize(b));
                    if (r >= 3) t.push_back(time_ms(a, b) * 1000.0);
                }
                const double us = median(t);
                printf("%s,%d,%d,%zu,%.2f,%.3f\n", mode == 0 ? "k_write" : "k_read", s, d, nb, us, nb / us / 1e3);
                CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
            }
        }
    }

    if (!no_peer) {
        printf("\nlat_mode,src,dst,iters,one_way_us\n");
        const int iters = 2000;
        for (int s = 0; s < ndev; ++s) for (int d = 0; d < ndev; ++d) {
            if (s == d) continue;
            CK(cudaSetDevice(s)); CK(cudaMemset(flag[s], 0, 64)); CK(cudaDeviceSynchronize());
            CK(cudaSetDevice(d)); CK(cudaMemset(flag[d], 0, 64)); CK(cudaDeviceSynchronize());
            // pong first so it is resident and spinning before ping starts
            CK(cudaSetDevice(d));
            k_pong<<<1, 1, 0, st[d]>>>((volatile int *) flag[d], (volatile int *) flag[s], iters);
            CK(cudaGetLastError());
            CK(cudaSetDevice(s));
            cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
            CK(cudaEventRecord(a, st[s]));
            k_ping<<<1, 1, 0, st[s]>>>((volatile int *) flag[d], (volatile int *) flag[s], iters, cyc[s]);
            CK(cudaGetLastError());
            CK(cudaEventRecord(b, st[s]));
            CK(cudaEventSynchronize(b));
            CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d]));
            const double us = time_ms(a, b) * 1000.0 / iters / 2.0;
            printf("flag_pingpong,%d,%d,%d,%.3f\n", s, d, iters, us);
            CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        }
    }
    printf("# done\n");
    return 0;
}
