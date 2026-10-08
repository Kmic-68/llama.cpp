// G0 extras (round 1): out-of-tree P2P microbenchmarks.
// (1) small-message latency: per directed pair, src kernel (1 block x 1024 threads) writes N bytes
//     into dst memory over P2P, fences, sets a flag; dst kernel waits for the flag and answers with a
//     flag in src memory. Per-iteration round trip timed with %globaltimer (ns). N = 16/64/256 KB.
//     Reported: median and p99 round trip (us), and median one-way estimate (= RT - flag-only RT/2 ... see notes).
// (2) all-to-all under load: all 3 GPUs simultaneously write to both peers (kernel P2P stores, and
//     separately copy-engine cudaMemcpyPeerAsync), at 64 KB and 20 MB. Per-GPU achieved send bandwidth.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void k_ping(int4 * remote_buf, size_t n4, volatile int * remote_flag, volatile int * local_flag, int iters, unsigned long long * rt) {
    __shared__ unsigned long long t0;
    for (int t = 1; t <= iters; ++t) {
        if (threadIdx.x == 0) t0 = gtime();
        __syncthreads();
        for (size_t i = threadIdx.x; i < n4; i += blockDim.x) remote_buf[i] = make_int4(t, t, t, t);
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            *remote_flag = t;
            __threadfence_system();
            while (*local_flag != t) { }
            rt[t - 1] = gtime() - t0;
        }
        __syncthreads();
    }
}

__global__ void k_pong(volatile int * local_flag, volatile int * remote_flag, int iters) {
    for (int t = 1; t <= iters; ++t) {
        while (*local_flag != t) { }
        *remote_flag = t;
        __threadfence_system();
    }
}

__global__ void k_store2(int4 * a, int4 * b, size_t n4, int v) {
    // first half of the grid writes peer a, second half peer b
    const int half = gridDim.x / 2;
    int4 * dst = blockIdx.x < half ? a : b;
    const int bid = blockIdx.x < half ? blockIdx.x : blockIdx.x - half;
    for (size_t i = bid * (size_t) blockDim.x + threadIdx.x; i < n4; i += (size_t) half * blockDim.x) dst[i] = make_int4(v, v, v, v);
}

int main() {
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    for (int i = 0; i < nd; ++i) for (int j = 0; j < nd; ++j) if (i != j) {
        CK(cudaSetDevice(i)); cudaError_t pe = cudaDeviceEnablePeerAccess(j, 0);
        if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) CK(pe); cudaGetLastError();
    }
    const size_t maxb = 20 << 20;
    std::vector<void *> rx(nd * nd, nullptr);  // rx[d*nd+s]: landing buffer on d for sender s
    std::vector<int *> flag(nd);
    std::vector<unsigned long long *> rt(nd);
    std::vector<cudaStream_t> st(nd), st2(nd);
    const int iters = 1000;
    for (int d = 0; d < nd; ++d) {
        CK(cudaSetDevice(d));
        for (int s = 0; s < nd; ++s) if (s != d) CK(cudaMalloc(&rx[d * nd + s], maxb));
        CK(cudaMalloc(&flag[d], 64 * nd));
        CK(cudaMalloc(&rt[d], iters * sizeof(unsigned long long)));
        CK(cudaStreamCreateWithFlags(&st[d], cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&st2[d], cudaStreamNonBlocking));
    }
    std::vector<void *> src(nd);
    for (int d = 0; d < nd; ++d) { CK(cudaSetDevice(d)); CK(cudaMalloc(&src[d], maxb)); CK(cudaMemset(src[d], d + 1, maxb)); }
    CK(cudaDeviceSynchronize());

    printf("test,src,dst,bytes,median_us,p99_us\n");
    const size_t sizes[] = { 0, 16 * 1024, 64 * 1024, 256 * 1024 };
    for (int s = 0; s < nd; ++s) for (int d = 0; d < nd; ++d) {
        if (s == d) continue;
        for (size_t nb : sizes) {
            CK(cudaSetDevice(s)); CK(cudaMemset(flag[s], 0, 64 * nd)); CK(cudaDeviceSynchronize());
            CK(cudaSetDevice(d)); CK(cudaMemset(flag[d], 0, 64 * nd)); CK(cudaDeviceSynchronize());
            k_pong<<<1, 1, 0, st[d]>>>((volatile int *) flag[d], (volatile int *) flag[s], iters);
            CK(cudaGetLastError());
            CK(cudaSetDevice(s));
            k_ping<<<1, 1024, 0, st[s]>>>((int4 *) rx[d * nd + s], nb / 16, (volatile int *) flag[d], (volatile int *) flag[s], iters, rt[s]);
            CK(cudaGetLastError());
            CK(cudaStreamSynchronize(st[s]));
            CK(cudaSetDevice(d)); CK(cudaStreamSynchronize(st[d]));
            std::vector<unsigned long long> h(iters);
            CK(cudaSetDevice(s)); CK(cudaMemcpy(h.data(), rt[s], iters * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            std::vector<double> v(h.begin() + 50, h.end());  // drop warmup iterations
            std::sort(v.begin(), v.end());
            printf("write_flag_rt,%d,%d,%zu,%.3f,%.3f\n", s, d, nb, v[v.size() / 2] / 1e3, v[(size_t) (v.size() * 0.99)] / 1e3);
        }
    }

    printf("\ntest,mode,bytes_per_peer,gpu,median_us,send_GBps,per_link_GBps\n");
    const size_t a2a[] = { 64 * 1024, 20 << 20 };
    for (size_t nb : a2a) for (int mode = 0; mode < 2; ++mode) {
        std::vector<std::vector<double>> tms(nd);
        for (int rep = 0; rep < 20; ++rep) {
            std::vector<cudaEvent_t> a(nd), b(nd);
            for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaEventCreate(&a[g])); CK(cudaEventCreate(&b[g])); }
            CK(cudaDeviceSynchronize());
            for (int g = 0; g < nd; ++g) {
                CK(cudaSetDevice(g));
                const int p1 = (g + 1) % nd, p2 = (g + 2) % nd;
                CK(cudaEventRecord(a[g], st[g]));
                if (mode == 0) {
                    k_store2<<<2 * 112, 256, 0, st[g]>>>((int4 *) rx[p1 * nd + g], (int4 *) rx[p2 * nd + g], nb / 16, rep);
                    CK(cudaGetLastError());
                } else {
                    CK(cudaMemcpyPeerAsync(rx[p1 * nd + g], p1, src[g], g, nb, st[g]));
                    CK(cudaMemcpyPeerAsync(rx[p2 * nd + g], p2, src[g], g, nb, st2[g]));
                    cudaEvent_t j; CK(cudaEventCreateWithFlags(&j, cudaEventDisableTiming));
                    CK(cudaEventRecord(j, st2[g])); CK(cudaStreamWaitEvent(st[g], j, 0));
                }
                CK(cudaEventRecord(b[g], st[g]));
            }
            for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaEventSynchronize(b[g])); float ms; CK(cudaEventElapsedTime(&ms, a[g], b[g])); if (rep >= 3) tms[g].push_back((double) ms * 1000.0); }
        }
        for (int g = 0; g < nd; ++g) {
            std::sort(tms[g].begin(), tms[g].end()); const double med = tms[g][tms[g].size() / 2];
            printf("all_to_all,%s,%zu,%d,%.2f,%.3f,%.3f\n", mode == 0 ? "k_write" : "ce_push", nb, g, med, 2.0 * nb / med / 1e3, nb / med / 1e3);
        }
    }
    printf("# note: all_to_all reports the median of 17 reps per GPU; GPUs are launched back to back from one host thread, so start skew of a few us is included at 64 KB\n");
    return 0;
}
