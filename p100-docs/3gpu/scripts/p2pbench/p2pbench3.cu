// Out-of-tree 3-GPU AllReduce shootout (round 2; not part of the fork's source).
//
// Algorithms (sum of one partial vector per GPU, result on all GPUs):
//   a  meta: the meta backend's generic butterfly replicated with peer copies (cudaMemcpyPeerAsync on the
//            source stream, event, add kernel on the destination): fold 2->0, swap 0<->1, copy 0->2.
//   b  oneshot: one kernel per GPU; every GPU writes its whole partial into both peers, flags, then sums the
//            three partials in fixed order (x0 + x1) + x2 locally.
//   c  twoshot: reduce-scatter (each GPU owns a third and sums it from 3 sources in fixed order) + all-gather.
//   d  ring: 2 reduce-scatter steps + 2 all-gather steps around 0->1->2->0.
// Wire: f32 or f16 (b-d convert to f16 before sending; (a) converts with a kernel before each copy).
// Per-block flags only (block b on one GPU talks to block b on the peers), no intra-grid sync.
//
// Timing, per iteration (iterations enqueued 32 at a time behind a 30 ms GPU sleep kernel, so host enqueue cost is excluded):
//   e2e:    cudaEvents on GPU0. Start = after GPU0's stream has waited for all three GPUs to be ready, end =
//           after it has waited for all three to finish. Includes launches and cross-device event waits.
//   kernel: (b-d only) %globaltimer inside the kernel, from the end of a start barrier between the three GPUs
//           to the last block's end, max over GPUs. Excludes launch latency.
// Correctness: after timing, all three outputs are compared bitwise (must be identical for b-d) and against
// a double-precision host sum.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t err_ = (x); if (err_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(err_)); exit(1); } } while (0)

constexpr int NG = 3;
constexpr int MAXB = 64;
constexpr int NKIND = 6;          // 0 barrier, 1..4 data phases
constexpr int NSLOT = 4;          // receive slots per GPU (by source GPU for b/c, by step for d)
constexpr int THREADS = 512;

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

struct WF32 { using T = float4;
    static __device__ __forceinline__ T pack(float4 v) { return v; }
    static __device__ __forceinline__ float4 unpack(T w) { return w; } };
struct WF16 { using T = uint2;
    static __device__ __forceinline__ T pack(float4 v) {
        __half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
        return make_uint2(*reinterpret_cast<unsigned *>(&a), *reinterpret_cast<unsigned *>(&b)); }
    static __device__ __forceinline__ float4 unpack(T w) {
        __half2 a = *reinterpret_cast<__half2 *>(&w.x), b = *reinterpret_cast<__half2 *>(&w.y);
        float2 fa = __half22float2(a), fb = __half22float2(b);
        return make_float4(fa.x, fa.y, fb.x, fb.y); } };

__device__ __forceinline__ float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

template <typename T> __device__ __forceinline__ T ldcg(const T * p);
template <> __device__ __forceinline__ float4 ldcg<float4>(const float4 * p) { return __ldcg(p); }
template <> __device__ __forceinline__ uint2  ldcg<uint2>(const uint2 * p) { return __ldcg(p); }

struct Ctx {                       // per GPU, pointers valid on that GPU (peer pointers via UVA)
    const float4 * part;           // local partial, n4 groups of 4 floats
    float4 * out;                  // result
    void * rx[NG];                 // rx[p] = receive area on GPU p (NSLOT * n4 wire groups)
    int * flag[NG];                // flag[p] = flag array on GPU p: [kind][slot][block]
    unsigned long long * tt;       // [2*MAXB]: per-block start/end globaltimer
};

__device__ __forceinline__ int fidx(int kind, int slot, int b) { return (kind * NSLOT + slot) * MAXB + b; }

__device__ __forceinline__ void signal(int * remote_flags, int kind, int slot, int b, int it) {
    // caller: all threads' stores issued; fence then flag from thread 0
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) { *((volatile int *) remote_flags + fidx(kind, slot, b)) = it; }
}
__device__ __forceinline__ void wait_flag(int * local_flags, int kind, int slot, int b, int it) {
    if (threadIdx.x == 0) { while (*((volatile int *) local_flags + fidx(kind, slot, b)) < it) { } }
    __syncthreads();
    __threadfence();
}

__device__ __forceinline__ void block_range(long n4, int nb, long & lo, long & hi) {
    const long per = (n4 + nb - 1) / nb;
    lo = min(n4, (long) blockIdx.x * per); hi = min(n4, lo + per);
}
__device__ __forceinline__ void seg_range(long lo, long hi, int s, long & a, long & z) {
    const long len = hi - lo, per = (len + NG - 1) / NG;
    a = min(hi, lo + s * per); z = min(hi, a + per);
}

__device__ __forceinline__ void start_barrier(const Ctx & c, int g, int b, int it) {
    if (threadIdx.x == 0) {
        for (int p = 0; p < NG; ++p) if (p != g) *((volatile int *) c.flag[p] + fidx(0, g, b)) = it;
        __threadfence_system();
        for (int p = 0; p < NG; ++p) if (p != g) while (*((volatile int *) c.flag[g] + fidx(0, p, b)) < it) { }
        c.tt[b] = gtime();
    }
    __syncthreads();
}
__device__ __forceinline__ void stamp_end(const Ctx & c, int b) {
    __syncthreads();
    if (threadIdx.x == 0) c.tt[MAXB + b] = gtime();
}

template <typename W>
__global__ void __launch_bounds__(THREADS) k_oneshot(Ctx c, int g, long n4, int it) {
    using T = typename W::T;
    const int b = blockIdx.x; long lo, hi; block_range(n4, gridDim.x, lo, hi);
    start_barrier(c, g, b, it);
    for (long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const T w = W::pack(c.part[i]);
        for (int p = 0; p < NG; ++p) if (p != g) ((T *) c.rx[p])[(long) g * n4 + i] = w;
    }
    for (int p = 0; p < NG; ++p) if (p != g) { signal(c.flag[p], 1, g, b, it); }
    for (int p = 0; p < NG; ++p) if (p != g) wait_flag(c.flag[g], 1, p, b, it);
    const T * rx = (const T *) c.rx[g];
    for (long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float4 x[NG];
        for (int s = 0; s < NG; ++s) x[s] = s == g ? W::unpack(W::pack(c.part[i])) : W::unpack(ldcg(rx + (long) s * n4 + i));
        float4 acc = x[0]; acc = add4(acc, x[1]); acc = add4(acc, x[2]);
        c.out[i] = acc;
    }
    stamp_end(c, b);
}

template <typename W>
__global__ void __launch_bounds__(THREADS) k_twoshot(Ctx c, int g, long n4, int it) {
    using T = typename W::T;
    const int b = blockIdx.x; long lo, hi; block_range(n4, gridDim.x, lo, hi);
    start_barrier(c, g, b, it);
    // reduce-scatter: send my partial of segment p to its owner p (slot g)
    for (int p = 0; p < NG; ++p) if (p != g) {
        long a, z; seg_range(lo, hi, p, a, z);
        for (long i = a + threadIdx.x; i < z; i += blockDim.x) ((T *) c.rx[p])[(long) g * n4 + i] = W::pack(c.part[i]);
    }
    for (int p = 0; p < NG; ++p) if (p != g) signal(c.flag[p], 1, g, b, it);
    for (int p = 0; p < NG; ++p) if (p != g) wait_flag(c.flag[g], 1, p, b, it);
    const T * rx = (const T *) c.rx[g];
    long a, z; seg_range(lo, hi, g, a, z);
    for (long i = a + threadIdx.x; i < z; i += blockDim.x) {
        float4 x[NG];
        for (int s = 0; s < NG; ++s) x[s] = s == g ? c.part[i] : W::unpack(ldcg(rx + (long) s * n4 + i));
        float4 acc = x[0]; acc = add4(acc, x[1]); acc = add4(acc, x[2]);
        const T w = W::pack(acc);
        c.out[i] = W::unpack(w);   // owner keeps the wire-rounded value so all GPUs end bit-identical
        for (int p = 0; p < NG; ++p) if (p != g) ((T *) c.rx[p])[(long) g * n4 + i] = w;   // all-gather: own segment, slot g
    }
    for (int p = 0; p < NG; ++p) if (p != g) signal(c.flag[p], 2, g, b, it);
    for (int p = 0; p < NG; ++p) if (p != g) {
        wait_flag(c.flag[g], 2, p, b, it);
        long a2, z2; seg_range(lo, hi, p, a2, z2);
        for (long i = a2 + threadIdx.x; i < z2; i += blockDim.x) c.out[i] = W::unpack(ldcg(rx + (long) p * n4 + i));
    }
    stamp_end(c, b);
}

template <typename W>
__global__ void __launch_bounds__(THREADS) k_ring(Ctx c, int g, long n4, int it) {
    using T = typename W::T;
    const int b = blockIdx.x; long lo, hi; block_range(n4, gridDim.x, lo, hi);
    start_barrier(c, g, b, it);
    const int nx = (g + 1) % NG;
    const T * rx = (const T *) c.rx[g];
    T * tx = (T *) c.rx[nx];
    long a, z;
    // RS step 0: send raw partial of segment (g+2)%3 to next (slot 0)
    seg_range(lo, hi, (g + 2) % NG, a, z);
    for (long i = a + threadIdx.x; i < z; i += blockDim.x) tx[0 * n4 + i] = W::pack(c.part[i]);
    signal(c.flag[nx], 1, 0, b, it);
    // RS step 1: receive segment (g+1)%3 (prev's (prev+2)%3), add own, send to next (slot 1)
    wait_flag(c.flag[g], 1, 0, b, it);
    seg_range(lo, hi, (g + 1) % NG, a, z);
    for (long i = a + threadIdx.x; i < z; i += blockDim.x) tx[1 * n4 + i] = W::pack(add4(W::unpack(ldcg(rx + 0 * n4 + i)), c.part[i]));
    signal(c.flag[nx], 2, 1, b, it);
    // owner: receive two-way sum of segment g, add own -> full sum; AG step 0: send to next (slot 2)
    wait_flag(c.flag[g], 2, 1, b, it);
    seg_range(lo, hi, g, a, z);
    for (long i = a + threadIdx.x; i < z; i += blockDim.x) {
        const T w = W::pack(add4(W::unpack(ldcg(rx + 1 * n4 + i)), c.part[i]));
        c.out[i] = W::unpack(w);
        tx[2 * n4 + i] = w;
    }
    signal(c.flag[nx], 3, 2, b, it);
    // AG step 1: receive segment (g+2)%3 (prev's own), store, forward to next (slot 3)
    wait_flag(c.flag[g], 3, 2, b, it);
    seg_range(lo, hi, (g + 2) % NG, a, z);
    for (long i = a + threadIdx.x; i < z; i += blockDim.x) { const T w = ldcg(rx + 2 * n4 + i); c.out[i] = W::unpack(w); tx[3 * n4 + i] = w; }
    signal(c.flag[nx], 4, 3, b, it);
    // receive segment (g+1)%3
    wait_flag(c.flag[g], 4, 3, b, it);
    seg_range(lo, hi, (g + 1) % NG, a, z);
    for (long i = a + threadIdx.x; i < z; i += blockDim.x) c.out[i] = W::unpack(ldcg(rx + 3 * n4 + i));
    stamp_end(c, b);
}

// (a) helpers
__global__ void k_add_f32(float4 * dst, const float4 * x, const float4 * y, long n4) {
    for (long i = blockIdx.x * (long) blockDim.x + threadIdx.x; i < n4; i += (long) gridDim.x * blockDim.x) dst[i] = add4(x[i], y[i]);
}
__global__ void k_add_f16(float4 * dst, const float4 * x, const uint2 * y, long n4) {
    for (long i = blockIdx.x * (long) blockDim.x + threadIdx.x; i < n4; i += (long) gridDim.x * blockDim.x) dst[i] = add4(x[i], WF16::unpack(y[i]));
}
__global__ void k_to_f16(uint2 * dst, const float4 * x, long n4) {
    for (long i = blockIdx.x * (long) blockDim.x + threadIdx.x; i < n4; i += (long) gridDim.x * blockDim.x) dst[i] = WF16::pack(x[i]);
}
__global__ void k_from_f16(float4 * dst, const uint2 * x, long n4) {
    for (long i = blockIdx.x * (long) blockDim.x + threadIdx.x; i < n4; i += (long) gridDim.x * blockDim.x) dst[i] = WF16::unpack(x[i]);
}

struct Dev {
    float4 * part; float4 * out; void * rx; int * flag; unsigned long long * tt;
    float4 * tmp; float4 * tmp2; uint2 * stage; uint2 * rstage; uint2 * rstage2;     // for (a)
    cudaStream_t st;
};

static Dev D[NG];
static long g_maxn4;

static void setup(long maxn4) {
    g_maxn4 = maxn4;
    for (int i = 0; i < NG; ++i) for (int j = 0; j < NG; ++j) if (i != j) {
        CK(cudaSetDevice(i)); cudaError_t pe = cudaDeviceEnablePeerAccess(j, 0);
        if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) CK(pe); cudaGetLastError();
    }
    for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        CK(cudaMalloc(&D[g].part, maxn4 * 16)); CK(cudaMalloc(&D[g].out, maxn4 * 16));
        CK(cudaMalloc(&D[g].rx, (size_t) NSLOT * maxn4 * 16));
        CK(cudaMalloc(&D[g].flag, NKIND * NSLOT * MAXB * sizeof(int))); CK(cudaMemset(D[g].flag, 0, NKIND * NSLOT * MAXB * sizeof(int)));
        CK(cudaMalloc(&D[g].tt, 32 * 2 * MAXB * sizeof(unsigned long long)));
        CK(cudaMalloc(&D[g].tmp, maxn4 * 16)); CK(cudaMalloc(&D[g].tmp2, maxn4 * 16));
        CK(cudaMalloc(&D[g].stage, maxn4 * 8)); CK(cudaMalloc(&D[g].rstage, maxn4 * 8)); CK(cudaMalloc(&D[g].rstage2, maxn4 * 8));
        CK(cudaStreamCreateWithFlags(&D[g].st, cudaStreamNonBlocking));
    }
}

static Ctx ctx_of(int g) {
    Ctx c; c.part = D[g].part; c.out = D[g].out; c.tt = D[g].tt;
    for (int p = 0; p < NG; ++p) { c.rx[p] = D[p].rx; c.flag[p] = D[p].flag; }
    return c;
}

static int g_epoch = 0;

// enqueue one iteration of algorithm alg on all GPUs (after the caller's start waits)
static void enqueue(char alg, bool f16, long n4, int nb, int slot) {
    const int it = ++g_epoch;
    const int ab = (int) std::min<long>(224, (n4 + 255) / 256);
    if (alg == 'a') {
        // buffers: tmp/tmp2 = f32 receive A/B, stage = f16 send, rstage/rstage2 = f16 receive A/B
        cudaEvent_t e1, e2, e3, e4;
        // stage 1: fold 2 -> 0
        CK(cudaSetDevice(2));
        if (!f16) CK(cudaMemcpyPeerAsync(D[0].tmp, 0, D[2].part, 2, n4 * 16, D[2].st));
        else { k_to_f16<<<ab, 256, 0, D[2].st>>>(D[2].stage, D[2].part, n4); CK(cudaMemcpyPeerAsync(D[0].rstage, 0, D[2].stage, 2, n4 * 8, D[2].st)); }
        CK(cudaEventCreateWithFlags(&e1, cudaEventDisableTiming)); CK(cudaEventRecord(e1, D[2].st));
        CK(cudaSetDevice(0)); CK(cudaStreamWaitEvent(D[0].st, e1, 0));
        if (!f16) k_add_f32<<<ab, 256, 0, D[0].st>>>(D[0].out, D[0].part, D[0].tmp, n4);
        else k_add_f16<<<ab, 256, 0, D[0].st>>>(D[0].out, D[0].part, D[0].rstage, n4);
        // stage 2: swap 0 <-> 1 (the 1 -> 0 copy depends only on GPU1 being ready)
        CK(cudaSetDevice(1));
        if (!f16) CK(cudaMemcpyPeerAsync(D[0].tmp2, 0, D[1].part, 1, n4 * 16, D[1].st));
        else { k_to_f16<<<ab, 256, 0, D[1].st>>>(D[1].stage, D[1].part, n4); CK(cudaMemcpyPeerAsync(D[0].rstage2, 0, D[1].stage, 1, n4 * 8, D[1].st)); }
        CK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming)); CK(cudaEventRecord(e2, D[1].st));
        CK(cudaSetDevice(0));
        if (!f16) CK(cudaMemcpyPeerAsync(D[1].tmp, 1, D[0].out, 0, n4 * 16, D[0].st));
        else { k_to_f16<<<ab, 256, 0, D[0].st>>>(D[0].stage, D[0].out, n4); CK(cudaMemcpyPeerAsync(D[1].rstage, 1, D[0].stage, 0, n4 * 8, D[0].st)); }
        CK(cudaEventCreateWithFlags(&e3, cudaEventDisableTiming)); CK(cudaEventRecord(e3, D[0].st));
        CK(cudaStreamWaitEvent(D[0].st, e2, 0));
        // (0 + 2) + 1 on GPU0; 1 + (0 + 2) on GPU1 (a single add commutes exactly)
        if (!f16) k_add_f32<<<ab, 256, 0, D[0].st>>>(D[0].out, D[0].out, D[0].tmp2, n4);
        else k_add_f16<<<ab, 256, 0, D[0].st>>>(D[0].out, D[0].out, D[0].rstage2, n4);
        CK(cudaSetDevice(1)); CK(cudaStreamWaitEvent(D[1].st, e3, 0));
        if (!f16) k_add_f32<<<ab, 256, 0, D[1].st>>>(D[1].out, D[1].part, D[1].tmp, n4);
        else k_add_f16<<<ab, 256, 0, D[1].st>>>(D[1].out, D[1].part, D[1].rstage, n4);
        // stage 3: copy back 0 -> 2
        CK(cudaSetDevice(0));
        if (!f16) CK(cudaMemcpyPeerAsync(D[2].out, 2, D[0].out, 0, n4 * 16, D[0].st));
        else { k_to_f16<<<ab, 256, 0, D[0].st>>>(D[0].stage, D[0].out, n4); CK(cudaMemcpyPeerAsync(D[2].rstage, 2, D[0].stage, 0, n4 * 8, D[0].st)); }
        CK(cudaEventCreateWithFlags(&e4, cudaEventDisableTiming)); CK(cudaEventRecord(e4, D[0].st));
        CK(cudaSetDevice(2)); CK(cudaStreamWaitEvent(D[2].st, e4, 0));
        if (f16) k_from_f16<<<ab, 256, 0, D[2].st>>>(D[2].out, D[2].rstage, n4);
        CK(cudaEventDestroy(e1)); CK(cudaEventDestroy(e2)); CK(cudaEventDestroy(e3)); CK(cudaEventDestroy(e4));
        return;
    }
    for (int g = 0; g < NG; ++g) {
        CK(cudaSetDevice(g));
        Ctx c = ctx_of(g); c.tt += (size_t) slot * 2 * MAXB;
        if (alg == 'b') { if (f16) k_oneshot<WF16><<<nb, THREADS, 0, D[g].st>>>(c, g, n4, it); else k_oneshot<WF32><<<nb, THREADS, 0, D[g].st>>>(c, g, n4, it); }
        if (alg == 'c') { if (f16) k_twoshot<WF16><<<nb, THREADS, 0, D[g].st>>>(c, g, n4, it); else k_twoshot<WF32><<<nb, THREADS, 0, D[g].st>>>(c, g, n4, it); }
        if (alg == 'd') { if (f16) k_ring<WF16><<<nb, THREADS, 0, D[g].st>>>(c, g, n4, it); else k_ring<WF32><<<nb, THREADS, 0, D[g].st>>>(c, g, n4, it); }
        CK(cudaGetLastError());
    }
}

struct Res { double e2e_med, e2e_p99, k_med, k_p99; };

static double pct(std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t) (q * v.size()))]; }

__global__ void k_sleep(unsigned long long ns) { const unsigned long long t0 = gtime(); while (gtime() - t0 < ns) { } }


// Iterations are enqueued in batches of BATCH behind a 30 ms GPU-side sleep kernel on GPU0, so the whole batch is
// enqueued before it starts and host enqueue cost is not part of the measured GPU-side latency (NOHOLD=1 disables).
static Res measure(char alg, bool f16, long n4, int nb, int iters, int skip) {
    const int BATCH = 32;
    std::vector<cudaEvent_t> ea(iters), eb(iters);
    CK(cudaSetDevice(0));
    for (int i = 0; i < iters; ++i) { CK(cudaEventCreate(&ea[i])); CK(cudaEventCreate(&eb[i])); }
    std::vector<double> ke;
    std::vector<unsigned long long> h((size_t) BATCH * 2 * MAXB);
    for (int i0 = 0; i0 < iters; i0 += BATCH) {
        const int i1 = std::min(iters, i0 + BATCH);
        static const bool nohold = getenv("NOHOLD") != nullptr;
        if (!nohold) { CK(cudaSetDevice(0)); k_sleep<<<1, 1, 0, D[0].st>>>(30000000ull); CK(cudaGetLastError()); }
        std::vector<cudaEvent_t> tmpev;
        for (int i = i0; i < i1; ++i) {
            // start: GPU1 and GPU2 ready (after the previous iteration ended everywhere), GPU0 waits for them
            for (int g = 1; g < NG; ++g) { cudaEvent_t r; CK(cudaSetDevice(g)); CK(cudaEventCreateWithFlags(&r, cudaEventDisableTiming)); CK(cudaEventRecord(r, D[g].st));
                CK(cudaSetDevice(0)); CK(cudaStreamWaitEvent(D[0].st, r, 0)); tmpev.push_back(r); }
            CK(cudaSetDevice(0)); CK(cudaEventRecord(ea[i], D[0].st));
            for (int g = 1; g < NG; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamWaitEvent(D[g].st, ea[i], 0)); }
            enqueue(alg, f16, n4, nb, i - i0);
            for (int g = 1; g < NG; ++g) { cudaEvent_t d; CK(cudaSetDevice(g)); CK(cudaEventCreateWithFlags(&d, cudaEventDisableTiming)); CK(cudaEventRecord(d, D[g].st));
                CK(cudaSetDevice(0)); CK(cudaStreamWaitEvent(D[0].st, d, 0)); tmpev.push_back(d); }
            CK(cudaSetDevice(0)); CK(cudaEventRecord(eb[i], D[0].st));
            for (int g = 1; g < NG; ++g) { CK(cudaSetDevice(g)); CK(cudaStreamWaitEvent(D[g].st, eb[i], 0)); }
        }
        CK(cudaSetDevice(0)); CK(cudaEventSynchronize(eb[i1 - 1]));
        for (auto e : tmpev) CK(cudaEventDestroy(e));
        if (alg != 'a') {
            std::vector<double> worst(i1 - i0, 0.0);
            for (int g = 0; g < NG; ++g) {
                CK(cudaSetDevice(g)); CK(cudaMemcpy(h.data(), D[g].tt, h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                for (int i = i0; i < i1; ++i) {
                    const unsigned long long * t = h.data() + (size_t) (i - i0) * 2 * MAXB;
                    unsigned long long t0 = ~0ull, t1 = 0;
                    for (int b = 0; b < nb; ++b) { t0 = std::min(t0, t[b]); t1 = std::max(t1, t[MAXB + b]); }
                    worst[i - i0] = std::max(worst[i - i0], (double) (t1 - t0) / 1e3);
                }
            }
            for (int i = i0; i < i1; ++i) if (i >= skip) ke.push_back(worst[i - i0]);
        }
    }
    std::vector<double> e;
    for (int i = skip; i < iters; ++i) { float ms; CK(cudaEventElapsedTime(&ms, ea[i], eb[i])); e.push_back(ms * 1e3); }
    for (int i = 0; i < iters; ++i) { CK(cudaEventDestroy(ea[i])); CK(cudaEventDestroy(eb[i])); }
    Res r; r.e2e_med = pct(e, 0.5); r.e2e_p99 = pct(e, 0.99);
    r.k_med = ke.empty() ? NAN : pct(ke, 0.5); r.k_p99 = ke.empty() ? NAN : pct(ke, 0.99);
    return r;
}

int main(int argc, char ** argv) {
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    if (nd < NG) { fprintf(stderr, "need 3 GPUs\n"); return 1; }
    const bool quick = argc > 1 && !strcmp(argv[1], "quick");
    std::vector<long> sizes = { 16 << 10, 20 << 10, 64 << 10, 128 << 10, 256 << 10, 4 << 20, 20 << 20, 40 << 20 };  // f32 bytes of the vector
    if (quick) sizes = { 16 << 10, 4 << 20 };
    const long maxn4 = (40 << 20) / 16;
    setup(maxn4);
    // inputs: random signs and magnitudes over 2^-8..2^8 so summation order changes the rounding
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-1.f, 1.f); std::uniform_int_distribution<int> ex(-8, 8);
    std::vector<std::vector<float>> hp(NG, std::vector<float>(maxn4 * 4));
    for (int g = 0; g < NG; ++g) for (auto & x : hp[g]) x = ldexpf(u(rng), ex(rng));
    for (int g = 0; g < NG; ++g) { CK(cudaSetDevice(g)); CK(cudaMemcpy(D[g].part, hp[g].data(), maxn4 * 16, cudaMemcpyHostToDevice)); }
    const int nbs[] = { 2, 4, 8, 16, 28, 56 };
    printf("alg,wire,bytes_f32,wire_bytes,blocks,e2e_med_us,e2e_p99_us,kernel_med_us,kernel_p99_us,algbw_GBps,identical,max_rel_err\n");
    fflush(stdout);
    for (long bytes : sizes) {
        const long n4 = bytes / 16;
        const bool small = bytes <= (256 << 10);
        const int iters = quick ? 50 : (small ? 2000 : 60), skip = quick ? 5 : (small ? 100 : 6);
        for (int wf = 0; wf < 2; ++wf) for (char alg : { 'a', 'b', 'c', 'd' }) {
            const bool f16 = wf == 1;
            int best_nb = 0;
            if (alg != 'a') {   // pick the block count by a short pre-run (median e2e)
                double bt = 1e30;
                for (int nb : nbs) {
                    if (nb > MAXB || (long) nb * 3 > n4) continue;
                    if (getenv("VERBOSE")) fprintf(stderr, "pre %c f16=%d n4=%ld nb=%d\n", alg, (int) f16, n4, nb);
                    Res r = measure(alg, f16, n4, nb, small ? 200 : 12, small ? 20 : 2);
                    if (r.e2e_med < bt) { bt = r.e2e_med; best_nb = nb; }
                }
            }
            if (getenv("VERBOSE")) fprintf(stderr, "measure %c f16=%d n4=%ld nb=%d\n", alg, (int) f16, n4, best_nb);
            Res r = measure(alg, f16, n4, best_nb, iters, skip);
            // correctness
            std::vector<std::vector<float>> o(NG, std::vector<float>(n4 * 4));
            for (int g = 0; g < NG; ++g) { CK(cudaSetDevice(g)); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(o[g].data(), D[g].out, n4 * 16, cudaMemcpyDeviceToHost)); }
            const bool ident = !memcmp(o[0].data(), o[1].data(), n4 * 16) && !memcmp(o[0].data(), o[2].data(), n4 * 16);
            double mre = 0;
            for (long i = 0; i < n4 * 4; ++i) {
                const double ref = (double) hp[0][i] + hp[1][i] + hp[2][i];
                double err = 0; for (int g = 0; g < NG; ++g) err = std::max(err, fabs(o[g][i] - ref));
                mre = std::max(mre, err / (fabs((double) hp[0][i]) + fabs((double) hp[1][i]) + fabs((double) hp[2][i]) + 1e-30));
            }
            const long wire = f16 ? bytes / 2 : bytes;
            printf("%c,%s,%ld,%ld,%d,%.2f,%.2f,%.2f,%.2f,%.3f,%d,%.3g\n", alg, f16 ? "f16" : "f32", bytes, wire, best_nb,
                   r.e2e_med, r.e2e_p99, r.k_med, r.k_p99, bytes / r.e2e_med / 1e3, ident ? 1 : 0, mre);
            fflush(stdout);
        }
    }
    printf("# max_rel_err = max over elements and GPUs of |out - double sum| / (|x0|+|x1|+|x2|)\n");
    printf("# algbw = f32 vector bytes / e2e median. blocks = best of {2,4,8,16,28,56} by a short pre-run (b-d)\n");
    return 0;
}
