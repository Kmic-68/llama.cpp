// p100_dma.cuh: a decode layer's missing experts copied by the card's copy engine (FNX_P100_DMA=1), beside the row kernels of the ones in VRAM, so the copy takes no SM away from them
// (the side-stream copy kernel did, and lost at T = 2). The plan's list goes to a mailbox in mapped host memory (k_dma_publish: each expert's host copies and slot), a host thread
// of the card's issues two cudaMemcpyAsync an expert on a stream of its own and then writes the list's number into a word in VRAM (cuStreamWriteValue32); k_dma_wait spins on that
// word before the copied experts' rows. mikro/over_dma_test.cu measured it on one layer: 36-52 us less with 1-4 experts missing, the same bits.
#pragma once
#include <cuda.h>
#include <sched.h>

#include <cctype>
#include <cstring>
#include <string>

#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <vector>

#include "p100_over.hpp"

namespace fnx {
namespace {

#include "p100_dma_box.cuh"
struct DmaCard {
    int dev = 0;
    DmaBox* box = nullptr;      // host address
    DmaBox* box_d = nullptr;    // its device address
    DmaBox* box2 = nullptr;     // FNX_PREFETCH: the next layer's guessed experts, copied on cs2
    DmaBox* box2_d = nullptr;
    cudaStream_t cs2 = nullptr;
    DmaDev* d = nullptr;
    cudaStream_t cs = nullptr;
    std::thread th;
    std::atomic<bool> stop{false};
    // FNX_PF_REFILL=1: the prompt path's slots after a read, filled between the lists a few experts at a time, on a stream of their own
    cudaStream_t fs = nullptr;
    std::mutex fill_mu;
    std::vector<DmaFill> fill;
    size_t fill_k = 0, fill_seen = 0;   // issued: [0, fill_k); seen landed: [0, fill_seen)
    std::atomic<bool> fill_on{false};
    std::atomic<int> landed[EC_POOLS];  // each pool's count landed (DmaFill::done), read by the engine before each window
};

__global__ void __launch_bounds__(32) k_dma_publish(ExpertCache ec, int p, const int* __restrict__ list, DmaBox* box, DmaDev* d, size_t gu_bytes, size_t dn_bytes) {
    const int n = min(list[0], DMA_MAX), lane = threadIdx.x;
    for (int k = lane; k < n; k += 32) {
        const int g = list[1 + k];
        char* at = ec.pool[p] + (size_t) (ec.slot_of[g] - ec.s0[p]) * ec.slot_bytes[p];
        box->ent[k] = DmaEnt{ec.host_gu[g], at, ec.host_dn[g], at + ec.dn_off[p]};
    }
    __syncwarp();
    if (lane == 0) {
        if (n == 0) {
            d->want = 0;
            return;
        }
        box->n = n;
        box->gu_bytes = gu_bytes;
        box->dn_bytes = dn_bytes;
        const unsigned s = ++d->seq;
        d->want = s;
        __threadfence_system();
        box->seq = s;
        __threadfence_system();
    }
}

__global__ void __launch_bounds__(32) k_dma_wait(DmaDev* d) {
    if (threadIdx.x != 0) return;
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const long long t0 = clock64();
    while ((int) (*(volatile unsigned*) &d->flag - want) < 0) {
        if (clock64() - t0 > 4000000000LL) {   // ~3 s: the host thread is gone
            atomicOr(&d->err, 1u);
            break;
        }
    }
    __threadfence();
}

// FNX_PREFETCH: k_dma_publish's twin for a layer's guessed experts (box2), published after the layer before it, and the wait that layer's
// window does before its first phase (the copies have had the attention half's time; want2 0: nothing was guessed out of a slot)
__device__ __forceinline__ void dma_publish2(const ExpertCache& ec, int p, const int* __restrict__ list, DmaBox* box, DmaDev* d, size_t gu_bytes, size_t dn_bytes) {
    const int n = min(list[0], DMA_MAX), lane = threadIdx.x;
    for (int k = lane; k < n; k += 32) {
        const int g = list[1 + k];
        char* at = ec.pool[p] + (size_t) (ec.slot_of[g] - ec.s0[p]) * ec.slot_bytes[p];
        box->ent[k] = DmaEnt{ec.host_gu[g], at, ec.host_dn[g], at + ec.dn_off[p]};
    }
    __syncwarp();
    if (lane == 0) {
        if (n == 0) {
            d->want2 = 0;
            return;
        }
        box->n = n;
        box->gu_bytes = gu_bytes;
        box->dn_bytes = dn_bytes;
        const unsigned s = ++d->seq2;
        d->want2 = s;
        __threadfence_system();
        box->seq = s;
        __threadfence_system();
    }
}
__global__ void __launch_bounds__(32) k_dma_publish2(ExpertCache ec, int p, const int* __restrict__ list, DmaBox* box, DmaDev* d, size_t gu_bytes, size_t dn_bytes) {
    dma_publish2(ec, p, list, box, d, gu_bytes, dn_bytes);
}
// FNX_FOLD 4: k_pf_plan and k_dma_publish2 in one launch
__global__ void __launch_bounds__(32) k_pf_plan_publish(const int* __restrict__ guess, int T, int layer, ExpertCache ec, const int* __restrict__ want, int* list, DmaBox* box, DmaDev* d,
                                                        size_t gu_bytes, size_t dn_bytes) {
    pf_plan(guess, T, layer, ec, want, list);
    __syncwarp();
    dma_publish2(ec, ec.pool_of[layer], list, box, d, gu_bytes, dn_bytes);
}

__global__ void __launch_bounds__(32) k_dma_wait2(DmaDev* d) {
    if (threadIdx.x != 0) return;
    const unsigned want = *(volatile unsigned*) &d->want2;
    if (want == 0) return;
    const long long t0 = clock64();
    while ((int) (*(volatile unsigned*) &d->flag2 - want) < 0) {
        if (clock64() - t0 > 4000000000LL) {
            atomicOr(&d->err, 2u);
            break;
        }
    }
    __threadfence();
}

// FNX_NOC_FILE split: the waits for a list's gate|up parts only (flag_gu / flag2_gu, written after them)
__global__ void __launch_bounds__(32) k_dma_wait_gu(DmaDev* d, int second) {
    if (threadIdx.x != 0) return;
    const unsigned want = second ? *(volatile unsigned*) &d->want2 : *(volatile unsigned*) &d->want;
    if (want == 0) return;
    const volatile unsigned* flag = second ? &d->flag2_gu : &d->flag_gu;
    const long long t0 = clock64();
    while ((int) (*flag - want) < 0) {
        if (clock64() - t0 > 4000000000LL) {
            atomicOr(&d->err, second ? 2u : 1u);
            break;
        }
    }
    __threadfence();
}

void dma_pin(int dev) {   // the card's own CPU (its NUMA node), as the engine's card threads
    char id[64] = {};
    if (cudaDeviceGetPCIBusId(id, sizeof id, dev) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    for (char* c = id; *c; ++c) *c = (char) std::tolower((unsigned char) *c);
    std::ifstream f(std::string("/sys/bus/pci/devices/") + id + "/local_cpulist");
    std::string s;
    if (!f || !std::getline(f, s)) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    size_t i = 0;
    while (i < s.size()) {
        const size_t j = s.find(',', i);
        const std::string part = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str()), b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) CPU_SET(c, &set);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    sched_setaffinity(0, sizeof set, &set);
}

// FNX_PF_REFILL=1, between the lists: once the last batch has landed its counts go to the engine (Engine::pf_refill_tick) and the next few
// experts are issued (FNX_PF_REFILL_BATCH, 4), so a list's copies wait behind one small batch at most; off when all have landed
void dma_fill_step(DmaCard* c) {
    const cudaError_t q = cudaStreamQuery(c->fs);
    if (q == cudaErrorNotReady) return;
    std::lock_guard<std::mutex> g(c->fill_mu);
    if (q != cudaSuccess) {   // the slots stay lent until the next read lends them again
        std::fprintf(stderr, "fnx: FNX_PF_REFILL: copy failed (%s), the slots stay lent\n", cudaGetErrorString(q));
        c->fill.clear();
        c->fill_k = c->fill_seen = 0;
        c->fill_on.store(false, std::memory_order_release);
        return;
    }
    for (; c->fill_seen < c->fill_k; ++c->fill_seen) {
        const DmaFill& f = c->fill[c->fill_seen];
        c->landed[f.pool].store(f.done, std::memory_order_release);
    }
    if (c->fill_k >= c->fill.size()) {
        c->fill_on.store(false, std::memory_order_release);
        return;
    }
    static const size_t batch = [] {
        const char* e = std::getenv("FNX_PF_REFILL_BATCH");
        return (size_t) std::max(1, e ? std::atoi(e) : 4);
    }();
    for (const size_t end = std::min(c->fill.size(), c->fill_k + batch); c->fill_k < end; ++c->fill_k) {
        const DmaFill& f = c->fill[c->fill_k];
        cudaMemcpyAsync(f.dst_gu, f.src_gu, f.gu_bytes, cudaMemcpyHostToDevice, c->fs);
        cudaMemcpyAsync(f.dst_dn, f.src_dn, f.dn_bytes, cudaMemcpyHostToDevice, c->fs);
    }
}
// the card's copier: spins while lists come (decode), sleeps 100 us between looks after 20 ms without one
void dma_loop(DmaCard* c) {
    cudaSetDevice(c->dev);
    dma_pin(c->dev);
    unsigned last = 0, last2 = 0;
    auto idle_since = std::chrono::steady_clock::now();
    while (!c->stop.load(std::memory_order_relaxed)) {
        const unsigned s = c->box->seq;
        if (s == last) {
            if (const unsigned s2 = c->box2->seq; s2 != last2) {   // FNX_PREFETCH: guessed experts, after any missing ones
                std::atomic_thread_fence(std::memory_order_acquire);
                const int n2 = std::min(c->box2->n, DMA_MAX);
                const size_t gb2 = c->box2->gu_bytes, db2 = c->box2->dn_bytes;
                DmaEnt e2[DMA_MAX];
                for (int k = 0; k < n2; ++k) e2[k] = c->box2->ent[k];
                for (int k = 0; k < n2; ++k) cudaMemcpyAsync(e2[k].dst_gu, e2[k].src_gu, gb2, cudaMemcpyHostToDevice, c->cs2);   // gate|up first
                cuStreamWriteValue32((CUstream) c->cs2, (CUdeviceptr) &reinterpret_cast<DmaDev*>(c->d)->flag2_gu, s2, 0);
                for (int k = 0; k < n2; ++k) cudaMemcpyAsync(e2[k].dst_dn, e2[k].src_dn, db2, cudaMemcpyHostToDevice, c->cs2);
                cuStreamWriteValue32((CUstream) c->cs2, (CUdeviceptr) &reinterpret_cast<DmaDev*>(c->d)->flag2, s2, 0);
                last2 = s2;
                idle_since = std::chrono::steady_clock::now();
                continue;
            }
            if (c->fill_on.load(std::memory_order_acquire)) {   // FNX_PF_REFILL: awake while the slots are filled
                dma_fill_step(c);
                idle_since = std::chrono::steady_clock::now();
                continue;
            }
            if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(20)) std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const int n = std::min(c->box->n, DMA_MAX);
        const size_t gb = c->box->gu_bytes, db = c->box->dn_bytes;
        DmaEnt e[DMA_MAX];
        for (int k = 0; k < n; ++k) e[k] = c->box->ent[k];
        DmaDev* dd = reinterpret_cast<DmaDev*>(c->d);
        for (int k = 0; k < n; ++k) {   // gate|up first, each part's landing in gu_done[k] (split 2)
            cudaMemcpyAsync(e[k].dst_gu, e[k].src_gu, gb, cudaMemcpyHostToDevice, c->cs);
            cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &dd->gu_done[k], s, 0);
        }
        cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &dd->flag_gu, s, 0);
        for (int k = 0; k < n; ++k) {
            cudaMemcpyAsync(e[k].dst_dn, e[k].src_dn, db, cudaMemcpyHostToDevice, c->cs);
            cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &dd->dn_done[k], s, 0);
        }
        cuStreamWriteValue32((CUstream) c->cs, (CUdeviceptr) &reinterpret_cast<DmaDev*>(c->d)->flag, s, 0);
        last = s;
        idle_since = std::chrono::steady_clock::now();
    }
}

}  // namespace

bool p100_dma_on() {
    static const bool on = [] {
        const char* e = std::getenv("FNX_P100_DMA");
        return e && std::atoi(e) != 0;
    }();
    return on;
}

void* p100_dma_start(int dev) {
    auto* c = new DmaCard;
    c->dev = dev;
    if (cudaHostAlloc((void**) &c->box, sizeof(DmaBox), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &c->box_d, c->box, 0) != cudaSuccess || cudaMalloc(&c->d, sizeof(DmaDev)) != cudaSuccess ||
        cudaMemset(c->d, 0, sizeof(DmaDev)) != cudaSuccess || cudaStreamCreateWithFlags(&c->cs, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error(std::string("FNX_P100_DMA: ") + cudaGetErrorString(cudaGetLastError()));
    std::memset((void*) c->box, 0, sizeof(DmaBox));
    if (cudaHostAlloc((void**) &c->box2, sizeof(DmaBox), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &c->box2_d, c->box2, 0) != cudaSuccess || cudaStreamCreateWithFlags(&c->cs2, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error(std::string("FNX_PREFETCH: ") + cudaGetErrorString(cudaGetLastError()));
    std::memset((void*) c->box2, 0, sizeof(DmaBox));
    if (cudaStreamCreateWithFlags(&c->fs, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error(std::string("FNX_PF_REFILL: ") + cudaGetErrorString(cudaGetLastError()));
    for (auto& l : c->landed) l.store(0);
    c->th = std::thread(dma_loop, c);
    return c;
}

void p100_dma_stop(void* h) {
    auto* c = (DmaCard*) h;
    if (!c) return;
    p100_dma_fill_stop(h);
    c->stop = true;
    if (c->th.joinable()) c->th.join();
    cudaSetDevice(c->dev);
    cudaStreamSynchronize(c->cs);
    cudaStreamDestroy(c->cs);
    cudaStreamDestroy(c->fs);
    cudaStreamSynchronize(c->cs2);
    cudaStreamDestroy(c->cs2);
    cudaFreeHost((void*) c->box2);
    cudaFree(c->d);
    cudaFreeHost((void*) c->box);
    delete c;
}

void p100_dma_fill(void* h, std::vector<DmaFill> jobs, const int* landed) {
    auto* c = (DmaCard*) h;
    std::lock_guard<std::mutex> g(c->fill_mu);
    c->fill = std::move(jobs);
    c->fill_k = c->fill_seen = 0;
    for (int p = 0; p < EC_POOLS; ++p) c->landed[p].store(landed[p], std::memory_order_release);
    c->fill_on.store(true, std::memory_order_release);
}

void p100_dma_fill_stop(void* h) {   // nothing more issued, and what was issued has landed
    auto* c = (DmaCard*) h;
    if (!c) return;
    {
        std::lock_guard<std::mutex> g(c->fill_mu);
        c->fill.clear();
        c->fill_k = c->fill_seen = 0;
        c->fill_on.store(false, std::memory_order_release);
    }
    cudaStreamSynchronize(c->fs);
}

void p100_dma_landed(void* h, int* out) {
    auto* c = (DmaCard*) h;
    for (int p = 0; p < EC_POOLS; ++p) out[p] = c->landed[p].load(std::memory_order_acquire);
}
void p100_dma_publish2(void* h, int layer, const ExpertCache& ec, const int* list, size_t gu_stride, size_t dn_stride, void* st) {
    auto* c = (DmaCard*) h;
    k_dma_publish2<<<1, 32, 0, (cudaStream_t) st>>>(ec, ec.pool_of[layer], list, c->box2_d, c->d, gu_stride, dn_stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dma publish2: ") + cudaGetErrorString(e));
}

void* p100_dma_box(void* h) { return ((DmaCard*) h)->box_d; }
void* p100_dma_dev(void* h) { return ((DmaCard*) h)->d; }
void p100_pf_plan_publish(void* h, const int* guess, int T, int layer, const ExpertCache& ec, const int* want, int* list, size_t gu_stride, size_t dn_stride, void* st) {
    auto* c = (DmaCard*) h;
    k_pf_plan_publish<<<1, 32, 0, (cudaStream_t) st>>>(guess, T, layer, ec, want, list, c->box2_d, c->d, gu_stride, dn_stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("pf plan publish: ") + cudaGetErrorString(e));
}

void p100_dma_wait2(void* h, void* st) {
    auto* c = (DmaCard*) h;
    k_dma_wait2<<<1, 32, 0, (cudaStream_t) st>>>(c->d);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dma wait2: ") + cudaGetErrorString(e));
}

void p100_dma_publish(void* h, int layer, const ExpertCache& ec, const int* list, size_t gu_stride, size_t dn_stride, void* st) {
    auto* c = (DmaCard*) h;
    const int p = ec.pool_of[layer];
    k_dma_publish<<<1, 32, 0, (cudaStream_t) st>>>(ec, p, list, c->box_d, c->d, gu_stride, dn_stride);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dma publish: ") + cudaGetErrorString(e));
}

const unsigned* p100_dma_done(void* h, int part) {
    auto* c = (DmaCard*) h;
    return part ? c->d->dn_done : c->d->gu_done;
}
const unsigned* p100_dma_want_ptr(void* h) { return &((DmaCard*) h)->d->want; }
unsigned* p100_dma_err_ptr(void* h) { return &((DmaCard*) h)->d->err; }

void p100_dma_wait_gu(void* h, void* st) {
    auto* c = (DmaCard*) h;
    k_dma_wait_gu<<<1, 32, 0, (cudaStream_t) st>>>(c->d, 0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dma wait gu: ") + cudaGetErrorString(e));
}

void p100_dma_wait2_gu(void* h, void* st) {
    auto* c = (DmaCard*) h;
    k_dma_wait_gu<<<1, 32, 0, (cudaStream_t) st>>>(c->d, 1);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dma wait2 gu: ") + cudaGetErrorString(e));
}

void p100_dma_wait(void* h, void* st) {
    auto* c = (DmaCard*) h;
    k_dma_wait<<<1, 32, 0, (cudaStream_t) st>>>(c->d);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("dma wait: ") + cudaGetErrorString(e));
}

}  // namespace fnx
