#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <vector>
#include <cstdint>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <memory>

struct llama_model;
struct llama_device;
struct llama_model_tensor_buft_override;

// GPU-resident LRU expert pools: the decode expert matmuls read VRAM
// pools, the LRU bookkeeping and fills run as graph nodes. activated by
// --moe-lru-slots; the pooled layer set follows the placement parameters
struct llama_moe_lru {
    static constexpr int N_KIND = 3; // 0 = gate, 1 = up, 2 = down
    // graph nodes the pooled path adds per layer: decide, 2 views, 3 fills
    static constexpr int NODES_PER_LAYER = 6;
    enum kind { KIND_GATE = 0, KIND_UP = 1, KIND_DOWN = 2 };

    // rewrite the expert overrides to the pinned host buffer; call before
    // the expert tensors are created. storage must outlive the loading phase
    static bool init_model_overrides(struct llama_model_loader & ml, const std::vector<llama_device> & devices,
                                     std::vector<llama_model_tensor_buft_override> & storage, int n_slot);

    // LLAMA_MOE_LRU_EXCLUSIVE=1: each expert lives either in a VRAM pool slot or in a host slot of the context (misses swap)
    static bool exclusive_requested();
    // LLAMA_MOE_LRU_REFILL=1 (implies exclusive): a miss drops the victim; a host thread refills its host slot from the mapped
    // model weights (no write-back from the GPU)
    static bool refill_requested();

    ~llama_moe_lru();
    // disk refill control block shared with the backend (nullptr unless refill is active)
    struct ggml_lru_ctrl * ctrl() const { return refill_ctrl; }
    int pool_index(int il) const { return pooled(il) ? layer_to_pool[il] : -1; }
    bool is_exclusive() const { return exclusive; }

    // lookahead: router tensors of a pooled layer (predict its experts from the previous layer's input)
    ggml_tensor * router_get(int il, int which) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].router[which]; }
    // the next pooled layer after il, or -1
    int next_pooled(int il) const {
        for (int nx = il + 1; nx < (int) layer_to_pool.size(); nx++) { if (pooled(nx)) return nx; }
        return -1;
    }

    // true if the device can run the LRU ops; capability probe instead of
    // device-type filtering, so meta devices pass through (tensor split)
    static bool backend_supports(ggml_backend_dev_t dev);

    llama_moe_lru(const llama_model & model, const std::vector<ggml_backend_t> & gpu_backends, int n_slot);

    // il is the model layer number; the pooled set may be any subset (-ot)
    bool pooled(int il) const { return il >= 0 && il < (int) layer_to_pool.size() && layer_to_pool[il] >= 0; }
    int n_pooled() const { return n_layers; }
    int n_slots() const { return n_slot; }   // the uniform default (--moe-lru-slots)
    // this layer's slots: the default unless LLAMA_MOE_LRU_SLOTS=<file> gives the layer its own budget
    int n_slots(int il) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].n_slot; }
    // the smallest per-layer budget: the decide-vs-assemble choice uses it so every layer of a batch takes the SAME path (LB1: a per-layer
    // choice mixed the paths for 13-38-token batches and the next layer's prefetched assembly stayed live -> a 6.9 GiB compute reserve)
    int n_slots_min() const { return n_slot_min; }

    // large batches assemble the experts from the pools on the GPU (LLAMA_MOE_LRU_ASSEMBLE=0: scheduler upload of every expert)
    bool assemble() const { return use_assemble; }
    // build the next pooled layer's assembly one layer ahead (opt-in: LLAMA_MOE_LRU_PREFETCH=1)
    bool prefetch() const { return use_assemble && use_prefetch; }

    // per-layer and total counters at teardown (hit rates)
    void log_final_stats();

    // host-computed misses (LLAMA_MOE_LRU_HOST=<fills per layer and pass>, exclusive cache): beyond that many fills a decode
    // miss is computed by a CPU thread pool from its host slot while the GPU computes the cached experts
    bool host_enabled() const { return host_max_fill >= 0; }
    int  host_max_fill_get() const { return host_max_fill; }
    int  host_every_get() const { return host_every; }
    struct ggml_lru_host_req * host_req(int il) const { GGML_ASSERT(pooled(il)); return host_reqs.empty() ? nullptr : host_reqs[layer_to_pool[il]]; }
    int  host_writer() const { return n_split > 1 ? 1 : 2; }

    // call only when pooled(il)
    ggml_tensor * state_get(int il) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].state; }
    ggml_tensor * pool_get(int il, int kind) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].pool[kind]; }
    ggml_tensor * exps_get(int il, int kind) const { GGML_ASSERT(pooled(il)); return layers[layer_to_pool[il]].exps[kind]; }

private:
    struct per_layer {
        ggml_backend_buffer_type_t buft = nullptr; // this layer's device buft
        int n_slot = 0;                            // this layer's VRAM slots (per-layer budget or the default)
        ggml_tensor * pool[N_KIND] = {};   // [ne0, ne1, n_slot] in VRAM
        ggml_tensor * exps[N_KIND] = {};  // the kernels' source: expert weights in pinned host memory (exclusive: = hpool)
        ggml_tensor * hpool[N_KIND] = {}; // exclusive: [ne0, ne1, n_expert - n_slot] pinned host slots
        ggml_tensor * mexps[N_KIND] = {}; // refill: the model's (mapped) expert tensors, the refill source
        ggml_tensor * router[3] = {};     // ffn_gate_inp, ffn_gate_inp_b, ffn_exp_probs_b
        std::vector<int32_t> host_expert; // exclusive: host slot -> expert at start
        ggml_tensor * state = nullptr;     // i32 [n_expert + 2*n_slot + 3]
                                           // (owner | slot_of | last_use | tick | cum_ids | cum_fills)
    };

    int n_layers = 0, n_slot = 0, n_slot_min = 0;
    bool exclusive = false;
    int n_split = 1;   // devices an expert is sliced over (-sm tensor): the host pool keeps each device's slices contiguous
    // copy expert e of layer L (kind k) from the mapped model into host slot hs, scattered into the per-device regions
    void host_slot_store(const struct per_layer & L, int k, int32_t e, int64_t hs) const;

    // disk refill
    struct ggml_lru_ctrl * refill_ctrl = nullptr;
    std::thread refill_thread;
    std::atomic<bool> refill_stop{false};
    void refill_loop();
    // out-of-order completion: refill copies run on workers, swap-only entries are published as soon as they are ready
    struct refill_task { int32_t li, v, hs; uint32_t idx; };
    std::vector<std::thread> refill_workers;
    std::mutex refill_mtx;
    std::condition_variable refill_cv;
    std::deque<refill_task> refill_queue;
    std::unique_ptr<std::atomic<uint8_t>[]> refill_processed;   // per ring index: finished (tail may pass it)
    void refill_worker();
    // host-computed misses
    int host_max_fill = -1;
    int host_every = 0;   // LLAMA_MOE_LRU_HOST_EVERY
    std::vector<struct ggml_lru_host_req *> host_reqs;   // per pooled layer, in device-readable host memory
    std::vector<std::thread> host_threads;
    std::atomic<bool> host_stop{false};
    struct host_work;
    std::unique_ptr<host_work> hw;
    void host_init(ggml_backend_buffer_type_t host_buft, int64_t n_embd, int n_used);
    void host_dispatch_loop();
    void host_worker_loop(int w);
    void host_process(int li);
    bool use_assemble = true;
    bool use_prefetch = false;
    int64_t n_expert = 0;

    // model layer -> index into layers, -1 when not pooled
    std::vector<int> layer_to_pool;

    std::vector<ggml_context_ptr> pool_ctxs; // per-layer pool/state tensor objects
    std::vector<ggml_backend_buffer_ptr> bufs;
    std::vector<per_layer> layers;
};
