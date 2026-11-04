#include "llama-moe-lru.h"
#include "llama.h"

#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml-cpu.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <cmath>
#include <functional>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

bool llama_moe_lru::init_model_overrides(llama_model_loader & ml, const std::vector<llama_device> & devices,
                                     std::vector<llama_model_tensor_buft_override> & storage, int n_slot) {
    if (n_slot <= 0) {
        return false; // LRU disabled, keep the overrides as-is
    }
    if (exclusive_requested()) {
        return false; // exclusive cache: the context copies the experts out of the (mapped) model tensors, no pinning
    }

    ggml_backend_buffer_type_t pinned_buft = nullptr;
    for (const auto & d : devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(d.dev, &props);
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(d.dev);
        if (host && host != ggml_backend_cpu_buffer_type()) {
            pinned_buft = host;
            break;
        }
    }
    if (!pinned_buft) {
        // -sm tensor: the model's device list holds the meta device, which has no host buffer type -> ask the registry
        for (size_t i = 0; i < ggml_backend_dev_count() && !pinned_buft; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            ggml_backend_buffer_type_t host = ggml_backend_dev_host_buffer_type(dev);
            if (host && host != ggml_backend_cpu_buffer_type()) {
                pinned_buft = host;
            }
        }
    }
    if (!pinned_buft) {
        return false; // no GPU backend with a pinned host buffer in this build
    }

    const llama_model_tensor_buft_override * cur = ml.tensor_buft_overrides;
    if (!cur) {
        return false;
    }

    std::vector<llama_model_tensor_buft_override> & transformed = storage;
    transformed.clear();
    int n_pinned = 0;
    for (const auto * o = cur; o->pattern != nullptr; ++o) {
        transformed.push_back(*o);
        if (o->buft == ggml_backend_cpu_buffer_type() && strstr(o->pattern, "exps") != nullptr) {
            transformed.back().buft = pinned_buft;
            n_pinned++;
        }
    }
    if (n_pinned == 0) {
        return false;
    }

    transformed.push_back({ nullptr, nullptr });
    ml.tensor_buft_overrides = transformed.data();
    LLAMA_LOG_INFO("llama_moe_lru: %d expert weight override(s) redirected to the pinned host buffer\n", n_pinned);
    return true;
}

// LLAMA_MOE_LRU_PROFILE=<file>: one line per model layer, "il e0 e1 e2 ..." with the layer's experts most used first
// (tools: bench-tools/expertcache/make_lru_profile.py from a LLAMA_MOE_TRACE). The pool starts with each layer's top n_slot.
static const std::map<int, std::vector<int32_t>> & llama_moe_lru_profile() {
    static const std::map<int, std::vector<int32_t>> prof = [] {
        std::map<int, std::vector<int32_t>> m;
        const char * path = getenv("LLAMA_MOE_LRU_PROFILE");
        if (path == nullptr || *path == 0) {
            return m;
        }
        FILE * f = fopen(path, "r");
        if (f == nullptr) {
            LLAMA_LOG_WARN("llama_moe_lru: cannot open profile %s\n", path);
            return m;
        }
        char line[1 << 16];
        while (fgets(line, sizeof(line), f)) {
            char * p = line;
            char * end = nullptr;
            const long il = strtol(p, &end, 10);
            if (end == p) {
                continue;
            }
            p = end;
            std::vector<int32_t> v;
            while (true) {
                const long e = strtol(p, &end, 10);
                if (end == p) break;
                v.push_back((int32_t) e);
                p = end;
            }
            m[(int) il] = std::move(v);
        }
        fclose(f);
        LLAMA_LOG_INFO("llama_moe_lru: profile %s (%zu layers)\n", path, m.size());
        return m;
    }();
    return prof;
}

// LLAMA_MOE_LRU_SLOTS=<file> (2026-10-21, gate EL1/slot-budget simulation): per-layer VRAM slot budgets, one "<layer> <slots>" pair per
// line (# comments allowed); layers not listed keep --moe-lru-slots. Misses concentrate in a few layers (Flash-Next: layer 0 85 % hit,
// late layers 99 %), so the same VRAM spread unevenly saves fills (bench-tools expertcache/slotbudget.py sizes it from a trace).
static const std::map<int, int> & llama_moe_lru_budget() {
    static const std::map<int, int> budget = [] {
        std::map<int, int> m;
        const char * path = getenv("LLAMA_MOE_LRU_SLOTS");
        if (path == nullptr || *path == 0) {
            return m;
        }
        FILE * f = fopen(path, "r");
        if (f == nullptr) {
            throw std::runtime_error(std::string("llama_moe_lru: cannot open slot budget file ") + path);
        }
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '#') {
                continue;
            }
            int il = -1, ns = -1;
            if (sscanf(line, "%d %d", &il, &ns) == 2 && il >= 0 && ns > 0) {
                m[il] = ns;
            }
        }
        fclose(f);
        LLAMA_LOG_INFO("llama_moe_lru: slot budgets %s (%zu layers)\n", path, m.size());
        return m;
    }();
    return budget;
}

// exclusive host pool layout: for kind k, device d's slices of all host slots are contiguous ([d][host slot][slice]) so a
// prefill assembly moves them in one transfer and the fills read contiguous slices. Slices follow the meta split of the
// experts: gate/up by rows (ne[1]), down by columns (ne[0], each row's d-th segment).
void llama_moe_lru::host_slot_store(const per_layer & L, int k, int32_t e, int64_t hs) const {
    const ggml_tensor * m = L.mexps[k];
    const int64_t n_host = L.hpool[k]->ne[2];
    const size_t es = m->nb[2];
    const size_t slice = es / n_split;
    const char * srce = (const char *) m->data + (size_t) e * es;
    char * base = (char *) L.hpool[k]->data;
    for (int d = 0; d < n_split; d++) {
        char * dstp = base + ((size_t) d * n_host + hs) * slice;
        if (n_split == 1 || k != KIND_DOWN) {   // rows: device d holds rows [d*r, (d+1)*r) -- contiguous
            memcpy(dstp, srce + (size_t) d * slice, slice);
        } else {                                // columns: each row's d-th segment
            const size_t seg = m->nb[1] / n_split;
            for (int64_t r = 0; r < m->ne[1]; r++) {
                memcpy(dstp + (size_t) r * seg, srce + (size_t) r * m->nb[1] + (size_t) d * seg, seg);
            }
        }
    }
}

bool llama_moe_lru::refill_requested() {
    const char * e = getenv("LLAMA_MOE_LRU_REFILL");
    return e != nullptr && atoi(e) != 0;
}

bool llama_moe_lru::exclusive_requested() {
    const char * e = getenv("LLAMA_MOE_LRU_EXCLUSIVE");
    return (e != nullptr && atoi(e) != 0) || refill_requested();
}

llama_moe_lru::~llama_moe_lru() {
    if (!host_threads.empty()) {
        host_stop = true;
        for (auto & t : host_threads) {
            t.join();
        }
    }
    if (refill_thread.joinable()) {
        refill_stop = true;
        refill_cv.notify_all();
        refill_thread.join();
        for (auto & w : refill_workers) {
            w.join();
        }
    }
}

// entries complete out of order: one is ready when every device finished the fill of the step that freed its slot (the
// slot still held the incoming expert until then). A swap-only entry (every device wrote the victim back) is published at once,
// a refill entry goes to a worker that copies the victim from the mapped model and then publishes. The tail passes finished
// entries only. Two pending entries never share a slot: a slot is reassigned only after its occupant was loaded back into a
// pool, which waited for the occupant's publication.
void llama_moe_lru::refill_worker() {
    ggml_lru_ctrl * c = refill_ctrl;
    while (true) {
        refill_task t;
        {
            std::unique_lock<std::mutex> lk(refill_mtx);
            refill_cv.wait(lk, [this] { return refill_stop.load() || !refill_queue.empty(); });
            if (refill_queue.empty()) {
                return;   // stop requested
            }
            t = refill_queue.front();
            refill_queue.pop_front();
        }
        const per_layer & L = layers[t.li];
        for (int k = 0; k < N_KIND; k++) {
            host_slot_store(L, k, t.v, t.hs);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        ((volatile int32_t *) c->content)[(size_t) t.li * c->n_host + t.hs] = t.v;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        refill_processed[t.idx % c->cap].store(1, std::memory_order_release);
    }
}

void llama_moe_lru::refill_loop() {
    ggml_lru_ctrl * c = refill_ctrl;
    uint32_t tail = c->tail, scan = tail;
    struct waiting { int32_t li, v, hs, step; uint32_t idx; };
    std::vector<waiting> wait_list;
    while (!refill_stop.load(std::memory_order_relaxed)) {
        bool progress = false;
        const uint32_t head = c->head;
        for (; scan != head; scan++) {   // new entries (device 0 writes the entry before advancing head)
            const int32_t * en = c->entries + 4 * (size_t) (scan % c->cap);
            wait_list.push_back({ en[0], en[1], en[2], en[3], scan });
            progress = true;
        }
        for (size_t i = 0; i < wait_list.size(); ) {
            const waiting & w = wait_list[i];
            bool ready = true;
            for (uint32_t d = 0; d < c->n_dev && ready; d++) {
                ready = (int64_t) ((volatile uint32_t *) c->done)[(size_t) d * c->n_layer + w.li] >= 3 * (int64_t) w.step;
            }
            if (!ready) {
                i++;
                continue;
            }
            // every device's decide for this step ran before its fill, so its mode for the entry is written
            bool any_left = false;
            for (uint32_t d = 0; d < c->n_dev; d++) {
                any_left |= ((volatile int32_t *) c->mode)[(size_t) d * c->cap + w.idx % c->cap] == 0;
            }
            if (any_left) {
                {
                    std::lock_guard<std::mutex> lk(refill_mtx);
                    refill_queue.push_back({ w.li, w.v, w.hs, w.idx });
                }
                refill_cv.notify_one();
            } else {
                std::atomic_thread_fence(std::memory_order_seq_cst);
                ((volatile int32_t *) c->content)[(size_t) w.li * c->n_host + w.hs] = w.v;
                std::atomic_thread_fence(std::memory_order_seq_cst);
                refill_processed[w.idx % c->cap].store(1, std::memory_order_release);
            }
            wait_list[i] = wait_list.back();
            wait_list.pop_back();
            progress = true;
        }
        while (tail != scan && refill_processed[tail % c->cap].load(std::memory_order_acquire)) {
            refill_processed[tail % c->cap].store(0, std::memory_order_relaxed);
            tail++;
            c->tail = tail;
            progress = true;
        }
        if (!progress) {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }
}

bool llama_moe_lru::backend_supports(ggml_backend_dev_t dev) {
    // capability probe: excludes GPU backends without LRU kernels
    // (e.g. Vulkan); meta devices pass by forwarding to their children
    ggml_init_params ip = { 16 * ggml_tensor_overhead(), nullptr, /*no_alloc =*/ true };
    ggml_context_ptr ctx(ggml_init(ip));
    if (!ctx) {
        return false;
    }
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    ggml_tensor * st  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8);
    ggml_tensor * dec = ggml_lru_decide(ctx.get(), ids, st, 1, 1);
    return ggml_backend_dev_supports_op(dev, dec);
}

llama_moe_lru::llama_moe_lru(const llama_model & model, const std::vector<ggml_backend_t> & gpu_backends, int n_slot) {
    std::unordered_map<ggml_backend_dev_t, ggml_backend_t> backend_of;
    for (ggml_backend_t b : gpu_backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(b);
        if (dev) {
            backend_of[dev] = b;
        }
    }
    if (backend_of.empty()) {
        throw std::runtime_error("llama_moe_lru: no GPU backends given");
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(backend_of.begin()->first);
    for (size_t i = 0; i < ggml_backend_dev_count() && (host_buft == nullptr || host_buft == ggml_backend_cpu_buffer_type()); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            host_buft = ggml_backend_dev_host_buffer_type(d);
        }
    }

    const int64_t n_expert = model.hparams.n_expert;
    const int64_t n_used_h = model.hparams.n_expert_used();
    const int64_t n_layer_tot = model.hparams.n_layer();

    const int max_batch = (int) (n_slot / n_used_h);
    if (max_batch < 1) {
        throw std::runtime_error("llama_moe_lru: pool too small for one token's experts");
    }

    std::vector<int> pooled_ids;
    std::vector<ggml_backend_buffer_type_t> pending_buft;
    for (int il = 0; il < (int) n_layer_tot; il++) {
        const ggml_tensor * up = model.layers[il].ffn_up_exps;
        const ggml_tensor * down = model.layers[il].ffn_down_exps;
        if (!up || !down) continue;
        if (model.layers[il].ffn_gate_up_exps) {
            throw std::runtime_error("llama_moe_lru: merged gate_up experts not supported");
        }
        if (up->buffer && ggml_backend_buffer_is_host(up->buffer)) {
            if (!up->data) {
                throw std::runtime_error("llama_moe_lru: expert weights not mapped");
            }
            // pageable host memory hangs the fill kernel (exclusive: the kernels read the context's pinned host pool instead)
            if (exclusive_requested()) {
                // the experts are copied out byte for byte: a CPU repack buffer (e.g. Q8_0/Q4_0 interleaved) would be garbage
                const char * bn = ggml_backend_buft_name(ggml_backend_buffer_get_type(up->buffer));
                if (bn && strstr(bn, "REPACK")) {
                    throw std::runtime_error("llama_moe_lru: exclusive cache needs plain expert weights, got a repack buffer (use --no-repack)");
                }
            } else if (ggml_backend_buffer_get_type(up->buffer) != host_buft) {
                throw std::runtime_error("llama_moe_lru: expert weights are host but not pinned "
                                         "(use --n-cpu-moe or -ot with --moe-lru-slots and a non-lazy load mode)");
            }
            pooled_ids.push_back(il);
            // pool locality = layer locality: follow the router weight's device
            ggml_backend_dev_t cdev = nullptr;
            const ggml_tensor * gate_inp = model.layers[il].ffn_gate_inp;
            if (gate_inp && gate_inp->buffer) {
                cdev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(gate_inp->buffer));
            }
            if (!backend_of.count(cdev)) {
                cdev = backend_of.begin()->first;
            }
            pending_buft.push_back(ggml_backend_dev_buffer_type(cdev));
        }
    }
    if (pooled_ids.empty()) {
        throw std::runtime_error("llama_moe_lru: no layers with host-side expert weights");
    }

    this->n_layers = (int) pooled_ids.size();
    {
        const char * env = getenv("LLAMA_MOE_LRU_ASSEMBLE");
        use_assemble = env == nullptr || atoi(env) != 0;
        // FZD (10-13): prefetch + side stream = identical text but no faster (the zero-copy assembly competes with the matmuls
        // for the SMs) and ~1.2 GB/GPU fewer pool slots -> opt-in
        const char * envp = getenv("LLAMA_MOE_LRU_PREFETCH");
        use_prefetch = envp != nullptr && atoi(envp) != 0;
    }
    this->n_slot = n_slot;
    this->n_expert = n_expert;
    // per-layer slot budgets (LLAMA_MOE_LRU_SLOTS); every layer needs one token's experts and fewer slots than experts to stay exclusive
    std::vector<int> layer_slots(this->n_layers, n_slot);
    {
        const auto & budget = llama_moe_lru_budget();
        for (int li = 0; li < this->n_layers; li++) {
            auto it = budget.find(pooled_ids[li]);
            if (it != budget.end()) {
                layer_slots[li] = it->second;
            }
            if (layer_slots[li] < n_used_h || layer_slots[li] > n_expert) {
                throw std::runtime_error("llama_moe_lru: layer " + std::to_string(pooled_ids[li]) + " budget " +
                                         std::to_string(layer_slots[li]) + " slots is outside [n_expert_used, n_expert]");
            }
        }
    }
    // slot loan (LLAMA_MOE_LRU_LEND=<slots per layer>): that many slots more per layer, whose memory is also the compute buffer
    // (ggml-cuda maps it twice). Decode uses them; a larger batch empties them first. Needs the exclusive cache with copy-engine
    // or SM fills, no disk refill, no host-computed misses, no victim stash.
    std::vector<int> layer_base(layer_slots);
    {
        const char * el = getenv("LLAMA_MOE_LRU_LEND");
        const int want = el ? atoi(el) : 0;
        const char * es = getenv("GGML_CUDA_LRU_STASH");
        if (want > 0 && (!exclusive_requested() || refill_requested() || getenv("LLAMA_MOE_LRU_HOST") != nullptr ||
                         (es != nullptr && atoi(es) != 0) || lend_fn_map() == nullptr)) {
            LLAMA_LOG_WARN("llama_moe_lru: LLAMA_MOE_LRU_LEND needs the exclusive cache, CUDA, GGML_CUDA_LRU_STASH=0, no disk "
                           "refill and no LLAMA_MOE_LRU_HOST: off\n");
        } else if (want > 0) {
            lend_slots = want;
            for (int li = 0; li < this->n_layers; li++) {
                layer_slots[li] = std::min<int>(layer_base[li] + want, (int) n_expert - 1);
            }
        }
    }
    const int slots_min = *std::min_element(layer_slots.begin(), layer_slots.end());
    const int slots_max = *std::max_element(layer_slots.begin(), layer_slots.end());
    this->n_slot_min = slots_min;
    // exclusive: every expert lives in exactly one place -- the first n_slot of a layer in the VRAM pool, the rest in a pinned
    // host pool of (n_expert - n_slot) slots owned by this context; a miss swaps the incoming expert with the pool's victim
    this->exclusive = exclusive_requested() && slots_max < n_expert;
    const int64_t n_host = this->exclusive ? n_expert - n_slot : 0;   // uniform layers only (disk refill's control block)
    if (this->exclusive && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        int n_gpu = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            n_gpu += ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        this->n_split = std::max(1, n_gpu);
    }

    // model layer -> pool index; the pooled set may be any subset (-ot)
    layer_to_pool.assign(n_layer_tot, -1);
    for (int li = 0; li < this->n_layers; li++) {
        layer_to_pool[pooled_ids[li]] = li;
    }

    layers.resize(this->n_layers);

    for (int li = 0; li < this->n_layers; li++) {
        const int il = pooled_ids[li];
        per_layer & L = layers[li];
        L.buft = pending_buft[li];
        L.n_slot = layer_slots[li];
        L.n_base = layer_base[li];
        const int64_t n_slot = L.n_slot;                          // this layer's slots (shadows the default)
        const int64_t n_base = lend_slots > 0 ? L.n_base : n_slot;   // slot loan: slots >= n_base are borrowed
        const int64_t n_host = this->exclusive ? n_expert - n_base : 0;

        ggml_tensor * exps[N_KIND] = {
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_down_exps,
        };
        L.router[0] = model.layers[il].ffn_gate_inp;
        L.router[1] = model.layers[il].ffn_gate_inp_b;
        L.router[2] = model.layers[il].ffn_exp_probs_b;
        for (int k = 0; k < N_KIND; k++) {
            if (!exps[k] || !exps[k]->data) {
                throw std::runtime_error("llama_moe_lru: expert tensor missing or unmapped");
            }
            GGML_ASSERT(ggml_is_contiguous(exps[k]));
            L.exps[k] = exps[k];
        }

        // standard allocator: correct data pointers and init for meta
        // buffer types; names must be set before alloc, the meta split
        // callback classifies tensors by name
        static const char * kind_name[N_KIND] = { "gate", "up", "down" };
        ggml_context_ptr pctx(ggml_init({ (N_KIND + 2) * ggml_tensor_overhead(), nullptr, true }));
        for (int k = 0; k < N_KIND; k++) {
            ggml_tensor * t = ggml_new_tensor_3d(pctx.get(), exps[k]->type,
                                                 exps[k]->ne[0], exps[k]->ne[1], n_slot);
            ggml_format_name(t, "blk.%d.moe_lru_pool.%s", il, kind_name[k]);
            L.pool[k] = t;
        }
        // +tick +cum_ids +cum_fills (+ exclusive: expert -> host slot)
        // (+ refill: this device's log counter)
        // (+ slot loan: n_base, n_active)
        L.state = ggml_new_tensor_1d(pctx.get(), GGML_TYPE_I32,
                                     n_expert + 2 * (int64_t) n_slot + 3 + (this->exclusive ? 2 * n_expert + 1 : 0) + (lend_slots > 0 ? 2 : 0));
        ggml_format_name(L.state, "blk.%d.moe_lru_state", il);

        if (lend_slots > 0) {
            lend_fn_pools()((int) n_base);   // VMM chunks; the pool tensors record their borrowed chunks
        }
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(pctx.get(), L.buft);
        if (lend_slots > 0) {
            lend_fn_pools()(0);
        }
        if (!b) {
            throw std::runtime_error("llama_moe_lru: pool alloc failed");
        }
        bufs.emplace_back(b);
        pool_ctxs.emplace_back(std::move(pctx));

        // owner = -1, slot owner = -1, last_use = 0, tick = 1, counters = 0
        const int64_t n_state = n_expert + 2 * (int64_t) n_slot + 3 + (this->exclusive ? 2 * n_expert + 1 : 0) + (lend_slots > 0 ? 2 : 0);
        std::vector<int32_t> h(n_state, -1);
        for (int64_t i = n_expert + n_slot; i < n_expert + 2 * n_slot; i++) h[i] = i - n_expert - n_slot < n_base ? 0 : -1;   // borrowed: first
        h[n_expert + 2 * n_slot] = 1;      // tick
        h[n_expert + 2 * n_slot + 1] = 0;  // cum_ids
        h[n_expert + 2 * n_slot + 2] = 0;  // cum_fills
        // initial placement: the profile's most-used experts (LLAMA_MOE_LRU_PROFILE) fill the pool, the most popular in the highest
        // slots (LRU evicts the lowest slot on ties, so the least popular go first); without a profile the exclusive pool takes
        // experts 0..n_slot-1 and the inclusive pool starts empty. Exclusive: every other expert gets a host slot.
        std::vector<int32_t> chosen;
        {
            const auto & prof = llama_moe_lru_profile();
            auto it = prof.find(il);
            std::vector<bool> taken(n_expert, false);
            if (it != prof.end()) {
                for (int32_t e : it->second) {
                    if ((int64_t) chosen.size() >= n_base) break;
                    if (e >= 0 && e < n_expert && !taken[e]) { chosen.push_back(e); taken[e] = true; }
                }
            }
            if (this->exclusive) {   // the exclusive pool must be full: pad with the lowest unused experts
                for (int32_t e = 0; e < n_expert && (int64_t) chosen.size() < n_base; e++) {
                    if (!taken[e]) { chosen.push_back(e); taken[e] = true; }
                }
            }
        }
        {
            int32_t * owner = h.data(), * slot_of = h.data() + n_expert;
            for (size_t r = 0; r < chosen.size(); r++) {
                const int32_t s = (int32_t) (n_base - 1 - (int64_t) r);
                owner[chosen[r]] = s;
                slot_of[s] = chosen[r];
            }
        }
        if (!chosen.empty()) {
            for (int k = 0; k < N_KIND; k++) {
                const size_t es = exps[k]->nb[2];
                std::vector<uint8_t> stage((size_t) n_slot * es, 0);
                for (size_t r = 0; r < chosen.size(); r++) {
                    memcpy(stage.data() + (size_t) (n_base - 1 - (int64_t) r) * es, (const char *) exps[k]->data + (size_t) chosen[r] * es, es);
                }
                // full pool layout; the meta buffer splits it per device
                ggml_backend_tensor_set(L.pool[k], stage.data(), 0, stage.size());
            }
        }
        if (this->exclusive) {
            int32_t * owner = h.data(), * hslot = h.data() + n_expert + 2 * n_slot + 3;
            L.host_expert.clear();
            for (int64_t e = 0; e < n_expert; e++) {
                if (owner[e] >= 0) {
                    hslot[e] = -1;
                } else {
                    hslot[e] = (int32_t) L.host_expert.size();
                    L.host_expert.push_back((int32_t) e);
                }
            }
            GGML_ASSERT((int64_t) L.host_expert.size() == n_host);
            h[2 * n_expert + 2 * n_slot + 3] = 0;   // refill log counter
            if (lend_slots > 0) {
                h[3 * n_expert + 2 * n_slot + 4] = (int32_t) n_base;   // borrowed from here
                h[3 * n_expert + 2 * n_slot + 5] = (int32_t) n_base;   // active: lent out until the context takes them
            }
            for (int64_t hs = 0; hs < n_host; hs++) {
                h[2 * n_expert + 2 * n_slot + 4 + hs] = L.host_expert[hs];   // host slot -> expert
            }

            ggml_context_ptr hctx(ggml_init({ N_KIND * ggml_tensor_overhead(), nullptr, true }));
            for (int k = 0; k < N_KIND; k++) {
                L.hpool[k] = ggml_new_tensor_3d(hctx.get(), exps[k]->type, exps[k]->ne[0], exps[k]->ne[1], n_host);
                ggml_format_name(L.hpool[k], "blk.%d.moe_lru_host.%s", il, kind_name[k]);
            }
            ggml_backend_buffer_t hb = ggml_backend_alloc_ctx_tensors_from_buft(hctx.get(), host_buft);
            if (!hb) {
                throw std::runtime_error("llama_moe_lru: host pool alloc failed");
            }
            bufs.emplace_back(hb);
            pool_ctxs.emplace_back(std::move(hctx));
            for (int k = 0; k < N_KIND; k++) {
                L.mexps[k] = exps[k];
                if (n_split > 1) {
                    GGML_ASSERT(exps[k]->nb[2] % n_split == 0);
                    GGML_ASSERT(k != KIND_DOWN ? exps[k]->ne[1] % n_split == 0 : exps[k]->nb[1] % n_split == 0);
                }
                for (int64_t hs = 0; hs < n_host; hs++) {
                    host_slot_store(L, k, L.host_expert[hs], hs);
                }
                L.exps[k] = L.hpool[k];   // the kernels' source: host slots instead of expert rows
            }
#if defined(__linux__) && defined(MADV_PAGEOUT)
            // the experts now live in the pools; drop their (mmapped) source pages so the page cache does not fill up while the
            // pinned host pools grow: on a model larger than RAM that pressure swapped the desktop out during load. PAGEOUT keeps
            // the data either way (file pages are re-read on demand, anonymous memory goes to swap). Disk refill reads the source
            // again, so it keeps the pages.
            if (!refill_requested()) {
                const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
                for (int k = 0; k < N_KIND; k++) {
                    const uintptr_t a = ((uintptr_t) L.mexps[k]->data + pg - 1) & ~(pg - 1);
                    const uintptr_t b = ((uintptr_t) L.mexps[k]->data + ggml_nbytes(L.mexps[k])) & ~(pg - 1);
                    if (b > a) {
                        madvise((void *) a, b - a, MADV_PAGEOUT);
                    }
                }
            }
#endif
        }
        ggml_backend_tensor_set(L.state, h.data(), 0, h.size() * 4);
    }

    if (lend_slots > 0) {
        const char * eh = getenv("LLAMA_MOE_LRU_LEND_HEAD");   // MiB per device kept for decode graphs (default 768)
        lend_head = (size_t) (eh ? atoi(eh) : 768) << 20;
        lend_arena = lend_fn_map()(lend_head);
        LLAMA_LOG_WARN("llama_moe_lru: slot loan: +%d slots per layer, arena %.1f MiB per device\n", lend_slots, lend_arena / 1048576.0);
    }

    // disk refill: control block in device-readable host memory + the refill thread. The done tally needs every device that
    // fills a layer: all GPUs under -sm tensor, or a single GPU
    if (this->exclusive && refill_requested() && slots_min != slots_max) {
        LLAMA_LOG_WARN("llama_moe_lru: disk refill needs uniform slots (LLAMA_MOE_LRU_SLOTS given), using the write-back swap\n");
    } else if (this->exclusive && refill_requested()) {
        uint32_t n_dev = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            n_dev += ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        if (model.split_mode() != LLAMA_SPLIT_MODE_TENSOR && n_dev != 1) {
            LLAMA_LOG_WARN("llama_moe_lru: disk refill needs -sm tensor or one GPU, using the write-back swap\n");
        } else {
            const uint32_t cap = 16384;
            const size_t sz = sizeof(ggml_lru_ctrl) + 64 + (size_t) cap * 4 * sizeof(int32_t) +
                              (size_t) n_layers * n_host * sizeof(int32_t) + (size_t) 16 * n_layers * sizeof(uint32_t) +
                              (size_t) 16 * cap * sizeof(int32_t);
            ggml_backend_buffer_t cb = ggml_backend_buft_alloc_buffer(host_buft, sz);
            if (!cb) {
                throw std::runtime_error("llama_moe_lru: refill control block alloc failed");
            }
            bufs.emplace_back(cb);
            char * base = (char *) ggml_backend_buffer_get_base(cb);
            memset(base, 0, sz);
            ggml_lru_ctrl * c = (ggml_lru_ctrl *) base;
            char * p = base + ((sizeof(ggml_lru_ctrl) + 63) & ~(size_t) 63);
            c->cap = cap; c->n_layer = (uint32_t) n_layers; c->n_host = (uint32_t) n_host; c->n_dev = n_dev;
            c->entries = (int32_t *) p;  p += (size_t) cap * 4 * sizeof(int32_t);
            c->content = (int32_t *) p;  p += (size_t) n_layers * n_host * sizeof(int32_t);
            c->done    = (uint32_t *) p;  p += (size_t) 16 * n_layers * sizeof(uint32_t);
            c->mode    = (int32_t *) p;
            {
                // hybrid: swap instead of refill while this many entries wait for the host (0 = always swap, huge = never)
                const char * eb = getenv("LLAMA_MOE_LRU_REFILL_BACKLOG");
                c->swap_backlog = eb ? (uint32_t) atoi(eb) : 16u;
            }
            for (int li = 0; li < n_layers; li++) {
                for (int64_t hs = 0; hs < n_host; hs++) {
                    c->content[(size_t) li * n_host + hs] = layers[li].host_expert[hs];
                }
            }
            refill_ctrl = c;
            refill_processed.reset(new std::atomic<uint8_t>[cap]);
            for (uint32_t i = 0; i < cap; i++) {
                refill_processed[i].store(0);
            }
            const char * et = getenv("LLAMA_MOE_LRU_REFILL_THREADS");
            const int n_workers = std::max(1, et ? atoi(et) : 4);
            for (int i = 0; i < n_workers; i++) {
                refill_workers.emplace_back([this] { refill_worker(); });
            }
            refill_thread = std::thread([this] { refill_loop(); });
            LLAMA_LOG_INFO("llama_moe_lru: disk refill on (%u devices, ring %u, swap when backlog >= %u)\n", n_dev, cap, c->swap_backlog);
        }
    }

    {
        const char * eh = getenv("LLAMA_MOE_LRU_HOST");
        if (eh != nullptr && atoi(eh) >= 0) {
            if (!this->exclusive || refill_ctrl != nullptr) {
                LLAMA_LOG_WARN("llama_moe_lru: LLAMA_MOE_LRU_HOST needs the exclusive cache without disk refill, off\n");
            } else {
                host_max_fill = atoi(eh);
                const char * ee = getenv("LLAMA_MOE_LRU_HOST_EVERY");
                host_every = ee ? std::max(0, atoi(ee)) : 0;
                host_init(host_buft, model.hparams.n_embd, (int) n_used_h);
            }
        }
    }

    size_t pool_bytes = 0;
    for (const auto & b : bufs) {
        pool_bytes += ggml_backend_buffer_get_size(b.get());
    }
    int64_t slots_total = 0;
    for (int ns : layer_slots) {
        slots_total += ns;
    }
    LLAMA_LOG_INFO("llama_moe_lru: %d pooled layers x %d slots (per layer %d..%d, total %lld), %d experts, topk %lld, "
                   "max verification batch %d, %.2f GiB%s\n",
                   this->n_layers, n_slot, slots_min, slots_max, (long long) slots_total, (int) n_expert, (long long) n_used_h,
                   (int) (slots_min / n_used_h),
                   pool_bytes / 1073741824.0, this->exclusive ? " (exclusive: incl. the pinned host pool)" : "");
}


void llama_moe_lru::log_final_stats() {
    int64_t total_ids = 0, total_fills = 0;

    // pool index -> model layer number, for the per-layer log lines
    std::vector<int> pool_to_layer(n_layers);
    for (int il = 0; il < (int) layer_to_pool.size(); il++) {
        if (layer_to_pool[il] >= 0) {
            pool_to_layer[layer_to_pool[il]] = il;
        }
    }

    for (int li = 0; li < n_layers; li++) {
        std::vector<int32_t> h(3);
        const int64_t off = (n_expert + 2 * (int64_t) layers[li].n_slot) * 4;
        ggml_backend_tensor_get(layers[li].state, h.data(), off, 3 * 4);
        total_ids += h[1];
        total_fills += h[2];
        if (h[1] > 0) {
            LLAMA_LOG_INFO("llama_moe_lru: layer %d: ids %lld fills %lld hit %.1f%%",
                           pool_to_layer[li], (long long) h[1], (long long) h[2],
                           100.0 * (1.0 - (double) h[2] / h[1]));
        }
    }
    if (total_ids > 0) {
        LLAMA_LOG_INFO("llama_moe_lru: final: %d layers, %lld ids, %lld fills, hit %.2f%%",
                       n_layers, (long long) total_ids, (long long) total_fills,
                       100.0 * (1.0 - (double) total_fills / total_ids));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// host-computed misses (LLAMA_MOE_LRU_HOST). A request names (entry, expert, host slot) triples and carries the layer's MoE
// input; the dispatcher computes up/gate (rows split over the workers), the SwiGLU, then down, from the expert's host slot,
// with ggml-cpu's own dot products (the CPU backend's arithmetic for these types), and writes the unweighted outputs back.

struct llama_moe_lru::host_work {
    int n_workers = 0;
    // one phase at a time: n_jobs row chunks, claimed with next, counted in done; gen announces a phase to the workers
    std::atomic<uint64_t> gen{0};
    std::atomic<int> next{0}, done{0};
    int n_jobs = 0;
    std::function<void(int)> job;
    std::vector<uint32_t> last_seq;   // per pooled layer
    // LLAMA_MOE_LRU_HOST_STATS=1: requests with work, entries, processing time
    bool stats = false;
    uint64_t n_req = 0, n_ent = 0, n_empty = 0;
    double us = 0.0, us_max = 0.0;
    // scratch
    std::vector<uint8_t> qx;          // [n_tok] activations in the gate/up vec_dot type
    std::vector<float>   g, u;        // [n_ent][n_ff]
    std::vector<uint8_t> qh;          // [n_ent] SwiGLU output in the down vec_dot type
};

void llama_moe_lru::host_init(ggml_backend_buffer_type_t host_buft, int64_t n_embd, int n_used) {
    const int max_ent = GGML_LRU_HOST_MAX_TOK * n_used;
    const size_t a = 64;
    const size_t sz_one = ((sizeof(ggml_lru_host_req) + a - 1) / a) * a + ((size_t) max_ent * 3 * sizeof(int32_t) + a - 1) / a * a +
                          (size_t) GGML_LRU_HOST_MAX_TOK * n_embd * sizeof(float) + (size_t) max_ent * n_embd * sizeof(float);
    const size_t sz = sz_one * n_layers + a;
    ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(host_buft, sz);
    if (!b) {
        throw std::runtime_error("llama_moe_lru: host request alloc failed");
    }
    bufs.emplace_back(b);
    char * p = (char *) ggml_backend_buffer_get_base(b);
    memset(p, 0, sz);
    p = (char *) (((uintptr_t) p + a - 1) & ~(uintptr_t) (a - 1));
    host_reqs.resize(n_layers);
    for (int li = 0; li < n_layers; li++) {
        char * q = p + (size_t) li * sz_one;
        auto * r = (ggml_lru_host_req *) q;
        q += ((sizeof(ggml_lru_host_req) + a - 1) / a) * a;
        r->ent = (int32_t *) q; q += ((size_t) max_ent * 3 * sizeof(int32_t) + a - 1) / a * a;
        r->x   = (float *) q;   q += (size_t) GGML_LRU_HOST_MAX_TOK * n_embd * sizeof(float);
        r->out = (float *) q;
        r->n_embd = (int32_t) n_embd;
        r->max_ent = max_ent;
        host_reqs[li] = r;
    }
    // every layer must be computable: types with CPU dot products, down columns split on block boundaries
    for (int li = 0; li < n_layers; li++) {
        for (int k = 0; k < N_KIND; k++) {
            const ggml_tensor * t = layers[li].hpool[k];
            const auto * tr = ggml_get_type_traits_cpu(t->type);
            if (tr->vec_dot == nullptr || ggml_get_type_traits_cpu(tr->vec_dot_type)->from_float == nullptr) {
                throw std::runtime_error(format("llama_moe_lru: no CPU dot product for %s", ggml_type_name(t->type)));
            }
            if (k == KIND_DOWN && (t->ne[0] / n_split) % ggml_blck_size(t->type) != 0) {
                throw std::runtime_error("llama_moe_lru: host misses need block-aligned down slices");
            }
        }
    }
    hw.reset(new host_work());
    hw->last_seq.assign(n_layers, 0);
    hw->stats = getenv("LLAMA_MOE_LRU_HOST_STATS") != nullptr;
    const char * et = getenv("LLAMA_MOE_LRU_HOST_THREADS");
    hw->n_workers = std::max(0, et ? atoi(et) : 6);
    for (int w = 0; w < hw->n_workers; w++) {
        host_threads.emplace_back([this, w] { host_worker_loop(w); });
    }
    host_threads.emplace_back([this] { host_dispatch_loop(); });
    LLAMA_LOG_INFO("llama_moe_lru: host misses on: %d fills per layer and pass, the rest on %d+1 CPU threads (%.1f MiB requests)\n",
                   host_max_fill, hw->n_workers, sz / 1048576.0);
}

static inline void llama_moe_lru_pause() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
}

void llama_moe_lru::host_worker_loop(int /*w*/) {
    uint64_t seen = 0;
    auto idle_since = std::chrono::steady_clock::now();
    while (!host_stop.load(std::memory_order_relaxed)) {
        const uint64_t g = hw->gen.load(std::memory_order_acquire);
        if (g == seen) {
            // spin while requests come in (they are ~1 ms apart in a decode pass), sleep when idle
            if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(5)) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            } else {
                llama_moe_lru_pause();
            }
            continue;
        }
        seen = g;
        for (int j; (j = hw->next.fetch_add(1, std::memory_order_relaxed)) < hw->n_jobs; ) {
            hw->job(j);
            hw->done.fetch_add(1, std::memory_order_release);
        }
        idle_since = std::chrono::steady_clock::now();
    }
}

void llama_moe_lru::host_dispatch_loop() {
    auto idle_since = std::chrono::steady_clock::now();
    while (!host_stop.load(std::memory_order_relaxed)) {
        bool any = false;
        for (int li = 0; li < n_layers; li++) {
            ggml_lru_host_req * r = host_reqs[li];
            const uint32_t sq = r->seq;
            if (sq == hw->last_seq[li]) {
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            hw->last_seq[li] = sq;
            any = true;
            if (r->done != sq) {
                const auto t0 = std::chrono::steady_clock::now();
                host_process(li);
                std::atomic_thread_fence(std::memory_order_release);
                r->done = sq;
                if (hw->stats) {
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    hw->n_req++; hw->n_ent += r->n_ent; hw->us += us; hw->us_max = std::max(hw->us_max, us);
                    if (hw->n_req % 4096 == 0) {
                        LLAMA_LOG_INFO("llama_moe_lru: host misses: %llu requests (+%llu empty), %.2f experts and %.1f us each (max %.0f)\n",
                                       (unsigned long long) hw->n_req, (unsigned long long) hw->n_empty, (double) hw->n_ent / hw->n_req,
                                       hw->us / hw->n_req, hw->us_max);
                        hw->us_max = 0.0;
                    }
                }
            } else {
                hw->n_empty++;
            }
        }
        if (any) {
            idle_since = std::chrono::steady_clock::now();
        } else if (std::chrono::steady_clock::now() - idle_since > std::chrono::milliseconds(5)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        } else {
            llama_moe_lru_pause();
        }
    }
}

void llama_moe_lru::host_process(int li) {
    ggml_lru_host_req * r = host_reqs[li];
    const per_layer & L = layers[li];
    const int n_ent  = r->n_ent;
    const int n_tok  = r->n_tok;
    const int n_embd = r->n_embd;
    const int n_used = r->max_ent / GGML_LRU_HOST_MAX_TOK;
    if (n_ent <= 0) {
        return;
    }
    const ggml_tensor * tg = L.hpool[KIND_GATE];
    const ggml_tensor * tu = L.hpool[KIND_UP];
    const ggml_tensor * td = L.hpool[KIND_DOWN];
    const int64_t n_ff = tg->ne[1];
    const auto * trg = ggml_get_type_traits_cpu(tg->type);
    const auto * tru = ggml_get_type_traits_cpu(tu->type);
    const auto * trd = ggml_get_type_traits_cpu(td->type);
    GGML_ASSERT(trg->vec_dot_type == tru->vec_dot_type);
    const size_t qx_row = ggml_row_size(trg->vec_dot_type, n_embd);
    const size_t qh_row = ggml_row_size(trd->vec_dot_type, n_ff);
    host_work & W = *hw;
    W.qx.resize(qx_row * n_tok);
    W.g.resize((size_t) n_ent * n_ff);
    W.u.resize((size_t) n_ent * n_ff);
    W.qh.resize(qh_row * n_ent);
    for (int t = 0; t < n_tok; t++) {
        ggml_get_type_traits_cpu(trg->vec_dot_type)->from_float(r->x + (size_t) t * n_embd, W.qx.data() + t * qx_row, n_embd);
    }

    // host slot layout ([device][host slot][slice]): gate/up rows by device, down columns by device (host_slot_store)
    const int64_t n_host = tg->ne[2];
    auto run = [&](int n_jobs, const std::function<void(int)> & job) {
        W.job = job;
        W.n_jobs = n_jobs;
        W.next.store(0, std::memory_order_relaxed);
        W.done.store(0, std::memory_order_relaxed);
        W.gen.fetch_add(1, std::memory_order_release);
        for (int j; (j = W.next.fetch_add(1, std::memory_order_relaxed)) < n_jobs; ) {
            job(j);
            W.done.fetch_add(1, std::memory_order_release);
        }
        while (W.done.load(std::memory_order_acquire) < n_jobs) {
            llama_moe_lru_pause();
        }
    };

    // the window's entries grouped by expert: an expert several tokens route to is decoded once for all of them
    // (ggml_cpu_vec_dot_multi; LLAMA_MOE_LRU_HOST_MULTI=0: one vec_dot per entry and row, as before)
    static const bool multi = [] { const char * e = getenv("LLAMA_MOE_LRU_HOST_MULTI"); return !e || atoi(e) != 0; }();
    std::vector<std::vector<int>> groups;   // entries per distinct host slot
    {
        std::vector<std::pair<int64_t, int>> by_hs;
        for (int e = 0; e < n_ent; e++) by_hs.push_back({ (int64_t) r->ent[3 * e + 2], e });
        std::sort(by_hs.begin(), by_hs.end());
        for (size_t i = 0; i < by_hs.size(); i++) {
            if (!multi || i == 0 || by_hs[i].first != by_hs[i - 1].first || groups.back().size() >= 8) groups.emplace_back();
            groups.back().push_back(by_hs[i].second);
        }
    }
    const int n_grp = (int) groups.size();

    // gate and up: chunks of 64 rows per (expert group, kind)
    const int rc = 64;
    const int n_rc = (int) ((n_ff + rc - 1) / rc);
    run(n_grp * 2 * n_rc, [&](int j) {
        const int gi = j / (2 * n_rc), k = (j / n_rc) % 2, c = j % n_rc;
        const std::vector<int> & grp = groups[gi];
        const int ny = (int) grp.size();
        const int64_t hs = r->ent[3 * grp[0] + 2];
        const ggml_tensor * t = k == 0 ? tg : tu;
        const auto * tr = k == 0 ? trg : tru;
        const size_t slice = t->nb[2] / n_split;
        const int64_t r_d = t->ne[1] / n_split;
        const void * ys[8];
        for (int q = 0; q < ny; q++) ys[q] = W.qx.data() + (r->ent[3 * grp[q]] / n_used) * qx_row;
        float sv[8];
        for (int64_t row = (int64_t) c * rc; row < std::min<int64_t>(n_ff, (int64_t) (c + 1) * rc); row++) {
            const int64_t d = row / r_d, lr = row % r_d;
            const char * w = (const char *) t->data + ((size_t) d * n_host + hs) * slice + (size_t) lr * t->nb[1];
            if (!ggml_cpu_vec_dot_multi(t->type, (int) t->ne[0], sv, w, ys, ny)) {
                for (int q = 0; q < ny; q++) tr->vec_dot((int) t->ne[0], sv + q, 0, w, 0, ys[q], 0, 1);
            }
            for (int q = 0; q < ny; q++) ((k == 0 ? W.g.data() : W.u.data()) + (size_t) grp[q] * n_ff)[row] = sv[q];
        }
    });
    // SwiGLU (silu(gate) * up, as ggml_swiglu_split), quantized for down
    for (int e = 0; e < n_ent; e++) {
        float * g = W.g.data() + (size_t) e * n_ff;
        const float * u = W.u.data() + (size_t) e * n_ff;
        for (int64_t i = 0; i < n_ff; i++) {
            g[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        }
        ggml_get_type_traits_cpu(trd->vec_dot_type)->from_float(g, W.qh.data() + e * qh_row, n_ff);
    }
    // down: chunks of 256 output rows per entry; each row is the sum of its per-device column segments
    const int dc = 256;
    const int64_t n_out = td->ne[1];
    const int n_dc = (int) ((n_out + dc - 1) / dc);
    const size_t dslice = td->nb[2] / n_split;
    const size_t seg = td->nb[1] / n_split;
    const int64_t cols = td->ne[0] / n_split;
    const size_t qseg = ggml_row_size(trd->vec_dot_type, cols);
    run(n_grp * n_dc, [&](int j) {
        const int gi = j / n_dc, c = j % n_dc;
        const std::vector<int> & grp = groups[gi];
        const int ny = (int) grp.size();
        const int64_t hs = r->ent[3 * grp[0] + 2];
        float acc[8], sd[8];
        const void * ys[8];
        for (int64_t row = (int64_t) c * dc; row < std::min<int64_t>(n_out, (int64_t) (c + 1) * dc); row++) {
            for (int q = 0; q < ny; q++) acc[q] = 0.0f;
            for (int d = 0; d < n_split; d++) {
                const char * w = (const char *) td->data + ((size_t) d * n_host + hs) * dslice + (size_t) row * seg;
                for (int q = 0; q < ny; q++) ys[q] = W.qh.data() + grp[q] * qh_row + d * qseg;
                if (!ggml_cpu_vec_dot_multi(td->type, (int) cols, sd, w, ys, ny)) {
                    for (int q = 0; q < ny; q++) trd->vec_dot((int) cols, sd + q, 0, w, 0, ys[q], 0, 1);
                }
                for (int q = 0; q < ny; q++) acc[q] += sd[q];
            }
            for (int q = 0; q < ny; q++) (r->out + (size_t) grp[q] * n_embd)[row] = acc[q];
        }
    });
}

// ---------------------------------------------------------------------------------------------------------------------------
// slot loan (LLAMA_MOE_LRU_LEND): the ggml-cuda entry points, and lending the borrowed slots out / taking them back
static void * llama_moe_lru_cuda_fn(const char * name) {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}
llama_moe_lru::lend_pools_t   llama_moe_lru::lend_fn_pools()   { return (lend_pools_t)   llama_moe_lru_cuda_fn("ggml_backend_cuda_lend_pools"); }
llama_moe_lru::lend_compute_t llama_moe_lru::lend_fn_compute() { return (lend_compute_t) llama_moe_lru_cuda_fn("ggml_backend_cuda_lend_compute"); }
llama_moe_lru::lend_map_t     llama_moe_lru::lend_fn_map()     { return (lend_map_t)     llama_moe_lru_cuda_fn("ggml_backend_cuda_lend_map"); }
llama_moe_lru::lend_active_t  llama_moe_lru::lend_fn_active()  { return (lend_active_t)  llama_moe_lru_cuda_fn("ggml_backend_cuda_lend_active"); }

// on: decode takes the borrowed slots (they are empty). off: empty them (their experts still have their host slots) and stop
// using them, before a batch whose buffers overwrite them. The caller has synchronized the backends.
void llama_moe_lru::lend_set(bool on) {
    if (lend_slots <= 0 || on == lend_on) {
        return;
    }
    for (auto & L : layers) {
        const int64_t ns = L.n_slot, nb = L.n_base;
        const int64_t n_state = ggml_nelements(L.state);
        std::vector<int32_t> h(n_state);
        ggml_backend_tensor_get(L.state, h.data(), 0, n_state * 4);
        int32_t * owner = h.data(), * slot_of = h.data() + n_expert, * last_use = h.data() + n_expert + ns;
        for (int64_t s = nb; s < ns; s++) {
            if (slot_of[s] >= 0) {
                owner[slot_of[s]] = -1;
                slot_of[s] = -1;
            }
            last_use[s] = -1;
        }
        h[3 * n_expert + 2 * ns + 5] = (int32_t) (on ? ns : nb);
        ggml_backend_tensor_set(L.state, h.data(), 0, n_state * 4);
    }
    lend_on = on;
}
