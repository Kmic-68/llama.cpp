// p100_over.hpp: a decode layer's missing experts copied on a stream of their own while the ones in VRAM are computed (p100_moe_rows.cuh).
#pragma once
#include "kernels.hpp"

#include <vector>

namespace fnx {
// FNX_NOC_FILE (2026-11-08 night, mikro/noc_patch.py): modes read before each window, "g2 split memo" (at start FNX_ROWS_G2, FNX_DMA_SPLIT,
// FNX_MEMO); a change is in graph_key, so the window graphs are captured again
struct NocMode {
    int g2 = 0;      // phase 2 (the copied experts): lanes a row, 0 as phases 0-1 (gate|up 2, down 1), 4 or 8 (128 threads a block)
    int split = 0;   // 1: phase 2 gate|up waits for the gate|up parts only, its down for the rest (the copier sends gate|up first);
                     // 2: each copied expert's blocks wait for its own part (perexp_patch.py)
    int memo = 0;    // the routing memo (memo_patch.py)
};
const NocMode& noc_mode();
int noc_gen();
bool noc_tick();
void p100_dma_wait_gu(void* h, void* st);
// memo (memo_patch.py): a window's / a prompt chunk's picks into the memo ([cap][NL][TOPK], mapped host memory), the lookup rows'
// guesses merged into the prefetch's (a row whose memo guess is -1 keeps its own), and the counts behind the fnx: noc lines
void memo_put_win(const int* topi, int T, const int* ctl, int layer, uint16_t* memo, int cap, void* st);
void memo_put_rows(const int* topi, int C, int pos0, int layer, uint16_t* memo, int cap, void* st);
void memo_merge(const int* mg, int layer, int T, int* guess, void* st);
void memo_stat(const int* topi, const int* mg, const int* probe, const int* cache_list, int layer, int T, int pf_n, const int* pf_list,
               unsigned long long* stat, void* st);
void p100_dma_wait2_gu(void* h, void* st);
// split 2 (perexp_patch.py): the copier's per-expert words (part 0: gate|up, 1: down; [DMA_MAX]), the list's number and the error word,
// and phase 2 with its blocks waiting for their expert's part
const unsigned* p100_dma_done(void* h, int part);
const unsigned* p100_dma_want_ptr(void* h);
unsigned* p100_dma_err_ptr(void* h);
void p100_moe_rows_phase_w(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, int T, const uint16_t* x, float* out, int phase, void* st,
                           const unsigned* done, const unsigned* want, const int* list, int layer, unsigned* err);
bool p100_overlap_on();
void p100_moe_rows_phase(const DW& A, size_t sa, const Job* jobs, const int* njobs, int njobs_cap, int T, const uint16_t* x, float* out, int phase, void* st);
void p100_jobs_split(Job* gu, Job* dn, const int* njobs, int layer, const int* list, void* st);
void p100_cache_copy(int layer, size_t gu_stride, size_t dn_stride, const ExpertCache& ec, const int* list, void* st);
void p100_swiglu_after(const float* gu, int ffs, int P, uint16_t* h16, const int* list, void* st);
void p100_tail_slots(const ExpertCache& ec, int p, int count, bool lend, void* st);
// a prompt chunk's attention with Strata's P100 kernels (p100_attn.cuh, int8 K/V)
bool p100_attn_on();
size_t p100_attn_scratch_floats();
void p100_attn_prompt(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells, const int* n_cells, int card, int C, int r0, int T,
                      float* part, float* scratch, void* st);
// FNX_P100_DMA=1: a layer's missing experts by the copy engine, a host thread of the card's (p100_dma.cuh)
bool p100_dma_on();
void* p100_dma_start(int dev);
void p100_dma_stop(void* h);
void p100_dma_publish(void* h, int layer, const ExpertCache& ec, const int* list, size_t gu_stride, size_t dn_stride, void* st);
void p100_dma_wait(void* h, void* st);
// FNX_PF_REFILL=1: the copier also fills the prompt path's slots after a read, between its lists: an entry an expert, done its pool's count
// once it has landed (Engine::pf_refill_start, pf_refill_tick)
struct DmaFill {
    const char* src_gu;
    char* dst_gu;
    const char* src_dn;
    char* dst_dn;
    size_t gu_bytes, dn_bytes;
    int pool, done;
};
void p100_dma_fill(void* h, std::vector<DmaFill> jobs, const int* landed);
void p100_dma_fill_stop(void* h);
void p100_dma_landed(void* h, int* out);
// FNX_PREFETCH=1: layer `layer`'s guessed experts (the first *want of each row's guess) not in a slot take one (p100_moe_rows.cuh), the
// copier copies them on a stream of its own (box2), and the layer's window waits for them before its first phase (p100_dma.cuh)
void p100_pf_plan(const int* guess, int T, int layer, const ExpertCache& ec, const int* want, int* list, void* st);
void p100_dma_publish2(void* h, int layer, const ExpertCache& ec, const int* list, size_t gu_stride, size_t dn_stride, void* st);
void p100_dma_wait2(void* h, void* st);
// FNX_FOLD (IDEJE 39): the mailbox and the words in VRAM for the router's last block (RouteTail, kernels.hpp), and the guessed experts' plan
// and publish in one launch
void* p100_dma_box(void* h);
void* p100_dma_dev(void* h);
void p100_pf_plan_publish(void* h, const int* guess, int T, int layer, const ExpertCache& ec, const int* want, int* list, size_t gu_stride, size_t dn_stride, void* st);
// and a window's (FNX_P100_ATTN_DEC=1): attn()'s partials, its scratch p100_attn_dec_floats(T) floats
bool p100_attn_dec_on();
size_t p100_attn_dec_floats(int T);
void p100_attn_window(const float* qg, const uint16_t* kc, const uint16_t* vc, const KvPages& g, const int* pos0, const int* cells, const int* n_cells, int card, int T,
                      float* part, float* scratch, void* st);
}  // namespace fnx
