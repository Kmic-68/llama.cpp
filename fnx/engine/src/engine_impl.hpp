// The engine's per-card buffers (engine.cu, prefill.cu).
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "kernels.hpp"
#include "model.hpp"
#include "owned.hpp"

namespace fnx {

// a lane's state on one card: its page table, recurrent states and histories, QSA raw-key rings; the engine's
// current-lane pointers in CardBuf are a copy of the lane's (Engine::use_lane)
struct LaneDev {
    int* pt = nullptr;
    std::vector<float*> S, chist;
    float* ple_hist = nullptr;
    std::vector<uint16_t*> ring;      // per layer (attention layers only)
    uint16_t* mtp_ring = nullptr;
    int* ctl = nullptr;               // [2]: the lane's position and tag in a batched window
    int* mrope_buf = nullptr;         // the lane's rotary table ([max_ctx][3]), made at its first image
    int* mrope = nullptr;             // mrope_buf while its sequence holds an image, else null (rotary position = cell)
};

struct CardBuf {
    int dev;
    cudaStream_t st;
    cudaStream_t st2;                 // the shared expert, beside the routed ones
    cudaEvent_t fork, join;
    cudaStream_t st3 = nullptr;       // FNX_P100_OVERLAP: the missing experts' copy, beside the row kernels
    void* dma = nullptr;              // FNX_P100_DMA: the copy engine's thread for them instead (p100_dma.cuh)
    cudaEvent_t fork2 = nullptr, join2 = nullptr;
    float *R, *mixed, *bo_a, *bo_f, *inj_a, *inj_f, *hcs;
    float* cvec = nullptr;            // FNX_CVEC: NL x N directions (Engine::cvec_), null without a vector
    void* xq;   // q8_1 rows
    float *qkvz, *ab;
    float* gdn_scratch;
    float *qg, *k, *v, *att;
    float* att_part;                  // the attention splits' partials
    float *logits, *topw;
    int *topi, *njobs;
    Job *jobs_gu, *jobs_dn, *dense[MAXT + 1];
    float *gu, *part;
    uint16_t *xh16 = nullptr, *h16 = nullptr;   // FNX_P100_ROWS: mixed and h as FP16, for the row kernel
    // edge0_patch.py (FNX_LATENT): D's output, its f16 rows (t * LAT_G + group), z and its q8_1, U's output [T][N]
    float *lxl = nullptr, *lz = nullptr, *latu = nullptr;
    uint16_t* lxl16 = nullptr;
    uint8_t* lzq = nullptr;
    // edge0_patch6.py (FNX_LAT_SPLIT): host mailboxes of D's half (lxg) and of z's part (lzs), written by the other card; *_next: the other card's
    unsigned long long *lxg_in = nullptr, *lxg_next = nullptr, *lzs_in = nullptr, *lzs_next = nullptr;
    float* sh_gu;                     // the shared expert: [T][gate | up]
    void* sh_hq;                      // its h as q8_1
    float *ple_emb, *ple_key, *ple_val;
    float* head_logits;
    int *tk_pen_tok = nullptr, *tk_pen_cnt = nullptr, *tk_pen_n = nullptr, *tk_i = nullptr;   // top_tokens()
    float* tk_v = nullptr;
    unsigned* tk_err = nullptr;
    float* tk_par = nullptr;
    // top_tokens()'s logit_bias: BIAS_LISTS slots of this card's tokens of a list (made at the first list), each slot's list
    // (its key, 0: none) and count; per row its start and count (tk_bias_rows: they hold a list's rows, else all zero)
    int *tk_bias_tok = nullptr, *tk_bias_off = nullptr, *tk_bias_n = nullptr;
    float* tk_bias_val = nullptr;
    uint64_t tk_bias_key[BIAS_LISTS] = {};
    int tk_bias_cnt[BIAS_LISTS] = {};
    bool tk_bias_rows = false;
    float* best_v;
    int* best_i;
    unsigned long long* am_keys;      // argmax's per-row keys
    // state
    std::vector<float*> S, chist;
    std::vector<uint16_t*> kc, vc;   // f16: each attention layer's K and V in the paged pool (its offsets in a page)
    uint16_t* pool = nullptr;        // the paged K/V pool (kernels.hpp): n_pages pages of KVP_PAGE halves
    int n_pages = 0;
    int* pt = nullptr;               // the page table (device)
    KvPages pg;
    // QSA per attention layer (and the draft layer's): the raw-key ring and the pooled keys (f16); shared scratch
    std::vector<uint16_t*> qsa_ring, qsa_pooled;
    uint16_t *mtp_qsa_ring = nullptr, *mtp_qsa_pooled = nullptr;
    float *qsa_y, *qsa_q, *qsa_scores;
    int *qsa_cells, *qsa_ncells;
    float* ple_hist;
    // what commit() cuts a window back to: the state after each row, each kept count's histories
    std::vector<float*> snap, hsnap;
    // every recurrent layer's snap in one block: between a commit and the next verify nothing reads it, so the prompt
    // path borrows it for scratch
    float* snap_block = nullptr;
    size_t snap_block_bytes = 0;
    float* ple_hsnap;
    Ring ring;
    AttRing att_ring;   // the attention partials' merge
    int* ctl;                         // [0] window position, [1] ring tag (as unsigned)
    int* mrope = nullptr;             // the current lane's rotary table (its LaneDev::mrope)
    cudaEvent_t ev0, ev1;
    char* stage[2] = {nullptr, nullptr};   // a parked state's copies pass through these page-locked halves (engine.cu)
    cudaEvent_t stage_ev[2] = {nullptr, nullptr};
    std::vector<LaneDev> lanes;
    // window graphs, one per size, updated in place when a window of that size runs for other lanes (graph memory is
    // ~8-14 MB each a card: one per lane or per set of lanes does not fit beside 262k of K/V)
    struct GraphSlot {
        cudaGraphExec_t exec = nullptr;
        std::vector<int> key;   // the lane(s) it was captured for
    };
    std::vector<GraphSlot> win_graphs = std::vector<GraphSlot>(MAXT + 1), mtp_graphs = std::vector<GraphSlot>(MAXT + 1),
                           batch_graphs = std::vector<GraphSlot>(MAXT + 1);
    // FNX_PRECAP: batched-window graphs that were not evicted by a window of another layout. batch_graphs[T] stays the graph the next
    // window of T rows launches (and the tables enqueue_rows reads); a layout change puts the slot's graph and its tables here, and a
    // window of a layout held here swaps it back. Entries are the same on every card (same index = same layout)
    struct GraphEntry {
        cudaGraphExec_t exec = nullptr;
        std::vector<int> key;
        const int** lane_rows = nullptr;
        float** lane_gdn = nullptr;
        const int** seg_ctl = nullptr;
        int* seg_row0 = nullptr;
        unsigned long used = 0;   // the engine's tick of its last use
    };
    // FNX_CAPTURE_AFTER: the tables of a window run without a graph (verify_rows' eager windows); they stand in for the slot's while it runs
    struct EagerTab {
        const int** lane_rows = nullptr;
        float** lane_gdn = nullptr;
        const int** seg_ctl = nullptr;
        int* seg_row0 = nullptr;
    } eager_tab;
    std::vector<std::vector<GraphEntry>> batch_cache = std::vector<std::vector<GraphEntry>>(MAXT + 1);
    std::vector<unsigned long> batch_slot_used = std::vector<unsigned long>(MAXT + 1, 0);
    // batch_graphs[T]'s lane rows (LaneRows): [MAXT] each row's lane's ctl, then [MAXT] its page table; written when
    // the graph is captured, read by it
    std::vector<const int**> lane_rows = std::vector<const int**>(MAXT + 1, nullptr);
    // and its rows' recurrent state then conv history per layer (GdnLanes): [NL][MAXT] S, then [NL][MAXT] hist
    std::vector<float**> lane_gdn = std::vector<float**>(MAXT + 1, nullptr);
    // a verify_rows() graph's (in batch_graphs[T], its key telling it from step()'s): lane_rows' positions point into
    // row_pos (each row's position, written every window), lane_gdn holds the lanes' (segments') pointers in segment
    // order, seg_ctl each segment's lane ctl, seg_row0 its first window row
    int* row_pos = nullptr;
    std::vector<const int**> seg_ctl = std::vector<const int**>(MAXT + 1, nullptr);
    std::vector<int*> seg_row0 = std::vector<int*>(MAXT + 1, nullptr);
    CopyJob* rows_commit_jobs = nullptr;   // commit_rows()'s copies
    // the draft layer: token embeddings in, its normed inputs, fc outputs, source rows, residual, K/V, head
    uint16_t *mtp_kc = nullptr, *mtp_vc = nullptr;   // f16
    float *mtp_emb = nullptr, *mtp_en, *mtp_e2, *mtp_src, *mtp_hn, *mtp_h2, *mtp_R, *mtp_logits, *mtp_bv;
    int *mtp_bi, *mtp_ctl;
    // the batched draft layer (Engine::draft_lanes): per lane [pos0, tag], per row its cell, and the rows' pointer table (the main window's lane_rows, for the draft rows)
    int *mtp_lctl = nullptr, *mtp_row_pos = nullptr;
    const int** mtp_lane_rows = nullptr;
    // per lane of a batched draft window: the last row's penalties and (cands) its TOPK_N candidates
    int *mtp_pen_l = nullptr, *mtp_tk_i_l = nullptr;
    float *mtp_pen_par_l = nullptr, *mtp_tk_v_l = nullptr;
    unsigned* mtp_tk_err_l = nullptr;
    int* mtp_pen = nullptr;        // the last row's penalties (penalize_row's pen, this card's draft-vocabulary indexes)
    float* mtp_pen_par = nullptr;
    float* mtp_tk_v = nullptr;     // the last row's TOPK_N candidates (draft(cands)); mtp_zero: no penalties for top_tokens
    int *mtp_tk_i = nullptr, *mtp_zero = nullptr;
    unsigned* mtp_tk_err = nullptr;
    CopyJob* commit_jobs = nullptr;   // commit()'s copies (copy_many)
    // recorded after each upload of commit()'s (commit_rows()'s) table from its one page-locked host copy: the next
    // commit waits on it before writing that copy again
    cudaEvent_t commit_ev = nullptr, rows_commit_ev = nullptr;
    unsigned* route_cnt = nullptr;   // FNX_ROUTE_COUNT, card 0: [decode | prompt][NL][NEXP] picks
    unsigned* ekeep = nullptr;       // FNX_EXPERT_MASK (maska_patch.py): [NL][NEXP / 32] bits, set: the expert is kept
    unsigned* route_trace = nullptr;              // FNX_ROUTE_TRACE=M, card 0, mapped host memory: decode's picks in order (M million)
    unsigned long long* route_trace_len = nullptr;
    unsigned long long route_trace_cap = 0;
    int* cache_list = nullptr;       // moe_cache's scratch (with an ExpertCache)
    // FNX_PREFETCH_PROBE=1, card 0: layer l + 1's router on layer l's FFN input, its top 10 per row (guess [NL][MAXT *
    // TOPK]) scored against layer l + 1's picks (cnt [7]: probe_counts)
    float* probe_logits = nullptr;
    int* probe_guess = nullptr;
    unsigned long long* probe_cnt = nullptr;
    // FNX_PREFETCH=1, every card: layer l + 1's router on layer l's FFN input (beside the shared expert), its guess, the experts it
    // plans into slots (pf_list, cache_list's form) and how many guesses a row to take (pf_want, 0: none; from the host)
    float* pf_logits = nullptr;
    int* pf_guess = nullptr;
    int* pf_list = nullptr;
    int* pf_want = nullptr;
    // memo (memo_patch.py): the window's lookup rows' guesses ([NL][MAXT][TOPK], -1 none; [NL * MAXT * TOPK]: any), the guesses the
    // probe made for this layer (kept at the layer before), card 0 only: counts
    int* memo_guess = nullptr;
    int* memo_probe = nullptr;
};

// what engine_common.cu reads of this model (engine_parts.hpp says what each is)
constexpr int ROW_W = D;   // the HC streams of N
constexpr int LANE_ROW_COLS = 2;
inline void lane_row_extra(const int**, int, const LaneDev&) {}
inline void card_lane(CardBuf& b, const LaneDev& L) {
    b.pt = L.pt;
    b.pg.pt = L.pt;
    b.S = L.S;
    b.chist = L.chist;
    b.ple_hist = L.ple_hist;
    for (int l = 0; l < NL; ++l)
        if (is_attn(l)) b.qsa_ring[l] = L.ring[l];
    b.mtp_qsa_ring = L.mtp_ring;
    b.mrope = L.mrope;
}
inline cudaError_t reset_hists(CardBuf& b) { return cudaMemsetAsync(b.ple_hist, 0, (size_t) 9 * D * 4, b.st); }
// the PLE history (lane L's, or the current lane's when L is null) from row `at`'s snapshot
constexpr int HIST_JOBS = 1;
inline int hist_jobs(CopyJob* jobs, const CardBuf& b, const LaneDev* L, size_t at) {
    jobs[0] = {L ? L->ple_hist : b.ple_hist, b.ple_hsnap + at * 9 * D, (size_t) 9 * D * 4};
    return 1;
}
inline const float* draft_src(const CardBuf& b, bool from_main) { return from_main ? b.R : b.mtp_R; }
constexpr bool MTP_HEAD_EVERY_ROW = true;
inline unsigned* err_word(CardBuf& b) { return b.ring.err; }
inline const char* window_error(unsigned err, bool draft) {
    if (err & 1u) return "a ring sum timed out";
    if (err & 2u) return "a ring sum met a NaN or inf (or, on the fp16 ring, a value past its range)";
    if (err & 4u) return "an hc read's grid barrier timed out";
    if (err & 16u) return draft ? "the draft layer's logits hold a NaN or inf" : "the head's logits hold a NaN or inf";
    if (err) return "an error bit no message names";
    return nullptr;
}
// mtp: the draft layer is attached (its K/V and QSA keys are part of the state); hist_half: the conv history half the
// next window reads
std::vector<std::pair<char*, size_t>> state_pieces(CardBuf& b, const CardW& w, bool mtp, int pos, bool kv, unsigned hist_half,
                                                   const std::vector<int>& pages);

// FNX_PROF=1: what a window's steps cost on card 0: time stamps between them, captured with the graph and read after the window
enum ProfKind { PK_HC, PK_QKVZ, PK_GDN, PK_OUT, PK_SUM, PK_IDX, PK_QKV, PK_ATT, PK_WO, PK_ROUTER, PK_CACHE, PK_GU, PK_DOWN, PK_JOIN, PK_HEAD, PK_PLE, PK_N };
void prof_begin(int T);
void prof_mark(CardBuf& b, int kind);
void prof_collect(int T);

// FNX_PROF=1 on the prompt path
enum PfKind { PF_HC, PF_QKVZ, PF_AB, PF_GDN, PF_OUT, PF_QKV, PF_IDX, PF_SEL, PF_ATT, PF_WO, PF_SUM, PF_ROUTER, PF_GU, PF_DOWN, PF_SHARED, PF_DRAFT, PF_END, PF_N };
void pf_prof_begin();
void pf_prof_mark(CardBuf& b, int kind);
void pf_prof_report(int rows);

}  // namespace fnx
