// p100_dma_box.cuh: the copy engine's mailbox and its words in VRAM (p100_dma.cuh); the router's last block writes them too (FNX_FOLD,
// moe.cuh). Included inside fnx's unnamed namespace.
#ifndef FNX_P100_DMA_BOX
#define FNX_P100_DMA_BOX
constexpr int DMA_MAX = MAXT * TOPK;   // a layer's distinct picks, at most
struct DmaEnt {
    const char* src_gu;
    char* dst_gu;
    const char* src_dn;
    char* dst_dn;
};
struct DmaBox {   // mapped host memory: the card writes the entries and n, then seq
    volatile unsigned seq;
    int n;
    size_t gu_bytes, dn_bytes;
    DmaEnt ent[DMA_MAX];
};
struct DmaDev {   // VRAM
    unsigned seq;    // the card's count of lists
    unsigned want;   // the last list's number, 0 when it was empty
    unsigned flag;   // the host thread's: the last list it copied
    unsigned err;    // 1: a wait timed out, 2: a wait for guessed experts did
    unsigned seq2, want2, flag2;   // FNX_PREFETCH: the same for the lists of guessed experts (box2, cs2)
    unsigned flag_gu, flag2_gu;    // the lists whose gate|up parts have landed (the copier sends them first; FNX_NOC_FILE split)
    unsigned gu_done[DMA_MAX], dn_done[DMA_MAX];   // split 2: the list whose k-th expert's gate|up / down part has landed
};
#endif
