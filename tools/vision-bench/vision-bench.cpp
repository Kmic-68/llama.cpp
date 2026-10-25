// vision-bench: time the vision encoder (mmproj) alone and dump its output embeddings, for comparing projector
// quantizations and encoder settings without loading the text model.
// usage: llama-vision-bench --model TEXT.gguf --mmproj F.gguf [--dev CUDA1] [--max-tokens N] [--reps R] [--fa on|off] [--out PREFIX] img...
// prints per image: tokens, encode ms (min/median of R), device memory used by the projector; writes PREFIX.<i>.f32
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static size_t dev_used(ggml_backend_dev_t d) {
    size_t f = 0, t = 0;
    if (d) ggml_backend_dev_memory(d, &f, &t);
    return t - f;
}

int main(int argc, char ** argv) {
    std::string mmproj, model_path, dev_name, out;
    int max_tok = 0, reps = 3;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    std::vector<std::string> imgs;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "--mmproj")     mmproj   = next();
        else if (a == "--model")      model_path = next();
        else if (a == "--dev")        dev_name = next();
        else if (a == "--max-tokens") max_tok  = std::stoi(next());
        else if (a == "--reps")       reps     = std::stoi(next());
        else if (a == "--out")        out      = next();
        else if (a == "--fa")         { std::string v = next(); fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO; }
        else imgs.push_back(a);
    }
    if (mmproj.empty() || imgs.empty()) { fprintf(stderr, "usage: %s --mmproj F.gguf [opts] img...\n", argv[0]); return 1; }
    ggml_backend_load_all();
    mtmd_helper_log_set([](ggml_log_level l, const char * t, void *) { if (l >= GGML_LOG_LEVEL_WARN) fputs(t, stderr); }, nullptr);

    ggml_backend_dev_t dev = dev_name.empty() ? nullptr : ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev_name.empty() && !dev) { fprintf(stderr, "no device %s\n", dev_name.c_str()); return 1; }
    ggml_backend_dev_t mdev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    const size_t used0 = dev_used(mdev);

    mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu = true; p.device = dev; p.print_timings = false; p.n_threads = 4; p.flash_attn_type = fa; p.warmup = true;
    if (max_tok > 0) p.image_max_tokens = max_tok;
    auto t0 = std::chrono::steady_clock::now();
    // the text model is loaded vocab-only (no weights): mtmd needs its tokenizer and n_embd
    llama_model * model = nullptr;
    if (!model_path.empty()) {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        model = llama_model_load_from_file(model_path.c_str(), mp);
        if (!model) { fprintf(stderr, "failed to load vocab from %s\n", model_path.c_str()); return 1; }
    }
    mtmd_context * ctx = mtmd_init_from_file(mmproj.c_str(), model, p);
    if (!ctx) { fprintf(stderr, "failed to load %s\n", mmproj.c_str()); return 1; }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t used_load = dev_used(mdev);
    printf("load %.0f ms, device mem after load+warmup %.1f MiB\n", load_ms, (used_load - used0) / 1048576.0);

    for (size_t k = 0; k < imgs.size(); k++) {
        auto bw = mtmd_helper_bitmap_init_from_file(ctx, imgs[k].c_str(), false, mtmd_helper_init_opt_default());
        if (!bw.bitmap) { fprintf(stderr, "cannot read %s\n", imgs[k].c_str()); return 1; }
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        std::string prompt = mtmd_default_marker();
        mtmd_input_text txt = { prompt.c_str(), prompt.size(), false, true };
        const mtmd_bitmap * bm = bw.bitmap;
        if (mtmd_tokenize(ctx, chunks, &txt, &bm, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        const mtmd_input_chunk * img = nullptr;
        for (size_t c = 0; c < mtmd_input_chunks_size(chunks); c++) {
            auto ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = ch;
        }
        if (!img) { fprintf(stderr, "no image chunk\n"); return 1; }
        const size_t n_tok = mtmd_input_chunk_get_n_tokens(img);
        std::vector<double> ms;
        size_t peak = 0;
        for (int r = 0; r < reps; r++) {
            auto t = std::chrono::steady_clock::now();
            if (mtmd_encode_chunk(ctx, img) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
            peak = std::max(peak, dev_used(mdev));
        }
        std::vector<double> s = ms; std::sort(s.begin(), s.end());
        // n_embd of the projector output = total floats / tokens; Qwen3-VL: 5120 (no deepstack here)
        const size_t n_embd = 5120;
        printf("%s: %zu tokens, encode min %.1f ms median %.1f ms, device mem %.1f MiB\n", imgs[k].c_str(), n_tok,
               s.front(), s[s.size() / 2], (peak - used0) / 1048576.0);
        if (!out.empty()) {
            std::string fn = out + "." + std::to_string(k) + ".f32";
            FILE * f = fopen(fn.c_str(), "wb");
            fwrite(mtmd_get_output_embd(ctx), sizeof(float), n_tok * n_embd, f);
            fclose(f);
        }
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
    }
    mtmd_free(ctx);
    if (model) llama_model_free(model);
    return 0;
}
