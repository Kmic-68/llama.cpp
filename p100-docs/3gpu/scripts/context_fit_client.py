#!/usr/bin/env python3
"""Round 4 context-fit client (runs on the test machine against a llama-server on 127.0.0.1:18080).
  fit <corpus_file> <n_prompt_tokens> : tokenize the corpus, send exactly n tokens as one prompt (no prompt cache),
        generate 64 tokens, print the server's timings and truncation fields as one JSON line."""
import json, sys, time, urllib.request
PORT = 18080
def post(path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
def main():
    mode = sys.argv[1]
    if mode != "fit": raise SystemExit("unknown mode")
    text = open(sys.argv[2], encoding="utf-8", errors="replace").read(); n = int(sys.argv[3])
    toks = post("/tokenize", {"content": text})["tokens"]
    assert len(toks) >= n, f"corpus too short: {len(toks)} tokens"
    props = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=30) as r: props = json.loads(r.read())
    except Exception as e: props = {"error_reading_props": str(e)[:80]}
    t0 = time.time()
    r = post("/completion", {"prompt": toks[:n], "n_predict": 64, "temperature": 0.0, "top_k": 1, "seed": 42, "cache_prompt": False, "stream": False})
    t = r.get("timings", {})
    print(json.dumps({"n_requested": n, "corpus_tokens": len(toks), "n_ctx_slot": (props.get("default_generation_settings") or {}).get("n_ctx"),
        "prompt_n": t.get("prompt_n"), "prompt_tps": t.get("prompt_per_second"), "prompt_s": round((t.get("prompt_ms") or 0) / 1e3, 1),
        "predicted_n": t.get("predicted_n"), "predicted_tps": t.get("predicted_per_second"), "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
        "truncated": r.get("truncated"), "tokens_evaluated": r.get("tokens_evaluated"), "tokens_cached": r.get("tokens_cached"), "stop_type": r.get("stop_type"),
        "wall_s": round(time.time() - t0, 1), "text_head": (r.get("content") or "")[:80]}), flush=True)
if __name__ == "__main__":
    main()
