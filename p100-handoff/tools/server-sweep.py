import json,urllib.request,time,sys
S="/tmp/claude-1000/-home-kaden-llama-opt/f73b5a33-7d91-4dd3-a3b7-19ecb9e11747/scratchpad"
prompt=open(f"{S}/long262k_uniq.txt",errors="replace").read()
def run(nmax):
    body={"prompt":prompt,"n_predict":512,"temperature":0,"top_k":1,"seed":42,
          "cache_prompt":True,"speculative.n_max":nmax,"speculative.p_min":0.2}
    req=urllib.request.Request("http://127.0.0.1:8099/completion",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req, timeout=5400) as r:
        d=json.loads(r.read())
    t=d.get("timings",{})
    line=(f"n_max={nmax}  pred={t.get('predicted_n')} "
          f"{t.get('predicted_per_second',0):.3f} t/s  "
          f"prompt_n={t.get('prompt_n')} prompt={t.get('prompt_per_second',0):.1f} t/s  "
          f"wall={time.time()-t0:.1f}s")
    print(line, flush=True)
    open(f"{S}/sweep.log","a").write(line+"\n")
for k in [int(x) for x in sys.argv[1:]]:
    run(k)
