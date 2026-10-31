#!/usr/bin/env python3
"""hitcurve.py TRACE OUT.json [S_MIN S_MAX STEP]: expected expert-cache misses per token for every average slot count S (per-layer
budgets optimized for the same VRAM, as slot-budget.py), from a decode routing trace (LLAMA_MOE_TRACE). Also writes the budget
file for each S next to OUT (slots-S.txt). The model's per-layer expert sizes come from the trace's layers (uniform unless
SIZES.json {layer: MiB} is given as $SIZES)."""
import collections, heapq, json, os, struct, sys

trace, out = sys.argv[1], sys.argv[2]
s_min, s_max, step = (int(x) for x in sys.argv[3:6]) if len(sys.argv) > 5 else (60, 200, 5)
sizes = {int(k): v for k, v in json.load(open(os.environ["SIZES"])).items()} if os.environ.get("SIZES") else {}

seqs = collections.defaultdict(list); last_ntok = {}
f = open(trace, 'rb')
while True:
    h = f.read(20)
    if len(h) < 20:
        break
    mg, il, kind, ne0, ne1 = struct.unpack('<4siiii', h)
    pos = f.tell(); n = None
    for c in (ne0 * ne1, ne0 * ne1 * last_ntok.get(il, 1), ne0 + (ne1 - 1) * 512):
        f.seek(pos + 4 * c); nx = f.read(4)
        if nx == b'MOET' or nx == b'':
            n = c; break
    f.seek(pos); raw = f.read(4 * n)
    if kind == 0:
        last_ntok[il] = ne1
        if ne1 == 1:
            seqs[il].append(struct.unpack(f'<{n}i', raw))

def curve(seq, smax):
    # LRU stack distances: a reference at depth d hits for every cache of more than d slots
    stack = []; hist = collections.Counter(); cold = 0
    for s in seq:
        for e in s:
            if e in stack:
                d = stack.index(e); hist[d] += 1; stack.pop(d)
            else:
                cold += 1
            stack.insert(0, e)
    total = sum(hist.values()) + cold
    c = []; hits = 0
    for sz in range(smax + 1):
        c.append(total - hits); hits += hist.get(sz, 0)
    return c, total

layers = sorted(seqs)
steps = min(len(seqs[il]) for il in layers)
smax = 512
cur = {}; tot = {}
for il in layers:
    cur[il], tot[il] = curve(seqs[il], smax)
size = lambda il: sizes.get(il, 1.0)
res = {"layers": layers, "steps": steps, "refs_per_token": sum(tot.values()) / steps, "points": {}}
outdir = os.path.dirname(os.path.abspath(out))
for S in range(s_min, s_max + 1, step):
    budget = sum(S * size(il) for il in layers)
    alloc = {il: 8 for il in layers}; used = sum(8 * size(il) for il in layers)
    heap = [(-(cur[il][8] - cur[il][9]) / size(il), il) for il in layers]; heapq.heapify(heap)
    while heap:
        g, il = heapq.heappop(heap)
        if used + size(il) > budget or alloc[il] >= smax - 1:
            continue
        alloc[il] += 1; used += size(il)
        heapq.heappush(heap, (-(cur[il][alloc[il]] - cur[il][alloc[il] + 1]) / size(il), il))
    misses = sum(cur[il][alloc[il]] for il in layers) / steps
    res["points"][S] = {"misses_per_token": misses, "hit": 1 - misses / res["refs_per_token"]}
    with open(os.path.join(outdir, f"slots-{S}.txt"), "w") as o:
        for il in layers:
            o.write(f"{il} {alloc[il]}\n")
    print(S, round(misses, 1), round(100 * res["points"][S]["hit"], 2))
json.dump(res, open(out, "w"), indent=1)
