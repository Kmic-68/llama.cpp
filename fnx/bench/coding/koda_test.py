#!/usr/bin/env python3
"""koda_test.py (2026-11-18, q4hitri): runs koda_naloge.py's tasks against an OpenAI-style chat endpoint.

koda_f(post, out_json, log)  FUNKCIJE: reasoning low (Koda's default for Flash-Next), greedy, the answer's code run against
                             the hidden asserts.
koda_u(post, out_json, log)  UREJANJE: "orodje" tasks as Koda works (read result given, the model must call edit/write; an
                             edit whose oldString is not in the file exactly once is refused like OpenCode's and counted as
                             a failed edit, the model may try again, at most 8 turns); "cela" tasks return the whole file.
post(body) -> the endpoint's JSON answer. Both return a summary dict and write every answer to out_json.

Standalone (a quick check against a running engine, e.g. production through the node, nothing else changes):
  python3 koda_test.py URL MODEL f|u [ID ...]       URL like http://127.0.0.1:43140/v1/chat/completions"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import koda_naloge as K

SUFFIX_EN = " Reply with the code only, in one ```python block."
SUFFIX_SL = " Odgovori samo s kodo, v enem bloku ```python."

KODA_SISTEM = (
    "Si Koda, programerski agent v terminalu. Uporabnikove datoteke spreminjaš z orodji. Orodje `edit` v datoteki zamenja "
    "besedilo `oldString` z `newString`: oldString mora biti dobesedno prepisan iz datoteke (brez številke vrstice in znaka '| ' "
    "pred njo, z enakimi zamiki) in se mora v datoteki pojaviti natanko enkrat, razen če nastaviš replaceAll. Orodje `write` "
    "zapiše celo datoteko, `read` jo prebere. Ko je naloga opravljena, na kratko povej, kaj si spremenil, brez novih klicev orodij.")

KODA_SISTEM_EN = (
    "You are Koda, a coding agent in a terminal. You change the user's files with tools. The `edit` tool replaces `oldString` "
    "with `newString` in a file: oldString must be copied verbatim from the file (without the line number and the '| ' before "
    "it, with the same indentation) and must occur in the file exactly once, unless you set replaceAll. The `write` tool writes "
    "a whole file, `read` reads one. When the task is done, say briefly what you changed, without further tool calls.")

TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file. Lines come numbered as 00001| text.",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "Absolute path"}},
                    "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "edit",
     "description": "Replace oldString with newString in a file. oldString must match the file's text exactly (without the "
                    "line-number prefix) and occur once, unless replaceAll is true.",
     "parameters": {"type": "object", "properties": {
         "filePath": {"type": "string", "description": "Absolute path"},
         "oldString": {"type": "string", "description": "The exact text to replace"},
         "newString": {"type": "string", "description": "The text to replace it with"},
         "replaceAll": {"type": "boolean", "description": "Replace every occurrence"}},
         "required": ["filePath", "oldString", "newString"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write a whole file.",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["filePath", "content"]}}},
]


def _is_sl(prompt):
    return prompt.startswith("Napiši") or "funkcijo v Pythonu" in prompt


def _timing(out):
    t = out.get("timings") or {}
    return t.get("predicted_n") or 0, (t.get("predicted_ms") or 0) / 1000.0


def koda_f(post, out_json, log, ids=None):
    res, ok_n, ntok, tsec = {}, 0, 0, 0.0
    for fid, prompt, tests, _ in K.FUNKCIJE:
        if ids and fid not in ids:
            continue
        body = {"model": "strata", "messages": [{"role": "user", "content": prompt + (SUFFIX_SL if _is_sl(prompt) else SUFFIX_EN)}],
                "max_tokens": 6000, "temperature": 0.0, "reasoning_effort": "low", "stream": False}
        t0 = time.time()
        try:
            out = post(body)
            msg = out["choices"][0]["message"]
            text = msg.get("content") or ""
            ok, err = K.run_code(K.code_of(text), tests)
        except Exception as e:                                   # a failed request counts as a wrong answer
            out, msg, text, ok, err = {}, {}, "", False, f"zahteva: {e!r}"[:300]
        n, s = _timing(out)
        ntok += n
        tsec += s
        ok_n += ok
        res[fid] = {"prav": ok, "napaka": err, "odgovor": text, "razmislek_znakov": len(msg.get("reasoning_content") or ""),
                    "n": n, "tok_s": round(n / s, 2) if s else None, "wall": round(time.time() - t0, 1)}
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    narobe = ",".join(k for k, v in res.items() if not v["prav"])
    en = [k for k in res if K.angleska(k)]
    en_ok = sum(res[k]["prav"] for k in en)
    log(f"  koda_f: {ok_n}/{len(res)} prav (anglesko {en_ok}/{len(en)}), {ntok} zetonov, {ntok / tsec if tsec else 0:.2f} tok/s; "
        f"narobe: {narobe or '-'}")
    return {"prav": ok_n, "od": len(res), "prav_en": en_ok, "od_en": len(en), "zetonov": ntok,
            "tok_s": round(ntok / tsec, 2) if tsec else None, "narobe": narobe}


def _args(call):
    a = call.get("function", {}).get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            return None
    return a if isinstance(a, dict) else None


def _tool(name, args, files, path, stats):
    """Runs one tool call on the in-memory file; returns the tool result text."""
    if args is None:
        stats["slabih_klicev"] += 1
        return "Error: the arguments are not valid JSON."
    fp = str(args.get("filePath") or "")
    if Path(fp).name != Path(path).name:
        stats["slabih_klicev"] += 1
        return f"Error: file not found: {fp}"
    text = files[path]
    if name == "read":
        return K.numbered(text)
    if name == "write":
        files[path] = str(args.get("content") or "")
        stats["write"] += 1
        return "File written successfully."
    if name == "edit":
        old, new = args.get("oldString"), args.get("newString")
        if not isinstance(old, str) or not isinstance(new, str) or old == new or not old:
            stats["napacnih_urejanj"] += 1
            return "Error: oldString and newString must be different strings, oldString not empty."
        n = text.count(old)
        if n == 0:
            stats["napacnih_urejanj"] += 1
            return "Error: oldString not found in content. It must match the file exactly, including whitespace and indentation."
        if n > 1 and not args.get("replaceAll"):
            stats["napacnih_urejanj"] += 1
            return "Error: Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match."
        files[path] = text.replace(old, new) if args.get("replaceAll") else text.replace(old, new, 1)
        stats["edit"] += 1
        return "Edit applied successfully."
    stats["slabih_klicev"] += 1
    return f"Error: unknown tool {name}."


def _orodje(post, path, original, instr, stats, english=False):
    files = {path: original}
    msgs = [{"role": "system", "content": KODA_SISTEM_EN if english else KODA_SISTEM},
            {"role": "user", "content": instr},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_read_0", "type": "function",
             "function": {"name": "read", "arguments": json.dumps({"filePath": path})}}]},
            {"role": "tool", "tool_call_id": "call_read_0", "content": K.numbered(original)}]
    last = ""
    for turn in range(8):
        out = post({"model": "strata", "messages": msgs, "tools": TOOLS, "max_tokens": 4000, "temperature": 0.0,
                    "reasoning_effort": "low", "stream": False})
        n, s = _timing(out)
        stats["zetonov"] += n
        stats["s"] += s
        msg = out["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        last = msg.get("content") or ""
        if not calls:
            break
        msgs.append({"role": "assistant", "content": last, "tool_calls": calls})
        for i, c in enumerate(calls):
            stats["klicev"] += 1
            result = _tool(c.get("function", {}).get("name"), _args(c), files, path, stats)
            msgs.append({"role": "tool", "tool_call_id": c.get("id") or f"call_{turn}_{i}", "content": result})
    stats["korakov"] = turn + 1
    return files[path], last


def koda_u(post, out_json, log, ids=None):
    res, ok_n, total = {}, 0, {"zetonov": 0, "s": 0.0, "napacnih_urejanj": 0, "enako_ref": 0}
    for uid, mode, path, original, instr, tests, pairs in K.UREJANJE:
        if ids and uid not in ids:
            continue
        stats = {"zetonov": 0, "s": 0.0, "klicev": 0, "edit": 0, "write": 0, "napacnih_urejanj": 0, "slabih_klicev": 0}
        t0 = time.time()
        try:
            if mode == "orodje":
                final, reply = _orodje(post, path, original, instr, stats, K.angleska(uid))
            else:
                out = post({"model": "strata", "messages": [{"role": "user", "content": instr + "\n\n```python\n" + original + "```"}],
                            "max_tokens": 8000, "temperature": 0.0, "reasoning_effort": "low", "stream": False})
                n, s = _timing(out)
                stats["zetonov"] += n
                stats["s"] += s
                reply = out["choices"][0]["message"].get("content") or ""
                final = K.code_of(reply)
            ok, err = K.run_code(final, tests)
        except Exception as e:
            final, reply, ok, err = "", "", False, f"zahteva: {e!r}"[:300]
        ref = K.apply_ref(original, pairs)
        same = final.rstrip() == ref.rstrip()
        ok_n += ok
        for k in ("zetonov", "s", "napacnih_urejanj"):
            total[k] += stats[k]
        total["enako_ref"] += same
        res[uid] = {"prav": ok, "napaka": err, "nacin": mode, "enako_ref": same, "koncna": final, "odgovor": reply,
                    "wall": round(time.time() - t0, 1), **stats,
                    "tok_s": round(stats["zetonov"] / stats["s"], 2) if stats["s"] else None}
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    narobe = ",".join(k for k, v in res.items() if not v["prav"])
    en = [k for k in res if K.angleska(k)]
    en_ok = sum(res[k]["prav"] for k in en)
    log(f"  koda_u: {ok_n}/{len(res)} prav (anglesko {en_ok}/{len(en)}), napacnih urejanj {total['napacnih_urejanj']}, enako ref "
        f"{total['enako_ref']}, {total['zetonov']} zetonov, {total['zetonov'] / total['s'] if total['s'] else 0:.2f} tok/s; "
        f"narobe: {narobe or '-'}")
    return {"prav": ok_n, "od": len(res), "prav_en": en_ok, "od_en": len(en), "napacnih_urejanj": total["napacnih_urejanj"],
            "enako_ref": total["enako_ref"], "zetonov": total["zetonov"],
            "tok_s": round(total["zetonov"] / total["s"], 2) if total["s"] else None, "narobe": narobe}


# English questions with one checkable answer (reasoning low, greedy): the answer must be among the reply's last tokens.
ANGL = [
    ("A train leaves at 14:35 and travels for 2 hours and 50 minutes. At what time does it arrive? End with the time only.", ["17:25"]),
    ("How many positive integers n with n <= 100 are divisible by 3 or by 5? End with the number only.", ["47"]),
    ("What is the remainder when 2^100 is divided by 7? End with the number only.", ["2"]),
    ("In how many distinct ways can the letters of the word LEVEL be arranged? End with the number only.", ["30"]),
    ("What is the sum of all prime numbers below 30? End with the number only.", ["129"]),
    ("Anna is older than Ben, Ben is older than Carl, and Carl is older than Dan. Who is the second youngest? End with the name only.", ["carl"]),
    ("What is the capital city of Australia? End with the city name only.", ["canberra"]),
    ("Which chemical element has the symbol W? End with the element's name only.", ["tungsten"]),
    ("A shirt costs $40 after a 20% discount. What was its original price in dollars? End with the number only.", ["50"]),
    ("How many ordered pairs of positive integers (a, b) satisfy a*b = 3600 and gcd(a, b) = 1? End with the number only.", ["8"]),
    ("What is the remainder when 3^2026 is divided by 100? End with the number only.", ["29"]),
    ("In triangle ABC, AB = 13, BC = 14 and CA = 15. What is the length of the altitude from A to BC? End with the number only.", ["12"]),
    ("A bag holds 5 red and 3 blue balls. Two are drawn without replacement. What is the probability that both are red? End with the reduced fraction only.", ["5/14"]),
    ("Who wrote the novel Pride and Prejudice? End with the author's surname only.", ["austen"]),
    ("Write 0.375 as a fraction in lowest terms. End with the fraction only.", ["3/8"]),
    ("How many trailing zeros does 125! (125 factorial) have? End with the number only.", ["31"]),
]
# Open English prompts, read by a person later; checked here only for empty answers and for characters outside Latin script.
ANGL_ODPRTO = [
    "Explain in about 150 words how a heat pump heats a house in winter.",
    "Write a short, polite email to a customer explaining that their order will arrive three days late because of a supplier delay.",
    "What is the difference between a process and a thread? Answer in five sentences.",
    "Summarize the main idea of the TCP three-way handshake for a beginner, in under 120 words.",
]


def angl(post, out_json, log):
    import re
    res, ok_n, foreign = {}, 0, 0
    for i, (q, want) in enumerate(ANGL, 1):
        try:
            out = post({"model": "strata", "messages": [{"role": "user", "content": q}], "max_tokens": 6000, "temperature": 0.0,
                        "reasoning_effort": "low", "stream": False})
            text = (out["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            out, text = {}, f"zahteva: {e!r}"[:300]
        toks = re.findall(r"\d+[:.]\d+|\d+/\d+|\d+|[a-z]+", text[-80:].lower())
        ok = any(w in toks[-3:] for w in want)
        ok_n += ok
        res[f"q{i}"] = {"vprasanje": q, "odgovor": text, "prav": ok, "n": _timing(out)[0]}
    for i, q in enumerate(ANGL_ODPRTO, 1):
        try:
            out = post({"model": "strata", "messages": [{"role": "user", "content": q}], "max_tokens": 1200, "temperature": 0.0,
                        "reasoning_effort": "none", "stream": False})
            text = (out["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            out, text = {}, f"zahteva: {e!r}"[:300]
        odd = [c for c in text if ord(c) > 0x24F and c not in "–—‘’“”…•€£×→←≤≥°·"]
        bad = not text or bool(odd)
        foreign += bad
        res[f"o{i}"] = {"vprasanje": q, "odgovor": text, "tuji_znaki": "".join(odd[:20]), "slabo": bad, "n": _timing(out)[0]}
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    narobe = ",".join(k for k, v in res.items() if not v.get("prav", True) or v.get("slabo"))
    log(f"  angl: {ok_n}/{len(ANGL)} prav, odprtih s tujimi znaki ali praznih {foreign}/{len(ANGL_ODPRTO)}; narobe: {narobe or '-'}")
    return {"prav": ok_n, "od": len(ANGL), "odprto_slabo": foreign, "narobe": narobe}


if __name__ == "__main__":
    import urllib.request
    url, model, which, *ids = sys.argv[1:]

    def post(body):
        body = {**body, "model": model}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return json.load(r)

    out = Path(__file__).resolve().parent / f"koda_test-{which}.json"
    if which == "a":
        print(angl(post, out, print))
    else:
        print({"f": koda_f, "u": koda_u}[which](post, out, print, set(ids) or None))
