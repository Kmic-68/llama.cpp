# Coding and English tasks with hidden tests

The tasks we used on 18-19 Nov 2026 to check that the fast, lossy UD-Q4_K_XL profile does not code worse than the exact
one (docs/GUIDE.md, "Fast UD-Q4_K_XL"). We wrote them for that test, so no model saw them in training. The file and
variable names are partly Slovenian, as is our work; every task asked in Slovenian also has an English twin (`_en`), and
our verdict counted the English ones.

| File | What it is |
|---|---|
| `koda_naloge.py` | the tasks. `FUNKCIJE`: 45 functions or classes (41 in English) with hidden asserts and our reference solution each. `UREJANJE`: 26 edits: 11 done as a coding agent works (the file arrives as a `read` tool result and the model must call `edit` with an `oldString` copied exactly, or `write`), 4 whole files of 127-161 lines to return with one change and tests for every function in them, and 11 English twins |
| `koda_test.py` | the runner: `koda_f` (functions), `koda_u` (edits), `angl` (16 English questions with one answer, 4 open prompts) |
| `preveri_naloge.py` | checks the tasks without a model: every reference passes its tests, every original file fails its edit's tests |

Run them against any OpenAI-style chat endpoint (Strata's server, this engine, llama.cpp, ...), Python 3.10+ on Linux:

```bash
python3 bench/coding/preveri_naloge.py                                                    # ends with "KONEC uspeh"
python3 bench/coding/koda_test.py http://127.0.0.1:8080/v1/chat/completions strata f      # functions
python3 bench/coding/koda_test.py http://127.0.0.1:8080/v1/chat/completions strata u      # edits
python3 bench/coding/koda_test.py http://127.0.0.1:8080/v1/chat/completions strata a      # English
```

Each run prints a summary (`prav` = passed, `anglesko` = English only, `napacnih urejanj` = `edit` calls refused because
the `oldString` was not in the file exactly once, `narobe` = the failed tasks) and writes every answer to
`koda_test-f.json`, `-u.json` or `-a.json` beside the script. The requests use temperature 0 and
`"reasoning_effort": "low"`, the way our coding agent calls the model.

**Warning:** the model's code runs on your machine (`python3 -I` in a temporary folder, 20 s timeout, no network expected).
Run it in a container or as an unprivileged user if that matters to you.

One failed task in a run is noise in this set: the exact UD-Q4_K_XL profile fails one function too (`racunalo`, an
expression evaluator). Our numbers are in docs/GUIDE.md, "Fast UD-Q4_K_XL".
