# v4 prompt renderer and graders: clean-room statement

**Dates:** first version 2026-10-30; moved to the v4src schema on 2026-10-31.

**Files:** all in `bench/v4/`:
- `prompts_v4.py`
- `grade_v4.py`
- `lcb_grade_v4.py`
- `split_lcb_tests.py`
- `test_v4.py`
- `test_lcb_v4.py`

## What was read
- **v4 data (2026-10-31):** the calibration set `route_data/v4src/`, which our own clean-room loaders built from the original public releases. We read:
  - `DATA_SPEC_v4.md`
  - `MANIFEST*.json`
  - the data files in `data/*.jsonl`
- **Earlier development (2026-10-30), before v4src existed:**
  - Our older row files `route_data/train*.jsonl` and the 520-row pilot set (`route_data/v4/pilot.jsonl`) with model responses from the pilot run.
  - Our own old FinQA loader `route_data/loaders/finqa.py`, read while diagnosing the pilot. It had dropped the FinQA tables to copy a RouterArena data quirk; v4src does not, and nothing of it is used here.
- **Background knowledge:** the public conventions of the source benchmarks. No web search was done.
- **Library:** `sacrebleu` 2.6.0, used for chrF.
- **LiveCodeBench test cases:** these come from the public HF release. They were decoded with our own restricted unpickler, which refuses every global.

## What was NOT read
- Nothing under the RouterArena repository: none of its templates, configs, scorer, data or code-execution helper.
- None of our lab files that used them: `bench/routerarena/*` (prompts, grade_matrix, ra_score, offline_eval, lcb_grade*, lcb_fetch_ra), any `livecodebench_util.py`, and any file mentioning `eval_config`.
- `SPEC.md` and `quotas.md`.
- No LLM was called.

## Schema (v4src)
Each row has these fields:
- `id`, `benchmark`, `subset`, `task`
- `context`, `question`, `options`
- `answer`, `aliases`, `meta`

`task` is one of `mc | binary | numeric | short | reading | translate | code`. For `mc` and `binary`, `answer` is the 0-based index into `options`. For every other task, the references are `[answer] + aliases`. The older row format is no longer supported.

## Templates (verbatim; one user message, no system prompt)
`{context}` is the row's context followed by a blank line, or empty. Options are lettered `A.`, `B.`, … in row order, so letter *i* is option index *i*.

```
mc (also binary): {context}{question}\n\nOptions:\n{options}\n{note}\nChoose the single best option. You may reason briefly first. End your response with a final line of the form "Answer: X", where X is the option letter.
numeric:   {context}{question}\n\nSolve the problem. You may show your work. End your response with a final line of the form "Answer: <final answer>", giving only the final result (a number, or yes/no if the question asks for it), without units.
short:     {context}{question}\n{note}\nYou may reason first if needed. End your response with a final line of the form "Answer: <your answer>", where the answer is as short as possible (a name, word, phrase, number or expression).
reading:   Read the passage and answer the question.\n\nPassage:\n{context}\n\nQuestion: {question}\n\nEnd your response with a final line of the form "Answer: <your answer>", a short phrase or sentence.
translate: Translate the following {src} text into {tgt}.\n\n{question}\n\nEnd your response with a final line of the form "Answer: <translation>", containing only the translation.
code_stdin: {question}\n\nWrite a complete Python 3 program that reads the input from standard input and writes the answer to standard output. Return the whole program in a single ```python code block.
code_func: {question}\n\nComplete the following starter code:\n```python\n{starter}\n```\n\nReturn the complete solution (the full class or function) in a single ```python code block.
```

No template contains benchmark-specific wording. Only two things are added, both driven by fields in the data:
- **`{note}` on binary rows:** when `meta.label_semantics` reads `1 = <meaning>`, the note is `(Option B means: <meaning>.)`. This applies to the ETHICS rows, whose labels are otherwise bare, such as no/yes.
- **`{note}` on short rows:** when the question contains the field marker `@placeholder` (ReCoRD), the note is `Give the word or name that should replace "@placeholder".`
- **Language names** come from `meta.src_lang_name` and `meta.tgt_lang_name`. All 18 WMT19 directions in the data are covered, including into English and between German and French.
- **Code variant:** `code_func` is used when `meta.starter_code` is non-empty, and `code_stdin` otherwise.

## Grading rules (`grade(row, response) -> float in [0,1]`, or `None` for code)
- **Extraction:** the grader uses the text after the last `Answer:` line, or the next non-empty line if that one is empty. If there is no `Answer:` line, it falls back to the last `\boxed{}`, then to the last non-empty line.
- **mc / binary:**
  - A letter is accepted as `C`, `(C)`, `C.`, `C) text` or `**C**`.
  - The exact option text is also accepted.
  - With no `Answer:` line, `\boxed{C}`, "the answer is (C)" and "correct option is C" are accepted too.
  - The letter must equal `answer`.
- **numeric:** the grader compares against each reference and accepts the best match.
  - **yes/no golds:** the first yes/no in the answer must match.
  - **Most benchmarks:** the first number in the answer, after removing `$`, `,` and `%` and evaluating `a/b` or `\frac{a}{b}`, must be within relative 1e-6.
  - **finqa:** the answer passes if it meets any of these:
    - it equals the gold after rounding to the gold's number of decimals, or is within 0.1%;
    - it differs only by ×100 or ÷100, where exactly one side has `%`;
    - it is `0.235` for a gold of `23.5`;
    - it agrees within 0.1% after a 1000^k rescale (k = ±1..3), for amounts of magnitude ≥ 1. This covers answers given "in millions".
- **short:** the matcher depends on the benchmark, because the answer formats differ:
  - **math:** a light LaTeX normaliser is applied to both sides. It removes `\left`/`\right`, `^\circ`, `\$`, `%`, `\text{}`, spaces, trailing `.` and `,`, a leading `x=` and thousands commas. It also turns `\dfrac`/`\tfrac` into `\frac`, `\frac12` into `\frac{1}{2}`, `\sqrt3` into `\sqrt{3}`, and `(A)` into `A`. The results must be equal, or numerically equal when the gold is a plain number or fraction.
  - **qanta:** the accepted strings come from each quiz-bowl answer line: the main answer, the `{required}` part alone, and the `[or …]`/`[accept …]` alternatives. `prompt on` alternatives are not accepted. A response passes if, after normalisation, it equals or contains an accepted string, or is a trailing part of one with at least 4 letters, such as a surname.
  - **chessinstruct:** the answer is compared after removing case, spaces, quotes and brackets, which also covers list answers. A JSON object echoed from the stated output format is unpacked first. For White/Black golds, the answer must name the gold side and not the other. For move golds, the first UCI move must match.
  - **asdiv:** answers such as `17 (units); 9 (units)`, `11:9` or `11:45 p.m.` must give the same numbers in the same order. Word answers such as History or Yes must match after normalisation.
  - **Everything else (ReCoRD):** the answer must equal a reference after normalisation, or contain it with at most 3 extra words.
- **reading:** the score is the best over all references (NarrativeQA has two). An answer passes if either rule holds:
  - token F1 ≥ **0.5**, or
  - it covers at least half of the reference's content words (at least 2 when the reference has 2 or more), with no more than 4×max(reference, 3) content words. Content words are non-stopwords with a crude ing/ed/es/s/e suffix stem.
- **translate:** passes if sentence chrF from sacrebleu (default settings) against any reference is at least **40**.
  - On test rows, correct but freely worded translations scored 44–88; untranslated or wrong-language output scored 5–25.
  - With one reference, some valid translations will still fail.
- **Normalisation for text:** lowercase, fold accents, turn punctuation and `_` into spaces, and drop a/an/the.

## LiveCodeBench
- **`split_lcb_tests.py`:**
  - It streams `data/livecodebench.jsonl` and writes `data/livecodebench_tests.jsonl.gz`, one row per item: `{"id", "gidx", "fn_name", "public_tests", "private_tests"}`. That came to 200 rows, 6,715 tests and 568 MB.
  - It rewrites `livecodebench.jsonl` as a small file of 424 KB. The tests are removed from `meta`, and `n_public_tests`, `n_private_tests` and `tests_file` are added.
  - It checks the round trip before replacing anything, and keeps the original as `livecodebench.jsonl.full`.
- **`lcb_grade_v4.py`** uses only the standard library. Tests can come from three places, given as `--tests`, one or more JSONL or `.gz` files:
  - the v4src rows, with tests in `meta`;
  - the split test file;
  - HF-style rows, with base64 → zlib → restricted-unpickle → JSON.
- **Running each test:**
  - Public tests run first, then private tests.
  - Each test gets a fresh `python -I` subprocess with its own session.
  - A 6 s timeout kills the whole process group.
  - The rlimits are set in the child before exec: address space 2 GB, CPU time, 64 MB file size, no core dumps.
  - The run stops at the first failure.
- **Checking results:**
  - **stdin tasks:** each output line has trailing whitespace stripped, and trailing blank lines are ignored. Tokens must match exactly, unless both parse as floats; then they must agree within relative 1e-6.
  - **functional tasks:** each input line is one JSON argument to `Solution().<fn_name>`. The result is compared with the JSON-decoded expected output, using the same float tolerance; booleans must stay booleans.
- **Output:** rows of `{"gidx", "model", "score", "reason"}`.
- **Isolation:** model-generated code runs only in `docker run --rm --network none --memory 4g --cpus 2`. Our tests run only solutions we wrote by hand.

## Tests and checks (2026-10-31)
- **`test_v4.py`:** 3 real rows from every benchmark in `data/` (22 benchmarks, 4,264 rows), each with a correct, a wrong and a messy-but-correct response. It also includes the 2026-10-30 pilot regressions for FinQA and NarrativeQA, rewritten for the new rows.
- **Check on all 4,064 non-code rows:** "Answer: <gold>" (the letter for mc/binary) passes on every row, and a junk answer fails on every row.
- **Rendering:** every row is rendered.
- **`test_lcb_v4.py`:** covers the old-format and v4src tasks.
- **Result:** 122 passed.

## Data notes
- **FinQA:** the data now includes tables. Some references are odd as released: `answer` and `exe_ans` sometimes disagree in sign or value, and one answer is a text fragment. Since both are references, either one passes.
- **MATH:** a few answers are option labels such as `\text{(A)}`.
- **ChessInstruct:** the invalid task types are dropped in v4src.
- **GeographyData:** has no rows.
