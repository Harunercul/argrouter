# argrouter v4 calibration data: specification (written before any v4 data was drawn)

Date: 2026-10-31. This file fixes how the v4 calibration set is built. It is part of the v4 pre-registration.

## Sources
The 23 public benchmarks that RouterArena's paper names as its sources (stated openly; that list is the only thing taken
from RouterArena besides the exclusion hashes): AIME, ArcMMLU, AsDiv, ChessInstruct, ETHICS, FinQA, GeoBench,
GeographyData, GSM8K, LiveCodeBench, MATH, MathQA, MedMCQA, MMLU, MMLU-Pro, MusicTheoryBench, NarrativeQA, OpenTDB,
PubMedQA, QANTA, SocialIQA, SuperGLUE, WMT19. Each is loaded from its own public release (Hugging Face or the authors'
repository), in its native schema. Loaders must not open RouterArena's repository, data files, templates, configs or
code, nor argrouter's older loaders/specs (route_data/loaders, SPEC.md, quotas.md, train/, train_all.jsonl).

## Allocation (fixed)
- Budget: **200 items per benchmark**, the same for every benchmark. If a benchmark has fewer usable items after
  exclusion, take all of them and record the shortfall. No top-ups from other benchmarks.
- Inside a benchmark, the budget is split **evenly across the benchmark's own natural subsets** (all of them: every
  MMLU-Pro category, every MMLU subject, every OpenTDB category, every WMT19 language pair and direction the release
  ships, every SuperGLUE task, every ETHICS subset, every ChessInstruct task type, every LiveCodeBench platform),
  round-robin; a subset with too few items gives its remaining share back to the round-robin.
- Split preference: train, then validation/dev, then test. Uniform random with `random.Random(20261031)`.
- Items needing modalities we cannot send as text (images, audio) are dropped, counted, and replaced by the round-robin.
- Every table, passage or context the source provides is kept in full (e.g. FinQA tables serialised as text).

## Exclusion
`exclude_hashes.json` (sha256 73fc86d5…5284): drop a candidate when sha256(norm(question)) is listed, with
`norm(s) = re.sub(r"\W+", " ", str(s).lower()).strip()[:300]`; also drop it when sha256("<family>:<native id>") is
listed (family names as given in the file). Each loader logs how many candidates each rule removed.

## Output schema (ours)
One JSONL per benchmark in `route_data/v4src/data/<benchmark>.jsonl`:
`{"id": "<benchmark>/<subset>/<native id>", "benchmark", "subset", "split", "task": "mc|numeric|short|reading|translate|code|binary",
"context": str, "question": str, "options": [str], "answer": <canonical>, "aliases": [str], "meta": {...}, "source": "<hf id or url>@<revision>"}`
Canonical answers: `mc` → 0-based int index into `options`; `numeric` → string as the source gives it; `short`/`reading`
→ the reference string (+ `aliases`); `translate` → reference translation, with `meta.src_lang`/`meta.tgt_lang`;
`binary` → `options` holds the two labels in the source's order and `answer` the index; `code` → `meta` holds the
test cases (public and private, decoded) and `meta.starter_code`/`meta.fn_name` when present.
A `MANIFEST.json` lists per benchmark: source + revision, subsets, available/excluded/taken counts, shortfalls.
