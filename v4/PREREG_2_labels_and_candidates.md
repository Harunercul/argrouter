# argrouter v4 pre-registration, part 2: calibration data, prompts, graders, candidate models

Date: 2026-10-31. Committed before any model is run on the v4 calibration set. Part 1 (data spec, exclusion
hashes) is commit 824258c.

## Calibration set (built)
4,264 items from 21 public benchmarks, drawn by the loaders in [`loaders/`](loaders/) under
[`DATA_SPEC_v4.md`](DATA_SPEC_v4.md) (incl. Amendment 1: data validity). Per-benchmark sources, revisions,
counts and exclusions: [`MANIFEST.json`](MANIFEST.json) and [`MANIFEST/`](MANIFEST/). Item list with question
hashes: [`items.tsv`](items.tsv). GeographyData: no public source found, 0 items. MusicTheoryBench: 64 items
(source exhausted after exclusion).

## Prompts and graders
Written independently of RouterArena (who wrote them, what they read and did not read:
[`pipeline/CLEANROOM.md`](pipeline/CLEANROOM.md)). One generic template per task type; our own answer
grader and our own LiveCodeBench test runner (model code runs only in a network-less container).

| file | sha256 |
|---|---|
| pipeline/prompts_v4.py | `1b8c2b6052ddedc18d05a77d9bcffe37861b60e06578832e8619bb0a8935763c` |
| pipeline/grade_v4.py | `c9b9653bdf559a7c733751243026fb08cdf1db21fa9cc17263c94a48a2bff359` |
| pipeline/lcb_grade_v4.py | `e1fee2c1e7373df277f14be2873ed630b03825e9190980b9bf0330a257cf6b19` |
| pipeline/run_v4.py | `c4868ef97e21cfd9c477a82c73ce958640b1c5cde5c9d3e81bfdf67ad163956f` |
| all.jsonl (4,264 items, not published; rebuild with the loaders) | `86028ed7a780ad7fe1f5a7845c81b3134b4a16e5e244e24904f085afbd686e2d` |
| livecodebench_tests.jsonl.gz (not published) | `bc1dce52194305f0fa1d3a988eefc1bf19e99128458640fe221fe7f99bcc90dc` |
| MANIFEST.json | `e0a2c66587c083e17d3cc2ec08e1714a95751828eeeca46b93c0d2d75d19fda4` |

## Candidate models (rule and result)
**Rule.** Candidates are the models on the accuracy–cost Pareto front of our pilot, plus any model within 1.0
accuracy point of the front at its cost. Pilot: 16 models × 520 items, equal weight per benchmark, prompts and
grader as above, FinQA and LiveCodeBench left out (FinQA's pilot items lacked their tables; code was not graded).

**Disclosure.** The pilot items were drawn from our earlier calibration pool, which had been sized from
RouterArena's per-source counts; it is used here only to shortlist models, never to train or calibrate v4.
The 16 pilot models were chosen by hand from OpenRouter, informed by our v1–v3 experience and by the models
that leaderboard entries use. The v4 router itself is trained and calibrated only on the 4,264-item set above.

| pilot model | accuracy % | $ / 1K items | on front |
|---|---:|---:|---|
| google/gemma-4-31b-it | 80.3 | 0.16 | yes |
| openai/gpt-6-luna | 80.9 | 0.24 | yes |
| google/gemini-3.1-flash-lite (reasoning none) | 83.4 | 0.38 | yes |
| qwen/qwen3-235b-a22b-2507 | 81.5 | 0.48 | |
| anthropic/claude-haiku-5.5 | 84.5 | 0.51 | yes |
| xiaomi/mimo-v2.6-flash | 80.9 | 0.70 | |
| deepseek/deepseek-v4-flash-0731 | 84.9 | 0.74 | yes |
| deepseek/deepseek-v4-flash | 82.4 | 0.82 | |
| openai/gpt-6-luna-pro | 82.0 | 0.85 | |
| google/gemini-3-flash-preview (reasoning none) | 85.0 | 0.86 | yes |
| z-ai/glm-5.3-flash | 83.8 | 0.88 | |
| inclusionai/ling-3.1-flash | 83.2 | 1.17 | |
| google/gemini-3.7-flash (reasoning low) | 87.6 | 1.48 | yes |
| deepseek/deepseek-v4.1-flash-1017 | 84.5 | 1.61 | |
| google/gemini-3.8-flash (reasoning low) | 87.7 | 1.68 | yes |
| minimax/minimax-m3.1-flash | 81.4 | 2.26 | |

**Result: 8 candidates** (no model fell within 1.0 point of the front without being on it): gemma-4-31b-it,
gpt-6-luna, gemini-3.1-flash-lite@none, claude-haiku-5.5, deepseek-v4-flash-0731, gemini-3-flash-preview@none,
gemini-3.7-flash@low, gemini-3.8-flash@low. Each is called on all 4,264 items with `run_v4.py` (model + our
messages, provider defaults, `reasoning.effort` only where marked).

## What comes next (part 3, before any RouterArena run)
The routing policy (pool, decision rule, thresholds) is chosen on this set only, with every benchmark weighted
equally and at the providers' standard (non-introductory) list prices, also reported at the price table's current
prices. The frozen router (code + weights + config, with sha256) is committed as part 3; only then is it run on
RouterArena, once.
