# argrouter pre-registration, part 5: the frozen v5 router

Date: 2026-11-01. Committed before the v5 router routes any RouterArena query. Part 4 (the v4 result and the v5
plan) is commit c015c0e, with its amendment in 5898e84. Everything below follows the rule of part 4, applied to
our calibration set only.

## Labels
- **Calls.** The 8 new candidates were each called once on all 4,264 items with the part-2 prompts and runner.
  That was 34,112 calls, $45.63 billed.
- **Failures.** Only qwen/qwen3.8-flash had any: 24 failed with upstream 429 or read timeouts.
- **Retries.** `run_v4.py` crashes on resume before making a call (it reads the trailing newline of a finished
  file as a row). So retries went through [`pipeline/retry_v5.py`](pipeline/retry_v5.py), which changes only that
  check.
  - 23 of the 24 succeeded.
  - One item, chessinstruct/FIND_FINAL_SCORE/19598, failed on every attempt.
  - Part 4 admits a model only if every item is graded, so **qwen3.8-flash is not a candidate**.
- **Code.** Code answers were graded with the part-2 sandboxed grader:
  [`lcb_graded_v5.jsonl`](lcb_graded_v5.jsonl), 1,455 of 1,600 programs passing.
- **Prices.** As in [`prices_ra.json`](prices_ra.json).

## Single candidates
"hard" uses the part-4 weights (floor 0.5) and "equal" uses equal benchmark weights. Both are arena-style scores
on our set, not RouterArena scores.

| model | hard acc % | hard $/1K | hard score | equal score |
|---|---:|---:|---:|---:|
| nex-agi/nex-n2.5-pro | 79.2 | 0.733 | 75.69 | 84.23 |
| google/gemini-3.7-flash@low | 81.1 | 1.910 | 75.16 | 83.22 |
| google/gemini-3.8-flash@low | 81.4 | 2.461 | 74.72 | 82.69 |
| deepseek/deepseek-v4.1-flash | 79.6 | 1.563 | 74.44 | 82.96 |
| deepseek/deepseek-v4.1-flash-1017 | 80.1 | 1.853 | 74.44 | 82.74 |
| z-ai/glm-5.3-flash | 77.8 | 0.848 | 74.21 | 82.75 |
| deepseek/deepseek-v4-flash-0731 | 78.0 | 1.089 | 73.95 | 83.06 |
| openai/gpt-6-luna-pro | 77.6 | 0.982 | 73.83 | 82.62 |
| openai/gpt-6-luna | 75.4 | 0.294 | 73.82 | 83.80 |
| anthropic/claude-haiku-5.5 | 75.8 | 0.521 | 73.37 | 83.19 |
| google/gemma-4-31b-it | 73.7 | 0.203 | 72.70 | 82.98 |
| xiaomi/mimo-v2.6-flash | 76.4 | 1.204 | 72.46 | 81.03 |
| google/gemini-3-flash-preview@none | 76.0 | 1.020 | 72.45 | 81.93 |
| xiaomi/mimo-v2.6-pro | 79.7 | 3.987 | 71.93 | 79.87 |
| google/gemini-3.1-flash-lite@none | 72.3 | 0.486 | 70.49 | 80.79 |

## Pool and λ, by the part-4 rule
Full log: [`fit_router_v5.log`](fit_router_v5.log). Code: [`pipeline/fit_router_v5.py`](pipeline/fit_router_v5.py),
which runs in the lab layout next to the v4 scripts.

**Forward selection.**
1. nex-agi/nex-n2.5-pro: objective 75.69.
2. google/gemini-3.8-flash@low: 76.77.
3. Stop. The best third addition, z-ai/glm-5.3-flash, scored 76.85, a gain of 0.08, under the 0.10 threshold.

**λ.** Leave-one-wrapper-out, 5 folds × 2 seeds, hard weights:

| λ | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 | **0.16** |
|---|---:|---:|---:|---:|---:|---:|
| hard | 75.25 | 75.59 | 75.96 | 76.18 | 76.55 | **76.65** |
| equal | 83.46 | 83.81 | 84.24 | 84.61 | 85.02 | 85.15 |

λ = 0.16 is the edge of the pre-registered grid, and the scores were still rising there. We keep the grid as
pre-registered.

**What the router does on our set.** Rendered with prompt variant A, it sends 4,024 of 4,264 items to
nex-n2.5-pro and 240 to gemini-3.8-flash@low. Those 240 are 199 of the 200 chess items and a few NarrativeQA,
SocialIQA, MedMCQA and QANTA items. In effect it is one cheap model with an escalation for chess. We report
it as such.

## Frozen artefacts
| file | sha256 |
|---|---|
| router_v5.npz (pool, λ = 0.16, all weights; RouterV4 format) | `8dd5bf420b5405017549b259972e0e163feecdd390b1fee248f9a9e890d3aba5` |
| ../v4/pipeline/route_v4.py (unchanged router class) | `41c85d24fb6241b64d6700aca22351595661246830e6f9d0bb3b33b2f2edff40` |
| pipeline/ra_predict_v5.py | `03d012a08ae812524592f3de0f0044e8fbe20cbbbea86e22cd8381a6cca5b6ac` |
| pipeline/fit_router_v5.py | `0a2960ee9c9346b79cbcee197e26d404cfc0b286403156be2ad2b899f2ce76af` |
| pipeline/retry_v5.py | `e93052a3cc89f97eaf593af0ac2441e0187edb62c6983b4d7c22a039bbeffe20` |
| prices_ra.json | `cc4f0416b7d483ed871f5166acc3a4af3f997af4f1746c872a6009453c72411e` |
| lcb_graded_v5.jsonl | `33e85410a3791ac557438974d9c7e22ce2922ef96715d2045ce28c0dfb3c6b30` |

## The RouterArena run (after this commit, once)
1. `ra_predict_v5.py --split full` and `--split robustness`, on RouterArena main with the pool's price profiles
   added (OpenRouter base prices, with source links).
2. RouterArena's `llm_inference/run.py` makes one fresh call per routed query through OpenRouter. No cached
   answer from v1 to v4 is used.
3. RouterArena's `llm_evaluation/run.py` scores the calls. We submit once, and the submission discloses the v1 to
   v4 results, including v4's 73.56, and parts 1 to 5.

Failed calls are retried with the same model. No routing decision changes after this commit.
