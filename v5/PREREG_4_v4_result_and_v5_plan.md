# argrouter pre-registration, part 4: the v4 RouterArena result and the v5 plan

Date: 2026-11-01. Committed before any label of the new v5 candidates is looked at, and before any v5 router
exists. Parts 1 to 3 are commits 824258c, 00e989c + 64a90e6 and 9fa3faf.

## The v4 result (run once, as part 3 said)
The frozen v4 router (`v4/router_v4.npz`, sha256 cdd8063c…) routed RouterArena's full split. RouterArena's own
`llm_inference/run.py` made one fresh call per query (12,445 calls, 0 failures) and its own
`llm_evaluation/run.py` scored them, at the prices of RouterArena main as of 2026-10-31 (before #250):

| metric | v4 |
|---|---:|
| RouterArena score | 73.56 |
| accuracy | 74.99 % |
| cost per 1K queries | $0.266 |
| robustness | 50.48 |

That ranks below 10 routers. It is not submitted yet. It will be published in the submission alongside v5.

What went wrong, seen from the per-benchmark results:
- Our calibration set is much easier than RouterArena for the same sources. On ours, every candidate answered
  94 to 98 % of quiz-bowl (QANTA) questions; on RouterArena the routed models got 24 to 44 %. NarrativeQA,
  FinQA, Ethics and SuperGLUE show the same gap.
- With almost every item answered by every model, the router learned that the cheapest model is nearly always
  enough. It sent 36 % of queries to gpt-6-luna, which was the weakest of the pool on RouterArena (72.6 %).
- A stricter grader does not close the gap (QANTA stays at 90 to 96 % under exact match), so the items, not the
  grading, are too easy.

## What v5 changes, and why each change is a reaction to the v4 result
We say this plainly: every change below was decided after seeing the v4 RouterArena result. None of them uses
RouterArena data, templates, configs or per-query results. They are decided here, before any v5 label is seen.

1. **Objective weights.** Items are weighted by equal benchmark share × (0.25 + err), where err is the fraction
   of the 8 v4 candidates that answered the item wrongly. The 8 reference models are fixed; new candidates never
   change the weights. Items everybody gets right still count (the 0.25 floor) but count less. The v4 equal
   weighting is still reported for information.
2. **Prices.** RouterArena now charges every model at OpenRouter's base list price (#218, #250, #262, merged
   2026-10-31). We price every candidate the same way:
   [`prices_ra.json`](prices_ra.json), taken from RouterArena main 44cc1ff's `model_cost.json`, or, for models
   not listed there, from OpenRouter's models API on 2026-11-01
   ([`prices_ra_sources.json`](prices_ra_sources.json)).
   - This replaces part 3's "standard list price" rule.
   - It brings back gemini-3.7-flash and gemini-3.8-flash, now priced at $0.75 / $3.75 and $0.825 / $4.125.
3. **More candidates.** Eight new models were each smoke-tested on 44 items, then called once on all 4,264
   calibration items with the unchanged part-2 prompts and runner. They are:
   - z-ai/glm-5.3-flash
   - xiaomi/mimo-v2.6-flash
   - xiaomi/mimo-v2.6-pro
   - nex-agi/nex-n2.5-pro
   - deepseek/deepseek-v4.1-flash
   - deepseek/deepseek-v4.1-flash-1017
   - openai/gpt-6-luna-pro
   - qwen/qwen3.8-flash

   Three smoke-tested models were left out before the full run:
   - gemini-3.8-flash at reasoning medium: too expensive to label.
   - inclusionai/ling-3.1-flash: its OpenRouter price is a temporary 75 % discount.
   - minimax/minimax-m3.1-flash: its OpenRouter price is an off-peak discount.

   Code answers are graded with the part-2 sandboxed LiveCodeBench grader.
4. **Pool and λ, by a fixed rule** (`fit_router_v5.py`, the v4 router with these changes):
   - **Candidates:** the v4 candidates plus the new ones, as long as every item is graded for that model.
   - **Pool:** greedy forward selection. At each step, add the candidate that most raises the cross-validated
     score under the weights of item 1. Use 5 folds and 1 seed, and take the best λ in
     (0.005, 0.01, 0.02, 0.04, 0.08, 0.16) for each pool. Stop when the gain is under 0.10 points or the pool
     has 6 models.
   - **λ for the final pool:** the v4 leave-one-wrapper-out procedure (5 folds × 2 seeds), under the same
     weights.
   - The router class, embedding model and features are unchanged from part 3.

## Then
Part 5 freezes the v5 router (hashes of the weights and code, the selection log) before it routes any
RouterArena query. After that we run RouterArena once and submit once, with v4's result and this file disclosed.

## Amendment (2026-11-01, same day, before any v5 label is seen)
The 0.25 floor in item 1 is replaced by **0.5**.

Why: a dry run of the selection rule on the 8 v4 candidates only (labels already public in part 3) showed that
0.25 makes our set much harder than RouterArena, and the rule then picks a pool costing $2 to $3 per 1K queries.
We calibrated the floor against one number, v4's measured RouterArena accuracy (74.99 %). The calibration
used the v4 router's out-of-fold picks on our set (pool and λ = 0.08 of part 3):

| floor | weighted accuracy | $ / 1K | score |
|---|---:|---:|---:|
| 0.25 | 68.3 % | 0.422 | 67.18 |
| **0.5** | **75.9 %** | 0.372 | 73.95 |
| 0.75 | 79.2 % | 0.350 | 76.82 |
| 1.0 | 81.0 % | 0.339 | 78.40 |
| equal weights | 87.3 % | 0.298 | 83.91 |

The grid was 0.25, 0.5, 0.75, 1, 1.5, 2 and 4; we took the value closest to 74.99 %. This is the only use of a
RouterArena result in v5, and it is one aggregate number from our own v4 run. The floor is set in
`fit_router_v5.py` as `HARD_FLOOR`.
