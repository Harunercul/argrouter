# argrouter v4 pre-registration, part 3: the frozen router

Date: 2026-10-31. Committed before the v4 router is run on any RouterArena query. Part 1 (data spec, exclusion
hashes) is commit 824258c; part 2 (calibration set, prompts, graders, candidate models) is 00e989c + 64a90e6.
Everything below was decided on the 4,264-item v4 calibration set only. No RouterArena data, template, config or
result was used.

## Labels
The 8 candidates of part 2 were each called once on all 4,264 items with `run_v4.py` (34,112 calls, 0 errors,
$29.06 billed). Answers were graded with `grade_v4.py`. LiveCodeBench code (1,600 programs) was graded with
`lcb_grade_v4.py` inside a macOS sandbox that blocks all network access and file writes outside a scratch folder
and denies reads of the home directory (1,392 passed). The per-item code scores are in
[`lcb_graded.jsonl`](lcb_graded.jsonl).

All costs are recomputed from billed token counts (reasoning tokens are part of output tokens) at standard list
prices: [`prices_standard.json`](prices_standard.json).
- gemini-3.7-flash and gemini-3.8-flash are priced at $1.50 / $7.50. That is Google's post-introductory price
  for the 3.8 Flash tier, which runs at $0.75 / $3.75 until 2026-12-31 (ai.google.dev/gemini-api/docs/pricing).
- deepseek-v4-flash-0731 is priced at $0.14 / $0.28, DeepSeek's list price for v4-flash. OpenRouter's displayed
  price, $0.119 / $0.238, is one provider's price, and providers range from $0.06 to $0.44.
- claude-haiku-5.5 is priced at $0.10 / $0.50 (claude.com/pricing).
- gemini-3.1-flash-lite and gemini-3-flash-preview use Google's standard prices.
- gemma-4-31b-it is priced at OpenRouter's listed $0.20 / $0.40.

Every benchmark has equal weight. The table gives each candidate alone (arena-style score on our set, not a
RouterArena score):

| model | accuracy % | $ / 1K | score |
|---|---:|---:|---:|
| gemini-3.8-flash (reasoning low) | 91.3 | 3.768 | 80.55 |
| gemini-3.7-flash (reasoning low) | 91.2 | 3.385 | 80.88 |
| deepseek-v4-flash-0731 | 88.6 | 0.975 | 82.69 |
| gemini-3-flash-preview (reasoning none) | 87.6 | 0.965 | 81.93 |
| claude-haiku-5.5 | 87.1 | 0.423 | 83.19 |
| gpt-6-luna | 86.7 | 0.234 | 83.80 |
| gemma-4-31b-it | 85.3 | 0.185 | 82.98 |
| gemini-3.1-flash-lite (reasoning none) | 84.4 | 0.457 | 80.79 |

## Decisions, in the order they were made
1. **Pool.** We dropped gemini-3.7-flash and gemini-3.8-flash.
   - Google's pricing page does not list 3.7 Flash, so it has no official price source.
   - 3.8 Flash is on an introductory price.
   - A cross-validated router over all 8, using the cost regressor of step 2, scored the same with or without the
     two at λ ≥ 0.12 on every seed (`explore_v4.py`-style CV, 3 seeds). At those settings it never picked them.

   The pool is the other 6 candidates.
2. **Policy.** We compared single models, draft-and-escalate cascades (two cheap drafts, escalate on
   disagreement), 2-of-3 majority voting and a learned router (`explore_v4.py`, 5-fold CV). The learned router with
   a per-query cost estimate was best.
   - *Correction, disclosed:* the first CV runs charged each candidate the realised cost of the held-out item. That
     cost is unknown when routing, and it inflated the router by about 1 point. We replaced it with a cost
     regressor fitted on the training folds before any decision was taken from those runs.
3. **Prompt-format robustness.** We wrote a second prompt format of our own, variant B
   (`prompts_v4b.py`: a system message, different wording, a `\boxed{}` answer). We ran the 6 pool models on 840
   items in that format (`run_variant_v4.py`, $2.35; item ids in
   [`robust_item_ids.txt`](robust_item_ids.txt)).
   - A router fitted on variant-A labels still beat every single model under variant-B labels (`robust_v4.py`).
   - A router trained only on bare question text lost about 0.3 to 0.6 points on an unseen wrapper. Training on
     several renderings removed that loss.
4. **Router.** Implemented in [`pipeline/fit_router_v4.py`](pipeline/fit_router_v4.py); inference is in
   [`pipeline/route_v4.py`](pipeline/route_v4.py).
   - **Probability of a correct answer:** for each pool model, a logistic regression on a bge-small-en-v1.5
     embedding of the query text.
   - **Expected cost:** a ridge regression on log cost over [embedding, log text length], converted with the
     lognormal mean.
   - **Decision:** route to argmax P − λ·K.
   - **Training text:** every item appears three times: bare content, prompt variant A and prompt variant B. At
     inference the text is embedded exactly as received (its last 4,000 characters), with no parsing or stripping
     of any wrapper.
5. **λ.** We chose λ by leave-one-wrapper-out CV: train on two renderings, test on the third, 5 folds × 2 seeds.
   The grid value with the highest mean wins. Means: 0.06 → 84.20, **0.08 → 84.29**, 0.12 → 84.21,
   0.16 → 84.19, 0.24 → 83.89. The best single model scores 83.80. Full log:
   [`fit_router_v4.log`](fit_router_v4.log).

## Frozen artefacts
| file | sha256 |
|---|---|
| router_v4.npz (pool, λ = 0.08, all weights) | `cdd8063c303b250c04f52efae463c2d1404712daaf4ce4ec039d279f770c7b66` |
| pipeline/route_v4.py | `41c85d24fb6241b64d6700aca22351595661246830e6f9d0bb3b33b2f2edff40` |
| pipeline/ra_predict_v4.py | `9641bef63e42e1c873de9dbe9b58cdcd4a16e4404aa538eb6e39aa351e26188d` |
| pipeline/fit_router_v4.py | `e9eedfd3c497c7aa999c067c0ba2101670a991bbaeaec51092da71df0ec47974` |
| pipeline/analyze_full_v4.py | `746b3bde283a2fc8342fe1f6a84c1970a85055d562735b93741bf049c13a708a` |
| pipeline/explore_v4.py | `40579f0d89f3fe4b0758bf498afc9af89d002d38c2f4239e36eb9179a2223329` |
| pipeline/prompts_v4b.py | `134866303e5bbf199febdaebae7d703a23edc51731fb591821b379517f543c09` |
| pipeline/run_variant_v4.py | `3e4d15d9c875a209bdad83552c7c2d3c8c46cf7c1dfee5f3e6ea4b37b32d4a42` |
| pipeline/robust_v4.py | `67d20d3d3d639481f6f574a9580c1dc4c1664953bc8bf4b39a5c6460e0b68555` |
| prices_standard.json | `954708866fccd266993fad4a2cd647fe42efeefd69f86832a2e5d02c7184a207` |
| lcb_graded.jsonl | `73daa320c571c39f145ecc6a331e02f3664aedfde8799db56599b6db324f75b2` |

Part 2's prompts_v4.py, grade_v4.py, lcb_grade_v4.py and run_v4.py are unchanged; their hashes match part 2 and
its amendment.

## The RouterArena run (after this commit, once)
1. `ra_predict_v4.py --split full` and `--split robustness`. The prompt text each query carries in the dataset is
   routed by the frozen router.
2. RouterArena's own `llm_inference/run.py` makes one fresh call per routed query through OpenRouter. "-reasoning-
   none" selects the same model with reasoning effort none. No cached answers from our earlier v1 to v3 runs are
   used.
3. The prediction files, the config and the price entries for new models, with official price links, are
   submitted once. The submission discloses our v1 to v3 RouterArena runs and what was changed after seeing them
   (see the README's version history).

Any failed calls are retried with the same model. No routing decision changes after this commit.
