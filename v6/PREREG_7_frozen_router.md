# argrouter pre-registration, part 7: the frozen v6 router

Date: 2026-11-02. Committed before the v6 router routes any RouterArena query. Part 6 (commit 6576333) excluded
nex-n2.5-pro for having a single provider. Everything below is the part-4 rule, rerun on the remaining 14
candidates. The labels, prices, weights and code are those of part 5.

## Selection
Full log: [`fit_router_v6.log`](fit_router_v6.log). Command:
`fit_router_v5.py --prices prices_ra.json --only <the 14 candidates>`.

**Forward selection:**
1. google/gemini-3.7-flash@low: objective 75.16. It is the best single candidate once nex is gone.
2. deepseek/deepseek-v4.1-flash: objective 76.01. The runners-up were glm-5.3-flash at 75.81 and
   deepseek-v4.1-flash-1017 at 75.79.
3. Stop. The best third addition, glm-5.3-flash, adds 0.00.

**λ**, leave-one-wrapper-out over 5 folds × 2 seeds, under hard weights:

| λ | 0.005 | 0.01 | 0.02 | **0.04** | 0.08 | 0.16 |
|---|---:|---:|---:|---:|---:|---:|
| hard | 75.41 | 75.62 | 75.65 | **75.73** | 75.69 | 75.63 |
| equal | 83.64 | 83.89 | 84.01 | 84.16 | 84.19 | 84.17 |

**On our set**, rendered with prompt variant A, the router sends 2,769 of 4,264 items to deepseek-v4.1-flash and 1,495 to gemini-3.7-flash@low. The
gemini picks include all 200 chess items and are otherwise spread across the benchmarks (SocialIQA 135,
PubMedQA 97, NarrativeQA 95, ...).

**Expected RouterArena score.** The v5 LOWO score was 76.65; this pool's is 75.73. v4 scored 73.95 under the
same weights and measured 73.56 on RouterArena. By that rough calibration, v6 should land around 75 to 75.5.
That is below the current #1. We run it anyway: the rule picks the router, not the expected rank.

## Frozen artefacts
| file | sha256 |
|---|---|
| router_v6.npz (pool, λ = 0.04, all weights; RouterV4 format) | `dfe06e0557f7d275d1768df7c6cc6814d10d449d9f8d824965e12ce082c745c0` |
| ../v4/pipeline/route_v4.py (unchanged router class) | `41c85d24fb6241b64d6700aca22351595661246830e6f9d0bb3b33b2f2edff40` |
| pipeline/ra_predict_v6.py | `3554b11d9a8bb95c36f1b0048277e8fadb27e8b6eb7e5ebf7cff830ff610ad88` |
| ../v5/pipeline/fit_router_v5.py (unchanged) | `0a2960ee9c9346b79cbcee197e26d404cfc0b286403156be2ad2b899f2ce76af` |
| fit_router_v6.log | `316c8d3d61602d6fea9afa35436de32d8e0b429411b3bddb1fc9732f2339b54b` |

The prices and code labels are part 5's ([`../v5/prices_ra.json`](../v5/prices_ra.json),
[`../v5/lcb_graded_v5.jsonl`](../v5/lcb_graded_v5.jsonl)), and their hashes are unchanged.

## The RouterArena run (after this commit, once)
The procedure is part 5's, with this router:
- RouterArena main 44cc1ff with the model_inference.py port for "@effort";
- the alias google/gemini-3.7-flash@low on the existing gemini-3.7-flash profile ($0.75 / $3.75);
- deepseek-v4.1-flash at its existing profile price ($0.045 / $0.50).

Every call is a fresh call. Failed calls are retried with the same model, and no routing decision changes after
this commit.
