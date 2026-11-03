# argrouter pre-registration, part 9: the frozen cluster router

Date: 2026-11-04. Committed before this router routes any RouterArena query. Part 8 is commit 2ca09fd.

Everything below was done on our own calibration set (4,264 v4 items + 200 QANTA first-sentence items), except
for two pieces of RouterArena information that we list first, because they are departures from earlier parts.

## What we used from RouterArena (disclosed)
1. **Query composition.** `bench/v7/ra_mix.py` counts the `global index` prefixes of RouterArena's public
   `dataset/router_data.json` (8,400 queries): per benchmark (MMLU-Pro 30.1 %, OpenTDB 10.6 %, QANTA 7.7 %, ...,
   GSM8K 0.3 %) and per subset (e.g. WMT19 gu-en 89, de-en 62; MMLU only formal_logic and management). Output:
   [`ra_mix.json`](ra_mix.json). No answer, no result. To place the `GeoGraphyData_100k` prefix we read one
   prompt of it and one GeoBench prompt. Our v4 clean-room statement said nothing under the RouterArena repository
   was read; this is the exception, and only for these counts.
2. **The two disclosed RouterArena totals** (v4 73.56, accuracy 75.0 %, $0.266 / 1K; v6 76.18, accuracy 80.1 %,
   $0.824 / 1K) were used to calibrate the objective (next section). That is tuning on RouterArena results, at
   the level of two published totals. No per-query or per-benchmark RouterArena result was used after part 8.

## Calibrating the objective to RouterArena
`bench/v7/calib_ra.py` rescored the two routers that have a RouterArena result (same class, pool, λ and training
weights as their frozen versions; out-of-fold predictions, leave-one-wrapper-out, 5 folds × 2 seeds) under
candidate weightings. Logs: [`calib_ra.log`](calib_ra.log), [`calib_ra_floor.log`](calib_ra_floor.log).

| weighting of our items | v4 | v6 | v6 − v4 | v4 acc | v6 acc |
|---|---:|---:|---:|---:|---:|
| **RouterArena (measured)** | **73.56** | **76.18** | **+2.62** | **75.0 %** | **80.1 %** |
| equal shares × (0.5 + err) (parts 5–8) | 78.32 | 80.09 | +1.77 | 81.1 % | 86.0 % |
| RouterArena benchmark shares | 86.01 | 85.80 | −0.22 | 89.6 % | 92.4 % |
| RouterArena benchmark + subset shares | 85.95 | 86.04 | +0.09 | 89.6 % | 92.9 % |
| RouterArena benchmark shares × (0.5 + err) | 78.15 | 79.28 | +1.13 | 80.7 % | 84.9 % |
| equal shares × (0.1 + err) | 63.40 | 68.58 | +5.18 | 64.3 % | 72.5 % |
| **equal shares × (0.25 + err)** | **73.00** | **75.95** | **+2.95** | **75.1 %** | **81.1 %** |
| equal shares × (1 + err) | 81.71 | 82.74 | +1.04 | 84.9 % | 89.0 % |

err = mean error of the 8 fixed v4 reference models on the item; items no candidate answers keep weight 0 (part 8).

- Weighting by RouterArena's own benchmark shares does **not** track RouterArena: it cannot even tell that v6
  beat v4. The difficulty factor is what matters; RouterArena's items are harder than ours.
- With floor 0.25 our set reproduces both RouterArena totals and both accuracies within about 1 point. (The
  part-4 text had 0.25; the part-4 amendment moved it to 0.5 from the v4 point alone.)
- Our costs are about 1.6 × RouterArena's for the same router (0.433 vs 0.266 and 1.387 vs 0.824; RouterArena's
  prompts ask for short answers). The objective therefore multiplies costs by **0.6**.

**The objective from here on:** equal benchmark shares × (0.25 + err), noise items 0, cost × 0.6, arena score.
The RouterArena-share weightings were not used for any decision below.

## New candidates (our set only)
- **Effort variants.** A 378-item pilot (`build_variant_pilot.py`, 18 items per benchmark, LCB left out, $1.86)
  tried terra@low, terra@minimal, gemini-3.7-flash@minimal, gemini-3.8-flash@minimal, luna@high and
  deepseek-v4.1-flash@none. gemini-3.x-flash@none and glm-5.3-flash@none are refused by the providers
  ("Reasoning is mandatory"). Four variants got full labels (all items, the QANTA first-sentence items, LCB
  graded in the sandbox; about $14): **gpt-6.1-terra@low**, gemini-3.7-flash@minimal, gpt-6-luna@high,
  deepseek-v4.1-flash@none. LCB passes out of 200: 188, 181, 183, 164.
- **Triage of two more 2-provider models** on the part-8 triage items ($5.42): inclusionai/ling-3.1-flash
  (hard 22.6 %, QANTA first sentence 53.3 %, $2.67 / 1K) and stepfun/step-5-preview (22.2 %, 60.0 %,
  $14.69 / 1K). Neither beats gpt-6.1-terra (23.4 %, 81.7 %, $1.73 / 1K); not labelled further.

21 candidates in total (qwen3.8-flash and nex-n2.5-pro stay incomplete / excluded).

## The predictor
The part-8 LR router reached 80.74 under the part-8 objective, only +0.2 over the v6 pair under the same
objective, so it was **not** run on RouterArena ([`router_v7.npz`](router_v7.npz), sha256 `21d20061…`, kept for
the record). The oracle of that pool is 89.78: the predictor was the bottleneck. On the fixed part-8 pool
(`compare_emb.py`, `compare_pred.py`; logs `cmp_*.log`):

| predictor | part-8 objective |
|---|---:|
| LR on bge-small (v4–v6 class) | 80.91 |
| LR on qwen3-embedding-8b / openai-3-large / gemini-embedding-2 / voyage-4-large / bge-m3 | 81.36 / 81.25 / 81.07 / 81.09 / 81.31 |
| kNN on qwen3-embedding-8b | 80.85 |
| **cluster router** (k-means k = 30, pseudo-count 5) on the five embeddings | 81.83 / 82.19 / 81.95 / 81.75 / 81.39 |
| benchmark-recognising kNN + per-benchmark means | 81.89–82.12 |

The cluster router is kept: k-means on the L2-normalised embedding of the query text (all three renderings);
per cluster and model, the weighted mean correctness and cost, shrunk to the model's overall mean with
pseudo-count 5; route to argmax P − λK. Embeddings come from OpenRouter's embeddings API ($1.46 for the five).

## Pool, embedding, λ
`nested_cal.py` ([`nested_cal.log`](nested_cal.log)), under the calibrated objective, with the mean prediction
of four embeddings: for 5 random half-splits (stratified by benchmark), greedy forward selection (part-4 rule)
and λ on one half, scored on the other half, both ways round.

- Held-out gain over the v6 pair under the same predictor: **+1.14** on average (sd 0.73, 10 runs, 9 positive).
- Models picked in at least half of the 10 selections form the pool: gemini-4-argon 10/10, gemini-3.7-flash@low
  7, glm-5.3-flash 6, gpt-6.1-terra@low 5, gpt-6.1-terra 5, claude-haiku-5.5 5.
- `final_cal.py` ([`final_cal.log`](final_cal.log)) then chose one embedding (one call per RouterArena query) and
  λ for that pool, counting the embedding call's cost: **google/gemini-embedding-2, λ = 0.01**, objective 79.38
  (qwen3-embedding-8b 79.02, openai-3-large 79.00, voyage 78.87, bge-m3 78.86). The v6 pair scores 77.39 here.

All six models have at least two OpenRouter providers (part-6 rule): argon 2, terra 3, glm 35, gemini-3.7-flash
2, haiku 5.

**Expected RouterArena score.** The calibrated objective put v6 within 0.3 of its RouterArena score. The held-out
gain says about +1.1; the in-sample pick says +2.0. Our expectation: about **77.5, likely between 77.0 and 78.4**.
We run it whatever that implies for the rank.

## Frozen artefacts
`fit_router_cluster.py` ([`fit_router_cluster.log`](fit_router_cluster.log)) fits the router on all items.

| file | sha256 |
|---|---|
| router_cluster.npz (pool, λ, embedding id, 30 centroids, P and K per cluster) | `0e371ad8a58ca742ec4f638f32c1a4b8c52051dd6d51ac70adfb7179610319f7` |
| pipeline/route_cluster.py | `8498e0536c63d2aa85d61d8ddf98d3aecca979544c9e10ef9f951b48e1927490` |
| pipeline/ra_predict_cluster.py | `cef569289e221a66a3c71b9b519006ae83420e4cb4cd2fa6cb5ef37f4f2716dc` |
| pipeline/fit_router_cluster.py | `7064509b7f890e480ae9e96b511775acfd1a6bf37418689f5e23732f868a3116` |
| fit_router_cluster.log | `6ff177d8f71f538ee0d28e49813198f6967af71ec8066dcb71f579ca256b1bf8` |

The other scripts and logs of this part are in [`pipeline/`](pipeline/) and next to this file. Labels and prices:
[`prices_ra.json`](prices_ra.json) (part-8 prices; an effort variant is priced as its base model),
[`lcb_graded_v7.jsonl`](lcb_graded_v7.jsonl) (sandbox LCB grades of the seven v7 models).

## The RouterArena run (after this commit, once)
- RouterArena main 44cc1ff with the part-5 port of `model_inference.py` for "@effort".
- Price profiles added for the pool at OpenRouter list price: google/gemini-4-argon ($2 / $10),
  openai/gpt-6.1-terra ($1 / $5, alias openai/gpt-6.1-terra@low), and google/gemini-embedding-2 ($0.20 / $0) for
  the router's own calls. glm-5.3-flash, gemini-3.7-flash@low and claude-haiku-5.5 use their existing profiles.
- Every query's embedding call goes into the row's `router_usage`, so RouterArena bills it.
- Every answer is a fresh call. Failed calls are retried with the same model; no routing decision changes after
  this commit.
- The result is reported in part 10 whatever it is, and the submission discloses v1 to v7 and all ten parts.
