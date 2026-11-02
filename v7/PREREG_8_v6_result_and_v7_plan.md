# argrouter pre-registration, part 8: the v6 result and the v7 plan

Date: 2026-11-03. This plan is committed before any full-set label of the new v7 candidates, or any QANTA
start-of-question label, is looked at. Parts 6 and 7 are commits 6576333 and 39b7417.

## The v6 result (run once, as part 7 said)
The frozen v6 router routed RouterArena's full split. RouterArena's `llm_inference/run.py` made 9,209 fresh calls:
- 572 calls failed with HTTP 429 and were retried with the same model; 0 failures remained.
- RouterArena's `llm_evaluation/run.py` scored them.

| metric | v6 |
|---|---:|
| RouterArena score | 76.18 |
| accuracy | 80.08 % |
| cost per 1K queries | $0.824 |
| robustness | 73.57 |

**v6 is not submitted.** We continue to a v7 and will submit once, disclosing v1 to v6.

**What we looked at.** After v6 we looked at RouterArena's per-benchmark accuracy of the v6 run, for example
QANTA 51.6 %, NarrativeQA 47.7 %, MMLU-Pro 86.9 %. Two of the decisions below react to it:
- the QANTA start-of-question items;
- the stronger candidates.

That is tuning on RouterArena results, and we disclose it here. No RouterArena query, template, config or
per-query result is used to build data.

## Candidate triage (done before this commit, disclosed)
`bench/v7/build_triage.py` took 312 items from our own calibration set:
- the 12 items per benchmark (LiveCodeBench left out) that the 8 fixed v4 reference models most often got wrong;
- 60 QANTA questions reduced to their first sentence.

Thirteen models, none of them reference models, were called once on each ($27.01). Selected results:

| model | hard 252 % | QANTA first sentence % | $ / 1K on these items |
|---|---:|---:|---:|
| google/gemini-4-argon | 31.3 | 83.3 | 5.71 |
| minimax/minimax-m3.1-flash | 26.2 | 31.7 | 2.16 |
| google/gemini-3.8-flash@medium | 25.4 | 81.7 | 5.40 |
| openai/gpt-6.1-terra | 23.4 | 81.7 | 1.71 |
| google/gemini-3.7-flash@low (v6 pool) | 20.2 | 80.0 | 2.10 |
| deepseek/deepseek-v4.1-flash (v6 pool) | 19.4 | 66.7 | 2.46 |

The other models were too expensive for what they added:
- mistral-large-4 ($19.29 / 1K);
- kimi-k2.6 ($35.43 / 1K);
- deepseek-v4-pro ($19.05 / 1K);
- gemini-3.7-flash@medium.

Three others were weaker than the two pool models on both columns: beam-1.0, nemotron-3.5-super and
deepseek-v4.1-flash@high. Single-provider models were not tried (part 6 rule): particle-1, mercury-3,
glm-5.3-flashx, command-a-plus, qwen3.8-omni-flash, muse-spark-1.3.

Of the 252 hard items, **111 were answered correctly by none of the 13 models**. Spot checks show these are
mostly gold-label errors in the source datasets or grader misses. Examples:
- a MathQA item whose listed answer is wrong;
- a GeoBench true/false item;
- NarrativeQA "empty and without purpose" against "it is purposeless".

## What v7 changes
1. **Three new candidates**, each called once on all 4,264 calibration items with the unchanged part-2 prompts
   and runner. Code answers are graded with the part-2 sandbox grader.
   - **openai/gpt-6.1-terra** at $1.00 / $5.00 (5 providers).
   - **google/gemini-4-argon** at $2.00 / $10.00 (6 providers). That is the list price; every provider currently
     shows a 50 % discount, which we do not use.
   - **minimax/minimax-m3.1-flash** at $0.15 / $0.60 (14 providers). OpenRouter lists $0.075 / $0.30 only
     between 15:00 and 01:00 UTC and $0.15 / $0.60 otherwise. We use the higher price and register it so in
     RouterArena.
2. **QANTA start-of-question items.** QANTA's own evaluation scores a system after the first sentence as well as
   at the end of the question (Rodriguez et al. 2019).
   - We add, for each of our 200 QANTA items, the same question cut to the release's `first_sentence` field,
     with the same answer, aliases and grader (`bench/v7/build_qanta_s1.py`).
   - Every candidate is called once on these 200 items.
   - QANTA keeps its one benchmark share, split equally between the two settings.
3. **Label noise.** Items that no candidate answers correctly get weight 0 in the objective. They cannot change
   which model is picked, and with the part-4 difficulty weights they would get the largest weight.

## What does not change
- the v4 grader;
- prices of the existing candidates (`v5/prices_ra.json`);
- the difficulty weights (floor 0.5, the 8 fixed v4 reference models). The new QANTA items use the error of the
  same 8 models on the first-sentence item itself, since all 8 are called on it;
- the 2-provider rule of part 6;
- the forward selection (5 folds, 1 seed, λ grid, stop under +0.10 or at 6 models);
- leave-one-wrapper-out λ;
- the router class.

We also checked whether NarrativeQA and WMT19 should be graded with their official continuous metrics (ROUGE-L,
BLEU) instead of the v4 pass rules. On the existing labels it does not change the ranking of candidates on those
benchmarks, so the grader stays as it is.

## Then
Part 9 freezes the v7 router (hashes, selection log) before it routes any RouterArena query. Then there is one
RouterArena run and one submission, disclosing v1 to v6 and parts 1 to 9.
