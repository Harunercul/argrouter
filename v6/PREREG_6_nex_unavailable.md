# argrouter pre-registration, part 6: v5 could not be run, nex-n2.5-pro is excluded

Date: 2026-11-02. Committed before the v6 router exists and before any v5 RouterArena call is scored.

## What happened to the v5 run
The frozen v5 router (part 5, commit 91d9eb5) routed RouterArena's full split. Its pool was nex-agi/nex-n2.5-pro
and google/gemini-3.8-flash@low. RouterArena's `llm_inference/run.py` then called the routed models.
- Every gemini-3.8-flash@low call succeeded.
- nex-n2.5-pro has a single provider on OpenRouter (Nex AGI). From the second day of the run, most of its calls
  failed with `HTTP 400 routed_dp_rank=N out of range [0, 1)`, a provider-side error.
- We retried with the same model, as part 5 says: 11 passes over about 6 hours, 5 minutes apart. The failures
  went from 1,250 to 233, but in the last pass only 27 of 260 calls succeeded.
- We stopped there. **The v5 run is not scored.** `llm_evaluation/run.py` was not run on it, and no v5 answer
  was looked at.

A router that cannot answer its queries cannot be submitted. Its pool was chosen without regard to availability,
which part 4 did not model.

## The one change
**A candidate must be served by at least 2 providers on OpenRouter**, checked on 2026-11-02 with OpenRouter's
`/api/v1/models/{id}/endpoints`. nex-n2.5-pro has 1 provider. Every other candidate has 5 or more:

| model | endpoints |
|---|---:|
| anthropic/claude-haiku-5.5 | 10 |
| deepseek/deepseek-v4-flash-0731 | 29 |
| google/gemini-3-flash-preview | 6 |
| google/gemini-3.1-flash-lite | 8 |
| google/gemini-3.7-flash | 7 |
| google/gemini-3.8-flash | 7 |
| google/gemma-4-31b-it | 18 |
| openai/gpt-6-luna | 7 |
| openai/gpt-6-luna-pro | 6 |
| deepseek/deepseek-v4.1-flash | 40 |
| deepseek/deepseek-v4.1-flash-1017 | 44 |
| xiaomi/mimo-v2.6-flash | 9 |
| xiaomi/mimo-v2.6-pro | 5 |
| z-ai/glm-5.3-flash | 40 |
| **nex-agi/nex-n2.5-pro** | **1** |

So nex-n2.5-pro is the only candidate excluded, together with qwen3.8-flash, which was already excluded in part 5.

Nothing else changes. These all stay exactly as in parts 4 and 5:
- labels, prices and weights (floor 0.5);
- the forward-selection rule and its stopping rule;
- the λ grid and the leave-one-wrapper-out choice of λ;
- the router class and the scripts.

The rule is rerun with `fit_router_v5.py --only` on the remaining 14 candidates, with its output in `v6/`.

## Then
Part 7 freezes the resulting router, with its hashes and the selection log, before it routes any RouterArena
query. After that we make one RouterArena run and one submission. The submission discloses v1 to v4, v4's 73.56,
this unscored v5 run and parts 1 to 7.
