# Benchmark pre-registration

Committed **before** any paid benchmark run. Nothing below may change without a
dated amendment appended to this file explaining why.

This document exists because cost-router benchmarks are easy to game and the
category has, deservedly, stopped believing them. Published and commercial
routers have repeatedly failed to beat trivial baselines under independent
evaluation. We state our claim, our margin and our baselines up front so that a
skeptical reader can check whether we moved the goalposts afterwards.

---

## 1. Headline claim (the only number that goes in the README)

> **CSR@NI(δ)** — "*N%* all-in cost reduction versus the always-strong baseline,
> with quality statistically non-inferior at a pre-registered margin
> δ = **1.0 accuracy point**, by paired TOST at 95% confidence, n = …"

Four properties, each chosen to close a known attack:

| Property | Closes |
|---|---|
| Reported in **dollars** | "fraction of strong-model calls isn't money" |
| **All-in** (router overhead included) | "you didn't count the router" |
| **Non-inferiority test**, not a failed significance test | "absence of evidence isn't evidence of absence" |
| δ **pre-registered** | "you picked the margin after seeing the curve" |

Reported **per stratum and traffic-weighted**. Never pooled-only.
We will never headline a maximum over datasets.

## 2. Baselines we must beat

Tier 1 — omitting any of these invalidates the result:

1. **Always-strong** — quality ceiling, cost denominator
2. **Always-cheap** — cost floor
3. **Zero Router** — random mixture along the convex hull of individual models
4. **Best Single** — the single highest-average-quality model, chosen in hindsight
5. **Random router** at matched strong-call fraction
6. **Oracle** (per-item cheapest-correct) and **anti-oracle**, as the achievable range

Tier 2 — "is routing even the right tool?" Nobody in the literature runs these;
an infrastructure engineer will ask within five minutes:

7. **Strong model at reduced reasoning effort**
8. **Prompt caching + prompt hygiene alone**, on always-strong
9. **Batch API** on always-strong, for the latency-tolerant share

> **Caching is run ENABLED AND WARMED on every arm, and the cache-hit rate of
> each arm is published alongside its cost.** A cost claim measured against a
> cache-disabled baseline is not a weak claim, it is a void one. Routing
> fragments model-scoped caches, so this is the measurement most likely to go
> against us — which is exactly why it is pre-registered.

## 3. Cost accounting

Three figures published together, never one alone:

- **Inference cost** — model calls only (what everyone else reports)
- **All-in cost** — inference + classifier + verifier + every retried attempt
  + router training amortized over a stated request volume, with the
  amortization curve shown
- **Cost per completed task** — all-in ÷ items that actually passed

Recorded per `(item, rep, call)` from the provider's own response, never
estimated. `usd_cost_recorded` must reconcile against `usd_cost_recomputed`.
Token-count tables are published separately from dollars so readers can
recompute with their own prices after ours go stale.

## 4. Quality grading

- **Tier 1 — verifiable (target ≥55% of items).** Exact match with
  normalization, unit tests, constraint checkers. **The headline is computed on
  Tier 1+2 only.** A reader who rejects LLM judges entirely still has a number.
- **Tier 2 — programmatic-structural.** Atomic-claim entailment with a fixed,
  separately-validated NLI model; schema validity; refusal detection.
- **Tier 3 — judged.** Secondary metric only. Pairwise, blinded,
  position-swapped with verdict flips scored as ties and the conflict rate
  published; length-controlled; judge drawn from outside both routed model
  families; a second-family judge re-run as sensitivity; human-labeled
  calibration subset with κ published including the disagreements.

## 5. Statistics

Paired per-item differences (not two independent summary scores). Clustered
standard errors where items share a source or a conversation, with cluster
counts reported. Power analysis run **before** spending. Any stratum whose noise
floor exceeds δ cannot support a claim and will be reported as inconclusive
rather than quietly pooled away.

## 6. Data discipline

Train/validation/test split drawn at random, stratified by stratum tag and
**never** by baseline score. RouterArena and RouterBench are reserved strictly
for external validation: never used for training, threshold tuning or model
selection. Decontamination report published with method, threshold and the
matches removed.

## 7. Failure disclosure

We publish the result when routing **loses**. Strata where routing does not pay
are reported as such and named in the README. The project's claim is that its
cost number is correct — not that routing always wins.

## 8. Reproducibility

Pinned dataset revisions by commit SHA, exact model snapshot IDs with
`model_served` asserted per call, all seeds, verbatim prompts and judge
templates, the full cost ledger, every raw model output, an `errors.jsonl`
sidecar kept separate from results, and notebooks that regenerate every number
and figure — including the headline — from the raw rows.

---

*Amendments must be appended below with a date and a reason. None yet.*
