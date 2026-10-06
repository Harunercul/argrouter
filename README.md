# argrouter — pick the LLM, and how hard it should think, per query

argrouter routes each request to the model **and reasoning effort** that maximise
`P(correct) − λ · expected cost`, where the cost is what that model actually bills on
similar requests (hidden reasoning tokens included), not its list price. Two models with
the same price tag can differ 10× in real cost because one of them thinks for 3,000
tokens before answering.

**On [RouterArena](https://github.com/RouteWorks/RouterArena)** (ICLR 2026, the open router
leaderboard) argrouter scores **76.30**, ahead of the current #1 (76.28), at
**$0.56 per 1,000 queries**. [Details below.](#routerarena)

The cost engine underneath is usable on its own:

```python
from argrouter.catalog.pricing import Catalog, expected_cost

catalog = Catalog.from_snapshot("prices.json")        # vendored, offline, SHA-pinned
model   = catalog.get("anthropic/claude-sonnet-5.5")  # unknown id raises; never $0

cost = expected_cost(
    model,
    input_tokens=20_000,            # a RAG call: long prompt, short answer
    expected_output_tokens=500,     # forecast, not a YAML constant
    cached_input_tokens=18_000,     # cache state enters the decision
)

print(cost.total_usd)      # 0.0141
print(cost.as_dict())      # every component separately auditable
```

```bash
pip install argrouter      # one dependency: httpx
```

> **Status: alpha.** The cost engine, provider selection and router inference API are
> released and tested. Router training and the trained weights are not part of the
> open-source package.

---

## Why cost-aware LLM routing usually gets the cost wrong

Output tokens are typically 60–90% of an LLM bill, and the field does not
forecast them. Three failures are common, and all three are in shipped code
today:

| Failure | Consequence |
|---|---|
| Adding `input_rate + output_rate` | Ranking is wrong for any workload that isn't 1:1 |
| A constant `expected_output_tokens` in YAML | The number that dominates the bill is a guess you typed |
| Unpriced model treated as `$0` | The model nobody priced always wins "cheapest" |

argrouter's position is narrow and checkable: **be the project whose cost number
is right.** Not another selection algorithm — a correct denominator.

## Does routing break prompt caching?

Usually yes, and this is the strongest argument against routers in general.
Prompt caches are **model-scoped**. Cache reads cost 0.1× base input — and as
little as 0.025× on some models — so switching models mid-session can forfeit a
75–97% discount to chase a smaller routing saving.

argrouter treats this as a first-class cost term rather than a footnote:

- **Cache state is priced at decision time.** The forfeited cache discount is
  subtracted from the candidate's expected saving *before* it is compared.
- **Minimum-cacheable thresholds are honoured.** Below the per-model minimum,
  providers silently do not cache and charge full rate. argrouter records that
  explicitly instead of quietly over-estimating the discount.
- **Session affinity** keeps a conversation on its model unless the measured
  saving exceeds the measured cache loss.

When we publish a benchmark, the baseline runs with **caching fully enabled and
warmed**, and **the cache-hit rate of every arm is published next to its cost**.
A cost claim measured against a cache-disabled baseline is void, and we would
rather say that ourselves than have it said in a comment thread.

## When argrouter will not help you

- **Long agentic sessions on one model with a warm cache.** Keep the cache. Use
  cost tracking only.
- **Uniformly hard workloads.** Published routers beat random routing by ~14% on
  knowledge-dense benchmarks while still needing the frontier model for over half
  of calls. Some traffic is simply not routable.
- **Single-shape workloads.** If every request looks the same, choose the model
  once at build time. You do not need a router.
- **Workloads where output *style* matters.** Switching models changes voice.

You should know this before you install it, not after.

## RouterArena

[RouterArena](https://github.com/RouteWorks/RouterArena) (ICLR 2026) scores routers on 8,400
queries from 23 public benchmarks: accuracy, weighted with log-scaled cost (β = 0.1).
argrouter on all 8,400 queries, scored with RouterArena's own evaluation code:

| router | arena | accuracy | $ / 1K queries |
|---|---:|---:|---:|
| **argrouter** | **76.30** | 79.40% | 0.56 |
| KT-ModelRouter (current #1) | 76.28 | 78.14% | 0.27 |
| Sqwish Router (#2) | 76.21 | 79.76% | 0.70 |
| Divyam (#3) | 75.85 | 78.59% | 0.48 |
| NotDiamond (powers OpenRouter Auto) | 57.29 | 60.83% | 4.10 |
| RouteLLM | 48.07 | 47.04% | 0.27 |

**How it was run.** The pool is five models; reasoning effort is part of the choice
(gemini-3.8-flash at low effort, gemma-4-31b, glm-5.3-flash, gpt-6-luna,
mimo-v2.6-flash), chosen by greedy forward selection. The router reads **only the
question content**: the instruction and answer-format paragraphs are dropped with generic
rules, and no RouterArena file is read. It was trained on 3,562 held-out items from the
same public sources, every RouterArena item excluded, with sources weighted equally.

**What we got wrong along the way**, because a leaderboard number is only worth its
history:

| version | arena | why it is not the submission |
|---|---:|---|
| v1 | 75.90 | first run |
| v2 | 77.42 | routed on full prompts (which include RouterArena's per-dataset instruction text) and weighted by RouterArena's dataset mix. Both count as fitting to RouterArena under its rules ([#213](https://github.com/RouteWorks/RouterArena/pull/213)), so v2 was withdrawn before submission. |
| **v3** | **76.30** | content-only routing, equal source weights; λ and pool fixed before the run. |

Routing on content only did not cost accuracy on our own data; it was the equal source
weighting that shifted traffic to cheaper models (cost −34%, accuracy −2.3 points vs v2).
Nothing was tuned after seeing a RouterArena result. Submission to the official
leaderboard is pending.

## How much does it actually save?

**Model routing:** not measured yet. When it is, the number will be reported as
all-in cost (including this library's own overhead and every retried call), per
stratum, against **the best fixed single model** — not against the most expensive
model in the pool, which is how savings in this category are usually inflated.

**Provider selection (same model, many providers): measured, and the honest answer
is "about the same as OpenRouter's own price sort".** Two runs, 377 billed calls,
4 open-weight models × 3 workloads, every cost taken from OpenRouter's `usage.cost`
([raw data and method](bench/results/README.md)):

| strategy | billed, both runs | vs thrift |
|---|---:|---:|
| OpenRouter default routing | $0.03123 | +67% |
| OpenRouter `sort: price` + same quantization floor | $0.01892 | +1% |
| rank providers by input + output price (LiteLLM/Plano rule) | $0.01885 | +1% |
| argrouter | $0.01874 | — |

What this means: if you call open-weight models through OpenRouter, **turning on
any price-aware provider sort saves ~40% over the default**, and that is most of
the win. argrouter's mix-aware ranking only changes the pick when two providers'
input/output prices cross (here: `gpt-oss-120b`, 1–4% cheaper on the output-heavy
workloads, worse on RAG whenever its pick was busy); the rest of the
gap between price-aware strategies is availability noise, not ranking. argrouter
also enforces a quality floor (≥ fp8, ≥ 99% uptime) that the price sort does not.

We have pre-committed to publishing the result **when routing loses**.

## Install

```bash
pip install argrouter          # core: httpx only
pip install "argrouter[server]"  # optional /v1/decision sidecar
```

Python 3.10+. Fully typed, `py.typed` shipped.

## How it fits with your existing gateway

argrouter does **not** reimplement the OpenAI wire format. Over half of the open
issues on the largest gateway in this space are request-translation bugs; that is
a maintenance burden with no upside for a routing project.

Instead it is a decision layer: a `/v1/decision` sidecar and plugins for gateways
that already own the bytes. **We win the decision; your gateway keeps the
transport.**

## Supply chain

The dominant package in this category was compromised on PyPI, and that is a
standing cost to everyone shipping here. argrouter commits to:

- **One runtime dependency** (`httpx`)
- **PyPI Trusted Publishing** with **PEP 740 attestations** — no long-lived token
- **No install-time code execution**
- A **vendored, SHA-pinned price snapshot** — no network call at import

## Prior art and credit

The price catalog is seeded from [OpenRouter](https://openrouter.ai/)'s public
models API, which co-locates pricing with quality indices, and from
[models.dev](https://models.dev/). The candidate-filtering and scoring shape is
informed by
[vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)
(Apache-2.0), which is the strongest open implementation of multi-factor
selection. [RouteLLM](https://github.com/lm-sys/RouteLLM) defined the evaluation
vocabulary this project is measured in, and
[RouterArena](https://github.com/RouteWorks/RouterArena) provides the benchmark, scorer
and price table the leaderboard result above uses.

This project was briefly published as `thriftllm`. It was renamed to avoid confusion with
[ThriftLLM](https://arxiv.org/abs/2501.04901) (Huang et al., 2025), an unrelated paper on
budget-constrained LLM ensemble selection, and with the thriftllm.com gateway.

## License

Apache-2.0. Contributions under [DCO](https://developercertificate.org/)
(`git commit -s`). See [LICENSE](LICENSE).
