# thriftllm — route every LLM call to the cheapest model that can actually do it

Most "cost-aware" routers rank models by `input_rate + output_rate`. That sum
assumes a 1:1 input/output ratio, and almost no real workload has one. thriftllm
computes the **expected cost of this specific request** — forecast output length,
tiered rates, cache state, reasoning tokens — and ranks on **$ per quality point**.

```python
from thriftllm.catalog.pricing import Catalog, expected_cost

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
pip install thriftllm      # one dependency: httpx
```

> **Status: pre-release.** The pricing core and its tests are written. The
> router, the benchmark and the measured savings number are not. This README
> states no savings percentage because none has been measured yet — see
> [PRE-REGISTRATION.md](PRE-REGISTRATION.md) for the claim we have committed to
> making and the margin we committed to before running anything.

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

thriftllm's position is narrow and checkable: **be the project whose cost number
is right.** Not another selection algorithm — a correct denominator.

## Does routing break prompt caching?

Usually yes, and this is the strongest argument against routers in general.
Prompt caches are **model-scoped**. Cache reads cost 0.1× base input — and as
little as 0.025× on some models — so switching models mid-session can forfeit a
75–97% discount to chase a smaller routing saving.

thriftllm treats this as a first-class cost term rather than a footnote:

- **Cache state is priced at decision time.** The forfeited cache discount is
  subtracted from the candidate's expected saving *before* it is compared.
- **Minimum-cacheable thresholds are honoured.** Below the per-model minimum,
  providers silently do not cache and charge full rate. thriftllm records that
  explicitly instead of quietly over-estimating the discount.
- **Session affinity** keeps a conversation on its model unless the measured
  saving exceeds the measured cache loss.

When we publish a benchmark, the baseline runs with **caching fully enabled and
warmed**, and **the cache-hit rate of every arm is published next to its cost**.
A cost claim measured against a cache-disabled baseline is void, and we would
rather say that ourselves than have it said in a comment thread.

## When thriftllm will not help you

- **Long agentic sessions on one model with a warm cache.** Keep the cache. Use
  cost tracking only.
- **Uniformly hard workloads.** Published routers beat random routing by ~14% on
  knowledge-dense benchmarks while still needing the frontier model for over half
  of calls. Some traffic is simply not routable.
- **Single-shape workloads.** If every request looks the same, choose the model
  once at build time. You do not need a router.
- **Workloads where output *style* matters.** Switching models changes voice.

You should know this before you install it, not after.

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
| thriftllm | $0.01874 | — |

What this means: if you call open-weight models through OpenRouter, **turning on
any price-aware provider sort saves ~40% over the default**, and that is most of
the win. thriftllm's mix-aware ranking only changes the pick when two providers'
input/output prices cross (here: `gpt-oss-120b`, 1–4% cheaper on the output-heavy
workloads, worse on RAG whenever its pick was busy); the rest of the
gap between price-aware strategies is availability noise, not ranking. thriftllm
also enforces a quality floor (≥ fp8, ≥ 99% uptime) that the price sort does not.

We have pre-committed to publishing the result **when routing loses**.

## Install

```bash
pip install thriftllm          # core: httpx only
pip install "thriftllm[server]"  # optional /v1/decision sidecar
```

Python 3.10+. Fully typed, `py.typed` shipped.

## How it fits with your existing gateway

thriftllm does **not** reimplement the OpenAI wire format. Over half of the open
issues on the largest gateway in this space are request-translation bugs; that is
a maintenance burden with no upside for a routing project.

Instead it is a decision layer: a `/v1/decision` sidecar and plugins for gateways
that already own the bytes. **We win the decision; your gateway keeps the
transport.**

## Supply chain

The dominant package in this category was compromised on PyPI, and that is a
standing cost to everyone shipping here. thriftllm commits to:

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
vocabulary this project is measured in.

## License

Apache-2.0. Contributions under [DCO](https://developercertificate.org/)
(`git commit -s`). See [LICENSE](LICENSE).
