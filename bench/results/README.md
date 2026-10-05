# Provider-selection benchmark

`bench/provider_bench.py` — same model, four ways of choosing the provider, every
call billed by OpenRouter (`usage.cost`, not an estimate).

| arm | what it sends |
|---|---|
| `or_default` | no `provider` field — what most users get |
| `or_price` | `{"sort": "price", "quantizations": ["fp8","int8","bf16","fp16","fp32"]}` |
| `naive_sum` | same eligibility policy as thrift, providers ordered by input + output price |
| `thrift` | same policy, ordered by expected cost with a learned output-length forecast |

Models: `openai/gpt-oss-120b`, `z-ai/glm-5.3-flash`, `deepseek/deepseek-v4-flash-0731`,
`qwen/qwen3-235b-a22b-2507`. Workloads: `rag` (~2.3k in / ~100 out),
`gen` (~100 in / 1200 out), `balanced` (~900 in / ~450 out). 4 interleaved
repetitions per run, temperature 0, concurrency 6.

| run | or_default | or_price | naive_sum | thrift | failed (thrift/naive) |
|---|---:|---:|---:|---:|---|
| [2026-10-06 00:48](provider_bench_20261006-004828.md) | $0.01469 | $0.00903 | $0.00845 | $0.00833 | 3 / 3 |
| [2026-10-06 01:02](provider_bench_20261006-010257.md) | $0.01654 | $0.00989 | $0.01040 | $0.01041 | 0 / 1 |

## Findings

1. **Default routing costs ~67% more than any price-aware strategy.** This is the
   one large, repeatable effect.
2. **Between price-aware strategies the difference is within noise.** The winner
   flipped between runs. Most cost variance comes from which provider is
   rate-limited at that moment, not from the ranking rule.
3. **Mix-aware ranking matters only where providers' price ratios cross.** Among
   these four models that happened once: `gpt-oss-120b`, where DekaLLM is cheaper
   on input and DeepInfra on output. thrift picks per workload; the price sort does
   not.
4. **Run 1 found a real bug, fixed in run 2.** Pinning only the 3 cheapest
   providers with `allow_fallbacks: false` failed 6 requests when the cheapest
   provider was rate-limited upstream. `Selection.openrouter_provider()` now lists
   every eligible provider by default: 6 failures → 0.
5. The price sort is not quality-equivalent: on `glm-5.3-flash` it was served by
   providers that thrift's policy rejects (uptime 96%, status degraded).

## Limits

Small sample (~$0.09 total spend), one time window, OpenRouter only. Prompt
caching was active for every arm (cached tokens appear in all four), so it is not
a confound between arms but does lower absolute costs. Raw rows are in the
`.jsonl` files; endpoint snapshots taken at run start are in `.endpoints.json`.
