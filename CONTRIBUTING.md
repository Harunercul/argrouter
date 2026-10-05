# Contributing

## Sign-off (DCO), not a CLA

Contributions are accepted under the
[Developer Certificate of Origin](https://developercertificate.org/). Sign your
commits:

```bash
git commit -s -m "your message"
```

**Why a DCO and not a CLA.** A CLA exists to let a maintainer relicense your
work later. We do not intend to do that: core contributions stay under
Apache-2.0 permanently. Apache-2.0 §5 already gives this project inbound code
on the same terms as outbound, together with each contributor's §3 patent
grant — so a CLA would add a legal hurdle without adding anything we need.

If this project ever has commercial components, they will be written by the
maintainers in a separately-licensed directory, not by relicensing anything you
contributed. That directory does not exist today, and the license will stay a
clean `Apache-2.0` identifier until it does.

## Before you open a PR

```bash
uv sync --all-extras --group dev   # or: pip install -e . && pip install pytest mypy ruff
pytest                             # 90% coverage gate is enforced
mypy src/
ruff check .
```

`pip` is supported as a first-class path throughout. `uv` is what the
maintainers use day to day; nothing in this project requires it.

## What we especially want

- **Price-catalog corrections.** If a model's cache multiplier, context tier or
  per-request fee is wrong in the snapshot, that is a real bug — expected cost
  is the whole point of this project.
- **Counter-examples to the routing thesis.** Workloads where routing loses. We
  publish those; see `PRE-REGISTRATION.md`.
- **Benchmark scrutiny.** If a baseline looks untuned or a cost figure looks
  flattering, say so in an issue. That critique is more valuable here than a
  feature.
