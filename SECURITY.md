# Security Policy

## Reporting a vulnerability

Report privately via GitHub Security Advisories
("Security" tab → "Report a vulnerability"). Please do not open a public issue.

Expect an acknowledgement within 72 hours.

## Supply chain commitments

This package is in a category that has suffered a credential-harvesting
compromise on PyPI, so these are stated as commitments rather than intentions:

- **Publishing**: PyPI Trusted Publishing (OIDC). No long-lived API token exists.
- **Attestations**: every release carries PEP 740 attestations, verifiable
  against this repository and the workflow that built it.
- **Install-time execution**: none. No build hooks, no post-install scripts.
- **Runtime dependencies**: one (`httpx`). Every addition is a deliberate,
  reviewed decision.
- **Price data**: vendored as a SHA-pinned snapshot. No network call at import.

## Verifying a release

```bash
pip download argrouter --no-deps -d /tmp/t
# PyPI publishes attestations alongside the wheel; check the project's
# "Provenance" section on pypi.org/project/argrouter/
```
