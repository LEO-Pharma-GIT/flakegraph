# Security Policy

## Reporting A Vulnerability

Please do not open a public issue for a security problem. Report it through
GitHub's private vulnerability reporting for this repository:

https://github.com/LEO-Pharma-R-D-Data-Analytics/flakegraph/security/advisories/new

Include what you found, how to reproduce it, and which component it affects
(the Python package, the console under `react/`, the Helm chart under
`deploy/`, or a container image). A report is acknowledged within five
working days. We will work with you on a fix and coordinate disclosure; you
will be credited in the advisory unless you prefer not to be.

## Supported Versions

Security fixes land on `main` and in the next release. Older releases are
not patched separately.

## Scope

FlakeGraph processes documents and writes what it extracts, including source
text, to local artifacts or Snowflake tables, and it calls the OCR, LLM and
embedding providers a deployment configures. Findings that are particularly
relevant:

- Credentials or tokens reaching logs, artifacts, run reports, staged
  specifications or job tables (the redaction boundaries in
  `src/kg_processor/application/redaction.py`).
- Path handling that lets a manifest, upload or configuration read or
  replace files outside the intended directory.
- Authentication or authorisation gaps in the console, the inference
  sidecar or the OCR shim.
- Chart defaults that expose a service without the intended gate.

Deployments run behind an operator's own network, identity provider and
provider accounts; problems in those systems are out of scope here but we
are happy to point you to the right place.
