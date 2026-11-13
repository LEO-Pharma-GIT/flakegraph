# Contributing to FlakeGraph

Thank you for considering a contribution. This guide covers how to set up a
development environment, what a change is expected to carry, and how it gets
reviewed.

## Development Setup

FlakeGraph needs Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
The console needs [Bun](https://bun.sh/).

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check src tests
uv run mypy src
uv run pytest
```

The unit suite runs without any external service. Integration tests that need
a service skip themselves unless the corresponding variable is set:

| Marker | Variable | Needs |
| --- | --- | --- |
| `live` | `KG_RUN_<PROVIDER>_LIVE=1` | A reachable provider endpoint and credentials |
| `postgres` | `KG_TEST_POSTGRES_DSN` | A PostgreSQL database the tests may create schemas in |
| `spark` | `KG_RUN_SPARK_INTEGRATION=1` | The `distributed-spark` extra and a Java 17+ runtime |
| `helm` | none | The `helm` binary on `PATH` |

Deselect a group with `uv run pytest -m "not helm"`.

For the console:

```bash
cd react
bun install
bun run lint
bun run test
```

## What A Change Should Carry

- Tests. Behaviour changes come with a unit test that would fail without
  them; contract tests under `tests/unit` pin documentation, configuration
  profiles and deployment artifacts, so update them with the thing they pin.
- Docstrings and comments that explain why, in prose. Every production
  module, class and function carries a docstring (ruff enforces it), and a
  comment should say what a reader could not work out from the code alone.
- No secrets, account identifiers, hostnames or internal names in code,
  configuration, fixtures or documentation. Profiles reference credentials
  through `${KG_*}` placeholders only.
- Dependency changes update `uv.lock` (`uv lock`) and, when they add a
  package, `THIRD_PARTY_NOTICES.md`. Required dependencies must use permissive
  licences unless a weak-copyleft exception is recorded in the
  [Copyleft Summary](THIRD_PARTY_NOTICES.md#copyleft-summary): the LGPL-3.0
  `psycopg`, `psycopg-binary` and `psycopg-pool` packages and the MPL-2.0
  `certifi` bundle. Strong-copyleft dependencies belong in opt-in extras; the
  `leiden` extra shows that pattern.
- Every Python module under `src/kg_processor` starts with an
  `# SPDX-License-Identifier: Apache-2.0` header.

## Commits And Pull Requests

- Keep commits focused; one logical change per commit, with a subject line
  that says what the change does.
- Open pull requests against `main`. Describe what changed and why, and how
  it was verified.
- CI runs the lint, type and test steps above; a green run is required
  before review.

## Licensing Of Contributions

By contributing you agree that your contribution is licensed under the
[Apache License 2.0](LICENSE) that covers the project, as described in
section 5 of that licence. No separate contributor licence agreement is
required.

## Reporting Problems

Use GitHub issues for bugs and feature requests. Security problems go
through the process in [SECURITY.md](SECURITY.md), not a public issue.

## Code Of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
