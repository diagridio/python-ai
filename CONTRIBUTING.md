# Contributing

## Prerequisites

The following tools are required for local development:

- **Python ≥ 3.11, < 3.14** — `.python-version` pins the local interpreter to 3.13
- **[uv](https://docs.astral.sh/uv/)** — package manager and virtual environment tool
- **Docker** (daemon running)
- **kind** — local Kubernetes clusters
- **kubectl** — Kubernetes CLI
- **helm** — Kubernetes package manager
- **piko** — tunnelling binary used by the generated quickstart projects

> **Note:** `diagridpy init` runs a preflight check for all five binaries
> (`docker`, `kind`, `kubectl`, `helm`, `piko`) and *prompts* before installing
> any that are missing — it aborts rather than proceeding in a non-interactive
> shell. See `REQUIRED_BINARIES` in `diagrid/cli/utils/deps.py`.

## Setup

```bash
git clone <repo>
cd ai-python
uv sync --all-packages --extra all --group test --group dev
```

## Running unit tests

There is no `addopts` in `[tool.pytest.ini_options]`, so nothing is excluded by
default: `-m "not integration"` is what makes a run unit-only. Without it,
pytest also collects `tests/e2e/` (needs a running Dapr sidecar) and
`tests/cli/utils/test_deps_functional.py` (downloads binaries over the network).

| What | Command |
|------|---------|
| Unit tests only — same as CI | `make test-unit` or `uv run pytest tests -m "not integration"` |
| Everything, including e2e + functional | `uv run pytest tests` or `make test` |
| Everything, with a coverage report | `make test-cov` |

Always pass `tests` as the path. Bare `pytest` from the repo root fails
collection: the `examples/*/test_crash_recovery.py` and
`examples/*/test_retry.py` scripts are standalone `dapr run` programs, not
pytest tests, despite the names.

## Running integration / functional tests locally

Integration tests perform real network downloads and require internet access:

```bash
uv run pytest tests/cli/utils/test_deps_functional.py -m integration -v
```

What each test does:

- Downloads `kind`, `kubectl`, `helm` binaries from their upstream release URLs into a temporary directory
- Verifies the file is non-empty, has the executable bit set (Linux/macOS), and the binary responds to `--version`
- `test_docker_daemon_running_on_linux` is skipped on non-Linux

Expected runtime: ~2–3 minutes depending on network speed.

## Linting and type-checking

```bash
make format      # uv run ruff format diagrid tests
make lint        # uv run flake8 diagrid tests ...
make typecheck   # uv run mypy --config-file mypy.ini
```

Or individually:

```bash
uv run ruff format
uv run flake8 diagrid tests --ignore=E501,F401,W503,E203,E704
uv run mypy --config-file mypy.ini
```

> **Note:** the CI format gate (`.github/workflows/build.yaml`) runs
> `uv run ruff format` with **no path arguments**, which covers `examples/` too.
> `make format` and the pre-push hook narrow it to `diagrid tests`, so a change
> under `examples/` can pass every local check and still fail CI. Run the bare
> form before pushing.
>
> `ruff` is only the formatter here — nothing runs `ruff check`. Linting is
> flake8, and its `--ignore` list lives in no config file: it is repeated in the
> `Makefile`, `.pre-commit-config.yaml` and `build.yaml`, so `flake8` without it
> reports a large number of findings CI does not enforce.

## Pre-commit hooks

Hooks run at `pre-push` (not pre-commit) and cover: trailing whitespace, YAML lint,
ruff format, flake8, mypy, and the unit test suite.

```bash
make hooks-install   # installs the pre-push hook into .git/hooks/
make hooks-run       # runs all hooks against all files right now
```

## Local Kubernetes / Helm development

```bash
make cluster-up      # start registry mirrors + create kind cluster
make helm-install    # helm upgrade --install catalyst-agents from local chart
make cluster-down    # kind delete cluster
make helm-test       # lint + template validation + chainsaw e2e (requires kind cluster)
```

The chainsaw e2e suite (`make helm-test-chainsaw`) creates its own `catalyst-agents-test`
kind cluster, runs the Chainsaw tests against it, and deletes the cluster on exit.
