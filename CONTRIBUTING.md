# Contributing

## Prerequisites

The following tools are required for local development:

- **Python ≥ 3.11**
- **[uv](https://docs.astral.sh/uv/)** — package manager and virtual environment tool
- **Docker** (daemon running)
- **kind** — local Kubernetes clusters
- **kubectl** — Kubernetes CLI
- **helm** — Kubernetes package manager

> **Note:** Running `diagridpy init` will auto-install `kind`, `kubectl`, and `helm` if they are missing.

## Setup

```bash
git clone <repo>
cd ai-python
uv sync --all-packages --group test --group dev
```

## Running unit tests

| What | Command |
|------|---------|
| All unit tests (default) | `uv run pytest tests` |
| Unit tests only (explicit) | `make test-unit` or `uv run pytest tests -m "not integration"` |
| With coverage report | `make test-cov` |
| Verbose + short tracebacks | `make test` |

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
uv run ruff format diagrid tests
uv run flake8 diagrid tests --ignore=E501,F401,W503,E203,E704
uv run mypy --config-file mypy.ini
```

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
