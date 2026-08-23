# AGENTS.md

Working notes for an AI assistant in this repo: the things that are easy to get
wrong. The README says what the project is; this does not repeat it.

Verified 2026-08-23 against `main`. Every `diagrid` CLI claim was checked against
**CLI v1.66.0** (API server 1.93.0) signed in to production — run
`diagrid version` and re-check if yours is newer.

## Three distributions, one namespace package

| Source | PyPI distribution | Contents |
|---|---|---|
| `diagrid/agent/**` + root `pyproject.toml` | `diagrid` | framework integrations; owns the `diagridpy` script |
| `diagrid/core/` | `diagrid-core` | auth, Catalyst API client, config |
| `diagrid/cli/` | `diagrid-cli` | the `click` + `rich` CLI |

All three carry the same version and ship together: `pypi-release.yaml` is a
manual `workflow_dispatch` that bumps all three `pyproject.toml` files and the
Helm chart, commits and tags straight on `main`, then publishes three wheels.

- **`diagrid/` and `diagrid/agent/` have no `__init__.py`, on purpose.** They are
  implicit namespace packages so three separately built wheels can install into
  one import path. Adding `diagrid/__init__.py` or `diagrid/agent/__init__.py`
  breaks that split, and it is why `mypy.ini` needs `namespace_packages` and
  `explicit_package_bases`. `diagrid/core/__init__.py` and
  `diagrid/agent/core/__init__.py` do exist — those are ordinary packages.
- The root wheel **excludes** `diagrid.core*` and `diagrid.cli*`
  (`[tool.setuptools.packages.find].exclude`); they ship as their own wheels.
- The console script is **`diagridpy`**, not `diagrid`. `diagrid` is the separate
  Go Catalyst CLI (see *Catalyst*). `diagridpy` has three commands: `init`,
  `deploy`, `chaos`.
- uv workspace members: `diagrid/core`, `diagrid/cli`, and every
  `examples/<framework>` **except** `examples/holmesgpt`. `--all-packages`
  installs the examples too.

## Extras, and the holmesgpt fork

Eleven framework extras — `langgraph`, `crewai`, `adk`, `strands`,
`openai_agents`, `claude_agents`, `pydantic_ai`, `deepagents`, `langchain`,
`smolagents`, `holmesgpt` — plus `agent-core` (shared runtime) and `all`. Note
that PyPI normalises the underscores, so `diagrid[openai-agents]` and
`diagrid[openai_agents]` are the same extra.

- **`all` deliberately omits `holmesgpt`.** HolmesGPT's pins on
  fastapi/uvicorn/cachetools/mcp/httpx cannot coexist with the other frameworks,
  so `[tool.uv].conflicts` declares it incompatible with every other extra
  *including* `all`. Asking for both cannot resolve. Sync it alone, without
  `--all-packages` (the example members drag in conflicting extras):
  `uv sync --extra holmesgpt --group test`.
- Every pin and every entry in `override-dependencies` carries a comment saying
  which resolution failure it exists for — the whole OTel stack held at
  1.39.1/0.60b1, `numpy<2.5`, `crewai<1.16`, `mcp==1.26.0`. Read the comment
  before "tidying" one; these are not style choices.
- **`dapr-agents` is a git dependency tracking `main`**, marked TEMPORARY in
  `[tool.uv.sources]`. It is unpinned, so `uv lock --upgrade` moves it to
  whatever `main` was that day. `deps-check.yaml` runs `uv lock --check` on every
  PR, so an out-of-date lock fails CI.
- `[tool.uv] exclude-newer = "1 days"` — a fresh resolution ignores anything
  published in the last day. Invisible when installing from the lock; surprising
  when `uv lock --upgrade` skips a release you just watched land.
- `requires-python = ">=3.11,<3.14"`, `.python-version` pins the local
  interpreter to **3.13**, and `mypy.ini` targets **3.11**. CI builds 3.11, 3.12
  and 3.13 on Linux and 3.13 on Windows, so a 3.11-only type error is a real
  failure even though your venv is 3.13.

## What CI runs — and three commands that are not what they look like

PR jobs: `lint`, `build` (×4 OS/Python), `holmesgpt` (×2), `deps-check` (×4),
`e2e-ollama`, `sast`, `codeql`. The two Helm jobs are path-filtered to
`charts/**` and `charts-tests/**`. Reproduce the Python side with exactly this:

```bash
uv sync --all-packages --extra all --group test
uv run ruff format                                              # no path args
uv run flake8 diagrid tests --ignore=E501,F401,W503,E203,E704
uv run mypy --config-file mypy.ini
uv run pytest tests -m "not integration"
```

- **The CI format gate is bare `uv run ruff format`, with no path arguments.**
  It covers every tracked `.py`, `examples/` included. `make format`, the
  pre-push hook and `CONTRIBUTING.md` all run `ruff format diagrid tests`, which
  misses exactly the 38 files under `examples/`. So an `examples/` change can
  pass every local check and still fail `lint`, which reformats and then fails if
  `git status` is dirty. Run the bare form before pushing.
- **The flake8 `--ignore` list is load-bearing and lives in no config file.** It
  is repeated verbatim in the `Makefile`, `.pre-commit-config.yaml` and
  `build.yaml`. Bare `flake8 diagrid tests` reports over a thousand findings CI
  does not care about — E501 (line length) and F401 (unused imports) are both
  switched off here.
- **`ruff` is the formatter only.** Nothing runs `ruff check`; flake8 is the
  linter. Do not "fix" `ruff check` findings as if they were CI failures.
- **`-m "not integration"` is the only thing keeping the unit run unit-sized.**
  There is no `addopts` in `[tool.pytest.ini_options]`, so plain `pytest tests`
  collects the lot, including `tests/e2e/` (needs a Dapr sidecar) and
  `tests/cli/utils/test_deps_functional.py` (downloads kind/kubectl/helm over the
  network). `make test` and `make test-cov` use the plain form; **`make test-unit`
  is the target that matches CI.**
- **Always give pytest a path.** Bare `pytest` from the repo root dies with
  collection errors: `examples/*/test_crash_recovery.py` and `test_retry.py` are
  standalone `dapr run` scripts, not pytest tests, despite the names.
- **`mypy` pip-installs stub packages into the venv on first run**
  (`install_types` + `non_interactive` in `mypy.ini`) — a couple of dozen
  `types-*` wheels, outside uv's lock. The next `uv sync` prunes them and the
  next mypy reinstalls them, so a slow first mypy after a sync is normal, not a
  hang. `mypy.ini` sets `files = diagrid/**/*.py`: **`tests/` is not
  type-checked.**
- **`tests/agent/holmesgpt/` silently collects zero tests** in the normal
  `--extra all` venv — its `conftest.py` sets `collect_ignore_glob` when `holmes`
  is not importable. A green `pytest tests` says nothing about HolmesGPT; the
  separate `holmesgpt` CI job is what covers it.
- Pre-commit hooks run at **pre-push**, not pre-commit: `make hooks-install` runs
  `pre-commit install --hook-type pre-push`.

## The test tree's two traps

- **Never name a directory under `tests/` after a top-level import package.**
  `pythonpath = ["."]` plus pytest's prepend import mode puts `tests/agent/` on
  `sys.path`, so `tests/agent/crewai/__init__.py` shadows the real `crewai`.
  Four directories collide today — `crewai`, `langgraph`, `pydantic_ai`,
  `deepagents` — and the workarounds are the price: `_import_unshadowed` in
  `tests/e2e/conftest.py` and the `sys.path` / `__path__` surgery in
  `tests/agent/deepagents/conftest.py`. Three later ones dodge it by suffix
  (`langchain_ext`, `smolagents_ext`, `strands_ext`) precisely because
  `langchain`, `smolagents` and `strands` *are* importable top-level names. The
  rest are safe only by accident: `adk`, `openai_agents`, `claude_agents` and
  `holmesgpt` import as `google.adk`, `agents`, `claude_agent_sdk` and `holmes`.
  Check with `python -c "import <name>"` and use the `_ext` suffix if it resolves.
- **`DaprClient()` blocks for 60 seconds with no sidecar.** The autouse fixture
  that patches `DaprHealth.wait_for_sidecar` lives in `tests/agent/conftest.py`
  and therefore covers only `tests/agent/`. A new test under `tests/cli/` or
  `tests/core/` that constructs a `DaprClient` will look like a hang. Mock it, or
  put the test under `tests/agent/`.

## Running the e2e suite

`tests/e2e/` needs **no Catalyst credentials** — no API key, no project, nothing
in `~/.diagrid`. It runs against a local `dapr run` sidecar plus a local
OpenAI-compatible endpoint, which is exactly what `e2e-ollama.yaml` does on every
PR: `dapr init`, Ollama with `qwen3:0.6b`, then `pytest tests/e2e/ -m integration`.

```bash
dapr init                                    # once; needs Docker
DAPR_E2E=1 uv run pytest tests/e2e -m "integration and not ollama" -v
# with an LLM:
OLLAMA_ENDPOINT=http://localhost:11434/v1 OLLAMA_MODEL=qwen3:0.6b \
  OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 \
  uv run pytest tests/e2e -m integration -v
```

- **With none of those env vars set, the whole suite skips and exits 0.**
  `tests/e2e/conftest.py` starts a sidecar only when `OLLAMA_ENDPOINT` or
  `DAPR_E2E` is set, and `skip_without_prerequisites` skips any test whose Dapr
  port is closed. Green is not the same as passed — read the skip count.
- The fixture owns app-id `e2e-test` on fixed ports 50001/3500 and will
  `dapr stop` whatever is already using them. Do not run it next to your own
  `dapr run`.
- Markers: `integration` (Dapr sidecar, no LLM), `ollama` (Dapr + LLM), `chaos`
  (Chaos Mesh on a cluster, gated on `CHAOS_ENABLED`). Only `integration` is
  declared in `pyproject.toml`; `ollama` and `chaos` are registered in
  `tests/e2e/conftest.py`.

## Runtime model

- **The workflow name is a wire identifier, and it embeds the agent name.**
  `build_workflow_name` in `diagrid/agent/core/workflow/naming.py` produces
  `dapr.<framework-lowercased>.<TitleCaseName>.workflow`, and
  `sanitize_agent_name` TitleCases the agent name (`catering-coordinator` →
  `CateringCoordinator`). So **renaming an agent renames its registered
  workflow** and in-flight instances cannot replay.
- **Never guess an activity name — read the `register_activity` call.** There is
  no convention. Some frameworks use bare names (`call_llm_activity`,
  `execute_tool_activity`, `execute_node_activity`,
  `evaluate_condition_activity`), some prefix them
  (`langchain_call_llm_activity`, `smolagents_call_model_activity`,
  `strands_call_model`, `claude_call_llm`, `holmes_call_llm`), and
  `DaprAgentWorkflow` in `diagrid/agent/strands/durable_agent.py` is the one that
  builds them from the agent name — `dapr.strands.<TitleCaseName>.call_model` and
  `.execute_tool` — so **there, renaming the agent moves the activities too.**
- **Every workflow↔activity hop crosses JSON.** The `workflow.py` in each
  framework package passes `.to_dict()` into `ctx.call_activity` and calls
  `.from_dict()` on the result; Dapr serialises in between. Channel values must
  be JSON-native, or install a custom serialiser with `set_serializer()`. Values
  come back as the types JSON produces, not the types you put in.
- **Component discovery is by exact name and fails silently.**
  `diagrid/agent/core/discovery.py` calls `DaprClient.get_metadata()` once (cached
  for the process) and matches components named exactly `agent-configuration`,
  `agent-memory`, `agent-pubsub`, `agent-registry`, `agent-runtime`. On any error
  it returns empty defaults with no exception. There is **no env-var override** —
  unlike the Go SDK you cannot point it at a differently named store. None of the
  local component files here declare those names
  (`tests/e2e/resources/statestore.yaml` is `statestore`,
  `examples/*/resources/statestore.yaml` is `agent-workflow`), so local runs
  discover nothing, by design.
- `DaprStateStore` defaults to store name `agent-memory`
  (`diagrid/agent/core/state/store.py`). `DaprClient` is created lazily and
  reused; do not open and close one per call.
- Telemetry is off unless configured. `diagrid/agent/core/observability.py` reads
  `OTEL_SDK_DISABLED`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`,
  `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_{LOGGING,TRACING}_ENABLED` and
  `OTEL_{LOGS,TRACES}_EXPORTER`.
- `diagridpy init` calls `preflight_check()`, which requires **five** binaries —
  `docker`, `kind`, `kubectl`, `helm`, `piko` — and prompts (`click.confirm`)
  before installing the missing ones. It will abort rather than proceed in a
  non-interactive shell.

## Catalyst — only what a contributor here needs (CLI v1.66.0, 2026-08-23)

- **`diagridpy` (this repo) and `diagrid` (the Go Catalyst CLI) are different
  tools.** Nothing in `tests/` and nothing in CI talks to Catalyst; the local
  loop is `dapr run` plus kind. Catalyst enters only via `diagridpy init` and
  `diagridpy deploy`, which authenticate by device code against
  `https://api.r1.diagrid.io`.
- **Run `diagrid project list` before creating anything** — an org normally
  already has an auto-provisioned `default` project, and
  `diagridpy init --existing-project` selects one instead of adding another.
- **`--enable-agent-infrastructure` is not on `diagrid project create` at all**,
  and on `project update` it is rejected for a cloud project with the managed KV
  store — it applies to BYOC/private-region projects, or cloud projects without it.
- **`diagrid agent` fronts *your* app; `diagrid managed-agent` is a
  Catalyst-hosted Durable Agent**, hidden from `diagrid --help` and restricted to
  Diagrid employees. An unqualified "create an agent" picks the wrong one.
- **Never print a project's `.status.apiToken`** — it is a live credential. Pass
  it by reference (`--api-key`, `DIAGRID_API_KEY`); keep it out of logs and PRs.
- Deeper Catalyst detail lives in the `diagridio/catalyst-ai` plugin. That repo
  is private, so nothing here depends on it — the above stands on its own.

## What is not tested

The unit run is entirely offline and finishes in about a minute. What it never
reaches:

- **`diagrid/cli/infra/`: only `helm.py` and `kind.py` have a test file.**
  `catalyst_operator.py`, `chaos.py`, `docker.py` and `kubectl.py` have none —
  and those are the modules that shell out to real binaries.
- **`diagrid/cli/commands/chaos.py` has no test file** (`deploy.py` and `init.py`
  do). Neither does `diagrid/cli/utils/console.py` or `process.py`.
- **`diagrid/core/auth/token.py` and `diagrid/core/catalyst/appids.py` have no
  test file** — token refresh and App ID creation, both auth-critical.
- **`diagrid/agent/core/plugins/context.py` and `spi.py` have no test file**
  (only `registry.py`), and neither does
  `diagrid/agent/core/metadata/mixins.py`.
- `tests/cli/utils/test_deps_functional.py` really downloads `kind`, `kubectl`
  and `helm` — but **not `piko`**, which is the one downloader that resolves an
  unpinned upstream `latest` tag (`andydunstall/piko`) and so is the one most
  likely to break.
- **Most framework runners are untested.** Only `claude_agents`, `deepagents`,
  `pydantic_ai` and `holmesgpt` have a `test_runner.py`; `adk`, `crewai`,
  `langchain`, `langgraph`, `openai_agents`, `smolagents` and `strands` do not.
  Nor do `adk/plugin.py`, `strands/durable_agent.py` or `strands/hooks.py`.
- Every sidecar-touching path is covered only by `tests/e2e/`, which needs a live
  Dapr. If you change `discovery.py`, `state/store.py`, `pubsub/` or any
  `runner.py`, the unit suite cannot catch the regression — run the e2e suite.

## Conventions

- Every file under `diagrid/` opens with the two-line BUSL header
  (`Copyright (c) 2026-Present Diagrid Inc.` /
  `SPDX-License-Identifier: BUSL-1.1`). Nothing enforces it — put it on new
  files yourself.
- Type annotations on public signatures; `from __future__ import annotations` is
  the house style. Every package under `diagrid/` carries an empty `py.typed`
  marker — add one to any new subpackage. A new subpackage of `diagrid-core` or
  `diagrid-cli` also has to be listed in that distribution's
  `[tool.setuptools] packages` array; the root `diagrid` distribution discovers
  its own automatically.
- Pydantic v2 `BaseModel` with `Field(alias=...)` for camelCase API fields and
  `model_config = {"populate_by_name": True}`. Dataclasses with
  `to_dict()`/`from_dict()` for anything crossing an activity boundary.
- Sign off your commits: `git commit -s`. Conventional-commit subjects are the
  norm. Neither is enforced by a check, and the history is mixed on both — sign
  off anyway.
- `main` is **not** branch-protected and no status check is required, so a red PR
  can be merged. Do not lean on the gate — run the five commands above yourself.
