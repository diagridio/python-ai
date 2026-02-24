# CLAUDE.md — ai-python Repository Instructions

## Project Overview

UV workspace monorepo with 3 Python packages (Python 3.11+):

- **diagrid** (root) — Agent framework integrations (LangGraph, CrewAI, ADK, Strands, OpenAI Agents) with Dapr runtime
- **diagrid-core** (`diagrid/core/`) — Shared auth, Catalyst API client, config
- **diagrid-cli** (`diagrid/cli/`) — CLI tool (click + rich)

## Build & Quality Commands

```bash
# Install all dependencies
uv sync --all-packages

# Install with test deps and all framework extras
uv sync --all-packages --extra all --group test

# Format (auto-fixes in place)
uv run ruff format

# Lint
uv run flake8 diagrid tests --ignore=E501,F401,W503,E203,E704

# Type check
uv run mypy --config-file mypy.ini

# Unit tests (excludes integration)
uv run pytest tests -m "not integration"
```

## Code Conventions

- **Type annotations**: Required on all public function signatures
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: stdlib → third-party → local, one import per line for local modules
- **Models**: Pydantic v2 `BaseModel` with `Field(alias=...)` for camelCase API fields; use `model_config = {"populate_by_name": True}`
- **Dapr patterns**: Lazy-initialized `DaprClient` (reusable, avoid repeated open/close), JSON serialization for state/pubsub
- **Dataclasses**: Used for internal workflow models (LangGraph); provide `to_dict()`/`from_dict()` serialization
- **Telemetry**: OpenTelemetry SDK with gRPC OTLP exporter; no-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset

## Agent Guardrails

### CI Fix Agent (`claude-ci-fix.yaml`)
- **CAN**: Fix formatting issues, lint errors, type errors, and test failures reported by CI
- **CAN**: Run `uv run ruff format`, `uv run flake8`, `uv run mypy`, `uv run pytest`
- **CANNOT**: Refactor code beyond what's needed to fix the specific failure
- **CANNOT**: Add new features or change behavior
- **CANNOT**: Modify files listed in the "Never Modify" section

### Review Agent (`claude-pr-review.yaml`)
- **CAN**: Read code, run diffs, analyze changes
- **CAN**: Post structured review comments categorized as architectural or non-architectural
- **CANNOT**: Modify any files
- **CANNOT**: Push commits

### Implementation Agent (`claude-implement.yaml`)
- **CAN**: Implement ONLY the specific non-architectural suggestions listed in its prompt
- **CAN**: Run quality commands to verify changes don't break anything
- **CANNOT**: Make architectural changes (new patterns, restructuring, new dependencies)
- **CANNOT**: Implement suggestions not explicitly listed in its prompt
- **CANNOT**: Modify files listed in the "Never Modify" section

## Never Modify

These files must not be changed by any agent:

- `pyproject.toml` (root and package-level)
- `uv.lock`
- `mypy.ini`
- `charts/` (Helm charts)
- `.github/workflows/` (workflow definitions)
- Do not add new dependencies
- Do not change public APIs in `diagrid/core/`
- Do not delete or skip existing tests
