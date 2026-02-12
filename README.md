# Diagrid

Diagrid Catalyst AI packages for building agents with metadata introspection.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver

## Installation

```bash
# Install all dependencies including workspace members
make test-install

# Or manually with uv
uv sync --group dev --group test
```

## Pre-commit Hooks

This project uses pre-commit hooks that run automatically on `git push` to ensure code quality.

### Setup

```bash
# Install pre-push hooks
make hooks-install
```

### What Gets Checked

The hooks automatically run before each push:
- **File hygiene** - Trailing whitespace, end of files, YAML validation
- **Ruff format** - Auto-formats code
- **Flake8** - Lints code for style issues
- **MyPy** - Type checking
- **Unit tests** - Runs all unit tests (excludes integration tests)

### Manual Execution

```bash
# Run all hooks manually on all files
make hooks-run

# Uninstall hooks
make hooks-uninstall
```

## Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run only unit tests (no integration)
make test-unit
```

View coverage report: `open htmlcov/index.html`

## Development

```bash
# Format code
make format

# Lint code
make lint

# Type check
make typecheck
```

## Workspace Structure

This project uses a `uv` workspace:

- **diagrid** - Main package
- **diagrid-agent-core** - Workspace member (`diagrid/agent/core`) with LangGraph/Strands support

Dependencies for the workspace member are managed in `diagrid/agent/core/pyproject.toml`.

## Make Targets

| Command | Description |
|---------|-------------|
| `make test-install` | Install all dependencies |
| `make test` | Run tests |
| `make test-cov` | Run tests with coverage |
| `make hooks-install` | Install pre-push hooks |
| `make hooks-run` | Run hooks manually |
| `make format` | Format code with Ruff |
| `make lint` | Lint with Flake8 |
| `make typecheck` | Type check with MyPy |
