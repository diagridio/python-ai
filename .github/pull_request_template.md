## What and why

<!-- What this changes and the problem it solves. Link the issue: "Fixes #123". -->

## How it was tested

<!-- Commands you ran and what they showed. -->

## Checklist

- [ ] `uv run ruff format` (no path arguments, so `examples/` is covered too)
- [ ] `uv run flake8 diagrid tests --ignore=E501,F401,W503,E203,E704`
- [ ] `uv run mypy --config-file mypy.ini`
- [ ] `uv run pytest tests -m "not integration"`
- [ ] New or changed behaviour has tests
- [ ] New files under `diagrid/` carry the license header
- [ ] Commits are signed off (`git commit -s`)
