"""Kubectl operations."""

from __future__ import annotations

import subprocess

from diagrid.cli.utils.process import CommandError, is_verbose, run, run_capture


def get_current_context() -> str:
    """Get the current kubectl context."""
    return run_capture("kubectl", "config", "current-context")


def apply_manifest(path: str, *, namespace: str | None = None) -> None:
    """Apply a Kubernetes manifest file."""
    args = ["kubectl", "apply", "-f", path]
    if namespace:
        args.extend(["-n", namespace])
    run(*args)


def apply_stdin(manifest: str, *, namespace: str | None = None) -> None:
    """Apply a Kubernetes manifest from a string via stdin."""
    args = ["kubectl", "apply", "-f", "-"]
    if namespace:
        args.extend(["-n", namespace])

    if is_verbose():
        import sys

        proc = subprocess.run(
            args, input=manifest, text=True, stdout=sys.stdout, stderr=sys.stderr
        )
    else:
        proc = subprocess.run(args, input=manifest, text=True, capture_output=True)

    if proc.returncode != 0:
        stderr = proc.stderr if not is_verbose() else ""
        raise CommandError("kubectl", proc.returncode, stderr)
