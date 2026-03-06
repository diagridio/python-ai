# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Subprocess wrapper with streaming output."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# Environment passed to subprocesses — strip macOS MallocStackLogging noise.
_CLEAN_ENV = {k: v for k, v in os.environ.items() if k != "MallocStackLogging"}


class CommandNotFoundError(Exception):
    """Raised when a required command is not found in PATH."""


class CommandError(Exception):
    """Raised when a subprocess exits with non-zero status."""

    def __init__(self, cmd: str, returncode: int, stderr: str = "") -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        msg = f"Command '{cmd}' failed with exit code {returncode}"
        if stderr:
            msg += f"\n{stderr.strip()}"
        super().__init__(msg)


# Module-level verbose flag, toggled by the CLI group.
_verbose = False


def set_verbose(verbose: bool) -> None:
    """Set the global verbose flag."""
    global _verbose
    _verbose = verbose


def is_verbose() -> bool:
    """Return the current verbose flag."""
    return _verbose


def has_command(name: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(name) is not None


def run(
    *args: str,
    cwd: str | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command.

    By default, output is suppressed (captured and discarded) unless the
    global verbose flag is set or ``capture=True`` is passed.

    Args:
        args: Command and arguments.
        cwd: Working directory.
        capture: If True, capture and return output.
        check: If True, raise CommandError on non-zero exit.

    Returns:
        CompletedProcess with stdout/stderr if captured.
    """
    if not args:
        raise ValueError("No command provided")

    if not has_command(args[0]):
        raise CommandNotFoundError(
            f"'{args[0]}' not found. Please install it and try again."
        )

    kwargs: dict = {"cwd": cwd, "env": _CLEAN_ENV}  # type: ignore[type-arg]

    if capture or not _verbose:
        # Capture output — either to return it or to suppress it.
        kwargs["capture_output"] = True
        kwargs["text"] = True
    else:
        # Verbose: stream to terminal.
        kwargs["stdout"] = sys.stdout
        kwargs["stderr"] = sys.stderr

    result = subprocess.run(list(args), **kwargs)

    if check and result.returncode != 0:
        stderr = result.stderr if (capture or not _verbose) else ""
        raise CommandError(args[0], result.returncode, stderr)

    return result


def run_capture(*args: str, cwd: str | None = None) -> str:
    """Run a command and return its stdout. Raises on error."""
    result = run(*args, cwd=cwd, capture=True, check=True)
    return result.stdout.strip()
