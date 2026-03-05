# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Rich console output helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from diagrid.cli.utils.process import is_verbose

console = Console()
error_console = Console(stderr=True)


def info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue]>[/bold blue] {message}")


def success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow]![/bold yellow] {message}")


def error(message: str) -> None:
    """Print an error message to stderr."""
    error_console.print(f"[bold red]✗[/bold red] {message}")


def step(number: int, total: int, message: str) -> None:
    """Print a numbered step."""
    console.print(f"[bold cyan][{number}/{total}][/bold cyan] {message}")


@contextmanager
def spinner(number: int, total: int, message: str) -> Generator[None, None, None]:
    """Show a numbered step with a spinner while the block executes.

    In verbose mode, prints the step text and yields without a spinner
    so that subprocess output can stream to the terminal uninterrupted.
    """
    if is_verbose():
        step(number, total, message)
        yield
    else:
        with console.status(
            f"[bold cyan][{number}/{total}][/bold cyan] {message}",
            spinner="dots",
        ):
            yield


def print_summary(title: str, lines: list[str]) -> None:
    """Print a summary panel."""
    body = Text()
    for line in lines:
        body.append(line + "\n")
    console.print(Panel(body, title=title, border_style="green"))
