"""Rich console output helpers."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

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


def print_summary(title: str, lines: list[str]) -> None:
    """Print a summary panel."""
    body = Text()
    for line in lines:
        body.append(line + "\n")
    console.print(Panel(body, title=title, border_style="green"))
