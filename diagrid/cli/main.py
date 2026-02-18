"""Diagrid CLI entry point."""

from __future__ import annotations

import click

from diagrid.cli.commands.deploy import deploy
from diagrid.cli.commands.init import init
from diagrid.cli.utils.process import set_verbose


@click.group()
@click.version_option(version="0.1.0", prog_name="diagridpy")
@click.option("-v", "--verbose", is_flag=True, help="Show subprocess output")
def cli(verbose: bool) -> None:
    """Diagrid CLI for Catalyst agent development."""
    set_verbose(verbose)


cli.add_command(init)
cli.add_command(deploy)


if __name__ == "__main__":
    cli()
