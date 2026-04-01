# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid CLI entry point."""

from __future__ import annotations

import click

from diagrid.cli.commands.chaos import chaos
from diagrid.cli.commands.deploy import deploy
from diagrid.cli.commands.init import init
from diagrid.cli.utils.process import set_verbose
from diagrid.core.config.constants import PROD_API_URL, STAGING_API_URL


@click.group()
@click.version_option(version="0.1.0", prog_name="diagridpy")
@click.option("-v", "--verbose", is_flag=True, help="Show subprocess output")
@click.option(
    "--env",
    type=click.Choice(["prod", "staging"]),
    default=None,
    help="Target environment: prod or staging",
)
@click.option("--api", default=None, hidden=True, help="Override Diagrid API URL")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, env: str | None, api: str | None) -> None:
    """Diagrid CLI for Catalyst agent development."""
    set_verbose(verbose)
    ctx.ensure_object(dict)
    if api:
        ctx.obj["api_url"] = api
    elif env == "staging":
        ctx.obj["api_url"] = STAGING_API_URL
    elif env == "prod":
        ctx.obj["api_url"] = PROD_API_URL
    else:
        ctx.obj["api_url"] = None


cli.add_command(init)
cli.add_command(deploy)
cli.add_command(chaos)


if __name__ == "__main__":
    cli()
