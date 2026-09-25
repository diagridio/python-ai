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

# ``diagrid-core`` ships the reporter. This distribution only pins
# ``diagrid-core>=0.1.0`` (directly or through ``diagrid-cli``), so an
# application holding an older ``diagrid-core`` must keep working: fall back
# to a no-op.
try:
    from diagrid.core.analytics import report_usage
except ImportError:  # pragma: no cover - older diagrid-core

    def report_usage(package: str) -> None:
        return None


@click.group()
# Read the version from the installed ``diagrid-cli`` distribution rather than
# hardcoding it. The release workflow bumps ``diagrid/cli/pyproject.toml`` but
# never touched a literal here, so a hardcoded string silently goes stale.
@click.version_option(package_name="diagrid-cli", prog_name="diagridpy")
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
    # One anonymous usage event per process, never blocking. See
    # ``diagrid.core.analytics`` and the README "Usage analytics" section.
    report_usage("diagrid-cli")
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
