# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid CLI entry point."""

from __future__ import annotations

import logging
from typing import Any

import click

from diagrid.cli.commands.chaos import chaos
from diagrid.cli.commands.deploy import deploy
from diagrid.cli.commands.init import init
from diagrid.cli.utils.process import set_verbose
from diagrid.core import analytics
from diagrid.core.config.constants import PROD_API_URL, STAGING_API_URL


def _report_cli_usage() -> None:
    """Report one anonymous usage event to Reo. Never raises, never prints."""
    try:
        analytics.report_usage_blocking("diagrid-cli", sinks=(analytics.REO_SINK,))
    except Exception:
        logging.getLogger(__name__).debug("cli usage report failed", exc_info=True)


class _UsageReportingGroup(click.Group):
    """A group that reports usage before it parses anything.

    ``--help`` and ``--version`` are eager options: click handles them while
    parsing, before the group callback runs. ``main`` is the one point every
    invocation passes through, including ``CliRunner.invoke``.
    """

    def main(self, *args: Any, **kwargs: Any) -> Any:
        _report_cli_usage()
        return super().main(*args, **kwargs)


@click.group(cls=_UsageReportingGroup)
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
