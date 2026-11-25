# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Diagrid CLI entry point."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import click

from diagrid.cli import telemetry_notice
from diagrid.cli.commands.chaos import chaos
from diagrid.cli.commands.deploy import deploy
from diagrid.cli.commands.init import init
from diagrid.cli.utils.process import set_verbose
from diagrid.core.config.constants import PROD_API_URL, STAGING_API_URL

# ``diagrid.core.analytics`` first shipped in diagrid-core 0.5.1, but this
# package pins ``diagrid-core>=0.1.0``. With an older core the CLI must still
# run, so fall back to no reporting.
try:
    from diagrid.core import analytics
except ImportError:  # pragma: no cover - older diagrid-core
    analytics = None  # type: ignore[assignment]


def _is_completion_request(prog_name: str | None, complete_var: str | None) -> bool:
    """Return True when click will treat this invocation as shell completion.

    Mirrors ``click.Command._main_shell_completion``: the variable is
    ``complete_var`` when given, else ``_{PROG_NAME}_COMPLETE`` upper-cased with
    ``-`` and ``.`` in the program name replaced by ``_``. Click then exits
    after printing the script or the completions, so such a call must neither
    report nor block (``eval "$(_DIAGRIDPY_COMPLETE=zsh_source diagridpy)"``
    runs on every new terminal).
    """
    try:
        if complete_var is None:
            name = prog_name or os.path.basename(sys.argv[0])
            name = name.replace("-", "_").replace(".", "_")
            complete_var = f"_{name}_COMPLETE".upper()
        return bool(os.environ.get(complete_var))
    except Exception:
        return False


def _report_cli_usage() -> None:
    """Report one anonymous usage event to Reo. Never raises.

    The first run that is about to send prints a one-time notice to stderr
    first; nothing else is ever printed.
    """
    if analytics is None:
        return
    try:
        if analytics.REO_SINK.configured() and analytics.reo_allowed():
            telemetry_notice.show_notice_once()
        analytics.report_usage_blocking("diagrid-cli", sinks=(analytics.REO_SINK,))
    except Exception:
        logging.getLogger(__name__).debug("cli usage report failed", exc_info=True)


class _UsageReportingGroup(click.Group):
    """A group that reports usage before it parses anything.

    ``--help`` and ``--version`` are eager options: click handles them while
    parsing, before the group callback runs. ``main`` is the one point every
    invocation passes through, including ``CliRunner.invoke``.
    """

    def main(
        self,
        args: Any = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        *rest: Any,
        **kwargs: Any,
    ) -> Any:
        if not _is_completion_request(prog_name, complete_var):
            _report_cli_usage()
        return super().main(args, prog_name, complete_var, *rest, **kwargs)


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
