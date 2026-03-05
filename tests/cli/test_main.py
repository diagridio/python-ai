# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for CLI main entry point — env/api flag handling."""

from __future__ import annotations
import click
from click.testing import CliRunner

from diagrid.cli.main import cli
from diagrid.core.config.constants import PROD_API_URL, STAGING_API_URL


def _invoke_noop(*args: str) -> "click.testing.Result":
    """Invoke the CLI with a no-op subcommand to inspect ctx.obj."""

    @cli.command(name="_noop", hidden=True)
    @click.pass_context
    def _noop(ctx: click.Context) -> None:
        # Print api_url so the test can inspect it
        api_url = (ctx.obj or {}).get("api_url", "NOT_SET")
        click.echo(f"api_url={api_url}")

    runner = CliRunner()
    result = runner.invoke(cli, [*args, "_noop"])
    # Remove the ephemeral command after each test to keep state clean
    cli.commands.pop("_noop", None)
    return result


def test_cli_env_staging_sets_api_url() -> None:
    """--env staging sets api_url to STAGING_API_URL in ctx.obj."""
    result = _invoke_noop("--env", "staging")
    assert result.exit_code == 0, result.output
    assert f"api_url={STAGING_API_URL}" in result.output


def test_cli_env_prod_sets_api_url() -> None:
    """--env prod sets api_url to PROD_API_URL in ctx.obj."""
    result = _invoke_noop("--env", "prod")
    assert result.exit_code == 0, result.output
    assert f"api_url={PROD_API_URL}" in result.output


def test_cli_api_override_sets_api_url() -> None:
    """--api sets a custom api_url in ctx.obj."""
    result = _invoke_noop("--api", "https://custom.example.com")
    assert result.exit_code == 0, result.output
    assert "api_url=https://custom.example.com" in result.output


def test_cli_no_env_sets_none() -> None:
    """No env/api flags → api_url is None in ctx.obj."""
    result = _invoke_noop()
    assert result.exit_code == 0, result.output
    assert "api_url=None" in result.output
