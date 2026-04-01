# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""diagridpy chaos command — apply, delete, and inspect Chaos Mesh experiments."""

from __future__ import annotations

import os

import click

from diagrid.cli.infra.chaos import (
    DEFAULT_EXPERIMENTS,
    ChaosConfig,
    apply_chaos,
    chaos_mesh_installed,
    delete_chaos,
    get_chaos_status,
    render_experiments,
)
from diagrid.cli.utils import console
from diagrid.core.config.constants import DEFAULT_NAMESPACE


_VALID_EXPERIMENTS = ("pod", "network", "http", "stress")


@click.group()
def chaos() -> None:
    """Manage Chaos Mesh experiments for agent resilience testing."""


@chaos.command()
@click.option(
    "--intensity",
    type=click.Choice(["low", "medium", "high"]),
    default=None,
    help="Chaos intensity preset (or set CHAOS_INTENSITY env)",
)
@click.option(
    "--experiments",
    default=None,
    help="Comma-separated experiments: pod,network,http,stress",
)
@click.option("--target-app-id", default=None, help="Scope chaos to a single agent")
@click.option("--namespace", default=DEFAULT_NAMESPACE, help="Target namespace")
@click.option("--dry-run", is_flag=True, help="Print rendered YAML without applying")
def start(
    intensity: str | None,
    experiments: str | None,
    target_app_id: str | None,
    namespace: str,
    dry_run: bool,
) -> None:
    """Start chaos experiments against deployed agents."""
    resolved_intensity = intensity or os.environ.get("CHAOS_INTENSITY", "medium")

    if experiments:
        parsed = tuple(e.strip() for e in experiments.split(","))
        invalid = [e for e in parsed if e not in _VALID_EXPERIMENTS]
        if invalid:
            raise click.ClickException(
                f"Invalid experiments: {', '.join(invalid)}. "
                f"Valid: {', '.join(_VALID_EXPERIMENTS)}"
            )
        experiment_tuple = parsed
    else:
        experiment_tuple = DEFAULT_EXPERIMENTS

    config = ChaosConfig(
        intensity=resolved_intensity,
        experiments=experiment_tuple,
        namespace=namespace,
        frequency="",
        duration="",
        target_app_id=target_app_id,
    )

    if dry_run:
        for name, yaml_str in render_experiments(config):
            console.info(f"--- {name} ---")
            click.echo(yaml_str)
        return

    if not chaos_mesh_installed(namespace):
        raise click.ClickException(
            "Chaos Mesh CRDs not found. Re-run 'diagridpy init' to install "
            "the Chaos Mesh controller via the Helm chart."
        )

    console.info(f"Applying {resolved_intensity} chaos: {', '.join(experiment_tuple)}")
    if target_app_id:
        console.info(f"Scoped to app: {target_app_id}")

    applied = apply_chaos(config)
    console.success(f"Applied {len(applied)} experiment group(s): {', '.join(applied)}")


@chaos.command()
@click.option("--namespace", default=DEFAULT_NAMESPACE, help="Target namespace")
def stop(namespace: str) -> None:
    """Stop and delete all chaos experiments."""
    console.info(f"Deleting chaos resources in {namespace}...")
    delete_chaos(namespace)
    console.success("All chaos experiments removed")


@chaos.command()
@click.option("--namespace", default=DEFAULT_NAMESPACE, help="Target namespace")
def status(namespace: str) -> None:
    """Show active chaos experiments."""
    output = get_chaos_status(namespace)
    click.echo(output)
