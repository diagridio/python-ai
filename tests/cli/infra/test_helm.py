"""Tests for helm operations."""

from __future__ import annotations

from unittest.mock import patch

from diagrid.cli.infra.helm import install_dapr_agents


@patch("diagrid.cli.infra.helm.run")
def test_install_dapr_agents(mock_run: object) -> None:
    """install_dapr_agents calls helm upgrade --install with OCI chart."""
    install_dapr_agents("sk-test-key")

    assert mock_run.call_count == 1  # type: ignore[attr-defined]
    args = mock_run.call_args[0]  # type: ignore[attr-defined]

    assert args == (
        "helm",
        "upgrade",
        "--install",
        "catalyst-agents",
        "oci://ghcr.io/diagridio/charts/catalyst-agents",
        "--version",
        "0.1.0",
        "--namespace",
        "catalyst-agents",
        "--create-namespace",
        "--set",
        "llm.apiKey=sk-test-key",
    )
