"""Tests for helm operations."""

from __future__ import annotations

from unittest.mock import patch

from diagrid.cli.infra.helm import install_dapr_agents


@patch("diagrid.cli.infra.helm.run")
def test_install_dapr_agents(mock_run: object) -> None:
    """install_dapr_agents calls helm repo add, update, upgrade --install."""
    install_dapr_agents("sk-test-key")

    assert mock_run.call_count == 3  # type: ignore[attr-defined]
    calls = mock_run.call_args_list  # type: ignore[attr-defined]

    # First call: helm repo add
    assert calls[0][0][0] == "helm"
    assert calls[0][0][1] == "repo"
    assert calls[0][0][2] == "add"
    assert calls[0][0][3] == "dapr-agents"
    assert calls[0][0][4] == "https://caspergn.github.io/dapr-agents-dev/"

    # Second call: helm repo update
    assert calls[1][0][0] == "helm"
    assert calls[1][0][1] == "repo"
    assert calls[1][0][2] == "update"

    # Third call: helm upgrade --install
    assert calls[2][0][0] == "helm"
    assert calls[2][0][1] == "upgrade"
    assert calls[2][0][2] == "--install"
    assert calls[2][0][3] == "dapr-agents"
    assert calls[2][0][4] == "dapr-agents/dapr-agents"
