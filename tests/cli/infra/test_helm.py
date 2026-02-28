"""Tests for helm operations."""

from __future__ import annotations

from unittest.mock import patch

from diagrid.cli.infra.helm import install_dapr_agents


@patch("diagrid.cli.infra.helm.run")
def test_install_dapr_agents_default_version(mock_run: object) -> None:
    """install_dapr_agents omits --version when using default empty version."""
    install_dapr_agents("sk-test-key")

    assert mock_run.call_count == 1  # type: ignore[attr-defined]
    args = mock_run.call_args[0]  # type: ignore[attr-defined]

    assert args == (
        "helm",
        "upgrade",
        "--install",
        "catalyst-agents",
        "oci://ghcr.io/diagridio/charts/catalyst-agents",
        "--namespace",
        "catalyst-agents",
        "--create-namespace",
        "--set",
        "llm.apiKey=sk-test-key",
    )
    # Ensure --version is NOT in the args
    assert "--version" not in args


@patch("diagrid.cli.infra.helm.run")
def test_install_dapr_agents_explicit_version(mock_run: object) -> None:
    """install_dapr_agents includes --version when version is specified."""
    install_dapr_agents("sk-test-key", version="1.2.3")

    assert mock_run.call_count == 1  # type: ignore[attr-defined]
    args = mock_run.call_args[0]  # type: ignore[attr-defined]

    assert "--version" in args
    version_idx = args.index("--version")
    assert args[version_idx + 1] == "1.2.3"
