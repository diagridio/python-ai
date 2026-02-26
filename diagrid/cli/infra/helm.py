"""Helm chart operations."""

from __future__ import annotations

from diagrid.cli.utils.process import run
from diagrid.core.config.constants import (
    DEFAULT_HELM_CHART_VERSION,
    DEFAULT_HELM_OCI_CHART,
    DEFAULT_NAMESPACE,
)


def install_dapr_agents(
    llm_api_key: str,
    *,
    google_api_key: str = "",
    chart: str = DEFAULT_HELM_OCI_CHART,
    version: str = DEFAULT_HELM_CHART_VERSION,
    namespace: str = DEFAULT_NAMESPACE,
    release_name: str = "catalyst-agents",
) -> None:
    """Install the catalyst-agents Helm chart from OCI registry."""
    args = [
        "helm",
        "upgrade",
        "--install",
        release_name,
        chart,
        "--version",
        version,
        "--namespace",
        namespace,
        "--create-namespace",
        "--set",
        f"llm.apiKey={llm_api_key}",
    ]
    if google_api_key:
        args.extend(["--set", f"llm.googleApiKey={google_api_key}"])
    run(*args)
