"""Helm chart operations."""

from __future__ import annotations

from diagrid.cli.utils.process import run
from diagrid.core.config.constants import (
    DEFAULT_HELM_CHART,
    DEFAULT_HELM_REPO_NAME,
    DEFAULT_HELM_REPO_URL,
    DEFAULT_NAMESPACE,
)


def add_repo(
    name: str = DEFAULT_HELM_REPO_NAME,
    url: str = DEFAULT_HELM_REPO_URL,
) -> None:
    """Add a Helm repo."""
    run("helm", "repo", "add", name, url)


def update_repos() -> None:
    """Update Helm repos."""
    run("helm", "repo", "update")


def install_dapr_agents(
    llm_api_key: str,
    *,
    chart: str = DEFAULT_HELM_CHART,
    namespace: str = DEFAULT_NAMESPACE,
    release_name: str = "dapr-agents",
) -> None:
    """Install the dapr-agents Helm chart from the remote repo."""
    add_repo()
    update_repos()
    run(
        "helm",
        "upgrade",
        "--install",
        release_name,
        chart,
        "--namespace",
        namespace,
        "--create-namespace",
        "--set",
        f"llm.apiKey={llm_api_key}",
    )
