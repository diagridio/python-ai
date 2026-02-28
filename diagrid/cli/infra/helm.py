"""Helm chart operations."""

from __future__ import annotations

from diagrid.cli.utils import console
from diagrid.cli.utils.process import CommandError, run, run_capture
from diagrid.core.config.constants import (
    DEFAULT_HELM_CHART_VERSION,
    DEFAULT_HELM_OCI_CHART,
    DEFAULT_NAMESPACE,
)


def _adopt_existing_secret_for_helm(
    secret_name: str,
    *,
    namespace: str,
    release_name: str,
) -> None:
    """Label/annotate a pre-existing secret so Helm can adopt it.

    If the secret doesn't exist or is already Helm-managed, this is a no-op.
    Failures are logged as warnings (non-fatal) to match the graceful pattern
    used by ``_patch_llm_secret`` in the deploy command.
    """
    try:
        managed_by = run_capture(
            "kubectl",
            "get",
            "secret",
            secret_name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.metadata.labels.app\\.kubernetes\\.io/managed-by}",
        )
    except CommandError:
        # Secret doesn't exist yet — nothing to adopt.
        return

    if managed_by == "Helm":
        return

    try:
        run(
            "kubectl",
            "label",
            "secret",
            secret_name,
            "-n",
            namespace,
            "app.kubernetes.io/managed-by=Helm",
            f"app.kubernetes.io/instance={release_name}",
            "--overwrite",
        )
        run(
            "kubectl",
            "annotate",
            "secret",
            secret_name,
            "-n",
            namespace,
            f"meta.helm.sh/release-name={release_name}",
            f"meta.helm.sh/release-namespace={namespace}",
            "--overwrite",
        )
        console.info(
            f"Adopted existing secret '{secret_name}' for Helm release '{release_name}'."
        )
    except CommandError as exc:
        console.warning(f"Could not adopt secret '{secret_name}' for Helm: {exc}")


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
    _adopt_existing_secret_for_helm(
        "llm-secret",
        namespace=namespace,
        release_name=release_name,
    )
    args = [
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
    ]
    if version:
        args.extend(["--version", version])
    if google_api_key:
        args.extend(["--set", f"llm.googleApiKey={google_api_key}"])
    run(*args)
