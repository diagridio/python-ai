# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for helm operations."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from diagrid.cli.infra.helm import (
    _adopt_existing_secret_for_helm,
    install_dapr_agents,
)
from diagrid.cli.utils.process import CommandError


# ---------------------------------------------------------------------------
# install_dapr_agents — existing tests (with adoption mock)
# ---------------------------------------------------------------------------


@patch("diagrid.cli.infra.helm._adopt_existing_secret_for_helm")
@patch("diagrid.cli.infra.helm.run")
def test_install_dapr_agents_default_version(
    mock_run: MagicMock, mock_adopt: MagicMock
) -> None:
    """install_dapr_agents omits --version when using default empty version."""
    install_dapr_agents("sk-test-key")

    assert mock_run.call_count == 1
    args = mock_run.call_args[0]

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


@patch("diagrid.cli.infra.helm._adopt_existing_secret_for_helm")
@patch("diagrid.cli.infra.helm.run")
def test_install_dapr_agents_explicit_version(
    mock_run: MagicMock, mock_adopt: MagicMock
) -> None:
    """install_dapr_agents includes --version when version is specified."""
    install_dapr_agents("sk-test-key", version="1.2.3")

    assert mock_run.call_count == 1
    args = mock_run.call_args[0]

    assert "--version" in args
    version_idx = args.index("--version")
    assert args[version_idx + 1] == "1.2.3"


@patch("diagrid.cli.infra.helm._adopt_existing_secret_for_helm")
@patch("diagrid.cli.infra.helm.run")
def test_install_dapr_agents_calls_adopt_before_helm(
    mock_run: MagicMock, mock_adopt: MagicMock
) -> None:
    """install_dapr_agents calls the adoption helper before the helm command."""
    install_dapr_agents("sk-test-key")

    mock_adopt.assert_called_once_with(
        "llm-secret",
        namespace="catalyst-agents",
        release_name="catalyst-agents",
    )
    # Adoption must happen before the helm run call.
    assert mock_adopt.call_count == 1
    assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# _adopt_existing_secret_for_helm
# ---------------------------------------------------------------------------

NS = "catalyst-agents"
RELEASE = "catalyst-agents"


@patch("diagrid.cli.infra.helm.console")
@patch("diagrid.cli.infra.helm.run")
@patch("diagrid.cli.infra.helm.run_capture")
def test_adopt_secret_not_found(
    mock_capture: MagicMock, mock_run: MagicMock, mock_console: MagicMock
) -> None:
    """No label/annotate when the secret doesn't exist."""
    mock_capture.side_effect = CommandError("kubectl", 1, "not found")

    _adopt_existing_secret_for_helm("llm-secret", namespace=NS, release_name=RELEASE)

    mock_run.assert_not_called()
    mock_console.info.assert_not_called()


@patch("diagrid.cli.infra.helm.console")
@patch("diagrid.cli.infra.helm.run")
@patch("diagrid.cli.infra.helm.run_capture")
def test_adopt_secret_already_helm_managed(
    mock_capture: MagicMock, mock_run: MagicMock, mock_console: MagicMock
) -> None:
    """No label/annotate when the secret is already Helm-managed."""
    mock_capture.return_value = "Helm"

    _adopt_existing_secret_for_helm("llm-secret", namespace=NS, release_name=RELEASE)

    mock_run.assert_not_called()
    mock_console.info.assert_not_called()


@patch("diagrid.cli.infra.helm.console")
@patch("diagrid.cli.infra.helm.run")
@patch("diagrid.cli.infra.helm.run_capture")
def test_adopt_secret_labels_and_annotates(
    mock_capture: MagicMock, mock_run: MagicMock, mock_console: MagicMock
) -> None:
    """Labels and annotates an existing secret without Helm ownership."""
    mock_capture.return_value = ""

    _adopt_existing_secret_for_helm("llm-secret", namespace=NS, release_name=RELEASE)

    assert mock_run.call_count == 2
    label_call, annotate_call = mock_run.call_args_list

    assert label_call == call(
        "kubectl",
        "label",
        "secret",
        "llm-secret",
        "-n",
        NS,
        "app.kubernetes.io/managed-by=Helm",
        f"app.kubernetes.io/instance={RELEASE}",
        "--overwrite",
    )
    assert annotate_call == call(
        "kubectl",
        "annotate",
        "secret",
        "llm-secret",
        "-n",
        NS,
        f"meta.helm.sh/release-name={RELEASE}",
        f"meta.helm.sh/release-namespace={NS}",
        "--overwrite",
    )
    mock_console.info.assert_called_once()


@patch("diagrid.cli.infra.helm.console")
@patch("diagrid.cli.infra.helm.run")
@patch("diagrid.cli.infra.helm.run_capture")
def test_adopt_secret_label_failure_is_non_fatal(
    mock_capture: MagicMock, mock_run: MagicMock, mock_console: MagicMock
) -> None:
    """A failure to label/annotate logs a warning but doesn't raise."""
    mock_capture.return_value = ""
    mock_run.side_effect = CommandError("kubectl", 1, "permission denied")

    _adopt_existing_secret_for_helm("llm-secret", namespace=NS, release_name=RELEASE)

    mock_console.warning.assert_called_once()
