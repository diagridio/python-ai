# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for kind cluster provisioning."""

from __future__ import annotations

from unittest.mock import call, patch

from diagrid.cli.infra.kind import _ensure_container, cluster_exists, kind_available
from diagrid.cli.utils.process import CommandError


@patch("diagrid.cli.infra.kind.has_command", return_value=True)
def test_kind_available(mock_has: object) -> None:
    assert kind_available() is True


@patch("diagrid.cli.infra.kind.has_command", return_value=False)
def test_kind_not_available(mock_has: object) -> None:
    assert kind_available() is False


@patch("diagrid.cli.infra.kind.kind_available", return_value=False)
def test_cluster_exists_no_kind(mock_avail: object) -> None:
    assert cluster_exists("test") is False


@patch("diagrid.cli.infra.kind.kind_available", return_value=True)
@patch(
    "diagrid.cli.infra.kind.run_capture", return_value="catalyst-agents\nother-cluster"
)
def test_cluster_exists_found(mock_run: object, mock_avail: object) -> None:
    assert cluster_exists("catalyst-agents") is True


@patch("diagrid.cli.infra.kind.kind_available", return_value=True)
@patch("diagrid.cli.infra.kind.run_capture", return_value="other-cluster")
def test_cluster_exists_not_found(mock_run: object, mock_avail: object) -> None:
    assert cluster_exists("catalyst-agents") is False


# --- _ensure_container tests ---


@patch("diagrid.cli.infra.kind.run")
@patch("diagrid.cli.infra.kind.run_capture", return_value="true")
def test_ensure_container_running_is_noop(
    mock_capture: object, mock_run: object
) -> None:
    """Running container requires no action."""
    _ensure_container("my-container", ["-d", "--name", "my-container", "img"])
    mock_run.assert_not_called()  # type: ignore[attr-defined]


@patch("diagrid.cli.infra.kind.run")
@patch("diagrid.cli.infra.kind.run_capture", return_value="false")
def test_ensure_container_stopped_starts_it(
    mock_capture: object, mock_run: object
) -> None:
    """Stopped container is restarted via docker start."""
    _ensure_container("my-container", ["-d", "--name", "my-container", "img"])
    mock_run.assert_called_once_with("docker", "start", "my-container")  # type: ignore[attr-defined]


@patch("diagrid.cli.infra.kind.run")
@patch(
    "diagrid.cli.infra.kind.run_capture",
    side_effect=CommandError("docker", 1, "No such object"),
)
def test_ensure_container_missing_creates_it(
    mock_capture: object, mock_run: object
) -> None:
    """Missing container is created via docker run."""
    run_args = ["-d", "--name", "my-container", "img"]
    _ensure_container("my-container", run_args)
    mock_run.assert_called_once_with("docker", "run", *run_args)  # type: ignore[attr-defined]
