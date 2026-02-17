"""Tests for kind cluster provisioning."""

from __future__ import annotations

from unittest.mock import patch

from diagrid.cli.infra.kind import cluster_exists, kind_available


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
@patch("diagrid.cli.infra.kind.run_capture", return_value="dapr-agents\nother-cluster")
def test_cluster_exists_found(mock_run: object, mock_avail: object) -> None:
    assert cluster_exists("dapr-agents") is True


@patch("diagrid.cli.infra.kind.kind_available", return_value=True)
@patch("diagrid.cli.infra.kind.run_capture", return_value="other-cluster")
def test_cluster_exists_not_found(mock_run: object, mock_avail: object) -> None:
    assert cluster_exists("dapr-agents") is False
