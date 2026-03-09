# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for Catalyst project operations."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx

from diagrid.core.catalyst.models import Project
from diagrid.core.catalyst.projects import create_project, get_project, list_projects


def _mock_client(response_json: dict) -> MagicMock:  # type: ignore[type-arg]
    mock = MagicMock()
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = response_json
    mock.post.return_value = mock_resp
    mock.get.return_value = mock_resp
    return mock


def test_create_project() -> None:
    client = _mock_client(
        {
            "metadata": {"name": "test-proj"},
            "spec": {"defaultAgentInfrastructureEnabled": True},
        }
    )
    proj = create_project(client, "test-proj", agent_infrastructure_enabled=True)
    assert isinstance(proj, Project)
    assert proj.metadata is not None
    assert proj.metadata.name == "test-proj"
    client.post.assert_called_once()


def test_list_projects() -> None:
    client = _mock_client(
        {"items": [{"metadata": {"name": "p1"}}, {"metadata": {"name": "p2"}}]}
    )
    projects = list_projects(client)
    assert len(projects) == 2
    assert projects[0].metadata is not None
    assert projects[0].metadata.name == "p1"


def test_get_project() -> None:
    client = _mock_client({"metadata": {"name": "my-proj"}})
    proj = get_project(client, "my-proj")
    assert proj.metadata is not None
    assert proj.metadata.name == "my-proj"
    client.get.assert_called_once_with("/projects/my-proj")
