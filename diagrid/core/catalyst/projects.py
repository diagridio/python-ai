# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Catalyst project operations."""

from __future__ import annotations

from diagrid.core.catalyst.client import CatalystClient
from diagrid.core.catalyst.models import Project, ProjectMetadata


def create_project(
    client: CatalystClient,
    name: str,
    *,
    agent_infrastructure_enabled: bool = False,
) -> Project:
    """Create a new Catalyst project."""
    payload = {
        "apiVersion": "cra.diagrid.io/v1beta1",
        "kind": "Project",
        "metadata": {"name": name},
        "spec": {
            "displayName": name,
            "defaultAgentInfrastructureEnabled": agent_infrastructure_enabled,
        },
    }
    resp = client.post("/projects", json_data=payload)
    if resp.content:
        return Project.model_validate(resp.json())
    return Project(metadata=ProjectMetadata(name=name))


def list_projects(client: CatalystClient) -> list[Project]:
    """List all projects."""
    resp = client.get("/projects")
    data = resp.json()
    items = data.get("items", [])
    return [Project.model_validate(p) for p in items]


def get_project(client: CatalystClient, name: str) -> Project:
    """Get a project by name."""
    resp = client.get(f"/projects/{name}")
    return Project.model_validate(resp.json())
