"""Catalyst AppID operations."""

from __future__ import annotations

from diagrid.core.catalyst.client import CatalystClient
from diagrid.core.catalyst.models import AppID, AppIDMetadata


def create_appid(
    client: CatalystClient,
    project: str,
    name: str,
    *,
    app_port: int | None = None,
) -> AppID:
    """Create a new AppID in a project."""
    payload: dict = {  # type: ignore[type-arg]
        "apiVersion": "cra.diagrid.io/v1beta1",
        "kind": "AppIdentity",
        "metadata": {"name": name},
    }
    if app_port is not None:
        payload["spec"] = {"appPort": app_port}
    resp = client.post(f"/projects/{project}/appids", json_data=payload)
    if resp.content:
        return AppID.model_validate(resp.json())
    return AppID(metadata=AppIDMetadata(name=name))


def get_appid(client: CatalystClient, project: str, name: str) -> AppID:
    """Get an AppID by name, including its status (apiToken etc.)."""
    resp = client.get(f"/projects/{project}/appids/{name}")
    return AppID.model_validate(resp.json())


def list_appids(client: CatalystClient, project: str) -> list[AppID]:
    """List all AppIDs in a project."""
    resp = client.get(f"/projects/{project}/appids")
    data = resp.json()
    items = data.get("items", [])
    return [AppID.model_validate(a) for a in items]
