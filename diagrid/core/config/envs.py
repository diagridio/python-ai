"""Fetch environment configuration from the Diagrid API."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from .constants import DEFAULT_API_URL


class EnvConfig(BaseModel):
    """Environment configuration from {apiURL}/cli.envs.json.

    Field aliases match the Go CLI JSON field names exactly.
    """

    api_url: str = Field(alias="apiUrl")
    auth_audience: str = Field(alias="authAudience")
    auth_client_id: str = Field(alias="authClientId")
    auth_domain: str = Field(alias="authDomain")

    model_config = {"populate_by_name": True}


async def get_env_config(
    client: httpx.AsyncClient,
    api_url: str = DEFAULT_API_URL,
) -> EnvConfig:
    """Fetch environment config from {apiURL}/cli.envs.json."""
    url = f"{api_url.rstrip('/')}/cli.envs.json"
    resp = await client.get(url)
    resp.raise_for_status()
    env = EnvConfig.model_validate(resp.json())
    # Force-set apiURL so it matches what the user provided
    env.api_url = api_url.rstrip("/")
    return env


def get_env_config_sync(
    api_url: str = DEFAULT_API_URL,
) -> EnvConfig:
    """Fetch environment config synchronously."""
    url = f"{api_url.rstrip('/')}/cli.envs.json"
    with httpx.Client() as client:
        resp = client.get(url)
        resp.raise_for_status()
        env = EnvConfig.model_validate(resp.json())
        env.api_url = api_url.rstrip("/")
        return env
