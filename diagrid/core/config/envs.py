"""Fetch environment configuration from the Diagrid API."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from .constants import DEFAULT_API_URL


class EnvConfig(BaseModel):
    """Environment configuration from {apiURL}/cli.envs.json.

    Field aliases match the Go CLI JSON field names exactly.
    """

    api_url: str = Field(alias="apiUrl", default="")
    auth_audience: str = Field(alias="authAudience", default="")
    auth_client_id: str = Field(alias="authClientId", default="")
    # Preferred field — present in current API responses
    issuer_url: str = Field(alias="issuerUrl", default="")
    # Legacy / deprecated
    auth_domain: str = Field(alias="authDomain", default="")

    model_config = {"populate_by_name": True}

    @property
    def device_authorization_endpoint(self) -> str:
        """Derive device code URL from issuerUrl (preferred) or legacy authDomain."""
        if self.issuer_url:
            return f"{self.issuer_url.rstrip('/')}/oauth/device/code"
        return f"https://{self.auth_domain}/oauth/device/code"

    @property
    def token_endpoint(self) -> str:
        """Derive token URL from issuerUrl (preferred) or legacy authDomain."""
        if self.issuer_url:
            return f"{self.issuer_url.rstrip('/')}/oauth/token"
        return f"https://{self.auth_domain}/oauth/token"


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
