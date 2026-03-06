# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Token models for OAuth2 flows."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DeviceCodeResponse(BaseModel):
    """Response from POST /oauth/device/code."""

    device_code: str = Field(alias="device_code")
    user_code: str = Field(alias="user_code")
    verification_uri: str = Field(alias="verification_uri")
    verification_uri_complete: str = Field(alias="verification_uri_complete")
    expires_in: int = Field(alias="expires_in")
    interval: int = Field(alias="interval")

    model_config = {"populate_by_name": True}


class TokenResponse(BaseModel):
    """Response from POST /oauth/token."""

    access_token: str = Field(alias="access_token")
    token_type: str = Field(default="", alias="token_type")
    refresh_token: str = Field(default="", alias="refresh_token")
    expires_in: int = Field(default=0, alias="expires_in")
    id_token: str = Field(default="", alias="id_token")

    model_config = {"populate_by_name": True}


class AuthContext(BaseModel):
    """Authentication context used across the CLI."""

    api_url: str
    org_id: str
    project_id: str = ""
    access_token: str = ""
    api_key: str = ""

    @property
    def bearer_token(self) -> str:
        return self.access_token

    @property
    def auth_header(self) -> dict[str, str]:
        if self.api_key:
            return {"x-diagrid-api-key": self.api_key}
        return {
            "Authorization": f"Bearer {self.access_token}",
            "x-diagrid-orgid": self.org_id,
        }
