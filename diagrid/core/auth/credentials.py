"""Credential store for Diagrid CLI (~/.diagrid/creds).

The credential file is base64-encoded JSON, byte-compatible with the Go CLI.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from diagrid.core.auth.token import TokenResponse
from diagrid.core.config.constants import CREDS_PATH
from diagrid.core.config.envs import EnvConfig


class Credential(BaseModel):
    """Stored credential. Field aliases match Go CLI JSON names exactly."""

    subject: str = Field(default="")
    env: EnvConfig | None = Field(default=None)
    token_response: TokenResponse | None = Field(default=None, alias="token_response")
    client_id: str = Field(default="", alias="client_id")
    client_secret: str = Field(default="", alias="client_secret")
    invalid: bool = Field(default=False)
    default_org: str = Field(default="", alias="default_org")
    orgs: dict[str, list[str]] = Field(default_factory=dict)
    expires_at: datetime | None = Field(default=None, alias="expires_at")
    timestamp: datetime | None = Field(default=None)
    api_key: str = Field(default="", alias="apiKey")

    model_config = {"populate_by_name": True}

    @property
    def bearer_token(self) -> str:
        if self.token_response:
            return self.token_response.access_token
        return ""


class CredentialStore(Protocol):
    def get(self) -> Credential: ...
    def set(self, cred: Credential) -> None: ...
    def unset(self) -> None: ...


class FileCredentialStore:
    """File-based credential store at ~/.diagrid/creds.

    Stores credentials as base64-encoded JSON, matching the Go CLI format.
    """

    def __init__(self, file_path: Path | None = None) -> None:
        self.file_path = file_path or CREDS_PATH

    def get(self) -> Credential:
        if not self.file_path.exists():
            return Credential()
        raw = self.file_path.read_text()
        if not raw.strip():
            return Credential()
        decoded = base64.b64decode(raw)
        return Credential.model_validate(json.loads(decoded))

    def set(self, cred: Credential) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = cred.model_dump(by_alias=True, mode="json")
        encoded = base64.b64encode(json.dumps(data).encode()).decode()
        self.file_path.write_text(encoded)
        self.file_path.chmod(0o600)

    def unset(self) -> None:
        self.set(Credential())
