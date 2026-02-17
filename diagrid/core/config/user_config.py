"""User configuration store (~/.diagrid/config.json).

Byte-compatible with the Go CLI's userconfig package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .constants import CONFIG_PATH


class OrgInfo(BaseModel):
    id: str = Field(default="", alias="id")
    name: str = Field(default="", alias="name")

    model_config = {"populate_by_name": True}


class UserConfig(BaseModel):
    """User configuration. Field aliases match Go CLI JSON names."""

    current_org_id: str = Field(default="", alias="currentOrgID")
    current_org_name: str = Field(default="", alias="currentOrgName")
    current_project_id: str = Field(default="", alias="currentProjectID")
    current_product: str = Field(default="", alias="currentProduct")
    current_plan: str = Field(default="", alias="currentPlan")
    product_last_org: dict[str, OrgInfo] = Field(
        default_factory=dict, alias="productLastOrg"
    )

    model_config = {"populate_by_name": True}


class UserConfigStore(Protocol):
    def get(self) -> UserConfig: ...
    def set(self, config: UserConfig) -> None: ...
    def unset(self) -> None: ...


class FileUserConfigStore:
    """File-based user config store at ~/.diagrid/config.json."""

    def __init__(self, file_path: Path | None = None) -> None:
        self.file_path = file_path or CONFIG_PATH

    def get(self) -> UserConfig:
        if not self.file_path.exists():
            return UserConfig()
        data = self.file_path.read_text()
        if not data.strip():
            return UserConfig()
        return UserConfig.model_validate(json.loads(data))

    def set(self, config: UserConfig) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(by_alias=True, exclude_none=True)
        self.file_path.write_text(json.dumps(data))
        self.file_path.chmod(0o600)

    def unset(self) -> None:
        self.set(UserConfig())
