"""Tests for user config store."""

from __future__ import annotations

import json
from pathlib import Path

from diagrid.core.config.user_config import FileUserConfigStore, UserConfig


def test_user_config_roundtrip(tmp_path: Path) -> None:
    """UserConfig can be stored and loaded."""
    store = FileUserConfigStore(tmp_path / "config.json")

    config = UserConfig(
        current_org_id="org-123",
        current_project_id="proj-456",
        current_product="catalyst",
    )
    store.set(config)
    loaded = store.get()

    assert loaded.current_org_id == "org-123"
    assert loaded.current_project_id == "proj-456"
    assert loaded.current_product == "catalyst"


def test_user_config_json_aliases(tmp_path: Path) -> None:
    """JSON output uses Go-compatible field names."""
    store = FileUserConfigStore(tmp_path / "config.json")
    config = UserConfig(current_org_id="org-1", current_product="catalyst")
    store.set(config)

    raw = json.loads((tmp_path / "config.json").read_text())
    assert "currentOrgID" in raw
    assert "currentProduct" in raw
    assert raw["currentOrgID"] == "org-1"


def test_user_config_missing_file(tmp_path: Path) -> None:
    """Missing file returns empty config."""
    store = FileUserConfigStore(tmp_path / "nonexistent.json")
    config = store.get()
    assert config.current_org_id == ""


def test_user_config_empty_file(tmp_path: Path) -> None:
    """Empty file returns empty config."""
    path = tmp_path / "config.json"
    path.write_text("")
    store = FileUserConfigStore(path)
    config = store.get()
    assert config.current_org_id == ""


def test_user_config_file_permissions(tmp_path: Path) -> None:
    """Config file must have 0600 permissions."""
    store = FileUserConfigStore(tmp_path / "config.json")
    store.set(UserConfig())
    mode = (tmp_path / "config.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_unset_clears_config(tmp_path: Path) -> None:
    store = FileUserConfigStore(tmp_path / "config.json")
    store.set(UserConfig(current_org_id="will-be-cleared"))
    store.unset()
    config = store.get()
    assert config.current_org_id == ""
