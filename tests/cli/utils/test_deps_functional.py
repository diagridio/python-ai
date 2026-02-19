"""Functional integration tests for diagrid.cli.utils.deps.

These tests perform real network downloads and require internet access.
Run with: uv run pytest tests/cli/utils/test_deps_functional.py -m integration -v
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest

from diagrid.cli.utils import deps


@pytest.mark.integration
def test_download_kind_real(tmp_path: Path) -> None:
    deps._download_kind(tmp_path, deps._arch())
    ext = ".exe" if sys.platform == "win32" else ""
    binary = tmp_path / f"kind{ext}"
    assert binary.exists()
    assert binary.stat().st_size > 0
    if sys.platform != "win32":
        assert binary.stat().st_mode & stat.S_IEXEC
    result = subprocess.run([str(binary), "--version"], capture_output=True)
    assert result.returncode == 0


@pytest.mark.integration
def test_download_kubectl_real(tmp_path: Path) -> None:
    deps._download_kubectl(tmp_path, deps._arch())
    ext = ".exe" if sys.platform == "win32" else ""
    binary = tmp_path / f"kubectl{ext}"
    assert binary.exists()
    assert binary.stat().st_size > 0
    if sys.platform != "win32":
        assert binary.stat().st_mode & stat.S_IEXEC
    result = subprocess.run([str(binary), "version", "--client"], capture_output=True)
    assert result.returncode == 0


@pytest.mark.integration
def test_download_helm_real(tmp_path: Path) -> None:
    deps._download_helm(tmp_path, deps._arch())
    ext = ".exe" if sys.platform == "win32" else ""
    binary = tmp_path / f"helm{ext}"
    assert binary.exists()
    assert binary.stat().st_size > 0
    if sys.platform != "win32":
        assert binary.stat().st_mode & stat.S_IEXEC
    result = subprocess.run([str(binary), "version"], capture_output=True)
    assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="Docker daemon test is Linux-only")
def test_docker_daemon_running_on_linux() -> None:
    assert deps._docker_daemon_running() is True
