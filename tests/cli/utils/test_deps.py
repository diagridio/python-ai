# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for diagrid.cli.utils.deps — preflight dependency checks."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tarfile
import zipfile
from io import BytesIO

import click
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from diagrid.cli.utils import deps
from diagrid.cli.utils.deps import (
    _arch,
    _docker_daemon_running,
    _ensure_in_path,
    _install_dir,
    _missing_binaries,
    preflight_check,
)


# ---------------------------------------------------------------------------
# Platform / arch helpers
# ---------------------------------------------------------------------------


def test_arch_x86_64() -> None:
    with patch("platform.machine", return_value="x86_64"):
        assert _arch() == "amd64"


def test_arch_aarch64() -> None:
    with patch("platform.machine", return_value="aarch64"):
        assert _arch() == "arm64"


def test_arch_arm64() -> None:
    with patch("platform.machine", return_value="arm64"):
        assert _arch() == "arm64"


# ---------------------------------------------------------------------------
# _install_dir
# ---------------------------------------------------------------------------


def test_install_dir_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert _install_dir() == Path.home() / ".local" / "bin"


def test_install_dir_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _install_dir() == Path.home() / ".local" / "bin"


def test_install_dir_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\test\\AppData\\Local")
    result = _install_dir()
    assert (
        result
        == Path("C:\\Users\\test\\AppData\\Local") / "Programs" / "diagrid" / "bin"
    )


# ---------------------------------------------------------------------------
# _ensure_in_path
# ---------------------------------------------------------------------------


def test_ensure_in_path_adds_new_dir(tmp_path: Path) -> None:
    original = os.environ.get("PATH", "")
    try:
        _ensure_in_path(tmp_path)
        assert str(tmp_path) in os.environ["PATH"]
    finally:
        os.environ["PATH"] = original


def test_ensure_in_path_skips_if_already_present(tmp_path: Path) -> None:
    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(tmp_path) + os.pathsep + original_path
    try:
        _ensure_in_path(tmp_path)
        # Should not be duplicated
        assert os.environ["PATH"].count(str(tmp_path)) == 1
    finally:
        os.environ["PATH"] = original_path


# ---------------------------------------------------------------------------
# _missing_binaries
# ---------------------------------------------------------------------------


def test_missing_binaries_none_missing() -> None:
    with patch("diagrid.cli.utils.deps.shutil.which", return_value="/usr/bin/fake"):
        assert _missing_binaries() == []


def test_missing_binaries_all_missing() -> None:
    with patch("diagrid.cli.utils.deps.shutil.which", return_value=None):
        assert _missing_binaries() == ["docker", "kind", "kubectl", "helm", "piko"]


def test_missing_binaries_subset_missing() -> None:
    def _which(name: str) -> str | None:
        return None if name in ("kind", "helm") else "/usr/bin/" + name

    with patch("diagrid.cli.utils.deps.shutil.which", side_effect=_which):
        assert _missing_binaries() == ["kind", "helm"]


# ---------------------------------------------------------------------------
# _docker_daemon_running
# ---------------------------------------------------------------------------


def test_docker_daemon_running_true() -> None:
    mock_result = MagicMock(returncode=0)
    with patch("subprocess.run", return_value=mock_result):
        assert _docker_daemon_running() is True


def test_docker_daemon_running_false_nonzero() -> None:
    mock_result = MagicMock(returncode=1)
    with patch("subprocess.run", return_value=mock_result):
        assert _docker_daemon_running() is False


def test_docker_daemon_not_installed() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert _docker_daemon_running() is False


def test_docker_daemon_timeout() -> None:
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 10)):
        assert _docker_daemon_running() is False


# ---------------------------------------------------------------------------
# preflight_check — happy path
# ---------------------------------------------------------------------------


def test_preflight_check_all_present_daemon_running() -> None:
    """No output, no installs when everything is already available."""
    with (
        patch("diagrid.cli.utils.deps.shutil.which", return_value="/usr/bin/fake"),
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=True),
    ):
        # Should complete without raising
        preflight_check()


# ---------------------------------------------------------------------------
# preflight_check — user declines
# ---------------------------------------------------------------------------


def test_preflight_check_user_declines_exits() -> None:
    def _which(name: str) -> str | None:
        return None if name == "kind" else "/usr/bin/" + name

    with (
        patch("diagrid.cli.utils.deps.shutil.which", side_effect=_which),
        patch("diagrid.cli.utils.deps.click.confirm", return_value=False),
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=True),
    ):
        with pytest.raises(SystemExit) as exc_info:
            preflight_check()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# preflight_check — user accepts, installs succeed
# ---------------------------------------------------------------------------


def test_preflight_check_user_accepts_and_installs() -> None:
    missing = {"kind", "kubectl"}
    call_count: dict[str, int] = {}

    def _which(name: str) -> str | None:
        # After first check cycle, simulate all found
        call_count[name] = call_count.get(name, 0) + 1
        if call_count[name] <= len(missing) + 1 and name in missing:
            return None
        return "/usr/bin/" + name

    mock_install = MagicMock()

    with (
        patch("diagrid.cli.utils.deps.shutil.which", side_effect=_which),
        patch("diagrid.cli.utils.deps.click.confirm", return_value=True),
        patch("diagrid.cli.utils.deps._install_binary", mock_install),
        patch(
            "diagrid.cli.utils.deps._missing_binaries",
            side_effect=[
                ["kind", "kubectl"],  # first call — what to install
                [],  # second call — post-install verification
            ],
        ),
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=True),
    ):
        preflight_check()

    assert mock_install.call_count == 2
    mock_install.assert_any_call("kind")
    mock_install.assert_any_call("kubectl")


def test_preflight_check_install_still_fails_raises() -> None:
    """If a binary is still missing after install, a ClickException is raised."""
    with (
        patch("diagrid.cli.utils.deps.click.confirm", return_value=True),
        patch("diagrid.cli.utils.deps._install_binary", MagicMock()),
        patch(
            "diagrid.cli.utils.deps._missing_binaries",
            side_effect=[
                ["helm"],  # before install
                ["helm"],  # after install — still missing
            ],
        ),
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=True),
    ):
        with pytest.raises(click.ClickException, match="Failed to install"):
            preflight_check()


# ---------------------------------------------------------------------------
# preflight_check — docker daemon not running
# ---------------------------------------------------------------------------


def test_preflight_check_starts_docker_when_not_running() -> None:
    mock_start = MagicMock()

    with (
        patch("diagrid.cli.utils.deps.shutil.which", return_value="/usr/bin/fake"),
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=False),
        patch("diagrid.cli.utils.deps._start_or_wait_for_docker", mock_start),
    ):
        preflight_check()

    mock_start.assert_called_once()


# ---------------------------------------------------------------------------
# _install_binary dispatch
# ---------------------------------------------------------------------------


def test_install_binary_docker_delegates() -> None:
    mock_docker = MagicMock()
    with patch("diagrid.cli.utils.deps._install_docker", mock_docker):
        deps._install_binary("docker")
    mock_docker.assert_called_once()


def test_install_binary_brew_on_mac() -> None:
    mock_brew = MagicMock()
    with (
        patch.object(sys, "platform", "darwin"),
        patch(
            "diagrid.cli.utils.deps.shutil.which", return_value="/usr/local/bin/brew"
        ),
        patch("diagrid.cli.utils.deps._install_via_brew", mock_brew),
    ):
        deps._install_binary("kind")
    mock_brew.assert_called_once_with("kind")


def test_install_binary_download_on_linux() -> None:
    mock_dl = MagicMock()
    with (
        patch.object(sys, "platform", "linux"),
        patch("diagrid.cli.utils.deps._install_via_download", mock_dl),
    ):
        deps._install_binary("kind")
    mock_dl.assert_called_once_with("kind")


def test_install_binary_mac_no_brew_falls_back_to_download() -> None:
    mock_dl = MagicMock()
    with (
        patch.object(sys, "platform", "darwin"),
        patch("diagrid.cli.utils.deps.shutil.which", return_value=None),
        patch("diagrid.cli.utils.deps._install_via_download", mock_dl),
    ):
        deps._install_binary("helm")
    mock_dl.assert_called_once_with("helm")


# ---------------------------------------------------------------------------
# _install_via_brew
# ---------------------------------------------------------------------------


def test_install_via_brew_success() -> None:
    mock_result = MagicMock(returncode=0)
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        deps._install_via_brew("kind")
    mock_run.assert_called_once_with(["brew", "install", "kind"], capture_output=True)


def test_install_via_brew_kubectl_uses_formula_name() -> None:
    mock_result = MagicMock(returncode=0)
    with patch("subprocess.run", return_value=mock_result) as mock_run:
        deps._install_via_brew("kubectl")
    mock_run.assert_called_once_with(
        ["brew", "install", "kubernetes-cli"], capture_output=True
    )


def test_install_via_brew_failure_raises() -> None:
    mock_result = MagicMock(returncode=1, stderr=b"error")
    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(click.ClickException, match="brew install"):
            deps._install_via_brew("kind")


# ---------------------------------------------------------------------------
# _download_kind / _download_kubectl — smoke tests with mocked network
# ---------------------------------------------------------------------------


def test_download_kind_linux(tmp_path: Path) -> None:
    fake_binary = b"\x7fELF fake-kind-binary"

    class _FakeResponse:
        def read(self) -> bytes:
            return b'{"tag_name": "v0.25.0"}'

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def _urlopen(req: object) -> "_FakeResponse":
        return _FakeResponse()

    def _urlretrieve(url: str, dest: str | Path) -> None:
        Path(dest).write_bytes(fake_binary)

    with (
        patch.object(sys, "platform", "linux"),
        patch("urllib.request.urlopen", side_effect=_urlopen),
        patch("urllib.request.urlretrieve", side_effect=_urlretrieve),
    ):
        deps._download_kind(tmp_path, "amd64")

    kind_bin = tmp_path / "kind"
    assert kind_bin.exists()
    assert kind_bin.read_bytes() == fake_binary
    if sys.platform != "win32":
        assert kind_bin.stat().st_mode & stat.S_IEXEC


def test_download_kubectl_linux(tmp_path: Path) -> None:
    fake_binary = b"\x7fELF fake-kubectl"

    class _StableResp:
        def read(self) -> bytes:
            return b"v1.30.0"

        def __enter__(self) -> "_StableResp":
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def _urlopen(url: object) -> "_StableResp":
        return _StableResp()

    def _urlretrieve(url: str, dest: str | Path) -> None:
        Path(dest).write_bytes(fake_binary)

    with (
        patch.object(sys, "platform", "linux"),
        patch("urllib.request.urlopen", side_effect=_urlopen),
        patch("urllib.request.urlretrieve", side_effect=_urlretrieve),
    ):
        deps._download_kubectl(tmp_path, "amd64")

    kubectl_bin = tmp_path / "kubectl"
    assert kubectl_bin.exists()
    assert kubectl_bin.read_bytes() == fake_binary
    if sys.platform != "win32":
        assert kubectl_bin.stat().st_mode & stat.S_IEXEC


def test_download_helm_linux(tmp_path: Path) -> None:
    """_download_helm extracts the helm binary from a .tar.gz archive."""
    fake_binary = b"\x7fELF fake-helm"

    # Build a minimal tar.gz that mimics the real Helm archive layout
    archive_buf = BytesIO()
    with tarfile.open(fileobj=archive_buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="linux-amd64/helm")
        info.size = len(fake_binary)
        tf.addfile(info, BytesIO(fake_binary))
    archive_bytes = archive_buf.getvalue()

    class _FakeTagResp:
        def read(self) -> bytes:
            return b'{"tag_name": "v3.15.0"}'

        def __enter__(self) -> "_FakeTagResp":
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def _urlopen(req: object) -> "_FakeTagResp":
        return _FakeTagResp()

    def _urlretrieve(url: str, dest: str | Path) -> None:
        Path(dest).write_bytes(archive_bytes)

    with (
        patch.object(sys, "platform", "linux"),
        patch("urllib.request.urlopen", side_effect=_urlopen),
        patch("urllib.request.urlretrieve", side_effect=_urlretrieve),
    ):
        deps._download_helm(tmp_path, "amd64")

    helm_bin = tmp_path / "helm"
    assert helm_bin.exists()
    assert helm_bin.read_bytes() == fake_binary
    if sys.platform != "win32":
        assert helm_bin.stat().st_mode & stat.S_IEXEC


def test_download_helm_windows(tmp_path: Path) -> None:
    """_download_helm extracts helm.exe from a .zip on Windows."""
    fake_binary = b"MZ fake-helm-windows"

    archive_buf = BytesIO()
    with zipfile.ZipFile(archive_buf, mode="w") as zf:
        zf.writestr("windows-amd64/helm.exe", fake_binary)
    archive_bytes = archive_buf.getvalue()

    class _FakeTagResp:
        def read(self) -> bytes:
            return b'{"tag_name": "v3.15.0"}'

        def __enter__(self) -> "_FakeTagResp":
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def _urlopen(req: object) -> "_FakeTagResp":
        return _FakeTagResp()

    def _urlretrieve(url: str, dest: str | Path) -> None:
        Path(dest).write_bytes(archive_bytes)

    with (
        patch.object(sys, "platform", "win32"),
        patch("urllib.request.urlopen", side_effect=_urlopen),
        patch("urllib.request.urlretrieve", side_effect=_urlretrieve),
    ):
        deps._download_helm(tmp_path, "amd64")

    helm_bin = tmp_path / "helm.exe"
    assert helm_bin.exists()
    assert helm_bin.read_bytes() == fake_binary


# ---------------------------------------------------------------------------
# _start_or_wait_for_docker
# ---------------------------------------------------------------------------


def test_start_or_wait_for_docker_linux_succeeds_immediately() -> None:
    with (
        patch.object(sys, "platform", "linux"),
        patch("subprocess.run") as mock_run,
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=True),
    ):
        deps._start_or_wait_for_docker()

    # systemctl start docker should have been called
    mock_run.assert_called_once_with(
        ["sudo", "systemctl", "start", "docker"], capture_output=True
    )


def test_start_or_wait_for_docker_mac_opens_app() -> None:
    with (
        patch.object(sys, "platform", "darwin"),
        patch("subprocess.run") as mock_run,
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=True),
    ):
        deps._start_or_wait_for_docker()

    mock_run.assert_called_once_with(["open", "-a", "Docker"], capture_output=True)


def test_start_or_wait_for_docker_prompts_if_timeout() -> None:
    """If daemon doesn't start within 60s, user is prompted, then re-checked.

    monotonic call sequence:
      1. deadline = monotonic() + 60  → 0.0 + 60 = 60.0
      2. while monotonic() < deadline → 61.0 < 60.0 is False — loop body never runs
    Then prompt fires, then _docker_daemon_running() is called once (returns True).
    """
    with (
        patch.object(sys, "platform", "darwin"),
        patch("subprocess.run"),
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=True),
        patch("diagrid.cli.utils.deps.time.monotonic", side_effect=[0.0, 61.0]),
        patch("diagrid.cli.utils.deps.click.prompt") as mock_prompt,
    ):
        deps._start_or_wait_for_docker()

    mock_prompt.assert_called_once()


def test_start_or_wait_for_docker_raises_if_still_not_running() -> None:
    with (
        patch.object(sys, "platform", "darwin"),
        patch("subprocess.run"),
        patch("diagrid.cli.utils.deps._docker_daemon_running", return_value=False),
        patch("diagrid.cli.utils.deps.time.monotonic", side_effect=[0.0, 61.0, 61.0]),
        patch("diagrid.cli.utils.deps.click.prompt"),
    ):
        with pytest.raises(click.ClickException, match="Docker daemon is not running"):
            deps._start_or_wait_for_docker()


# ---------------------------------------------------------------------------
# Download URL construction (no network, mocks urlretrieve)
# ---------------------------------------------------------------------------
#
# These tests pin the asset names returned by upstream release pages so that
# a future change in this code immediately tells you which URL changed,
# instead of waiting for `tests/cli/utils/test_deps_functional.py` to fail
# nightly on a 404.


_KIND_TAG = "v0.30.0"
_HELM_TAG = "v3.20.0"
_KUBECTL_VERSION = "v1.32.0"


def _capture_download_url(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch urlretrieve and capture every URL it is called with."""
    urls: list[str] = []

    def fake_urlretrieve(url: str, dest: object) -> tuple[object, object]:
        urls.append(url)
        # Touch the destination so callers that chmod/inspect it don't crash.
        Path(str(dest)).write_bytes(b"")
        return dest, None

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    return urls


@pytest.mark.parametrize(
    ("platform_name", "arch", "expected_asset", "expected_dest_name"),
    [
        ("linux", "amd64", "kind-linux-amd64", "kind"),
        ("linux", "arm64", "kind-linux-arm64", "kind"),
        ("darwin", "amd64", "kind-darwin-amd64", "kind"),
        ("darwin", "arm64", "kind-darwin-arm64", "kind"),
        # Regression: kind's Windows asset has no `.exe` suffix on the
        # download URL (was the cause of nightly integration.yaml 404s),
        # but the local destination must still be `kind.exe` so Windows
        # can execute it.
        ("win32", "amd64", "kind-windows-amd64", "kind.exe"),
    ],
)
def test_download_kind_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_name: str,
    arch: str,
    expected_asset: str,
    expected_dest_name: str,
) -> None:
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(
        "diagrid.cli.utils.deps._github_latest_tag",
        lambda repo: _KIND_TAG,
    )
    urls = _capture_download_url(monkeypatch)

    deps._download_kind(tmp_path, arch)

    assert urls == [
        f"https://github.com/kubernetes-sigs/kind/releases/download/{_KIND_TAG}/{expected_asset}"
    ]
    assert (tmp_path / expected_dest_name).exists()


@pytest.mark.parametrize(
    ("platform_name", "arch", "expected_segment", "expected_dest_name"),
    [
        ("linux", "amd64", "linux/amd64/kubectl", "kubectl"),
        ("linux", "arm64", "linux/arm64/kubectl", "kubectl"),
        ("darwin", "amd64", "darwin/amd64/kubectl", "kubectl"),
        ("darwin", "arm64", "darwin/arm64/kubectl", "kubectl"),
        ("win32", "amd64", "windows/amd64/kubectl.exe", "kubectl.exe"),
    ],
)
def test_download_kubectl_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_name: str,
    arch: str,
    expected_segment: str,
    expected_dest_name: str,
) -> None:
    monkeypatch.setattr(sys, "platform", platform_name)

    fake_resp = MagicMock()
    fake_resp.read.return_value = _KUBECTL_VERSION.encode()
    fake_resp.__enter__ = lambda s: s
    fake_resp.__exit__ = lambda *a: False
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: fake_resp)
    urls = _capture_download_url(monkeypatch)

    deps._download_kubectl(tmp_path, arch)

    assert urls == [
        f"https://dl.k8s.io/release/{_KUBECTL_VERSION}/bin/{expected_segment}"
    ]
    assert (tmp_path / expected_dest_name).exists()


@pytest.mark.parametrize(
    ("platform_name", "arch", "expected_archive"),
    [
        ("linux", "amd64", f"helm-{_HELM_TAG}-linux-amd64.tar.gz"),
        ("linux", "arm64", f"helm-{_HELM_TAG}-linux-arm64.tar.gz"),
        ("darwin", "amd64", f"helm-{_HELM_TAG}-darwin-amd64.tar.gz"),
        ("darwin", "arm64", f"helm-{_HELM_TAG}-darwin-arm64.tar.gz"),
        ("win32", "amd64", f"helm-{_HELM_TAG}-windows-amd64.zip"),
    ],
)
def test_download_helm_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_name: str,
    arch: str,
    expected_archive: str,
) -> None:
    """Pin the Helm download URL pattern across OS/arch combinations.

    Helm extracts an archive after download — we don't need to verify the
    archive contents here, just that the URL is what we expect. We mock
    the extraction so the test stays fast and offline.
    """
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(
        "diagrid.cli.utils.deps._github_latest_tag",
        lambda repo: _HELM_TAG,
    )
    urls = _capture_download_url(monkeypatch)
    # Skip archive extraction — we only care about the URL here.
    monkeypatch.setattr(
        "zipfile.ZipFile",
        lambda *a, **kw: MagicMock(
            __enter__=lambda s: s, __exit__=lambda *a: False, namelist=lambda: []
        ),
    )
    monkeypatch.setattr(
        "tarfile.open",
        lambda *a, **kw: MagicMock(
            __enter__=lambda s: s, __exit__=lambda *a: False, getmembers=lambda: []
        ),
    )
    # Helm copies an extracted binary to install_dir; tolerate it not finding one.
    try:
        deps._download_helm(tmp_path, arch)
    except (FileNotFoundError, StopIteration, KeyError, IndexError):
        # Extraction returns nothing in the mock; the URL was already captured.
        pass

    assert urls == [f"https://get.helm.sh/{expected_archive}"]
