# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Dependency preflight check and auto-install for diagridpy commands."""

from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import click

from diagrid.cli.utils import console


REQUIRED_BINARIES = ["docker", "kind", "kubectl", "helm", "piko"]

_BREW_FORMULAS = {
    "kind": "kind",
    "kubectl": "kubernetes-cli",
    "helm": "helm",
}

_WINGET_IDS = {
    "kind": "Kubernetes.kind",
    "kubectl": "Kubernetes.kubectl",
    "helm": "Helm.Helm",
}


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------


def _is_mac() -> bool:
    return sys.platform == "darwin"


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_windows() -> bool:
    return sys.platform == "win32"


def _arch() -> str:
    """Return normalized architecture string (amd64 or arm64)."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


# ---------------------------------------------------------------------------
# PATH management
# ---------------------------------------------------------------------------


def _install_dir() -> Path:
    """Return the directory where auto-installed binaries are placed."""
    if _is_windows():
        local_app_data = os.environ.get(
            "LOCALAPPDATA", str(Path.home() / "AppData" / "Local")
        )
        return Path(local_app_data) / "Programs" / "diagrid" / "bin"
    return Path.home() / ".local" / "bin"


def _ensure_in_path(directory: Path) -> None:
    """Add directory to os.environ['PATH'] if not already present."""
    dir_str = str(directory)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if dir_str not in path_entries:
        os.environ["PATH"] = dir_str + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------


def _missing_binaries() -> list[str]:
    """Return list of required binaries not found in PATH."""
    return [name for name in REQUIRED_BINARIES if shutil.which(name) is None]


# ---------------------------------------------------------------------------
# Docker daemon check
# ---------------------------------------------------------------------------


def _docker_daemon_running() -> bool:
    """Check if the Docker daemon is responsive."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------


def preflight_check() -> None:
    """Ensure all required binaries exist and Docker daemon is running."""
    # Pick up previously auto-installed binaries before checking
    install_dir = _install_dir()
    _ensure_in_path(install_dir)

    missing = _missing_binaries()

    if missing:
        names = ", ".join(missing)
        console.warning(f"Missing required tools: {names}")
        if not click.confirm("  Install missing tools automatically?", default=True):
            console.error(
                "Required tools must be installed to continue. "
                "Please install them manually and retry."
            )
            raise SystemExit(0)

        for name in missing:
            console.info(f"Installing {name}...")
            _install_binary(name)
            console.success(f"{name} installed")

        # Verify every binary is now resolvable
        still_missing = _missing_binaries()
        if still_missing:
            raise click.ClickException(
                f"Failed to install: {', '.join(still_missing)}. "
                "Please install them manually and retry."
            )

        console.success("All required tools are ready")

    if not _docker_daemon_running():
        console.warning(
            "Docker is installed but not running. Starting Docker Desktop..."
        )
        _start_or_wait_for_docker()
        console.success("Docker daemon started")


# ---------------------------------------------------------------------------
# Install dispatch
# ---------------------------------------------------------------------------


def _install_binary(name: str) -> None:
    """Install a single binary using the best available strategy."""
    if name == "docker":
        _install_docker()
        return

    if _is_mac() and shutil.which("brew") and name in _BREW_FORMULAS:
        _install_via_brew(name)
    elif _is_windows() and shutil.which("winget") and name in _WINGET_IDS:
        _install_via_winget(name)
    else:
        _install_via_download(name)


def _install_via_brew(name: str) -> None:
    """Install a binary via Homebrew."""
    formula = _BREW_FORMULAS.get(name, name)
    result = subprocess.run(["brew", "install", formula], capture_output=True)
    if result.returncode != 0:
        raise click.ClickException(
            f"brew install {formula} failed:\n{result.stderr.decode()}"
        )


def _install_via_winget(name: str) -> None:
    """Install a binary via winget."""
    winget_id = _WINGET_IDS.get(name, name)
    result = subprocess.run(
        [
            "winget",
            "install",
            "--id",
            winget_id,
            "-e",
            "--accept-source-agreements",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"winget install {winget_id} failed:\n{result.stderr.decode()}"
        )


def _install_via_download(name: str) -> None:
    """Install a binary via direct download to the local install directory."""
    install_dir = _install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)

    arch = _arch()
    if name == "kind":
        _download_kind(install_dir, arch)
    elif name == "kubectl":
        _download_kubectl(install_dir, arch)
    elif name == "helm":
        _download_helm(install_dir, arch)
    elif name == "piko":
        _download_piko(install_dir, arch)
    else:
        raise click.ClickException(f"No download strategy for '{name}'")

    _ensure_in_path(install_dir)


# ---------------------------------------------------------------------------
# Direct download helpers
# ---------------------------------------------------------------------------


def _make_executable(path: Path) -> None:
    """Set executable bits on a file (no-op on Windows)."""
    if not _is_windows():
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _github_latest_tag(repo: str) -> str:
    """Return the latest release tag from a GitHub repo (e.g. 'kubernetes-sigs/kind')."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"User-Agent": "diagridpy"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        data = json.loads(resp.read())
    return data["tag_name"]


def _download_kind(install_dir: Path, arch: str) -> None:
    """Download the kind binary from GitHub releases."""
    version = _github_latest_tag("kubernetes-sigs/kind")
    os_name = "darwin" if _is_mac() else "windows" if _is_windows() else "linux"
    ext = ".exe" if _is_windows() else ""
    filename = f"kind-{os_name}-{arch}{ext}"
    url = f"https://github.com/kubernetes-sigs/kind/releases/download/{version}/{filename}"
    dest = install_dir / f"kind{ext}"
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    _make_executable(dest)


def _download_kubectl(install_dir: Path, arch: str) -> None:
    """Download kubectl from the official Kubernetes release URL."""
    version_url = "https://dl.k8s.io/release/stable.txt"
    with urllib.request.urlopen(version_url) as resp:  # noqa: S310
        version = resp.read().decode().strip()

    os_name = "darwin" if _is_mac() else "windows" if _is_windows() else "linux"
    ext = ".exe" if _is_windows() else ""
    url = f"https://dl.k8s.io/release/{version}/bin/{os_name}/{arch}/kubectl{ext}"
    dest = install_dir / f"kubectl{ext}"
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    _make_executable(dest)


def _download_helm(install_dir: Path, arch: str) -> None:
    """Download Helm from the official Helm release URL."""
    version = _github_latest_tag("helm/helm")
    os_name = "darwin" if _is_mac() else "windows" if _is_windows() else "linux"

    if _is_windows():
        archive_name = f"helm-{version}-{os_name}-{arch}.zip"
    else:
        archive_name = f"helm-{version}-{os_name}-{arch}.tar.gz"

    url = f"https://get.helm.sh/{archive_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / archive_name
        urllib.request.urlretrieve(url, archive_path)  # noqa: S310

        if _is_windows():
            with zipfile.ZipFile(archive_path) as zf:
                for member in zf.namelist():
                    if member.endswith("helm.exe"):
                        data = zf.read(member)
                        dest = install_dir / "helm.exe"
                        dest.write_bytes(data)
                        break
        else:
            with tarfile.open(archive_path) as tf:
                for tar_member in tf.getmembers():
                    if tar_member.name.endswith("/helm"):
                        fileobj = tf.extractfile(tar_member)
                        if fileobj:
                            dest = install_dir / "helm"
                            dest.write_bytes(fileobj.read())
                            _make_executable(dest)
                        break


def _download_piko(install_dir: Path, arch: str) -> None:
    """Download the piko binary from GitHub releases."""
    version = _github_latest_tag("andydunstall/piko")
    os_name = "darwin" if _is_mac() else "linux"
    filename = f"piko-{os_name}-{arch}"
    url = f"https://github.com/andydunstall/piko/releases/download/{version}/{filename}"
    dest = install_dir / "piko"
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    _make_executable(dest)


# ---------------------------------------------------------------------------
# Docker install
# ---------------------------------------------------------------------------


def _install_docker() -> None:
    """Install Docker based on the current platform."""
    if _is_linux():
        _install_docker_linux()
    elif _is_mac():
        _install_docker_mac()
    elif _is_windows():
        _install_docker_windows()
    else:
        raise click.ClickException(
            f"Unsupported platform for Docker install: {sys.platform}"
        )


def _install_docker_linux() -> None:
    """Install Docker on Linux using the official install script."""
    console.info("Downloading Docker install script from get.docker.com...")
    script_url = "https://get.docker.com"
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
        script_path = f.name
        req = urllib.request.Request(script_url, headers={"User-Agent": "diagridpy"})
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            f.write(resp.read())

    try:
        result = subprocess.run(["sudo", "sh", script_path], capture_output=True)
        if result.returncode != 0:
            raise click.ClickException(
                f"Docker install script failed:\n{result.stderr.decode()}"
            )
    finally:
        os.unlink(script_path)

    user = os.environ.get("USER", "")
    if user:
        subprocess.run(["sudo", "usermod", "-aG", "docker", user], capture_output=True)
    subprocess.run(
        ["sudo", "systemctl", "enable", "--now", "docker"], capture_output=True
    )
    console.warning(
        "Docker installed. You may need to run 'newgrp docker' or log out and back "
        "in for group membership to take effect."
    )


def _install_docker_mac() -> None:
    """Install Docker Desktop on macOS."""
    if shutil.which("brew"):
        result = subprocess.run(
            ["brew", "install", "--cask", "docker"], capture_output=True
        )
        if result.returncode != 0:
            raise click.ClickException(
                f"brew install --cask docker failed:\n{result.stderr.decode()}"
            )
        _start_or_wait_for_docker()
    else:
        raise click.ClickException(
            "Docker Desktop is required but not installed.\n"
            "Please install it from https://www.docker.com/products/docker-desktop/ "
            "and retry."
        )


def _install_docker_windows() -> None:
    """Install Docker Desktop on Windows."""
    if shutil.which("winget"):
        result = subprocess.run(
            [
                "winget",
                "install",
                "--id",
                "Docker.DockerDesktop",
                "-e",
                "--accept-source-agreements",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise click.ClickException(
                f"winget install Docker.DockerDesktop failed:\n{result.stderr.decode()}"
            )
        _start_or_wait_for_docker()
    else:
        raise click.ClickException(
            "Docker Desktop is required but not installed.\n"
            "Please install it from https://www.docker.com/products/docker-desktop/ "
            "and retry."
        )


# ---------------------------------------------------------------------------
# Docker daemon start / poll
# ---------------------------------------------------------------------------


def _start_or_wait_for_docker() -> None:
    """Attempt to start the Docker daemon and poll until it responds."""
    if _is_linux():
        subprocess.run(["sudo", "systemctl", "start", "docker"], capture_output=True)
    elif _is_mac():
        subprocess.run(["open", "-a", "Docker"], capture_output=True)
    elif _is_windows():
        subprocess.Popen(["Docker Desktop.exe"], shell=True)  # noqa: S602

    # Poll for up to 60 seconds
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if _docker_daemon_running():
            return
        time.sleep(2)

    # Still not running — ask the user to start it manually
    click.prompt(
        "Docker is not responding. Please start Docker manually and press Enter to continue",
        default="",
        prompt_suffix="",
        show_default=False,
    )
    if not _docker_daemon_running():
        raise click.ClickException(
            "Docker daemon is not running. Please start Docker and retry."
        )
