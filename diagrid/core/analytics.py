# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Anonymous usage reporting for the Diagrid Python packages.

PyPI publishes aggregate download counts only. This module reports one event
per package per process with the package version and the host platform, so
Diagrid can see which versions run, and where. No application data is
collected. See the "Usage analytics" section of the README, including how to
opt out.

One event per process means one event per replica per restart on Kubernetes.
The numbers count process starts, not deployments or users.

The call never blocks and never raises: it runs on a daemon thread and swallows
every failure. The one second timeout bounds each socket operation, not DNS
resolution, so the thread itself can outlive a second on a network that
blackholes lookups. The calling thread never waits on it. Blocked egress and
air-gapped clusters are normal conditions, not faults.

Set ``USAGE_ENDPOINT`` to an empty string to turn the module into a no-op.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import urllib.parse
import urllib.request
from importlib import metadata

logger = logging.getLogger(__name__)

# Scarf event-collection route for python-ai (package c9dadc82, owner Diagrid).
# The route records the request and redirects nowhere. An empty string disables
# reporting entirely.
USAGE_ENDPOINT = "https://diagrid.gateway.scarf.sh/python-ai"
USAGE_TIMEOUT_SECONDS = 1.0

# The cross-ecosystem DO_NOT_TRACK convention, Scarf's own variable, and a
# Diagrid-specific opt-out. Any of them set to a truthy value disables reporting.
OPT_OUT_ENV_VARS = ("DO_NOT_TRACK", "SCARF_NO_ANALYTICS", "DIAGRID_NO_ANALYTICS")
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# ``CI`` is the convention most vendors follow. The rest cover vendors that set
# their own flag but not ``CI``. Presence is enough for the last two.
_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "TF_BUILD",
    "BUILDKITE",
    "JENKINS_URL",
)

# The Dapr SDK variables that point a process at Catalyst.
_DAPR_ENDPOINT_ENV_VARS = ("DAPR_GRPC_ENDPOINT", "DAPR_HTTP_ENDPOINT")
_CATALYST_HOST_SUFFIX = "diagrid.io"

_DIMENSION_MAX_LEN = 64

_reported_lock = threading.Lock()
_reported_packages: set[str] = set()


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY_VALUES


def usage_reporting_disabled() -> bool:
    """Return True when the user opted out through any supported variable."""
    return any(_is_truthy(os.environ.get(name, "")) for name in OPT_OUT_ENV_VARS)


def running_in_ci() -> bool:
    """Return True when a well-known CI variable is set.

    Reported as the ``ci`` dimension so pipeline runs can be separated from
    real usage on the dashboard.
    """
    return any(
        _is_truthy(os.environ.get(name, "")) for name in _CI_ENV_VARS[:6]
    ) or any(os.environ.get(name, "").strip() for name in _CI_ENV_VARS[6:])


def detect_target() -> str:
    """Return ``catalyst`` when the process points at Diagrid Catalyst, else ``dapr``.

    Catalyst is configured through the Dapr SDK variables: the endpoint host is
    under ``diagrid.io``, and Catalyst issues ``DAPR_API_TOKEN``. A self-hosted
    sidecar with API token authentication also reads as ``catalyst``. That is an
    approximation, and the dashboard reads it as one.
    """
    for name in _DAPR_ENDPOINT_ENV_VARS:
        host = urllib.parse.urlsplit(os.environ.get(name, "").strip()).hostname or ""
        if host == _CATALYST_HOST_SUFFIX or host.endswith("." + _CATALYST_HOST_SUFFIX):
            return "catalyst"
    if os.environ.get("DAPR_API_TOKEN", "").strip():
        return "catalyst"
    return "dapr"


def package_version(package: str) -> str:
    """Return the installed version of ``package``, or ``unknown``."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def _clean_dimension(value: object) -> str:
    return str(value).strip()[:_DIMENSION_MAX_LEN]


def build_url(package: str, version: str, **dimensions: object) -> str:
    """Build the event URL for ``package``. Exposed for tests.

    The defaults describe the package and the host. ``dimensions`` are added by
    the caller (for example ``kind`` and ``framework``) and override a default
    of the same name. Empty values are dropped.
    """
    params: dict[str, str] = {
        "package": package,
        "version": version,
        "core_version": package_version("diagrid-core"),
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "python_version": platform.python_version(),
        "target": detect_target(),
        "ci": "true" if running_in_ci() else "false",
    }
    for key, value in dimensions.items():
        if value is None:
            continue
        cleaned = _clean_dimension(value)
        if cleaned:
            params[key] = cleaned
    return f"{USAGE_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _send_event(package: str, dimensions: dict[str, object]) -> None:
    """Send one event and swallow every failure."""
    try:
        version = package_version(package)
        request = urllib.request.Request(
            build_url(package, version, **dimensions),
            headers={"User-Agent": f"{package}/{version}"},
        )
        with urllib.request.urlopen(request, timeout=USAGE_TIMEOUT_SECONDS):
            pass
        logger.debug("usage event sent for %s %s", package, version)
    except Exception as exc:
        logger.debug("usage event for %s not sent: %r", package, exc)


def report_usage(package: str, **dimensions: object) -> None:
    """Report one usage event per package per process, on a daemon thread.

    ``package`` is the distribution that triggered the call, for example
    ``diagrid`` for an agent runner. ``dimensions`` are extra query parameters
    such as ``kind`` and ``framework``. Never blocks the caller and never
    raises. Does nothing while ``USAGE_ENDPOINT`` is empty or when the user
    opted out.
    """
    try:
        if not USAGE_ENDPOINT:
            return
        with _reported_lock:
            if package in _reported_packages:
                return
            _reported_packages.add(package)

        if usage_reporting_disabled():
            logger.debug(
                "usage reporting for %s is disabled by the environment", package
            )
            return

        thread = threading.Thread(
            target=_send_event,
            args=(package, dict(dimensions)),
            name="diagrid-usage",
            daemon=True,
        )
        thread.start()
    except Exception:
        pass
