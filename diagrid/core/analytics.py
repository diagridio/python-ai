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

Events go to every destination in ``USAGE_SINKS``: Scarf and Reo. Reo is gated
more strictly than Scarf (see ``reo_allowed``): it never receives events from
Kubernetes, Catalyst or CI. Set ``USAGE_ENDPOINT`` to an empty string to turn
the Scarf destination into a no-op, and ``REO_ENDPOINT`` to an empty string to
do the same for Reo.

The destinations are sent to one after another on the same daemon thread, so a
slow Scarf send can delay the Reo send by up to the timeout.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any

logger = logging.getLogger(__name__)

# Scarf event-collection route for python-ai (package c9dadc82, owner Diagrid).
# The route records the request and redirects nowhere. An empty string disables
# reporting entirely.
USAGE_ENDPOINT = "https://diagrid.gateway.scarf.sh/python-ai"
USAGE_TIMEOUT_SECONDS = 1.0

# Reo event-collection endpoint. Reo accepts a GET with the meta keys as query
# parameters and no authentication. An empty string disables the destination.
REO_ENDPOINT = "https://telemetry.diagrid.io/data"

# The cross-ecosystem DO_NOT_TRACK convention, Scarf's own variable, and a
# Diagrid-specific opt-out. Any of them set to a truthy value disables reporting.
OPT_OUT_ENV_VARS = ("DO_NOT_TRACK", "SCARF_NO_ANALYTICS", "DIAGRID_NO_ANALYTICS")
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# ``CI`` is the convention most vendors follow. The rest cover vendors that set
# their own flag but not ``CI``.
_CI_TRUTHY_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "TF_BUILD",
)
# Vendors that set a value rather than a flag. Presence is enough.
_CI_PRESENCE_ENV_VARS = ("BUILDKITE", "JENKINS_URL")

# The Dapr SDK variables that point a process at Catalyst.
_DAPR_ENDPOINT_ENV_VARS = ("DAPR_GRPC_ENDPOINT", "DAPR_HTTP_ENDPOINT")
_CATALYST_HOST_SUFFIX = "diagrid.io"

_DIMENSION_MAX_LEN = 64

_reported_lock = threading.Lock()
_reported_packages: set[tuple[str, str]] = set()


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
    if any(_is_truthy(os.environ.get(name, "")) for name in _CI_TRUTHY_ENV_VARS):
        return True
    return any(os.environ.get(name, "").strip() for name in _CI_PRESENCE_ENV_VARS)


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


def _default_dimensions(package: str, version: str) -> dict[str, str]:
    """Return the dimensions every destination reports: the package and the host."""
    return {
        "package": package,
        "version": version,
        "core_version": package_version("diagrid-core"),
        "os": platform.system().lower(),
        "arch": platform.machine().lower(),
        "python_version": platform.python_version(),
        "target": detect_target(),
        "ci": "true" if running_in_ci() else "false",
    }


def _merged_dimensions(
    package: str, version: str, dimensions: Mapping[str, object]
) -> dict[str, str]:
    """Return the defaults overlaid with the caller's dimensions, cleaned.

    ``None`` and empty values are dropped and every value is capped. Shared by
    every destination so they cannot drift apart.
    """
    params = _default_dimensions(package, version)
    for key, value in dimensions.items():
        if value is None:
            continue
        cleaned = _clean_dimension(value)
        if cleaned:
            params[key] = cleaned
    return params


def build_url(package: str, version: str, **dimensions: object) -> str:
    """Build the Scarf event URL for ``package``. Exposed for tests.

    The defaults describe the package and the host. ``dimensions`` are added by
    the caller (for example ``kind`` and ``framework``) and override a default
    of the same name. Empty values are dropped.
    """
    params = _merged_dimensions(package, version, dimensions)
    return f"{USAGE_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _always() -> bool:
    return True


@dataclass(frozen=True)
class Sink:
    """One usage-analytics destination.

    ``allowed`` is that destination's gate, evaluated on the calling thread
    before anything is sent. ``build_request`` turns a package and the caller's
    dimensions into the request to send. ``configured`` is False while the
    destination has no endpoint, which makes it a no-op that is not even counted
    as reported.
    """

    name: str
    allowed: Callable[[], bool]
    build_request: Callable[[str, Mapping[str, Any]], urllib.request.Request]
    configured: Callable[[], bool] = _always


def _scarf_request(
    package: str, dimensions: Mapping[str, Any]
) -> urllib.request.Request:
    version = package_version(package)
    return urllib.request.Request(
        build_url(package, version, **dimensions),
        headers={"User-Agent": f"{package}/{version}"},
    )


SCARF_SINK = Sink(
    name="scarf",
    allowed=lambda: not usage_reporting_disabled(),
    build_request=_scarf_request,
    configured=lambda: bool(USAGE_ENDPOINT),
)


def reo_allowed() -> bool:
    """Return True when a Reo event may be sent. Fails closed.

    Stricter than Scarf's gate. All of these must hold: no opt-out variable is
    set, the process is not inside a Kubernetes cluster (no exceptions), it is
    not pointed at Catalyst, and it is not running in CI. Anything that cannot
    be determined, including an exception, means no event.
    """
    try:
        if usage_reporting_disabled():
            return False
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            return False
        if detect_target() != "dapr":
            return False
        return not running_in_ci()
    except Exception:
        return False


# Collected name -> the key Reo registered. Names absent from this table (target
# and ci) are not sent to Reo: the gate makes both constant on every Reo event.
_REO_KEY_MAP = {
    "package": "product",
    "version": "version",
    "python_version": "python",
    "os": "platform",
    "arch": "arch",
    "kind": "kind",
    "framework": "framework",
    "framework_version": "framework_version",
    "core_version": "core_version",
}


def build_reo_url(package: str, version: str, **dimensions: object) -> str:
    """Build the Reo ``pypi_download`` event URL for ``package``. Exposed for tests."""
    merged = _merged_dimensions(package, version, dimensions)
    params = {"activity_type": "pypi_download"}
    for collected, reo_key in _REO_KEY_MAP.items():
        if collected in merged:
            params[reo_key] = merged[collected]
    return f"{REO_ENDPOINT}?{urllib.parse.urlencode(params)}"


def _reo_request(package: str, dimensions: Mapping[str, Any]) -> urllib.request.Request:
    version = package_version(package)
    return urllib.request.Request(
        build_reo_url(package, version, **dimensions),
        headers={"User-Agent": f"{package}/{version}"},
    )


REO_SINK = Sink(
    name="reo",
    allowed=reo_allowed,
    build_request=_reo_request,
    configured=lambda: bool(REO_ENDPOINT),
)

USAGE_SINKS: tuple[Sink, ...] = (SCARF_SINK, REO_SINK)


def _send_event(
    package: str,
    dimensions: dict[str, object],
    sinks: Sequence[Sink] = USAGE_SINKS,
) -> None:
    """Send one event to each sink and swallow every failure."""
    for sink in sinks:
        try:
            request = sink.build_request(package, dimensions)
            with urllib.request.urlopen(request, timeout=USAGE_TIMEOUT_SECONDS):
                pass
            logger.debug("usage event sent for %s to %s", package, sink.name)
        except Exception as exc:
            logger.debug(
                "usage event for %s not sent to %s: %r", package, sink.name, exc
            )


def report_usage(
    package: str, *, sinks: Sequence[Sink] = USAGE_SINKS, **dimensions: object
) -> None:
    """Report one usage event per package per sink per process, on a daemon thread.

    ``package`` is the distribution that triggered the call, for example
    ``diagrid`` for an agent runner. ``dimensions`` are extra query parameters
    such as ``kind`` and ``framework``. Never blocks the caller and never
    raises. A sink is skipped while it has no endpoint or when its gate is
    closed, for example when the user opted out.
    """
    try:
        selected: list[Sink] = []
        for sink in sinks:
            try:
                if not sink.configured():
                    continue
                key = (sink.name, package)
                with _reported_lock:
                    if key in _reported_packages:
                        continue
                    _reported_packages.add(key)

                if not sink.allowed():
                    logger.debug(
                        "usage reporting for %s to %s is disabled", package, sink.name
                    )
                    continue
                selected.append(sink)
            except Exception:
                continue
        if not selected:
            return

        thread = threading.Thread(
            target=_send_event,
            args=(package, dict(dimensions)),
            kwargs={"sinks": tuple(selected)},
            name="diagrid-usage",
            daemon=True,
        )
        thread.start()
    except Exception:
        pass
