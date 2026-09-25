# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Anonymous usage reporting for the Diagrid Python packages.

PyPI publishes aggregate download counts only. This module reports one event
per process with the package version and the host platform, so Diagrid can see
which versions run, and where. No application data is collected. See the
"Usage analytics" section of the README, including how to opt out.

The call never blocks and never raises: it runs on a daemon thread with a short
timeout and swallows every failure. Blocked egress and air-gapped clusters are
normal conditions, not faults.

``USAGE_ENDPOINT`` is empty until the Scarf event-collection package exists.
While it is empty, this module does nothing at all.
"""

from __future__ import annotations

import os
import platform
import threading
import urllib.parse
import urllib.request
from importlib import metadata

# TODO(scarf): set to the Scarf event-collection URL for python-ai, for example
# "https://diagrid.gateway.scarf.sh/python-ai". An empty string disables
# reporting entirely.
USAGE_ENDPOINT = ""
USAGE_TIMEOUT_SECONDS = 1.0

# The cross-ecosystem DO_NOT_TRACK convention, Scarf's own variable, and a
# Diagrid-specific opt-out. Any of them set to a truthy value disables reporting.
OPT_OUT_ENV_VARS = ("DO_NOT_TRACK", "SCARF_NO_ANALYTICS", "DIAGRID_NO_ANALYTICS")
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

_reported_lock = threading.Lock()
_reported = False


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY_VALUES


def usage_reporting_disabled() -> bool:
    """Return True when the user opted out through any supported variable."""
    return any(_is_truthy(os.environ.get(name, "")) for name in OPT_OUT_ENV_VARS)


def package_version(package: str) -> str:
    """Return the installed version of ``package``, or ``unknown``."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def build_url(package: str) -> str:
    """Build the event URL for ``package``. Exposed for tests."""
    params = urllib.parse.urlencode(
        {
            "package": package,
            "version": package_version(package),
            "os": platform.system().lower(),
            "arch": platform.machine().lower(),
            "python_version": platform.python_version(),
        }
    )
    return f"{USAGE_ENDPOINT}?{params}"


def _send_event(package: str) -> None:
    """Send one event and swallow every failure."""
    try:
        request = urllib.request.Request(
            build_url(package),
            headers={"User-Agent": f"{package}/{package_version(package)}"},
        )
        with urllib.request.urlopen(request, timeout=USAGE_TIMEOUT_SECONDS):
            pass
    except Exception:
        pass


def report_usage(package: str) -> None:
    """Report one usage event per process, on a background daemon thread.

    ``package`` is the distribution that triggered the call, for example
    ``diagrid`` for an agent runner or ``diagrid-cli`` for the CLI. Never blocks
    the caller and never raises. Does nothing while ``USAGE_ENDPOINT`` is empty
    or when the user opted out.
    """
    global _reported

    try:
        if not USAGE_ENDPOINT:
            return
        with _reported_lock:
            if _reported:
                return
            _reported = True

        if usage_reporting_disabled():
            return

        thread = threading.Thread(
            target=_send_event, args=(package,), name="diagrid-usage", daemon=True
        )
        thread.start()
    except Exception:
        pass
