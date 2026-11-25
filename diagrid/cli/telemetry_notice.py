# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The one-time notice ``diagridpy`` prints before it first reports usage.

Usage reporting is opt-out, so a CLI user who never reads the README would
otherwise not know it happens. The first run that is about to send an event
prints this notice to stderr and records that in a marker file, the way
``dotnet``, Homebrew and the Azure CLI do.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import click

NOTICE = (
    "diagridpy sends anonymous usage data to Diagrid (Reo): package version, "
    "OS, architecture and Python version.\n"
    "To turn it off, set DO_NOT_TRACK=1 or DIAGRID_NO_ANALYTICS=1. Details: "
    "https://github.com/diagridio/python-ai#usage-analytics\n"
    "\n"
)

MARKER_NAME = "telemetry-notice-shown"


def marker_path() -> Path:
    """Return the marker file path, resolved now rather than at import.

    Same rule as ``diagrid.core.config.constants.DIAGRID_DIR``: ``$DIAGRID_HOME``
    when set, else ``~/.diagrid``. That constant is computed at import, so it
    would miss a ``DIAGRID_HOME`` set afterwards.
    """
    home = os.environ.get("DIAGRID_HOME")
    base = Path(home) if home else Path.home() / ".diagrid"
    return base / MARKER_NAME


def show_notice_once() -> None:
    """Print the notice to stderr unless it was shown before. Never raises.

    The marker is written after printing. If it can't be written, the notice
    shows again on the next run, which is accepted.
    """
    try:
        marker = marker_path()
        if marker.exists():
            return
        click.echo(NOTICE, err=True, nl=False)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except Exception:
        logging.getLogger(__name__).debug(
            "telemetry notice marker could not be written", exc_info=True
        )
