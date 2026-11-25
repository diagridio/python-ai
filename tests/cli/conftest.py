# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_diagrid_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the telemetry notice marker out of the developer's real home.

    ``diagridpy`` writes ``telemetry-notice-shown`` under ``$DIAGRID_HOME`` (or
    ``~/.diagrid``) the first time it reports. Pointing ``DIAGRID_HOME`` at a
    per-test directory means no CLI test touches the real one, and every test
    starts as a first run.
    """
    monkeypatch.setenv("DIAGRID_HOME", str(tmp_path / "diagrid-home"))
