# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Checks on the PyPI metadata of the three distributions built from this repo."""

from __future__ import annotations

import email.parser
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PACKAGE_DIRS = {
    "diagrid": ROOT,
    "diagrid-core": ROOT / "diagrid" / "core",
    "diagrid-cli": ROOT / "diagrid" / "cli",
}

DESCRIPTIONS = {
    "diagrid": (
        "Durable execution for AI agents: LangGraph, CrewAI, Pydantic AI, "
        "OpenAI Agents SDK, Google ADK, Strands and more, on Dapr Workflow"
    ),
    "diagrid-core": (
        "Shared auth, configuration and Catalyst API client for the Diagrid "
        "durable AI agent integrations"
    ),
    "diagrid-cli": (
        "diagridpy CLI: scaffold, deploy and chaos-test durable AI agents on "
        "Dapr and Diagrid Catalyst"
    ),
}

ROOT_KEYWORDS = [
    "durable-execution",
    "durable-workflows",
    "ai-agents",
    "agents",
    "dapr",
    "dapr-workflow",
    "workflow",
    "diagrid",
    "catalyst",
    "langgraph",
    "crewai",
    "pydantic-ai",
    "openai-agents",
    "google-adk",
    "strands-agents",
    "langchain",
    "smolagents",
    "deepagents",
    "claude-agent-sdk",
    "fault-tolerance",
]

CLASSIFIERS = {
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: Other/Proprietary License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
}

URLS = {
    "Homepage": "https://www.diagrid.io/catalyst",
    "Documentation": "https://docs.diagrid.io",
    "Source": "https://github.com/diagridio/python-ai",
    "Issues": "https://github.com/diagridio/python-ai/issues",
    "Examples": "https://github.com/diagridio/python-ai/tree/main/examples",
}

# A markdown link or image whose target is neither absolute nor an anchor.
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)[^)]*\)")


def _project(name: str) -> dict:
    with open(PACKAGE_DIRS[name] / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


@pytest.mark.parametrize("name", sorted(PACKAGE_DIRS))
def test_description(name: str) -> None:
    assert _project(name)["description"] == DESCRIPTIONS[name]


@pytest.mark.parametrize("name", sorted(PACKAGE_DIRS))
def test_classifiers(name: str) -> None:
    assert set(_project(name)["classifiers"]) == CLASSIFIERS


@pytest.mark.parametrize("name", sorted(PACKAGE_DIRS))
def test_urls(name: str) -> None:
    assert _project(name)["urls"] == URLS


@pytest.mark.parametrize("name", sorted(PACKAGE_DIRS))
def test_keywords_present(name: str) -> None:
    keywords = _project(name)["keywords"]
    assert keywords
    assert len(keywords) == len(set(keywords))
    assert {"durable-execution", "dapr", "diagrid", "catalyst"} <= set(keywords)


def test_root_keywords() -> None:
    assert _project("diagrid")["keywords"] == ROOT_KEYWORDS


@pytest.mark.parametrize("name", sorted(PACKAGE_DIRS))
def test_license_declares_busl_file(name: str) -> None:
    assert _project(name)["license"] == {"file": "LICENSE.md"}
    license_file = PACKAGE_DIRS[name] / "LICENSE.md"
    # The sub-packages carry a copy because setuptools only reads license files
    # from inside the project directory. The copy must not drift from the root.
    assert license_file.read_bytes() == (ROOT / "LICENSE.md").read_bytes()
    assert "Business Source License 1.1" in license_file.read_text()


@pytest.mark.parametrize("name", sorted(PACKAGE_DIRS))
def test_readme_has_no_relative_links(name: str) -> None:
    readme = PACKAGE_DIRS[name] / _project(name)["readme"]
    assert RELATIVE_LINK.findall(readme.read_text()) == []


_PREPARE_METADATA = (
    "import sys\n"
    "from setuptools import build_meta\n"
    "print(build_meta.prepare_metadata_for_build_wheel(sys.argv[1]))\n"
)


@pytest.mark.parametrize("name", sorted(PACKAGE_DIRS))
def test_built_wheel_metadata(name: str, tmp_path: Path) -> None:
    """Run the setuptools backend and check the METADATA a wheel would carry."""
    pytest.importorskip("setuptools")
    result = subprocess.run(
        [sys.executable, "-c", _PREPARE_METADATA, str(tmp_path)],
        cwd=PACKAGE_DIRS[name],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    dist_info = result.stdout.strip().splitlines()[-1]
    raw = (tmp_path / dist_info / "METADATA").read_text(encoding="utf-8")
    metadata = email.parser.Parser().parsestr(raw)

    assert metadata["Name"] == name
    assert metadata["Summary"] == DESCRIPTIONS[name]
    assert metadata["Keywords"] == ",".join(_project(name)["keywords"])
    assert set(metadata.get_all("Classifier", [])) == CLASSIFIERS
    assert metadata.get_all("Project-URL") == [
        f"{label}, {url}" for label, url in URLS.items()
    ]
    assert metadata["License-File"] == "LICENSE.md"
    assert "Business Source License 1.1" in metadata["License"]
    assert metadata["Description-Content-Type"] == "text/markdown"


def test_root_readme_opening_line() -> None:
    lines = [
        line
        for line in (ROOT / "README.md").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "Durable execution for AI agents and workflows."
