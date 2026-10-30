"""Checks that the repository's community-health files are present and valid."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_DIR = REPO_ROOT / ".github"
ADVISORY_URL = "https://github.com/diagridio/python-ai/security/advisories/new"
OWNING_TEAM = "@diagridio/diagrid-engineering-ai"


def test_codeowners_assigns_every_path_to_the_owning_team() -> None:
    codeowners = GITHUB_DIR / "CODEOWNERS"
    assert codeowners.is_file()
    rules = [
        line.split()
        for line in codeowners.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert ["*", OWNING_TEAM] in rules, "a catch-all '*' rule must name the team"
    for pattern, *owners in rules:
        assert OWNING_TEAM in owners, f"{pattern} does not name {OWNING_TEAM}"


def test_license_stays_busl() -> None:
    license_file = REPO_ROOT / "LICENSE.md"
    assert license_file.is_file()
    assert "Business Source License 1.1" in license_file.read_text(encoding="utf-8")
    assert not (REPO_ROOT / "LICENSE").exists(), "the license lives in LICENSE.md"


def test_security_policy_points_at_private_reporting() -> None:
    policy = REPO_ROOT / "SECURITY.md"
    assert policy.is_file()
    assert ADVISORY_URL in policy.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["bug_report.yml", "feature_request.yml"])
def test_issue_forms_are_valid(name: str) -> None:
    yaml = pytest.importorskip("yaml")
    form = yaml.safe_load(
        (GITHUB_DIR / "ISSUE_TEMPLATE" / name).read_text(encoding="utf-8")
    )
    assert form["name"]
    assert form["description"]
    body = form["body"]
    assert body, "an issue form needs at least one body element"
    ids = [item["id"] for item in body if item["type"] != "markdown"]
    assert len(ids) == len(set(ids)), "issue form element ids must be unique"
    assert any(item.get("validations", {}).get("required") for item in body), (
        "an issue form should require at least one field"
    )


def test_issue_template_config_routes_security_reports_privately() -> None:
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(
        (GITHUB_DIR / "ISSUE_TEMPLATE" / "config.yml").read_text(encoding="utf-8")
    )
    urls = [link["url"] for link in config["contact_links"]]
    assert ADVISORY_URL in urls


def test_pull_request_template_present() -> None:
    template = GITHUB_DIR / "pull_request_template.md"
    assert template.is_file()
    assert "How it was tested" in template.read_text(encoding="utf-8")
