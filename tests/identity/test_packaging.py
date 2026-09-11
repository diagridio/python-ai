# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Guards that ``diagrid[identity]`` declares everything the module imports.

The dev environment supplies starlette and PyJWT through ``fastapi`` and the
test group whether or not this package declares them, so comparing the extra
against the source's own imports is what catches an undeclared one.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path
from typing import Dict, Set

IDENTITY_EXTRA = "identity"
FIRST_PARTY_ROOT = "diagrid"

# PyPI distribution name -> the name ``import`` sees, where they differ.
_DISTRIBUTION_TO_IMPORT: Dict[str, str] = {"pyjwt": "jwt"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IDENTITY_PACKAGE = _REPO_ROOT / FIRST_PARTY_ROOT / "identity"


def _declared_import_names() -> Set[str]:
    """Import names provided by the ``identity`` extra's requirements."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    extras = pyproject["project"]["optional-dependencies"]
    names = set()
    for requirement in extras[IDENTITY_EXTRA]:
        # Trim at the first character that cannot appear in a name.
        distribution = requirement.split(";")[0].split("[")[0]
        for separator in ("<", ">", "=", "!", "~", " "):
            distribution = distribution.split(separator)[0]
        distribution = distribution.strip().lower().replace("-", "_")
        names.add(_DISTRIBUTION_TO_IMPORT.get(distribution, distribution))
    return names


def _third_party_import_names() -> Set[str]:
    """Top-level third-party modules imported anywhere in ``diagrid/identity``."""
    names = set()
    for source_file in sorted(_IDENTITY_PACKAGE.glob("*.py")):
        tree = ast.parse(source_file.read_text(), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return {
        name
        for name in names
        if name != FIRST_PARTY_ROOT and name not in sys.stdlib_module_names
    }


def test_identity_extra_declares_every_third_party_import():
    undeclared = _third_party_import_names() - _declared_import_names()
    assert not undeclared, (
        f"diagrid/identity imports {sorted(undeclared)}, which the "
        f"'{IDENTITY_EXTRA}' extra does not declare — a clean "
        f"'pip install diagrid[{IDENTITY_EXTRA}]' would fail to import."
    )


def test_identity_extra_declares_nothing_unused():
    unused = _declared_import_names() - _third_party_import_names()
    assert not unused, (
        f"the '{IDENTITY_EXTRA}' extra declares {sorted(unused)}, which "
        "diagrid/identity no longer imports."
    )


def test_identity_requires_pyjwt_with_its_crypto_extra():
    """Plain PyJWT ships no RS256/ES256 backend; ``verifier.py`` needs one."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    extras = pyproject["project"]["optional-dependencies"]
    pyjwt = next(
        req for req in extras[IDENTITY_EXTRA] if req.lower().startswith("pyjwt")
    )
    assert "[crypto]" in pyjwt, (
        f"the '{IDENTITY_EXTRA}' extra declares {pyjwt!r}; it must request "
        "pyjwt[crypto] so cryptography is installed rather than inherited "
        "by accident from another dependency."
    )


def test_identity_is_part_of_the_all_extra():
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    all_extra = pyproject["project"]["optional-dependencies"]["all"]
    assert any(IDENTITY_EXTRA in requirement for requirement in all_extra), (
        f"'{IDENTITY_EXTRA}' is missing from the 'all' extra"
    )
