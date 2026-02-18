"""API key parsing and org ID extraction."""

from __future__ import annotations

import os
import re

import jwt

from diagrid.core.config.constants import ENV_API_KEY


class InvalidAPIKeyError(Exception):
    """Raised when an API key is invalid or cannot be parsed."""


def get_api_key(env_key: str | None = None, flag_key: str | None = None) -> str | None:
    """Get API key from flag, then env var. Returns None if not set."""
    if flag_key:
        return flag_key
    return env_key or os.environ.get(ENV_API_KEY)


def extract_org_id_from_api_key(api_key: str) -> str:
    """Extract the organization ID from a Diagrid API key's JWT claims.

    The API key is a JWT whose claims include org info in the
    `https://diagrid.io/org_<orgID>/roles` pattern.
    """
    try:
        # Decode without verification — we just need the claims
        claims = jwt.decode(api_key, options={"verify_signature": False})
    except jwt.DecodeError as exc:
        raise InvalidAPIKeyError(f"Failed to decode API key: {exc}") from exc

    org_regex = re.compile(r"^https://diagrid\.io/org_([a-zA-Z0-9-]+)/roles$")
    for key in claims:
        match = org_regex.match(key)
        if match:
            return match.group(1)

    raise InvalidAPIKeyError("No organization claim found in API key")
