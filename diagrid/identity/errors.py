# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Wire error codes and the failure types the identity surface raises.

Dependency-free on purpose: :mod:`diagrid.identity` re-exports everything
here, and importing that package has to work on a bare install.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "IdentityNotConfiguredError",
    "OAuthErrorCodes",
    "TokenVerificationError",
    "VerifierNotReadyError",
]


class OAuthErrorCodes:
    """The ``oauth.*`` codes returned as ``{"error": "<code>"}``.

    A frozen wire contract shared with every other Diagrid SDK.
    """

    MISSING_TOKEN: Final[str] = "oauth.missing_token"
    NOT_CONFIGURED: Final[str] = "oauth.not_configured"
    VERIFIER_UNAVAILABLE: Final[str] = "oauth.verifier_unavailable"
    EXPIRED: Final[str] = "oauth.expired"
    INVALID_ISSUER: Final[str] = "oauth.invalid_issuer"
    INVALID_AUDIENCE: Final[str] = "oauth.invalid_audience"
    INVALID_SIGNATURE: Final[str] = "oauth.invalid_signature"
    DECODE_ERROR: Final[str] = "oauth.decode_error"
    INVALID_TOKEN: Final[str] = "oauth.invalid_token"
    MISSING_SCOPE: Final[str] = "oauth.missing_scope"


class VerifierNotReadyError(Exception):
    """JWKS key material has not loaded yet."""


class TokenVerificationError(Exception):
    """Signature or claim validation failed.

    Attributes:
        code: The :class:`OAuthErrorCodes` value the middleware puts on the
            wire for this failure.
    """

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        super().__init__(message or code)


class IdentityNotConfiguredError(RuntimeError):
    """Identity coordinates could not be resolved, or resolved unusably.

    Subclasses :class:`RuntimeError`, which is what the middleware catches.
    """
