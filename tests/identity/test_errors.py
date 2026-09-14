from diagrid.identity import (
    IdentityNotConfiguredError,
    OAuthErrorCodes,
    TokenVerificationError,
    VerifierNotReadyError,
)


class TestOAuthErrorCodes:
    """The ten codes are a frozen wire contract; pin every literal."""

    def test_codes_are_the_exact_wire_literals(self):
        assert OAuthErrorCodes.MISSING_TOKEN == "oauth.missing_token"
        assert OAuthErrorCodes.NOT_CONFIGURED == "oauth.not_configured"
        assert OAuthErrorCodes.VERIFIER_UNAVAILABLE == "oauth.verifier_unavailable"
        assert OAuthErrorCodes.EXPIRED == "oauth.expired"
        assert OAuthErrorCodes.INVALID_ISSUER == "oauth.invalid_issuer"
        assert OAuthErrorCodes.INVALID_AUDIENCE == "oauth.invalid_audience"
        assert OAuthErrorCodes.INVALID_SIGNATURE == "oauth.invalid_signature"
        assert OAuthErrorCodes.DECODE_ERROR == "oauth.decode_error"
        assert OAuthErrorCodes.INVALID_TOKEN == "oauth.invalid_token"
        assert OAuthErrorCodes.MISSING_SCOPE == "oauth.missing_scope"

    def test_code_compares_equal_to_a_plain_string(self):
        """``exc.code == OAuthErrorCodes.EXPIRED`` has to keep working."""
        exc = TokenVerificationError(OAuthErrorCodes.EXPIRED)
        assert exc.code == "oauth.expired"


class TestIdentityNotConfiguredError:
    def test_is_a_runtime_error(self):
        """``except RuntimeError`` call sites predate the named type."""
        assert issubclass(IdentityNotConfiguredError, RuntimeError)


class TestVerifierNotReadyError:
    def test_carries_its_message(self):
        assert str(VerifierNotReadyError("JWKS loading")) == "JWKS loading"
