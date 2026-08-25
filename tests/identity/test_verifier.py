import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from diagrid.identity.verifier import (
    JWKSVerifier,
    TokenVerificationError,
    VerifierNotReady,
    _discover_from_env,
    _discover_from_metadata,
    build_verifier,
)


def _generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key


def _sign_token(private_key, payload, headers=None):
    return pyjwt.encode(
        payload,
        private_key,
        algorithm="RS256",
        headers=headers,
    )


class TestJWKSVerifier:
    def test_verify_valid_token(self):
        private_key = _generate_rsa_keypair()
        payload = {
            "sub": "alice@example.com",
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = _sign_token(private_key, payload)

        public_key = private_key.public_key()
        mock_jwk = MagicMock()
        mock_jwk.key = public_key

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        with patch.object(verifier, "_ensure_client") as mock_client:
            mock_pyjwk_client = MagicMock()
            mock_pyjwk_client.get_signing_key_from_jwt.return_value = mock_jwk
            mock_client.return_value = mock_pyjwk_client

            result = verifier.verify(token)
            assert result["sub"] == "alice@example.com"
            assert result["iss"] == "https://oidc.example.com"

    def test_verify_expired_token(self):
        private_key = _generate_rsa_keypair()
        payload = {
            "sub": "alice@example.com",
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) - 3600,
            "iat": int(time.time()) - 7200,
        }
        token = _sign_token(private_key, payload)

        public_key = private_key.public_key()
        mock_jwk = MagicMock()
        mock_jwk.key = public_key

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        with patch.object(verifier, "_ensure_client") as mock_client:
            mock_pyjwk_client = MagicMock()
            mock_pyjwk_client.get_signing_key_from_jwt.return_value = mock_jwk
            mock_client.return_value = mock_pyjwk_client

            with pytest.raises(TokenVerificationError) as exc_info:
                verifier.verify(token)
            assert exc_info.value.code == "oauth.expired"

    def test_verify_wrong_issuer(self):
        private_key = _generate_rsa_keypair()
        payload = {
            "sub": "alice@example.com",
            "iss": "https://wrong-issuer.com",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = _sign_token(private_key, payload)

        public_key = private_key.public_key()
        mock_jwk = MagicMock()
        mock_jwk.key = public_key

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        with patch.object(verifier, "_ensure_client") as mock_client:
            mock_pyjwk_client = MagicMock()
            mock_pyjwk_client.get_signing_key_from_jwt.return_value = mock_jwk
            mock_client.return_value = mock_pyjwk_client

            with pytest.raises(TokenVerificationError) as exc_info:
                verifier.verify(token)
            assert exc_info.value.code == "oauth.invalid_issuer"

    def test_verify_bad_signature(self):
        sign_key = _generate_rsa_keypair()
        wrong_key = _generate_rsa_keypair()
        payload = {
            "sub": "alice@example.com",
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = _sign_token(sign_key, payload)

        mock_jwk = MagicMock()
        mock_jwk.key = wrong_key.public_key()

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        with patch.object(verifier, "_ensure_client") as mock_client:
            mock_pyjwk_client = MagicMock()
            mock_pyjwk_client.get_signing_key_from_jwt.return_value = mock_jwk
            mock_client.return_value = mock_pyjwk_client

            with pytest.raises(TokenVerificationError) as exc_info:
                verifier.verify(token)
            assert exc_info.value.code == "oauth.invalid_signature"


class TestDiscovery:
    def test_discover_from_env(self):
        with patch.dict(
            "os.environ",
            {"DIAGRID_DP_SENTRY_ISSUER": "https://oidc.test.com/org/region"},
        ):
            coords = _discover_from_env()
            assert coords is not None
            assert coords.issuer == "https://oidc.test.com/org/region"
            assert coords.jwks_uri == "https://oidc.test.com/org/region/jwks.json"

    def test_discover_from_env_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _discover_from_env() is None

    def test_discover_from_metadata_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": "test-app",
            "identity": {
                "issuer": "https://oidc.test.com/org/region",
                "jwks_uri": "https://oidc.test.com/org/region/jwks.json",
            },
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"DAPR_HTTP_PORT": "3500"}),
            patch("diagrid.identity.verifier.httpx.get", return_value=mock_resp),
        ):
            coords = _discover_from_metadata()
            assert coords is not None
            assert coords.issuer == "https://oidc.test.com/org/region"

    def test_discover_from_metadata_no_port(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _discover_from_metadata() is None

    def test_discover_from_metadata_no_identity_block(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "test-app"}
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"DAPR_HTTP_PORT": "3500"}),
            patch("diagrid.identity.verifier.httpx.get", return_value=mock_resp),
        ):
            assert _discover_from_metadata() is None

    def test_build_verifier_explicit(self):
        with patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance
            build_verifier(
                issuer="https://oidc.example.com",
                jwks_uri="https://oidc.example.com/jwks.json",
            )
            mock_cls.assert_called_once_with(
                issuer="https://oidc.example.com",
                jwks_uri="https://oidc.example.com/jwks.json",
                audience="",
            )
            mock_instance.warm.assert_called_once()

    def test_build_verifier_no_config_raises(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "diagrid.identity.verifier._discover_from_metadata", return_value=None
            ),
        ):
            with pytest.raises(RuntimeError, match="Cannot discover"):
                build_verifier()
