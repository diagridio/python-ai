import logging
import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from diagrid.identity.verifier import (
    JWKSVerifier,
    TokenVerificationError,
    _IdentityCoordinates,
    _discover_from_env,
    _discover_from_metadata,
    _discover_from_remote,
    build_verifier,
)


def _metadata_response(identity=None):
    mock_resp = MagicMock()
    body = {"id": "test-app"}
    if identity is not None:
        body["identity"] = identity
    mock_resp.json.return_value = body
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


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

    def test_verify_malformed_token(self):
        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        with patch.object(verifier, "_ensure_client") as mock_client:
            mock_pyjwk_client = MagicMock()
            mock_pyjwk_client.get_signing_key_from_jwt.side_effect = pyjwt.DecodeError(
                "Not enough segments"
            )
            mock_client.return_value = mock_pyjwk_client

            with pytest.raises(TokenVerificationError) as exc_info:
                verifier.verify("not-a-jwt")
            assert exc_info.value.code == "oauth.decode_error"


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
            patch("diagrid.identity.verifier.httpx2.get", return_value=mock_resp),
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
            patch("diagrid.identity.verifier.httpx2.get", return_value=mock_resp),
        ):
            assert _discover_from_metadata() is None

    def test_discover_from_remote_success(self):
        mock_resp = _metadata_response(
            {
                "issuer": "https://oidc.test.com/org/region",
                "jwks_uri": "https://oidc.test.com/org/region/jwks.json",
            }
        )

        with (
            patch.dict(
                "os.environ",
                {"DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443/"},
                clear=True,
            ),
            patch(
                "diagrid.identity.verifier.httpx2.get", return_value=mock_resp
            ) as mock_get,
        ):
            coords = _discover_from_remote()

        assert coords is not None
        assert coords.issuer == "https://oidc.test.com/org/region"
        assert coords.jwks_uri == "https://oidc.test.com/org/region/jwks.json"
        assert (
            mock_get.call_args.args[0] == "https://http-prj1.region:30443/v1.0/metadata"
        )

    def test_discover_from_remote_no_endpoint(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _discover_from_remote() is None

    def test_discover_from_remote_sends_api_token(self):
        mock_resp = _metadata_response({"issuer": "https://oidc.test.com/org/region"})

        with (
            patch.dict(
                "os.environ",
                {
                    "DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443",
                    "DAPR_API_TOKEN": "diagrid://v1/org/prj/token",
                },
                clear=True,
            ),
            patch(
                "diagrid.identity.verifier.httpx2.get", return_value=mock_resp
            ) as mock_get,
        ):
            assert _discover_from_remote() is not None

        assert mock_get.call_args.kwargs["headers"] == {
            "dapr-api-token": "diagrid://v1/org/prj/token"
        }

    def test_discover_from_remote_omits_api_token_when_absent(self):
        mock_resp = _metadata_response({"issuer": "https://oidc.test.com/org/region"})

        with (
            patch.dict(
                "os.environ",
                {"DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443"},
                clear=True,
            ),
            patch(
                "diagrid.identity.verifier.httpx2.get", return_value=mock_resp
            ) as mock_get,
        ):
            assert _discover_from_remote() is not None

        assert mock_get.call_args.kwargs["headers"] == {}

    def test_discover_from_remote_request_failure(self):
        with (
            patch.dict(
                "os.environ",
                {"DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443"},
                clear=True,
            ),
            patch(
                "diagrid.identity.verifier.httpx2.get",
                side_effect=Exception("connection refused"),
            ),
        ):
            assert _discover_from_remote() is None

    def test_discover_from_remote_warns_when_unreachable(self, caplog):
        with (
            patch.dict(
                "os.environ",
                {
                    "DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443",
                    "DAPR_API_TOKEN": "diagrid://v1/org/prj/token",
                },
                clear=True,
            ),
            patch(
                "diagrid.identity.verifier.httpx2.get",
                side_effect=RuntimeError("connection refused"),
            ),
            caplog.at_level(logging.WARNING, logger="diagrid.identity.verifier"),
        ):
            assert _discover_from_remote() is None

        assert "https://http-prj1.region:30443/v1.0/metadata" in caplog.text
        assert "RuntimeError: connection refused" in caplog.text
        assert "diagrid://v1/org/prj/token" not in caplog.text

    def test_discover_from_metadata_warns_when_unreachable(self, caplog):
        with (
            patch.dict("os.environ", {"DAPR_HTTP_PORT": "3500"}, clear=True),
            patch(
                "diagrid.identity.verifier.httpx2.get",
                side_effect=RuntimeError("connection refused"),
            ),
            caplog.at_level(logging.WARNING, logger="diagrid.identity.verifier"),
        ):
            assert _discover_from_metadata() is None

        assert "http://127.0.0.1:3500/v1.0/metadata" in caplog.text

    def test_discover_from_remote_silent_when_endpoint_unset(self, caplog):
        with (
            patch.dict("os.environ", {}, clear=True),
            caplog.at_level(logging.WARNING, logger="diagrid.identity.verifier"),
        ):
            assert _discover_from_remote() is None

        assert caplog.text == ""

    def test_discover_from_remote_warns_on_plaintext_token(self, caplog):
        mock_resp = _metadata_response({"issuer": "https://oidc.test.com/org/region"})

        with (
            patch.dict(
                "os.environ",
                {
                    "DAPR_HTTP_ENDPOINT": "http://localhost:3500",
                    "DAPR_API_TOKEN": "diagrid://v1/org/prj/token",
                },
                clear=True,
            ),
            patch(
                "diagrid.identity.verifier.httpx2.get", return_value=mock_resp
            ) as mock_get,
            caplog.at_level(logging.WARNING, logger="diagrid.identity.verifier"),
        ):
            assert _discover_from_remote() is not None

        # Warned, but still sent: plain http is valid for a self-hosted sidecar.
        assert "non-https" in caplog.text
        assert "diagrid://v1/org/prj/token" not in caplog.text
        assert mock_get.call_args.kwargs["headers"] == {
            "dapr-api-token": "diagrid://v1/org/prj/token"
        }

    def test_discover_from_remote_no_warning_over_https(self, caplog):
        mock_resp = _metadata_response({"issuer": "https://oidc.test.com/org/region"})

        with (
            patch.dict(
                "os.environ",
                {
                    "DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443",
                    "DAPR_API_TOKEN": "diagrid://v1/org/prj/token",
                },
                clear=True,
            ),
            patch("diagrid.identity.verifier.httpx2.get", return_value=mock_resp),
            caplog.at_level(logging.WARNING, logger="diagrid.identity.verifier"),
        ):
            assert _discover_from_remote() is not None

        assert caplog.text == ""

    def test_discover_from_remote_malformed_body(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        for body in (["not", "an", "object"], {"identity": {"issuer": 123}}):
            mock_resp.json.return_value = body
            with (
                patch.dict(
                    "os.environ",
                    {"DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443"},
                    clear=True,
                ),
                patch("diagrid.identity.verifier.httpx2.get", return_value=mock_resp),
            ):
                assert _discover_from_remote() is None

    def test_build_verifier_prefers_local_over_remote(self):
        local = _IdentityCoordinates(
            issuer="https://local.test.com",
            jwks_uri="https://local.test.com/jwks.json",
            audience="",
        )

        with (
            patch(
                "diagrid.identity.verifier._discover_from_metadata", return_value=local
            ),
            patch("diagrid.identity.verifier._discover_from_remote") as mock_remote,
            patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls,
        ):
            build_verifier()

        mock_remote.assert_not_called()
        assert mock_cls.call_args.kwargs["issuer"] == "https://local.test.com"

    def test_build_verifier_falls_back_to_remote(self):
        remote = _IdentityCoordinates(
            issuer="https://remote.test.com",
            jwks_uri="https://remote.test.com/jwks.json",
            audience="",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "diagrid.identity.verifier._discover_from_metadata", return_value=None
            ),
            patch(
                "diagrid.identity.verifier._discover_from_remote", return_value=remote
            ),
            patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls,
        ):
            build_verifier()

        assert mock_cls.call_args.kwargs["issuer"] == "https://remote.test.com"
        assert (
            mock_cls.call_args.kwargs["jwks_uri"] == "https://remote.test.com/jwks.json"
        )

    def test_build_verifier_remote_failure_falls_back_to_env(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "DAPR_HTTP_ENDPOINT": "https://http-prj1.region:30443",
                    "DIAGRID_DP_SENTRY_ISSUER": "https://oidc.env.com/org/region",
                },
                clear=True,
            ),
            patch(
                "diagrid.identity.verifier.httpx2.get",
                side_effect=Exception("connection refused"),
            ),
            patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls,
        ):
            build_verifier()

        assert mock_cls.call_args.kwargs["issuer"] == "https://oidc.env.com/org/region"

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

    def test_build_verifier_honours_discovered_jwks_uri(self):
        """A sidecar publishing a jwks_uri away from its issuer means it.

        Deriving ``issuer + /jwks.json`` ahead of the discovered value would
        point the verifier at an endpoint that need not exist, and every
        request would then fail with ``oauth.verifier_unavailable``.
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": "test-app",
            "identity": {
                "issuer": "https://sentry.acme",
                "jwks_uri": "https://keys.acme/jwks",
            },
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"DAPR_HTTP_PORT": "3500"}),
            patch("diagrid.identity.verifier.httpx2.get", return_value=mock_resp),
            patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls,
        ):
            build_verifier()
            mock_cls.assert_called_once_with(
                issuer="https://sentry.acme",
                jwks_uri="https://keys.acme/jwks",
                audience="",
            )

    def test_build_verifier_pinned_issuer_ignores_foreign_jwks_uri(self):
        """A pinned issuer is never checked against another issuer's keys.

        When the app names an issuer explicitly and the sidecar advertises a
        different one, adopting the advertised ``jwks_uri`` would let a token
        minted by that other issuer, claiming the pinned one, verify.
        """
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": "test-app",
            "identity": {
                "issuer": "https://other.acme",
                "jwks_uri": "https://keys.other.acme/jwks",
            },
        }
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"DAPR_HTTP_PORT": "3500"}),
            patch("diagrid.identity.verifier.httpx2.get", return_value=mock_resp),
            patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls,
        ):
            build_verifier(issuer="https://pinned.acme")
            mock_cls.assert_called_once_with(
                issuer="https://pinned.acme",
                jwks_uri="https://pinned.acme/jwks.json",
                audience="",
            )

    def test_build_verifier_no_config_raises(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "diagrid.identity.verifier._discover_from_metadata", return_value=None
            ),
        ):
            with pytest.raises(RuntimeError, match="Cannot discover"):
                build_verifier()
