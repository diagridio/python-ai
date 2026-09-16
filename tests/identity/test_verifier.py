import logging
import time
from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from diagrid.identity import (
    IdentityNotConfiguredError,
    OAuthErrorCodes,
    TokenVerifier,
)
from diagrid.identity.verifier import (
    JWKSVerifier,
    TokenVerificationError,
    IdentityCoordinates,
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
            assert exc_info.value.code == OAuthErrorCodes.EXPIRED

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
            assert exc_info.value.code == OAuthErrorCodes.INVALID_ISSUER

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
            assert exc_info.value.code == OAuthErrorCodes.INVALID_SIGNATURE

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
            assert exc_info.value.code == OAuthErrorCodes.DECODE_ERROR


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
        local = IdentityCoordinates(
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
        remote = IdentityCoordinates(
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


class TestTokenVerifierProtocol:
    def test_jwks_verifier_satisfies_the_protocol(self):
        """Structural conformance: JWKSVerifier does not inherit the Protocol."""
        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )
        assert isinstance(verifier, TokenVerifier)
        assert TokenVerifier not in JWKSVerifier.__mro__

    def test_an_object_without_verify_does_not_satisfy_it(self):
        assert not isinstance(object(), TokenVerifier)


class TestJWKSTransport:
    """The key set is the whole root of trust, so plaintext is refused.

    An on-path attacker who can rewrite an http JWKS response mints tokens
    this verifier accepts.
    """

    def test_non_loopback_http_jwks_uri_refused(self):
        with pytest.raises(IdentityNotConfiguredError, match="allow_insecure_jwks"):
            build_verifier(
                issuer="http://oidc.example.com",
                jwks_uri="http://oidc.example.com/jwks.json",
            )

    def test_http_jwks_uri_discovered_from_env_refused(self):
        with patch.dict(
            "os.environ", {"DIAGRID_DP_SENTRY_ISSUER": "http://oidc.example.com"}
        ):
            with pytest.raises(IdentityNotConfiguredError, match="allow_insecure_jwks"):
                build_verifier()

    def test_loopback_http_jwks_uri_allowed(self):
        """The local sidecar publishes a loopback metadata endpoint."""
        with patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls:
            build_verifier(
                issuer="http://127.0.0.1:9000",
                jwks_uri="http://127.0.0.1:9000/jwks.json",
            )
            mock_cls.assert_called_once_with(
                issuer="http://127.0.0.1:9000",
                jwks_uri="http://127.0.0.1:9000/jwks.json",
                audience="",
            )

    def test_localhost_http_jwks_uri_allowed(self):
        with patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls:
            build_verifier(
                issuer="http://localhost:9000",
                jwks_uri="http://localhost:9000/jwks.json",
            )
            mock_cls.assert_called_once_with(
                issuer="http://localhost:9000",
                jwks_uri="http://localhost:9000/jwks.json",
                audience="",
            )

    def test_allow_insecure_jwks_opts_in(self):
        with patch("diagrid.identity.verifier.JWKSVerifier") as mock_cls:
            build_verifier(
                issuer="http://oidc.example.com",
                jwks_uri="http://oidc.example.com/jwks.json",
                allow_insecure_jwks=True,
            )
            mock_cls.assert_called_once_with(
                issuer="http://oidc.example.com",
                jwks_uri="http://oidc.example.com/jwks.json",
                audience="",
            )

    @pytest.mark.parametrize(
        "jwks_uri",
        [
            "file:///etc/diagrid/jwks.json",
            "ftp://keys.example.com/jwks.json",
            "keys.example.com/jwks.json",
        ],
    )
    def test_allow_insecure_jwks_relaxes_http_only(self, jwks_uri):
        """The opt-in widens the rule to plain http and to nothing else.

        Opting into plaintext says nothing about loading signing keys off the
        local filesystem or over some other transport, so every scheme but
        http and https stays refused however the flag is set.
        """
        for allow_insecure_jwks in (False, True):
            with pytest.raises(IdentityNotConfiguredError, match="https"):
                build_verifier(
                    issuer="https://oidc.example.com",
                    jwks_uri=jwks_uri,
                    allow_insecure_jwks=allow_insecure_jwks,
                )

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.1.attacker.example",
            "127.evil.example",
            "localhost.attacker.example",
        ],
    )
    def test_loopback_exemption_is_not_a_prefix_match(self, host):
        """A hostname that merely starts like loopback is not loopback.

        The exemption is decided by parsing the host, so an attacker-controlled
        DNS name cannot borrow it to serve a plaintext key set — the key set is
        the entire root of trust.
        """
        with pytest.raises(IdentityNotConfiguredError, match="allow_insecure_jwks"):
            build_verifier(
                issuer=f"http://{host}",
                jwks_uri=f"http://{host}/jwks.json",
            )


class TestIdentityNotConfigured:
    def test_build_verifier_raises_the_named_error(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "diagrid.identity.verifier._discover_from_metadata", return_value=None
            ),
        ):
            with pytest.raises(IdentityNotConfiguredError, match="Cannot discover"):
                build_verifier()


class TestPublicSurface:
    def test_coordinates_are_not_public_api(self):
        """``IdentityCoordinates`` is discovery's internal result type."""
        from diagrid.identity import verifier as verifier_module

        assert "IdentityCoordinates" not in verifier_module.__all__


def _verify_with_mocked_keys(verifier, token, public_key):
    """Run ``verifier.verify`` with *public_key* standing in for the JWKS."""
    mock_jwk = MagicMock()
    mock_jwk.key = public_key

    with patch.object(verifier, "_ensure_client") as mock_client:
        mock_pyjwk_client = MagicMock()
        mock_pyjwk_client.get_signing_key_from_jwt.return_value = mock_jwk
        mock_client.return_value = mock_pyjwk_client
        return verifier.verify(token)


class TestRejectedAlgorithms:
    """Only RS256 and ES256 are accepted, and a refusal is invalid_token.

    ``oauth.decode_error`` is reserved for a token that is genuinely
    unparseable; an ``alg`` outside the allowlist parses fine and is refused
    on its merits.
    """

    def test_unsecured_alg_none_token_is_invalid_token(self):
        private_key = _generate_rsa_keypair()
        token = pyjwt.encode(
            {
                "sub": "alice@example.com",
                "iss": "https://oidc.example.com",
                "exp": int(time.time()) + 3600,
            },
            key=None,
            algorithm="none",
        )

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        with pytest.raises(TokenVerificationError) as exc_info:
            _verify_with_mocked_keys(verifier, token, private_key.public_key())
        assert exc_info.value.code == OAuthErrorCodes.INVALID_TOKEN

    def test_symmetric_hs256_token_is_invalid_token(self):
        """An HS256 token signed with the public key as the HMAC secret.

        The classic confusion attack: it is refused for its algorithm, before
        any signature comparison.
        """
        private_key = _generate_rsa_keypair()
        token = pyjwt.encode(
            {
                "sub": "alice@example.com",
                "iss": "https://oidc.example.com",
                "exp": int(time.time()) + 3600,
            },
            "a-secret-the-attacker-picked-long-enough-for-sha256",
            algorithm="HS256",
        )

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        with pytest.raises(TokenVerificationError) as exc_info:
            _verify_with_mocked_keys(verifier, token, private_key.public_key())
        assert exc_info.value.code == OAuthErrorCodes.INVALID_TOKEN


class TestAudience:
    def test_wrong_audience_is_rejected(self):
        private_key = _generate_rsa_keypair()
        token = _sign_token(
            private_key,
            {
                "sub": "alice@example.com",
                "iss": "https://oidc.example.com",
                "aud": "someone-elses-service",
                "exp": int(time.time()) + 3600,
            },
        )

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com",
            jwks_uri="https://example.com/jwks.json",
            audience="this-service",
        )

        with pytest.raises(TokenVerificationError) as exc_info:
            _verify_with_mocked_keys(verifier, token, private_key.public_key())
        assert exc_info.value.code == OAuthErrorCodes.INVALID_AUDIENCE

    def test_audience_is_not_checked_when_none_is_configured(self):
        """No configured audience means the claim is not asserted on.

        Discovery leaves ``audience`` empty when the sidecar publishes none,
        and a token carrying an ``aud`` must still verify in that case.
        """
        private_key = _generate_rsa_keypair()
        token = _sign_token(
            private_key,
            {
                "sub": "alice@example.com",
                "iss": "https://oidc.example.com",
                "aud": "some-other-service",
                "exp": int(time.time()) + 3600,
            },
        )

        verifier = JWKSVerifier(
            issuer="https://oidc.example.com", jwks_uri="https://example.com/jwks.json"
        )

        payload = _verify_with_mocked_keys(verifier, token, private_key.public_key())
        assert payload["sub"] == "alice@example.com"


class TestClaimCheckOrder:
    """One order of claim checks, so a doubly-defective token has one answer.

    Required claims, then ``exp``, then ``iss``, then ``aud`` — the order
    documented on :meth:`JWKSVerifier.verify`.
    """

    _NOW = int(time.time())
    _ISSUER = "https://oidc.example.com"
    _AUDIENCE = "this-service"

    @pytest.mark.parametrize(
        ("name", "overrides", "dropped", "expected"),
        [
            (
                "expired beats a wrong issuer",
                {"iss": "https://wrong.example.com", "exp": _NOW - 3600},
                (),
                OAuthErrorCodes.EXPIRED,
            ),
            (
                "expired beats a wrong audience",
                {"aud": "someone-else", "exp": _NOW - 3600},
                (),
                OAuthErrorCodes.EXPIRED,
            ),
            (
                "a wrong issuer beats a wrong audience",
                {"iss": "https://wrong.example.com", "aud": "someone-else"},
                (),
                OAuthErrorCodes.INVALID_ISSUER,
            ),
            (
                "a missing required claim beats everything",
                {"exp": _NOW - 3600},
                ("sub",),
                OAuthErrorCodes.INVALID_TOKEN,
            ),
        ],
    )
    def test_the_first_failing_check_names_the_code(
        self, name, overrides, dropped, expected
    ):
        private_key = _generate_rsa_keypair()
        payload = {
            "sub": "alice@example.com",
            "iss": self._ISSUER,
            "aud": self._AUDIENCE,
            "exp": self._NOW + 3600,
            **overrides,
        }
        for claim in dropped:
            payload.pop(claim)
        token = _sign_token(private_key, payload)

        verifier = JWKSVerifier(
            issuer=self._ISSUER,
            jwks_uri="https://example.com/jwks.json",
            audience=self._AUDIENCE,
        )

        with pytest.raises(TokenVerificationError) as exc_info:
            _verify_with_mocked_keys(verifier, token, private_key.public_key())
        assert exc_info.value.code == expected, name
