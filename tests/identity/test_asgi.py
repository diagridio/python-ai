import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from diagrid.identity import OAuthConfig, VerifiedUser
from diagrid.identity.asgi import OAuthMiddleware
from diagrid.identity.outbound import current_user_token, clear_current_token
from diagrid.identity.verifier import TokenVerificationError, VerifierNotReady


def _make_app(config=None):
    async def invoke(request: Request):
        user: VerifiedUser = request.state.user
        return JSONResponse(
            {
                "subject": user.subject,
                "tenant": user.tenant,
                "scopes": sorted(user.scopes),
                "issuer_id": user.issuer_id,
            }
        )

    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[Route("/invoke", invoke, methods=["POST"]), Route("/health", health)]
    )
    app.add_middleware(OAuthMiddleware, config=config)
    return app


def _mock_verifier(payload=None, side_effect=None):
    mock = MagicMock()
    if side_effect:
        mock.verify.side_effect = side_effect
    else:
        mock.verify.return_value = payload or {}
    return mock


class TestOAuthMiddleware:
    def test_valid_token(self):
        config = OAuthConfig(scopes=frozenset({"agent.invoke"}))
        app = _make_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        payload = {
            "sub": "alice@example.com",
            "tid": "acme-corp",
            "scp": ["agent.invoke", "admin"],
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) + 3600,
        }
        verifier = _mock_verifier(payload)

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post(
                "/invoke", headers={"X-Diagrid-User-Token": "Bearer fake.jwt.token"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["subject"] == "alice@example.com"
        assert body["tenant"] == "acme-corp"
        assert "agent.invoke" in body["scopes"]

    def test_missing_token_rejected(self):
        config = OAuthConfig(require_auth=True)
        app = _make_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        with patch.object(
            OAuthMiddleware, "_get_verifier", return_value=_mock_verifier()
        ):
            resp = client.post("/invoke")

        assert resp.status_code == 401
        assert resp.json()["error"] == "oauth.missing_token"

    def test_missing_token_allowed_when_not_required(self):
        config = OAuthConfig(require_auth=False)
        app = Starlette(routes=[Route("/health", lambda r: JSONResponse({"ok": True}))])
        app.add_middleware(OAuthMiddleware, config=config)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert resp.status_code == 200

    def test_invalid_signature_returns_401(self):
        config = OAuthConfig()
        app = _make_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        verifier = _mock_verifier(
            side_effect=TokenVerificationError("oauth.invalid_signature")
        )

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post(
                "/invoke", headers={"X-Diagrid-User-Token": "Bearer bad.token"}
            )

        assert resp.status_code == 401
        assert resp.json()["error"] == "oauth.invalid_signature"

    def test_missing_scope_returns_403(self):
        config = OAuthConfig(scopes=frozenset({"admin.write"}))
        app = _make_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        payload = {
            "sub": "bob@example.com",
            "scp": ["agent.invoke"],
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) + 3600,
        }
        verifier = _mock_verifier(payload)

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post(
                "/invoke", headers={"X-Diagrid-User-Token": "Bearer fake.jwt"}
            )

        assert resp.status_code == 403
        assert resp.json()["error"] == "oauth.missing_scope"

    def test_verifier_not_ready_returns_503(self):
        config = OAuthConfig()
        app = _make_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        verifier = _mock_verifier(side_effect=VerifierNotReady("JWKS loading"))

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post(
                "/invoke", headers={"X-Diagrid-User-Token": "Bearer fake.jwt"}
            )

        assert resp.status_code == 503

    def test_authorization_header_ignored(self):
        config = OAuthConfig(require_auth=True)
        app = _make_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        with patch.object(
            OAuthMiddleware, "_get_verifier", return_value=_mock_verifier()
        ):
            resp = client.post("/invoke", headers={"Authorization": "Bearer some.jwt"})

        assert resp.status_code == 401
        assert resp.json()["error"] == "oauth.missing_token"

    def test_outbound_contextvar_set_during_request(self):
        config = OAuthConfig()
        captured_token = []

        async def handler(request: Request):
            captured_token.append(current_user_token())
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(OAuthMiddleware, config=config)
        client = TestClient(app, raise_server_exceptions=False)

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}
        verifier = _mock_verifier(payload)

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            client.post(
                "/test", headers={"X-Diagrid-User-Token": "Bearer the.raw.token"}
            )

        assert captured_token == ["the.raw.token"]
        assert current_user_token() is None

    def test_scope_extraction_from_space_delimited_string(self):
        config = OAuthConfig(scopes=frozenset({"read"}))
        app = _make_app(config)
        client = TestClient(app, raise_server_exceptions=False)

        payload = {
            "sub": "alice",
            "scope": "read write",
            "iss": "x",
            "exp": int(time.time()) + 3600,
        }
        verifier = _mock_verifier(payload)

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert "read" in resp.json()["scopes"]
