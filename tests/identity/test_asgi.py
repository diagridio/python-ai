import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.middleware.base import BaseHTTPMiddleware
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
        user: VerifiedUser = request.state.diagrid_user
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

    def test_verified_user_on_namespaced_state_attribute(self):
        config = OAuthConfig()
        captured = []

        async def handler(request: Request):
            captured.append(request.state.diagrid_user)
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(OAuthMiddleware, config=config)
        client = TestClient(app, raise_server_exceptions=False)

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}
        verifier = _mock_verifier(payload)

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post("/test", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert isinstance(captured[0], VerifiedUser)
        assert captured[0].subject == "alice"

    def test_app_state_user_left_untouched(self):
        """The middleware must never write ``request.state.user``.

        That slot belongs to the application; writing it is the collision
        this namespacing exists to remove.
        """
        config = OAuthConfig()
        state_user_set = {}

        async def handler(request: Request):
            try:
                request.state.user
                state_user_set["value"] = True
            except AttributeError:
                state_user_set["value"] = False
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(OAuthMiddleware, config=config)
        client = TestClient(app, raise_server_exceptions=False)

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}
        verifier = _mock_verifier(payload)

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post("/test", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert state_user_set["value"] is False

    @pytest.mark.parametrize("oauth_outermost", [True, False])
    def test_app_state_user_survives_oauth_middleware(self, oauth_outermost):
        """An app that sets ``state.user`` itself still reads its own object.

        ``request.state`` is one namespace shared by the whole stack, so if
        both middlewares wrote ``state.user`` the one running second would
        win.  Starlette makes the last-registered middleware outermost,
        hence run first, so both orders are exercised.
        """
        config = OAuthConfig()
        app_user = object()
        captured = {}

        async def handler(request: Request):
            captured["app"] = request.state.user
            captured["diagrid"] = request.state.diagrid_user
            return JSONResponse({"ok": True})

        class AppAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                request.state.user = app_user
                return await call_next(request)

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        if oauth_outermost:
            app.add_middleware(AppAuthMiddleware)
            app.add_middleware(OAuthMiddleware, config=config)
        else:
            app.add_middleware(OAuthMiddleware, config=config)
            app.add_middleware(AppAuthMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}
        verifier = _mock_verifier(payload)

        with patch.object(OAuthMiddleware, "_get_verifier", return_value=verifier):
            resp = client.post("/test", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert captured["app"] is app_user
        assert captured["diagrid"].subject == "alice"

    def test_no_verified_user_when_auth_not_required(self):
        """An unauthenticated request must not populate the namespaced slot.

        Downstream code distinguishes "no verified caller" by the attribute
        being absent, so a ``None`` sentinel here would break ``hasattr``.
        """
        config = OAuthConfig(require_auth=False)
        diagrid_user_set = {}

        async def handler(request: Request):
            try:
                request.state.diagrid_user
                diagrid_user_set["value"] = True
            except AttributeError:
                diagrid_user_set["value"] = False
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/health", handler)])
        app.add_middleware(OAuthMiddleware, config=config)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")

        assert resp.status_code == 200
        assert diagrid_user_set["value"] is False
