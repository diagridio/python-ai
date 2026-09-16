import asyncio
import logging
import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from diagrid.identity import IdentityNotConfiguredError, OAuthConfig, VerifiedUser
from diagrid.identity.asgi import OAuthMiddleware, verified_user
from diagrid.identity.outbound import current_user_token
from diagrid.identity.verifier import TokenVerificationError, VerifierNotReadyError


async def _echo_user(request: Request) -> JSONResponse:
    user: VerifiedUser = request.state.diagrid_user
    return JSONResponse(
        {
            "subject": user.subject,
            "tenant": user.tenant,
            "scopes": sorted(user.scopes),
            "issuer_id": user.issuer_id,
        }
    )


def _make_app(config=None, verifier=None):
    async def health(request: Request):
        return JSONResponse({"status": "ok"})

    app = Starlette(
        routes=[
            Route("/invoke", _echo_user, methods=["POST"]),
            Route("/health", health),
        ]
    )
    app.add_middleware(OAuthMiddleware, config=config, verifier=verifier)
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
        payload = {
            "sub": "alice@example.com",
            "tid": "acme-corp",
            "scp": ["agent.invoke", "admin"],
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) + 3600,
        }
        app = _make_app(config, _mock_verifier(payload))
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/invoke", headers={"X-Diagrid-User-Token": "Bearer fake.jwt.token"}
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["subject"] == "alice@example.com"
        assert body["tenant"] == "acme-corp"
        assert "agent.invoke" in body["scopes"]

    def test_missing_token_rejected(self):
        app = _make_app(OAuthConfig(require_auth=True), _mock_verifier())
        client = TestClient(app, raise_server_exceptions=False)

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
        verifier = _mock_verifier(
            side_effect=TokenVerificationError("oauth.invalid_signature")
        )
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/invoke", headers={"X-Diagrid-User-Token": "Bearer bad.token"}
        )

        assert resp.status_code == 401
        assert resp.json()["error"] == "oauth.invalid_signature"

    def test_missing_scope_returns_403(self):
        config = OAuthConfig(scopes=frozenset({"admin.write"}))
        payload = {
            "sub": "bob@example.com",
            "scp": ["agent.invoke"],
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) + 3600,
        }
        app = _make_app(config, _mock_verifier(payload))
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/invoke", headers={"X-Diagrid-User-Token": "Bearer fake.jwt"}
        )

        assert resp.status_code == 403
        assert resp.json()["error"] == "oauth.missing_scope"

    def test_verifier_not_ready_returns_503(self):
        verifier = _mock_verifier(side_effect=VerifierNotReadyError("JWKS loading"))
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/invoke", headers={"X-Diagrid-User-Token": "Bearer fake.jwt"}
        )

        assert resp.status_code == 503

    def test_authorization_header_ignored(self):
        app = _make_app(OAuthConfig(require_auth=True), _mock_verifier())
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"Authorization": "Bearer some.jwt"})

        assert resp.status_code == 401
        assert resp.json()["error"] == "oauth.missing_token"

    def test_outbound_contextvar_set_during_request(self):
        captured_token = []

        async def handler(request: Request):
            captured_token.append(current_user_token())
            return JSONResponse({"ok": True})

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(
            OAuthMiddleware, config=OAuthConfig(), verifier=_mock_verifier(payload)
        )
        client = TestClient(app, raise_server_exceptions=False)

        client.post("/test", headers={"X-Diagrid-User-Token": "Bearer the.raw.token"})

        assert captured_token == ["the.raw.token"]
        assert current_user_token() is None

    def test_scope_extraction_from_space_delimited_string(self):
        config = OAuthConfig(scopes=frozenset({"read"}))
        payload = {
            "sub": "alice",
            "scope": "read write",
            "iss": "x",
            "exp": int(time.time()) + 3600,
        }
        app = _make_app(config, _mock_verifier(payload))
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert "read" in resp.json()["scopes"]

    def test_verified_user_on_namespaced_state_attribute(self):
        captured = []

        async def handler(request: Request):
            captured.append(request.state.diagrid_user)
            return JSONResponse({"ok": True})

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(
            OAuthMiddleware, config=OAuthConfig(), verifier=_mock_verifier(payload)
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/test", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert isinstance(captured[0], VerifiedUser)
        assert captured[0].subject == "alice"

    def test_app_state_user_left_untouched(self):
        """The middleware must never write ``request.state.user``.

        That slot belongs to the application; writing it is the collision
        this namespacing exists to remove.
        """
        state_user_set = {}

        async def handler(request: Request):
            try:
                request.state.user
                state_user_set["value"] = True
            except AttributeError:
                state_user_set["value"] = False
            return JSONResponse({"ok": True})

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(
            OAuthMiddleware, config=OAuthConfig(), verifier=_mock_verifier(payload)
        )
        client = TestClient(app, raise_server_exceptions=False)

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

        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}
        verifier = _mock_verifier(payload)

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        if oauth_outermost:
            app.add_middleware(AppAuthMiddleware)
            app.add_middleware(OAuthMiddleware, config=config, verifier=verifier)
        else:
            app.add_middleware(OAuthMiddleware, config=config, verifier=verifier)
            app.add_middleware(AppAuthMiddleware)
        client = TestClient(app, raise_server_exceptions=False)

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


class TestVerifiedUserAccessor:
    def test_returns_the_verified_caller(self):
        captured = []

        async def handler(request: Request):
            captured.append(verified_user(request))
            return JSONResponse({"ok": True})

        payload = {
            "sub": "alice",
            "scp": ["agent.invoke"],
            "iss": "x",
            "exp": int(time.time()) + 3600,
        }

        app = Starlette(routes=[Route("/test", handler, methods=["POST"])])
        app.add_middleware(
            OAuthMiddleware, config=OAuthConfig(), verifier=_mock_verifier(payload)
        )
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/test", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert isinstance(captured[0], VerifiedUser)
        assert captured[0].subject == "alice"
        assert captured[0].has_scope("agent.invoke")

    def test_returns_none_on_an_unauthenticated_request(self):
        captured = []

        async def handler(request: Request):
            captured.append(verified_user(request))
            return JSONResponse({"ok": True})

        app = Starlette(routes=[Route("/health", handler)])
        app.add_middleware(OAuthMiddleware, config=OAuthConfig(require_auth=False))
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")

        assert resp.status_code == 200
        assert captured == [None]


class TestVerifierInjection:
    """A caller can supply the verifier, on the middleware rather than config."""

    def test_an_injected_verifier_is_used(self):
        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}
        verifier = _mock_verifier(payload)

        app = Starlette(routes=[Route("/invoke", _echo_user, methods=["POST"])])
        app.add_middleware(OAuthMiddleware, config=OAuthConfig(), verifier=verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        assert resp.json()["subject"] == "alice"
        verifier.verify.assert_called_once_with("t")

    def test_an_injected_verifier_is_never_rebuilt(self):
        """Injection replaces discovery outright rather than seeding a cache."""
        payload = {"sub": "alice", "iss": "x", "exp": int(time.time()) + 3600}

        app = Starlette(routes=[Route("/invoke", _echo_user, methods=["POST"])])
        app.add_middleware(
            OAuthMiddleware, config=OAuthConfig(), verifier=_mock_verifier(payload)
        )
        client = TestClient(app, raise_server_exceptions=False)

        with patch("diagrid.identity.asgi.build_verifier") as mock_build:
            resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 200
        mock_build.assert_not_called()

    def test_no_verifier_and_no_coordinates_is_503(self):
        """The fail-closed path: a build failure rejects, it does not 500."""
        app = Starlette(routes=[Route("/invoke", _echo_user, methods=["POST"])])
        app.add_middleware(OAuthMiddleware, config=OAuthConfig())
        client = TestClient(app, raise_server_exceptions=False)

        with patch(
            "diagrid.identity.asgi.build_verifier",
            side_effect=IdentityNotConfiguredError("Cannot discover"),
        ):
            resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 503
        assert resp.json()["error"] == "oauth.not_configured"
        assert resp.headers["Cache-Control"] == "no-store"


class TestUnexpectedVerifierFailure:
    """An unexpected verifier exception still answers with the error envelope.

    The failure has to be caught in the middleware, so the tests drive a
    request through it rather than calling the verifier directly.
    """

    def test_unexpected_exception_returns_503_verifier_unavailable(self):
        verifier = _mock_verifier(side_effect=ValueError("kaboom"))
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 503
        assert resp.json() == {"error": "oauth.verifier_unavailable"}
        assert resp.headers["Cache-Control"] == "no-store"

    def test_unexpected_runtime_error_returns_503_verifier_unavailable(self):
        verifier = _mock_verifier(side_effect=RuntimeError("kaboom"))
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 503
        assert resp.json() == {"error": "oauth.verifier_unavailable"}
        assert resp.headers["Cache-Control"] == "no-store"

    def test_unexpected_exception_logged_at_warning(self, caplog):
        verifier = _mock_verifier(side_effect=ValueError("kaboom"))
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        with caplog.at_level(logging.WARNING, logger="diagrid.identity.asgi"):
            client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert any(
            "ValueError" in rec.getMessage() and "kaboom" in rec.getMessage()
            for rec in caplog.records
        )

    def test_cancellation_is_not_reported_as_a_verifier_failure(self):
        """A client disconnect has to reach the server, not become a 503."""
        verifier = _mock_verifier(side_effect=asyncio.CancelledError())
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code != 503
        assert resp.text == ""


class TestSpecificFailuresSurviveTheBroadCatch:
    """The broad catch runs after the specific ones, so codes keep their meaning.

    Ordered first, it would turn a correctly-rejected token into a 503.
    """

    def test_expired_token_still_401_expired(self):
        verifier = _mock_verifier(
            side_effect=TokenVerificationError("oauth.expired", "token has expired")
        )
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 401
        assert resp.json() == {"error": "oauth.expired"}
        assert resp.headers["Cache-Control"] == "no-store"

    def test_verifier_reported_missing_scope_still_403(self):
        verifier = _mock_verifier(
            side_effect=TokenVerificationError("oauth.missing_scope")
        )
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 403
        assert resp.json() == {"error": "oauth.missing_scope"}

    def test_config_enforced_missing_scope_still_403(self):
        config = OAuthConfig(scopes=frozenset({"admin.write"}))
        payload = {
            "sub": "bob@example.com",
            "scp": ["agent.invoke"],
            "iss": "https://oidc.example.com",
            "exp": int(time.time()) + 3600,
        }
        app = _make_app(config, _mock_verifier(payload))
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 403
        assert resp.json() == {"error": "oauth.missing_scope"}

    def test_verifier_not_ready_still_503_verifier_unavailable(self):
        verifier = _mock_verifier(side_effect=VerifierNotReadyError("JWKS loading"))
        app = _make_app(OAuthConfig(), verifier)
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/invoke", headers={"X-Diagrid-User-Token": "Bearer t"})

        assert resp.status_code == 503
        assert resp.json() == {"error": "oauth.verifier_unavailable"}
