# Diagrid Verified Identity Example

A two-route Starlette service that verifies the calling user's Catalyst token and
propagates that same caller onto an outbound call. Deliberately minimal — no agent,
no model, no workflow, no state store, just the `diagrid.identity` surface.

## How it works

The install is two lines of app code:

```python
config = OAuthConfig()
app.add_middleware(OAuthMiddleware, config=config)
```

`OAuthConfig()` is fail-closed by default: `require_auth=True`, so a request with no
token never reaches a handler. Issuer, audience and JWKS URI are left unset, so the
sidecar's `/v1.0/metadata` endpoint supplies them. No scopes are required of every
caller — the scope check is demonstrated in the handler instead.

Three things follow from those two lines:

- **Inbound verification** — `OAuthMiddleware` verifies `X-Diagrid-User-Token` on
  every request and rejects a bad one with `{"error": "<code>"}` before any handler
  runs.
- **Reading the caller** — `GET /whoami` calls `verified_user(request)`, the typed
  accessor, and returns `subject`, `tenant`, `scopes` and `hasRead`
  (`user.has_scope("read")`).
- **Outbound propagation** — `GET /downstream` makes one GET to `$DOWNSTREAM_URL`
  (default `http://localhost:8081/whoami`) through
  `diagrid.identity.http.AsyncClient`, which attaches the caller's identity headers
  at send time so the callee verifies the same user. It returns
  `{"downstream": "<the raw body>"}`, or 502 `{"error": "downstream_unreachable"}`
  if the call fails. The client is built **once**, in the app's `lifespan`, and shared
  by every request: the token is read from the request context at send time rather
  than baked in at construction, so concurrent callers each carry their own identity.
  Building one per request would throw away the connection pool for nothing.

## Run it

Install the SDK from the repo root, then the ASGI server this example runs on.
The `identity` extra ships starlette but no server, and this example is not a uv
workspace member, so its own dependencies are not resolved by the sync.

```bash
uv sync --all-packages --extra identity
uv pip install uvicorn
```

Then run the service:

```bash
cd examples/identity
python3 identity_service.py
```

It listens on `http://127.0.0.1:8080`. Run it behind a Catalyst sidecar so inbound
requests carry a real token and the middleware can discover its coordinates:

```bash
diagrid dev run -- python3 identity_service.py
```

`diagrid dev run` is what supplies the identity coordinates. A plain `dapr run`
sidecar will not do: an OSS Dapr sidecar publishes no `identity` block on
`/v1.0/metadata`, so discovery finds nothing.

Without a sidecar that publishes coordinates, the two failure modes are:

- a request carrying **no** token is 401 `{"error":"oauth.missing_token"}` — the
  token check happens before any discovery, so this works with no sidecar at all;
- a request carrying a token is 503 `{"error":"oauth.not_configured"}`, because
  there is nothing to verify it against.

To run against an issuer you name yourself instead — a local test JWKS server, say —
set the coordinates explicitly: `OAuthConfig(issuer=..., jwks_uri=...)`, adding
`allow_insecure_jwks=True` if that endpoint is plain http on a non-loopback host.

## Try it

With a valid token — this is the header the Catalyst sidecar sets for you:

```bash
curl -s http://127.0.0.1:8080/whoami -H "X-Diagrid-User-Token: Bearer $TOKEN"
{"subject":"alice@example.com","tenant":"acme","scopes":["read","write"],"hasRead":true}

curl -s http://127.0.0.1:8080/downstream -H "X-Diagrid-User-Token: Bearer $TOKEN"
{"downstream":"{\"subject\":\"alice@example.com\",...}"}
```

Every rejection is `{"error": "<code>"}` with `Cache-Control: no-store`:

```bash
# No token at all
curl -s -o /dev/null -w '%{http_code} ' http://127.0.0.1:8080/whoami \
  && curl -s http://127.0.0.1:8080/whoami
401 {"error":"oauth.missing_token"}

# Garbage token
curl -s http://127.0.0.1:8080/whoami -H "X-Diagrid-User-Token: Bearer not-a-jwt"
401 {"error":"oauth.decode_error"}

# Expired token
curl -s http://127.0.0.1:8080/whoami -H "X-Diagrid-User-Token: Bearer $EXPIRED"
401 {"error":"oauth.expired"}
```

And if the downstream service is not up:

```bash
curl -s http://127.0.0.1:8080/downstream -H "X-Diagrid-User-Token: Bearer $TOKEN"
502 {"error":"downstream_unreachable"}
```

`downstream_unreachable` is this example's own error string, not an SDK code.

## Notes

- **Unauthenticated routes.** Set `OAuthConfig(require_auth=False)` to let health and
  readiness endpoints share the app. A token that *is* present is still verified, and
  an invalid one still rejected.
- **Local development.** `OAuthConfig(allow_insecure_jwks=True)` permits a plaintext
  JWKS URI on a non-loopback host. It relaxes the rule to plain http only — a
  `file://` JWKS URI, or any other scheme, stays refused. It is a local-development
  escape hatch only: the key set is the entire root of trust, so it must not be set
  in production.
- **Where the token comes from.** The Catalyst sidecar signs the calling user's
  identity into the `X-Diagrid-User-Token` header on every inbound request. The
  middleware accepts it with or without the `Bearer ` prefix.
- **A client you cannot replace.** `AsyncClient` is the outbound path — the app never
  assembles identity headers itself. When the client is handed to you already built,
  install the same behaviour as a request hook instead:
  `httpx2.AsyncClient(event_hooks={"request": [attach_identity_headers_async]})`, or
  `attach_identity_headers` on a synchronous client. Both come from
  `diagrid.identity.http` and carry the same send-time read, header clearing and
  origin pinning the factories do.
