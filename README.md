# Durable Workflows for AI agents

**Make your AI agents resilient to failure and outages**

The `diagrid` package is an extension SDK for the open-source [Dapr](https://github.com/dapr/dapr) project to build durable, fault-tolerant AI agents. It integrates seamlessly with popular agent frameworks, wrapping them in Dapr Workflows to ensure agents can recover from failures, persist state across restarts, and scale effectively.

Get started with [Diagrid Catalyst for free](https://diagrid.ws/get-catalyst).

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Features

- **Multi-Framework Support:** Native integrations for LangGraph, CrewAI, Google ADK, Strands, PydanticAI, OpenAI Agents, Claude Agent SDK, LangChain, Smolagents, LangChain Deep Agents, and HolmesGPT.
- **Durability:** Agent state is automatically persisted in the database of your choice. If your process crashes, the agent resumes from the last successful step.
- **Fault Tolerance:** Built-in retries and error handling powered by Dapr.
- **Observability:** Deep insights into agent execution, tool calls, and state transitions.

## Installation

Install the base package along with the extension for your chosen framework:

```bash
# For LangGraph
pip install "diagrid[langgraph]"

# For CrewAI
pip install "diagrid[crewai]"

# For Google ADK
pip install "diagrid[adk]"

# For Strands
pip install "diagrid[strands]"

# For Pydantic AI
pip install "diagrid[pydantic_ai]"

# For OpenAI Agents
pip install "diagrid[openai_agents]"

# For LangChain Deep Agents
pip install "diagrid[deepagents]"

# For Claude Agent SDK
pip install "diagrid[claude_agents]"

# For LangChain
pip install "diagrid[langchain]"

# For Smolagents
pip install "diagrid[smolagents]"

# For HolmesGPT (install in a dedicated environment — see note below)
pip install "diagrid[holmesgpt]"
```

> **Note:** `diagrid[holmesgpt]` is intentionally not part of `diagrid[all]`. HolmesGPT ships strict pins on `fastapi`, `uvicorn`, `cachetools`, `mcp`, and `httpx[socks]` that conflict with the looser constraints used by the other agent extras. Install it in its own environment.

## Verified identity

Catalyst signs the calling user's identity into an `X-Diagrid-User-Token` header on
every inbound request. `diagrid.identity` verifies it before your handler runs, and
puts it back on the calls your agent makes on the caller's behalf.

```bash
pip install "diagrid[identity]"
```

Two lines wire it up:

```python
from diagrid.identity import OAuthConfig
from diagrid.identity.asgi import OAuthMiddleware

app.add_middleware(OAuthMiddleware, config=OAuthConfig(scopes={"agent.invoke"}))
```

### Reading the verified caller

```python
from fastapi import Request

from diagrid.identity.asgi import verified_user


@app.post("/invoke")
async def invoke(request: Request):
    user = verified_user(request)  # VerifiedUser | None
    return {
        "subject": user.subject,
        "tenant": user.tenant,
        "scopes": list(user.scopes),
        "admin": user.has_scope("admin.write"),
    }
```

`VerifiedUser` carries `subject`, `tenant`, `scopes`, `claims` and `issuer_id`, plus
`has_scope(scope)`. `scopes` keeps set semantics but iterates in sorted order, so a
response that echoes it is stable across requests and across SDKs.

`verified_user()` returns `None` only when the request carried no token and
`require_auth=False` allowed it through. A token that *is* present is always
verified, and an invalid one never reaches the handler.

### Outbound calls on behalf of the caller

The sidecar mints the token for the *inbound* request, so an outbound call has to
carry it explicitly. Use the identity-aware client and it rides along:

```python
from diagrid.identity.http import AsyncClient

client = AsyncClient()  # an httpx2.AsyncClient that sends the caller's token
```

The token is read from the inbound request context at *send* time, not baked in at
construction, so one long-lived client is safe to share: concurrent requests each
carry their own caller's token. The header is cleared before it is set, and it only
ever travels to the origin the caller addressed — a redirect away from that origin
drops it.

For a client you cannot replace, install the same behaviour as a request hook:

```python
import httpx2

from diagrid.identity.http import attach_identity_headers_async

client = httpx2.AsyncClient(event_hooks={"request": [attach_identity_headers_async]})
```

`attach_identity_headers` is the synchronous counterpart. A call made with no inbound
user context — a cron, pub/sub or scheduled trigger — proceeds unauthenticated with
the header omitted rather than raising.

### `OAuthConfig`

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `scopes` | `FrozenSet[str]` | `frozenset()` | Scopes every caller must carry; a token short of one gets 403. |
| `issuer` | `Optional[str]` | `None` | Expected `iss`. Discovered when unset. |
| `audience` | `Optional[str]` | `None` | Expected `aud`. Discovered when unset. |
| `jwks_uri` | `Optional[str]` | `None` | JWKS endpoint. Discovered when unset. |
| `require_auth` | `bool` | `True` | Reject a request that carries no token. The default is fail-closed, and so is `OAuthConfig()`. |
| `allow_insecure_jwks` | `bool` | `False` | Opt in to a plaintext JWKS URI on a non-loopback host. The key set is the root of trust, so https is otherwise required; loopback is exempt because that is where the local sidecar serves. |

Coordinates are resolved explicit config first, then the sidecar's `/v1.0/metadata`
endpoint, then the `DIAGRID_DP_SENTRY_ISSUER` / `DIAGRID_DP_SENTRY_AUDIENCE`
environment variables.

Tokens are accepted for RS256 and ES256 only, must carry `exp`, `iss` and `sub`, are
allowed 120s of clock skew, and the key set is cached for 300s.

### Rejections

Every rejection is `{"error": "<code>"}` with `Cache-Control: no-store`. The codes
are constants on `OAuthErrorCodes`.

| Status | Code | When |
| --- | --- | --- |
| 401 | `oauth.missing_token` | No `X-Diagrid-User-Token`, and `require_auth=True`. |
| 401 | `oauth.expired` | `exp` is in the past, beyond the skew allowance. |
| 401 | `oauth.invalid_issuer` | `iss` is not the expected issuer. |
| 401 | `oauth.invalid_audience` | `aud` is not the expected audience. |
| 401 | `oauth.invalid_signature` | Signature does not verify against the key set. |
| 401 | `oauth.decode_error` | The token is malformed. |
| 401 | `oauth.invalid_token` | Any other claim validation failure. |
| 403 | `oauth.missing_scope` | Verified, but short of `OAuthConfig.scopes`. |
| 503 | `oauth.not_configured` | No identity coordinates could be resolved. |
| 503 | `oauth.verifier_unavailable` | Key material has not loaded yet. |

## Prerequisites

- **Python:** 3.11 or higher

## Examples & Quickstarts

Two paths to your first running agent:

- **Managed quickstart templates** — `diagridpy init my-project --framework langgraph` bootstraps a Catalyst project, local kind cluster, Helm chart, and framework template in one command. Templates are cloned from [`diagridio/catalyst-quickstarts`](https://github.com/diagridio/catalyst-quickstarts).
- **In-repo code samples** — clone this repo and run any framework's example directly. See [`examples/`](examples/) for the index, or jump straight to a framework: [`langgraph`](examples/langgraph/), [`crewai`](examples/crewai/), [`adk`](examples/adk/), [`strands`](examples/strands/), [`openai_agents`](examples/openai_agents/), [`claude_agents`](examples/claude_agents/), [`pydantic_ai`](examples/pydantic_ai/), [`deepagents`](examples/deepagents/), [`langchain`](examples/langchain/), [`smolagents`](examples/smolagents/), [`holmesgpt`](examples/holmesgpt/).

## Getting Started with Diagrid Catalyst

Diagrid Catalyst is a fully managed workflow engine for AI agents, built on the open-source CNCF Dapr Workflow project. It's the easiest way to test the different agentic integrations for free.

See [quickstarts](https://docs.diagrid.io/getting-started/quickstarts/ai-agents/) to get started in less than 5 minutes.

## How It Works

This SDK leverages [Dapr Workflows](https://docs.dapr.io/developing-applications/building-blocks/workflow/) to orchestrate agent execution.
1.  **Orchestration:** The agent's control loop is modeled as a workflow.
2.  **Activities:** Each tool execution or LLM call is modeled as a durable activity.
3.  **State Store:** Dapr saves the workflow state to a configured state store (e.g., Redis, CosmosDB) after every step.

Your code can run anywhere (local machine, Kubernetes, EC2, etc.) while the fully managed workflow engine takes care of the agent's execution state, making it crash-proof and resilient to any outage or failure.
