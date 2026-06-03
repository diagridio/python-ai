# Diagrid Examples

Two ways to get started with Diagrid Catalyst — pick whichever fits your goal.

## Which should I use?

- **I want to run a full Catalyst-managed agent end-to-end** → use a **managed quickstart template** below.
- **I want to read the source, run locally, or test fault tolerance** → use an **in-repo code sample** below.

---

## Managed quickstart templates

Bootstrap a complete project (Catalyst project, AppID, local kind cluster, Helm chart, framework template) in one command. Cloned from [`diagridio/catalyst-quickstarts`](https://github.com/diagridio/catalyst-quickstarts).

```bash
diagridpy init my-project --framework <framework>
```

| Framework        | Command                                                                                |
|------------------|----------------------------------------------------------------------------------------|
| `dapr-agents`    | `diagridpy init my-project --framework dapr-agents` (default)                          |
| `langgraph`      | `diagridpy init my-project --framework langgraph`                                      |
| `crewai`         | `diagridpy init my-project --framework crewai`                                         |
| `adk`            | `diagridpy init my-project --framework adk`                                            |
| `strands`        | `diagridpy init my-project --framework strands`                                        |
| `openai-agents`  | `diagridpy init my-project --framework openai-agents`                                  |
| `pydantic-ai`    | `diagridpy init my-project --framework pydantic-ai`                                    |
| `deepagents`     | `diagridpy init my-project --framework deepagents`                                     |
| `orchestrator`   | `diagridpy init my-project --framework orchestrator` — multi-agent template (crewai + openai-agents + adk working together) |

See the [`diagridpy init` docs](../README.md#diagridpy-init) for what each step does (auth, project creation, cluster provisioning, Helm install).

---

## In-repo code samples

Clone this repo and run any of these directly. Each subdirectory has its own README with prerequisites and `dapr run` commands.

| Framework | Directory | What's inside |
|-----------|-----------|---------------|
| LangGraph | [`langgraph/`](langgraph/) — [README](langgraph/README.md) | Simple graph, conditional routing, ReAct agent, crash recovery, retry |
| CrewAI | [`crewai/`](crewai/) — [README](crewai/README.md) | Simple agent, crash recovery, retry |
| Google ADK | [`adk/`](adk/) — [README](adk/README.md) | Simple agent, crash recovery, retry |
| Strands | [`strands/`](strands/) — [README](strands/README.md) | Simple agent, crash recovery, retry |
| OpenAI Agents | [`openai_agents/`](openai_agents/) — [README](openai_agents/README.md) | Simple agent, crash recovery |
| Pydantic AI | [`pydantic_ai/`](pydantic_ai/) | Simple agent, subagent workflows, incident-response demo, crash recovery, retry |
| Deep Agents | [`deepagents/`](deepagents/) | Simple agent, crash recovery, retry |
| HolmesGPT | [`holmesgpt/`](holmesgpt/) | CLI runner, SSE server, schedule + park + resume phases, ask test |

## Shared prerequisites

Most examples need:

- **Dapr CLI** initialized: `dapr init` ([install](https://docs.dapr.io/getting-started/install-dapr-cli/))
- **Redis** on `localhost:6379` (handled by `dapr init`, or `docker run -d --name redis -p 6379:6379 redis:latest`)
- **An LLM API key** in your env — `OPENAI_API_KEY`, `GOOGLE_API_KEY` (ADK), or `ANTHROPIC_API_KEY` (HolmesGPT) depending on the example
- **Dependencies** installed from the repo root: `uv sync --all-packages --extra <framework>`

The per-framework READMEs spell out exactly which are required.
