# Catalyst Agents Helm Chart

**Durable AI Agents with Diagrid Catalyst**

The `catalyst-agents` chart is a batteries-included Helm chart for developing durable, fault-tolerant AI agents using [Diagrid Catalyst](https://www.diagrid.io/catalyst). It deploys a complete local Kubernetes environment — including Dapr, observability tooling, and an LLM backend — so your agents can recover from failures, persist state across restarts, and scale effectively.

Coupled with the [Diagrid Python package](https://pypi.org/project/diagrid/) you'll have everything at hand to get started in no time.

See the [Diagrid QuickStarts repository](https://github.com/diagridio/catalyst-quickstarts) for getting started with building AI Agents with Catalyst.

Get started with [Catalyst for free](https://diagrid.ws/get-catalyst).

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Description

The `catalyst-agents` chart deploys everything needed to develop AI agents locally:

- **Dapr control plane** – sidecar injection, pub/sub, state stores, workflows, and conversation
  building blocks
- **Redis** – default state store and pub/sub broker, with optional RedisInsight UI
- **OpenTelemetry Collector** – receives traces, metrics, and logs from agents via OTLP
- **Loki + Tempo + kube-prometheus-stack (Grafana)** – full observability stack with pre-wired
  data sources
- **Diagrid Dashboard** – visual overview of agents, components, and traffic
- **Gateway** – optional ingress for local routing

## Prerequisites

| Tool | Version |
|---|---|
| kubectl | ≥ 1.28 |
| Helm | ≥ 3.17 |
| kind (or equivalent) | latest |
| Running container registry | localhost:5001 or GHCR |

## Installation

### Minimal install

```bash
helm install catalyst-agents \
  oci://ghcr.io/diagridio/charts/catalyst-agents \
  --version <VERSION> \
  --namespace catalyst-agents --create-namespace \
  --set llm.apiKey=<YOUR_KEY> \
  --set llm.googleApiKey=<YOUR_GOOGLE_KEY>
```

### Full install with overrides

```bash
helm install catalyst-agents \
  oci://ghcr.io/diagridio/charts/catalyst-agents \
  --version <VERSION> \
  --namespace catalyst-agents --create-namespace \
  --set registry=ghcr.io/myorg \
  --set llm.provider=openAI \
  --set llm.openAI.model=gpt-4o \
  --set llm.apiKey=<YOUR_OPENAI_KEY> \
  --set llm.googleApiKey=<YOUR_GOOGLE_KEY> \
  --set monitoring.enabled=true \
  --set redisInsight.enabled=true \
  --set diagridDashboard.enabled=true \
  --set gateway.enabled=true
```

### Upgrade

```bash
helm upgrade catalyst-agents \
  oci://ghcr.io/diagridio/charts/catalyst-agents \
  --version <VERSION> \
  --namespace catalyst-agents
```

### Uninstall

```bash
helm uninstall catalyst-agents --namespace catalyst-agents
```

## Values Reference

| Key | Default | Description |
|---|---|---|
| `registry` | `localhost:5001` | Container registry used to pull agent images |
| `global.logLevel` | `DEBUG` | Log level applied across all components |
| `monitoring.enabled` | `true` | Deploy the full observability stack (Loki, Tempo, Prometheus, Grafana, OTel Collector) |
| `dapr.enabled` | `true` | Deploy the Dapr control plane |
| `redis.enabled` | `true` | Deploy a Redis instance as state store and pub/sub broker |
| `redis.image` | `redis` | Redis container image |
| `redis.tag` | `6.2` | Redis image tag |
| `redis.port` | `6379` | Redis service port |
| `redis.password` | `""` | Redis password (empty = no auth) |
| `llm.provider` | `ollama` | LLM backend – `ollama` or `openAI` |
| `llm.ollama.enabled` | `true` | Enable Ollama as the LLM provider |
| `llm.ollama.model` | `llama3.2:latest` | Ollama model to use |
| `llm.ollama.endpoint` | `http://host.docker.internal:11434/v1` | Ollama API endpoint |
| `llm.openAI.model` | `gpt-4o-mini` | OpenAI model to use when `llm.provider=openAI` |
| `llm.apiKey` | `dummy-key` | API key passed to the LLM provider |
| `llm.googleApiKey` | `""` | Google API key for ADK agent |
| `opentelemetry.enabled` | `true` | Enable OpenTelemetry export from agents |
| `opentelemetry.endpoint` | `…opentelemetry-collector…:4317` | OTLP gRPC collector endpoint |
| `opentelemetry.protocol` | `grpc` | OTLP transport protocol (`grpc` or `http`) |
| `redisInsight.enabled` | `true` | Deploy RedisInsight UI (NodePort 30540) |
| `diagridDashboard.enabled` | `true` | Deploy Diagrid Dashboard UI (NodePort 30088) |
| `gateway.enabled` | `true` | Deploy the ingress gateway |

For the full set of values (including all sub-chart tunables), see
[`values.yaml`](./values.yaml).

## Dependencies

| Chart | Version | Source |
|---|---|---|
| loki | 6.6.x | https://grafana.github.io/helm-charts |
| tempo | 1.10.x | https://grafana.github.io/helm-charts |
| kube-prometheus-stack | 82.1.0 | https://prometheus-community.github.io/helm-charts |
| opentelemetry-collector | 0.145.0 | https://open-telemetry.github.io/opentelemetry-helm-charts |
| dapr | 1.17.0-rc.3 | https://dapr.github.io/helm-charts/ |
