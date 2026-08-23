# Catalyst Agents Helm Chart

**Durable AI Agents with Diagrid Catalyst**

The `catalyst-agents` chart is a batteries-included Helm chart for developing durable, fault-tolerant AI agents using [Diagrid Catalyst](https://www.diagrid.io/catalyst). It deploys the supporting tooling for a local Kubernetes environment — observability, chaos injection, and a local gateway — so you can trace, debug, and fault-test agents that recover from failures and persist state across restarts.

Coupled with the [Diagrid Python package](https://pypi.org/project/diagrid/) you'll have everything at hand to get started in no time.

See the [Diagrid QuickStarts repository](https://github.com/diagridio/catalyst-quickstarts) for getting started with building AI Agents with Catalyst.

Get started with [Catalyst for free](https://diagrid.ws/get-catalyst).

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Description

The `catalyst-agents` chart deploys the local supporting stack for AI agent development:

- **OpenTelemetry Collector** – receives traces, metrics, and logs from agents via OTLP
- **Loki + Tempo + kube-prometheus-stack (Grafana)** – full observability stack with pre-wired
  data sources, plus a ServiceMonitor for the collector
- **Chaos Mesh** – fault injection for agent resilience testing, with a job that installs its CRDs
- **Gateway** – an nginx reverse proxy that fronts Grafana for local routing
- **`llm-secret`** – holds the LLM API keys your agents read
- **Registry ConfigMap** – advertises the `localhost:5001` Kind registry
- **Catalyst `Project`** – created only when `catalystOperator.enabled` is set

The chart does **not** ship a Dapr control plane, Redis, RedisInsight, or a Diagrid Dashboard.

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
  --set chaos.enabled=true
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
| `registry` | `localhost:5001` | Container registry for agent images |
| `global.logLevel` | `DEBUG` | Log level for agent workloads |
| `monitoring.enabled` | `true` | Gate for the observability sub-charts (Loki, Tempo, kube-prometheus-stack, OTel Collector) and the collector ServiceMonitor |
| `chaos.enabled` | `true` | Gate for the `chaos-mesh` sub-chart and the CRD install job |
| `chaos.chartVersion` | see `values.yaml` | Chaos Mesh chart the CRD install job pulls; must match the `chaos-mesh` dependency version in `Chart.yaml` |
| `llm.provider` | `ollama` | LLM backend – `ollama` or `openAI` |
| `llm.ollama.enabled` | `true` | Enable Ollama as the LLM provider |
| `llm.ollama.model` | `llama3.2:latest` | Ollama model to use |
| `llm.ollama.endpoint` | `http://host.docker.internal:11434/v1` | Ollama API endpoint |
| `llm.openAI.model` | `gpt-4o-mini` | OpenAI model to use when `llm.provider=openAI` |
| `llm.apiKey` | `dummy-key` | Written to the `apiKey` field of the `llm-secret` secret |
| `llm.googleApiKey` | `""` | Written to the `googleApiKey` field of the `llm-secret` secret (used by the ADK agent) |
| `agents` | `{}` | Per-agent overrides |
| `gateway.enabled` | `true` | Gateway settings — note the gateway templates currently render unconditionally |
| `nameOverride` | _unset_ | Override the chart name used in generated resource names |
| `fullnameOverride` | _unset_ | Override the full name used in generated resource names |
| `catalystOperator.enabled` | _unset_ | Render the Catalyst `Project` resource (operator values are commented out by default) |
| `catalystOperator.region` | `aws-us-west` | Region for the Catalyst `Project` resource |
| `loki`, `tempo`, `kube-prometheus-stack`, `opentelemetry-collector`, `chaos-mesh` | see `values.yaml` | Value overrides passed straight to each sub-chart |

For the full set of values (including all sub-chart tunables), see
[`values.yaml`](./values.yaml).

## Dependencies

Five sub-charts, each gated by a condition:

| Chart | Source | Condition |
|---|---|---|
| loki | https://grafana.github.io/helm-charts | `monitoring.enabled` |
| tempo | https://grafana.github.io/helm-charts | `monitoring.enabled` |
| kube-prometheus-stack | https://prometheus-community.github.io/helm-charts | `monitoring.enabled` |
| opentelemetry-collector | https://open-telemetry.github.io/opentelemetry-helm-charts | `monitoring.enabled` |
| chaos-mesh | https://charts.chaos-mesh.org | `chaos.enabled` |

Pinned versions are deliberately not repeated here — Dependabot bumps them in
[`Chart.yaml`](./Chart.yaml), so a copy in this file goes stale within days.
Read `Chart.yaml` for the constraint and `Chart.lock` for what was last resolved.
