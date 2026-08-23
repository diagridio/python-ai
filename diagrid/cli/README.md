# Diagrid CLI

The Diagrid CLI (`diagridpy`) is a command-line tool for managing Diagrid Catalyst resources, deploying agents, and running chaos experiments.

## Community

Have questions, hit a bug, or want to share what you're building? Join the [Diagrid Community Discord](https://diagrid.ws/diagrid-community) to connect with the team and other users.

## Installation

The `diagridpy` executable ships with the main `diagrid` package:

```bash
pip install diagrid
```

Installing the standalone `diagrid-cli` distribution gives you the CLI library code only — it declares no console script, so it installs no `diagridpy` executable.

## Usage

The CLI provides three commands. Run `diagridpy --help` to see all available commands.

Global flags:

- `-v`, `--verbose` — show subprocess output
- `--env {prod,staging}` — target the prod or staging Diagrid API

### Commands

#### Initialization
Initialize a local agent development environment (Catalyst project, Kind cluster, Helm install).

```bash
diagridpy init
```

#### Deployment
Build and deploy your agent to the Kind cluster.

```bash
diagridpy deploy
```

#### Chaos
Manage Chaos Mesh experiments for agent resilience testing.

```bash
# Start chaos experiments against deployed agents
diagridpy chaos start

# Show active chaos experiments
diagridpy chaos status

# Stop and delete all chaos experiments
diagridpy chaos stop
```

## Configuration

The CLI manages configuration and authentication.

- **Authentication:** Supports API key and device code authentication flows for connecting to Diagrid Catalyst.
- **Environment:** The global `--env {prod,staging}` flag selects which Diagrid API the CLI talks to.
