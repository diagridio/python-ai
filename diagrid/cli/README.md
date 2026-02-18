# Diagrid CLI

The Diagrid CLI (`diagrid-cli`) is a command-line tool for managing Diagrid Catalyst resources, deploying agents, and handling infrastructure tasks.

## Installation

The CLI is installed automatically when you install the main `diagrid` package. You can also install it standalone:

```bash
pip install diagrid-cli
```

## Usage

The CLI provides several command groups for different tasks. Run `diagrid --help` to see all available commands.

### Common Commands

#### Initialization
Initialize a new Diagrid Catalyst project.

```bash
diagrid init
```

#### Deployment
Deploy your agent to a target environment.

```bash
# Deploy to the currently configured context
diagrid deploy
```

#### Infrastructure
Manage local development infrastructure using Kind (Kubernetes in Docker) and Helm.

```bash
# Check if required tools (Docker, Helm, Kind, Kubectl) are installed
diagrid infra check

# Set up a local development cluster
diagrid infra setup
```

## Configuration

The CLI manages configuration and authentication contexts.

- **Authentication:** Supports API key and device code authentication flows for connecting to Diagrid Catalyst.
- **Contexts:** Switch between different environments (e.g., local, dev, prod).
