"""Constants for Diagrid CLI configuration."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_API_URL = "https://api.r1.diagrid.io"
PROD_API_URL = "https://api.r1.diagrid.io"
STAGING_API_URL = "https://catalyst.staging.diagrid.dev"

DIAGRID_DIR = Path(os.environ.get("DIAGRID_HOME", Path.home() / ".diagrid"))
CREDS_PATH = DIAGRID_DIR / "creds"
CONFIG_PATH = DIAGRID_DIR / "config.json"

# Environment variable names
ENV_API_KEY = "DIAGRID_API_KEY"
ENV_API_URL = "DIAGRID_API_URL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"

# OAuth2 scopes
AUTH_SCOPE = "openid profile email offline_access"

# Token refresh buffer (5 minutes before expiry)
TOKEN_REFRESH_BUFFER_SECONDS = 300

# Kind cluster defaults
DEFAULT_KIND_CLUSTER = "catalyst-agents"
DEFAULT_NAMESPACE = "catalyst-agents"

# Scripts repo — contains kind-config.yaml and the catalyst-agents helm chart
SCRIPTS_REPO_URL = "https://github.com/diagridio/catalyst-quickstarts"
SCRIPTS_REPO_BRANCH = "main"

# Kind config
KIND_NODE_IMAGE = (
    "kindest/node:v1.32.3"
    "@sha256:b36e76b4ad37b88539ce5e07425f77b29f73a8eaaebf3f1a8bc9c764401d118c"
)
KIND_CONFIG_YAML = """\
kind: Cluster
name: {cluster_name}
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: {node_image}
  extraPortMappings:
  - containerPort: 30080
    hostPort: 8080
    protocol: TCP
  - containerPort: 30443
    hostPort: 8443
    protocol: TCP
- role: worker
  image: {node_image}
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry]
    config_path = "/etc/containerd/certs.d"
"""

# Local Docker registry for kind
KIND_REGISTRY_NAME = "kind-registry"
KIND_REGISTRY_PORT = 5001

# Registry mirrors
REGISTRY_MIRRORS = [
    ("docker.io", "https://registry-1.docker.io"),
    ("ghcr.io", "https://ghcr.io"),
    ("quay.io", "https://quay.io"),
    ("gcr.io", "https://gcr.io"),
    ("registry.k8s.io", "https://registry.k8s.io"),
]

# Helm chart — OCI registry
DEFAULT_HELM_OCI_CHART = "oci://ghcr.io/diagridio/charts/catalyst-agents"
DEFAULT_HELM_CHART_VERSION = "0.1.0"

# Quickstart repo
QUICKSTART_REPO_URL = "https://github.com/diagridio/catalyst-quickstarts"
QUICKSTART_SUBDIRS = {
    "dapr-agents": "agents/dapr-agents/durable-agent",
    "langgraph": "agents/langgraph",
    "strands": "agents/strands",
    "openai-agents": "agents/openai-agents",
    "crewai": "agents/crewai",
    "adk": "agents/adk",
}
