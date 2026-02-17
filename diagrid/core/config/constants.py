"""Constants for Diagrid CLI configuration."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_API_URL = "https://api.diagrid.io"

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
DEFAULT_KIND_CLUSTER = "dapr-agents"
DEFAULT_NAMESPACE = "dapr-agents"

# Scripts repo — contains kind-config.yaml and the local dapr-agents helm chart
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

# Helm chart — remote repo
DEFAULT_HELM_REPO_NAME = "dapr-agents"
DEFAULT_HELM_REPO_URL = "https://caspergn.github.io/dapr-agents-dev/"
DEFAULT_HELM_CHART = "dapr-agents/dapr-agents"

# Quickstart repo
QUICKSTART_REPO_URL = "https://github.com/diagridio/catalyst-quickstarts"
QUICKSTART_SUBDIR = "durable-agent/python"
