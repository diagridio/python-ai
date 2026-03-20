# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Chaos Mesh experiment management — presets, templates, apply/delete/status."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

from diagrid.cli.infra.kubectl import apply_stdin
from diagrid.cli.utils.process import CommandError, run, run_capture


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChaosConfig:
    """Immutable configuration for a chaos experiment run."""

    intensity: str
    experiments: tuple[str, ...]
    namespace: str
    frequency: str
    duration: str
    target_app_id: str | None = None


# ---------------------------------------------------------------------------
# Intensity presets (ported from cloudgrid/tools/chaos/apply-chaos.sh)
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, str | int]] = {
    "low": {
        "pod_kill": 10,
        "network_delay": "50ms",
        "packet_loss": 5,
        "partition_duration": "30s",
        "cpu_workers": 1,
        "memory_size": "128MB",
        "http_abort": 5,
        "http_delay": "100ms",
        "frequency": "@every 10m",
        "duration": "1m",
    },
    "medium": {
        "pod_kill": 30,
        "network_delay": "200ms",
        "packet_loss": 15,
        "partition_duration": "2m",
        "cpu_workers": 2,
        "memory_size": "512MB",
        "http_abort": 20,
        "http_delay": "500ms",
        "frequency": "@every 5m",
        "duration": "2m",
    },
    "high": {
        "pod_kill": 50,
        "network_delay": "500ms",
        "packet_loss": 30,
        "partition_duration": "5m",
        "cpu_workers": 4,
        "memory_size": "1GB",
        "http_abort": 40,
        "http_delay": "2s",
        "frequency": "@every 2m",
        "duration": "3m",
    },
}

DEFAULT_EXPERIMENTS: tuple[str, ...] = ("pod", "network", "http", "stress")


# ---------------------------------------------------------------------------
# YAML templates (Chaos Mesh CRDs)
# ---------------------------------------------------------------------------

_POD_CHAOS_TEMPLATE = Template("""\
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: pod-kill-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: PodChaos
  podChaos:
    action: pod-kill
    mode: fixed-percent
    value: "${pod_kill}"
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    duration: ${duration}
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: pod-failure-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: PodChaos
  podChaos:
    action: pod-failure
    mode: one
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    duration: ${duration}
""")

_NETWORK_CHAOS_TEMPLATE = Template("""\
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: network-delay-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: NetworkChaos
  networkChaos:
    action: delay
    mode: all
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    delay:
      latency: "${network_delay}"
      jitter: "50ms"
      correlation: "50"
    duration: ${duration}
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: network-loss-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: NetworkChaos
  networkChaos:
    action: loss
    mode: all
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    loss:
      loss: "${packet_loss}"
      correlation: "50"
    duration: ${duration}
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: network-partition-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: NetworkChaos
  networkChaos:
    action: partition
    mode: all
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    direction: both
    duration: ${partition_duration}
""")

_HTTP_CHAOS_TEMPLATE = Template("""\
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: http-abort-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: HTTPChaos
  httpChaos:
    mode: all
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    target: Request
    port: 8080
    abort: true
    duration: ${duration}
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: http-delay-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: HTTPChaos
  httpChaos:
    mode: all
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    target: Request
    port: 8080
    delay: ${http_delay}
    duration: ${duration}
""")

_STRESS_CHAOS_TEMPLATE = Template("""\
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: cpu-stress-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: StressChaos
  stressChaos:
    mode: one
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    stressors:
      cpu:
        workers: ${cpu_workers}
        load: 80
    duration: ${duration}
---
apiVersion: chaos-mesh.org/v1alpha1
kind: Schedule
metadata:
  name: memory-stress-schedule
  namespace: ${namespace}
spec:
  schedule: "${frequency}"
  historyLimit: 3
  concurrencyPolicy: Forbid
  type: StressChaos
  stressChaos:
    mode: one
    selector:
      namespaces: [${target_namespace}]
${label_selector_block}\
    stressors:
      memory:
        workers: 1
        size: ${memory_size}
    duration: ${duration}
""")

_EXPERIMENT_TEMPLATES: dict[str, Template] = {
    "pod": _POD_CHAOS_TEMPLATE,
    "network": _NETWORK_CHAOS_TEMPLATE,
    "http": _HTTP_CHAOS_TEMPLATE,
    "stress": _STRESS_CHAOS_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _build_label_selector_block(target_app_id: str | None) -> str:
    """Return the YAML block for label-based pod selection."""
    if target_app_id:
        return f'      labelSelectors:\n        app: "{target_app_id}"\n'
    return ""


def _resolve_preset(config: ChaosConfig) -> dict[str, str]:
    """Merge preset defaults with config overrides into template variables."""
    preset = dict(PRESETS.get(config.intensity, PRESETS["medium"]))
    return {
        "namespace": config.namespace,
        "target_namespace": config.namespace,
        "frequency": config.frequency or str(preset["frequency"]),
        "duration": config.duration or str(preset["duration"]),
        "pod_kill": str(preset["pod_kill"]),
        "network_delay": str(preset["network_delay"]),
        "packet_loss": str(preset["packet_loss"]),
        "partition_duration": str(preset["partition_duration"]),
        "cpu_workers": str(preset["cpu_workers"]),
        "memory_size": str(preset["memory_size"]),
        "http_abort": str(preset["http_abort"]),
        "http_delay": str(preset["http_delay"]),
        "label_selector_block": _build_label_selector_block(config.target_app_id),
    }


def render_experiments(config: ChaosConfig) -> list[tuple[str, str]]:
    """Render chaos experiment YAML for the given config.

    Returns a list of ``(experiment_name, yaml_string)`` tuples.
    """
    variables = _resolve_preset(config)
    results: list[tuple[str, str]] = []
    for experiment in config.experiments:
        template = _EXPERIMENT_TEMPLATES.get(experiment)
        if template is None:
            continue
        rendered = template.safe_substitute(variables)
        results.append((experiment, rendered))
    return results


# ---------------------------------------------------------------------------
# Cluster operations
# ---------------------------------------------------------------------------


def chaos_mesh_installed(namespace: str) -> bool:
    """Check whether Chaos Mesh CRDs are present in the cluster."""
    try:
        run_capture("kubectl", "get", "crd", "podchaos.chaos-mesh.org")
        return True
    except CommandError:
        return False


def apply_chaos(config: ChaosConfig) -> list[str]:
    """Apply chaos experiments to the cluster.

    Returns a list of experiment names that were applied.
    """
    applied: list[str] = []
    for name, yaml_str in render_experiments(config):
        apply_stdin(yaml_str, namespace=config.namespace)
        applied.append(name)
    return applied


def delete_chaos(namespace: str) -> None:
    """Delete all chaos resources in *namespace*."""
    run(
        "kubectl",
        "delete",
        "schedule,podchaos,networkchaos,stresschaos,httpchaos",
        "--all",
        "-n",
        namespace,
        check=False,
    )


def get_chaos_status(namespace: str) -> str:
    """Return a human-readable summary of active chaos experiments."""
    try:
        return run_capture(
            "kubectl",
            "get",
            "schedule,podchaos,networkchaos,stresschaos,httpchaos",
            "-n",
            namespace,
        )
    except CommandError:
        return "No chaos experiments found."
