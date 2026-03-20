# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Catalyst k8s operator helpers — AppID CRD management."""

from __future__ import annotations

import time

from diagrid.cli.infra.kubectl import apply_stdin
from diagrid.cli.utils.process import CommandError, run_capture


_APPID_CRD_TEMPLATE = """\
apiVersion: catalyst.diagrid.io/v1alpha1
kind: AppID
metadata:
  name: {app_id}
  namespace: {namespace}
  labels:
    catalyst.diagrid.io/project: "{project}"
spec:
  appId: {app_id}
"""


def create_appid_crd(project: str, app_id: str, namespace: str) -> None:
    """Create a Catalyst AppID CRD so the webhook can inject credentials."""
    manifest = _APPID_CRD_TEMPLATE.format(
        project=project,
        app_id=app_id,
        namespace=namespace,
    )
    apply_stdin(manifest, namespace=namespace)


def wait_for_appid_ready(
    project: str,
    app_id: str,
    namespace: str,
    timeout: int = 60,
) -> None:
    """Wait for the AppID CRD status to have an API token.

    Raises ``TimeoutError`` if the token is not available within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            token = run_capture(
                "kubectl",
                "get",
                "appid",
                app_id,
                "-n",
                namespace,
                "-o",
                "jsonpath={.status.apiToken}",
            )
            if token.strip():
                return
        except CommandError:
            pass
        time.sleep(2)
    raise TimeoutError(
        f"AppID '{app_id}' in project '{project}' did not become ready "
        f"within {timeout}s"
    )


def catalyst_operator_active(namespace: str) -> bool:
    """Check if the Catalyst operator CRDs are installed."""
    try:
        run_capture("kubectl", "get", "crd", "appids.catalyst.diagrid.io")
        return True
    except CommandError:
        return False
