# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the deploy command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from diagrid.cli.commands.deploy import deploy
from diagrid.cli.main import cli
from diagrid.core.config.constants import STAGING_API_URL

_MOCK_CONN = {
    "app_id": "test-agent",
    "api_token": "diagrid://v1/fake-token",
    "http_endpoint": "https://http-prj123.cloud.r1.diagrid.io:443",
    "grpc_endpoint": "https://grpc-prj123.cloud.r1.diagrid.io:443",
}


@patch("diagrid.cli.commands.deploy.rollout_restart")
@patch("diagrid.cli.commands.deploy._patch_llm_secret")
@patch(
    "diagrid.cli.commands.deploy._resolve_llm_keys",
    return_value={"OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""},
)
@patch("diagrid.cli.commands.deploy._ensure_registry_healthy")
@patch("diagrid.cli.commands.deploy.preflight_check")
@patch("diagrid.cli.commands.deploy.DeviceCodeAuth")
@patch("diagrid.cli.commands.deploy._get_connection_details", return_value=_MOCK_CONN)
@patch("diagrid.cli.commands.deploy.build_image", return_value="agent:latest")
@patch(
    "diagrid.cli.commands.deploy.push_to_registry",
    return_value="localhost:5001/agent:latest",
)
@patch("diagrid.cli.commands.deploy.apply_stdin")
def test_deploy_full_flow(
    mock_apply: MagicMock,
    mock_push: MagicMock,
    mock_build: MagicMock,
    mock_conn: MagicMock,
    mock_auth: MagicMock,
    mock_preflight: MagicMock,
    mock_registry: MagicMock,
    mock_resolve_keys: MagicMock,
    mock_patch_secret: MagicMock,
    mock_rollout: MagicMock,
) -> None:
    """Deploy command runs all steps."""
    mock_auth.return_value.authenticate.return_value = MagicMock()

    runner = CliRunner()
    result = runner.invoke(
        deploy,
        ["--api-key", "fake-key", "--image", "my-agent", "--tag", "v1"],
    )

    assert result.exit_code == 0, result.output
    mock_build.assert_called_once_with("my-agent", "v1")
    mock_push.assert_called_once()
    mock_apply.assert_called_once()
    mock_resolve_keys.assert_called_once()
    mock_patch_secret.assert_called_once()
    # Verify the manifest contains Dapr env vars
    manifest = mock_apply.call_args[0][0]
    assert "DAPR_API_TOKEN" in manifest
    assert "DAPR_HTTP_ENDPOINT" in manifest
    # Verify the manifest uses the registry-qualified image
    assert "localhost:5001/agent:latest" in manifest
    # Verify LLM secret env vars are injected
    assert "OPENAI_API_KEY" in manifest
    assert "GOOGLE_API_KEY" in manifest
    assert "llm-secret" in manifest


@patch("diagrid.cli.commands.deploy.rollout_restart")
@patch("diagrid.cli.commands.deploy._patch_llm_secret")
@patch(
    "diagrid.cli.commands.deploy._resolve_llm_keys",
    return_value={"OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""},
)
@patch("diagrid.cli.commands.deploy._ensure_registry_healthy")
@patch("diagrid.cli.commands.deploy.preflight_check")
@patch("diagrid.cli.commands.deploy.DeviceCodeAuth")
@patch("diagrid.cli.commands.deploy._get_connection_details", return_value=_MOCK_CONN)
@patch("diagrid.cli.commands.deploy.build_image", return_value="agent:latest")
@patch(
    "diagrid.cli.commands.deploy.push_to_registry",
    return_value="localhost:5001/agent:latest",
)
@patch("diagrid.cli.commands.deploy.apply_stdin")
def test_deploy_default_options(
    mock_apply: MagicMock,
    mock_push: MagicMock,
    mock_build: MagicMock,
    mock_conn: MagicMock,
    mock_auth: MagicMock,
    mock_preflight: MagicMock,
    mock_registry: MagicMock,
    mock_resolve_keys: MagicMock,
    mock_patch_secret: MagicMock,
    mock_rollout: MagicMock,
) -> None:
    """Deploy with defaults uses agent:latest."""
    mock_auth.return_value.authenticate.return_value = MagicMock()

    runner = CliRunner()
    result = runner.invoke(deploy, ["--api-key", "fake-key"])

    assert result.exit_code == 0, result.output
    mock_build.assert_called_once_with("agent", "latest")


@patch("diagrid.cli.commands.deploy.rollout_restart")
@patch("diagrid.cli.commands.deploy._patch_llm_secret")
@patch(
    "diagrid.cli.commands.deploy._resolve_llm_keys",
    return_value={"OPENAI_API_KEY": "", "GOOGLE_API_KEY": ""},
)
@patch("diagrid.cli.commands.deploy._ensure_registry_healthy")
@patch("diagrid.cli.commands.deploy.preflight_check")
@patch("diagrid.cli.commands.deploy.DeviceCodeAuth")
@patch("diagrid.cli.commands.deploy._get_connection_details", return_value=_MOCK_CONN)
@patch("diagrid.cli.commands.deploy.build_image", return_value="agent:latest")
@patch(
    "diagrid.cli.commands.deploy.push_to_registry",
    return_value="localhost:5001/agent:latest",
)
@patch("diagrid.cli.commands.deploy.apply_stdin")
def test_deploy_passes_api_url_from_context(
    mock_apply: MagicMock,
    mock_push: MagicMock,
    mock_build: MagicMock,
    mock_conn: MagicMock,
    mock_auth: MagicMock,
    mock_preflight: MagicMock,
    mock_registry: MagicMock,
    mock_resolve_keys: MagicMock,
    mock_patch_secret: MagicMock,
    mock_rollout: MagicMock,
) -> None:
    """Deploy passes the api_url from CLI context to DeviceCodeAuth."""
    mock_auth.return_value.authenticate.return_value = MagicMock()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--env", "staging", "deploy", "--api-key", "fake-key"],
    )

    assert result.exit_code == 0, result.output
    mock_auth.assert_called_once_with(
        api_url=STAGING_API_URL, api_key_flag="fake-key", no_browser=False
    )
