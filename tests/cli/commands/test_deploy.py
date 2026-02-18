"""Tests for the deploy command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from diagrid.cli.commands.deploy import deploy

_MOCK_CONN = {
    "app_id": "test-agent",
    "api_token": "diagrid://v1/fake-token",
    "http_endpoint": "https://http-prj123.cloud.r1.diagrid.io:443",
    "grpc_endpoint": "https://grpc-prj123.cloud.r1.diagrid.io:443",
}


@patch("diagrid.cli.commands.deploy.DeviceCodeAuth")
@patch("diagrid.cli.commands.deploy._get_connection_details", return_value=_MOCK_CONN)
@patch("diagrid.cli.commands.deploy.build_image", return_value="agent:latest")
@patch("diagrid.cli.commands.deploy.load_into_kind")
@patch("diagrid.cli.commands.deploy.apply_stdin")
def test_deploy_full_flow(
    mock_apply: MagicMock,
    mock_load: MagicMock,
    mock_build: MagicMock,
    mock_conn: MagicMock,
    mock_auth: MagicMock,
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
    mock_load.assert_called_once()
    mock_apply.assert_called_once()
    # Verify the manifest contains Dapr env vars
    manifest = mock_apply.call_args[0][0]
    assert "DAPR_API_TOKEN" in manifest
    assert "DAPR_HTTP_ENDPOINT" in manifest


@patch("diagrid.cli.commands.deploy.DeviceCodeAuth")
@patch("diagrid.cli.commands.deploy._get_connection_details", return_value=_MOCK_CONN)
@patch("diagrid.cli.commands.deploy.build_image", return_value="agent:latest")
@patch("diagrid.cli.commands.deploy.load_into_kind")
@patch("diagrid.cli.commands.deploy.apply_stdin")
def test_deploy_default_options(
    mock_apply: MagicMock,
    mock_load: MagicMock,
    mock_build: MagicMock,
    mock_conn: MagicMock,
    mock_auth: MagicMock,
) -> None:
    """Deploy with defaults uses agent:latest."""
    mock_auth.return_value.authenticate.return_value = MagicMock()

    runner = CliRunner()
    result = runner.invoke(deploy, ["--api-key", "fake-key"])

    assert result.exit_code == 0, result.output
    mock_build.assert_called_once_with("agent", "latest")
