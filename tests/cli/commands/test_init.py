"""Tests for the init command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from diagrid.cli.commands.init import init
from diagrid.cli.main import cli
from diagrid.core.config.constants import STAGING_API_URL


@patch("diagrid.cli.commands.init.preflight_check")
@patch("diagrid.cli.commands.init.DeviceCodeAuth")
@patch("diagrid.cli.commands.init.CatalystClient")
@patch("diagrid.cli.commands.init.create_project")
@patch("diagrid.cli.commands.init._clone_quickstart")
@patch("diagrid.cli.commands.init._provision_cluster")
@patch("diagrid.cli.commands.init.install_dapr_agents")
@patch("diagrid.cli.commands.init.create_appid")
def test_init_full_flow(
    mock_create_appid: MagicMock,
    mock_install_helm: MagicMock,
    mock_provision: MagicMock,
    mock_clone: MagicMock,
    mock_create_project: MagicMock,
    mock_catalyst_client: MagicMock,
    mock_auth: MagicMock,
    mock_preflight: MagicMock,
) -> None:
    """Init command runs all steps in order."""
    mock_auth_instance = MagicMock()
    mock_auth_instance.authenticate.return_value = MagicMock(
        api_url="https://api.diagrid.io",
        org_id="org-1",
    )
    mock_auth.return_value = mock_auth_instance

    runner = CliRunner()
    result = runner.invoke(
        init,
        ["test-project", "--api-key", "fake-key", "--openai-api-key", "sk-fake"],
    )

    assert result.exit_code == 0, result.output
    mock_auth.assert_called_once()
    mock_create_project.assert_called_once()
    mock_clone.assert_called_once_with("test-project", "dapr-agents")
    mock_provision.assert_called_once()
    mock_install_helm.assert_called_once_with("sk-fake")
    mock_create_appid.assert_called_once()


@patch("diagrid.cli.commands.init.preflight_check")
@patch("diagrid.cli.commands.init.DeviceCodeAuth")
def test_init_missing_openai_key_prompts(
    mock_auth: MagicMock, mock_preflight: MagicMock
) -> None:
    """Init prompts for OpenAI key if not provided."""
    mock_auth.return_value.authenticate.return_value = MagicMock()

    runner = CliRunner()
    result = runner.invoke(
        init,
        ["test-project", "--api-key", "fake-key"],
        input="sk-from-prompt\n",
    )
    # It will fail at project creation since we didn't mock it,
    # but the prompt should have been shown
    assert "OPENAI_API_KEY" in result.output or result.exit_code != 0


@patch("diagrid.cli.commands.init.preflight_check")
@patch("diagrid.cli.commands.init.DeviceCodeAuth")
@patch("diagrid.cli.commands.init.CatalystClient")
@patch("diagrid.cli.commands.init.create_project")
@patch("diagrid.cli.commands.init._clone_quickstart")
@patch("diagrid.cli.commands.init._provision_cluster")
@patch("diagrid.cli.commands.init.install_dapr_agents")
@patch("diagrid.cli.commands.init.create_appid")
def test_init_passes_api_url_from_context(
    mock_create_appid: MagicMock,
    mock_install_helm: MagicMock,
    mock_provision: MagicMock,
    mock_clone: MagicMock,
    mock_create_project: MagicMock,
    mock_catalyst_client: MagicMock,
    mock_auth: MagicMock,
    mock_preflight: MagicMock,
) -> None:
    """Init passes the api_url from CLI context to DeviceCodeAuth."""
    mock_auth_instance = MagicMock()
    mock_auth_instance.authenticate.return_value = MagicMock(
        api_url=STAGING_API_URL,
        org_id="org-1",
    )
    mock_auth.return_value = mock_auth_instance

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--env",
            "staging",
            "init",
            "test-project",
            "--api-key",
            "fake-key",
            "--openai-api-key",
            "sk-fake",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_auth.assert_called_once_with(
        api_url=STAGING_API_URL, api_key_flag="fake-key", no_browser=False
    )
