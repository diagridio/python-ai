# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The Strands direct-construction paths must produce canonical workflow names."""

from unittest import mock

import pytest

pytest.importorskip("strands")

from diagrid.agent.strands.durable_agent import DurableAgent  # noqa: E402
from diagrid.agent.strands.workflow import DaprAgentWorkflow  # noqa: E402


def test_dapr_agent_workflow_name_is_title_cased():
    wf = DaprAgentWorkflow(agent=mock.MagicMock(), name="venue-scout")
    assert wf.workflow_name == "dapr.strands.VenueScout.workflow"
    assert wf._activity_name == "dapr.strands.VenueScout.run_agent"


def test_durable_agent_name_is_title_cased():
    durable = DurableAgent(agent=mock.MagicMock(), name="catering-coordinator")
    assert durable._workflow_name == "dapr.strands.CateringCoordinator.workflow"
    assert durable._sanitized_name == "CateringCoordinator"
