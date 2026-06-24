# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for publishing the canonical workflow_name into agent metadata."""

from dapr_agents.agents.configs import AgentMetadata, AgentMetadataSchema

from diagrid.agent.core.metadata.metadata import apply_workflow_name
from diagrid.agent.core.types import SupportedFrameworks


def _make_metadata(name: str, agent_metadata=None) -> AgentMetadataSchema:
    return AgentMetadataSchema(
        version="edge",
        name=name,
        registered_at="2026-01-01T00:00:00",
        agent=AgentMetadata(
            appid="app-1",
            type="LangGraph",
            orchestrator=False,
            metadata=agent_metadata,
        ),
    )


class TestApplyWorkflowName:
    def test_sets_workflow_name_when_metadata_is_none(self):
        meta = _make_metadata("catering-coordinator")
        assert meta.agent.metadata is None

        apply_workflow_name(meta, SupportedFrameworks.LANGGRAPH, "catering-coordinator")

        assert meta.agent.metadata == {
            "workflow_name": "dapr.langgraph.CateringCoordinator.workflow"
        }

    def test_preserves_existing_metadata_keys(self):
        meta = _make_metadata(
            "my-agent", agent_metadata={"framework": "langgraph", "state_store": "s1"}
        )

        apply_workflow_name(meta, "LangGraph", "my-agent")

        assert meta.agent.metadata["framework"] == "langgraph"
        assert meta.agent.metadata["state_store"] == "s1"
        assert meta.agent.metadata["workflow_name"] == "dapr.langgraph.MyAgent.workflow"

    def test_accepts_plain_string_framework(self):
        meta = _make_metadata("venue-scout")

        apply_workflow_name(meta, "strands", "venue-scout")

        assert (
            meta.agent.metadata["workflow_name"] == "dapr.strands.VenueScout.workflow"
        )
