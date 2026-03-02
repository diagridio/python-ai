"""
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import gc
import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def find_agent_in_stack() -> Optional[Any]:
    """
    Walk up the call stack to find an agent/graph object.

    Currently supports:
    - LangGraph: CompiledStateGraph or SyncPregelLoop
    - Strands: DaprSessionManager
    - CrewAI: Agent or DaprWorkflowAgentRunner
    - ADK: LlmAgent or DaprWorkflowAgentRunner
    - OpenAI: Agent or DaprWorkflowAgentRunner

    Returns:
        The agent/graph object if found, None otherwise.
    """
    # First, try to find in the stack
    for frame_info in inspect.stack():
        frame_locals = frame_info.frame.f_locals

        # Look for 'self' in frame locals
        if "self" in frame_locals:
            obj = frame_locals["self"]
            obj_type = type(obj).__name__
            obj_module = type(obj).__module__

            # LangGraph support - CompiledStateGraph
            if obj_type == "CompiledStateGraph":
                return obj

            # Strands support - DaprSessionManager
            # Use gc to find the Agent that owns this session manager (similar to LangGraph checkpointer)
            if obj_type == "DaprSessionManager":
                referrers = gc.get_referrers(obj)
                for ref in referrers:
                    ref_type = type(ref).__name__
                    ref_module = type(ref).__module__
                    # Look for Strands Agent that owns this session manager
                    if ref_type == "Agent" and "strands" in ref_module:
                        return ref
                # Don't register bare DaprSessionManager - only register when Agent exists
                return None

            # CrewAI support
            if obj_type == "Agent" and "crewai" in obj_module:
                return obj
            if obj_type == "DaprWorkflowAgentRunner" and "crewai" in obj_module:
                return getattr(obj, "_agent", None)

            # ADK support
            if obj_type == "LlmAgent" and "google.adk" in obj_module:
                return obj
            if obj_type == "DaprWorkflowAgentRunner" and "adk" in obj_module:
                return getattr(obj, "_agent", None)

            # Pydantic AI support (check before OpenAI to avoid substring match)
            if obj_type == "Agent" and "pydantic_ai" in obj_module:
                return obj
            if obj_type == "DaprWorkflowAgentRunner" and "pydantic_ai" in obj_module:
                return getattr(obj, "_agent", None)

            # OpenAI Agents support
            if obj_type == "Agent" and "agents" in obj_module:
                return obj
            if obj_type == "DaprWorkflowAgentRunner" and "openai_agents" in obj_module:
                return getattr(obj, "_agent", None)

            # If we found a checkpointer, use gc to find the graph that references it
            if obj_type == "DaprCheckpointer":
                # Use garbage collector to find objects referencing this checkpointer
                referrers = gc.get_referrers(obj)
                for ref in referrers:
                    ref_type = type(ref).__name__
                    if ref_type == "CompiledStateGraph":
                        return ref

    return None


def detect_framework(agent: Any) -> Optional[str]:
    """
    Detect the framework type from an agent object.

    Args:
        agent: The agent object to inspect.

    Returns:
        Framework name string if detected, None otherwise.
    """
    agent_type = type(agent).__name__
    agent_module = type(agent).__module__

    # LangGraph
    if agent_type == "CompiledStateGraph" or "langgraph" in agent_module:
        return "langgraph"

    # Strands - detect both Agent class and DaprSessionManager
    if agent_type == "Agent" and "strands" in agent_module:
        return "strands"
    if agent_type == "DaprSessionManager":
        return "strands"

    # CrewAI
    if agent_type == "Agent" and "crewai" in agent_module:
        return "crewai"

    # ADK
    if agent_type == "LlmAgent" and "google.adk" in agent_module:
        return "adk"

    # Pydantic AI (check before OpenAI to avoid substring match)
    if agent_type == "Agent" and "pydantic_ai" in agent_module:
        return "pydantic_ai"

    # OpenAI Agents
    if agent_type == "Agent" and "agents" in agent_module:
        return "openai"

    return None
