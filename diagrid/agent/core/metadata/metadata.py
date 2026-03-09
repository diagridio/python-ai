# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import logging
import random
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Optional

from .introspection import (
    detect_framework,
    find_agent_in_stack,
)
from diagrid.agent.core.types import (
    SupportedFrameworks,
)

from dapr_agents.storage.daprstores.stateservice import (
    StateStoreService,
)
from dapr_agents.agents.configs import (
    AgentMetadataSchema,
    AgentRegistryConfig,
)

from dapr.clients.grpc._state import Concurrency, Consistency

from diagrid.agent.core.discovery import discover_components

logger = logging.getLogger(__name__)

_REGISTRY_AGENTS_KEY = "agents"


class AgentRegistryAdapter:
    @classmethod
    def create_from_stack(
        cls,
        registry: Optional[AgentRegistryConfig] = None,
        state_store_name: Optional[str] = None,
    ) -> Optional["AgentRegistryAdapter"]:
        """
        Auto-detect and create an AgentRegistryAdapter by walking the call stack.

        Args:
            registry: Optional registry configuration. If None, will attempt auto-discovery.
            state_store_name: Optional Dapr state store name resolved at runtime.

        Returns:
            AgentRegistryAdapter instance if agent found, None otherwise.
        """
        agent = find_agent_in_stack()
        if not agent:
            return None

        framework = detect_framework(agent)
        if not framework:
            return None

        return cls(
            registry=registry,
            framework=framework,
            agent=agent,
            state_store_name=state_store_name,
        )

    def __init__(
        self,
        registry: Optional[AgentRegistryConfig],
        framework: str,
        agent: Any,
        state_store_name: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._state_store_name = state_store_name

        try:
            from dapr.clients import DaprClient

            with DaprClient(http_timeout_seconds=10) as _client:
                resp = _client.get_metadata()
                self.appid = resp.application_id

            if self._registry is None:
                discovered = discover_components()
                if discovered.registry_store_name:
                    self._registry = AgentRegistryConfig(
                        store=StateStoreService(
                            store_name=discovered.registry_store_name
                        ),
                        team_name="default",
                    )
        except TimeoutError:
            logger.warning(
                "Dapr sidecar not responding; proceeding without auto-configuration."
            )

        if self._registry is None:
            return

        self.registry_state: StateStoreService = self._registry.store
        self._registry_prefix: str = "agents:"
        self._meta: Dict[str, str] = {"contentType": "application/json"}
        self._max_etag_attempts: int = 10
        self._save_options: Dict[str, Any] = {
            "concurrency": Concurrency.first_write,
            "consistency": Consistency.strong,
        }

        if not self._can_handle(framework):
            raise ValueError(f"Adapter cannot handle framework '{framework}'")

        _metadata = self._extract_metadata(agent)

        # We need to handle some null values here to avoid issues during registration
        if _metadata.agent.appid == "":
            _metadata.agent.appid = self.appid or ""

        if _metadata.registry:
            if _metadata.registry.name is None:
                _metadata.registry.name = self._registry.team_name
            if _metadata.registry.resource_name is None:
                _metadata.registry.resource_name = self.registry_state.store_name

        # Patch memory/agent statestore from runner's state store
        if self._state_store_name:
            if (
                _metadata.memory
                and _metadata.memory.short_term
                and not _metadata.memory.short_term.resource_name
            ):
                _metadata.memory.short_term.resource_name = self._state_store_name
            if _metadata.agent.metadata is None:
                _metadata.agent.metadata = {}
            if not _metadata.agent.metadata.get("state_store"):
                _metadata.agent.metadata["state_store"] = self._state_store_name

        self._register(_metadata)

    def _can_handle(self, framework: str) -> bool:
        """Check if this adapter can handle the given Agent."""

        def _normalize(s: str) -> str:
            return s.lower().replace("_", "").replace("-", "").replace(" ", "")

        for fw in SupportedFrameworks:
            if _normalize(framework) == _normalize(fw.value):
                self._framework = fw
                return True
        return False

    def _effective_team(self, team: Optional[str] = None) -> str:
        return team or (
            self._registry.team_name
            if self._registry and self._registry.team_name
            else "default"
        )

    def _team_registry_index_key(self, team: Optional[str] = None) -> str:
        return f"{self._registry_prefix}{self._effective_team(team)}:_index"

    def _agent_registry_key(self, agent_name: str, team: Optional[str] = None) -> str:
        return f"{self._registry_prefix}{self._effective_team(team)}:{agent_name}"

    def _registry_partition_key(self, team: Optional[str] = None) -> Dict[str, str]:
        meta = dict(self._meta)
        meta["partitionKey"] = f"{self._registry_prefix}{self._effective_team(team)}"
        return meta

    def _extract_metadata(self, agent: Any) -> AgentMetadataSchema:
        """Extract metadata from the given Agent."""

        try:
            schema_version = version("dapr-ext-agent_core")
        except PackageNotFoundError:
            schema_version = "edge"

        from diagrid.agent.core.metadata.mapping import (
            LangGraphMapper,
            StrandsMapper,
            CrewAIMapper,
            ADKMapper,
            OpenAIAgentsMapper,
            PydanticAIMapper,
        )

        framework_mappers = {
            SupportedFrameworks.LANGGRAPH: LangGraphMapper().map_agent_metadata,
            SupportedFrameworks.STRANDS: StrandsMapper().map_agent_metadata,
            SupportedFrameworks.CREWAI: CrewAIMapper().map_agent_metadata,
            SupportedFrameworks.ADK: ADKMapper().map_agent_metadata,
            SupportedFrameworks.OPENAI: OpenAIAgentsMapper().map_agent_metadata,
            SupportedFrameworks.PYDANTIC_AI: PydanticAIMapper().map_agent_metadata,
        }

        mapper = framework_mappers.get(self._framework)
        if not mapper:
            raise ValueError(f"Adapter cannot handle framework '{self._framework}'")

        return mapper(agent=agent, schema_version=schema_version)

    def _register(self, metadata: AgentMetadataSchema) -> None:
        """
        Register agent metadata in the team registry.

        Two-step operation:
        1. Save per-agent key (simple overwrite, no read-modify-write).
        2. Update index with ETag-protected retry loop (add agent name to list).
        """
        if not metadata.registry:
            raise ValueError("Registry metadata is required for registration")

        team = metadata.registry.name
        agent_name = metadata.name
        partition_meta = self._registry_partition_key(team)

        # Step 1: Save per-agent metadata key (contention-free overwrite)
        agent_key = self._agent_registry_key(agent_name, team)
        metadata_dict = metadata.model_dump(mode="json")
        self.registry_state.save(
            key=agent_key,
            value=metadata_dict,
            state_metadata=partition_meta,
        )

        # Step 2: Add agent name to index (ETag-protected)
        index_key = self._team_registry_index_key(team)
        self._ensure_registry_initialized(key=index_key, meta=partition_meta)

        attempts = self._max_etag_attempts
        for attempt in range(1, attempts + 1):
            try:
                current_index, etag = self.registry_state.load_with_etag(
                    key=index_key,
                    default={_REGISTRY_AGENTS_KEY: []},
                    state_metadata=partition_meta,
                )
                agents_list = current_index.get(_REGISTRY_AGENTS_KEY, [])  # type: ignore[union-attr]
                if agent_name not in agents_list:
                    agents_list.append(agent_name)
                    self.registry_state.save(
                        key=index_key,
                        value={_REGISTRY_AGENTS_KEY: agents_list},
                        etag=etag,
                        state_metadata=partition_meta,
                        state_options=self._save_options,
                    )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Conflict updating registry index (attempt %d/%d) for '%s': %s",
                    attempt,
                    attempts,
                    index_key,
                    exc,
                )
                if attempt == attempts:
                    logger.exception(
                        "Failed to update registry index after %d attempts.", attempts
                    )
                    return
                time.sleep(min(0.25 * attempt, 1.0) * (1 + random.uniform(0, 0.25)))

    def _remove_agent_entry(
        self,
        *,
        team: Optional[str],
        agent_name: str,
    ) -> None:
        """
        Delete a single agent record from the team registry.

        Two-step operation:
        1. Delete per-agent key (simple delete).
        2. Update index with ETag-protected retry loop (remove agent name from list).

        Args:
            team: Team identifier.
            agent_name: Agent name (key).
        """
        partition_meta = self._registry_partition_key(team)

        # Step 1: Delete per-agent metadata key
        agent_key = self._agent_registry_key(agent_name, team)
        try:
            self.registry_state.delete(key=agent_key, state_metadata=partition_meta)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete per-agent key '%s': %s", agent_key, exc)

        # Step 2: Remove agent name from index (ETag-protected)
        index_key = self._team_registry_index_key(team)
        attempts = self._max_etag_attempts
        for attempt in range(1, attempts + 1):
            try:
                current_index, etag = self.registry_state.load_with_etag(
                    key=index_key,
                    default={_REGISTRY_AGENTS_KEY: []},
                    state_metadata=partition_meta,
                )
                agents_list = current_index.get(_REGISTRY_AGENTS_KEY, [])  # type: ignore[union-attr]
                if agent_name not in agents_list:
                    break
                agents_list.remove(agent_name)
                self.registry_state.save(
                    key=index_key,
                    value={_REGISTRY_AGENTS_KEY: agents_list},
                    etag=etag,
                    state_metadata=partition_meta,
                    state_options=self._save_options,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Conflict updating registry index (attempt %d/%d) for '%s': %s",
                    attempt,
                    attempts,
                    index_key,
                    exc,
                )
                if attempt == attempts:
                    logger.exception(
                        "Failed to update registry index after %d attempts.", attempts
                    )
                    return
                time.sleep(min(0.25 * attempt, 1.0) * (1 + random.uniform(0, 0.25)))

    def _ensure_registry_initialized(self, *, key: str, meta: Dict[str, str]) -> None:
        """
        Ensure a registry document exists to create an ETag for concurrency control.

        Args:
            key: Registry document key.
            meta: Dapr state metadata to use for the operation.
        """
        _, etag = self.registry_state.load_with_etag(  # type: ignore[union-attr]
            key=key,
            default={},
            state_metadata=meta,
        )
        if etag is None:
            self.registry_state.save(  # type: ignore[union-attr]
                key=key,
                value={},
                etag=None,
                state_metadata=meta,
                state_options=self._save_options,
            )
