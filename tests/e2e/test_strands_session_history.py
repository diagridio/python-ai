"""E2E: Strands runner keeps conversation history per ``session_id``.

Runs against a real Dapr sidecar and its state store. Deterministic, no LLM
required (marked ``integration``, not ``ollama``): the model replies with how
many messages it received, so the second call proves it saw the first turn.
"""

import uuid
from typing import Any

import pytest

from tests.e2e.conftest import clear_dapr_registration


class _CountingModel:
    """Replies with the number of messages it was given."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: Any = None,
        **kw: Any,
    ) -> Any:
        self.seen.append(len(messages))
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"saw {len(messages)}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


@pytest.mark.integration
def test_second_call_sees_first_turn_via_state_store() -> None:
    """Two ``run_sync`` calls with one session_id share history; a new id does not."""
    from strands import Agent

    from diagrid.agent.core.state import DaprStateStore
    from diagrid.agent.strands import DaprWorkflowAgentRunner

    clear_dapr_registration()

    model = _CountingModel()
    agent = Agent(model=model, tools=[], callback_handler=None)  # type: ignore[arg-type]
    runner = DaprWorkflowAgentRunner(
        agent=agent,
        name="e2e-strands-history",
        state_store=DaprStateStore(store_name="statestore"),
    )
    session = f"e2e-hist-{uuid.uuid4().hex[:8]}"
    try:
        runner.start()
        first = runner.run_sync(task="hello", session_id=session, timeout=60)
        second = runner.run_sync(task="again", session_id=session, timeout=60)
        other = runner.run_sync(task="fresh", session_id=f"{session}-other", timeout=60)

        assert first is not None and first.result == "saw 1"
        # history (user + assistant) plus the new user turn
        assert second is not None and second.result == "saw 3", (
            f"second call did not see the first turn: {second}"
        )
        assert other is not None and other.result == "saw 1"
    finally:
        runner.shutdown()
        clear_dapr_registration()
