# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The LangSmith parent trace a run's node activities nest under."""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParentTrace:
    """A ``LangGraph`` run opened in LangSmith, for the activities to nest under."""

    run_id: str
    dotted_order: str
    client: Any

    @property
    def config(self) -> Dict[str, str]:
        """The keys that carry the trace to the workflow, in its config."""
        return {
            "langsmith_dotted_order": self.dotted_order,
            "langsmith_run_id": self.run_id,
        }

    def end(self) -> None:
        """Close the run. Harmless after the workflow closed it: it is idempotent."""
        try:
            self.client.update_run(self.run_id, end_time=datetime.now(timezone.utc))
            self.client.flush()
        except Exception as e:
            logger.debug(f"LangSmith parent trace close failed: {e}")


def start_parent_trace(inputs: Any, thread_id: str) -> Optional[ParentTrace]:
    """Open the parent trace when ``LANGSMITH_TRACING`` is on, else ``None``.

    A failure to open it is logged and the run goes untraced.
    """
    if os.environ.get("LANGSMITH_TRACING", "").lower() not in ("true", "1"):
        return None
    try:
        from langsmith import Client

        client = Client()
        run_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        dotted_order = f"{now.strftime('%Y%m%dT%H%M%S')}{now.microsecond:06d}Z{run_id}"
        client.create_run(
            id=run_id,
            name="LangGraph",
            run_type="chain",
            inputs=inputs,
            dotted_order=dotted_order,
            trace_id=run_id,
            extra={"metadata": {"thread_id": thread_id}},
        )
        return ParentTrace(run_id, dotted_order, client)
    except Exception as e:
        logger.debug(f"LangSmith parent trace creation failed: {e}")
        return None
