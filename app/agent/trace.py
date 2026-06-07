"""The single trace-event schema, consumed by BOTH the live SSE panel and
Langfuse. One shape so the two observability layers never drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

EventType = Literal["tool_call", "tool_result", "decision", "synthesis", "refusal"]


@dataclass
class TraceEvent:
    step: int
    type: EventType
    tool: Optional[str] = None  # vector_search | metadata_query | web_search | None
    input: str = ""             # query/args, truncated
    summary: str = ""           # human-readable result summary
    score: Optional[float] = None  # cosine distance for vector_search, else None
    ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
