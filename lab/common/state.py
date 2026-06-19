"""Shared graph-state schemas.

The state is the heart of a LangGraph graph: every node receives it and returns a
partial update that gets merged in. A *reducer* (declared via ``Annotated``) controls how
concurrent / repeated writes to a key are merged — without one, the last write wins.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class MessagesState(TypedDict):
    """The canonical chat state. ``add_messages`` appends (and de-dupes by id) rather
    than overwriting, so every node can contribute messages to one shared transcript."""

    messages: Annotated[list[BaseMessage], add_messages]


class SupervisorState(TypedDict):
    """Messages plus routing/bookkeeping fields used by the orchestrator module."""

    messages: Annotated[list[BaseMessage], add_messages]
    next: str  # which worker the supervisor picked (last-write-wins)
    step_count: Annotated[int, operator.add]  # incremented by nodes; reducer = sum
