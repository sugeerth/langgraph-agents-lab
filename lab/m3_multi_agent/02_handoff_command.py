"""Multi-agent #2 — the handoff primitive: ``Command``.

Concept: instead of static edges deciding where control goes, an agent can decide *at
runtime* by RETURNING a ``Command``. ``Command(goto="agent_b", update={...})`` does two
things in one object: it merges a state update AND names the next node to run. That is
the atomic "handoff" — agent A finishes its turn and explicitly hands the baton to B.

aha: a ``Command`` carries BOTH the state update and the next node — it is the handoff
primitive that replaces a hard-wired edge with a node-made decision.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from lab.common import MessagesState, banner, get_model, print_messages


def agent_a(state: MessagesState) -> Command[Literal["agent_b"]]:
    """Agent A does its work, then HANDS OFF to agent_b via a Command (no static edge)."""
    msg = get_model(script=["Agent A: research done — handing off to the writer."]).invoke(
        state["messages"]
    )
    # The return type annotation Command[Literal["agent_b"]] lets LangGraph draw the edge.
    return Command(goto="agent_b", update={"messages": [msg]})


def agent_b(state: MessagesState) -> dict:
    """Agent B receives control (and the appended message) and finishes."""
    msg = get_model(script=["Agent B: received the handoff, wrote the final answer."]).invoke(
        state["messages"]
    )
    return {"messages": [msg]}


def build_graph():
    """Note: there is NO ``add_edge('agent_a', 'agent_b')`` — the Command makes the jump."""
    g = StateGraph(MessagesState)
    g.add_node("agent_a", agent_a)
    g.add_node("agent_b", agent_b)
    g.add_edge(START, "agent_a")
    g.add_edge("agent_b", END)
    return g.compile()


def run_demo() -> None:
    banner("02 — explicit handoff with Command(goto=..., update=...)")
    out = build_graph().invoke({"messages": [HumanMessage("Write a memo.")]})
    print_messages(out, title="after the handoff")
    print("\nControl reached agent_b because agent_a RETURNED Command(goto='agent_b').")


if __name__ == "__main__":
    run_demo()
