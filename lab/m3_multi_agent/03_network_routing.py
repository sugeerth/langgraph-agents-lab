"""Multi-agent #3 — a network where the topology is emergent.

Concept: when EVERY node returns ``Command(goto=...)``, there is no central router and
almost no static wiring. A triage agent inspects the state and hands off to the right
specialist; each specialist does its job and goes to END. The "graph shape" is not drawn
up front — it emerges from the goto decisions each node makes at runtime.

aha: routing is distributed. The topology is emergent from each node's ``goto`` — there
is no supervisor; the agents themselves decide who runs next.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from lab.common import MessagesState, banner, get_model, print_messages


def triage(state: MessagesState) -> Command[Literal["billing", "tech"]]:
    """Look at the request and route to the specialist that should handle it."""
    text = state["messages"][-1].content.lower()
    dest = "billing" if ("refund" in text or "charge" in text or "bill" in text) else "tech"
    note = get_model(script=[f"Triage: this is a {dest} issue, routing now."]).invoke(
        state["messages"]
    )
    return Command(goto=dest, update={"messages": [note]})


def billing(state: MessagesState) -> Command:
    """Billing specialist — resolves and ends the conversation."""
    msg = get_model(script=["Billing: issued the refund. Resolved."]).invoke(state["messages"])
    return Command(goto=END, update={"messages": [msg]})


def tech(state: MessagesState) -> Command:
    """Tech specialist — resolves and ends the conversation."""
    msg = get_model(script=["Tech: applied a fix and restarted the service. Resolved."]).invoke(
        state["messages"]
    )
    return Command(goto=END, update={"messages": [msg]})


def build_graph():
    """Only START -> triage is wired statically; every other hop is a runtime goto."""
    g = StateGraph(MessagesState)
    g.add_node("triage", triage)
    g.add_node("billing", billing)
    g.add_node("tech", tech)
    g.add_edge(START, "triage")
    return g.compile()


def run_demo() -> None:
    banner("03 — network routing: each agent decides goto -> emergent topology")
    graph = build_graph()
    for request in ["I want a refund for a wrong charge.", "My app keeps crashing on launch."]:
        out = graph.invoke({"messages": [HumanMessage(request)]})
        print_messages(out, title=f"request: {request!r}")
    print("\nThe same graph routed billing vs tech purely from each node's goto.")


if __name__ == "__main__":
    run_demo()
