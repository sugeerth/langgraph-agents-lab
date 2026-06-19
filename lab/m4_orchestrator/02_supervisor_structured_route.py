"""Structured routing: the supervisor decides the next hop via a TYPED schema, not prose.

CONCEPT: free-text routing ("I think the researcher should go next") is fragile — you have
to parse it, and the model can phrase it a hundred ways. Instead force the router to emit a
pydantic ``RouteDecision`` whose ``next`` is a ``Literal`` of the valid worker names plus the
sentinel "FINISH". The model PHYSICALLY cannot return an invalid route.

aha: a typed route schema turns routing from brittle string-matching into a parseable,
testable contract.

Offline we script the two decisions the model would make (researcher, then FINISH); live,
``with_structured_output`` constrains Claude to the same schema.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import Annotated

from lab.common import SupervisorState, banner, get_model, print_messages

WORKERS = ["researcher", "writer"]


class RouteDecision(BaseModel):
    """The supervisor's only output: who acts next, or FINISH to stop."""

    next: Literal["researcher", "writer", "FINISH"]


def make_supervisor():
    """A model constrained to emit a RouteDecision. The script mirrors a plausible plan."""
    return get_model(
        script=[RouteDecision(next="researcher"), RouteDecision(next="FINISH")]
    ).with_structured_output(RouteDecision)


def build_graph():
    router = make_supervisor()

    def supervisor(state: SupervisorState) -> dict:
        decision = router.invoke(state["messages"])  # -> a RouteDecision instance
        return {"next": decision.next}

    def route(state: SupervisorState) -> str:
        """Conditional edge: send to the chosen worker, or to END on FINISH."""
        return END if state["next"] == "FINISH" else state["next"]

    def researcher(state: SupervisorState) -> dict:
        return {"messages": [AIMessage(content="researcher: gathered the data")], "step_count": 1}

    def writer(state: SupervisorState) -> dict:
        return {"messages": [AIMessage(content="writer: produced the report")], "step_count": 1}

    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_edge(START, "supervisor")
    # The supervisor's typed decision drives the conditional edge.
    g.add_conditional_edges("supervisor", route, {"researcher": "researcher", "writer": "writer", END: END})
    # Workers always report back to the supervisor for the next routing decision.
    for w in WORKERS:
        g.add_edge(w, "supervisor")
    return g.compile()


def run_demo():
    banner("02 — structured routing (typed RouteDecision instead of fragile free text)")
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage("research and report")], "next": "", "step_count": 0})
    print_messages(result, title="transcript")
    print(f"\nfinal route = {result['next']!r} (FINISH means the loop terminated cleanly)")
    return result


if __name__ == "__main__":
    run_demo()
