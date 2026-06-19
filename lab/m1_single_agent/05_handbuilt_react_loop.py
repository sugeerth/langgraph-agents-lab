"""Build the ReAct loop BY HAND so you can see exactly what it is.

CONCEPT: ReAct = Reason + Act in a CYCLE. The agent node calls the (tool-bound) model;
if the reply contains ``tool_calls`` we go run them, append the results, and loop back to
the agent; if it doesn't, we're done. The edge ``tools -> agent`` is the cycle.
aha: ReAct is literally a graph cycle, and ``.tool_calls`` on the last message is the
signal that decides whether to keep looping.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from lab.common import MessagesState, banner, calculator, get_model, print_messages
from lab.common.fake_model import tool_call


def _model():
    """Tool-bound model. Script: first turn asks for the calculator, then answers."""
    return get_model(
        script=[
            tool_call("calculator", {"a": 2, "b": 3, "op": "add"}),
            "The answer is 5.",
        ]
    ).bind_tools([calculator])


def should_continue(state: MessagesState) -> str:
    """Inspect the LAST message: tool_calls present -> go run tools, else -> END."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph():
    model = _model()

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([calculator]))
    g.add_edge(START, "agent")
    # The branch: agent -> (should_continue) -> {tools, END}
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")  # <-- this back-edge is the ReAct cycle
    return g.compile()


def run_demo() -> dict:
    graph = build_graph()
    banner("05 — a hand-built ReAct loop (the cycle is tools -> agent)")
    out = graph.invoke({"messages": [HumanMessage("add 2 and 3")]})
    print_messages(out, title="ReAct transcript")
    return out


if __name__ == "__main__":
    run_demo()
