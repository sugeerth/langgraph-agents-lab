"""The SAME ReAct graph as 05, but swap the hand-written ``should_continue`` for the
prebuilt ``tools_condition``.

CONCEPT: LangGraph ships the exact router you just wrote. ``tools_condition`` reads the
last message and returns ``"tools"`` if it has tool_calls, else ``"__end__"`` — so you map
those two keys to your tools node and END.
aha: ``tools_condition`` IS the ``should_continue`` from file 05, already written for you.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from lab.common import MessagesState, banner, calculator, get_model, print_messages
from lab.common.fake_model import tool_call


def _model():
    return get_model(
        script=[
            tool_call("calculator", {"a": 2, "b": 3, "op": "add"}),
            "The answer is 5.",
        ]
    ).bind_tools([calculator])


def build_graph():
    model = _model()

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([calculator]))
    g.add_edge(START, "agent")
    # tools_condition returns "tools" or "__end__"; map both to real destinations.
    g.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    g.add_edge("tools", "agent")
    return g.compile()


def run_demo() -> dict:
    graph = build_graph()
    banner("06 — tools_condition is the should_continue you wrote in 05")
    out = graph.invoke({"messages": [HumanMessage("add 2 and 3")]})
    print_messages(out, title="ReAct transcript (prebuilt router)")
    return out


if __name__ == "__main__":
    run_demo()
