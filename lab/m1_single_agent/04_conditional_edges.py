"""Branch the graph with ``add_conditional_edges`` and a plain-python router.

CONCEPT: control flow doesn't need an LLM. A *router function* inspects the state and
returns a string key; ``add_conditional_edges(source, router, {key: dest})`` maps that key
to the next node. Here we route on whether the user's text contains a digit.
aha: control flow is just a python function that returns which edge to take.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from lab.common import MessagesState, banner, print_messages


def route(state: MessagesState) -> str:
    """Plain-python router: any digit in the last human message -> 'math', else 'chat'."""
    text = state["messages"][-1].content
    return "math" if any(ch.isdigit() for ch in text) else "chat"


def math_node(state: MessagesState) -> dict:
    return {"messages": [AIMessage("Routing to the math branch.")]}


def chat_node(state: MessagesState) -> dict:
    return {"messages": [AIMessage("Routing to the chat branch.")]}


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("math", math_node)
    g.add_node("chat", chat_node)
    # The router runs FROM the START edge; its return value selects the destination node.
    g.add_conditional_edges(START, route, {"math": "math", "chat": "chat"})
    g.add_edge("math", END)
    g.add_edge("chat", END)
    return g.compile()


def run_demo() -> dict:
    graph = build_graph()
    banner("04 — control flow is a python router returning an edge key")
    math_out = graph.invoke({"messages": [HumanMessage("what is 2 + 2?")]})
    print_messages(math_out, title="contains a digit -> math branch")
    chat_out = graph.invoke({"messages": [HumanMessage("hello friend")]})
    print_messages(chat_out, title="no digit -> chat branch")
    return {"math": math_out, "chat": chat_out}


if __name__ == "__main__":
    run_demo()
