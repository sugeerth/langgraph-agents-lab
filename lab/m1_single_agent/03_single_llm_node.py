"""Wrap ONE model call as a graph node over MessagesState.

CONCEPT: an LLM node is nothing special — it is a pure ``state -> partial-state`` function
that happens to call a model. It reads ``state["messages"]``, invokes the model, and
returns the reply wrapped so the ``add_messages`` reducer appends it to the transcript.
aha: the LLM node is just another python function; the graph doesn't know or care that
Claude is inside it.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from lab.common import MessagesState, banner, get_model, print_messages

# Offline this returns a deterministic FakeChatModel; live it returns ChatAnthropic.
MODEL = get_model(script=["Hi! How can I help?"])


def chat_node(state: MessagesState) -> dict:
    """Pure state->state: invoke the model on the running transcript, append the reply."""
    reply = MODEL.invoke(state["messages"])
    return {"messages": [reply]}  # add_messages appends this AIMessage


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)
    return g.compile()


def run_demo() -> dict:
    graph = build_graph()
    banner("03 — a single LLM node is a pure state->state function")
    out = graph.invoke({"messages": [HumanMessage("hello there")]})
    print_messages(out, title="transcript (human in, AI reply appended)")
    return out


if __name__ == "__main__":
    run_demo()
