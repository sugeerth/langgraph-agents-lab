"""Multi-agent #1 — two agents sharing ONE state.

Concept: the simplest "multi-agent" system is just multiple nodes wired in sequence
over a single shared ``MessagesState``. A "researcher" gathers, a "writer" composes —
each is its own LLM node with its own scripted model, but they collaborate by reading
and appending to the *same* transcript (``add_messages`` appends every contribution).

aha: "multi-agent" can be nothing fancier than multiple nodes over shared state — the
shared state IS the communication channel.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from langgraph.graph import END, START, StateGraph

from lab.common import MessagesState, banner, get_model, print_messages


def researcher(state: MessagesState) -> dict:
    """First agent: reads the user's request, contributes findings to the transcript."""
    model = get_model(script=["Researcher: I gathered 3 facts about LangGraph state."])
    return {"messages": [model.invoke(state["messages"])]}


def writer(state: MessagesState) -> dict:
    """Second agent: reads everything so far (incl. the researcher's notes) and writes."""
    model = get_model(script=["Writer: Using the research, here is the final summary."])
    return {"messages": [model.invoke(state["messages"])]}


def build_graph():
    """START -> researcher -> writer -> END, all over one shared MessagesState."""
    g = StateGraph(MessagesState)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "writer")  # the writer sees the researcher's message
    g.add_edge("writer", END)
    return g.compile()


def run_demo() -> None:
    banner("01 — two agents over ONE shared state (researcher -> writer)")
    out = build_graph().invoke({"messages": [HumanMessage("Summarize LangGraph state.")]})
    print_messages(out, title="shared transcript")
    print("\nBoth agents appended to the same MessagesState — that's the whole trick.")


if __name__ == "__main__":
    run_demo()
