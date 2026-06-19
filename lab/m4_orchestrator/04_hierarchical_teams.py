"""Hierarchical teams: a whole TEAM is a compiled subgraph used as a single node.

CONCEPT: supervisors don't have to bottom out in plain workers — a "worker" can itself be an
entire team with its own internal supervisor and members. Because a compiled LangGraph is
just a Runnable over the state, you can drop one in as a node in a bigger graph. The top
supervisor sees the team as one opaque step.

aha: hierarchy = compiled graphs nested as nodes. The shared ``messages`` channel lets the
inner team contribute to the SAME transcript the top level reads.

Layout:
    top supervisor --> research_team (subgraph: team-lead -> searcher -> writer) --> END
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from lab.common import MessagesState, banner, print_messages


# ----------------------------------------------------------------- the research TEAM (subgraph)
def build_research_team():
    """A small self-contained team. Compiled, it becomes a Runnable we can nest."""

    def team_lead(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="[team] lead: assigning research")]}

    def searcher(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="[team] searcher: found 5 sources")]}

    def team_writer(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="[team] writer: wrote the section")]}

    team = StateGraph(MessagesState)
    team.add_node("team_lead", team_lead)
    team.add_node("searcher", searcher)
    team.add_node("team_writer", team_writer)
    team.add_edge(START, "team_lead")
    team.add_edge("team_lead", "searcher")
    team.add_edge("searcher", "team_writer")
    team.add_edge("team_writer", END)
    return team.compile()


# ----------------------------------------------------------------- the TOP-LEVEL graph
def build_graph():
    research_team = build_research_team()  # a compiled graph...

    def top_supervisor(state: MessagesState) -> dict:
        return {"messages": [AIMessage(content="[top] supervisor: delegating to research team")]}

    top = StateGraph(MessagesState)
    top.add_node("top_supervisor", top_supervisor)
    # ...nested directly as a node. It shares the `messages` channel with the parent.
    top.add_node("research_team", research_team)
    top.add_edge(START, "top_supervisor")
    top.add_edge("top_supervisor", "research_team")
    top.add_edge("research_team", END)
    return top.compile()


def run_demo():
    banner("04 — hierarchical teams (a compiled team subgraph used as one node)")
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage("produce a research section")]})
    print_messages(result, title="transcript (top level + nested team contributions)")
    team_msgs = [m for m in result["messages"] if "[team]" in str(m.content)]
    print(f"\nnested team contributed {len(team_msgs)} messages to the shared transcript")
    return result


if __name__ == "__main__":
    run_demo()
