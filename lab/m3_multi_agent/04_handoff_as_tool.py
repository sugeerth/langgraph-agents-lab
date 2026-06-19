"""Multi-agent #4 — handoff-as-a-TOOL (the from-scratch "swarm").

Concept: in modules 1-3 a node decided the handoff in Python. Here we let the *model*
trigger the handoff the same way it triggers any other tool: by calling one. The handoff
tool is a normal ``@tool`` — but instead of returning a string it returns a ``Command``
that names the next agent. A ``ToolNode`` executes the tool, and the ``Command`` it
returns routes control to the destination agent. This is exactly how langgraph-swarm
works under the hood.

VERIFIED in this version: a ``@tool`` returning ``Command(goto=...)`` (WITHOUT
``Command.PARENT``) works when the ``ToolNode`` lives in the SAME graph as the target
agents — the ToolNode propagates the goto. (``Command.PARENT`` only applies when the
ToolNode is nested inside a subgraph and must escape to its parent.)

aha: handoff-as-tool-call is the from-scratch version of a "swarm" — the model itself
chooses the next agent by calling a tool.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from lab.common import MessagesState, banner, get_model, print_messages
from lab.common.fake_model import tool_call


@tool
def transfer_to_writer(
    reason: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Hand off the conversation to the writer agent. Call this when research is done.

    Returning a Command makes this tool a HANDOFF: it both records a ToolMessage (so the
    tool call is properly answered) and routes control to the 'writer' node.
    """
    return Command(
        goto="writer",
        update={
            "messages": [
                ToolMessage(
                    content=f"Transferring to writer: {reason}",
                    name="transfer_to_writer",
                    tool_call_id=tool_call_id,
                )
            ]
        },
    )


def researcher(state: MessagesState) -> dict:
    """The model decides to hand off by CALLING the transfer tool (not by Python logic)."""
    model = get_model(
        script=[tool_call("transfer_to_writer", {"reason": "research complete"}, id="h1")]
    )
    return {"messages": [model.invoke(state["messages"])]}


def writer(state: MessagesState) -> dict:
    """The destination agent: takes over after the handoff tool routes control here."""
    return {"messages": [get_model(script=["Writer: composed the final report."]).invoke(state["messages"])]}


def build_graph():
    """researcher -> tools(handoff) --Command(goto=writer)--> writer -> END."""
    handoff_tools = ToolNode([transfer_to_writer])
    g = StateGraph(MessagesState)
    g.add_node("researcher", researcher)
    g.add_node("tools", handoff_tools)
    g.add_node("writer", writer)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "tools")  # researcher's tool call is executed by the ToolNode
    g.add_edge("writer", END)
    return g.compile()


def run_demo() -> None:
    banner("04 — handoff as a tool the model calls (from-scratch swarm)")
    out = build_graph().invoke({"messages": [HumanMessage("Research then write a report.")]})
    print_messages(out, title="tool-driven handoff")
    print("\nThe model CALLED transfer_to_writer; the tool's Command routed to 'writer'.")


if __name__ == "__main__":
    run_demo()
