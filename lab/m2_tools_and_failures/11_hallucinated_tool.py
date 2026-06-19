"""11 — The model calls a tool that doesn't exist.

CONCEPT: models sometimes invent a tool name that was never bound. A naive dispatcher would
``KeyError`` and crash. ToolNode instead returns a ``ToolMessage(status="error")`` saying the
name is invalid and listing the valid tools — so a hallucinated name is handled exactly like
any other tool error, and the model gets a chance to pick a real tool.

aha: hallucinated tool names are handled like any tool error — no special-casing needed.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode, create_react_agent

from lab.common import banner, calculator, get_model, print_messages
from lab.common.fake_model import tool_call


def demo_broken():
    """Model invents ``definitely_not_a_tool`` -> ToolNode reports an error, no crash."""
    res = ToolNode([calculator]).invoke(
        {"messages": [tool_call("definitely_not_a_tool", {})]}
    )
    return res["messages"][-1]


def demo_fixed():
    """In a loop, the model reads the 'not a valid tool' error and switches to a real tool."""
    model = get_model(
        script=[
            tool_call("definitely_not_a_tool", {}),  # turn 1: hallucinated name -> error
            tool_call("calculator", {"a": 2, "b": 3, "op": "add"}),  # turn 2: a REAL tool
            "The sum is 5.",  # turn 3: final answer
        ]
    )
    agent = create_react_agent(model, [calculator])
    return agent.invoke({"messages": [HumanMessage("add 2 and 3")]})


def run_demo():
    banner("11 — hallucinated tool name: ToolNode returns a 'not a valid tool' error message")
    bad = demo_broken()
    print(f"BROKEN: {type(bad).__name__}(status={bad.status!r}) {bad.content!r}")
    out = demo_fixed()
    print_messages(out, title="FIXED — model recovered by calling a real tool")
    print(f"\nfinal: {out['messages'][-1].content!r}")


if __name__ == "__main__":
    run_demo()
