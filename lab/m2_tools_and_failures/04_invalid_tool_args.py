"""04 — The model passes a bad argument type; pydantic validation catches it.

CONCEPT: every ``@tool`` has an inferred pydantic ``args_schema``. When the model supplies
an argument of the wrong type (e.g. a string where an int is required), LangGraph validates
BEFORE the tool body runs and surfaces the failure as a ``ToolMessage(status="error")``.

aha: arg validation is automatic and becomes a self-correction signal for the model.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode, create_react_agent

from lab.common import banner, bad_schema_tool, get_model, print_messages
from lab.common.fake_model import tool_call


def demo_broken():
    """Model hallucinates ``x="not-an-int"`` — validation fails before the body runs."""
    res = ToolNode([bad_schema_tool]).invoke(
        {"messages": [tool_call("bad_schema_tool", {"x": "not-an-int"})]}
    )
    return res["messages"][-1]


def demo_fixed():
    """In a full loop the model SEES the validation error and self-corrects with a real int."""
    model = get_model(
        script=[
            tool_call("bad_schema_tool", {"x": "not-an-int"}),  # turn 1: bad type -> error
            tool_call("bad_schema_tool", {"x": 42}),  # turn 2: corrected after seeing the error
            "The integer was accepted: 42.",  # turn 3: final answer
        ]
    )
    agent = create_react_agent(model, [bad_schema_tool])
    return agent.invoke({"messages": [HumanMessage("echo the integer")]})


def run_demo():
    banner("04 — invalid tool args: pydantic validation -> ToolMessage(status='error')")
    bad = demo_broken()
    print(f"BROKEN: {type(bad).__name__}(status={bad.status!r}) {bad.content!r}")
    out = demo_fixed()
    print_messages(out, title="FIXED — model retried with a valid int after the error")
    tools = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    print(f"\ntool messages: {len(tools)} (1 error, 1 success) | final: {out['messages'][-1].content!r}")


if __name__ == "__main__":
    run_demo()
