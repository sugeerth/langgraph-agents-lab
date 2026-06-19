"""01 — A tool raises, and the default ToolNode turns the exception into a message.

CONCEPT: ``create_react_agent`` runs tools through a ``ToolNode``. By default
(``handle_tool_errors=True``) the node CATCHES any exception the tool throws and feeds it
back to the model as a ``ToolMessage(status="error")`` instead of crashing the graph.

aha: errors become messages the model can react to — the loop survives a failing tool.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode, create_react_agent

from lab.common import banner, get_model, make_flaky_tool, print_messages
from lab.common.fake_model import tool_call


def _agent(default_msg: str):
    """A ReAct agent over a tool that ALWAYS fails (fail_times=999)."""
    # fail_times=999 => the tool never succeeds, so we always see the error path.
    tn = ToolNode([make_flaky_tool(fail_times=999)])  # handle_tool_errors=True by default
    model = get_model(
        script=[
            tool_call("flaky_tool", {"payload": "important"}),  # turn 1: call the tool (it raises)
            default_msg,  # turn 2: the model recovers and answers, having SEEN the error
        ],
        default=default_msg,
    )
    return create_react_agent(model, tn)


def demo_broken():
    """Without error handling a raising tool would bubble up and kill the run.

    Here there is no separate "broken" graph to crash — the point of file 03 is the
    crash. This returns the raw tool exception so the contrast is explicit: *this* is
    what would propagate if ToolNode did not catch it."""
    flaky = make_flaky_tool(fail_times=999)
    try:
        flaky.invoke({"payload": "important"})
        return None
    except Exception as exc:  # noqa: BLE001 - we want to show the raw error
        return exc


def demo_fixed():
    """The default ToolNode catches the exception and continues the loop."""
    out = _agent("The connection failed, but I logged the issue and moved on.").invoke(
        {"messages": [HumanMessage("process the payload")]}
    )
    return out


def run_demo():
    banner("01 — tool raises: default ToolNode catches it (handle_tool_errors=True)")
    print("BROKEN (raw, uncaught exception the tool throws):")
    print(f"  -> {type(demo_broken()).__name__}: {demo_broken()}")
    out = demo_fixed()
    print_messages(out, title="FIXED — the exception came back as a ToolMessage(status='error')")
    errs = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    print(f"\nerror ToolMessages: {len(errs)} | final answer: {out['messages'][-1].content!r}")


if __name__ == "__main__":
    run_demo()
