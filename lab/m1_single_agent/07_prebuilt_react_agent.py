"""The whole ReAct loop in ONE call: ``create_react_agent``.

CONCEPT: everything you assembled by hand in 05 (agent node + tools node + the
should_continue branch + the back-edge cycle) is exactly what ``create_react_agent`` wires
up internally. You give it a model, a list of tools, and an optional prompt.
aha: the prebuilt agent == files 05 and 06 collapsed into a single function call.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from lab.common import banner, calculator, get_model, print_messages
from lab.common.fake_model import tool_call


def build_graph():
    model = get_model(
        script=[
            tool_call("calculator", {"a": 21, "b": 21, "op": "add"}),
            "It is 42.",
        ]
    )
    return create_react_agent(model, [calculator], prompt="You are a helpful assistant.")


def run_demo() -> dict:
    agent = build_graph()
    banner("07 — create_react_agent == files 05-06 in one call")
    out = agent.invoke({"messages": [HumanMessage("add 21 and 21")]})
    print_messages(out, title="prebuilt ReAct transcript")
    return out


if __name__ == "__main__":
    run_demo()
