"""14 — Human-in-the-loop: pause BEFORE a destructive tool, then approve and resume.

CONCEPT: a destructive action (here ``dangerous_delete``) should not run unsupervised.
``create_react_agent(..., interrupt_before=["tools"], checkpointer=MemorySaver())`` pauses the
graph at the boundary *before* the tools node. The checkpointer holds the full state under a
``thread_id`` so a human can inspect the pending call, then ``invoke(None, cfg)`` resumes and
runs it.

aha: HITL = pause before a node; the checkpointer persists state so you can inspect & resume.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from lab.common import banner, dangerous_delete, get_model, print_messages
from lab.common.fake_model import tool_call


def build_graph():
    """A ReAct agent gated to pause before any tool call, backed by an in-memory checkpointer."""
    model = get_model(
        script=[tool_call("dangerous_delete", {"target": "prod-db"}), "Done — prod-db was deleted."],
        default="Done — prod-db was deleted.",
    )
    return create_react_agent(
        model,
        [dangerous_delete],
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],  # pause at the boundary before tools run
    )


def demo_broken():
    """Without the gate the destructive tool runs immediately, no chance to approve.

    Returns the state: a ToolMessage is already present (the delete happened unsupervised)."""
    model = get_model(
        script=[tool_call("dangerous_delete", {"target": "prod-db"}), "Done — prod-db was deleted."],
    )
    agent = create_react_agent(model, [dangerous_delete])  # NO interrupt, NO checkpointer
    return agent.invoke({"messages": [HumanMessage("delete prod-db")]})


def demo_fixed():
    """Pause before the tool, confirm it's pending, then resume to actually run it.

    Returns (paused_next, final_state) so a test can assert on both phases."""
    agent = build_graph()
    cfg = {"configurable": {"thread_id": "hitl-1"}}
    agent.invoke({"messages": [HumanMessage("delete prod-db")]}, cfg)
    paused_next = agent.get_state(cfg).next  # ('tools',) — paused before the delete
    final = agent.invoke(None, cfg)  # human approved -> resume and run the tool
    return paused_next, final


def run_demo():
    banner("14 — HITL: interrupt_before=['tools'] pauses, then resume to run the delete")
    broken = demo_broken()
    ran = any(isinstance(m, ToolMessage) for m in broken["messages"])
    print(f"BROKEN (no gate): destructive tool ran without approval? {ran}")
    paused_next, final = demo_fixed()
    print(f"FIXED: paused at {paused_next} (awaiting approval)")
    print_messages(final, title="FIXED — after approval the delete ran and the agent finished")


if __name__ == "__main__":
    run_demo()
