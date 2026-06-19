"""Three stream modes, same agent — see exactly what each one yields.

CONCEPT: ``graph.stream(...)`` is the same call every time; only ``stream_mode`` changes
what comes out of the iterator:
  * "values"   -> the FULL state after each super-step (good for a live progress view)
  * "updates"  -> only the DELTA each node returned, keyed by node name (good for logging)
  * "messages" -> a (message_chunk, metadata) stream for token-level / chat UIs

aha: values = full state per step, updates = per-node deltas, messages = token/message
stream. Pick the granularity your UI needs.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from lab.common import banner, calculator, get_model
from lab.common.fake_model import tool_call


def _fresh_agent():
    """A fresh agent each time — the fake model's script is consumed per run."""
    model = get_model(
        script=[tool_call("calculator", {"a": 2, "b": 3, "op": "add"}), "The sum is 5."],
        default="ok",
    )
    return create_react_agent(model, [calculator])


def stream_in(mode: str):
    """Collect everything one stream_mode yields for a single small run."""
    agent = _fresh_agent()
    return list(agent.stream({"messages": [HumanMessage("add 2 and 3")]}, stream_mode=mode))


def run_demo():
    banner('stream_mode="values" — full state snapshot after every super-step')
    for i, state in enumerate(stream_in("values")):
        print(f"  step {i}: state has {len(state['messages'])} message(s)")

    banner('stream_mode="updates" — only each node\'s delta, keyed by node name')
    for upd in stream_in("updates"):
        for node, delta in upd.items():
            n = len(delta.get("messages", [])) if isinstance(delta, dict) else "?"
            print(f"  node '{node}' -> added {n} message(s)")

    banner('stream_mode="messages" — (chunk, metadata) pairs for a chat/token UI')
    chunks = stream_in("messages")
    print(f"  yielded {len(chunks)} (chunk, metadata) tuples; first chunk type = {type(chunks[0][0]).__name__}")

    return {
        "values": stream_in("values"),
        "updates": stream_in("updates"),
        "messages": stream_in("messages"),
    }


if __name__ == "__main__":
    run_demo()
