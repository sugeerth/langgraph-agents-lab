"""Memory via a checkpointer keyed by thread_id.

CONCEPT: an agent has no memory of its own — you give it memory by attaching a
*checkpointer*. The checkpointer saves the graph state after every super-step, filed
under the ``thread_id`` you pass in the config. Re-invoke with the SAME thread_id and the
prior messages are loaded back in; use a DIFFERENT thread_id and you get a clean slate.

aha: memory = a checkpointer keyed by thread_id. Same id -> same conversation; new id ->
new conversation. The agent code does not change at all.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from lab.common import banner, get_model, print_messages


def build_graph():
    """A bare text agent (no tools) wired to an in-memory checkpointer."""
    # Script mirrors what a real model would say: it can recall the name on turn 2 of t1,
    # but on the fresh thread t2 it has never heard the name.
    model = get_model(
        script=[
            "Nice to meet you, Sugeerth!",  # t1, turn 1
            "Your name is Sugeerth.",  # t1, turn 2 — remembered because state was reloaded
            "I don't know your name yet.",  # t2, turn 1 — fresh thread, no history
        ],
        default="ok",
    )
    return create_react_agent(model, [], checkpointer=MemorySaver())


def run_demo():
    agent = build_graph()
    t1 = {"configurable": {"thread_id": "t1"}}
    t2 = {"configurable": {"thread_id": "t2"}}

    banner("Thread t1: two invokes share one conversation (memory persists)")
    agent.invoke({"messages": [HumanMessage("My name is Sugeerth.")]}, t1)
    out1 = agent.invoke({"messages": [HumanMessage("What is my name?")]}, t1)
    print_messages(out1, title="t1 transcript (grows across invokes)")

    banner("Thread t2: a brand-new thread_id => a fresh conversation (no memory of t1)")
    out2 = agent.invoke({"messages": [HumanMessage("What is my name?")]}, t2)
    print_messages(out2, title="t2 transcript (starts empty)")

    # t1 accumulated 4 messages (2 human + 2 ai); t2 only saw its single exchange.
    print(f"\nt1 messages: {len(out1['messages'])}  |  t2 messages: {len(out2['messages'])}")
    return {"t1": out1, "t2": out2}


if __name__ == "__main__":
    run_demo()
