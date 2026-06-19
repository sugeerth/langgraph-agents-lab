"""Time-travel debugging — every super-step is a checkpoint you can revisit.

CONCEPT: because a checkpointer snapshots state after every super-step, you get a full
history for free. ``get_state(cfg)`` returns the LATEST snapshot; ``get_state_history(cfg)``
returns ALL of them (newest first). Each snapshot's ``config`` carries a ``checkpoint_id``,
and invoking with that config REPLAYS the graph from exactly that point in the past.

aha: the checkpointer gives time-travel debugging — inspect any past state, then fork /
replay from it without re-running everything before it.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from lab.common import banner, calculator, get_model
from lab.common.fake_model import tool_call


def build_graph():
    """A one-tool ReAct agent so the history has several distinct super-steps."""
    model = get_model(
        script=[tool_call("calculator", {"a": 2, "b": 3, "op": "add"}), "The sum is 5."],
        # On replay the agent re-enters the loop; default keeps it answering, not looping.
        default="The sum is 5.",
    )
    return create_react_agent(model, [calculator], checkpointer=MemorySaver())


def run_demo():
    agent = build_graph()
    cfg = {"configurable": {"thread_id": "tt-1"}}

    banner("Run once, then walk the checkpoint history")
    agent.invoke({"messages": [HumanMessage("add 2 and 3")]}, cfg)

    current = agent.get_state(cfg)
    print(f"get_state -> latest snapshot has {len(current.values['messages'])} messages, next={current.next}")

    history = list(agent.get_state_history(cfg))  # newest -> oldest
    print(f"\nget_state_history -> {len(history)} snapshots (one per super-step):")
    for snap in history:
        step = snap.metadata.get("step")
        print(f"  step {step:>2} | next={str(snap.next):<14} | {len(snap.values.get('messages', []))} msgs")

    banner("Replay from a PAST checkpoint (time travel)")
    # Pick an early snapshot (just after the first model turn asked for the tool).
    past = history[-2]
    past_cfg = past.config  # carries the checkpoint_id
    print(f"Resuming from checkpoint_id={past_cfg['configurable']['checkpoint_id'][:8]}… (step {past.metadata.get('step')})")
    replayed = agent.invoke(None, past_cfg)  # invoke(None, <past cfg>) replays forward from there
    print(f"Replay produced a transcript of {len(replayed['messages'])} messages.")

    return {"current": current, "history": history, "replayed": replayed}


if __name__ == "__main__":
    run_demo()
