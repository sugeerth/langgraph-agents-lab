"""Durable memory with SqliteSaver — survives a process restart.

CONCEPT: ``MemorySaver`` keeps checkpoints in RAM, so they vanish when the process exits.
``SqliteSaver`` has the IDENTICAL interface but writes checkpoints to a file on disk, so a
*new* process can re-open the same db and continue the conversation. We simulate a
"restart" with two separate ``with`` blocks: the first writes, the second only reads.

NOTE: SqliteSaver is a CONTEXT MANAGER — it opens the db connection on enter and closes it
on exit, so you must build/invoke the agent INSIDE the ``with`` block.

aha: same interface as MemorySaver, but the checkpoints survive a restart.
"""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

from lab.common import banner, get_model

DB_PATH = "/Users/fullfocus/langgraph-agents-lab/lab_m5.sqlite"


def demo_persist():
    """Write a conversation in one db session, then read it back in a fresh session."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    cfg = {"configurable": {"thread_id": "durable-1"}}

    # --- Session 1: write something to disk -------------------------------------------
    with SqliteSaver.from_conn_string(DB_PATH) as cp:
        agent = create_react_agent(
            get_model(script=["Got it, Sugeerth — I'll remember that."], default="ok"),
            [],
            checkpointer=cp,
        )
        agent.invoke({"messages": [HumanMessage("My name is Sugeerth.")]}, cfg)

    # --- Session 2: a NEW agent + NEW connection to the SAME file ---------------------
    # No invoke needed — just read the persisted state to prove it survived.
    with SqliteSaver.from_conn_string(DB_PATH) as cp:
        agent = create_react_agent(get_model(script=["ok"], default="ok"), [], checkpointer=cp)
        snapshot = agent.get_state(cfg)

    survived = snapshot.values.get("messages", [])
    return {"survived": survived, "count": len(survived)}


def run_demo():
    banner("Durable memory: write in session 1, read back in session 2 (a 'restart')")
    result = demo_persist()
    print(f"Restored {result['count']} messages from {DB_PATH} after reopening the db:")
    for m in result["survived"]:
        print(f"  {type(m).__name__:>13} | {m.content}")
    # Clean up the artifact so the repo stays tidy (sqlite files are gitignored anyway).
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"\nCleaned up {DB_PATH}.")
    return result


if __name__ == "__main__":
    run_demo()
