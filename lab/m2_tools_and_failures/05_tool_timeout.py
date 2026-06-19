"""05 — A hanging tool: the TIMEOUT must live inside the tool, not the graph.

CONCEPT: LangGraph does NOT preempt a node that hangs — if a tool blocks forever, the whole
run blocks forever. So you enforce a deadline INSIDE the tool: run the slow call on a worker
thread and ``future.result(timeout=...)``. If the budget is exceeded you raise a
``TimeoutError`` — which (under the default ToolNode) becomes a recoverable ToolMessage.

CAVEAT (teachable): a Python thread can't be force-killed, so the orphaned worker keeps
running in the background; the timeout protects the *caller*, not the runaway thread.

aha: LangGraph won't kill a hung node — timeouts must live in the tool.
"""

from __future__ import annotations

import concurrent.futures

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent

from lab.common import banner, get_model, make_slow_tool, print_messages
from lab.common.fake_model import tool_call

_SLOW = make_slow_tool(delay=0.5)  # stands in for a hanging downstream call
_BUDGET = 0.1  # deadline well under the delay so the demo is fast + deterministic


def demo_broken():
    """No guard: invoking the slow tool directly just blocks for the full delay.

    Returns the (eventual) result to prove there is no timeout on the bare tool."""
    return _SLOW.invoke({"payload": "x"})  # returns only after 0.5s — nothing stopped it


@tool
def slow_with_timeout(payload: str) -> str:
    """Call a slow service but abandon it after a fixed time budget."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(lambda: _SLOW.invoke({"payload": payload}))
        try:
            return future.result(timeout=_BUDGET)
        except concurrent.futures.TimeoutError as exc:
            # Translate into a normal exception the ToolNode can render as a ToolMessage.
            raise TimeoutError(f"slow service exceeded {_BUDGET}s budget") from exc


def demo_fixed():
    """The wrapped tool raises TimeoutError -> default ToolNode -> error ToolMessage."""
    model = get_model(
        script=[tool_call("slow_with_timeout", {"payload": "x"}), "The service was too slow; I gave up."],
        default="The service was too slow; I gave up.",
    )
    agent = create_react_agent(model, [slow_with_timeout])
    return agent.invoke({"messages": [HumanMessage("call the slow service")]})


def run_demo():
    banner("05 — tool timeout: enforce the deadline INSIDE the tool")
    print("BROKEN: the bare slow tool blocks for the full delay, then returns:")
    print(f"  -> {demo_broken()!r}")
    out = demo_fixed()
    print_messages(out, title="FIXED — timeout raised, came back as an error ToolMessage")
    errs = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    print(f"\ntimeout error messages: {len(errs)}")


if __name__ == "__main__":
    run_demo()
