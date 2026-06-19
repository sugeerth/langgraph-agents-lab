"""Observability via callbacks — tracing is a config concern, not a code rewrite.

CONCEPT: you instrument a run by passing handlers through the config, NOT by editing your
graph. A ``BaseCallbackHandler`` receives lifecycle events — ``on_chat_model_start`` /
``on_llm_start`` (a model call begins), ``on_tool_start`` (a tool runs), and many more.
Drop one into ``config={"callbacks": [handler]}`` and the same agent is now traced.

The same hook surface backs production tracers — all env-gated and key-free by default:
  * LangSmith — set ``LANGSMITH_TRACING=true`` (+ ``LANGCHAIN_API_KEY``); auto-instruments.
  * Langfuse  — install ``langfuse`` and add its ``CallbackHandler`` to ``callbacks`` (uses
                ``LANGFUSE_*`` env vars). Nothing here phones home unless you opt in.

aha: tracing/observability is a config concern — attach a handler, don't rewrite the agent.
"""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from lab.common import banner, calculator, get_model
from lab.common.fake_model import tool_call


class CountingTracer(BaseCallbackHandler):
    """A minimal tracer: count model calls and tool calls, and log them."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_calls = 0
        self.events: list[str] = []

    # Chat models fire on_chat_model_start; older completion LLMs fire on_llm_start. Count both.
    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.llm_calls += 1
        self.events.append("chat_model_start")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1
        self.events.append("llm_start")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_calls += 1
        name = (serialized or {}).get("name", "?")
        self.events.append(f"tool_start:{name}")


def build_graph():
    model = get_model(
        script=[tool_call("calculator", {"a": 2, "b": 3, "op": "add"}), "The sum is 5."],
        default="ok",
    )
    return create_react_agent(model, [calculator])


def run_demo():
    agent = build_graph()
    tracer = CountingTracer()

    banner("Same agent, now traced — handler passed via config['callbacks']")
    agent.invoke(
        {"messages": [HumanMessage("add 2 and 3")]},
        config={"callbacks": [tracer]},  # <-- the ONLY change needed to observe the run
    )

    print(f"  model (llm/chat) calls : {tracer.llm_calls}")
    print(f"  tool calls            : {tracer.tool_calls}")
    print(f"  event log             : {tracer.events}")
    return tracer


if __name__ == "__main__":
    run_demo()
