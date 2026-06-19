"""Tests for module 5 — advanced patterns (memory, streaming, subgraphs, parallelism,
structured output, tracing). Everything runs OFFLINE via the fake model. We assert on
STRUCTURE (counts, shapes, types, raised errors), never on LLM prose."""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel


# ---------------------------------------------------------------- 01 memory across threads
def test_thread_id_isolates_memory(load_example):
    mod = load_example("m5_advanced.01_memory_saver_threads")
    out = mod.run_demo()
    # t1 ran two invokes that share one conversation; t2 is a brand-new thread.
    assert len(out["t1"]["messages"]) > len(out["t2"]["messages"])
    assert len(out["t1"]["messages"]) == 4  # 2 human + 2 ai accumulated on t1
    assert len(out["t2"]["messages"]) == 2  # only t2's own single exchange


# ---------------------------------------------------------------- 02 durable sqlite persist
def test_sqlite_state_survives_restart(load_example):
    mod = load_example("m5_advanced.02_sqlite_saver_persist")
    result = mod.run_demo()
    # The second db session (a simulated "restart") still saw the saved conversation.
    assert result["count"] >= 2
    assert len(result["survived"]) == result["count"]
    # run_demo cleans up the artifact.
    assert not os.path.exists(mod.DB_PATH)


# ---------------------------------------------------------------- 03 time travel
def test_state_history_has_multiple_snapshots(load_example):
    mod = load_example("m5_advanced.03_time_travel")
    out = mod.run_demo()
    assert len(out["history"]) >= 3  # several super-steps were snapshotted
    # Every snapshot carries a checkpoint_id (the thing that enables replay).
    assert all("checkpoint_id" in s.config["configurable"] for s in out["history"])
    # Replaying from a past checkpoint produced a transcript.
    assert len(out["replayed"]["messages"]) >= 1


# ---------------------------------------------------------------- 04 stream modes
def test_stream_modes_yield_expected_shapes(load_example):
    mod = load_example("m5_advanced.04_stream_modes")
    out = mod.run_demo()

    # values: every item is a full state dict containing "messages".
    assert len(out["values"]) >= 1
    assert all(isinstance(v, dict) and "messages" in v for v in out["values"])

    # updates: every item is a single-node delta dict keyed by node name.
    assert len(out["updates"]) >= 1
    assert all(isinstance(u, dict) and len(u) == 1 for u in out["updates"])
    node_names = {k for u in out["updates"] for k in u}
    assert "agent" in node_names and "tools" in node_names

    # messages: every item is a (chunk, metadata) tuple.
    assert len(out["messages"]) >= 1
    assert all(isinstance(item, tuple) and len(item) == 2 for item in out["messages"])


# ---------------------------------------------------------------- 05 token streaming
def test_token_streaming_emits_multiple_tokens(load_example):
    mod = load_example("m5_advanced.05_token_streaming")
    tokens = mod.run_demo()
    assert len(tokens) >= 3  # the fake model streams word-by-word
    assert all(isinstance(t, str) and t for t in tokens)
    # The text-only agent's final answer is exactly the concatenation of its tokens.
    assert "".join(tokens) == "LangGraph streams tokens one piece at a time."


def test_token_chunks_are_ai_message_chunks(load_example):
    mod = load_example("m5_advanced.05_token_streaming")
    agent = mod.build_graph()
    from langchain_core.messages import HumanMessage

    saw_chunk = False
    for chunk, _meta in agent.stream(
        {"messages": [HumanMessage("hi")]}, stream_mode="messages"
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            saw_chunk = True
    assert saw_chunk


# ---------------------------------------------------------------- 06 subgraphs
def test_subgraph_contributes_to_parent(load_example):
    mod = load_example("m5_advanced.06_subgraphs")
    out = mod.run_demo()
    # The subgraph's clean+shout steps ran -> uppercased + trailing "!".
    assert out["output"]["text"] == "HELLO FROM THE PARENT!"
    # Streaming with subgraphs=True surfaced at least one inside-subgraph event.
    assert out["saw_subgraph"] is True


# ---------------------------------------------------------------- 07 parallel fanout reducer
def test_parallel_fanout_broken_raises_fixed_merges(load_example):
    mod = load_example("m5_advanced.07_parallel_fanout_reducer")

    broken = mod.demo_broken()
    assert broken["raised"] is True  # no reducer -> concurrent writes collide

    fixed = mod.demo_fixed()
    # With operator.add both parallel writes are merged into one list.
    assert sorted(fixed["results"]) == ["A", "B"]


def test_parallel_fanout_run_demo_smoke(load_example):
    mod = load_example("m5_advanced.07_parallel_fanout_reducer")
    out = mod.run_demo()
    assert out["broken"]["raised"] is True
    assert sorted(out["fixed"]["results"]) == ["A", "B"]


# ---------------------------------------------------------------- 08 structured output
def test_structured_response_is_pydantic_instance(load_example):
    mod = load_example("m5_advanced.08_structured_output")
    out = mod.run_demo()
    structured = out["structured_response"]
    assert isinstance(structured, BaseModel)
    assert isinstance(structured, mod.Answer)
    assert structured.value == 5


# ---------------------------------------------------------------- 09 tracing hooks
def test_callback_counts_llm_and_tool_events(load_example):
    mod = load_example("m5_advanced.09_tracing_hooks")
    tracer = mod.run_demo()
    assert tracer.llm_calls >= 1  # at least one model call observed
    assert tracer.tool_calls >= 1  # at least one tool call observed
    assert any("tool_start" in e for e in tracer.events)


# ---------------------------------------------------------------- smoke: every example builds
@pytest.mark.parametrize(
    "dotted",
    [
        "m5_advanced.01_memory_saver_threads",
        "m5_advanced.02_sqlite_saver_persist",
        "m5_advanced.03_time_travel",
        "m5_advanced.04_stream_modes",
        "m5_advanced.05_token_streaming",
        "m5_advanced.06_subgraphs",
        "m5_advanced.07_parallel_fanout_reducer",
        "m5_advanced.08_structured_output",
        "m5_advanced.09_tracing_hooks",
    ],
)
def test_run_demo_does_not_raise(load_example, dotted):
    mod = load_example(dotted)
    assert mod.run_demo() is not None
