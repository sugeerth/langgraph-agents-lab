"""Module 2 tests — tool use and its failure modes.

Each example gets a smoke test (run_demo / build_graph does not raise) plus the KEY
assertion for its concept. We assert on STRUCTURE (message types, statuses, error vs
success branches, raised exceptions, paused-then-resumed graphs) — never on LLM prose."""

from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError


# --------------------------------------------------------------------------- 01
def test_01_default_toolnode_catches_exception(load_example):
    mod = load_example("m2_tools_and_failures.01_tool_raises_default")
    out = mod.demo_fixed()
    errs = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert errs, "the raising tool should produce at least one error ToolMessage"
    assert out["messages"][-1].content  # the loop survived and produced a final answer
    mod.run_demo()  # smoke


def test_01_broken_returns_raw_exception(load_example):
    mod = load_example("m2_tools_and_failures.01_tool_raises_default")
    assert isinstance(mod.demo_broken(), Exception)


# --------------------------------------------------------------------------- 02
def test_02_custom_error_string_and_callable(load_example):
    mod = load_example("m2_tools_and_failures.02_tool_raises_custom")
    string_msg, callable_msg = mod.demo_fixed()
    assert string_msg.content == "Retry with a smaller value."
    assert string_msg.status == "error"
    # the callable form embeds the exception type name it received
    assert "ValueError" in callable_msg.content
    assert callable_msg.status == "error"
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 03
def test_03_handling_off_propagates(load_example):
    mod = load_example("m2_tools_and_failures.03_tool_raises_off")
    with pytest.raises(Exception):
        mod.demo_broken()
    # and with handling on, the same failure is a recoverable error message
    fixed = mod.demo_fixed()
    assert isinstance(fixed, ToolMessage) and fixed.status == "error"
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 04
def test_04_invalid_args_become_error_message(load_example):
    mod = load_example("m2_tools_and_failures.04_invalid_tool_args")
    bad = mod.demo_broken()
    assert isinstance(bad, ToolMessage)
    assert bad.status == "error"
    assert "int" in bad.content.lower() or "valid" in bad.content.lower()
    # the full loop self-corrects: one error ToolMessage, then a successful one
    out = mod.demo_fixed()
    tools = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert any(t.status == "error" for t in tools)
    assert any(t.status != "error" for t in tools)
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 05
def test_05_timeout_inside_tool(load_example):
    mod = load_example("m2_tools_and_failures.05_tool_timeout")
    out = mod.demo_fixed()
    errs = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.status == "error"]
    assert errs, "the timeout should surface as an error ToolMessage"
    assert "budget" in errs[0].content.lower() or "slow" in errs[0].content.lower()
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 06
def test_06_retry_eventually_succeeds(load_example):
    mod = load_example("m2_tools_and_failures.06_retry_with_backoff")
    assert isinstance(mod.demo_broken(), Exception)  # single attempt fails
    result = mod.demo_fixed()
    assert isinstance(result, str) and "processed" in result  # retries cleared the failures
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 07
def test_07_fallback_answers_when_primary_fails(load_example):
    mod = load_example("m2_tools_and_failures.07_fallbacks")
    assert isinstance(mod.demo_broken(), Exception)
    fixed = mod.demo_fixed()
    assert "backup" in fixed
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 08
def test_08_rate_limit_is_retryable(load_example):
    mod = load_example("m2_tools_and_failures.08_rate_limit_429")
    assert isinstance(mod.demo_broken(), mod.RateLimitError)
    fixed = mod.demo_fixed()
    assert isinstance(fixed, str) and "answer" in fixed
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 09
def test_09_raw_loop_raises_recursion_error(load_example):
    mod = load_example("m2_tools_and_failures.09_infinite_loop_recursion")
    from langchain_core.messages import HumanMessage

    graph = mod.build_graph()
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("go")]}, config={"recursion_limit": 5})
    # demo_broken returns the caught instance
    assert isinstance(mod.demo_broken(limit=5), GraphRecursionError)
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 10
def test_10_graceful_cap_two_ways(load_example):
    mod = load_example("m2_tools_and_failures.10_graceful_cap")
    # (a) raw loop: broken raises, fixed catches into a partial answer
    assert isinstance(mod.demo_broken(limit=6), GraphRecursionError)
    capped = mod.demo_fixed(limit=6)
    assert capped["status"] == "capped" and "partial" in capped["answer"].lower()
    # (b) create_react_agent stops GRACEFULLY — no exception, no trailing tool_calls
    out = mod.demo_react_graceful(limit=8)
    last = out["messages"][-1]
    assert not last.tool_calls
    assert "need more steps" in last.content.lower()
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 11
def test_11_hallucinated_tool_is_error_not_crash(load_example):
    mod = load_example("m2_tools_and_failures.11_hallucinated_tool")
    bad = mod.demo_broken()
    assert isinstance(bad, ToolMessage)
    assert bad.status == "error"
    assert "valid tool" in bad.content.lower() or "not a valid" in bad.content.lower()
    # the loop recovers by calling a real tool and finishing
    out = mod.demo_fixed()
    tools = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert any(t.name == "calculator" for t in tools)
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 12
def test_12_silently_wrong_detected_via_transcript(load_example):
    mod = load_example("m2_tools_and_failures.12_wrong_tool_or_ignored_result")
    broken = mod.demo_broken()
    assert broken["_truth"] == 42.0
    assert broken["_mismatch"] is True  # model ignored the tool -> detectable mismatch
    # fixed: structured response is consistent with the tool's actual output
    fixed = mod.demo_fixed()
    assert fixed["structured_response"].value == 42.0
    assert fixed["_consistent"] is True
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 13
def test_13_deterministic_runs_are_identical(load_example):
    mod = load_example("m2_tools_and_failures.13_determinism")
    a, b = mod.demo_broken()
    assert a != b  # stochastic stand-in changes between calls
    r1, r2 = mod.demo_fixed()
    assert r1 == r2  # deterministic setup -> identical across runs
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 14
def test_14_interrupt_before_pauses_then_resumes(load_example):
    mod = load_example("m2_tools_and_failures.14_hitl_interrupt_before")
    paused_next, final = mod.demo_fixed()
    assert paused_next == ("tools",)  # paused at the boundary before the delete
    assert any(isinstance(m, ToolMessage) for m in final["messages"])  # ran after resume
    # broken path: with no gate the destructive tool ran immediately
    broken = mod.demo_broken()
    assert any(isinstance(m, ToolMessage) for m in broken["messages"])
    mod.run_demo()  # smoke


# --------------------------------------------------------------------------- 15
def test_15_dynamic_interrupt_approve_vs_deny_differs(load_example):
    mod = load_example("m2_tools_and_failures.15_hitl_dynamic_interrupt")
    res = mod.demo_fixed()
    approved = res["approved"]["result"]
    denied = res["denied"]["result"]
    assert approved != denied  # the two branches diverge
    assert "deleted" in approved.lower()
    assert "skipped" in denied.lower()
    mod.run_demo()  # smoke
