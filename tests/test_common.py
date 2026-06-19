"""Unit tests for the shared infrastructure. These must pass before any example is
trusted — the entire offline/CI promise rests on the fake model and toy tools behaving
exactly as documented."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from pydantic import BaseModel

from lab.common.fake_model import FakeChatModel, parallel_tool_calls, tool_call
from lab.common.tools import (
    bad_schema_tool,
    calculator,
    make_flaky_tool,
    make_slow_tool,
    mock_web_search,
)


# --------------------------------------------------------------------------- fake model
def test_text_response_is_consumed_in_order():
    m = FakeChatModel(script=["first", "second"])
    assert m.invoke([HumanMessage("a")]).content == "first"
    assert m.invoke([HumanMessage("b")]).content == "second"


def test_exhausted_script_falls_back_to_default():
    m = FakeChatModel(script=["only"], default="fallback")
    assert m.invoke([HumanMessage("a")]).content == "only"
    assert m.invoke([HumanMessage("b")]).content == "fallback"


def test_tool_call_helper_drives_react_loop():
    m = FakeChatModel(script=[tool_call("calculator", {"a": 2, "b": 3, "op": "add"}), "It is 5."])
    agent = create_react_agent(m, [calculator])
    out = agent.invoke({"messages": [HumanMessage("add 2 and 3")]})
    kinds = [type(x).__name__ for x in out["messages"]]
    assert "ToolMessage" in kinds  # the tool actually ran
    assert out["messages"][-1].content == "It is 5."


def test_parallel_tool_calls_all_execute():
    ai = parallel_tool_calls(
        [("calculator", {"a": 10, "b": 4, "op": "mul"}), ("mock_web_search", {"query": "langgraph"})]
    )
    res = ToolNode([calculator, mock_web_search]).invoke({"messages": [ai]})
    tools = [m for m in res["messages"] if isinstance(m, ToolMessage)]
    assert {t.name for t in tools} == {"calculator", "mock_web_search"}


def test_tools_condition_reads_tool_calls():
    assert tools_condition({"messages": [tool_call("calculator", {})]}) == "tools"
    assert tools_condition({"messages": [AIMessage(content="done")]}) == "__end__"


def test_with_structured_output_returns_schema():
    class Route(BaseModel):
        next: str

    m = FakeChatModel(script=[Route(next="researcher"), {"next": "writer"}])
    runnable = m.with_structured_output(Route)
    assert runnable.invoke([HumanMessage("x")]).next == "researcher"
    assert runnable.invoke([HumanMessage("y")]).next == "writer"  # dict coerced


def test_callable_entry_loops_forever():
    looper = lambda messages: tool_call("calculator", {"a": 1, "b": 1, "op": "add"})
    m = FakeChatModel(script=[], default=looper)
    for _ in range(3):
        assert m.invoke([HumanMessage("x")]).tool_calls  # always asks for a tool


def test_streaming_yields_multiple_tokens():
    m = FakeChatModel(script=["alpha beta gamma"])
    chunks = [c.content for c in m.stream([HumanMessage("x")]) if c.content]
    assert len(chunks) >= 3
    assert "".join(chunks) == "alpha beta gamma"


# --------------------------------------------------------------------------- tools
@pytest.mark.parametrize(
    "op,expected", [("add", "5.0"), ("sub", "-1.0"), ("mul", "6.0"), ("div", "0.6666666666666666")]
)
def test_calculator_ops(op, expected):
    assert calculator.invoke({"a": 2, "b": 3, "op": op}) == expected


def test_calculator_rejects_unknown_op():
    with pytest.raises(ValueError):
        calculator.invoke({"a": 1, "b": 1, "op": "pow"})


def test_mock_web_search_hit_and_miss():
    assert "LangGraph" in mock_web_search.invoke({"query": "what is langgraph"})
    assert "No results" in mock_web_search.invoke({"query": "quantum gastronomy"})


def test_flaky_tool_fails_then_succeeds():
    flaky = make_flaky_tool(fail_times=2)
    with pytest.raises(ValueError):
        flaky.invoke({"payload": "x"})
    with pytest.raises(ValueError):
        flaky.invoke({"payload": "x"})
    assert "processed" in flaky.invoke({"payload": "x"})  # third call succeeds


def test_bad_schema_tool_validates_via_toolnode():
    # The model "hallucinates" a non-integer arg; ToolNode surfaces the validation error.
    ai = tool_call("bad_schema_tool", {"x": "not-an-int"})
    res = ToolNode([bad_schema_tool]).invoke({"messages": [ai]})
    msg = res["messages"][-1]
    assert isinstance(msg, ToolMessage)
    assert msg.status == "error" or "validation" in msg.content.lower() or "int" in msg.content.lower()


def test_slow_tool_actually_sleeps():
    import time

    slow = make_slow_tool(delay=0.2)
    t0 = time.perf_counter()
    slow.invoke({"payload": "x"})
    assert time.perf_counter() - t0 >= 0.2
