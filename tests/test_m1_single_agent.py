"""Structure tests for Module 1 — single agent (bare StateGraph -> prebuilt ReAct).

We assert on STRUCTURE, never on model prose beyond the scripted strings the fake model
is guaranteed to emit: merged state keys, reducer arithmetic, the router branch taken,
and that a ToolMessage actually appears in the ReAct transcripts.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# --------------------------------------------------------------------------- 01
def test_01_partial_state_dicts_merge(load_example):
    mod = load_example("m1_single_agent.01_bare_state_graph")
    out = mod.run_demo()
    # value: 1 -> (+1) 2 -> (*10) 20 ; trail records both nodes ran in order.
    assert out["value"] == 20
    assert out["trail"] == ["a ran", "b ran"]


def test_01_build_graph_smoke(load_example):
    mod = load_example("m1_single_agent.01_bare_state_graph")
    out = mod.build_graph().invoke({"value": 0, "trail": []})
    assert out["value"] == 10  # (0 + 1) * 10


# --------------------------------------------------------------------------- 02
def test_02_reducers_accumulate_vs_last_write_wins(load_example):
    mod = load_example("m1_single_agent.02_state_reducers")
    out = mod.run_demo()
    # operator.add reducer summed 0 + 1 + 41.
    assert out["total"] == 42
    # add_messages appended: HumanMessage + 2 AIMessages.
    assert len(out["messages"]) == 3
    assert [type(m).__name__ for m in out["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "AIMessage",
    ]
    # No reducer on last_label -> the second write clobbers the first.
    assert out["last_label"] == "set-by-two"


# --------------------------------------------------------------------------- 03
def test_03_single_llm_node_appends_reply(load_example):
    mod = load_example("m1_single_agent.03_single_llm_node")
    out = mod.run_demo()
    assert len(out["messages"]) == 2  # human in, AI reply appended
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content == "Hi! How can I help?"


# --------------------------------------------------------------------------- 04
def test_04_router_picks_branch_by_digit(load_example):
    mod = load_example("m1_single_agent.04_conditional_edges")
    graph = mod.build_graph()
    math_out = graph.invoke({"messages": [HumanMessage("compute 2 + 2")]})
    chat_out = graph.invoke({"messages": [HumanMessage("hello friend")]})
    assert math_out["messages"][-1].content == "Routing to the math branch."
    assert chat_out["messages"][-1].content == "Routing to the chat branch."


def test_04_router_function_directly(load_example):
    mod = load_example("m1_single_agent.04_conditional_edges")
    assert mod.route({"messages": [HumanMessage("has a 7")]}) == "math"
    assert mod.route({"messages": [HumanMessage("no digits here")]}) == "chat"


# --------------------------------------------------------------------------- 05
def test_05_handbuilt_react_runs_tool_then_answers(load_example):
    mod = load_example("m1_single_agent.05_handbuilt_react_loop")
    out = mod.run_demo()
    kinds = [type(m).__name__ for m in out["messages"]]
    assert "ToolMessage" in kinds  # the tool actually ran inside the loop
    assert out["messages"][-1].content == "The answer is 5."
    # the calculator returned 5.0 as a ToolMessage
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[0].content == "5.0"


def test_05_should_continue_signal(load_example):
    mod = load_example("m1_single_agent.05_handbuilt_react_loop")
    from langgraph.graph import END

    # A message with tool_calls -> keep looping; a plain answer -> END.
    tc_msg = AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "c1", "type": "tool_call"}])
    assert mod.should_continue({"messages": [tc_msg]}) == "tools"
    assert mod.should_continue({"messages": [AIMessage("done")]}) == END


# --------------------------------------------------------------------------- 06
def test_06_tools_condition_matches_handbuilt(load_example):
    mod = load_example("m1_single_agent.06_tools_condition_builtin")
    out = mod.run_demo()
    kinds = [type(m).__name__ for m in out["messages"]]
    assert "ToolMessage" in kinds
    assert out["messages"][-1].content == "The answer is 5."


# --------------------------------------------------------------------------- 07
def test_07_prebuilt_react_agent(load_example):
    mod = load_example("m1_single_agent.07_prebuilt_react_agent")
    out = mod.run_demo()
    kinds = [type(m).__name__ for m in out["messages"]]
    assert "ToolMessage" in kinds  # prebuilt wired the tools node for us
    assert out["messages"][-1].content == "It is 42."
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[0].content == "42.0"  # 21 + 21
