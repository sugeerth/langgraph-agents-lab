"""Structure tests for Module 3 — multi-agent systems.

We assert on STRUCTURE only (which agents contributed, which node handled the case,
that control was actually routed), never on LLM prose. The whole suite runs offline
against the scripted FakeChatModel (forced by tests/conftest.py)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# --------------------------------------------------------------------------- 01 shared state
def test_01_both_agents_contribute_messages(load_example):
    mod = load_example("m3_multi_agent.01_shared_state_two_nodes")
    out = mod.build_graph().invoke({"messages": [HumanMessage("summarize")]})
    ai_contents = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    # Researcher AND writer each appended exactly one AIMessage to the shared transcript.
    assert len(ai_contents) == 2
    assert any("Researcher" in c for c in ai_contents)
    assert any("Writer" in c for c in ai_contents)
    # The shared state preserved the original human turn too.
    assert isinstance(out["messages"][0], HumanMessage)


def test_01_run_demo_smoke(load_example):
    load_example("m3_multi_agent.01_shared_state_two_nodes").run_demo()


# --------------------------------------------------------------------------- 02 handoff command
def test_02_handoff_reaches_agent_b(load_example):
    mod = load_example("m3_multi_agent.02_handoff_command")
    out = mod.build_graph().invoke({"messages": [HumanMessage("write a memo")]})
    contents = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    # Control reached agent_b: its message is present (and it ran last).
    assert any("Agent A" in c for c in contents)
    assert any("Agent B" in c for c in contents)
    assert "Agent B" in out["messages"][-1].content


def test_02_agent_a_returns_command_goto_agent_b(load_example):
    """The handoff is the Command itself: agent_a returns goto='agent_b' + an update."""
    from langgraph.types import Command

    mod = load_example("m3_multi_agent.02_handoff_command")
    cmd = mod.agent_a({"messages": [HumanMessage("x")]})
    assert isinstance(cmd, Command)
    assert cmd.goto == "agent_b"
    assert cmd.update["messages"]  # carries a state update too


def test_02_run_demo_smoke(load_example):
    load_example("m3_multi_agent.02_handoff_command").run_demo()


# --------------------------------------------------------------------------- 03 network routing
def test_03_billing_request_routes_to_billing(load_example):
    mod = load_example("m3_multi_agent.03_network_routing")
    out = mod.build_graph().invoke(
        {"messages": [HumanMessage("I need a refund for a wrong charge")]}
    )
    contents = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    assert any("Billing" in c for c in contents)
    assert not any("Tech:" in c for c in contents)  # tech specialist never ran


def test_03_tech_request_routes_to_tech(load_example):
    mod = load_example("m3_multi_agent.03_network_routing")
    out = mod.build_graph().invoke(
        {"messages": [HumanMessage("my app keeps crashing on launch")]}
    )
    contents = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    assert any("Tech:" in c for c in contents)
    assert not any("Billing:" in c for c in contents)  # billing specialist never ran


def test_03_triage_goto_is_state_dependent(load_example):
    """Routing is distributed: triage's goto changes with the request."""
    from langgraph.types import Command

    mod = load_example("m3_multi_agent.03_network_routing")
    bill = mod.triage({"messages": [HumanMessage("please refund this charge")]})
    tech = mod.triage({"messages": [HumanMessage("everything is broken")]})
    assert isinstance(bill, Command) and bill.goto == "billing"
    assert isinstance(tech, Command) and tech.goto == "tech"


def test_03_run_demo_smoke(load_example):
    load_example("m3_multi_agent.03_network_routing").run_demo()


# --------------------------------------------------------------------------- 04 handoff as tool
def test_04_handoff_tool_returns_command(load_example):
    """The handoff tool itself returns a Command(goto='writer') — the swarm primitive."""
    from langgraph.types import Command

    mod = load_example("m3_multi_agent.04_handoff_as_tool")
    cmd = mod.transfer_to_writer.invoke(
        {
            "name": "transfer_to_writer",
            "args": {"reason": "done"},
            "id": "h1",
            "type": "tool_call",
        }
    )
    assert isinstance(cmd, Command)
    assert cmd.goto == "writer"


def test_04_tool_call_routes_control_to_writer(load_example):
    mod = load_example("m3_multi_agent.04_handoff_as_tool")
    out = mod.build_graph().invoke({"messages": [HumanMessage("research then write")]})
    kinds = [type(m).__name__ for m in out["messages"]]
    # The model called the handoff tool, a ToolMessage answered it, and the writer ran last.
    assert "ToolMessage" in kinds
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert any(m.name == "transfer_to_writer" for m in tool_msgs)
    assert isinstance(out["messages"][-1], AIMessage)
    assert "Writer" in out["messages"][-1].content  # control reached the writer agent


def test_04_run_demo_smoke(load_example):
    load_example("m3_multi_agent.04_handoff_as_tool").run_demo()
