"""Tests for Module 4 — orchestration (supervisor, structured routing, map-reduce, teams).

We assert on STRUCTURE, never on model prose: which workers ran, that the loop terminated,
that map-reduce produced exactly one result per input item and reduced them, and that the
nested team contributed to the final shared state."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage


# --------------------------------------------------------------------- 01 supervisor from scratch
def test_01_supervisor_visits_both_workers_and_terminates(load_example):
    mod = load_example("m4_orchestrator.01_supervisor_from_scratch")
    result = mod.build_graph().invoke(
        {"messages": [HumanMessage("go")], "next": "", "step_count": 0}
    )
    # Both workers ran exactly once (step_count reducer = sum of the two +1 bumps).
    assert result["step_count"] == 2
    # The supervisor terminated by routing to FINISH.
    assert result["next"] == "FINISH"
    bodies = " ".join(str(m.content) for m in result["messages"])
    assert "researcher" in bodies and "writer" in bodies


def test_01_run_demo_smoke(load_example):
    mod = load_example("m4_orchestrator.01_supervisor_from_scratch")
    assert mod.run_demo()["step_count"] == 2


# --------------------------------------------------------------------- 02 structured routing
def test_02_route_decision_is_typed_literal(load_example):
    mod = load_example("m4_orchestrator.02_supervisor_structured_route")
    # The schema rejects an invalid worker name -> routing is a parseable contract.
    import pytest

    mod.RouteDecision(next="researcher")  # valid
    with pytest.raises(Exception):
        mod.RouteDecision(next="not-a-worker")


def test_02_supervisor_routes_via_structured_output_and_finishes(load_example):
    mod = load_example("m4_orchestrator.02_supervisor_structured_route")
    result = mod.build_graph().invoke(
        {"messages": [HumanMessage("go")], "next": "", "step_count": 0}
    )
    # Scripted plan: route to researcher, then FINISH -> loop terminates at FINISH.
    assert result["next"] == "FINISH"
    # The chosen worker actually ran and appended to the transcript.
    bodies = " ".join(str(m.content) for m in result["messages"])
    assert "researcher" in bodies
    assert result["step_count"] >= 1  # at least one worker did its bit


def test_02_run_demo_smoke(load_example):
    mod = load_example("m4_orchestrator.02_supervisor_structured_route")
    assert mod.run_demo()["next"] == "FINISH"


# --------------------------------------------------------------------- 03 map-reduce
def test_03_one_result_per_item_and_reduced(load_example):
    mod = load_example("m4_orchestrator.03_orchestrator_worker_mapreduce")
    items = ["a", "b", "c", "d", "e"]
    result = mod.build_graph().invoke({"items": items, "results": [], "summary": ""})
    # The map produced exactly one result per input item (the list reducer concatenated them).
    assert len(result["results"]) == len(items)
    assert set(result["results"]) == {f"processed:{it}" for it in items}
    # The reduce step read all of them.
    assert str(len(items)) in result["summary"]


def test_03_empty_input_produces_no_results(load_example):
    mod = load_example("m4_orchestrator.03_orchestrator_worker_mapreduce")
    result = mod.build_graph().invoke({"items": [], "results": [], "summary": ""})
    # Zero items -> zero Sends -> no worker runs (and synthesize, reached only via the
    # worker edge, is skipped). The map-reduce contributes nothing, as expected.
    assert result["results"] == []


def test_03_run_demo_smoke(load_example):
    mod = load_example("m4_orchestrator.03_orchestrator_worker_mapreduce")
    assert len(mod.run_demo()["results"]) == 4


# --------------------------------------------------------------------- 04 hierarchical teams
def test_04_nested_team_contributes_to_final_state(load_example):
    mod = load_example("m4_orchestrator.04_hierarchical_teams")
    result = mod.build_graph().invoke({"messages": [HumanMessage("go")]})
    bodies = [str(m.content) for m in result["messages"]]
    blob = " ".join(bodies)
    # The top supervisor ran...
    assert "[top]" in blob
    # ...and the nested team's three members all contributed to the SHARED transcript.
    team_msgs = [b for b in bodies if "[team]" in b]
    assert len(team_msgs) == 3
    assert any("searcher" in b for b in team_msgs)


def test_04_team_subgraph_runs_standalone(load_example):
    mod = load_example("m4_orchestrator.04_hierarchical_teams")
    # The team is a compiled graph on its own (this is what makes it nestable).
    team = mod.build_research_team()
    out = team.invoke({"messages": [HumanMessage("research")]})
    assert any("[team]" in str(m.content) for m in out["messages"])


def test_04_run_demo_smoke(load_example):
    mod = load_example("m4_orchestrator.04_hierarchical_teams")
    result = mod.run_demo()
    assert any("[team]" in str(m.content) for m in result["messages"])
