# LangGraph Agents Lab

A hands-on, **offline-runnable** teaching lab for learning how LLM agents actually work — built
as a progression you can read and run top to bottom:

> **single agent → tool use & its failure modes → multi-agent → orchestrator → advanced**

Every example is tiny and focused on **one concept**, and the whole thing runs with **zero API
keys** so you can read, run, break, and test agents without spending a cent.

🌐 **Showcase / learning path:** https://sugeerth.github.io/langgraph-agents-lab/
📦 **Built for:** LangGraph `0.3.34` · langchain-anthropic · Python 3.10+

---

## Why this exists

Most agent tutorials hand you `create_react_agent(...)` and stop. This lab does the opposite: you
**build the machinery by hand first** (state, nodes, edges, the ReAct loop) so the prebuilt stops
being magic — then you spend real time on the part tutorials skip: **how tool use fails, and what
LangGraph gives you to handle it.**

## Runs offline, deterministically

A scriptable `FakeChatModel` (in `lab/common/fake_model.py`) stands in for Claude. You hand it a
list of canned responses and it returns them in order, so every failure mode triggers on demand and
every example is testable in CI — no key, no cost, no flakiness.

```python
from lab.common import get_model
from lab.common.fake_model import tool_call

# OFFLINE: a scripted stand-in for Claude
model = get_model(script=[tool_call("calculator", {"a": 2, "b": 3, "op": "add"}), "The sum is 5."])
```

Set `ANTHROPIC_API_KEY` and the **exact same graphs** run against the real `claude-haiku-4-5`
(configurable via `LAB_MODEL`) — the script is simply ignored. Force offline anytime with
`LAB_USE_FAKE=1`.

---

## Quickstart

```bash
git clone https://github.com/sugeerth/langgraph-agents-lab
cd langgraph-agents-lab

./run.sh setup     # create .venv + install (uses uv if present, else venv+pip)
./run.sh test      # run the full pytest suite — offline, no keys
./run.sh list      # list every runnable example

# run a single example:
./run.sh m1_single_agent.07_prebuilt_react_agent
./run.sh m2_tools_and_failures.09_infinite_loop_recursion

# optional — use the real Claude instead of the fake model:
export ANTHROPIC_API_KEY=sk-...
```

> Manual setup: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`,
> then `PYTHONPATH=. pytest`.

---

## The learning path

Each module is a folder of numbered, one-concept-per-file examples. Read them in order.

### `m1_single_agent/` — from a bare graph to a ReAct agent
| file | concept | the "aha" |
|---|---|---|
| `01_bare_state_graph` | state + nodes + edges, **no LLM** | an agent is a state machine; the LLM is optional |
| `02_state_reducers` | `add_messages`, `operator.add` | a reducer is the merge rule for a state key |
| `03_single_llm_node` | wrap one model call as a node | the LLM node is a pure `state → state` function |
| `04_conditional_edges` | branch on state | control flow is a plain Python router |
| `05_handbuilt_react_loop` | the ReAct cycle, by hand | ReAct is literally a loop; `tool_calls` is the signal |
| `06_tools_condition_builtin` | swap in `tools_condition` | the prebuilt router == the one you wrote |
| `07_prebuilt_react_agent` | `create_react_agent` | the prebuilt == files 05–06 in one call |

### `m2_tools_and_failures/` — ⭐ the centerpiece
Each file deliberately **triggers** a failure, then applies the LangGraph **fix**. See the matrix
below.

### `m3_multi_agent/` — multiple specialists
| file | concept | the "aha" |
|---|---|---|
| `01_shared_state_two_nodes` | nodes over one shared state | "multi-agent" can just be multiple nodes |
| `02_handoff_command` | `Command(goto=, update=)` | one return carries both the update and the next node |
| `03_network_routing` | each agent routes to a peer | the topology is emergent from each node's `goto` |
| `04_handoff_as_tool` | handoff as a tool call | the from-scratch version of a "swarm" |

### `m4_orchestrator/` — central control
| file | concept | the "aha" |
|---|---|---|
| `01_supervisor_from_scratch` | a routing node + workers | a supervisor is one node whose job is routing |
| `02_supervisor_structured_route` | `with_structured_output` routing | typed routes beat fragile free-text |
| `03_orchestrator_worker_mapreduce` | `Send` fan-out + reducer | dynamic fan-out + a list-reducer = map-reduce |
| `04_hierarchical_teams` | a subgraph as a node | hierarchy = compiled graphs nested as nodes |

### `m5_advanced/` — the production surface ("many more")
| file | concept | the "aha" |
|---|---|---|
| `01_memory_saver_threads` | `MemorySaver` + `thread_id` | memory = a checkpointer keyed by thread |
| `02_sqlite_saver_persist` | durable `SqliteSaver` | same interface, survives restarts |
| `03_time_travel` | `get_state_history`, replay | every super-step is a snapshot |
| `04_stream_modes` | `values` / `updates` / `messages` | pick the mode by what you render |
| `05_token_streaming` | `stream_mode="messages"` | token streaming is just a stream mode |
| `06_subgraphs` | a compiled graph as a node | subgraphs encapsulate sub-workflows |
| `07_parallel_fanout_reducer` | parallel writes need a reducer | reducers are the merge contract |
| `08_structured_output` | `response_format` | agents return validated **data**, not prose |
| `09_tracing_hooks` | callbacks / LangSmith / Langfuse | tracing is a config concern, not a rewrite |

---

## The failure-modes matrix

The reason this lab exists. Real agentic tool use fails in many ways — most of them quietly.

| # | Failure | Trigger | LangGraph mechanism |
|---|---|---|---|
| 01–03 | **Tool raises an exception** | a tool throws | `ToolNode(handle_tool_errors=…)` — catch → ToolMessage, custom string, or fail-fast |
| 04 | **Invalid / missing arguments** | model passes a non-int | pydantic `ValidationError` surfaced as a ToolMessage (self-correction signal) |
| 05 | **Hanging / slow tool** | a tool blocks | `asyncio.wait_for` **inside** the tool — LangGraph won't kill a hung node |
| 06 | **Transient error** | flaky connection | `.with_retry()` — backoff + jitter |
| 07 | **Permanent failure** | primary always down | `.with_fallbacks([backup])` |
| 08 | **Rate limit (429)** | provider throttles | `.with_retry` + `InMemoryRateLimiter` |
| 09 | **Infinite tool-calling loop** | model never stops | `recursion_limit` → `GraphRecursionError` (raw graphs) |
| 10 | **…handled gracefully** | same loop | catch the error, or the prebuilt's built-in `remaining_steps` stop |
| 11 | **Hallucinated tool name** | calls a nonexistent tool | ToolNode returns a "tool not found" ToolMessage — no crash |
| 12 | **Wrong tool / ignored result** | answers from "memory" | detect via the transcript / `stream_mode="updates"`; mitigate with prompt + `response_format` |
| 13 | **Non-determinism** | same input, different output | `temperature=0` + the deterministic fake model |
| 14–15 | **Destructive action, no guardrail** | "delete prod" | `interrupt_before` / `interrupt()` + checkpointer → human approval, then resume |

---

## Project layout

```
lab/
  common/        shared infra: model factory, fake model, toy tools, state, printing
  m1_single_agent/        m2_tools_and_failures/   (★ centerpiece)
  m3_multi_agent/         m4_orchestrator/         m5_advanced/
tests/           one test module per lab module + test_common.py (all run offline)
docs/index.html  the showcase / learning-path website (GitHub Pages)
```

Every example exposes `run_demo()` and `if __name__ == "__main__": run_demo()`, and pulls its model
from `lab.common.get_model()` so it runs offline or live unchanged.

## Going further

- **Prebuilt multi-agent libs:** `langgraph-supervisor` (`create_supervisor`) and `langgraph-swarm`
  (`create_swarm`) package what modules 3–4 build by hand. We build from scratch on purpose.
- **Tracing:** set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`, or wire Langfuse — both are
  env-gated and off by default (see `m5_advanced/09_tracing_hooks.py`).

## Glossary

**state** the typed dict every node reads/writes · **node** a `state → partial-state` function ·
**edge** wiring between nodes · **reducer** how repeated/parallel writes to a key merge ·
**super-step** one tick of the graph (the unit the recursion limit counts) ·
**checkpointer** persists state per `thread_id` (memory + resume) · **interrupt** a pause point for
human-in-the-loop · **`Command`** a node's "update + go here next" · **`Send`** dynamic fan-out, one
branch per item.

---

*Destructive tools are simulated — they record intent and never touch your filesystem.*
