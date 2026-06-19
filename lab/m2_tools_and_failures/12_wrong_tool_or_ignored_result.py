"""12 — The dangerous failures are SILENT: a wrong answer with no exception.

CONCEPT: the scariest failure mode throws nothing. The tool runs and returns the right
number, but the model IGNORES the ToolMessage and answers from its (wrong) "prior knowledge".
Nothing errors — the transcript is the only evidence. You detect it by inspecting the message
log (or ``stream_mode="updates"``) and comparing the tool's output to the final answer; you
mitigate it with a stricter prompt and a typed ``response_format`` that forces the model to
commit to a value derived from the tool.

aha: the dangerous failures are silent, not exceptions — detect them via the transcript.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from lab.common import banner, calculator, get_model, print_messages
from lab.common.fake_model import tool_call


class Answer(BaseModel):
    """A typed final answer — forces the model to emit a single committed number."""

    value: float


def _tool_result_number(messages) -> float | None:
    """Pull the calculator's actual output out of the transcript (ground truth)."""
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "calculator":
            try:
                return float(m.content)
            except ValueError:
                return None
    return None


def demo_broken():
    """Tool correctly returns 42.0, but the scripted model 'knows' the answer is 99.

    Returns the full state so a test can detect the mismatch from the transcript."""
    model = get_model(
        script=[
            tool_call("calculator", {"a": 6, "b": 7, "op": "mul"}),  # tool will return 42.0
            "The answer is 99.",  # model IGNORES the tool result — silently wrong!
        ]
    )
    agent = create_react_agent(model, [calculator])
    out = agent.invoke({"messages": [HumanMessage("what is 6 * 7?")]})
    truth = _tool_result_number(out["messages"])
    stated = out["messages"][-1].content
    out["_mismatch"] = (truth is not None) and (str(int(truth)) not in stated)
    out["_truth"] = truth
    return out


def demo_fixed():
    """Stricter prompt + typed response_format anchors the answer to the tool output."""
    model = get_model(
        script=[
            tool_call("calculator", {"a": 6, "b": 7, "op": "mul"}),  # returns 42.0
            "Using the tool result, the answer is 42.",  # now consistent with the tool
            Answer(value=42.0),  # the LAST scripted entry -> structured_response
        ]
    )
    agent = create_react_agent(
        model,
        [calculator],
        prompt="You MUST base your final answer only on the tool's output. Never use prior knowledge.",
        response_format=Answer,
    )
    out = agent.invoke({"messages": [HumanMessage("what is 6 * 7?")]})
    truth = _tool_result_number(out["messages"])
    out["_consistent"] = (truth is not None) and (out["structured_response"].value == truth)
    out["_truth"] = truth
    return out


def run_demo():
    banner("12 — silently wrong: model ignores the tool result (detect via the transcript)")
    broken = demo_broken()
    print_messages(broken, title="BROKEN — tool said 42, model claimed 99")
    print(f"  tool truth={broken['_truth']}  mismatch_detected={broken['_mismatch']}")
    fixed = demo_fixed()
    print_messages(fixed, title="FIXED — stricter prompt + response_format")
    print(f"  structured_response={fixed['structured_response']}  consistent={fixed['_consistent']}")


if __name__ == "__main__":
    run_demo()
