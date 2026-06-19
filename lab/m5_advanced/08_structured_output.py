"""Structured output — make an agent return validated DATA, not just prose.

CONCEPT: pass ``response_format=<pydantic model>`` to ``create_react_agent`` and, after the
agent finishes its tool-using reasoning, it produces one more turn that is coerced into your
schema. The result lands in ``out["structured_response"]`` as a real, validated instance you
can ``.field`` into — no regex-parsing the prose.

SCRIPT ORDER for the offline fake model: tool turn(s) -> a final text answer -> THEN the
structured ``Answer(...)`` instance LAST (the structured turn happens after the chat turn).

aha: response_format turns a chatty agent into a typed function returning structured data.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

from lab.common import banner, calculator, get_model
from lab.common.fake_model import tool_call


class Answer(BaseModel):
    """The typed shape we want back instead of free-form text."""

    value: int
    explanation: str


def build_graph():
    model = get_model(
        script=[
            tool_call("calculator", {"a": 2, "b": 3, "op": "add"}),  # 1) use a tool
            "The sum of 2 and 3 is 5.",  # 2) a normal final text turn
            Answer(value=5, explanation="2 + 3 = 5"),  # 3) the structured turn, LAST
        ],
        default="done",
    )
    return create_react_agent(model, [calculator], response_format=Answer)


def run_demo():
    agent = build_graph()
    banner("Agent with response_format=Answer -> validated data, not prose")
    out = agent.invoke({"messages": [HumanMessage("add 2 and 3 and explain")]})

    structured = out["structured_response"]
    print(f"  type(structured_response) = {type(structured).__name__}")
    print(f"  .value       = {structured.value}")
    print(f"  .explanation = {structured.explanation!r}")
    print(f"  is Answer instance? {isinstance(structured, Answer)}")
    return out


if __name__ == "__main__":
    run_demo()
