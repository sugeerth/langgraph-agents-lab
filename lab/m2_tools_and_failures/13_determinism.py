"""13 — Determinism is why this whole lab is testable.

CONCEPT: real LLMs are stochastic; the same prompt can give different answers, which makes
failure modes hard to reproduce and assert on. This lab pins three things:
  1. ``temperature=0`` (greedy decoding) in live mode — the model factory's default.
  2. the deterministic ``FakeChatModel`` in offline mode — scripted, so output is fixed.
  3. pinned tool inputs — same args in, same result out (toy tools are pure).
Together these make a run REPRODUCIBLE, so CI can assert on structure without flakiness.

aha: reproducibility is exactly why this lab is testable.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from lab.common import banner, calculator
from lab.common.fake_model import FakeChatModel


def _run_once():
    """One full 'run': scripted model output + a pinned tool call. Returns a comparable tuple."""
    # A fresh model each run (independent cursor) given the SAME script.
    model = FakeChatModel(script=["The deterministic answer is 42."])
    answer = model.invoke([HumanMessage("question")]).content
    tool_out = calculator.invoke({"a": 6, "b": 7, "op": "mul"})  # pinned inputs -> pinned output
    return (answer, tool_out)


def demo_broken():
    """A stochastic stand-in (different output across calls) — NOT reproducible.

    We use a counter so the 'model' returns a different number each call, mimicking
    temperature>0. Returns the two differing outputs."""
    state = {"n": 0}

    def flaky_random(_messages):
        state["n"] += 1
        return f"answer #{state['n']}"  # changes every call -> can't assert on it

    model = FakeChatModel(script=[], default=flaky_random)
    return (model.invoke([HumanMessage("q")]).content, model.invoke([HumanMessage("q")]).content)


def demo_fixed():
    """Two independent runs with the deterministic setup -> byte-identical results."""
    return (_run_once(), _run_once())


def run_demo():
    banner("13 — determinism: identical output across runs makes the lab testable")
    a, b = demo_broken()
    print(f"BROKEN (stochastic): run1={a!r}  run2={b!r}  identical={a == b}")
    r1, r2 = demo_fixed()
    print(f"FIXED  (deterministic): run1={r1}  run2={r2}  identical={r1 == r2}")


if __name__ == "__main__":
    run_demo()
