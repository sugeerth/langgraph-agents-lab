"""07 — Graceful degradation: fall back to a backup when the primary always fails.

CONCEPT: retries help with *transient* errors, but some dependencies are simply down. For
that, ``.with_fallbacks([backup])`` swaps in an alternate runnable when the primary raises.
The caller gets an answer from the backup instead of an exception.

aha: graceful degradation, declaratively — wire the backup once, the chain self-heals.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from lab.common import banner, make_flaky_tool


# A primary that is permanently broken (down for the whole demo), and a cheap backup.
_PRIMARY = make_flaky_tool(fail_times=999)  # never succeeds
_BACKUP = RunnableLambda(lambda payload: f"[backup] cached answer for {payload!r}")


def demo_broken():
    """Calling the down primary on its own just raises."""
    try:
        return _PRIMARY.invoke({"payload": "report"})
    except Exception as exc:  # noqa: BLE001
        return exc


def demo_fixed():
    """Primary -> backup: the chain answers from the backup when the primary fails.

    Note the input shapes differ, so we adapt the primary's dict-input via a small lambda;
    the fallback is tried with the SAME original input ("report")."""
    primary_as_runnable = RunnableLambda(lambda payload: _PRIMARY.invoke({"payload": payload}))
    chain = primary_as_runnable.with_fallbacks([_BACKUP])
    return chain.invoke("report")


def run_demo():
    banner("07 — fallbacks: .with_fallbacks([backup]) for graceful degradation")
    broken = demo_broken()
    print(f"BROKEN (primary only): {type(broken).__name__}: {broken}")
    print(f"FIXED (primary->backup): {demo_fixed()!r}")


if __name__ == "__main__":
    run_demo()
