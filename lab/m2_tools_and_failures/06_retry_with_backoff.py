"""06 — Transient failures: retry the runnable until it succeeds.

CONCEPT: a flaky tool that fails its first N calls then works is a classic *transient*
error. The fix isn't try/except sprinkled through your nodes — it's ``.with_retry(...)`` on
the runnable. LangChain retries with exponential backoff and only the final failure (if any)
propagates.

aha: retries belong on the runnable — declare them, don't hand-roll them.
"""

from __future__ import annotations

from lab.common import banner, make_flaky_tool


def demo_broken():
    """One shot at a tool that needs 3 attempts -> it raises on the first call.

    Returns the exception so the contrast with the retried version is explicit."""
    flaky = make_flaky_tool(fail_times=2)  # fails twice, then succeeds
    try:
        return flaky.invoke({"payload": "ship-it"})
    except Exception as exc:  # noqa: BLE001
        return exc


def demo_fixed():
    """Same flaky tool, wrapped: the runnable retries and eventually returns the result."""
    flaky = make_flaky_tool(fail_times=2)
    resilient = flaky.with_retry(
        stop_after_attempt=3,  # 1 try + 2 retries = exactly enough to clear fail_times=2
        # retry_if_exception_type defaults to (Exception,); backoff is exponential by default.
    )
    return resilient.invoke({"payload": "ship-it"})


def run_demo():
    banner("06 — retry with backoff: .with_retry(stop_after_attempt=3)")
    broken = demo_broken()
    print(f"BROKEN (single attempt): {type(broken).__name__}: {broken}")
    print(f"FIXED (3 attempts):      {demo_fixed()!r}")


if __name__ == "__main__":
    run_demo()
