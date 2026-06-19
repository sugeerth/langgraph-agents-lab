"""08 — A 429 rate-limit is just a retryable transient error.

CONCEPT: provider rate limits surface as HTTP 429s. Reactively, they're handled exactly like
any transient failure: ``.with_retry(...)`` (with backoff) clears the spike. Proactively, you
can throttle BEFORE you send with ``langchain_core.rate_limiters.InMemoryRateLimiter`` (passed
as ``rate_limiter=`` to the chat model) so you stay under the limit in the first place.

aha: 429s are just retryable transient errors — backoff + an optional proactive limiter.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableLambda

from lab.common import banner

# from langchain_core.rate_limiters import InMemoryRateLimiter
#   limiter = InMemoryRateLimiter(requests_per_second=5, check_every_n_seconds=0.1)
#   get_model(rate_limiter=limiter)   # proactive: never exceed the limit (live mode only)


class RateLimitError(RuntimeError):
    """Stand-in for a provider 429 (e.g. anthropic.RateLimitError)."""


def _make_rate_limited_call(fail_times: int = 1):
    """A runnable that raises RateLimitError on its first ``fail_times`` calls, then succeeds."""
    state = {"calls": 0}

    def _call(prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise RateLimitError(f"429 Too Many Requests (attempt {state['calls']})")
        return f"answer to {prompt!r} (after {state['calls']} attempt(s))"

    return RunnableLambda(_call)


def demo_broken():
    """A single call during a rate-limit spike just raises the 429."""
    try:
        return _make_rate_limited_call(fail_times=1).invoke("summarize the doc")
    except RateLimitError as exc:
        return exc


def demo_fixed():
    """Wrap with retry that targets the 429 type: backoff rides out the spike."""
    resilient = _make_rate_limited_call(fail_times=1).with_retry(
        retry_if_exception_type=(RateLimitError,),
        stop_after_attempt=3,
    )
    return resilient.invoke("summarize the doc")


def run_demo():
    banner("08 — rate limit (429): treat it as a retryable transient error")
    broken = demo_broken()
    print(f"BROKEN (no retry): {type(broken).__name__}: {broken}")
    print(f"FIXED (with_retry): {demo_fixed()!r}")
    print("Proactive option: get_model(rate_limiter=InMemoryRateLimiter(...)) — see header.")


if __name__ == "__main__":
    run_demo()
