"""One factory so every example runs offline OR live without code changes.

``get_model()`` returns the deterministic ``FakeChatModel`` when there is no API key (or
when ``LAB_USE_FAKE=1``), and a real ``ChatAnthropic`` when ``ANTHROPIC_API_KEY`` is set.
Examples pass a ``script`` describing what a model *would* plausibly do; offline that
script drives the run, live it is simply ignored and Claude decides for itself.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from .fake_model import FakeChatModel

load_dotenv()  # pick up a local .env if present (gitignored); no-op otherwise

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def use_fake() -> bool:
    """Decide whether to use the offline fake model."""
    flag = os.environ.get("LAB_USE_FAKE", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    return not os.environ.get("ANTHROPIC_API_KEY")


def get_model(
    *,
    temperature: float = 0.0,
    script: list[Any] | None = None,
    default: Any = None,
    **anthropic_kwargs: Any,
):
    """Return a chat model. ``script``/``default`` only affect the offline fake model."""
    if use_fake():
        return FakeChatModel(script=script or [], default=default)
    from langchain_anthropic import ChatAnthropic

    model_name = os.environ.get("LAB_MODEL", DEFAULT_MODEL)
    return ChatAnthropic(model=model_name, temperature=temperature, **anthropic_kwargs)


def get_resilient_model(*, attempts: int = 3, **kwargs: Any):
    """A model wrapped with retries — used by the transient-error / rate-limit demos."""
    return get_model(**kwargs).with_retry(stop_after_attempt=attempts)
