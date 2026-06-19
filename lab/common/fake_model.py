"""A deterministic, scriptable chat model — the linchpin of the lab's offline mode.

Every example and every test can run with ZERO API keys because this model stands in
for Claude. It is *scriptable*: you hand it a list of canned responses and it returns
them one per ``.invoke()`` call, in order. That determinism is what lets us trigger
each tool-use failure mode on demand and assert on the result in CI.

It subclasses the SAME ``BaseChatModel`` that ``langchain-anthropic`` does, and
implements ``bind_tools`` / ``_generate`` / ``_stream`` / ``with_structured_output``, so
it is a drop-in for ``create_react_agent``, ``ToolNode``, ``tools_condition`` and the
structured-output path.

Script entries (consumed FIFO across calls):
  - ``str``                      -> a final-answer AIMessage(content=str)
  - ``AIMessage``                -> used as-is (build tool_calls explicitly via the helpers)
  - a pydantic ``BaseModel``     -> used by the ``with_structured_output`` path
  - ``dict``                     -> coerced into the schema by ``with_structured_output``
  - ``callable(messages)``       -> resolved at call time (e.g. an always-loops responder)

Use the helpers ``tool_call(...)`` / ``parallel_tool_calls(...)`` to script tool use.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr


# --------------------------------------------------------------------------- helpers
def tool_call(name: str, args: dict | None = None, *, id: str | None = None, text: str = "") -> AIMessage:
    """An AIMessage that asks to call one tool — the signal that drives the ReAct loop."""
    return AIMessage(
        content=text,
        tool_calls=[{"name": name, "args": args or {}, "id": id or f"call_{name}", "type": "tool_call"}],
    )


def parallel_tool_calls(specs: Sequence[tuple[str, dict]], *, text: str = "") -> AIMessage:
    """An AIMessage that asks to call several tools in one turn (parallel tool calls)."""
    return AIMessage(
        content=text,
        tool_calls=[
            {"name": n, "args": a or {}, "id": f"call_{i}_{n}", "type": "tool_call"}
            for i, (n, a) in enumerate(specs)
        ],
    )


def _tokenize(text: str) -> list[str]:
    """Split into word-ish tokens (keeping trailing spaces) so streaming looks real."""
    return re.findall(r"\S+\s*", text) or [text]


# --------------------------------------------------------------------------- the model
class FakeChatModel(BaseChatModel):
    """A scripted stand-in for a real chat model. Deterministic by construction."""

    script: list[Any] = []
    # Used when the script is exhausted. A callable -> evaluated each call (e.g. to loop
    # forever). Anything else -> returned verbatim. ``None`` -> a plain "done" message.
    default: Any = None

    _cursor: int = PrivateAttr(default=0)
    _bound_tools: list[Any] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    # -- scripting --------------------------------------------------------------------
    def _next_raw(self, messages: list[BaseMessage]) -> Any:
        if self._cursor < len(self.script):
            entry = self.script[self._cursor]
            self._cursor += 1
        elif self.default is not None:
            entry = self.default
        else:
            entry = "[fake] done"
        if callable(entry) and not isinstance(entry, (AIMessage, BaseMessage, BaseModel)):
            entry = entry(messages)
        return entry

    def _next_message(self, messages: list[BaseMessage]) -> AIMessage:
        entry = self._next_raw(messages)
        if isinstance(entry, AIMessage):
            return entry
        if isinstance(entry, BaseMessage):
            return AIMessage(content=entry.content)
        if isinstance(entry, BaseModel):
            return AIMessage(content=entry.model_dump_json())
        return AIMessage(content=str(entry))

    # -- BaseChatModel API ------------------------------------------------------------
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self._next_message(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        msg = self._next_message(messages)
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        emitted = False
        if content:
            for tok in _tokenize(content):
                if run_manager:
                    run_manager.on_llm_new_token(tok)
                yield ChatGenerationChunk(message=AIMessageChunk(content=tok))
                emitted = True
        if getattr(msg, "tool_calls", None):
            chunks = [
                {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc.get("id"), "index": i}
                for i, tc in enumerate(msg.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            emitted = True
        if not emitted:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))

    # -- tool calling -----------------------------------------------------------------
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        """Record the tools and return a runnable that still routes to our scripted core.

        We don't *use* the tools (responses are scripted), but ``create_react_agent``
        calls ``bind_tools`` and expects a tool-aware runnable back.
        """
        self._bound_tools = list(tools)
        return self.bind(tools=list(tools), **kwargs)

    # -- structured output ------------------------------------------------------------
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        """Return a runnable that emits the next scripted entry coerced into ``schema``."""

        def _run(messages: Any) -> Any:
            entry = self._next_raw(messages if isinstance(messages, list) else [])
            if isinstance(entry, schema) if isinstance(schema, type) else False:
                return entry
            if isinstance(entry, BaseModel):
                return entry
            if isinstance(entry, dict):
                return schema(**entry)
            if isinstance(entry, AIMessage):
                # tolerate a JSON-string answer
                try:
                    return schema(**json.loads(entry.content))
                except Exception:  # pragma: no cover - defensive
                    pass
            raise TypeError(
                f"FakeChatModel.with_structured_output: scripted entry {entry!r} cannot be "
                f"coerced into {getattr(schema, '__name__', schema)}"
            )

        return RunnableLambda(_run)
