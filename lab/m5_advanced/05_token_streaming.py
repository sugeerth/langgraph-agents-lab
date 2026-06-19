"""Token streaming is just a stream mode.

CONCEPT: to render an answer word-by-word (the ChatGPT typing effect), you don't need a
special API — you use ``stream_mode="messages"`` and iterate the (chunk, metadata) pairs.
Each ``chunk`` is an ``AIMessageChunk`` carrying a slice of ``.content``; concatenating the
slices rebuilds the full answer. (Our fake model streams word-by-word; a real model streams
sub-word tokens — the code is identical.)

aha: token streaming is just ``stream_mode="messages"`` — no extra machinery.
"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.prebuilt import create_react_agent

from lab.common import banner, get_model


def build_graph():
    """A TEXT-ONLY agent (no tools) so every chunk is part of the final answer."""
    model = get_model(script=["LangGraph streams tokens one piece at a time."], default="ok")
    return create_react_agent(model, [])


def stream_tokens():
    """Return the list of non-empty token strings, in arrival order."""
    agent = build_graph()
    tokens = []
    for chunk, _metadata in agent.stream(
        {"messages": [HumanMessage("explain streaming")]}, stream_mode="messages"
    ):
        # Only AI content chunks are tokens of the answer (ignore other message types).
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            tokens.append(chunk.content)
    return tokens


def run_demo():
    banner('Token streaming via stream_mode="messages"')
    print("assistant: ", end="", flush=True)
    tokens = stream_tokens()
    for tok in tokens:
        print(tok, end="", flush=True)  # arrives token-by-token in a real run
    print(f"\n\n({len(tokens)} tokens; rejoined = {''.join(tokens)!r})")
    return tokens


if __name__ == "__main__":
    run_demo()
