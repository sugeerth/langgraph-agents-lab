"""Shared test fixtures. The whole suite runs OFFLINE — we force the fake model and
scrub any real API key so tests never touch the network, even on a machine with a key."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force offline mode for the entire test session.
os.environ["LAB_USE_FAKE"] = "1"
os.environ.pop("ANTHROPIC_API_KEY", None)


def _load_example(dotted: str):
    """Import an example by its dotted path, e.g. 'm1_single_agent.01_bare_state_graph'.

    Example files are numerically prefixed (great for ordering, not importable by name),
    so we load them from disk. They use absolute imports (``from lab.common import ...``),
    which resolve because the project root is on ``sys.path``."""
    rel = dotted.replace(".", "/") + ".py"
    path = ROOT / "lab" / rel
    if not path.exists():
        raise FileNotFoundError(path)
    name = "labex_" + dotted.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def load_example():
    return _load_example
