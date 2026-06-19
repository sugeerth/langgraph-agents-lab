#!/usr/bin/env bash
# Convenience runner for the lab. Everything runs OFFLINE by default (no API key).
#
#   ./run.sh setup                                  create .venv + install
#   ./run.sh test                                   run the full pytest suite (offline)
#   ./run.sh m2_tools_and_failures.09_infinite_loop_recursion   run one example
#   ./run.sh list                                   list runnable examples
#
# To use the real Claude path instead of the fake model:  export ANTHROPIC_API_KEY=...
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"

case "${1:-help}" in
  setup)
    if command -v uv >/dev/null 2>&1; then
      uv venv .venv --python 3.12
      uv pip install --python .venv/bin/python -e ".[dev]"
    else
      python3 -m venv .venv
      ./.venv/bin/python -m pip install -U pip
      ./.venv/bin/python -m pip install -e ".[dev]"
    fi
    echo "Setup complete. Try: ./run.sh test"
    ;;
  test)
    PYTHONPATH=. "$PY" -m pytest "${@:2}"
    ;;
  list)
    find lab -name '[0-9][0-9]_*.py' | sed 's#lab/##; s#/#.#; s#\.py$##' | sort
    ;;
  help|-h|--help)
    sed -n '2,12p' "$0"
    ;;
  *)
    # treat the arg as module.file, e.g. m1_single_agent.01_bare_state_graph
    rel="lab/${1//.//}.py"
    if [[ ! -f "$rel" ]]; then
      echo "No such example: $rel" >&2
      echo "Run './run.sh list' to see available examples." >&2
      exit 1
    fi
    PYTHONPATH=. "$PY" "$rel"
    ;;
esac
