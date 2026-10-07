"""Internal path bootstrap shared by ``mock_tools.py`` / ``rollout_worker.py``.

Neither ``AIops-agent/agent`` nor ``data/cold_start`` is an installed
package, so importing the real hook/whitelist logic
(``agent.core.remediation``, ``agent.core.hooks``) and the cold-start
signal-tagging helpers (``data/cold_start/signals.py``) requires putting
their directories on ``sys.path`` first. This mirrors the exact bootstrap
pattern ``data/cold_start/collect_trajectories.py::_ensure_aiops_agent_on_path``
already uses, duplicated here (not imported) because ``verl_adapter``
should depend on stable, public entry points, not on another module's
private (underscore-prefixed) helper.

Not part of the public adapter surface (5 files the task asks for) — this
is small glue used by them, analogous to how ``signals.py`` supports
``collect_trajectories.py`` inside ``data/cold_start/``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # .../aiops-agentic-rl
AIOPS_AGENT_DIR = REPO_ROOT / "AIops-agent"
COLD_START_DIR = REPO_ROOT / "data" / "cold_start"


def ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def ensure_cold_start_on_path() -> None:
    p = str(COLD_START_DIR)
    if COLD_START_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def load_real_hook_modules() -> tuple[Any, Any]:
    """Import the real, pure-Python (no ``claude_agent_sdk`` needed) parts
    of AIops-agent: ``agent.core.remediation`` (the whitelist) and
    ``agent.core.hooks.guard_diagnose_ops`` (the PreToolUse hook). These two
    modules only import ``agent.config`` and stdlib — safe to import in any
    environment, including one with no API key / no docker / no k8s.
    """
    ensure_aiops_agent_on_path()
    from agent.core import remediation  # type: ignore
    from agent.core.hooks import guard_diagnose_ops  # type: ignore

    return remediation, guard_diagnose_ops


def load_signals_module() -> Any:
    """Import ``data/cold_start/signals.py`` (rule-based signal-type /
    repeat-call tagging), the single shared home for that logic per its own
    module docstring.
    """
    ensure_cold_start_on_path()
    import signals  # type: ignore

    return signals
