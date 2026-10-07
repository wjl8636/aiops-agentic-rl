"""Loads the sibling ``verl_adapter/`` task's mock rollout fixtures
(``data/cold_start/trajectories/mock/*.json``) into real
``reward.trajectory.Trajectory`` objects, for ``grpo/tests/`` to exercise
``grpo/reward_router.py`` against.

This is a thin FIELD-MAPPING layer, not a second parser: every ``Step``/
``Trajectory`` is built by calling the real dataclass constructors from
``reward/trajectory.py`` (so their ``__post_init__`` validation still runs).
The mapping exists only because the mock JSON's on-disk shape — produced by
the parallel ``verl_adapter/`` task, not by this task — diverges from the
``Trajectory``/``Step`` contract in three small, mechanical ways:

1. ``ground_truth: null`` in every current mock fixture, but
   ``Trajectory.ground_truth`` is a required ``dict`` (not ``Optional``) —
   mapped to ``{}``.
2. ``post_action_health`` on disk is shaped like a raw health-check-tool
   record (``{"checked", "healthy", "metric", "window_s"}``), not the
   contract's ``{"healthy_at_60s", "new_alert_triggered"}`` — mapped 1:1
   (``healthy`` -> ``healthy_at_60s``), with ``new_alert_triggered`` always
   ``False`` since the mock format has no such field and every current
   fixture is a simple "recovered or not" scenario, never a "recovered but
   also paged a new alert" one.
3. Top-level ``alert_id`` / ``meta`` keys are dropped — they're rollout
   provenance/bookkeeping, not part of the ``Trajectory`` contract.

If a future mock fixture's JSON shape changes, update the mapping here
rather than each test.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from reward.trajectory import Step, Trajectory

#: grpo/tests/mock_trajectories.py -> grpo/tests -> grpo -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_TRAJECTORY_DIR = REPO_ROOT / "data" / "cold_start" / "trajectories" / "mock"

_STEP_FIELDS = {
    "tool_name",
    "tool_input",
    "hook_decision",
    "observation",
    "hook_action",
    "hook_target",
    "signal_types_covered",
    "is_repeat_no_new_info",
    "tool_call_schema_valid",
}


def _step_from_dict(raw_step: dict[str, Any]) -> Step:
    """``Step(**raw_step)`` restricted to ``Step``'s real fields — every key
    in the mock JSON step dicts happens to already match a real ``Step``
    constructor kwarg 1:1, so no renaming is needed here (unlike the
    trajectory-level mapping above).
    """
    return Step(**{k: v for k, v in raw_step.items() if k in _STEP_FIELDS})


def _post_action_health_from_dict(raw_health: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw_health is None:
        return None
    return {
        "healthy_at_60s": bool(raw_health.get("healthy", False)),
        "new_alert_triggered": False,
    }


def load_mock_trajectory(path: Path) -> Trajectory:
    """Load one mock rollout JSON file into a real ``Trajectory``."""
    raw = json.loads(path.read_text())
    steps = [_step_from_dict(s) for s in raw["steps"]]
    return Trajectory(
        alert=raw["alert"],
        steps=steps,
        final_diagnosis=raw["final_diagnosis"],
        ground_truth=raw.get("ground_truth") or {},
        route=raw["route"],
        post_action_health=_post_action_health_from_dict(raw.get("post_action_health")),
    )


def load_all_mock_trajectories() -> dict[str, Trajectory]:
    """``{alert_id_stem: Trajectory}`` for every ``*.json`` fixture in
    ``data/cold_start/trajectories/mock/``, keyed by filename stem so tests
    can refer to them by name (e.g. ``"mock_fail_evidence"``).
    """
    return {p.stem: load_mock_trajectory(p) for p in sorted(MOCK_TRAJECTORY_DIR.glob("*.json"))}


def with_signal_coverage(traj: Trajectory, coverage_per_step: list[list[str]]) -> Trajectory:
    """Return a DEEP COPY of ``traj`` with ``steps[i].signal_types_covered``
    overwritten per ``coverage_per_step`` (one list per step, same length as
    ``traj.steps``). Everything else (route, final_diagnosis, ground_truth,
    post_action_health, hook decisions, observations) is left untouched.

    Used to build controlled variance-preservation fixtures: several copies
    of the SAME loaded mock trajectory that differ ONLY in which
    observability signal types each step is tagged as having covered, so
    that outcome-level reward (which depends on final_diagnosis/route/
    ground_truth/post_action_health, none of which changed) is provably
    IDENTICAL across the copies, isolating credit-assignment's
    contribution to any resulting reward variance.
    """
    if len(coverage_per_step) != len(traj.steps):
        raise ValueError(
            f"coverage_per_step must have one entry per step "
            f"({len(traj.steps)} steps, got {len(coverage_per_step)} entries)"
        )
    out = copy.deepcopy(traj)
    for step, types in zip(out.steps, coverage_per_step):
        step.signal_types_covered = list(types)
    return out
