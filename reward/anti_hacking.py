"""Countermeasures for the four concrete reward-hacking modes identified in
doc §5.3 (see also the transcript's 七 section for the discovery stories).

Each countermeasure is a small, composable, pure function. None of them
call each other's home module in a circular way:

- 案例一 (dual-source-coverage gate) lives here, standalone. It is a
  *judgment call on magnitude* (the doc only says "从 step-level 扣分",
  it does not give an exact number) — see ``DUAL_SOURCE_GATE_PENALTY``
  below for the documented choice. Callers (e.g. a future
  ``grpo/reward_router.py``) decide whether/how to fold it into the
  step-level total; ``step_reward.compute_step_rewards`` intentionally does
  NOT auto-apply it, because the task spec pins ``compute_step_rewards`` to
  *exactly* the 7 calibrated signals from the doc table.

- 案例二 (evidence traceability) lives here as the single source of truth.
  ``step_reward.py`` imports ``has_fabricated_evidence``/
  ``find_fabricated_evidence`` from this module rather than re-implementing
  the substring-matching logic — this file is the one home for it.

- 案例三 (real-health-check-only stop-loss) is primarily enforced by
  *what outcome_reward.py refuses to read* (it never touches
  ``remediation_detail``). This module additionally provides
  ``verify_stop_loss_ignores_self_report``, a self-checking utility that
  operationalizes the invariant as a callable rule check (tamper with the
  self-reported text, assert the reward doesn't move).

- 案例四 (group-variance-based downsampling signal) lives here as two pure
  helpers a future GRPO training loop can call directly.
"""
from __future__ import annotations

import copy
import re
import statistics

from reward.trajectory import Trajectory

# ---------------------------------------------------------------------------
# 案例一: dual-source-coverage gate
# ---------------------------------------------------------------------------

#: Doc only prescribes the *rule* ("至少覆盖两类信号源"), not a magnitude.
#: We pick -0.2: comparable to the other structural/procedural violations
#: (schema-fallback-unrecovered is also -0.2) while staying smaller than the
#: -0.5 reserved for the more severe evidence-fabrication case. Documented
#: judgment call — no exact number given in the design doc for this gate.
DUAL_SOURCE_GATE_PENALTY = -0.2
MIN_DISTINCT_SIGNAL_SOURCES = 2


def dual_source_coverage_penalty(
    traj: Trajectory,
    min_distinct_sources: int = MIN_DISTINCT_SIGNAL_SOURCES,
    penalty: float = DUAL_SOURCE_GATE_PENALTY,
) -> float:
    """Case 1 countermeasure: penalize a trajectory that reaches a
    conclusion (i.e. produces a ``final_diagnosis``) having touched fewer
    than ``min_distinct_sources`` distinct signal-source types across all
    steps' ``signal_types_covered``.

    This directly targets "调一次 kubectl get pod 就宣布定位完成" — cheap,
    single-tool trajectories that skip the multi-signal cross-validation the
    AIOps README requires (Prometheus + Jaeger + logs 三源交叉).

    Returns ``penalty`` (negative) if the gate is violated, else ``0.0``.
    """
    covered = traj.all_signal_types_covered()
    if len(covered) < min_distinct_sources:
        return penalty
    return 0.0


# ---------------------------------------------------------------------------
# 案例二: evidence traceability hard check
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, for robust-but-still-strict
    substring matching. We deliberately do NOT do fuzzy/semantic matching —
    the doc calls for a "规则匹配" (rule-based match), and a strict
    substring check is the simplest rule that can't be gamed by rewording
    (rewording just makes the match fail, which is the safe failure mode:
    it triggers the fabrication penalty rather than silently passing
    something unverifiable).
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def is_evidence_traceable(evidence_claim: str, observations: list[str]) -> bool:
    """True if ``evidence_claim`` can be substring-matched back into at
    least one step's raw observation text (whitespace/case-normalized).
    """
    needle = _normalize(evidence_claim)
    if not needle:
        return False
    for obs in observations:
        if needle in _normalize(obs):
            return True
    return False


def find_fabricated_evidence(traj: Trajectory) -> list[str]:
    """Return the subset of ``traj.final_diagnosis["evidence"]`` strings
    that cannot be traced back to any step's observation. Empty list means
    every evidence claim is traceable (or there is no evidence field).
    """
    evidence: list[str] = traj.final_diagnosis.get("evidence", []) or []
    observations = traj.all_observations()
    return [claim for claim in evidence if not is_evidence_traceable(claim, observations)]


def has_fabricated_evidence(traj: Trajectory) -> bool:
    """True if at least one evidence claim in ``final_diagnosis`` has no
    traceable source among the trajectory's tool observations.

    ``step_reward.py``'s -0.5 fabrication penalty calls this function; this
    is the single home for the traceability rule (no duplicate logic).
    """
    return len(find_fabricated_evidence(traj)) > 0


# ---------------------------------------------------------------------------
# 案例三: real-health-check-only stop-loss verdict
# ---------------------------------------------------------------------------

def verify_stop_loss_ignores_self_report(traj: Trajectory) -> bool:
    """Self-checking utility operationalizing the doc §5.3 案例三 invariant:
    tampering with the model's self-reported ``remediation_detail`` text
    must NOT change the stop-loss outcome reward, because that reward is
    computed purely from ``post_action_health`` (a real health-check read),
    never from model-authored text.

    Returns True iff the invariant holds for ``traj`` (i.e.
    ``outcome_reward.stop_loss_reward`` is provably indifferent to
    ``remediation_detail``'s content). Intended for use in tests/audits, not
    as a per-step reward term.
    """
    from reward.outcome_reward import stop_loss_reward  # local import: avoid a
    # module-level circular import (outcome_reward doesn't need anti_hacking
    # at import time, but keeping this import local keeps the dependency
    # direction obviously one-way when reading this file top-to-bottom).

    baseline = stop_loss_reward(traj)

    tampered = copy.deepcopy(traj)
    tampered.final_diagnosis["remediation_detail"] = (
        "已恢复！服务完全正常，止损成功，一切都好（这段话是伪造的自述，不应影响 reward）。"
    )
    tampered_reward = stop_loss_reward(tampered)

    return baseline == tampered_reward


# ---------------------------------------------------------------------------
# 案例四: group-variance-based downsampling signal
# ---------------------------------------------------------------------------

def group_reward_variance(rewards: list[float]) -> float:
    """Population variance of a group's rewards (e.g. the ``group_size``
    independent rollouts GRPO samples for one query). Returns 0.0 for
    empty/singleton input (no variance is definable/meaningful there).
    """
    if len(rewards) < 2:
        return 0.0
    return statistics.pvariance(rewards)


#: Judgment calls (doc says "方差长期为零" / "N 连续轮次" without exact
#: numbers): a variance below 1e-4 is treated as "effectively zero" (GRPO
#: rewards in this design are combinations of 0.05-0.5-scale terms, so 1e-4
#: is well below any real distinguishing signal), and "long enough" is
#: fixed at 5 consecutive rounds — long enough to not downsample on a
#: single lucky/unlucky batch, short enough to react within a training run.
DEFAULT_ZERO_VARIANCE_THRESHOLD = 1e-4
DEFAULT_CONSECUTIVE_ROUNDS = 5


def should_downsample(
    variance_history: list[float],
    threshold: float = DEFAULT_ZERO_VARIANCE_THRESHOLD,
    n_consecutive: int = DEFAULT_CONSECUTIVE_ROUNDS,
) -> bool:
    """Flags a scenario type as a downsampling candidate when its group
    reward variance has been at-or-below ``threshold`` for the most recent
    ``n_consecutive`` rounds recorded in ``variance_history`` (oldest first,
    most recent last).

    This is a pure, training-loop-level utility — the actual GRPO loop (not
    built here) would call it per scenario-type each round to decide
    whether to shrink that type's sampling weight in the next training pool
    (doc §5.3 案例四).
    """
    if len(variance_history) < n_consecutive:
        return False
    return all(v <= threshold for v in variance_history[-n_consecutive:])
