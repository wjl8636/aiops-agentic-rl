"""Step-level dense reward (doc §5.2, layer 1).

Implements exactly the 7 signals/magnitudes from the design doc's table —
no more, no fewer, so the calibrated numbers stay locked in and testable:

| signal                                                | magnitude |
|--------------------------------------------------------|-----------|
| tool call schema valid                                 | +0.05     |
| whitelist-hit-and-authorized (hook allow)               | +0.10     |
| whitelist-hit-but-unauthorized / write-deny (hook deny) | -0.15     |
| observation advances diagnosis                          | +0.10     |
| repeat call, no new info                                | -0.15     |
| evidence fabrication (final)                            | -0.50     |
| schema-fallback triggered, unrecovered (final)          | -0.20     |

The first five are genuinely per-step (computed independently for each
``Step``). The last two (evidence fabrication, schema fallback) are
trajectory-wide judgments — there is exactly one such flag per trajectory,
not one per step — so ``compute_step_rewards`` folds them into the reward of
the *last* step, which is where "the conclusion was reached" lives. This
keeps ``compute_step_rewards``'s output a clean ``list[float]`` of length
``len(traj.steps)`` that sums to the trajectory's total step-level reward,
which is what a GRPO training loop wants to add up per token/step.
"""
from __future__ import annotations

from reward.anti_hacking import has_fabricated_evidence
from reward.trajectory import Step, Trajectory

TOOL_SCHEMA_VALID_REWARD = 0.05
WHITELIST_ALLOW_AUTHORIZED_REWARD = 0.1
WHITELIST_DENY_PENALTY = -0.15
OBSERVATION_ADVANCES_DIAGNOSIS_REWARD = 0.1
REPEAT_NO_NEW_INFO_PENALTY = -0.15
EVIDENCE_FABRICATION_PENALTY = -0.5
SCHEMA_FALLBACK_UNRECOVERED_PENALTY = -0.2


def tool_call_schema_valid_reward(step: Step) -> float:
    """+0.05 when the SDK successfully parsed this step's tool-call schema."""
    return TOOL_SCHEMA_VALID_REWARD if step.tool_call_schema_valid else 0.0


def whitelist_allow_authorized_reward(step: Step) -> float:
    """+0.1 when the PreToolUse hook allowed this call — which, per
    ``remediation.classify_command``'s contract, only happens when the
    command matched the low-risk whitelist AND the target was authorized.
    """
    return WHITELIST_ALLOW_AUTHORIZED_REWARD if step.hook_decision == "allow" else 0.0


def whitelist_deny_penalty(step: Step) -> float:
    """-0.15 when the PreToolUse hook denied this call. Covers both
    sub-cases the doc groups under one magnitude: whitelist-hit-but-
    unauthorized-target (``classify_command`` returned "deny") and
    hit-a-write/destructive-deny-list (``_READONLY_DENY``/
    ``_WRITE_FALLBACK_DENY`` regex fallback in ``hooks.py``) — both surface
    identically as ``step.hook_decision == "deny"`` in the trajectory
    contract.
    """
    return WHITELIST_DENY_PENALTY if step.hook_decision == "deny" else 0.0


def observation_advances_diagnosis_reward(step: Step, prior_steps: list[Step]) -> float:
    """+0.1 when this step's observation "实质收窄" the suspect scope.

    Judgment call on how to detect "advances" from the trajectory contract
    (the doc describes the *intent* — "定位到具体 commit/文件/服务" or
    "指标从未知变已量化" — but doesn't give a field-level rule). We treat a
    step as advancing the diagnosis when both hold:
      1. the rollout layer did NOT flag it as ``is_repeat_no_new_info``, and
      2. it covers at least one signal type not covered by any earlier step
         in this trajectory (i.e. it contributes genuinely new evidence
         category, not just more of what we already had).

    This reuses fields already in the contract (no new rollout-layer work
    required) and is directionally exactly what the doc wants: cheap,
    repeated re-reads of the same signal type don't advance anything;
    a step that's the first to bring in e.g. Jaeger trace data does.
    """
    if step.is_repeat_no_new_info:
        return 0.0
    already_covered: set[str] = set()
    for s in prior_steps:
        already_covered.update(s.signal_types_covered)
    new_types = set(step.signal_types_covered) - already_covered
    return OBSERVATION_ADVANCES_DIAGNOSIS_REWARD if new_types else 0.0


def repeat_no_new_info_penalty(step: Step) -> float:
    """-0.15 when the rollout layer determined this call repeats an earlier
    (tool, target) pair without yielding new information.
    """
    return REPEAT_NO_NEW_INFO_PENALTY if step.is_repeat_no_new_info else 0.0


def evidence_fabrication_penalty(traj: Trajectory) -> float:
    """-0.5, flat, once per trajectory, if ANY claim in
    ``final_diagnosis["evidence"]`` cannot be traced back to a step
    observation. Flat (not -0.5 per fabricated claim) because that's the
    magnitude the doc's table pins for this row — one calibrated number per
    trajectory, not a per-claim multiplier.

    Delegates the traceability rule to ``anti_hacking.has_fabricated_evidence``
    (single home for that logic — see anti_hacking.py's module docstring).
    """
    return EVIDENCE_FABRICATION_PENALTY if has_fabricated_evidence(traj) else 0.0


def schema_fallback_unrecovered_penalty(traj: Trajectory) -> float:
    """-0.2, once per trajectory, when ``final_diagnosis`` carries the
    sibling flag ``schema_fallback_triggered=True`` (an unrecovered
    ``Diagnosis.fallback()`` occurred mid-trajectory).
    """
    return SCHEMA_FALLBACK_UNRECOVERED_PENALTY if traj.schema_fallback_triggered else 0.0


def compute_step_rewards(traj: Trajectory) -> list[float]:
    """Per-step step-level reward, ``len(result) == len(traj.steps)``.

    Each step's reward = the four genuinely-per-step signals for that step
    (schema valid, whitelist allow, whitelist deny, observation-advances,
    repeat-no-new-info — note a step can score at most one of
    allow/deny since hook_decision is a single value, and independently at
    most one of advances/repeat since those are also mutually exclusive by
    construction of ``is_repeat_no_new_info``). The two trajectory-wide
    penalties (evidence fabrication, unrecovered schema fallback) are added
    onto the LAST step only, since they are judgments about the trajectory's
    conclusion, not about any individual mid-trajectory tool call. Returns
    an empty list for a trajectory with no steps.
    """
    rewards: list[float] = []
    for i, step in enumerate(traj.steps):
        prior = traj.steps[:i]
        r = (
            tool_call_schema_valid_reward(step)
            + whitelist_allow_authorized_reward(step)
            + whitelist_deny_penalty(step)
            + observation_advances_diagnosis_reward(step, prior)
            + repeat_no_new_info_penalty(step)
        )
        rewards.append(r)

    if rewards:
        rewards[-1] += evidence_fabrication_penalty(traj) + schema_fallback_unrecovered_penalty(traj)

    return rewards
