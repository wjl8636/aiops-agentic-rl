"""GRPO reward router: composes ``reward/``'s already-complete step / credit-
assignment / outcome / anti-hacking functions into the single reward signal
a GRPO training loop actually consumes for ONE ``Trajectory`` or one GRPO
group (``group_size`` independent rollouts of the same query).

This module implements NO reward *logic* of its own for the three layers
(step-level dense, outcome-level, credit-assignment) — those are complete,
tested, and owned by the sibling ``reward/`` package (see
``reward/step_reward.py``, ``reward/outcome_reward.py``,
``reward/credit_assignment.py``, ``reward/anti_hacking.py``). This module's
only job is the *composition rule*: which numbers get added to which steps,
in what order.

Composition, in one place (see each function's docstring for the "why"):

1. Per-step dense reward = ``compute_step_rewards(traj)`` ELEMENT-WISE ADDED
   to ``compute_credit_assignment_rewards(traj)`` (doc §5.2 layer 1 + layer
   3; layer 3 exists specifically because layer 1 + outcome reward alone
   can still collapse a GRPO group's variance to zero — see
   ``reward/credit_assignment.py``'s module docstring — so it SUPPLEMENTS,
   never replaces, the layer-1 signal). See ``compute_per_step_rewards``.
2. Outcome-level reward = ``compute_outcome_rewards(traj)["total"]``,
   attached ONCE, to the trajectory's LAST step (standard GRPO/episodic-RL
   practice: a terminal/outcome reward prices the whole episode's result
   and is credited at the terminal step, not distributed across steps —
   see ``compute_trajectory_reward`` for the explicit justification).
3. Anti-hacking: ``dual_source_coverage_penalty`` (doc §5.3 案例一) is a
   trajectory-wide judgment (reads the union of signal types over ALL
   steps), so — exactly like ``compute_step_rewards``'s own trajectory-wide
   penalties (evidence fabrication, unrecovered schema fallback) — it is
   folded into the LAST step alongside the outcome reward.
4. Two router-level additions that live HERE (not in ``reward/``) because
   nothing in ``reward/`` implements them and the design doc requires them
   for a GRPO loop specifically (doc §5.1 "超过 12 轮还没收敛的截断给负奖"
   and §5.4 "调用轮数惩罚：超过 8 轮工具调用后每多一轮小幅扣分"): see
   ``turn_count_penalty`` and ``max_turns_truncation_penalty``. Both are
   documented judgment calls on magnitude (the doc gives the *rule*, not an
   exact number, exactly like ``anti_hacking.DUAL_SOURCE_GATE_PENALTY``).
5. ``compute_group_rewards`` batches the above over one GRPO group (doc
   §5.1: group_size = 6 independent rollouts per query) and reports the
   group's ``group_reward_variance`` (``reward.anti_hacking``), which is
   what a GRPO advantage-computation step and the §5.3 案例四 downsampling
   policy both need.

Explicitly NOT this module's job (doc §5.4, verified against
``reward/outcome_reward.py``'s own boundary):
- KL penalty against the SFT policy — that's applied by the veRL trainer
  itself (KL between policy and a frozen reference model), not by a reward
  *function*; see ``grpo/verl_config/aiops_grpo.yaml``'s
  ``actor_rollout_ref.actor.use_kl_loss``/``kl_loss_coef``.
- LLM-judge coherence — ``reward.outcome_reward.compute_outcome_rewards``
  already excludes it from ``"total"`` and asserts that boundary; this
  module reads ``"total"`` and never touches ``llm_judge_coherence`` itself,
  so the exclusion propagates here for free.
"""
from __future__ import annotations

from typing import Any

from reward.anti_hacking import (
    dual_source_coverage_penalty,
    group_reward_variance,
    should_downsample,
)
from reward.credit_assignment import compute_credit_assignment_rewards
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Router-level additions (doc §5.1 / §5.4) — NOT implemented anywhere in
# reward/, because reward/step_reward.py is deliberately pinned to "exactly
# the 7 calibrated signals from the doc table" (see its module docstring).
# Turn-budget policing is a GRPO-loop-level concern, so it lives here.
# ---------------------------------------------------------------------------

#: doc §5.4: "超过 8 轮工具调用后每多一轮小幅扣分" — turn count beyond which
#: each additional turn is nudged down. No exact threshold given beyond "8";
#: taken literally from the doc text.
TURN_COUNT_PENALTY_THRESHOLD = 8

#: "小幅扣分" (small deduction) — no magnitude given. Documented judgment
#: call: one order of magnitude below the smallest calibrated step_reward.py
#: term (+0.05 tool-schema-valid), so it nudges the "stop stalling" incentive
#: without ever being able to outweigh a single step's real dense signal.
TURN_COUNT_PENALTY_PER_EXTRA_TURN = -0.02

#: doc §5.1: "最长交互 12 轮，超过 12 轮还没收敛的截断给负奖" — hard turn
#: budget. Matches ``grpo/verl_config/aiops_grpo.yaml``'s
#: ``multi_turn.max_assistant_turns`` and is covered by the YAML's own shape
#: test (``grpo/tests/test_verl_config.py``) so the two numbers can't drift.
MAX_TURNS = 12

#: Doc gives no magnitude for the truncation penalty either. Documented
#: judgment call: pinned to the same magnitude as the single most severe
#: existing penalty (evidence fabrication, -0.5 in step_reward.py) — failing
#: to converge within the full turn budget is treated as comparably severe
#: a process failure, not a minor style issue.
MAX_TURNS_TRUNCATION_PENALTY = -0.5


def turn_count_penalty(traj: Trajectory) -> float:
    """Doc §5.4 turn-count penalty: 0 at/below the threshold, else
    ``(extra turns) * TURN_COUNT_PENALTY_PER_EXTRA_TURN`` (negative).
    """
    n = len(traj.steps)
    if n <= TURN_COUNT_PENALTY_THRESHOLD:
        return 0.0
    return (n - TURN_COUNT_PENALTY_THRESHOLD) * TURN_COUNT_PENALTY_PER_EXTRA_TURN


def max_turns_truncation_penalty(traj: Trajectory) -> float:
    """Doc §5.1 hard-truncation penalty: flat ``MAX_TURNS_TRUNCATION_PENALTY``
    if the trajectory ran past ``MAX_TURNS`` steps, else 0.

    Note the *decision* to stop generating at 12 turns is a rollout-layer
    concern (``verl_adapter/``, not built here) — by the time a
    ``Trajectory`` reaches this router it is already a finished rollout.
    This function only prices the fact that it took more than the budget,
    for whatever finished trajectory the rollout layer handed us (e.g. one
    that was allowed to run long, or one truncated at exactly 12 that we
    still want to price as "didn't converge in budget").
    """
    return MAX_TURNS_TRUNCATION_PENALTY if len(traj.steps) > MAX_TURNS else 0.0


def compute_per_step_rewards(traj: Trajectory) -> list[float]:
    """Layer 1 + layer 3, element-wise added — see module docstring point 1.

    Both ``compute_step_rewards`` and ``compute_credit_assignment_rewards``
    are already ``len(traj.steps)``-long lists over the exact same step
    sequence, so summing element-wise needs no alignment logic. Returns an
    empty list for a zero-step trajectory (mirrors both inputs).
    """
    step = compute_step_rewards(traj)
    credit = compute_credit_assignment_rewards(traj)
    assert len(step) == len(credit) == len(traj.steps), (
        "compute_step_rewards and compute_credit_assignment_rewards must both "
        f"return one value per step; got {len(step)} and {len(credit)} for "
        f"{len(traj.steps)} steps"
    )
    return [s + c for s, c in zip(step, credit)]


def compute_trajectory_reward(traj: Trajectory) -> dict[str, Any]:
    """Full, GRPO-ready reward breakdown for ONE trajectory.

    Returns a dict with:
      - ``per_step_rewards``: ``list[float]`` of length ``len(traj.steps)``
        (or a single synthetic element for a zero-step trajectory — see
        below), with every trajectory-wide/terminal term (outcome reward,
        dual-source-coverage penalty, turn-count penalty, max-turns
        truncation penalty) already folded into the LAST element, so
        ``sum(per_step_rewards) == total_reward`` always holds. This is the
        list a GRPO loop would attach to the token/step-level advantage
        computation.
      - ``total_reward``: the scalar sum, for GRPO's group-relative
        advantage computation (``compute_group_rewards`` below is the
        group-level wrapper around this).
      - ``outcome_breakdown``: the raw ``compute_outcome_rewards`` dict, for
        logging/debugging (includes the log-only ``llm_judge_coherence``,
        which is NOT part of ``total_reward`` — see module docstring).
      - ``dual_source_coverage_penalty`` / ``turn_count_penalty`` /
        ``max_turns_truncation_penalty``: the individual terminal
        adjustments, broken out for logging/debugging/tests.

    Zero-step trajectories: ``compute_per_step_rewards`` returns ``[]`` for
    these (nothing to attach a terminal reward to), but GRPO still needs a
    scalar reward for that rollout, so we represent it as a single
    synthetic element containing just the terminal terms. This is a
    documented edge case, not something the design doc addresses directly.
    """
    per_step = compute_per_step_rewards(traj)

    outcome = compute_outcome_rewards(traj)
    outcome_total = float(outcome["total"])
    dual_source_penalty = dual_source_coverage_penalty(traj)
    turn_penalty = turn_count_penalty(traj)
    truncation_penalty = max_turns_truncation_penalty(traj)

    terminal_addition = outcome_total + dual_source_penalty + turn_penalty + truncation_penalty

    if per_step:
        per_step[-1] += terminal_addition
    else:
        per_step = [terminal_addition]

    total_reward = sum(per_step)

    return {
        "per_step_rewards": per_step,
        "total_reward": total_reward,
        "outcome_breakdown": outcome,
        "dual_source_coverage_penalty": dual_source_penalty,
        "turn_count_penalty": turn_penalty,
        "max_turns_truncation_penalty": truncation_penalty,
    }


def compute_group_rewards(trajectories: list[Trajectory]) -> dict[str, Any]:
    """GRPO group-level reward: one call per group of ``group_size``
    (doc §5.1: 6) independent rollouts sampled for the SAME query.

    This is the calling convention referenced by
    ``grpo/verl_config/aiops_grpo.yaml``'s ``custom_reward_function`` entry
    (``grpo.reward_router:compute_group_rewards`` — see that file's header
    comment for exactly how this maps, and does NOT map cleanly, onto real
    veRL's per-SAMPLE ``compute_score(data_source, solution_str,
    ground_truth, extra_info)`` signature; that mismatch is flagged there,
    not glossed over here).

    Returns:
      - ``trajectory_rewards``: list of this group's per-trajectory
        ``compute_trajectory_reward`` breakdowns, same order as input.
      - ``total_rewards``: just the scalar totals, same order as input —
        what a GRPO advantage step actually subtracts the group mean from.
      - ``group_reward_variance``: ``reward.anti_hacking.group_reward_variance``
        of ``total_rewards`` — the group-collapse diagnostic (doc §5.3
        案例四); 0.0 for a group of size < 2.
      - ``group_size``: ``len(trajectories)``, for sanity-checking against
        the configured ``group_size`` (doc §5.1: 6).
    """
    per_traj = [compute_trajectory_reward(t) for t in trajectories]
    total_rewards = [r["total_reward"] for r in per_traj]
    return {
        "trajectory_rewards": per_traj,
        "total_rewards": total_rewards,
        "group_reward_variance": group_reward_variance(total_rewards),
        "group_size": len(trajectories),
    }


def route_trajectory_reward_to_variance_tracker(
    scenario_type: str,
    group_total_rewards: list[float],
    variance_history: dict[str, list[float]],
) -> dict[str, Any]:
    """Pluggable hook: feed one finished GRPO group's total rewards into the
    per-scenario-type variance history a training loop maintains, and
    report whether that scenario type is now a downsampling candidate (doc
    §5.3 案例四: "训练日志里盯每一组的 reward 方差，方差长期为零的场景类型从
    训练池里降采样").

    This function is NOT the training loop — it doesn't own persisted
    state, doesn't mutate its inputs, and doesn't decide sampling weights.
    It is the pure, testable "what should happen to the tracked history
    this round" step a training loop would call once per finished group:

        history = {}  # owned by the training loop, persisted across rounds
        for group in finished_groups_this_round:
            result = route_trajectory_reward_to_variance_tracker(
                group.scenario_type, group.total_rewards, history,
            )
            history = result["updated_variance_history"]
            if result["should_downsample"]:
                training_loop.shrink_sampling_weight(group.scenario_type)

    Args:
        scenario_type: the fault-kind/scenario bucket this group belongs to
            (doc's granularity: e.g. one of the four root-cause kinds, or
            the hybrid/capstone bucket — the training loop's choice, this
            function is agnostic to the exact taxonomy).
        group_total_rewards: this round's ``compute_group_rewards(...)
            ["total_rewards"]`` for that scenario type's group.
        variance_history: the CURRENT mapping of scenario_type -> list of
            past group variances (oldest-first). Not mutated in place — a
            new dict (with this round's variance appended under
            ``scenario_type``) is returned instead, so callers can hold
            onto the old value for comparison/rollback if they want.

    Returns a dict with ``scenario_type``, this round's ``variance``, the
    ``updated_variance_history`` (new dict, safe to keep as next round's
    input), and ``should_downsample`` (``reward.anti_hacking
    .should_downsample`` evaluated on the updated history for this
    scenario type).
    """
    variance = group_reward_variance(group_total_rewards)
    updated_history = dict(variance_history)
    updated_history[scenario_type] = list(variance_history.get(scenario_type, [])) + [variance]
    downsample = should_downsample(updated_history[scenario_type])
    return {
        "scenario_type": scenario_type,
        "variance": variance,
        "updated_variance_history": updated_history,
        "should_downsample": downsample,
    }
