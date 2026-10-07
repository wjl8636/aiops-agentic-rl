"""Fine-grained credit assignment (doc §5.2 layer 3) — the whole point is
non-zero variance even when outcome reward is identical across a group.
"""
from __future__ import annotations

import pytest

from reward.anti_hacking import group_reward_variance
from reward.credit_assignment import (
    SIGNAL_WEIGHTS_BY_KIND,
    compute_credit_assignment_rewards,
    cumulative_signal_types,
    evidence_value_reward,
)
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import REPEAT_NO_NEW_INFO_PENALTY
from reward.tests.fixtures import (
    _hybrid_wrong_route,
    hybrid_wrong_route_broad_coverage,
    hybrid_wrong_route_narrow_coverage,
)


def test_identical_outcome_but_credit_assignment_reward_differs():
    """The property the design doc is actually built around: two
    trajectories share an identical (wrong) route + identical outcome
    reward, yet the credit-assignment layer must still tell them apart
    because their signal coverage differs.
    """
    broad = hybrid_wrong_route_broad_coverage()
    narrow = hybrid_wrong_route_narrow_coverage()

    # Same wrong route, same everything outcome-reward looks at.
    assert broad.route == narrow.route == "auto_remediated"
    outcome_broad = compute_outcome_rewards(broad)
    outcome_narrow = compute_outcome_rewards(narrow)
    assert outcome_broad["total"] == pytest.approx(outcome_narrow["total"])

    # ...but credit assignment differs because coverage differs. Compare
    # the trajectory totals (each new signal type is now paid once, on the
    # step that introduces it, so the broad trajectory's SUM is what tracks
    # "explored more".).
    credit_broad = compute_credit_assignment_rewards(broad)
    credit_narrow = compute_credit_assignment_rewards(narrow)
    assert sum(credit_broad) != pytest.approx(sum(credit_narrow))

    variance = group_reward_variance([sum(credit_broad), sum(credit_narrow)])
    assert variance > 0.0


def test_evidence_value_reward_marginal_for_resource_kind():
    """Each step's credit == weight of ONLY the signal types this step
    first introduced (not the cumulative sum through this step).
    """
    traj = hybrid_wrong_route_broad_coverage()  # kind="resource"
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    # step0 first covers {prometheus}
    assert evidence_value_reward(traj, 0) == pytest.approx(weights["prometheus"])
    # step1 first covers {kubectl_status}
    assert evidence_value_reward(traj, 1) == pytest.approx(weights["kubectl_status"])
    # step2 first covers {logs}
    assert evidence_value_reward(traj, 2) == pytest.approx(weights["logs"])
    # step3 first covers {git_history} — NOT the cumulative sum through step 3
    assert evidence_value_reward(traj, 3) == pytest.approx(weights["git_history"])


def test_cumulative_signal_types_is_monotonic():
    traj = hybrid_wrong_route_broad_coverage()
    covered_sets = [cumulative_signal_types(traj, i) for i in range(len(traj.steps))]
    for earlier, later in zip(covered_sets, covered_sets[1:]):
        assert earlier <= later


def test_narrow_coverage_credit_zero_after_first_step():
    """Case A: repeated identical tool calls (same signal_type on every
    step) score credit only on the FIRST step; every subsequent repeat
    scores 0 so it can't farm credit off the same signal type.
    """
    traj = hybrid_wrong_route_narrow_coverage()  # 4 steps, all kubectl_status
    rewards = compute_credit_assignment_rewards(traj)
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    assert rewards[0] == pytest.approx(weights["kubectl_status"])
    assert rewards[1] == rewards[2] == rewards[3] == 0.0


def test_twelve_step_repeat_exploit_no_longer_dominates():
    """The exploit the design doc claims is neutralized: a 12-step
    trajectory that loops on the SAME tool call (identical signal_type on
    every step) used to earn ~18 credit under the old cumulative-sum rule
    (12 * w[kubectl_status] = 18.0), swamping the -0.15 repeat penalty.

    With marginal crediting, exactly one step gets paid (the first);
    the remaining 11 repeats score 0 credit — and each still costs
    the -0.15 repeat_no_new_info step-level penalty, so the trace is
    net-negative overall.
    """
    traj = _hybrid_wrong_route([["kubectl_status"]] * 12)
    # Mark steps 1..11 as repeats (as the rollout layer would).
    for step in traj.steps[1:]:
        step.is_repeat_no_new_info = True

    credit = compute_credit_assignment_rewards(traj)
    w_kubectl = SIGNAL_WEIGHTS_BY_KIND["resource"]["kubectl_status"]

    # Before (old cumulative rule): sum(credit) == 12 * 1.5 == 18.0
    # After (marginal rule): sum(credit) == 1 * 1.5 == 1.5
    assert sum(credit) == pytest.approx(w_kubectl)
    assert credit[0] == pytest.approx(w_kubectl)
    assert all(c == 0.0 for c in credit[1:])

    # Combined with repeat penalties (-0.15 * 11 = -1.65) this trace is
    # net-negative on credit + repeat penalties, closing the exploit.
    net = sum(credit) + REPEAT_NO_NEW_INFO_PENALTY * 11
    assert net < 0.0


def test_three_distinct_signals_each_pay_once():
    """Case B: three steps covering three DIFFERENT signal types — each
    step scores exactly its own weight.
    """
    traj = _hybrid_wrong_route(
        [["prometheus"], ["logs"], ["git_history"]]
    )
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    credit = compute_credit_assignment_rewards(traj)
    assert credit[0] == pytest.approx(weights["prometheus"])
    assert credit[1] == pytest.approx(weights["logs"])
    assert credit[2] == pytest.approx(weights["git_history"])


def test_mixed_new_and_repeat_only_new_steps_paid():
    """Case C: alternating new / repeat / new / repeat — only the two
    steps that introduce a new signal type score credit.
    """
    traj = _hybrid_wrong_route(
        [["prometheus"], ["prometheus"], ["logs"], ["logs"]]
    )
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    credit = compute_credit_assignment_rewards(traj)
    assert credit[0] == pytest.approx(weights["prometheus"])
    assert credit[1] == 0.0
    assert credit[2] == pytest.approx(weights["logs"])
    assert credit[3] == 0.0


def test_compute_credit_assignment_rewards_length_matches_steps():
    traj = hybrid_wrong_route_broad_coverage()
    assert len(compute_credit_assignment_rewards(traj)) == len(traj.steps)


def test_total_credit_equals_cumulative_weight_sum():
    """Sanity: summing the marginal per-step credits over the whole
    trajectory equals the weight sum over the final cumulative signal set —
    each newly-covered type contributes its weight exactly once.
    """
    traj = hybrid_wrong_route_broad_coverage()
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    final_covered = cumulative_signal_types(traj, len(traj.steps) - 1)
    expected = sum(weights[t] for t in final_covered)
    assert sum(compute_credit_assignment_rewards(traj)) == pytest.approx(expected)
