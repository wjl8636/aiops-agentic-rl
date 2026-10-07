"""Outcome-level reward — locks in exact magnitudes and the FixResult.verified
boundary (doc §7.2 / §5.2 layer 2).
"""
from __future__ import annotations

import copy

import pytest

from reward.outcome_reward import (
    HANDOFF_FIELD_REWARD,
    ROOT_CAUSE_FIELD_REWARD,
    ROUTE_MATCH_REWARD,
    STOP_LOSS_NEW_ALERT_PENALTY,
    STOP_LOSS_NOT_RECOVERED_REWARD,
    STOP_LOSS_SUCCESS_REWARD,
    compute_outcome_rewards,
    handoff_reward,
    llm_judge_coherence,
    root_cause_field_rewards,
    root_cause_reward,
    route_match_reward,
    stop_loss_reward,
)
from reward.tests.fixtures import (
    clean_dependency_success,
    fake_stop_loss_case,
    hybrid_wrong_route_broad_coverage,
    unauthorized_command_deny_case,
)


def test_magnitudes_match_doc_table():
    assert STOP_LOSS_SUCCESS_REWARD == 0.5
    assert STOP_LOSS_NOT_RECOVERED_REWARD == 0.0
    assert STOP_LOSS_NEW_ALERT_PENALTY == -0.3
    assert ROOT_CAUSE_FIELD_REWARD == 0.1
    assert ROUTE_MATCH_REWARD == 0.2
    assert HANDOFF_FIELD_REWARD == 0.05


def test_clean_dependency_case_exact_total():
    traj = clean_dependency_success()
    breakdown = compute_outcome_rewards(traj)
    assert breakdown["stop_loss_reward"] == pytest.approx(0.5)
    assert breakdown["root_cause_reward"] == pytest.approx(0.3)  # 3 fields all match
    assert breakdown["route_match_reward"] == pytest.approx(0.2)
    assert breakdown["handoff_reward"] == pytest.approx(0.0)  # not code_fix/hybrid
    assert breakdown["total"] == pytest.approx(1.0)


def test_stop_loss_ignores_self_reported_text_and_reads_real_health_check():
    """案例三: remediation_detail claims '已恢复' but the REAL health check
    says healthy_at_60s=False -> must score as failure (0.0), NOT success.
    """
    traj = fake_stop_loss_case()
    assert "已恢复" in traj.final_diagnosis["remediation_detail"]
    reward = stop_loss_reward(traj)
    assert reward == pytest.approx(STOP_LOSS_NOT_RECOVERED_REWARD)
    assert reward != STOP_LOSS_SUCCESS_REWARD


def test_stop_loss_new_alert_triggered_is_penalized():
    traj = fake_stop_loss_case()
    traj.post_action_health = {"healthy_at_60s": False, "new_alert_triggered": True}
    assert stop_loss_reward(traj) == pytest.approx(-0.3)


def test_stop_loss_not_applicable_for_non_online_op():
    traj = unauthorized_command_deny_case()  # remediation_type == "info_only"
    assert stop_loss_reward(traj) is None


def test_root_cause_partial_credit():
    traj = clean_dependency_success()
    traj.final_diagnosis["suspect_service"] = "wrong-service"
    per_field = root_cause_field_rewards(traj)
    assert per_field["suspect_service"] == 0.0
    assert per_field["kind"] == pytest.approx(0.1)
    assert per_field["remediation_type"] == pytest.approx(0.1)
    assert root_cause_reward(traj) == pytest.approx(0.2)


def test_route_mismatch_scores_zero():
    traj = hybrid_wrong_route_broad_coverage()
    assert traj.route != traj.ground_truth["route"]
    assert route_match_reward(traj) == 0.0


def test_handoff_scored_for_hybrid_scenario():
    traj = hybrid_wrong_route_broad_coverage()  # also_code_fix=True, all 3 handoff fields correct
    assert handoff_reward(traj) == pytest.approx(0.15)


def test_handoff_not_applicable_for_pure_online_op():
    traj = clean_dependency_success()
    traj.final_diagnosis["suspect_repo"] = "some-repo"  # even if present, shouldn't be scored
    assert handoff_reward(traj) == 0.0


def test_llm_judge_coherence_is_log_only_stub():
    traj = clean_dependency_success()
    assert llm_judge_coherence(traj) is None
    breakdown = compute_outcome_rewards(traj)
    assert breakdown["llm_judge_coherence"] is None
    # and it must not have been folded into "total"
    assert breakdown["total"] == pytest.approx(1.0)


@pytest.mark.parametrize("forbidden_key", ["verified", "fix_result", "fixresult", "pr_passed_ci"])
def test_fix_result_verified_leak_raises(forbidden_key):
    """Doc §7.2 boundary: FixResult.verified must never become a diagnosis
    outcome-reward input. Sneaking it into final_diagnosis must fail loudly.
    """
    traj = copy.deepcopy(clean_dependency_success())
    traj.final_diagnosis[forbidden_key] = True
    with pytest.raises(ValueError):
        compute_outcome_rewards(traj)
    with pytest.raises(ValueError):
        stop_loss_reward(traj)
