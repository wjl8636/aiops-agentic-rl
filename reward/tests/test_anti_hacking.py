"""Doc §5.3's four concrete reward-hacking countermeasures."""
from __future__ import annotations

import copy

import pytest

from reward.anti_hacking import (
    DEFAULT_CONSECUTIVE_ROUNDS,
    DEFAULT_ZERO_VARIANCE_THRESHOLD,
    DUAL_SOURCE_GATE_PENALTY,
    dual_source_coverage_penalty,
    find_fabricated_evidence,
    group_reward_variance,
    has_fabricated_evidence,
    should_downsample,
    verify_stop_loss_ignores_self_report,
)
from reward.tests.fixtures import (
    clean_dependency_success,
    fabricated_evidence_case,
    fake_stop_loss_case,
    hybrid_wrong_route_narrow_coverage,
    unauthorized_command_deny_case,
)


# --- 案例一: dual-source-coverage gate -------------------------------------

def test_dual_source_gate_passes_with_two_distinct_sources():
    traj = clean_dependency_success()  # prometheus + jaeger
    assert dual_source_coverage_penalty(traj) == 0.0


def test_dual_source_gate_penalizes_single_source():
    traj = unauthorized_command_deny_case()  # no signal_types_covered at all
    assert dual_source_coverage_penalty(traj) == pytest.approx(DUAL_SOURCE_GATE_PENALTY)


def test_dual_source_gate_penalizes_narrow_coverage_trajectory():
    traj = hybrid_wrong_route_narrow_coverage()  # kubectl_status only, repeated
    assert dual_source_coverage_penalty(traj) == pytest.approx(-0.2)


# --- 案例二: evidence traceability ------------------------------------------

def test_clean_trajectory_has_no_fabricated_evidence():
    traj = clean_dependency_success()
    assert find_fabricated_evidence(traj) == []
    assert has_fabricated_evidence(traj) is False


def test_fabricated_claim_is_detected_verbatim():
    traj = fabricated_evidence_case()
    fabricated = find_fabricated_evidence(traj)
    assert fabricated == ["kubectl top 显示 recommendation 内存 120MiB"]
    assert has_fabricated_evidence(traj) is True


def test_traceability_is_whitespace_and_case_normalized():
    traj = clean_dependency_success()
    # Same claim, reworded only in whitespace/case -> still traceable.
    traj.final_diagnosis["evidence"] = [
        "  RECOMMENDATION error_rate=0.42   (elevated, baseline 0.01)  "
    ]
    assert has_fabricated_evidence(traj) is False


# --- 案例三: real-health-check-only stop-loss verdict -----------------------

def test_verify_stop_loss_ignores_self_report_holds_for_fake_case():
    traj = fake_stop_loss_case()
    assert verify_stop_loss_ignores_self_report(traj) is True


def test_verify_stop_loss_ignores_self_report_holds_for_clean_case():
    traj = clean_dependency_success()
    assert verify_stop_loss_ignores_self_report(traj) is True


def test_verify_stop_loss_does_not_mutate_original_trajectory():
    traj = fake_stop_loss_case()
    original_detail = copy.deepcopy(traj.final_diagnosis["remediation_detail"])
    verify_stop_loss_ignores_self_report(traj)
    assert traj.final_diagnosis["remediation_detail"] == original_detail


# --- 案例四: group-variance-based downsampling ------------------------------

def test_group_reward_variance_zero_for_identical_rewards():
    assert group_reward_variance([0.5, 0.5, 0.5, 0.5]) == 0.0


def test_group_reward_variance_positive_for_differing_rewards():
    assert group_reward_variance([0.0, 1.0]) == pytest.approx(0.25)  # population variance


def test_group_reward_variance_trivial_for_short_input():
    assert group_reward_variance([]) == 0.0
    assert group_reward_variance([0.7]) == 0.0


def test_should_downsample_flags_long_zero_variance_streak():
    history = [0.0] * DEFAULT_CONSECUTIVE_ROUNDS
    assert should_downsample(history) is True


def test_should_downsample_false_when_streak_too_short():
    history = [0.0] * (DEFAULT_CONSECUTIVE_ROUNDS - 1)
    assert should_downsample(history) is False


def test_should_downsample_false_when_recent_round_has_variance():
    history = [0.0] * (DEFAULT_CONSECUTIVE_ROUNDS - 1) + [0.5]
    assert should_downsample(history) is False


def test_should_downsample_respects_threshold_boundary():
    history = [DEFAULT_ZERO_VARIANCE_THRESHOLD] * DEFAULT_CONSECUTIVE_ROUNDS
    assert should_downsample(history) is True
    history_above = [DEFAULT_ZERO_VARIANCE_THRESHOLD * 10] * DEFAULT_CONSECUTIVE_ROUNDS
    assert should_downsample(history_above) is False
