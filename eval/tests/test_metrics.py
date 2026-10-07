"""``eval/metrics.py`` — hand-constructed cases with known expected values.

Every case below builds ``Trajectory`` objects directly (no mock endpoints,
no file I/O) so the expected numbers can be computed by hand and asserted
exactly, per the task's "known expected values" requirement.
"""
from __future__ import annotations

import pytest

from eval.metrics import (
    avg_tool_call_rounds,
    compute_all,
    evidence_traceability_rate,
    ood_generalization_ratio,
    root_cause_field_accuracy,
    route_accuracy,
    stop_loss_success_rate,
    termination_decision_accuracy,
)
from reward.trajectory import Step, Trajectory


def _traj(
    *,
    steps=None,
    final_diagnosis=None,
    ground_truth=None,
    route="info_only",
    post_action_health=None,
) -> Trajectory:
    return Trajectory(
        alert={"alertname": "Test", "service": "svc"},
        steps=steps or [],
        final_diagnosis=final_diagnosis or {},
        ground_truth=ground_truth or {},
        route=route,
        post_action_health=post_action_health,
    )


# -- route_accuracy -----------------------------------------------------

def test_route_accuracy_exact():
    trajs = [
        _traj(route="feishu_online_op", ground_truth={"route": "feishu_online_op"}),
        _traj(route="auto_remediated", ground_truth={"route": "feishu_online_op"}),
        _traj(route="info_only", ground_truth={}),  # no route label -> excluded from denominator
    ]
    assert route_accuracy(trajs) == pytest.approx(0.5)


def test_route_accuracy_none_when_no_labels():
    trajs = [_traj(route="info_only", ground_truth={})]
    assert route_accuracy(trajs) is None


# -- root_cause_field_accuracy -------------------------------------------

def test_root_cause_field_accuracy_partial_credit():
    trajs = [
        _traj(
            final_diagnosis={"suspect_service": "ad", "kind": "resource", "remediation_type": "online_op"},
            ground_truth={"suspect_service": "ad", "kind": "dependency", "remediation_type": "online_op"},
        ),
        _traj(
            final_diagnosis={"suspect_service": "checkout", "kind": "resource", "remediation_type": "info_only"},
            ground_truth={"suspect_service": "ad", "kind": "dependency", "remediation_type": "info_only"},
        ),
    ]
    out = root_cause_field_accuracy(trajs)
    assert out["suspect_service"] == pytest.approx(0.5)  # 1/2 correct
    assert out["kind"] == pytest.approx(0.0)  # 0/2 correct
    assert out["remediation_type"] == pytest.approx(1.0)  # 2/2 correct
    assert out["overall"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)


def test_root_cause_field_accuracy_missing_field_is_none():
    trajs = [_traj(final_diagnosis={"suspect_service": "ad"}, ground_truth={"suspect_service": "ad"})]
    out = root_cause_field_accuracy(trajs)
    assert out["kind"] is None
    assert out["remediation_type"] is None
    assert out["overall"] == pytest.approx(1.0)  # only suspect_service was scoreable


# -- stop_loss_success_rate ----------------------------------------------

def test_stop_loss_success_rate():
    healthy = _traj(
        final_diagnosis={"remediation_type": "online_op"},
        post_action_health={"healthy_at_60s": True, "new_alert_triggered": False},
    )
    new_alert = _traj(
        final_diagnosis={"remediation_type": "online_op"},
        post_action_health={"healthy_at_60s": False, "new_alert_triggered": True},
    )
    not_applicable = _traj(final_diagnosis={"remediation_type": "info_only"})
    assert stop_loss_success_rate([healthy, new_alert, not_applicable]) == pytest.approx(0.5)


def test_stop_loss_success_rate_none_when_no_online_op():
    assert stop_loss_success_rate([_traj(final_diagnosis={"remediation_type": "info_only"})]) is None


# -- avg_tool_call_rounds --------------------------------------------------

def _mk_step() -> Step:
    return Step(tool_name="Bash", tool_input={"command": "x"}, hook_decision="passthrough")


def test_avg_tool_call_rounds():
    trajs = [_traj(steps=[_mk_step(), _mk_step()]), _traj(steps=[_mk_step()])]
    assert avg_tool_call_rounds(trajs) == pytest.approx(1.5)


def test_avg_tool_call_rounds_empty_batch_is_none():
    assert avg_tool_call_rounds([]) is None


# -- termination_decision_accuracy ----------------------------------------

def test_termination_decision_accuracy():
    on_target = _traj(steps=[_mk_step(), _mk_step()], ground_truth={"expected_step_count": 2})
    off_target = _traj(steps=[_mk_step()], ground_truth={"expected_step_count": 5})
    unlabeled = _traj(steps=[_mk_step()], ground_truth={})
    assert termination_decision_accuracy([on_target, off_target, unlabeled]) == pytest.approx(0.5)


def test_termination_decision_accuracy_tolerance():
    close_enough = _traj(steps=[_mk_step(), _mk_step(), _mk_step()], ground_truth={"expected_step_count": 2})
    assert termination_decision_accuracy([close_enough], tolerance=0) == pytest.approx(0.0)
    assert termination_decision_accuracy([close_enough], tolerance=1) == pytest.approx(1.0)


# -- evidence_traceability_rate --------------------------------------------

def test_evidence_traceability_rate():
    traceable = _traj(
        steps=[Step(tool_name="Bash", tool_input={}, hook_decision="passthrough", observation="cpu is at 99%")],
        final_diagnosis={"evidence": ["cpu is at 99%"]},
    )
    mixed = _traj(
        steps=[Step(tool_name="Bash", tool_input={}, hook_decision="passthrough", observation="mem ok")],
        final_diagnosis={"evidence": ["mem ok", "totally fabricated claim"]},
    )
    # 3 claims total, 2 traceable ("cpu is at 99%", "mem ok"), 1 fabricated.
    assert evidence_traceability_rate([traceable, mixed]) == pytest.approx(2 / 3, abs=1e-4)


def test_evidence_traceability_rate_none_when_no_evidence():
    assert evidence_traceability_rate([_traj(final_diagnosis={"evidence": []})]) is None


# -- ood_generalization_ratio ---------------------------------------------

def test_ood_generalization_ratio_within_doc_band():
    held_out = {"route_accuracy": 0.81, "root_cause_field_accuracy_overall": 0.83}
    ood = {"route_accuracy": 0.65, "root_cause_field_accuracy_overall": 0.68}
    ratio = ood_generalization_ratio(held_out, ood)
    # (0.65/0.81 + 0.68/0.83) / 2
    expected = ((0.65 / 0.81) + (0.68 / 0.83)) / 2
    assert ratio == pytest.approx(round(expected, 4))
    assert 0.75 <= ratio <= 0.85  # doc §6.3's "no cliff" band, for this hand-picked example


def test_ood_generalization_ratio_skips_missing_or_zero_keys():
    held_out = {"route_accuracy": 0.0, "root_cause_field_accuracy_overall": 0.8}
    ood = {"route_accuracy": 0.5, "root_cause_field_accuracy_overall": 0.4}
    # route_accuracy skipped (division by zero guard) -> ratio comes only from the second key
    assert ood_generalization_ratio(held_out, ood) == pytest.approx(0.5)


def test_ood_generalization_ratio_none_when_nothing_usable():
    assert ood_generalization_ratio({}, {}) is None


# -- compute_all ------------------------------------------------------------

def test_compute_all_returns_every_key():
    trajs = [
        _traj(
            steps=[_mk_step()],
            final_diagnosis={"suspect_service": "ad", "remediation_type": "online_op", "evidence": []},
            ground_truth={"suspect_service": "ad", "remediation_type": "online_op", "route": "auto_remediated"},
            route="auto_remediated",
            post_action_health={"healthy_at_60s": True, "new_alert_triggered": False},
        )
    ]
    out = compute_all(trajs)
    for key in (
        "n",
        "route_accuracy",
        "root_cause_field_accuracy",
        "root_cause_field_accuracy_overall",
        "stop_loss_success_rate",
        "avg_tool_call_rounds",
        "termination_decision_accuracy",
        "evidence_traceability_rate",
    ):
        assert key in out
    assert out["n"] == 1
    assert out["route_accuracy"] == pytest.approx(1.0)
