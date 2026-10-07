"""Critical integration test: ``mock_tools.py``'s canned rollout output must
plug into ``reward/``'s functions with zero adaptation glue (per the task's
core acceptance criterion for this adapter).
"""
from __future__ import annotations

import pytest

from reward.credit_assignment import compute_credit_assignment_rewards
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import Trajectory
from verl_adapter import mock_tools


@pytest.fixture(scope="module")
def mock_trajectories() -> list[Trajectory]:
    trajs = mock_tools.load_all_mock_trajectories()
    assert len(trajs) == 5  # data/cold_start/trajectories/mock/*.json has exactly 5 fixtures
    return trajs


def _by_alertname(trajs: list[Trajectory], alertname: str) -> Trajectory:
    return next(t for t in trajs if t.alert.get("alertname") == alertname)


def test_load_all_mock_trajectories_are_valid_trajectory_instances(mock_trajectories):
    for traj in mock_trajectories:
        assert isinstance(traj, Trajectory)
        assert len(traj.steps) >= 1
        assert all(step.hook_decision in ("allow", "deny", "passthrough") for step in traj.steps)


def test_step_rewards_compute_without_error(mock_trajectories):
    for traj in mock_trajectories:
        rewards = compute_step_rewards(traj)
        assert len(rewards) == len(traj.steps)
        assert all(isinstance(r, float) for r in rewards)


def test_outcome_rewards_compute_without_error(mock_trajectories):
    for traj in mock_trajectories:
        out = compute_outcome_rewards(traj)
        assert "total" in out
        assert isinstance(out["total"], float)


def test_credit_assignment_rewards_compute_without_error(mock_trajectories):
    for traj in mock_trajectories:
        rewards = compute_credit_assignment_rewards(traj)
        assert len(rewards) == len(traj.steps)
        assert all(isinstance(r, float) for r in rewards)


def test_dep_clean_scenario_hook_decision_is_a_real_deny(mock_trajectories):
    """mock_dep_clean's last step tries `docker restart product-catalog`;
    product-catalog is not in the default ALLOWED_TARGETS whitelist, so the
    REAL hook must deny it (this is not something mock_tools.py invents —
    it is what AIops-agent/agent/core/remediation.py::classify_command
    actually decides).
    """
    dep = _by_alertname(mock_trajectories, "MockHighErrorRate")
    assert dep.steps[-1].hook_decision == "deny"
    assert dep.steps[-1].hook_action == "restart_instance"
    assert dep.steps[-1].hook_target == "product-catalog"


def test_hookdeny_res_scenario_has_one_deny_and_one_allow(mock_trajectories):
    """mock_hookdeny_res: model first tries `docker exec -it ad top`
    (denied — exec is high-risk), then `docker update --cpus ...` (allowed —
    ad IS in the default whitelist). Both verdicts must come from the real
    hook, not be hand-set.
    """
    res = _by_alertname(mock_trajectories, "MockAdCpuThrottled")
    decisions = [s.hook_decision for s in res.steps]
    assert decisions.count("deny") == 1
    assert decisions.count("allow") == 1
    allow_step = next(s for s in res.steps if s.hook_decision == "allow")
    assert allow_step.hook_action == "scale_resources"
    assert allow_step.hook_target == "ad"


def test_evidence_fabrication_case_is_penalized(mock_trajectories):
    """mock_fail_evidence's final_diagnosis.evidence claims a kubectl top
    reading that no step ever observed -- the -0.5 fabrication penalty
    (folded into the last step) must trigger.
    """
    fail_evidence = _by_alertname(mock_trajectories, "MockFabricatedEvidence")
    rewards = compute_step_rewards(fail_evidence)
    assert rewards[-1] < 0


def test_ground_truth_is_populated_from_cold_start_fixtures(mock_trajectories):
    """ground_truth is None in the raw JSON (collect_trajectories.py leaves
    it for a later human-labeling step) -- mock_tools.py must fill it in
    from data/cold_start/ground_truth/<id>.json so outcome_reward's
    root_cause_field_rewards / route_match_reward have something to compare
    against (rather than silently scoring 0 for everything).
    """
    dep = _by_alertname(mock_trajectories, "MockHighErrorRate")
    assert dep.ground_truth.get("route") == "feishu_online_op"
    assert dep.ground_truth.get("suspect_service") == "product-catalog"
    out = compute_outcome_rewards(dep)
    assert out["route_match_reward"] > 0  # actual route matches ground truth for this clean scenario


def test_post_action_health_translated_to_required_schema(mock_trajectories):
    """Trajectory.post_action_health requires {"healthy_at_60s",
    "new_alert_triggered"} -- the raw mock fixture uses a different shape
    ({"checked","healthy","metric","window_s"}); mock_tools.py must
    translate it (Trajectory's __post_init__ would reject the untranslated
    shape outright).
    """
    dep = _by_alertname(mock_trajectories, "MockHighErrorRate")
    assert dep.post_action_health == {"healthy_at_60s": True, "new_alert_triggered": False}
    stop_loss = compute_outcome_rewards(dep)["stop_loss_reward"]
    assert stop_loss == 0.5  # real health check says healthy_at_60s=True -> success reward
