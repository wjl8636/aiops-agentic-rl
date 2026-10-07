"""Validation behavior of the Trajectory/Step contract itself."""
from __future__ import annotations

import pytest

from reward.trajectory import Step, Trajectory
from reward.tests.fixtures import clean_dependency_success


def test_clean_fixture_constructs_without_error():
    traj = clean_dependency_success()
    assert len(traj.steps) == 3
    assert traj.schema_fallback_triggered is False


def test_all_observations_and_signal_union():
    traj = clean_dependency_success()
    assert len(traj.all_observations()) == 3
    assert traj.all_signal_types_covered() == {"prometheus", "jaeger"}


def test_step_rejects_bad_hook_decision():
    with pytest.raises(ValueError):
        Step(tool_name="Bash", tool_input={}, hook_decision="maybe")  # type: ignore[arg-type]


def test_step_rejects_bad_signal_type():
    with pytest.raises(ValueError):
        Step(
            tool_name="Bash",
            tool_input={},
            hook_decision="passthrough",
            signal_types_covered=["not_a_real_signal"],
        )


def test_step_rejects_bad_hook_action():
    with pytest.raises(ValueError):
        Step(
            tool_name="Bash",
            tool_input={},
            hook_decision="allow",
            hook_action="delete_everything",
        )


def test_trajectory_rejects_bad_route():
    with pytest.raises(ValueError):
        Trajectory(
            alert={},
            steps=[],
            final_diagnosis={},
            ground_truth={},
            route="not_a_real_route",
        )


def test_trajectory_rejects_incomplete_post_action_health():
    with pytest.raises(ValueError):
        Trajectory(
            alert={},
            steps=[],
            final_diagnosis={},
            ground_truth={},
            route="info_only",
            post_action_health={"healthy_at_60s": True},  # missing new_alert_triggered
        )


def test_schema_fallback_sibling_flag_read_correctly():
    traj = Trajectory(
        alert={},
        steps=[],
        final_diagnosis={"schema_fallback_triggered": True},
        ground_truth={},
        route="feishu_low_confidence",
    )
    assert traj.schema_fallback_triggered is True
