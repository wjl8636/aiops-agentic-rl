"""Step-level dense reward — locks in the exact calibrated magnitudes from
doc §5.2 table 1, not just their sign.
"""
from __future__ import annotations

import copy

import pytest

from reward.step_reward import (
    EVIDENCE_FABRICATION_PENALTY,
    OBSERVATION_ADVANCES_DIAGNOSIS_REWARD,
    REPEAT_NO_NEW_INFO_PENALTY,
    SCHEMA_FALLBACK_UNRECOVERED_PENALTY,
    TOOL_SCHEMA_VALID_REWARD,
    WHITELIST_ALLOW_AUTHORIZED_REWARD,
    WHITELIST_DENY_PENALTY,
    compute_step_rewards,
    observation_advances_diagnosis_reward,
    repeat_no_new_info_penalty,
    tool_call_schema_valid_reward,
    whitelist_allow_authorized_reward,
    whitelist_deny_penalty,
)
from reward.trajectory import Step
from reward.tests.fixtures import (
    clean_dependency_success,
    fabricated_evidence_case,
    unauthorized_command_deny_case,
)


def test_magnitudes_match_doc_table():
    assert TOOL_SCHEMA_VALID_REWARD == 0.05
    assert WHITELIST_ALLOW_AUTHORIZED_REWARD == 0.1
    assert WHITELIST_DENY_PENALTY == -0.15
    assert OBSERVATION_ADVANCES_DIAGNOSIS_REWARD == 0.1
    assert REPEAT_NO_NEW_INFO_PENALTY == -0.15
    assert EVIDENCE_FABRICATION_PENALTY == -0.5
    assert SCHEMA_FALLBACK_UNRECOVERED_PENALTY == -0.2


def test_clean_dependency_trajectory_exact_step_rewards():
    traj = clean_dependency_success()
    rewards = compute_step_rewards(traj)
    # step0: schema(0.05) + passthrough(0) + advances-new-prometheus(0.1) = 0.15
    # step1: schema(0.05) + passthrough(0) + advances-new-jaeger(0.1)     = 0.15
    # step2: schema(0.05) + allow(0.1) + no-new-signal-type(0)            = 0.15
    assert rewards == pytest.approx([0.15, 0.15, 0.15])
    assert sum(rewards) == pytest.approx(0.45)


def test_unauthorized_deny_exact_reward():
    traj = unauthorized_command_deny_case()
    rewards = compute_step_rewards(traj)
    # schema(0.05) + deny(-0.15) + no signal coverage(0) = -0.10
    assert rewards == pytest.approx([-0.10])


def test_fabricated_evidence_applies_flat_penalty_on_last_step():
    clean = compute_step_rewards(clean_dependency_success())
    fabricated = compute_step_rewards(fabricated_evidence_case())
    # Same steps/observations as the clean case; only evidence differs.
    assert fabricated[-1] == pytest.approx(clean[-1] + EVIDENCE_FABRICATION_PENALTY)
    assert fabricated[-1] == pytest.approx(0.15 - 0.5)


def test_schema_fallback_penalty_applies_once_on_last_step():
    traj = clean_dependency_success()
    traj.final_diagnosis["schema_fallback_triggered"] = True
    rewards = compute_step_rewards(traj)
    assert rewards[-1] == pytest.approx(0.15 - 0.2)
    assert rewards[0] == pytest.approx(0.15)  # earlier steps untouched


def test_tool_call_schema_invalid_step_gets_no_bonus():
    step = Step(tool_name="Bash", tool_input={}, hook_decision="passthrough", tool_call_schema_valid=False)
    assert tool_call_schema_valid_reward(step) == 0.0


def test_repeat_no_new_info_penalty_direct():
    step = Step(tool_name="Bash", tool_input={}, hook_decision="passthrough", is_repeat_no_new_info=True)
    assert repeat_no_new_info_penalty(step) == pytest.approx(-0.15)


def test_observation_advances_requires_new_signal_type():
    prior = [Step(tool_name="Bash", tool_input={}, hook_decision="passthrough", signal_types_covered=["prometheus"])]
    same_type_again = Step(
        tool_name="Bash", tool_input={}, hook_decision="passthrough", signal_types_covered=["prometheus"]
    )
    new_type = Step(
        tool_name="Bash", tool_input={}, hook_decision="passthrough", signal_types_covered=["jaeger"]
    )
    assert observation_advances_diagnosis_reward(same_type_again, prior) == 0.0
    assert observation_advances_diagnosis_reward(new_type, prior) == pytest.approx(0.1)


def test_whitelist_allow_and_deny_are_mutually_exclusive():
    allow_step = Step(tool_name="Bash", tool_input={}, hook_decision="allow", hook_action="restart_instance")
    deny_step = Step(tool_name="Bash", tool_input={}, hook_decision="deny", hook_action="restart_instance")
    assert whitelist_allow_authorized_reward(allow_step) == pytest.approx(0.1)
    assert whitelist_deny_penalty(allow_step) == 0.0
    assert whitelist_allow_authorized_reward(deny_step) == 0.0
    assert whitelist_deny_penalty(deny_step) == pytest.approx(-0.15)


def test_empty_trajectory_yields_empty_rewards():
    traj = copy.deepcopy(clean_dependency_success())
    traj.steps = []
    assert compute_step_rewards(traj) == []
