from __future__ import annotations

import json

import collect_trajectories as ct

REQUIRED_STEP_KEYS = {
    "tool_name",
    "tool_input",
    "hook_decision",
    "hook_action",
    "hook_target",
    "observation",
    "signal_types_covered",
    "is_repeat_no_new_info",
}

REQUIRED_TRAJ_KEYS = {
    "alert",
    "alert_id",
    "steps",
    "final_diagnosis",
    "ground_truth",
    "route",
    "post_action_health",
    "meta",
}

EXPECTED_MOCK_IDS = {
    "mock_dep_clean",
    "mock_hookdeny_res",
    "mock_hybrid_leak",
    "mock_fail_evidence",
    "mock_fail_mismatch",
}


def test_check_env_structure():
    result = ct.check_env()
    assert result["aiops_agent_dir_exists"] is True
    # core（config/schema/remediation/hooks）不需要 claude_agent_sdk，这台机器上应该能 import 到
    assert result["core_importable"] is True
    assert isinstance(result["claude_agent_sdk_importable"], bool)
    assert isinstance(result["errors"], list)


def test_build_mock_trajectories_count_and_ids():
    trajs = ct.build_mock_trajectories()
    assert 3 <= len(trajs) <= 5 or len(trajs) == 5  # 任务要求 3~5 条；本实现给了 5 条
    ids = {t["alert_id"] for t in trajs}
    assert ids == EXPECTED_MOCK_IDS


def test_mock_trajectories_have_valid_shape_and_schema():
    _, remediation, schema, _ = ct._load_aiops_core()
    for traj in ct.build_mock_trajectories():
        assert REQUIRED_TRAJ_KEYS.issubset(traj.keys())
        assert isinstance(traj["alert"], dict)
        assert len(traj["steps"]) >= 1
        for step in traj["steps"]:
            assert REQUIRED_STEP_KEYS.issubset(step.keys())
            assert step["hook_decision"] in ("allow", "deny", "passthrough")
            assert isinstance(step["signal_types_covered"], list)
            assert isinstance(step["is_repeat_no_new_info"], bool)

        # final_diagnosis 必须能通过 AIops-agent 生产代码里真实的 Diagnosis pydantic 校验
        diag = schema.Diagnosis.model_validate(traj["final_diagnosis"])
        assert 0.0 <= diag.confidence <= 1.0
        assert len(diag.evidence) >= 1

        # route 是 run.py 里真实存在的枚举值之一
        assert traj["route"] in {
            "feishu_low_confidence",
            "feishu_online_op",
            "auto_remediated",
            "online_op_and_code_fix_pr",
            "auto_remediated_and_code_fix_pr",
            "code_fix_pr",
            "feishu_fix_unverified",
            "info_only",
        }
        # 采集阶段不应该编造人工标注
        assert traj["ground_truth"] is None


def test_mock_mode_writes_valid_json_files(tmp_path):
    written = ct.mock_mode(tmp_path)
    assert len(written) == 5
    for path in written:
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["alert_id"] == path.stem
        assert REQUIRED_TRAJ_KEYS.issubset(data.keys())


def test_hook_deny_and_allow_are_both_exercised_across_mocks():
    """至少要出现过 deny 和 allow 两种 hook 判定——覆盖「触发 hook deny」这条要求的场景。"""
    trajs = ct.build_mock_trajectories()
    decisions = {step["hook_decision"] for traj in trajs for step in traj["steps"]}
    assert "allow" in decisions
    assert "deny" in decisions


def test_hybrid_scenario_has_also_code_fix():
    trajs = {t["alert_id"]: t for t in ct.build_mock_trajectories()}
    hybrid = trajs["mock_hybrid_leak"]
    assert hybrid["final_diagnosis"]["also_code_fix"] is True
    assert hybrid["final_diagnosis"]["suspect_commit_hint"]
