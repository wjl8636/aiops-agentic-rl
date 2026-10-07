from __future__ import annotations

import collect_trajectories as ct
import prefix_split as ps


def test_subsample_count_matches_step_count():
    for traj in ct.build_mock_trajectories():
        n = len(traj["steps"])
        subsamples = ps.split_trajectory(traj)
        assert len(subsamples) == n


def test_only_last_subsample_targets_diagnosis():
    for traj in ct.build_mock_trajectories():
        subsamples = ps.split_trajectory(traj)
        target_types = [s["target_type"] for s in subsamples]
        assert target_types[:-1] == ["tool_call"] * (len(target_types) - 1)
        assert target_types[-1] == "diagnosis"


def test_prefix_slicing_is_correct():
    for traj in ct.build_mock_trajectories():
        steps = traj["steps"]
        subsamples = ps.split_trajectory(traj)
        for i, sample in enumerate(subsamples, start=1):
            if i < len(steps):
                assert sample["prefix_steps"] == steps[0 : i - 1]
                assert sample["target"] == {
                    "tool_name": steps[i - 1]["tool_name"],
                    "tool_input": steps[i - 1]["tool_input"],
                }
            else:
                assert sample["prefix_steps"] == steps
                assert sample["target"] == traj["final_diagnosis"]


def test_split_directory_total_matches_sum_of_steps(tmp_path):
    written = ct.mock_mode(tmp_path)
    total_steps = 0
    import json

    for p in written:
        total_steps += len(json.loads(p.read_text(encoding="utf-8"))["steps"])

    subsamples = ps.split_directory(tmp_path)
    assert len(subsamples) == total_steps


def test_malformed_structured_output_step_is_skipped_entirely():
    """带 `__unparsedToolInput` 的解析失败重试步：既不做目标、也不进前缀（§4.5-3）。"""
    traj = {
        "alert_id": "t_malformed",
        "alert": {"alertname": "T"},
        "steps": [
            {"tool_name": "Bash", "tool_input": {"command": "docker stats ad"}, "observation": "ad CPU 99%"},
            {
                "tool_name": "StructuredOutput",
                "tool_input": {"__unparsedToolInput": "{summary: 坏 JSON"},
                "observation": "Error parsing tool input",
            },
            {"tool_name": "Bash", "tool_input": {"command": "docker ps"}, "observation": "ad up"},
        ],
        "final_diagnosis": {"summary": "ok", "kind": "resource"},
    }
    import json

    subsamples = ps.split_trajectory(traj)
    # 3 步里 malformed 那步整步剔除 -> 有效 2 步 -> 2 条子样本（目标 docker stats + diagnosis）
    assert len(subsamples) == 2
    assert [s["target_type"] for s in subsamples] == ["tool_call", "diagnosis"]
    for s in subsamples:
        assert "__unparsedToolInput" not in json.dumps(s["target"], ensure_ascii=False)
        assert "__unparsedToolInput" not in json.dumps(s["prefix_steps"], ensure_ascii=False)
