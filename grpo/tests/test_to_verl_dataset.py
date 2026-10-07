"""Unit tests for ``grpo/to_verl_dataset.py``.

Uses small hand-built alert/ground-truth fixtures under a tmp_path (not the
real ``data/clean/split/train`` — that directory's real content changes as
more ground truth gets annotated, and these tests must stay stable
regardless of how many real alerts currently have ground truth).
"""
from __future__ import annotations

import json

from grpo.to_verl_dataset import (
    build_prompt_messages,
    build_row,
    convert,
    map_ground_truth,
    strip_scenario_hint,
    write_jsonl,
)

_ALERT = {
    "alertname": "TestAlert",
    "service": "cart",
    "labels": {"service": "cart", "severity": "critical", "job": "cartservice"},
    "annotations": {"summary": "s", "description": "d"},
    "startsAt": "2026-07-01T09:00:00Z",
    "scenario_hint": "[test] this must never reach the prompt",
}

_GT = {
    "remediation_type": "online_op",
    "suspect_service": "cart",
    "expect_route": "feishu_online_op",
    "expect_also_code_fix": True,
}


def _write_alert_and_gt(tmp_path, alert_id: str, alert: dict, gt: dict | None):
    alerts_dir = tmp_path / "alerts"
    gt_dir = tmp_path / "gt"
    alerts_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    (alerts_dir / f"{alert_id}.json").write_text(json.dumps(alert, ensure_ascii=False), encoding="utf-8")
    if gt is not None:
        (gt_dir / f"{alert_id}.json").write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    return alerts_dir, gt_dir


class TestStripScenarioHint:
    def test_removes_only_scenario_hint(self):
        stripped = strip_scenario_hint(_ALERT)
        assert "scenario_hint" not in stripped
        assert stripped["service"] == "cart"
        assert set(stripped.keys()) == set(_ALERT.keys()) - {"scenario_hint"}


class TestMapGroundTruth:
    def test_expect_route_becomes_route(self):
        mapped = map_ground_truth(_GT)
        assert mapped["route"] == "feishu_online_op"
        assert "expect_route" not in mapped

    def test_expect_also_code_fix_is_dropped(self):
        mapped = map_ground_truth(_GT)
        assert "expect_also_code_fix" not in mapped
        assert "also_code_fix" not in mapped

    def test_remediation_type_and_suspect_service_pass_through(self):
        mapped = map_ground_truth(_GT)
        assert mapped["remediation_type"] == "online_op"
        assert mapped["suspect_service"] == "cart"

    def test_unknown_keys_pass_through_unchanged(self):
        mapped = map_ground_truth({**_GT, "some_future_field": 1})
        assert mapped["some_future_field"] == 1


class TestBuildPromptMessages:
    def test_scenario_hint_never_reaches_the_prompt(self):
        messages = build_prompt_messages(_ALERT)
        for m in messages:
            assert "this must never reach the prompt" not in m["content"]

    def test_system_then_user_roles(self):
        messages = build_prompt_messages(_ALERT)
        assert [m["role"] for m in messages] == ["system", "user"]
        assert all(isinstance(m["content"], str) and m["content"] for m in messages)


class TestBuildRow:
    def test_row_schema_matches_verl_convention(self):
        row = build_row("seed_test", _ALERT, _GT, index=3, split="train")
        assert set(row.keys()) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
        assert row["reward_model"] == {
            "style": "rule",
            "ground_truth": {"remediation_type": "online_op", "suspect_service": "cart", "route": "feishu_online_op"},
        }
        assert row["extra_info"]["split"] == "train"
        assert row["extra_info"]["index"] == 3
        assert row["extra_info"]["alert_id"] == "seed_test"
        assert "scenario_hint" not in row["extra_info"]["alert"]


class TestConvert:
    def test_alert_with_ground_truth_is_converted(self, tmp_path):
        alerts_dir, gt_dir = _write_alert_and_gt(tmp_path, "seed_a", _ALERT, _GT)
        rows, skipped = convert(alerts_dir, gt_dir, split="train")
        assert len(rows) == 1
        assert skipped == []
        assert rows[0]["extra_info"]["alert_id"] == "seed_a"

    def test_alert_without_ground_truth_is_skipped_not_dropped_silently(self, tmp_path):
        alerts_dir, gt_dir = _write_alert_and_gt(tmp_path, "seed_a", _ALERT, _GT)
        _write_alert_and_gt(tmp_path, "seed_b", _ALERT, None)
        rows, skipped = convert(alerts_dir, gt_dir, split="train")
        assert len(rows) == 1
        assert skipped == ["seed_b"]

    def test_index_is_stable_ordering_by_filename(self, tmp_path):
        alerts_dir, gt_dir = _write_alert_and_gt(tmp_path, "seed_b", _ALERT, _GT)
        _write_alert_and_gt(tmp_path, "seed_a", _ALERT, _GT)
        rows, _ = convert(alerts_dir, gt_dir, split="train")
        assert [r["extra_info"]["alert_id"] for r in rows] == ["seed_a", "seed_b"]


class TestWriteJsonl:
    def test_round_trips_through_json_lines(self, tmp_path):
        alerts_dir, gt_dir = _write_alert_and_gt(tmp_path, "seed_a", _ALERT, _GT)
        rows, _ = convert(alerts_dir, gt_dir, split="train")
        out_path = tmp_path / "out" / "train.jsonl"
        write_jsonl(rows, out_path)

        lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["extra_info"]["alert_id"] == "seed_a"
