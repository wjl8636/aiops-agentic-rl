from __future__ import annotations

import json

import collect_trajectories as ct
import prefix_split as ps
import to_llamafactory_format as tlf

VALID_ROLES = {"human", "gpt", "function_call", "observation", "system"}


def _all_mock_records():
    records = []
    for traj in ct.build_mock_trajectories():
        records.extend(ps.split_trajectory(traj))
    return records


def test_convert_all_produces_expected_keys():
    converted = tlf.convert_all(_all_mock_records())
    assert len(converted) > 0
    for item in converted:
        assert set(item.keys()) == {"conversations", "system", "tools"}
        assert isinstance(item["system"], str) and item["system"]
        assert isinstance(item["tools"], str)
        json.loads(item["tools"])  # tools 字段必须是合法 JSON 字符串

        convo = item["conversations"]
        assert len(convo) == 2
        human, target = convo
        assert human["from"] == "human"
        assert target["from"] in ("function_call", "gpt")
        for turn in convo:
            assert turn["from"] in VALID_ROLES
            assert isinstance(turn["value"], str) and turn["value"]


def test_tool_call_target_is_valid_json_with_name_and_arguments():
    records = [r for r in _all_mock_records() if r["target_type"] == "tool_call"]
    assert records
    for r in records:
        item = tlf.convert_record(r)
        target_turn = item["conversations"][1]
        assert target_turn["from"] == "function_call"
        payload = json.loads(target_turn["value"])
        assert set(payload.keys()) == {"name", "arguments"}
        assert payload["name"] == r["target"]["tool_name"]


def test_diagnosis_target_is_valid_diagnosis_json():
    records = [r for r in _all_mock_records() if r["target_type"] == "diagnosis"]
    assert records
    for r in records:
        item = tlf.convert_record(r)
        target_turn = item["conversations"][1]
        assert target_turn["from"] == "gpt"
        payload = json.loads(target_turn["value"])
        for key in ("summary", "kind", "suspect_service", "remediation_type", "confidence", "evidence"):
            assert key in payload


def test_jsonl_roundtrip_and_dataset_info_snippet(tmp_path):
    in_path = tmp_path / "subsamples.json"
    out_path = tmp_path / "out.jsonl"
    dataset_info_out = tmp_path / "dataset_info_snippet.json"

    in_path.write_text(json.dumps(_all_mock_records(), ensure_ascii=False), encoding="utf-8")

    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(tlf.HERE / "to_llamafactory_format.py"),
            "--in",
            str(in_path),
            "--out",
            str(out_path),
            "--dataset-info-out",
            str(dataset_info_out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(_all_mock_records())
    for line in lines:
        obj = json.loads(line)
        assert "conversations" in obj

    snippet = json.loads(dataset_info_out.read_text(encoding="utf-8"))
    entry = snippet[tlf.DATASET_NAME]
    assert entry["formatting"] == "sharegpt"
    assert entry["file_name"] == out_path.name
    assert entry["columns"] == {"messages": "conversations", "system": "system", "tools": "tools"}
