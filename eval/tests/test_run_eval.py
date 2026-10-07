"""``eval/run_eval.py`` — loader produces the expected 5-item labeled set,
and ``run()``/``build_report()`` produce a valid, JSON-serializable report
against it."""
from __future__ import annotations

import json

import pytest

from eval.run_eval import DEFAULT_GT_DIR, DEFAULT_TRAJ_DIR, load_eval_set, run


def test_load_eval_set_shape():
    cases = load_eval_set()
    assert len(cases) == 5
    ids = {c.case_id for c in cases}
    assert ids == {
        "mock_dep_clean",
        "mock_fail_evidence",
        "mock_fail_mismatch",
        "mock_hookdeny_res",
        "mock_hybrid_leak",
    }
    for case in cases:
        assert "suspect_service" in case.ground_truth
        assert "remediation_type" in case.ground_truth
        assert "route" in case.ground_truth
        assert "expect_route" not in case.ground_truth  # must be renamed to "route"
        assert case.ground_truth["expected_step_count"] >= 0
        assert isinstance(case.alert, dict)


def test_loader_paths_exist():
    assert DEFAULT_TRAJ_DIR.is_dir()
    assert DEFAULT_GT_DIR.is_dir()


def test_run_perfect_endpoint_produces_valid_report():
    report = run("perfect")
    assert report["endpoint"] == "perfect"
    assert report["n_cases"] == 5
    assert report["n_passed"] == 5
    assert report["pass_rate"] == pytest.approx(1.0)
    assert report["metrics"]["route_accuracy"] == pytest.approx(1.0)
    assert report["metrics"]["n"] == 5
    # must be JSON-serializable (this is what gets written to eval/reports/)
    json.dumps(report, ensure_ascii=False)


def test_run_zero_shot_like_scores_lower_than_perfect():
    perfect = run("perfect")
    zero_shot = run("zero_shot_like")
    assert zero_shot["n_passed"] <= perfect["n_passed"]
    assert (zero_shot["metrics"]["route_accuracy"] or 0.0) <= (perfect["metrics"]["route_accuracy"] or 0.0)


def test_run_unknown_endpoint_raises():
    with pytest.raises(KeyError):
        run("not-a-real-endpoint")
