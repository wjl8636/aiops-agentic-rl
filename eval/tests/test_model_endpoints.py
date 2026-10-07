"""``eval/model_endpoints.py`` — every mock endpoint must return a valid
``Trajectory`` for every real labeled case, and ``PerfectDiagnosisEndpoint``
must be exactly correct on every field it has ground truth for (it's a
sanity double, not a doc tier — see its docstring)."""
from __future__ import annotations

import pytest

from eval.model_endpoints import (
    ENDPOINT_REGISTRY,
    OpusReferencePlaceholderEndpoint,
    PartiallyGoodEndpoint,
    PerfectDiagnosisEndpoint,
    SFTGRPOEndpoint,
    ZeroShotLikeEndpoint,
    build_endpoint,
)
from eval.run_eval import load_eval_set
from reward.trajectory import Trajectory


@pytest.fixture(scope="module")
def cases():
    cases = load_eval_set()
    assert len(cases) == 5  # the only 5 labeled mock trajectories in this repo
    return cases


@pytest.mark.parametrize(
    "endpoint_cls",
    [
        PerfectDiagnosisEndpoint,
        ZeroShotLikeEndpoint,
        PartiallyGoodEndpoint,
        SFTGRPOEndpoint,
        OpusReferencePlaceholderEndpoint,
    ],
)
def test_every_endpoint_returns_valid_trajectory_for_every_case(endpoint_cls, cases):
    endpoint = endpoint_cls()
    for case in cases:
        traj = endpoint.diagnose(case.alert, case.ground_truth)
        assert isinstance(traj, Trajectory)
        assert traj.alert == case.alert
        assert len(traj.steps) >= 1


def test_diagnose_without_ground_truth_raises(cases):
    endpoint = ZeroShotLikeEndpoint()
    with pytest.raises(ValueError):
        endpoint.diagnose(cases[0].alert, None)


def test_perfect_endpoint_matches_ground_truth_exactly(cases):
    endpoint = PerfectDiagnosisEndpoint()
    for case in cases:
        traj = endpoint.diagnose(case.alert, case.ground_truth)
        gt = case.ground_truth
        assert traj.route == gt["route"]
        assert traj.final_diagnosis["suspect_service"] == gt["suspect_service"]
        assert traj.final_diagnosis["remediation_type"] == gt["remediation_type"]


def test_perfect_endpoint_is_deterministic(cases):
    endpoint = PerfectDiagnosisEndpoint()
    case = cases[0]
    t1 = endpoint.diagnose(case.alert, case.ground_truth)
    t2 = endpoint.diagnose(case.alert, case.ground_truth)
    assert t1.route == t2.route
    assert t1.final_diagnosis == t2.final_diagnosis
    assert len(t1.steps) == len(t2.steps)


def test_registry_matches_documented_names():
    for key in ENDPOINT_REGISTRY:
        endpoint = build_endpoint(key)
        assert endpoint.name == key
