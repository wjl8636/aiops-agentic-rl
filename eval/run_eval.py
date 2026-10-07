"""Run one ``DiagnosisEndpoint`` against the (only) labeled eval set in this
repo and write a scored JSON report.

Labeled eval set construction
------------------------------
This repo has exactly one source of labeled ground truth today:
``data/cold_start/trajectories/mock/*.json`` (5 mock trajectories, each with
an ``alert`` field) paired 1:1 by filename stem with
``data/cold_start/ground_truth/*.json`` (each a small dict — see below).
``data/seeds/generated/*.json`` (30 seed alerts) and ``data/ood_fixtures/*``
(4 OOD alert payloads) have NO ground truth attached, so they cannot be
used here (they're fine as raw alert inputs for a future *unlabeled* smoke
test, but this module only builds *scored* eval sets).

Ground-truth field mapping (**important — do not assume this matches
``reward.trajectory.Trajectory.ground_truth``'s full field set**): the real
files under ``data/cold_start/ground_truth/`` only ever contain
``remediation_type``, ``suspect_service``, and ``expect_route`` — no
``kind``, and no diagnosis->fix handoff fields. This loader:
  - renames ``expect_route`` -> ``route`` (matching ``Trajectory.
    ground_truth``'s key name, since ``Trajectory`` / ``reward/*.py`` were
    written against the "route" key, not "expect_route"),
  - passes ``suspect_service``/``remediation_type`` straight through,
  - adds one eval-harness-only extension key, ``expected_step_count``,
    taken from the *actual* recorded step count in the paired mock
    trajectory file — this is the "correct stopping point" label
    ``metrics.termination_decision_accuracy`` needs and that the design doc
    doesn't otherwise give us a way to derive (see that function's
    docstring). This key is NOT part of ``Trajectory.ground_truth``'s
    documented field set, but ``Trajectory`` doesn't validate unknown keys,
    so it round-trips harmlessly through the reward package.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from eval import metrics
from eval.model_endpoints import ENDPOINT_REGISTRY, DiagnosisEndpoint, build_endpoint
from reward.trajectory import Trajectory

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_TRAJ_DIR = REPO_ROOT / "data" / "cold_start" / "trajectories" / "mock"
DEFAULT_GT_DIR = REPO_ROOT / "data" / "cold_start" / "ground_truth"
DEFAULT_REPORTS_DIR = HERE / "reports"

#: Ground-truth fields we pass straight through from data/cold_start/ground_truth/*.json.
_PASSTHROUGH_GT_FIELDS: tuple[str, ...] = (
    "suspect_service",
    "remediation_type",
    "kind",
    "suspect_repo",
    "suspect_commit_hint",
    "suspect_file_hint",
)


@dataclass
class EvalCase:
    """One labeled eval item: an alert plus the ground truth to score
    against (``Trajectory.ground_truth``-shaped, see module docstring)."""

    case_id: str
    alert: dict[str, Any]
    ground_truth: dict[str, Any]


def load_eval_set(traj_dir: Path = DEFAULT_TRAJ_DIR, gt_dir: Path = DEFAULT_GT_DIR) -> list[EvalCase]:
    """Build the labeled eval set from the mock trajectories' ``alert``
    field + the paired ground-truth file. Skips any trajectory file that
    has no matching ground-truth file (none currently, but keeps this
    loader safe if more unlabeled mock trajectories get added later).
    """
    cases: list[EvalCase] = []
    for traj_path in sorted(Path(traj_dir).glob("*.json")):
        case_id = traj_path.stem
        gt_path = Path(gt_dir) / f"{case_id}.json"
        if not gt_path.exists():
            continue
        traj_raw = json.loads(traj_path.read_text(encoding="utf-8"))
        gt_raw = json.loads(gt_path.read_text(encoding="utf-8"))

        ground_truth: dict[str, Any] = {}
        for field_name in _PASSTHROUGH_GT_FIELDS:
            if gt_raw.get(field_name) is not None:
                ground_truth[field_name] = gt_raw[field_name]
        route = gt_raw.get("expect_route", gt_raw.get("route"))
        if route is not None:
            ground_truth["route"] = route
        ground_truth["expected_step_count"] = len(traj_raw.get("steps") or [])

        cases.append(EvalCase(case_id=case_id, alert=traj_raw["alert"], ground_truth=ground_truth))
    return cases


def check_case(traj: Trajectory) -> dict[str, Any]:
    """Per-scenario pass/fail checks, reusing ``AIops-agent/eval/run.py``'s
    check semantics (kind match / suspect_service match / remediation_type
    match / route match), scored only over whichever of those 4 fields this
    case's ground truth actually has a label for.
    """
    gt = traj.ground_truth
    diag = traj.final_diagnosis
    checks: dict[str, bool] = {}
    if gt.get("kind") is not None:
        checks["kind"] = diag.get("kind") == gt["kind"]
    if gt.get("suspect_service") is not None:
        checks["suspect_service"] = (diag.get("suspect_service") or "").lower() == gt["suspect_service"].lower()
    if gt.get("remediation_type") is not None:
        checks["remediation_type"] = diag.get("remediation_type") == gt["remediation_type"]
    if gt.get("route") is not None:
        checks["route"] = traj.route == gt["route"]
    return {"checks": checks, "passed": all(checks.values()) if checks else False}


def run_endpoint(endpoint: DiagnosisEndpoint, cases: list[EvalCase]) -> list[Trajectory]:
    """Run ``endpoint`` once per case, in order. Returns one ``Trajectory``
    per case (same order/length as ``cases``)."""
    return [endpoint.diagnose(case.alert, case.ground_truth) for case in cases]


def build_report(endpoint_name: str, cases: list[EvalCase], trajectories: list[Trajectory]) -> dict[str, Any]:
    """Combine per-case pass/fail against expected-style checks with the
    batch-level RL metrics from ``eval.metrics``."""
    per_case: list[dict[str, Any]] = []
    for case, traj in zip(cases, trajectories):
        result = check_case(traj)
        per_case.append(
            {
                "case_id": case.case_id,
                "ground_truth": case.ground_truth,
                "route": traj.route,
                "final_diagnosis": {
                    k: traj.final_diagnosis.get(k)
                    for k in ("kind", "suspect_service", "remediation_type", "confidence", "evidence")
                },
                "num_steps": len(traj.steps),
                "checks": result["checks"],
                "passed": result["passed"],
            }
        )
    n_passed = sum(1 for row in per_case if row["passed"])
    return {
        "endpoint": endpoint_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "n_passed": n_passed,
        "pass_rate": round(n_passed / len(cases), 4) if cases else None,
        "per_case": per_case,
        "metrics": metrics.compute_all(trajectories),
        "caveat": (
            "Illustrative mock-endpoint numbers on a 5-item labeled eval set. NOT a "
            "reproduction of design-doc-reported percentages (those require real "
            "trained checkpoints + the doc's 100-item held-out set)."
        ),
    }


def run(endpoint_key: str, traj_dir: Path = DEFAULT_TRAJ_DIR, gt_dir: Path = DEFAULT_GT_DIR) -> dict[str, Any]:
    """Convenience one-shot entry point used by tests and other scripts:
    load the eval set, run ``endpoint_key``, return the report dict (does
    not write to disk)."""
    endpoint = build_endpoint(endpoint_key)
    cases = load_eval_set(traj_dir, gt_dir)
    trajectories = run_endpoint(endpoint, cases)
    return build_report(endpoint.name, cases, trajectories)


def _main(args: argparse.Namespace) -> None:
    report = run(args.endpoint)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out) if args.out else reports_dir / f"run_eval_{args.endpoint}_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[run_eval] endpoint={args.endpoint}  {report['n_passed']}/{report['n_cases']} passed "
          f"(pass_rate={report['pass_rate']})")
    for k, v in report["metrics"].items():
        print(f"  {k:34s} = {v}")
    print(f"[run_eval] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one mock/real diagnosis endpoint against the labeled eval set")
    parser.add_argument("--endpoint", choices=sorted(ENDPOINT_REGISTRY), default="perfect")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", help="explicit output path (default: <reports-dir>/run_eval_<endpoint>_<ts>.json)")
    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
