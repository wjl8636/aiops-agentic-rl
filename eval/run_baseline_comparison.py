"""3(-or-4)-way baseline comparison — the RL project's headline table (doc
§6.2): the same labeled eval set, run through zero-shot / SFT / SFT+GRPO /
Opus-reference, shaped like the doc's own 用户级/系统内部 tables.

In real use, ``eval/model_endpoints.py``'s mock classes get replaced one at
a time by real adapters (vLLM zero-shot -> vLLM SFT checkpoint -> vLLM
SFT+GRPO checkpoint -> ``AIops-agent/agent/agents/diagnose.py::diagnose()``
for Opus). This script's job is to prove the *comparison plumbing* — run
all tiers, compute the same metrics per tier, lay them out side by side —
works, not to reproduce the doc's actual percentages.

*** READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRINTS ***
The mock endpoints' win/loss ordering (zero_shot_like worst, sft_grpo_like
best of the trainable tiers, opus_reference_placeholder best overall) is
*scripted by construction* in ``model_endpoints.py`` (see each class's
``wrong_field_prob``/``n_signal_types``/etc.) — it demonstrates the harness
correctly measures a known-ordered set of inputs, and is directionally
consistent with the doc's staircase. It is NOT evidence about a real
Qwen3.5-9B model's zero-shot/SFT/SFT+GRPO behavior, and the exact
percentages are NOT the doc's reported 30/57/81/86% etc. — this repo has no
real checkpoints to run.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.model_endpoints import (
    OpusReferencePlaceholderEndpoint,
    PartiallyGoodEndpoint,
    PerfectDiagnosisEndpoint,
    SFTGRPOEndpoint,
    ZeroShotLikeEndpoint,
)
from eval.run_eval import DEFAULT_REPORTS_DIR, build_report, load_eval_set, run_endpoint

#: The doc's 4-column headline ladder (doc §6.2). "perfect" is deliberately
#: NOT one of these 4 — it's a 100%-by-construction sanity double, not a
#: doc tier (see model_endpoints.PerfectDiagnosisEndpoint's docstring); it
#: is still runnable/comparable via --include-perfect for harness sanity
#: checks, kept out of the headline table so nobody mistakes 100% for a
#: doc-tier number.
DOC_TIER_ENDPOINTS: tuple[type, ...] = (
    ZeroShotLikeEndpoint,
    PartiallyGoodEndpoint,
    SFTGRPOEndpoint,
    OpusReferencePlaceholderEndpoint,
)

#: doc §6.2 用户级 table rows this harness can actually score for the
#: diagnosis agent alone (the other two doc rows — end-to-end PR CI pass
#: rate and hybrid dual-route completion — need the code-fix agent's own
#: output, out of scope for this diagnosis-only mock harness).
USER_LEVEL_METRIC_KEYS: tuple[str, ...] = ("route_accuracy", "stop_loss_success_rate")

#: doc §6.2 系统内部 table rows.
SYSTEM_INTERNAL_METRIC_KEYS: tuple[str, ...] = (
    "root_cause_field_accuracy_overall",
    "evidence_traceability_rate",
    "avg_tool_call_rounds",
    "termination_decision_accuracy",
)


def run_comparison(include_perfect: bool = False) -> dict[str, Any]:
    cases = load_eval_set()
    endpoint_classes = list(DOC_TIER_ENDPOINTS)
    if include_perfect:
        endpoint_classes.append(PerfectDiagnosisEndpoint)

    reports: dict[str, dict[str, Any]] = {}
    for cls in endpoint_classes:
        endpoint = cls()
        trajectories = run_endpoint(endpoint, cases)
        reports[endpoint.name] = build_report(endpoint.name, cases, trajectories)

    def _row(keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        return {key: {name: reports[name]["metrics"].get(key) for name in reports} for key in keys}

    comparison_table = {
        "user_level": _row(USER_LEVEL_METRIC_KEYS),
        "system_internal": _row(SYSTEM_INTERNAL_METRIC_KEYS),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tiers": [cls().name for cls in endpoint_classes],
        "n_cases": len(cases),
        "reports": reports,
        "comparison_table": comparison_table,
        "caveat": (
            "Mock-endpoint numbers on this repo's 5-item labeled eval set, illustrative "
            "only. NOT the design doc's reported percentages (route accuracy "
            "~30/57/81/86%, root-cause 34/62/83%, etc.) — those require real trained "
            "Qwen3.5-9B checkpoints and the doc's full 100-item held-out set, neither of "
            "which exist in this session. This script proves the comparison harness "
            "correctly ranks a known-ordered set of scripted tiers, nothing more."
        ),
    }


def _print_table(comparison: dict[str, Any]) -> None:
    tiers = comparison["tiers"]
    print(f"[run_baseline_comparison] n_cases={comparison['n_cases']}  tiers={tiers}")
    for section_name, section in comparison["comparison_table"].items():
        print(f"\n[{section_name}]")
        header = "  " + "metric".ljust(34) + "".join(t.ljust(24) for t in tiers)
        print(header)
        for metric_key, per_tier in section.items():
            row = "  " + metric_key.ljust(34) + "".join(str(per_tier.get(t)).ljust(24) for t in tiers)
            print(row)


def _main(args: argparse.Namespace) -> None:
    comparison = run_comparison(include_perfect=args.include_perfect)
    _print_table(comparison)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = Path(args.out) if args.out else reports_dir / f"baseline_comparison_{ts}.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[run_baseline_comparison] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="3(-or-4)-way baseline comparison over the labeled eval set")
    parser.add_argument("--include-perfect", action="store_true", help="also run the 100%%-by-construction sanity double")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--out", help="explicit output path")
    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    main()
