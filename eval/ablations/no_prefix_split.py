"""Ablation: progressive prefix split (doc §4.4) vs. "no-split" (train on
full trajectories only).

Doc §6.4's claim: "去掉渐进式前缀拆分（用完整轨迹直接训 SFT）-> 终止决策准确率
从 88% 掉到 55%，平均工具调用轮数从 5.6 涨到 10 以上，死循环重现". Retraining a
model to measure that regression is out of scope here (no GPU, no 120-
trajectory corpus, no training loop in this repo) — what IS measurable with
real code that runs in this repo is the *training-signal composition* the
doc identifies as the actual mechanism behind the regression: "模型看到的
绝大多数训练样本都是'继续调工具'，收敛信号被稀释" (doc §4.4).

This script calls the REAL ``data/cold_start/prefix_split.py::split_directory``
(not reimplemented) on the 5 real mock trajectories in
``data/cold_start/trajectories/mock/`` and computes, for both variants:

- ``subsample_count``: how many separate, independently-loss-weighted
  training examples this variant produces from the same underlying
  trajectories.
- ``termination_example_ratio``: what fraction of those examples are
  *specifically and only* about the "should I stop and output a Diagnosis"
  decision (target_type == "diagnosis" in the real prefix_split output).

Progressive prefix split (real): a trajectory with N steps becomes N
subsamples — the N-1 "keep going" (tool_call target) subsamples and
exactly 1 dedicated "converge" (diagnosis target) subsample, each with its
own independent loss. termination_example_ratio = n_trajectories /
sum(N) — small, but every trajectory contributes one standalone,
undiluted, full-weight example dedicated entirely to the stop decision.

No-split baseline: the whole trajectory is ONE training example (this is
literally "只用完整轨迹样本训练" — the full-trajectory-only variant the doc's
ablation removes prefix-split down to). The "should I stop" decision is
never its own example; it is always bundled inside the one big multi-
decision example alongside every "keep going" step, sharing that example's
single gradient update. termination_example_ratio = 0 / n_trajectories =
0.0 exactly — there are zero *standalone, fully-weighted* examples
dedicated to the stop decision in this variant, which is the concrete,
computable form of "收敛信号被稀释" this script demonstrates.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.cold_start.prefix_split import DEFAULT_TRAJ_DIR, split_directory

#: Doc §4.4's own reported numbers, quoted (NOT computed by this script) for
#: context — this repo only has 5 real mock trajectories, not the doc's 120,
#: so the absolute counts below will not match; only the *ratio direction*
#: is what this ablation demonstrates on real, if small-scale, data.
DOC_REFERENCE = {
    "note": "Quoted from design doc §4.4/§6.4, NOT computed by this script (this repo "
    "has 5 mock trajectories, not the doc's 120).",
    "full_trajectories": 120,
    "avg_steps_per_trajectory": 8.2,
    "prefix_split_subsamples": 990,
    "termination_decision_accuracy_no_split": 0.55,
    "termination_decision_accuracy_prefix_split": 0.88,
    "avg_tool_call_rounds_no_split": ">10",
    "avg_tool_call_rounds_prefix_split": 5.6,
}


def run(traj_dir: Path = DEFAULT_TRAJ_DIR) -> dict[str, Any]:
    n_trajectories = len(list(Path(traj_dir).glob("*.json")))
    if n_trajectories == 0:
        raise RuntimeError(f"no trajectory files found under {traj_dir}")

    # --- real prefix-split variant (data/cold_start/prefix_split.py, unmodified) ---
    subsamples = split_directory(traj_dir)
    prefix_split_total = len(subsamples)
    prefix_split_converge = sum(1 for s in subsamples if s["target_type"] == "diagnosis")
    prefix_split_continue = prefix_split_total - prefix_split_converge
    prefix_split_ratio = round(prefix_split_converge / prefix_split_total, 4) if prefix_split_total else None

    # --- no-split baseline: exactly one whole-trajectory example per trajectory,
    # with zero examples standalone-dedicated to the converge decision ---
    no_split_total = n_trajectories
    no_split_converge = 0
    no_split_continue = no_split_total  # every example is a mixed "whole trajectory" example
    no_split_ratio = round(no_split_converge / no_split_total, 4) if no_split_total else None

    return {
        "n_trajectories": n_trajectories,
        "avg_steps_per_trajectory": round(prefix_split_total / n_trajectories, 3),
        "prefix_split": {
            "subsample_count": prefix_split_total,
            "converge_examples": prefix_split_converge,
            "continue_examples": prefix_split_continue,
            "termination_example_ratio": prefix_split_ratio,
        },
        "no_split": {
            "subsample_count": no_split_total,
            "converge_examples": no_split_converge,
            "continue_examples": no_split_continue,
            "termination_example_ratio": no_split_ratio,
        },
        "subsample_count_delta": prefix_split_total - no_split_total,
        "termination_example_ratio_delta": (
            round(prefix_split_ratio - no_split_ratio, 4)
            if prefix_split_ratio is not None and no_split_ratio is not None
            else None
        ),
        "doc_reference_990_subsample_style": DOC_REFERENCE,
    }


def main() -> None:
    result = run()
    print("[no_prefix_split] real computation over data/cold_start/trajectories/mock/*.json:")
    print(json.dumps({k: v for k, v in result.items() if k != "doc_reference_990_subsample_style"}, ensure_ascii=False, indent=2))
    print("\n[no_prefix_split] doc §4.4/§6.4 reference numbers (quoted, not computed here):")
    print(json.dumps(result["doc_reference_990_subsample_style"], ensure_ascii=False, indent=2))
    print(
        f"\n[no_prefix_split] prefix-split gives {result['prefix_split']['converge_examples']} "
        f"standalone, fully-weighted 'converge' example(s) out of "
        f"{result['prefix_split']['subsample_count']} total "
        f"(ratio={result['prefix_split']['termination_example_ratio']}); "
        f"no-split gives {result['no_split']['converge_examples']} out of "
        f"{result['no_split']['subsample_count']} (ratio={result['no_split']['termination_example_ratio']})."
    )


if __name__ == "__main__":
    main()
