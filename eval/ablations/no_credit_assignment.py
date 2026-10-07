"""Ablation: fine-grained credit assignment (doc §5.2 layer 3) vs. naive
outcome-reward/T redistribution (doc §5.3 案例四's explicit "no-op" claim).

Doc's own words (§5.3): "不是把 outcome reward 按轮次均分（reward/T 只是拆分
同一个数字、没引入新信息，全错的一组均分完还是零方差）". This script proves that
claim numerically, using the REAL reward functions in this repo (``reward.
step_reward.compute_step_rewards``, ``reward.credit_assignment.
compute_credit_assignment_rewards``, ``reward.outcome_reward.
compute_outcome_rewards``, ``reward.anti_hacking.group_reward_variance`` —
none reimplemented here).

Construction (a synthetic GRPO group, doc §5.2's "group_size 独立轨迹"):
6 synthetic ``Trajectory`` objects share the exact same alert and the exact
same WRONG final diagnosis/route (so every real outcome-reward component —
root-cause, route-match, handoff, stop-loss — is identically 0.0 across the
whole group by construction: this is the doc's "全错的一组" case). Each
member's steps are built so the 4 genuinely-per-step step-level reward
signals (schema-valid, hook allow/deny, repeat-penalty) are IDENTICAL
across every member too — only ONE thing differs member to member: how
many distinct observability signal types (``Step.signal_types_covered``)
member ``i`` touches on its first step (1 for member 0, up to 6 for member
5). This isolates credit assignment as the *only* possible source of
group variance, so any variance found in the "naive" pipeline below would
be a bug in this ablation's own construction, not a real effect — the
assertions at the bottom of ``run()`` catch that.

Two per-trajectory reward totals are then compared, exactly per the task's
framing:

- ``real_total`` = sum(step-level rewards) + sum(credit-assignment rewards)
  — doc's real layer-1 + layer-3 pipeline.
- ``naive_total`` = sum(step-level rewards) + outcome_reward.total, where
  "outcome_reward.total 按轮次均分再加总" mathematically nets back to
  outcome_reward.total itself (splitting a number into T equal pieces and
  summing them back up is the identity — that IS the "没引入新信息" the doc
  is describing) — so naive_total is just outcome_reward.total added on top
  of the (also-identical-across-the-group) step-level sum.

Because step-level sum and outcome_reward.total are both identical across
every group member by construction, ``group_reward_variance(naive_totals)``
comes out to EXACTLY 0.0 — not "small", zero — while
``group_reward_variance(real_totals)`` is strictly positive because
credit-assignment sums strictly increase with signal-type coverage. That
contrast is this ablation's demonstration of doc §6.4's claim ("混合根因场景
训练早期组内方差长期为零...最终 Route 判定准确率比完整方案掉 6-8 pp") — group
variance is exactly where the credit-assignment layer earns its keep.
"""
from __future__ import annotations

import json
from typing import Any

from reward.anti_hacking import group_reward_variance
from reward.credit_assignment import compute_credit_assignment_rewards
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import ALL_SIGNAL_TYPES, Step, Trajectory

GROUP_SIZE = 6

#: Shared, deliberately WRONG-on-every-field diagnosis (relative to
#: _GROUND_TRUTH below) so every real outcome_reward component is 0.0 for
#: every group member — the doc's "全错的一组" premise, made concrete.
_GROUND_TRUTH = {
    "suspect_service": "recommendation",
    "kind": "deploy_regression",
    "remediation_type": "code_fix",
    "route": "code_fix_pr",
}
_WRONG_ROUTE = "feishu_low_confidence"
_WRONG_DIAGNOSIS_FIELDS = {
    "suspect_service": "ad",
    "kind": "resource",
    "remediation_type": "info_only",  # not "online_op": keeps stop_loss_reward inapplicable (None), not just 0
    "also_code_fix": False,
    "evidence": [],  # empty: no fabrication penalty either way, keeps step-level rewards clean
}


def _build_member(member_index: int, n_signal_types: int) -> Trajectory:
    """One synthetic group member. ``n_signal_types`` (1..6) is the only
    thing that varies member to member — see module docstring."""
    types = list(ALL_SIGNAL_TYPES)[: max(1, min(n_signal_types, len(ALL_SIGNAL_TYPES)))]
    steps = [
        Step(
            tool_name="Bash",
            tool_input={"command": "mock-probe --breadth"},
            hook_decision="passthrough",
            observation=f"member {member_index} covers {len(types)} signal type(s): {types}",
            signal_types_covered=types,
            is_repeat_no_new_info=False,
        ),
        Step(
            tool_name="Bash",
            tool_input={"command": "mock-probe --repeat-1"},
            hook_decision="passthrough",
            observation="repeat probe, no new information",
            signal_types_covered=[],
            is_repeat_no_new_info=True,
        ),
        Step(
            tool_name="Bash",
            tool_input={"command": "mock-probe --repeat-2"},
            hook_decision="passthrough",
            observation="repeat probe, no new information",
            signal_types_covered=[],
            is_repeat_no_new_info=True,
        ),
    ]
    final_diagnosis = dict(_WRONG_DIAGNOSIS_FIELDS)
    alert = {
        "alertname": "MockCreditAssignmentAblation",
        "service": "recommendation",
        "labels": {"service": "recommendation", "severity": "critical", "job": "recommendation"},
        "annotations": {"summary": "synthetic group for the no-credit-assignment ablation"},
    }
    return Trajectory(
        alert=alert,
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=dict(_GROUND_TRUTH),
        route=_WRONG_ROUTE,
        post_action_health=None,
    )


def build_synthetic_group(group_size: int = GROUP_SIZE) -> list[Trajectory]:
    return [_build_member(i, n_signal_types=i + 1) for i in range(group_size)]


def run(group_size: int = GROUP_SIZE) -> dict[str, Any]:
    group = build_synthetic_group(group_size)

    step_sums = [sum(compute_step_rewards(t)) for t in group]
    credit_sums = [sum(compute_credit_assignment_rewards(t)) for t in group]
    outcome_totals = [compute_outcome_rewards(t)["total"] for t in group]

    # Construction sanity: step-level sum and outcome total must be
    # identical across the whole group, or this ablation isn't isolating
    # credit assignment the way its docstring claims.
    if len(set(round(s, 9) for s in step_sums)) != 1:
        raise AssertionError(f"step-level rewards are not identical across the group: {step_sums}")
    if len(set(round(o, 9) for o in outcome_totals)) != 1:
        raise AssertionError(f"outcome rewards are not identical across the group: {outcome_totals}")

    real_totals = [s + c for s, c in zip(step_sums, credit_sums)]
    naive_totals = [s + o for s, o in zip(step_sums, outcome_totals)]

    real_variance = group_reward_variance(real_totals)
    naive_variance = group_reward_variance(naive_totals)

    if naive_variance != 0.0:
        raise AssertionError(
            f"naive (outcome/T redistribution) variance should be exactly 0.0 by "
            f"construction, got {naive_variance}"
        )
    if not (real_variance > 0.0):
        raise AssertionError(
            f"real (step+credit-assignment) variance should be strictly positive, got {real_variance}"
        )

    return {
        "group_size": group_size,
        "step_level_sum_per_member": step_sums,
        "credit_assignment_sum_per_member": credit_sums,
        "outcome_reward_total_per_member": outcome_totals,
        "real_pipeline_total_per_member": real_totals,
        "naive_pipeline_total_per_member": naive_totals,
        "real_pipeline_group_variance": real_variance,
        "naive_pipeline_group_variance": naive_variance,
        "conclusion": (
            "naive outcome/T redistribution preserves EXACTLY zero group-relative "
            "variance on this all-wrong group (doc §5.3 案例四's claim, reproduced "
            "numerically); only the real step+credit-assignment pipeline keeps "
            "nonzero variance, driven entirely by differing signal-type coverage."
        ),
    }


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
