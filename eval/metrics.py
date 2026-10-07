"""RL-specific eval metrics (design doc §6.2/§6.3).

Operates on batches of ``reward.trajectory.Trajectory`` — every function
here takes ``trajectories: list[Trajectory]`` where each trajectory already
carries its own ``ground_truth`` (that's what ``Trajectory.ground_truth``
is for; see ``reward/trajectory.py``). This intentionally reuses the same
data contract the reward package trains against instead of inventing a
parallel "eval record" shape.

Naming deliberately mirrors ``AIops-agent/eval/metrics.py`` /
``aggregate.py`` where the concepts overlap (``route_accuracy``,
``suspect_service`` matching, "not applicable" -> ``None`` rather than 0)
so the vocabulary stays consistent for anyone reading both eval harnesses
— see that file's module docstring for the sibling convention.

Every accuracy/rate function returns ``None`` (not 0.0) when nothing in the
batch is applicable to that metric, matching ``AIops-agent/eval/aggregate.
py``'s ``_safe_div`` convention: "not scored" must be distinguishable from
"scored 0%".

Ground-truth caveat (read before trusting any number this module prints):
the only labeled dataset in this repo, ``data/cold_start/ground_truth/*.
json``, has exactly 5 items and does NOT carry a ``kind`` label at all (see
``run_eval.py``'s loader docstring) or the diagnosis->fix handoff fields.
Every metric below degrades gracefully (returns ``None`` for that
sub-component) rather than silently treating "no label" as "wrong".
"""
from __future__ import annotations

from typing import Optional

from reward.anti_hacking import find_fabricated_evidence
from reward.outcome_reward import STOP_LOSS_SUCCESS_REWARD, stop_loss_reward
from reward.trajectory import Trajectory

#: The 3-field "root cause" bundle scored in doc §6.2's 系统内部 table
#: ("Root cause 三字段准确率": 34% -> 62% -> 83%).
ROOT_CAUSE_FIELDS: tuple[str, ...] = ("suspect_service", "kind", "remediation_type")


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def route_accuracy(trajectories: list[Trajectory]) -> Optional[float]:
    """Fraction of trajectories whose ``route`` matches
    ``ground_truth["route"]``, over the subset that has a route label.
    Doc §6.2 用户级 "Route 判定准确率" (~30% -> 57% -> 81% -> 86% Opus).
    """
    scored = [t for t in trajectories if t.ground_truth.get("route") is not None]
    correct = sum(1 for t in scored if t.route == t.ground_truth["route"])
    return _safe_div(correct, len(scored))


def root_cause_field_accuracy(trajectories: list[Trajectory]) -> dict[str, Optional[float]]:
    """Per-field accuracy for the 3-field root-cause bundle, plus an
    ``"overall"`` key averaging whichever fields actually had a label in
    this batch. Doc §6.2 系统内部 "Root cause 三字段准确率".
    """
    out: dict[str, Optional[float]] = {}
    for field_name in ROOT_CAUSE_FIELDS:
        scored = [t for t in trajectories if t.ground_truth.get(field_name) is not None]
        correct = sum(1 for t in scored if t.final_diagnosis.get(field_name) == t.ground_truth[field_name])
        out[field_name] = _safe_div(correct, len(scored))
    applicable = [v for v in out.values() if v is not None]
    out["overall"] = round(sum(applicable) / len(applicable), 4) if applicable else None
    return out


def stop_loss_success_rate(trajectories: list[Trajectory]) -> Optional[float]:
    """Fraction of online-op trajectories that actually recovered within
    the 60s health-check window, per ``reward.outcome_reward.
    stop_loss_reward``'s real-health-check-only verdict (never the model's
    self-report — see that function's anti-hacking docstring). Doc §6.2
    用户级 "止损成功率" (~32% -> 60% -> 79% -> 82% Opus).
    """
    applicable = [t for t in trajectories if t.final_diagnosis.get("remediation_type") == "online_op"]
    successes = sum(1 for t in applicable if stop_loss_reward(t) == STOP_LOSS_SUCCESS_REWARD)
    return _safe_div(successes, len(applicable))


def avg_tool_call_rounds(trajectories: list[Trajectory]) -> Optional[float]:
    """Mean number of tool-call steps per trajectory. Doc §6.2 系统内部
    "平均工具调用轮数" (12.4"打满预算" -> 7.8 -> 5.6): a policy stuck in the
    "keep calling kubectl, never converge" failure mode inflates this.
    """
    if not trajectories:
        return None
    return round(sum(len(t.steps) for t in trajectories) / len(trajectories), 3)


def termination_decision_accuracy(trajectories: list[Trajectory], tolerance: int = 0) -> Optional[float]:
    """Fraction of trajectories that stopped within ``tolerance`` steps of
    the labeled "correct stopping point".

    That label is NOT part of ``reward.trajectory.Trajectory``'s stable
    ground_truth contract (see its docstring: suspect_service/kind/
    remediation_type/route/suspect_repo/suspect_commit_hint/
    suspect_file_hint only) — it is an eval-harness-only extension key,
    ``ground_truth["expected_step_count"]``, that ``run_eval.py``'s loader
    derives from the *real* recorded mock trajectory's own step count (see
    that module). Trajectories whose ground truth lacks this key are
    excluded from the denominator (not scored, not penalized). Doc §6.2
    系统内部 "终止决策准确率" (52% pre-split -> 88% post-split -> 91%).
    """
    scored = [t for t in trajectories if t.ground_truth.get("expected_step_count") is not None]
    correct = sum(
        1 for t in scored if abs(len(t.steps) - t.ground_truth["expected_step_count"]) <= tolerance
    )
    return _safe_div(correct, len(scored))


def evidence_traceability_rate(trajectories: list[Trajectory]) -> Optional[float]:
    """Fraction of individual ``final_diagnosis["evidence"]`` claims
    (pooled across every trajectory in the batch, not one bool per
    trajectory) that trace back to a real tool observation, per
    ``reward.anti_hacking.find_fabricated_evidence`` — reused rather than
    reimplemented, per the task's explicit "import from reward/" guidance.
    Doc §6.2 系统内部 "引用合法性覆盖率" (62% -> 91% -> 97%).
    """
    total = 0
    traceable = 0
    for t in trajectories:
        claims = t.final_diagnosis.get("evidence") or []
        if not claims:
            continue
        fabricated = set(find_fabricated_evidence(t))
        for claim in claims:
            total += 1
            if claim not in fabricated:
                traceable += 1
    return _safe_div(traceable, total)


#: Default metric keys compared by ``ood_generalization_ratio``, matching
#: the two doc §6.3 names exactly ("Route 判定准确率、Root cause 三字段准确率").
DEFAULT_OOD_RATIO_KEYS: tuple[str, ...] = ("route_accuracy", "root_cause_field_accuracy_overall")


def ood_generalization_ratio(
    held_out_metrics: dict[str, object],
    ood_metrics: dict[str, object],
    keys: tuple[str, ...] = DEFAULT_OOD_RATIO_KEYS,
) -> Optional[float]:
    """Doc §6.3's held-out-vs-OOD generalization check, as a pure numeric
    function over two already-computed metrics dicts (e.g. two
    ``compute_all()`` outputs) — this repo has no labeled 50-item OOD set
    (see ``data/ood_fixtures/*`` — 4 alert payloads with no ground truth
    attached), so this function is deliberately NOT wired to run an actual
    OOD eval; it is unit-tested on hand-constructed metrics dicts instead.

    Returns the mean of ``ood_metrics[k] / held_out_metrics[k]`` over every
    ``k`` in ``keys`` that is present and non-null in both dicts and
    non-zero in ``held_out_metrics`` (skips anything else rather than
    raising or dividing by zero). Returns ``None`` if no key was usable.

    Doc §6.3: this ratio landing in 75%-85% (no "断崖式下跌") is read as
    "the policy learned generalizable diagnosis skill, not the training
    distribution's specific alert templates".
    """
    ratios: list[float] = []
    for key in keys:
        h = held_out_metrics.get(key)
        o = ood_metrics.get(key)
        if h is None or o is None or not isinstance(h, (int, float)) or h == 0:
            continue
        ratios.append(float(o) / float(h))
    if not ratios:
        return None
    return round(sum(ratios) / len(ratios), 4)


def compute_all(trajectories: list[Trajectory]) -> dict[str, object]:
    """One-shot bundle of every metric above, for a batch of trajectories
    produced by a single endpoint against a single eval set. This is the
    dict shape ``run_eval.py``/``run_baseline_comparison.py`` embed into
    their JSON reports and what ``ood_generalization_ratio`` expects as
    input on both sides.
    """
    root_cause = root_cause_field_accuracy(trajectories)
    return {
        "n": len(trajectories),
        "route_accuracy": route_accuracy(trajectories),
        "root_cause_field_accuracy": root_cause,
        "root_cause_field_accuracy_overall": root_cause.get("overall"),
        "stop_loss_success_rate": stop_loss_success_rate(trajectories),
        "avg_tool_call_rounds": avg_tool_call_rounds(trajectories),
        "termination_decision_accuracy": termination_decision_accuracy(trajectories),
        "evidence_traceability_rate": evidence_traceability_rate(trajectories),
    }
