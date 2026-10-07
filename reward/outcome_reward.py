"""Outcome-level reward (doc §5.2, layer 2).

Boundary (doc §7.2, load-bearing — read before touching this file): the
diagnosis agent is the only thing being trained here, so outcome reward
only scores signals attributable to the *diagnosis* agent's decisions. The
code-fix agent's own output quality (whether its PR passes build+test,
i.e. ``FixResult.verified``) is explicitly and permanently OUT OF SCOPE —
see ``_assert_no_fix_result_leak`` below, which turns "accidentally wiring
FixResult.verified into this reward" from a silent bug into a loud
exception.
"""
from __future__ import annotations

from typing import Optional

from reward.trajectory import Trajectory

STOP_LOSS_SUCCESS_REWARD = 0.5
STOP_LOSS_NOT_RECOVERED_REWARD = 0.0
STOP_LOSS_NEW_ALERT_PENALTY = -0.3

ROOT_CAUSE_FIELD_REWARD = 0.1
ROOT_CAUSE_FIELDS = ("suspect_service", "kind", "remediation_type")

ROUTE_MATCH_REWARD = 0.2

HANDOFF_FIELD_REWARD = 0.05
HANDOFF_FIELDS = ("suspect_repo", "suspect_commit_hint", "suspect_file_hint")

#: Keys that must never appear in the buckets this module reads from, because
#: they belong to the code-fix agent's own verdict, not the diagnosis
#: agent's. See module docstring / doc §7.2.
_FORBIDDEN_FIX_RESULT_KEYS = ("verified", "fix_result", "fixresult", "pr_passed_ci")


def _assert_no_fix_result_leak(traj: Trajectory) -> None:
    """Fail loudly if ``FixResult.verified`` (or similarly-named fix-agent
    output) has been merged into ``final_diagnosis`` or ``ground_truth``.

    This function exists purely as a tripwire: doc §7.2 is explicit that
    "修复 Agent 是否 verified=true 不作为诊断 Agent 的直接奖励" — the code-fix
    agent's build+test outcome depends on its own capability and the target
    repo's complexity, and must not be credited/debited back onto the
    diagnosis policy. Since ``Trajectory`` has no ``fix_result``/``verified``
    field in its schema, this can currently only happen if a future caller
    stuffs it into one of the two dict buckets below — this check makes
    that mistake impossible to make silently.
    """
    for bucket_name, bucket in (
        ("final_diagnosis", traj.final_diagnosis),
        ("ground_truth", traj.ground_truth),
    ):
        leaked = [k for k in _FORBIDDEN_FIX_RESULT_KEYS if k in bucket]
        if leaked:
            raise ValueError(
                f"outcome_reward: {bucket_name} contains fix-agent-result key(s) "
                f"{leaked} — FixResult.verified must NEVER be used as diagnosis-agent "
                f"outcome reward input (doc §7.2). Remove it before scoring."
            )


def stop_loss_reward(traj: Trajectory) -> Optional[float]:
    """Stop-loss success reward for online_op (incl. hybrid) remediations.

    *** ANTI-HACKING (doc §5.3 案例三): reads ONLY ``traj.post_action_health``
    — a real health-check observation — and NEVER
    ``traj.final_diagnosis["remediation_detail"]`` or any other
    model-authored text. Do not "fix" this function by adding a fallback
    that inspects remediation_detail when post_action_health is missing;
    that is precisely the hacking pattern this design defends against. ***

    Applicability: only scored when the diagnosis claims an online
    operation happened, i.e. ``final_diagnosis["remediation_type"] ==
    "online_op"`` (this covers both a pure online-op remediation and the
    online_op-with-also_code_fix hybrid case, since hybrid keeps
    remediation_type == "online_op"). Returns ``None`` when not applicable
    (nothing to score — e.g. a pure code_fix or info_only diagnosis), so
    callers can distinguish "not scored" from "scored 0".

    When applicable:
      - ``post_action_health["new_alert_triggered"] is True``  -> -0.3
      - else ``post_action_health["healthy_at_60s"] is True``  -> +0.5
      - else (unhealthy, or health check missing entirely)     ->  0.0
    """
    _assert_no_fix_result_leak(traj)

    if traj.final_diagnosis.get("remediation_type") != "online_op":
        return None

    health = traj.post_action_health
    if health is None:
        # No real health-check observation available: cannot claim success.
        # This is the "未回归" (not recovered) bucket, not a crash.
        return STOP_LOSS_NOT_RECOVERED_REWARD
    if health.get("new_alert_triggered"):
        return STOP_LOSS_NEW_ALERT_PENALTY
    if health.get("healthy_at_60s"):
        return STOP_LOSS_SUCCESS_REWARD
    return STOP_LOSS_NOT_RECOVERED_REWARD


def root_cause_field_rewards(traj: Trajectory) -> dict[str, float]:
    """Per-field breakdown: +0.1 each for suspect_service/kind/
    remediation_type matching ground truth (allows partial credit).
    """
    out: dict[str, float] = {}
    for field_name in ROOT_CAUSE_FIELDS:
        predicted = traj.final_diagnosis.get(field_name)
        expected = traj.ground_truth.get(field_name)
        out[field_name] = (
            ROOT_CAUSE_FIELD_REWARD
            if expected is not None and predicted == expected
            else 0.0
        )
    return out


def root_cause_reward(traj: Trajectory) -> float:
    """Sum of the three root-cause field rewards (max +0.3)."""
    return sum(root_cause_field_rewards(traj).values())


def route_match_reward(traj: Trajectory) -> float:
    """+0.2 when the orchestrator's actual ``route`` matches
    ``ground_truth["route"]``.
    """
    expected = traj.ground_truth.get("route")
    if expected is not None and traj.route == expected:
        return ROUTE_MATCH_REWARD
    return 0.0


def _handoff_applicable(traj: Trajectory) -> bool:
    """Handoff fields are only scored for code_fix / hybrid scenarios."""
    remediation_type = traj.final_diagnosis.get("remediation_type")
    also_code_fix = bool(traj.final_diagnosis.get("also_code_fix"))
    return remediation_type == "code_fix" or also_code_fix


def handoff_field_rewards(traj: Trajectory) -> dict[str, float]:
    """Per-field breakdown for the diagnosis-to-fix handoff fields
    (suspect_repo/suspect_commit_hint/suspect_file_hint), +0.05 each, only
    when the scenario is code_fix or hybrid (also_code_fix=True). Returns
    all-zero when not applicable (no reward, no penalty — this signal
    simply doesn't apply to pure online_op/info_only diagnoses).
    """
    if not _handoff_applicable(traj):
        return {f: 0.0 for f in HANDOFF_FIELDS}
    out: dict[str, float] = {}
    for field_name in HANDOFF_FIELDS:
        predicted = traj.final_diagnosis.get(field_name)
        expected = traj.ground_truth.get(field_name)
        out[field_name] = (
            HANDOFF_FIELD_REWARD
            if expected is not None and predicted is not None and predicted == expected
            else 0.0
        )
    return out


def handoff_reward(traj: Trajectory) -> float:
    """Sum of the three handoff field rewards (max +0.15)."""
    return sum(handoff_field_rewards(traj).values())


def llm_judge_coherence(traj: Trajectory) -> Optional[float]:
    """Stub for the doc's "LLM Judge 论证连贯性" signal.

    Doc §5.2: "让 Claude Opus 对 Diagnosis 的 evidence 链条与结论的因果连贯度
    打分，只记录不进训练梯度，仅作训练后期人工抽检的对照" — i.e. this is
    LOG-ONLY and must NEVER be added into the trained reward total. This
    repo makes no live Opus/LLM calls (pure rule-based, offline module), so
    this stub always returns ``None``.

    TODO(future task, not this module): a real implementation would call an
    Opus judge here, e.g.
        response = anthropic_client.messages.create(
            model="claude-opus-...", ...,
            prompt=coherence_judge_prompt(traj.final_diagnosis, traj.all_observations()),
        )
        return parse_score(response)
    and the caller (an eval/monitoring harness, not the GRPO reward sum)
    would log the result for periodic human-audit comparison — it must
    never be summed into ``compute_outcome_rewards``'s trained total.
    """
    return None


def compute_outcome_rewards(traj: Trajectory) -> dict[str, object]:
    """Full outcome-level breakdown for one trajectory.

    Returns a dict with the individual components plus ``"total"`` (the sum
    of everything EXCEPT ``llm_judge_coherence``, which is intentionally
    excluded from the trained total per its docstring above).
    """
    _assert_no_fix_result_leak(traj)

    stop_loss = stop_loss_reward(traj)
    root_cause = root_cause_reward(traj)
    route = route_match_reward(traj)
    handoff = handoff_reward(traj)

    total = (stop_loss or 0.0) + root_cause + route + handoff

    return {
        "stop_loss_reward": stop_loss,
        "root_cause_field_rewards": root_cause_field_rewards(traj),
        "root_cause_reward": root_cause,
        "route_match_reward": route,
        "handoff_field_rewards": handoff_field_rewards(traj),
        "handoff_reward": handoff,
        "llm_judge_coherence": llm_judge_coherence(traj),  # log-only, NOT in total
        "total": total,
    }
