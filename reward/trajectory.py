"""Shared data contract: one rollout of the AIOps diagnosis agent, as consumed
by every reward function in this package.

WHY THIS MODULE EXISTS
-----------------------
``AIops-agent/agent/core/sdk_runner.py::run_sdk()`` does **not** return a
step-by-step tool-call history — it only returns the final
``{"raw", "structured", "degraded", "usage", "cost_usd", "num_turns",
"latency_s"}``. That means step-level reward can never be computed from
``run_sdk``'s return value alone: something has to sit *inside* the rollout
(intercepting each tool call, each ``PreToolUse`` hook decision, each tool
observation) and assemble a step-by-step record as the rollout happens.

That "something" is a **future, separate task** (the ``verl_adapter/``
rollout layer, not built here). This module defines the contract that
task must produce and this ``reward/`` package consumes. Treat every field
below as part of a stable interface — other code (this package, and later
``grpo/reward_router.py``) imports and depends on it.

We use plain ``@dataclass`` (not TypedDict) so that:
- construction is validated eagerly (``__post_init__``) instead of failing
  silently deep inside a reward function with a confusing KeyError/TypeError,
- editors/type-checkers get real attribute access (``step.hook_decision``,
  not ``step["hook_decision"]``),
- test fixtures read as plain, explicit constructor calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# --------------------------------------------------------------------------
# Enums mirrored from the real AIops-agent code (see task's "ground-truth
# interface facts"). We intentionally do NOT import AIops-agent at runtime:
# this
# package must stay a pure, dependency-free, offline-testable module, and
# duplicating these small literal sets here is cheap. If AIops-agent's
# enums ever change, update them here too.
# --------------------------------------------------------------------------

#: agent/core/hooks.py::guard_diagnose_ops / remediation.py::classify_command
HookDecision = Literal["allow", "deny", "passthrough"]
HOOK_DECISIONS: tuple[str, ...] = ("allow", "deny", "passthrough")

#: remediation.py::LOW_RISK_ACTIONS action IDs (stable across k8s/docker backends)
HookAction = Literal["restart_instance", "migrate_instance", "scale_resources"]
HOOK_ACTIONS: tuple[str, ...] = ("restart_instance", "migrate_instance", "scale_resources")

#: The observability signal-source categories the rollout layer tags per step.
#: Doc §5.3 案例一 / §5.2 第三层: "Prometheus 量 / Jaeger 链路 / logs 细节 /
#: kubectl 状态 / git 发版历史 / RAG 历史工单".
SignalType = Literal[
    "prometheus", "jaeger", "logs", "kubectl_status", "git_history", "rag_history"
]
ALL_SIGNAL_TYPES: tuple[str, ...] = (
    "prometheus", "jaeger", "logs", "kubectl_status", "git_history", "rag_history",
)

#: Diagnosis.kind (agent/core/schema.py)
Kind = Literal["dependency", "resource", "deploy_regression", "config"]
ALL_KINDS: tuple[str, ...] = ("dependency", "resource", "deploy_regression", "config")

#: Diagnosis.remediation_type (agent/core/schema.py)
RemediationType = Literal["online_op", "code_fix", "info_only"]
ALL_REMEDIATION_TYPES: tuple[str, ...] = ("online_op", "code_fix", "info_only")

#: Orchestrator's real route outcomes (given in task spec, verified against
#: the orchestrator's routing logic).
ALL_ROUTES: tuple[str, ...] = (
    "skipped_duplicate",
    "feishu_low_confidence",
    "auto_remediated",
    "auto_remediated_and_code_fix_pr",
    "online_op_and_code_fix_pr",
    "feishu_online_op",
    "code_fix_pr",
    "feishu_fix_unverified",
    "info_only",
)


@dataclass
class Step:
    """One tool-call turn inside a diagnosis rollout.

    Fields marked "rollout-layer computed" below are NOT recomputed by any
    function in ``reward/`` — they are consumed as given. The rollout layer
    (future ``verl_adapter/`` task) is responsible for populating them
    faithfully from the real tool-call/hook/observation stream; reward
    functions in this package only read them.

    Attributes:
        tool_name: e.g. "Bash", "Read", "Grep", "Glob", or a Skill/MCP tool name.
        tool_input: raw tool_input dict as passed to the tool (e.g.
            ``{"command": "kubectl top pod ..."}`` for Bash).
        hook_decision: the ``PreToolUse`` hook's verdict for this call.
            Mirrors ``guard_diagnose_ops``'s effective decision:
            "allow" == matched the remediation.py whitelist AND target was
            authorized; "deny" == matched-but-unauthorized OR hit a
            write/destructive regex deny list; "passthrough" == hook had no
            opinion (read-only call, implicitly allowed).
        hook_action: one of restart_instance/migrate_instance/scale_resources
            when ``classify_command`` matched a whitelist pattern (allow or
            deny); ``None`` for passthrough or non-Bash steps.
        hook_target: the instance/target name ``classify_command`` extracted
            (e.g. "recommendation"); ``None`` when not applicable.
        observation: raw text of the tool's output/result for this step.
        signal_types_covered: (rollout-layer computed) subset of
            ``ALL_SIGNAL_TYPES`` this step's observation actually touched —
            e.g. a ``curl .../metrics`` call tags ``["prometheus"]``, a
            Jaeger trace fetch tags ``["jaeger"]``. Multiple tags are valid
            (e.g. a combined dashboard read).
        is_repeat_no_new_info: (rollout-layer computed) True when the
            rollout layer determined, by comparing this call's
            (tool_name, target) against prior steps in the same trajectory,
            that this call repeats an earlier one without yielding new
            information. Reward functions consume this flag as-is; they do
            NOT re-derive it from tool_name/tool_input themselves.
        tool_call_schema_valid: whether the Claude Agent SDK's tool-call
            schema parsing succeeded for this step (the "工具调用格式合法"
            signal in doc §5.2). Defaults to True since the overwhelming
            majority of steps parse fine; the rollout layer should set this
            to False on the (rare) steps where SDK-level schema parsing of
            the tool call itself failed.
    """

    tool_name: str
    tool_input: dict[str, Any]
    hook_decision: HookDecision
    observation: str = ""
    hook_action: Optional[str] = None
    hook_target: Optional[str] = None
    signal_types_covered: list[str] = field(default_factory=list)
    is_repeat_no_new_info: bool = False
    tool_call_schema_valid: bool = True

    def __post_init__(self) -> None:
        if self.hook_decision not in HOOK_DECISIONS:
            raise ValueError(
                f"Step.hook_decision must be one of {HOOK_DECISIONS}, got {self.hook_decision!r}"
            )
        if self.hook_action is not None and self.hook_action not in HOOK_ACTIONS:
            raise ValueError(
                f"Step.hook_action must be one of {HOOK_ACTIONS} or None, got {self.hook_action!r}"
            )
        bad = [s for s in self.signal_types_covered if s not in ALL_SIGNAL_TYPES]
        if bad:
            raise ValueError(
                f"Step.signal_types_covered contains unknown signal type(s) {bad}; "
                f"must be a subset of {ALL_SIGNAL_TYPES}"
            )


@dataclass
class Trajectory:
    """One complete diagnosis-agent rollout, from alert to final route.

    Attributes:
        alert: the input alert payload (as given to the diagnosis agent).
        steps: ordered list of tool-call turns, see ``Step``.
        final_diagnosis: a ``Diagnosis.model_dump()``-shaped dict (fields:
            summary, kind, suspect_service, suspect_repo,
            suspect_commit_hint, suspect_file_hint, remediation_type,
            remediation_detail, confidence, evidence, time_window,
            also_code_fix, executed_actions), OPTIONALLY carrying an extra
            sibling key ``"schema_fallback_triggered": bool`` at the same
            dict level (not nested under any Diagnosis field) — set to True
            when a mid-trajectory schema validation failure occurred and was
            never successfully recovered by a retry (i.e. ``Diagnosis
            .fallback()`` had to be used). Absent/False means no unrecovered
            fallback occurred.
        ground_truth: human-labeled fields to score against — subset of
            {suspect_service, kind, remediation_type, route, suspect_repo,
            suspect_commit_hint, suspect_file_hint}.
        route: the orchestrator's actual route decision for this rollout,
            one of ``ALL_ROUTES``.
        post_action_health: ``None`` unless this rollout's remediation
            involved an online op, in which case it MUST be
            ``{"healthy_at_60s": bool, "new_alert_triggered": bool}``.

            *** THIS MUST COME FROM A REAL HEALTH-CHECK READ ***
            (a real ``kubectl get pod`` STATUS / ``/healthz`` / Prometheus
            metric read taken ~60s after the remediation action, per doc
            §5.3 案例三). It must NEVER be derived from — or fall back to —
            the model's own self-reported claim in
            ``final_diagnosis["remediation_detail"]`` (e.g. text like
            "服务已恢复"). Wiring this field to model-authored text is
            EXACTLY the reward-hacking failure mode doc §5.3 案例三
            describes and that ``outcome_reward.stop_loss_reward`` is
            designed to be immune to. If you are populating this field from
            a rollout layer: read a real signal, not a string the model
            wrote.
    """

    alert: dict[str, Any]
    steps: list[Step]
    final_diagnosis: dict[str, Any]
    ground_truth: dict[str, Any]
    route: str
    post_action_health: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.route not in ALL_ROUTES:
            raise ValueError(f"Trajectory.route must be one of {ALL_ROUTES}, got {self.route!r}")
        gt_route = self.ground_truth.get("route")
        if gt_route is not None and gt_route not in ALL_ROUTES:
            raise ValueError(
                f"Trajectory.ground_truth['route'] must be one of {ALL_ROUTES} or absent, "
                f"got {gt_route!r}"
            )
        gt_kind = self.ground_truth.get("kind")
        if gt_kind is not None and gt_kind not in ALL_KINDS:
            raise ValueError(
                f"Trajectory.ground_truth['kind'] must be one of {ALL_KINDS} or absent, got {gt_kind!r}"
            )
        gt_rt = self.ground_truth.get("remediation_type")
        if gt_rt is not None and gt_rt not in ALL_REMEDIATION_TYPES:
            raise ValueError(
                f"Trajectory.ground_truth['remediation_type'] must be one of "
                f"{ALL_REMEDIATION_TYPES} or absent, got {gt_rt!r}"
            )
        if self.post_action_health is not None:
            missing = {"healthy_at_60s", "new_alert_triggered"} - set(self.post_action_health)
            if missing:
                raise ValueError(
                    f"Trajectory.post_action_health is missing required key(s) {missing}"
                )

    # -- convenience read-only accessors used across reward/*.py ----------

    @property
    def schema_fallback_triggered(self) -> bool:
        """Sibling flag inside ``final_diagnosis`` (see class docstring)."""
        return bool(self.final_diagnosis.get("schema_fallback_triggered", False))

    def all_observations(self) -> list[str]:
        """Every step's raw observation text, in order."""
        return [s.observation for s in self.steps]

    def all_signal_types_covered(self) -> set[str]:
        """Union of ``signal_types_covered`` across every step."""
        out: set[str] = set()
        for s in self.steps:
            out.update(s.signal_types_covered)
        return out
