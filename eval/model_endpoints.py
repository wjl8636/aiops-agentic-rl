"""Pluggable model-endpoint interface for the RL eval harness.

``DiagnosisEndpoint`` is the seam between this harness and whatever actually
produces a diagnosis. In real use each concrete endpoint below would be
replaced by a thin adapter around one of:

- a vLLM OpenAI-compatible endpoint serving Qwen3.5-9B **zero-shot** (no
  fine-tuning) — see design doc §6.2, "zero-shot" column,
- the same vLLM endpoint serving the **Cold-Start SFT** LoRA checkpoint —
  doc §6.2, "SFT" column,
- the same vLLM endpoint serving the **SFT+GRPO** LoRA checkpoint — doc
  §6.2, "SFT+GRPO" column,
- the real Opus-backed diagnosis agent,
  ``AIops-agent/agent/agents/diagnose.py::diagnose()`` — doc §6.2, "Opus 参照"
  column.

None of those four things exist in *this* session (no live vLLM/Opus
access), so this module instead ships scripted mock endpoints that return
canned, ``Trajectory``-shaped output. They exist to prove the harness's
plumbing (loading, scoring, comparison, ablations) works end to end — their
accuracy numbers are illustrative stand-ins tuned to be *directionally*
consistent with the doc's staircase (zero-shot worst, SFT+GRPO best-of-the-
trainable-tiers, Opus-reference best overall), NOT reproductions of the
doc's actual reported percentages (30/57/81/86% etc.). Reproducing those
requires real trained checkpoints and the doc's full 100-item held-out set;
this repo's job is to prove the *eval harness* is correct and runnable.

WHY ``ground_truth`` IS A PARAMETER OF ``diagnose()``
------------------------------------------------------
``DiagnosisEndpoint.diagnose(alert, ground_truth=None)`` accepts an
*optional* ``ground_truth`` argument purely so the scripted mocks below can
script deterministic accuracy tiers (a "perfect" endpoint needs to know
what "correct" is in order to always return it). **A real production
adapter must never read this argument.** The eval harness (``run_eval.py``)
passes it through only because it already has the label in hand for
scoring afterwards anyway; wiring a real model to peek at it would be
exactly the kind of leakage a held-out/OOD eval is supposed to catch. Each
real adapter's ``diagnose()`` implementation should simply ignore the
parameter (or the harness could later be changed to not pass it to
non-mock endpoints at all — left as a TODO for whoever wires up the first
real adapter).
"""
from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from reward.trajectory import (
    ALL_REMEDIATION_TYPES,
    ALL_ROUTES,
    ALL_SIGNAL_TYPES,
    Step,
    Trajectory,
)

#: Small fixed pool of service names used when a mock endpoint needs to pick
#: a *wrong* ``suspect_service`` (rotated away from ground truth) or has no
#: ground truth to anchor on at all. Names lifted from the OTel-Demo-style
#: services the design doc references (§4.1).
SERVICE_POOL: tuple[str, ...] = (
    "product-catalog",
    "recommendation",
    "checkout",
    "ad",
    "currency",
    "payment",
    "cart",
    "frontend",
    "kafka",
    "shipping",
)


def _stable_rng(alert: dict[str, Any], salt: str) -> random.Random:
    """Deterministic per-(alert, endpoint) RNG.

    Same alert + same endpoint name always produces the same scripted
    output (tests can assert on exact values), but different endpoints
    diverge from each other and different alerts diverge from each other.
    """
    key = json.dumps(alert, sort_keys=True, default=str) + "|" + salt
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(seed)


class DiagnosisEndpoint(ABC):
    """Abstract seam between the eval harness and a diagnosis-producing
    model/agent. See module docstring for the real-adapter mapping and the
    ``ground_truth`` caveat.
    """

    #: Short, stable identifier used as a column header in comparison
    #: reports (e.g. "zero_shot_like", "sft_like").
    name: str = "base"

    @abstractmethod
    def diagnose(self, alert: dict[str, Any], ground_truth: Optional[dict[str, Any]] = None) -> Trajectory:
        """Run one diagnosis rollout for ``alert`` and return a
        ``reward.trajectory.Trajectory``. Real adapters must ignore
        ``ground_truth`` (see module docstring)."""
        raise NotImplementedError


class ScriptedMockEndpoint(DiagnosisEndpoint):
    """Base class for every mock endpoint below.

    Parametrized by four knobs, each tuned to move a different observable
    metric in ``eval/metrics.py`` in the direction the design doc describes
    for weaker vs. stronger tiers:

    - ``wrong_field_prob``: independent per-field probability of corrupting
      ``suspect_service``/``remediation_type``/``route`` away from ground
      truth, and of the post-remediation health check reading unhealthy.
      Higher -> lower route/root-cause accuracy and lower stop-loss success
      rate (doc §6.2 zero-shot ~30%/32% vs. SFT+GRPO ~81%/79%).
    - ``n_signal_types``: how many distinct observability signal categories
      the rollout bothers to touch. Lower -> narrower evidence base (also
      feeds ``reward.anti_hacking.dual_source_coverage_penalty`` if reused
      downstream) and a lower credit-assignment sum.
    - ``fabricate_evidence_prob``: probability the ``evidence`` list
      contains one claim that cannot be traced to any step's observation.
      Higher -> lower ``evidence_traceability_rate`` (doc: 62% zero-shot ->
      97% SFT+GRPO).
    - ``padding_steps``: extra no-new-information steps appended after the
      real signal-gathering steps. Higher -> higher ``avg_tool_call_rounds``
      and lower ``termination_decision_accuracy`` — this is the concrete
      stand-in for the doc's "反复调 kubectl 不收敛" failure mode (zero-shot
      12.4 rounds "打满预算" vs. SFT+GRPO 5.6).
    """

    def __init__(
        self,
        name: str,
        wrong_field_prob: float,
        n_signal_types: int,
        fabricate_evidence_prob: float,
        padding_steps: int,
    ) -> None:
        self.name = name
        self.wrong_field_prob = wrong_field_prob
        self.n_signal_types = n_signal_types
        self.fabricate_evidence_prob = fabricate_evidence_prob
        self.padding_steps = padding_steps

    # -- internals ----------------------------------------------------

    def _maybe_corrupt(self, rng: random.Random, truth_value: Any, pool: tuple[str, ...]) -> str:
        if truth_value is None:
            return rng.choice(pool)
        if rng.random() < self.wrong_field_prob:
            candidates = [v for v in pool if v != truth_value]
            return rng.choice(candidates) if candidates else truth_value
        return truth_value

    def diagnose(self, alert: dict[str, Any], ground_truth: Optional[dict[str, Any]] = None) -> Trajectory:
        if ground_truth is None:
            raise ValueError(
                f"{self.name}: scripted mock endpoints require a labeled `ground_truth` "
                "(they can only be used against a labeled eval set, never in production)."
            )
        rng = _stable_rng(alert, self.name)

        service = self._maybe_corrupt(rng, ground_truth.get("suspect_service"), SERVICE_POOL)
        remediation_type = self._maybe_corrupt(rng, ground_truth.get("remediation_type"), ALL_REMEDIATION_TYPES)
        route = self._maybe_corrupt(rng, ground_truth.get("route"), ALL_ROUTES)
        # This harness's only labeled ground-truth set (data/cold_start/ground_truth/*.json)
        # does not carry a `kind` label at all (see run_eval.py's loader docstring) — fall
        # back to a fixed default rather than fabricating a fake "ground truth" to corrupt.
        kind = ground_truth.get("kind") or "resource"

        signal_pool = list(ALL_SIGNAL_TYPES)
        rng.shuffle(signal_pool)
        chosen_types = signal_pool[: max(1, min(self.n_signal_types, len(signal_pool)))]

        steps: list[Step] = []
        observations: list[str] = []
        service_label = alert.get("service") or service
        for i, sig in enumerate(chosen_types):
            obs = f"[{self.name}] mock probe #{i} observed {sig} signal for {service_label}"
            observations.append(obs)
            steps.append(
                Step(
                    tool_name="Bash",
                    tool_input={"command": f"mock-probe --signal {sig}"},
                    hook_decision="passthrough",
                    observation=obs,
                    signal_types_covered=[sig],
                    is_repeat_no_new_info=False,
                )
            )
        for _ in range(self.padding_steps):
            steps.append(
                Step(
                    tool_name="Bash",
                    tool_input={"command": "mock-probe --signal " + (chosen_types[0] if chosen_types else "kubectl_status")},
                    hook_decision="passthrough",
                    observation=f"[{self.name}] repeat probe, no new information",
                    signal_types_covered=[],
                    is_repeat_no_new_info=True,
                )
            )

        post_action_health = None
        if remediation_type == "online_op":
            restart_obs = f"[{self.name}] {service} restarted, post-action health check pending"
            steps.append(
                Step(
                    tool_name="Bash",
                    tool_input={"command": f"docker restart {service}"},
                    hook_decision="allow",
                    hook_action="restart_instance",
                    hook_target=service,
                    observation=restart_obs,
                )
            )
            observations.append(restart_obs)
            healthy = rng.random() >= self.wrong_field_prob
            post_action_health = {
                "healthy_at_60s": healthy,
                "new_alert_triggered": (not healthy) and rng.random() < 0.3,
            }

        evidence = list(observations[:2])
        if rng.random() < self.fabricate_evidence_prob:
            evidence.append(
                f"[{self.name}] fabricated claim never actually observed in any tool call"
            )

        final_diagnosis: dict[str, Any] = {
            "summary": f"[{self.name} mock] scripted diagnosis for {alert.get('alertname', 'unknown-alert')}",
            "kind": kind,
            "suspect_service": service,
            "suspect_repo": ground_truth.get("suspect_repo"),
            "suspect_commit_hint": ground_truth.get("suspect_commit_hint"),
            "suspect_file_hint": ground_truth.get("suspect_file_hint"),
            "remediation_type": remediation_type,
            "remediation_detail": f"[{self.name} mock] scripted remediation_detail text "
            "(never read by any reward function — see reward/outcome_reward.py's "
            "anti-self-report guard).",
            "confidence": round(max(0.0, 1.0 - self.wrong_field_prob), 2),
            "evidence": evidence,
            "time_window": None,
            "also_code_fix": False,
            "executed_actions": [],
        }

        return Trajectory(
            alert=alert,
            steps=steps,
            final_diagnosis=final_diagnosis,
            ground_truth=dict(ground_truth),
            route=route,
            post_action_health=post_action_health,
        )


class PerfectDiagnosisEndpoint(ScriptedMockEndpoint):
    """Always returns the correct ground truth. Not a doc tier — this is a
    pure harness-sanity-check double: if this endpoint doesn't score 100% on
    every check, the bug is in the eval harness, not in a model.
    """

    def __init__(self) -> None:
        super().__init__(
            name="perfect",
            wrong_field_prob=0.0,
            n_signal_types=4,
            fabricate_evidence_prob=0.0,
            padding_steps=0,
        )


class ZeroShotLikeEndpoint(ScriptedMockEndpoint):
    """Illustrative stand-in for "Qwen3.5-9B zero-shot" (doc §6.2 worst
    column): wrong on most fields, narrow signal coverage, frequently
    fabricates evidence, and pads with no-op repeat steps (the "打满预算"
    death-loop the doc's §4.4 progressive-prefix-split fix targets).
    """

    def __init__(self) -> None:
        super().__init__(
            name="zero_shot_like",
            wrong_field_prob=0.75,
            n_signal_types=1,
            fabricate_evidence_prob=0.55,
            padding_steps=6,
        )


class PartiallyGoodEndpoint(ScriptedMockEndpoint):
    """Illustrative stand-in for "Cold-Start SFT" (doc §6.2 "SFT" column,
    ~57% route accuracy): right more often than wrong, moderate signal
    coverage, occasional evidence fabrication, some padding.
    """

    def __init__(self) -> None:
        super().__init__(
            name="sft_like",
            wrong_field_prob=0.35,
            n_signal_types=2,
            fabricate_evidence_prob=0.15,
            padding_steps=3,
        )


class SFTGRPOEndpoint(ScriptedMockEndpoint):
    """Illustrative stand-in for "SFT+GRPO" (doc §6.2 best trainable tier,
    ~81% route accuracy): mostly correct, broad signal coverage, rarely
    fabricates evidence, minimal padding.
    """

    def __init__(self) -> None:
        super().__init__(
            name="sft_grpo_like",
            wrong_field_prob=0.15,
            n_signal_types=3,
            fabricate_evidence_prob=0.05,
            padding_steps=1,
        )


class OpusReferencePlaceholderEndpoint(ScriptedMockEndpoint):
    """Illustrative placeholder for the doc's "Opus 参照上限" column
    (~86%/82%). **This is NOT a real Opus call** — a real implementation
    would adapt ``AIops-agent/agent/agents/diagnose.py::diagnose()``. Scored
    slightly below ``PerfectDiagnosisEndpoint`` (which is 100% by
    definition) to keep "perfect" reserved as a pure sanity double rather
    than conflated with the doc's real (imperfect) reference ceiling.
    """

    def __init__(self) -> None:
        super().__init__(
            name="opus_reference_placeholder",
            wrong_field_prob=0.08,
            n_signal_types=4,
            fabricate_evidence_prob=0.02,
            padding_steps=0,
        )


#: Convenience registry used by run_eval.py/run_baseline_comparison.py's CLIs.
ENDPOINT_REGISTRY: dict[str, type] = {
    "perfect": PerfectDiagnosisEndpoint,
    "zero_shot_like": ZeroShotLikeEndpoint,
    "sft_like": PartiallyGoodEndpoint,
    "sft_grpo_like": SFTGRPOEndpoint,
    "opus_reference_placeholder": OpusReferencePlaceholderEndpoint,
}


def build_endpoint(key: str) -> DiagnosisEndpoint:
    if key not in ENDPOINT_REGISTRY:
        raise KeyError(f"unknown endpoint {key!r}; choices: {sorted(ENDPOINT_REGISTRY)}")
    return ENDPOINT_REGISTRY[key]()
