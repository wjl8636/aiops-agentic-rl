"""Fine-grained credit assignment (doc §5.2, layer 3 / §5.3 案例四背景).

The problem this solves: in hybrid/capstone scenarios a whole GRPO group
(e.g. all 6 rollouts for one query) can share the same wrong outcome (route
misjudged), which makes group-relative-advantage zero-variance and wastes
the batch's gradient. Splitting outcome reward evenly across steps
(``reward / T``) does NOT help — it's the same number divided, carrying no
new information, so an all-wrong group is still exactly zero-variance after
division.

The fix (doc's own words): give each step a differentiated "evidence value"
reward — how many *distinct* observability signal categories the trajectory
has covered by that point, weighted by how relevant each category is to the
diagnosed fault ``kind``. Two trajectories that share an identical (wrong)
outcome but explored different signal coverage now get different
intermediate rewards, so the group keeps gradient signal even when outcome
reward alone would collapse to zero variance.

Credit is paid *marginally*, per step: each signal type contributes its
weight exactly once, to the step that first covered it. A step that only
re-visits already-covered categories scores 0 here — so a rollout that
loops on a single repeated tool call cannot farm credit off the same
signal type twelve times, and ``repeat_no_new_info -0.15`` from step-level
reward is the dominant signal on such stalled traces.
"""
from __future__ import annotations

from reward.trajectory import ALL_SIGNAL_TYPES, Trajectory

#: Per-(kind, signal_type) weight table. The doc names two concrete
#: examples ("依赖类故障 Jaeger 权重更高、资源类 kubectl top 权重更高") but
#: gives no exact numbers for any cell — every weight below is a documented
#: judgment call, chosen to be directionally consistent with how each fault
#: kind is actually diagnosed in the AIops-agent skills
#: (dependency-error.md / resource-cpu.md / queue-backlog.md / memory-oom.md):
#:
#: - dependency: a downstream dependency failure shows up first in Jaeger
#:   spans (which hop failed) and logs (the error body from that hop);
#:   Prometheus/kubectl are still useful (rule out resource causes) but
#:   secondary.
#: - resource: CPU/memory pressure is most directly visible via
#:   ``kubectl top``/pod status and Prometheus metrics; Jaeger/git history
#:   contribute comparatively little to "is this box resource-starved".
#: - deploy_regression: correlating a bad deploy needs git/deploy history
#:   above all else; logs are a close second (regression symptoms surface
#:   there); Jaeger/Prometheus are supporting evidence.
#: - config: misconfiguration typically surfaces as explicit errors in
#:   application logs, and RAG history (similar past tickets) is
#:   disproportionately useful for "we've seen this flag combination
#:   before"; kubectl/Prometheus/Jaeger are secondary.
SIGNAL_WEIGHTS_BY_KIND: dict[str, dict[str, float]] = {
    "dependency": {
        "jaeger": 1.5,
        "logs": 1.2,
        "prometheus": 1.0,
        "kubectl_status": 0.8,
        "rag_history": 0.8,
        "git_history": 0.6,
    },
    "resource": {
        "kubectl_status": 1.5,
        "prometheus": 1.3,
        "logs": 0.8,
        "rag_history": 0.8,
        "jaeger": 0.6,
        "git_history": 0.5,
    },
    "deploy_regression": {
        "git_history": 1.5,
        "logs": 1.2,
        "kubectl_status": 1.0,
        "prometheus": 0.8,
        "jaeger": 0.8,
        "rag_history": 0.8,
    },
    "config": {
        "logs": 1.5,
        "rag_history": 1.2,
        "kubectl_status": 0.8,
        "prometheus": 0.7,
        "jaeger": 0.7,
        "git_history": 0.6,
    },
}

#: Fallback weight table used when a trajectory's ``kind`` is unknown/absent
#: (e.g. very early in a rollout before the model has committed to a kind,
#: or a degraded/fallback diagnosis with kind="config" by default already
#: covered above — this is for the case where we have no diagnosis at all
#: yet). Flat weights: every signal type counts equally.
_DEFAULT_WEIGHTS: dict[str, float] = {t: 1.0 for t in ALL_SIGNAL_TYPES}


def _weights_for_kind(kind: str | None) -> dict[str, float]:
    if kind in SIGNAL_WEIGHTS_BY_KIND:
        return SIGNAL_WEIGHTS_BY_KIND[kind]
    return _DEFAULT_WEIGHTS


def cumulative_signal_types(traj: Trajectory, step_index: int) -> set[str]:
    """Distinct signal types covered by steps ``0..step_index`` inclusive."""
    covered: set[str] = set()
    for step in traj.steps[: step_index + 1]:
        covered.update(step.signal_types_covered)
    return covered


def _newly_covered_signal_types(traj: Trajectory, step_index: int) -> set[str]:
    """Signal types this step introduces for the first time in the trajectory
    (``cumulative(i) - cumulative(i-1)``). Empty on any step that only
    re-visits already-covered categories.
    """
    now = cumulative_signal_types(traj, step_index)
    prior = cumulative_signal_types(traj, step_index - 1) if step_index > 0 else set()
    return now - prior


def evidence_value_reward(traj: Trajectory, step_index: int) -> float:
    """Credit-assignment reward for step ``step_index``: sum of
    ``weight[kind][signal_type]`` over ONLY the signal types this step
    introduces *for the first time* in the trajectory (the marginal delta
    against the prior-step cumulative set). If this step covers no new
    signal type — the coverage set didn't grow — the reward is 0.

    Rationale: the previous version returned the cumulative-set weight sum
    at every step, which double-counted earlier discoveries and, worse,
    kept paying a flat per-step reward when a rollout stalled on a
    repeated tool call (12 identical ``kubectl get pods`` still each
    scored ``w[kubectl_status]``, letting repeat-hacking dominate the
    ``-0.15 repeat_no_new_info`` penalty). Marginal crediting fixes both:
    each signal type gets paid exactly once, on the step that first
    covered it; every subsequent step that adds nothing new scores 0.

    ``kind`` is read from ``traj.final_diagnosis["kind"]`` (the model's own
    claimed kind — using ground-truth kind here would leak the label into a
    reward the model can observe patterns from during training; using the
    model's own claim keeps this self-consistent: a model that misclassifies
    a resource fault as "dependency" gets scored against the dependency
    weight table it committed to, which is intentional — the credit signal
    tracks the model's own reasoning path, not an oracle).
    """
    kind = traj.final_diagnosis.get("kind")
    weights = _weights_for_kind(kind)
    new_types = _newly_covered_signal_types(traj, step_index)
    return sum((weights.get(t, 0.0) for t in new_types), 0.0)


def compute_credit_assignment_rewards(traj: Trajectory) -> list[float]:
    """Per-step credit-assignment reward, ``len(result) == len(traj.steps)``."""
    return [evidence_value_reward(traj, i) for i in range(len(traj.steps))]
