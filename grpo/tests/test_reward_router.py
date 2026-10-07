"""Unit tests for ``grpo/reward_router.py``, driven by the mock rollout
fixtures at ``data/cold_start/trajectories/mock/*.json`` (loaded via
``grpo/tests/mock_trajectories.py``).

What each test class proves (see individual docstrings for the "how" and
the exact numbers, verified by hand + printed and cross-checked before
being pinned into assertions):

- ``TestDeterminism``: routing the same ``Trajectory``/group twice gives
  bit-identical output.
- ``TestDirectionalSanity``: fabricating evidence, or getting an
  authorized whitelist action DENIED instead of ALLOWED, strictly lowers
  total reward — each via a minimal pair that changes exactly one thing
  (see docstrings for why a minimal pair, not an absolute comparison
  across unrelated mock files, is used here).
- ``TestVariancePreservation``: the doc's central claim for credit
  assignment (§5.2 layer 3 / §5.3 案例四) — a GRPO group that shares an
  identical outcome-level reward (because none of the mutated inputs
  affect ``final_diagnosis``/``route``/``ground_truth``/
  ``post_action_health``) still shows large, real reward variance after
  routing, because credit assignment differentiates on signal coverage
  that outcome reward is blind to. Includes an ablation proving this
  variance isn't an accident of step-level reward's own (much weaker)
  coverage-sensitivity.
- ``TestTurnBudgetPenalties`` / ``TestGroupRewardsShape``: the two
  router-level additions (turn-count penalty, max-turns truncation) and
  ``compute_group_rewards``'s/the variance-tracker hook's basic contracts.

One real finding surfaced while building these tests, worth documenting
here since it affects how the mock fixtures can be used: EVERY trajectory
in ``data/cold_start/trajectories/mock/`` currently has at least one
``final_diagnosis["evidence"]`` claim that is a close PARAPHRASE of a step
observation rather than a verbatim substring of it (e.g. "Prometheus
up{job=...}=0，错误率从 1% 跳升到 62%" vs the observation's "Prometheus:
up{job=...} = 0，过去 5 分钟内错误率从 1% 跳升到 62%。"). Since
``reward.anti_hacking.is_evidence_traceable`` is an intentionally strict,
un-fuzzy substring match (by design — see that module's docstring), EVERY
current mock fixture trips the evidence-fabrication penalty as-is. That's
real, reproducible behavior of the real ``reward/`` module against the
real mock data (not a bug in this router), but it means a clean "no
fabrication" baseline for the directional-sanity tests below has to be
built by patching one loaded mock trajectory's evidence to genuine
verbatim quotes, rather than picking an as-is mock file that already
qualifies. See ``_clean_hybrid_leak_variant`` below.
"""
from __future__ import annotations

import copy

import pytest

from grpo.reward_router import (
    MAX_TURNS,
    MAX_TURNS_TRUNCATION_PENALTY,
    TURN_COUNT_PENALTY_PER_EXTRA_TURN,
    TURN_COUNT_PENALTY_THRESHOLD,
    compute_group_rewards,
    compute_per_step_rewards,
    compute_trajectory_reward,
    max_turns_truncation_penalty,
    route_trajectory_reward_to_variance_tracker,
    turn_count_penalty,
)
from grpo.tests.mock_trajectories import load_all_mock_trajectories, with_signal_coverage
from reward.anti_hacking import group_reward_variance
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import Step, Trajectory

ALL_MOCK = load_all_mock_trajectories()


def _clean_hybrid_leak_variant() -> Trajectory:
    """``mock_hybrid_leak`` with its evidence rewritten to genuine verbatim
    quotes of two of its own step observations (removing the paraphrasing
    confound documented in this file's module docstring) and its
    ``ground_truth`` set to exactly match its own ``final_diagnosis``/
    ``route`` (making it "the right diagnosis" for tests that need a
    positive outcome-reward baseline). Everything else — the steps, the
    one authorized ``allow`` restart, the signal coverage — is untouched.
    """
    traj = copy.deepcopy(ALL_MOCK["mock_hybrid_leak"])
    traj.final_diagnosis["evidence"] = [
        traj.steps[0].observation,
        traj.steps[3].observation,
    ]
    traj.ground_truth = {
        "suspect_service": traj.final_diagnosis["suspect_service"],
        "kind": traj.final_diagnosis["kind"],
        "remediation_type": traj.final_diagnosis["remediation_type"],
        "route": traj.route,
        "suspect_repo": traj.final_diagnosis["suspect_repo"],
        "suspect_commit_hint": traj.final_diagnosis["suspect_commit_hint"],
        "suspect_file_hint": traj.final_diagnosis["suspect_file_hint"],
    }
    return traj


class TestDeterminism:
    def test_single_trajectory_is_deterministic(self):
        traj = ALL_MOCK["mock_hybrid_leak"]
        first = compute_trajectory_reward(traj)
        second = compute_trajectory_reward(traj)
        assert first == second

    def test_group_is_deterministic(self):
        group = [ALL_MOCK[k] for k in sorted(ALL_MOCK)]
        first = compute_group_rewards(group)
        second = compute_group_rewards(group)
        assert first == second

    def test_all_mock_trajectories_route_without_error(self):
        """Every current mock fixture loads and routes to a finite, real
        scalar (guards against the router crashing/NaN-ing on any of the
        5 real shapes — dependency/resource/deploy_regression kinds,
        online_op/info_only remediation types, present/absent
        post_action_health, allow/deny/passthrough hook decisions).
        """
        for name, traj in ALL_MOCK.items():
            result = compute_trajectory_reward(traj)
            assert isinstance(result["total_reward"], float), name
            assert result["total_reward"] == pytest.approx(sum(result["per_step_rewards"])), name


class TestDirectionalSanity:
    """Each test below is a MINIMAL PAIR: two trajectories identical except
    for exactly the one property under test. This isolates causation more
    cleanly than comparing two unrelated mock files (which differ in many
    confounded ways at once — different services, different step counts,
    different signal coverage, etc.) — see this file's module docstring
    for why the raw mock files alone can't cleanly demonstrate this.
    """

    def test_fabricated_evidence_lowers_total_reward_by_exactly_the_penalty(self):
        good = _clean_hybrid_leak_variant()
        bad = copy.deepcopy(good)
        bad.final_diagnosis["evidence"] = list(bad.final_diagnosis["evidence"]) + [
            "kubectl top 显示 recommendation 内存 9999MiB（此声明从未在任何工具调用中出现，纯编造）"
        ]

        good_reward = compute_trajectory_reward(good)["total_reward"]
        bad_reward = compute_trajectory_reward(bad)["total_reward"]

        # step_reward.EVIDENCE_FABRICATION_PENALTY is a flat -0.5 per
        # trajectory (not per fabricated claim) — appending exactly one
        # fabricated claim to an otherwise-clean trajectory should move the
        # total by exactly that flat amount, nothing more/less.
        assert bad_reward == pytest.approx(good_reward - 0.5)
        assert bad_reward < good_reward

    def test_denied_command_scores_lower_than_the_same_command_allowed(self):
        allowed = _clean_hybrid_leak_variant()
        denied = copy.deepcopy(allowed)
        # Flip ONLY the hook's verdict on the (originally authorized)
        # restart_instance action from allow -> deny — e.g. the target
        # instance turning out to be unauthorized, per doc §5.3 案例一's
        # "matched whitelist pattern but unauthorized target" case.
        denied.steps[-1].hook_decision = "deny"

        allowed_reward = compute_trajectory_reward(allowed)["total_reward"]
        denied_reward = compute_trajectory_reward(denied)["total_reward"]

        # Losing WHITELIST_ALLOW_AUTHORIZED_REWARD (+0.1) and gaining
        # WHITELIST_DENY_PENALTY (-0.15) is a swing of exactly -0.25;
        # nothing else in the trajectory changed (signal coverage, outcome
        # inputs, turn count are all untouched), so the router's other
        # terms must be unaffected.
        assert denied_reward == pytest.approx(allowed_reward - 0.25)
        assert denied_reward < allowed_reward

    def test_richer_observability_and_clean_evidence_beats_a_thin_mismatched_trajectory(self):
        """Real-mock-file (unmodified) sanity check: ``mock_dep_clean``
        (four observability steps covering four distinct signal types, plus
        a real health-check-confirmed stop-loss) scores well above
        ``mock_fail_mismatch`` (one step, one signal type, no stop-loss
        signal at all since it's an ``info_only`` diagnosis with no
        ``post_action_health``). Both still incur the paraphrased-evidence
        fabrication penalty documented above, so this is a real,
        directionally-obvious comparison rather than a razor-thin one.
        """
        clean_reward = compute_trajectory_reward(ALL_MOCK["mock_dep_clean"])["total_reward"]
        mismatch_reward = compute_trajectory_reward(ALL_MOCK["mock_fail_mismatch"])["total_reward"]
        assert clean_reward > mismatch_reward


class TestVariancePreservation:
    """The doc's central claim for credit assignment (§5.2 layer 3): a
    GRPO group that shares an identical outcome-level reward must still
    carry gradient signal via per-step credit assignment, differentiated
    by how broadly each rollout explored observability signals. Both
    fixtures below are built with ``with_signal_coverage`` on the SAME
    loaded ``mock_hybrid_leak`` trajectory, changing ONLY
    ``signal_types_covered`` per step — ``final_diagnosis``, ``route``,
    ``ground_truth`` (``{}`` for this fixture, per the loader), and
    ``post_action_health`` are byte-identical across every variant, so
    ``compute_outcome_rewards(...)["total"]`` is PROVABLY identical for
    all of them (asserted below, not just assumed).
    """

    BASE = ALL_MOCK["mock_hybrid_leak"]
    #: doc §5.2 layer-3 weight table entries used by kind="deploy_regression"
    #: (this trajectory's own claimed kind): git_history > logs >
    #: kubectl_status > prometheus == jaeger.
    ALL_FIVE_TYPES = ["git_history", "logs", "kubectl_status", "prometheus", "jaeger"]

    def test_identical_outcome_group_still_shows_nonzero_variance_after_routing(self):
        """Six coverage variants (0..5 distinct signal types touched,
        matching doc §5.1's group_size=6) of the SAME wrong/uncertain
        outcome. Outcome-only variance is exactly 0; the router's group
        variance (step + credit + outcome) is not — proving the
        composition preserves credit assignment's differentiation rather
        than washing it out.
        """
        variants = []
        for k in range(6):
            touched = self.ALL_FIVE_TYPES[:k]
            coverage = [
                [touched[i]] if i < k else ([touched[-1]] if touched else [])
                for i in range(len(self.BASE.steps))
            ]
            variants.append(with_signal_coverage(self.BASE, coverage))

        outcome_totals = [compute_outcome_rewards(v)["total"] for v in variants]
        assert group_reward_variance(outcome_totals) == pytest.approx(0.0)

        group = compute_group_rewards(variants)
        assert group["group_size"] == 6
        assert group["group_reward_variance"] > 0.0
        # Every variant's routed total must still equal its own
        # per-step-reward sum (composition invariant), even though they
        # differ from each other.
        for traj_reward in group["trajectory_rewards"]:
            assert traj_reward["total_reward"] == pytest.approx(sum(traj_reward["per_step_rewards"]))

    def test_credit_assignment_is_not_washed_out_by_step_and_outcome_alone(self):
        """Ablation: with credit assignment removed (i.e. comparing
        step-level + outcome only), the SAME six variants show far less
        variance than the router's full composition. This is the direct
        check that summing credit assignment INTO the per-step list (as
        ``compute_per_step_rewards`` does) preserves its contribution
        rather than it collapsing back toward step-level's much weaker,
        binary "did this step introduce any new type" signal.
        """
        variants = []
        for k in range(6):
            touched = self.ALL_FIVE_TYPES[:k]
            coverage = [
                [touched[i]] if i < k else ([touched[-1]] if touched else [])
                for i in range(len(self.BASE.steps))
            ]
            variants.append(with_signal_coverage(self.BASE, coverage))

        step_and_outcome_only = [
            sum(compute_step_rewards(v)) + compute_outcome_rewards(v)["total"] for v in variants
        ]
        full_totals = compute_group_rewards(variants)["total_rewards"]

        variance_without_credit = group_reward_variance(step_and_outcome_only)
        variance_with_credit = group_reward_variance(full_totals)

        assert variance_without_credit > 0.0  # step-level alone is not literally zero either
        assert variance_with_credit > variance_without_credit  # but credit assignment dominates

    def test_order_isolated_ablation_ties_step_and_outcome_exactly(self):
        """Stricter isolation than the breadth test above: two variants
        that touch the exact SAME final set of 5 signal types (so
        ``compute_step_rewards``'s per-step "new type introduced" bonus
        fires identically, index-for-index, in both — asserted below) and
        share the same outcome inputs, differing ONLY in the ORDER types
        are introduced.

        Credit assignment now pays each newly-covered signal type its
        weight EXACTLY ONCE (on the step that first introduces it), so
        the credit *total* is order-independent — two trajectories that
        cover the same 5 types in different orders score the same credit
        sum, hence the same total reward. This is the desired property:
        credit assignment differentiates on coverage BREADTH (§5.2 layer
        3), not on which order the model happened to discover things,
        which the doc never claims to reward. The variance-driving test
        above (``test_identical_outcome_group_still_shows_nonzero_
        variance_after_routing``) already proves credit assignment
        differentiates on coverage width, which is the property doc §5.2
        actually asks for.
        """
        front_loaded = with_signal_coverage(
            self.BASE, [["git_history"], ["logs"], ["kubectl_status"], ["prometheus"], ["jaeger"]]
        )
        back_loaded = with_signal_coverage(
            self.BASE, [["jaeger"], ["prometheus"], ["kubectl_status"], ["logs"], ["git_history"]]
        )

        assert compute_step_rewards(front_loaded) == compute_step_rewards(back_loaded)
        assert compute_outcome_rewards(front_loaded)["total"] == compute_outcome_rewards(back_loaded)["total"]

        front_total = compute_trajectory_reward(front_loaded)["total_reward"]
        back_total = compute_trajectory_reward(back_loaded)["total_reward"]
        # Order-independent credit total => identical trajectory totals.
        assert front_total == pytest.approx(back_total)


class TestComposition:
    def test_per_step_rewards_is_elementwise_sum_of_step_and_credit(self):
        traj = ALL_MOCK["mock_hookdeny_res"]
        combined = compute_per_step_rewards(traj)
        step_only = compute_step_rewards(traj)
        from reward.credit_assignment import compute_credit_assignment_rewards

        credit_only = compute_credit_assignment_rewards(traj)
        assert combined == [pytest.approx(s + c) for s, c in zip(step_only, credit_only)]

    def test_outcome_reward_attaches_only_to_the_last_step(self):
        traj = ALL_MOCK["mock_dep_clean"]
        per_step_before_outcome = compute_per_step_rewards(traj)
        full = compute_trajectory_reward(traj)
        outcome_total = compute_outcome_rewards(traj)["total"]

        # Every step except the last must be untouched by outcome reward.
        for i in range(len(traj.steps) - 1):
            assert full["per_step_rewards"][i] == pytest.approx(per_step_before_outcome[i])
        # The last step picks up outcome + dual-source + turn-budget terms.
        expected_last = (
            per_step_before_outcome[-1]
            + outcome_total
            + full["dual_source_coverage_penalty"]
            + full["turn_count_penalty"]
            + full["max_turns_truncation_penalty"]
        )
        assert full["per_step_rewards"][-1] == pytest.approx(expected_last)

    def test_zero_step_trajectory_gets_a_synthetic_terminal_reward(self):
        traj = Trajectory(
            alert={"service": "x"},
            steps=[],
            final_diagnosis={"remediation_type": "info_only", "evidence": []},
            ground_truth={},
            route="info_only",
        )
        result = compute_trajectory_reward(traj)
        assert len(result["per_step_rewards"]) == 1
        assert result["total_reward"] == pytest.approx(result["per_step_rewards"][0])


def _step(**overrides) -> Step:
    defaults = dict(
        tool_name="Bash",
        tool_input={"command": "true"},
        hook_decision="passthrough",
        observation="ok",
    )
    defaults.update(overrides)
    return Step(**defaults)


class TestTurnBudgetPenalties:
    def _trajectory_with_n_steps(self, n: int) -> Trajectory:
        return Trajectory(
            alert={"service": "x"},
            steps=[_step() for _ in range(n)],
            final_diagnosis={"remediation_type": "info_only", "evidence": []},
            ground_truth={},
            route="info_only",
        )

    def test_no_penalty_at_or_below_threshold(self):
        for n in (0, 1, TURN_COUNT_PENALTY_THRESHOLD):
            assert turn_count_penalty(self._trajectory_with_n_steps(n)) == 0.0

    def test_penalty_scales_linearly_beyond_threshold(self):
        n = TURN_COUNT_PENALTY_THRESHOLD + 3
        traj = self._trajectory_with_n_steps(n)
        assert turn_count_penalty(traj) == pytest.approx(3 * TURN_COUNT_PENALTY_PER_EXTRA_TURN)

    def test_no_truncation_penalty_at_or_below_max_turns(self):
        assert max_turns_truncation_penalty(self._trajectory_with_n_steps(MAX_TURNS)) == 0.0

    def test_truncation_penalty_applies_past_max_turns(self):
        traj = self._trajectory_with_n_steps(MAX_TURNS + 1)
        assert max_turns_truncation_penalty(traj) == MAX_TURNS_TRUNCATION_PENALTY

    def test_long_trajectory_total_reward_reflects_both_penalties(self):
        short = compute_trajectory_reward(self._trajectory_with_n_steps(TURN_COUNT_PENALTY_THRESHOLD))
        long_ = compute_trajectory_reward(self._trajectory_with_n_steps(MAX_TURNS + 2))
        # More (schema-valid, passthrough) steps alone would only ever add
        # +0.05 each; the long trajectory should still score lower overall
        # once turn-count + truncation penalties are included.
        assert long_["total_reward"] < short["total_reward"]


class TestGroupRewardsShape:
    def test_group_of_one_has_zero_variance(self):
        group = compute_group_rewards([ALL_MOCK["mock_dep_clean"]])
        assert group["group_size"] == 1
        assert group["group_reward_variance"] == 0.0

    def test_group_rewards_preserves_input_order(self):
        ordered = [ALL_MOCK["mock_dep_clean"], ALL_MOCK["mock_fail_mismatch"]]
        group = compute_group_rewards(ordered)
        assert group["total_rewards"][0] == pytest.approx(compute_trajectory_reward(ordered[0])["total_reward"])
        assert group["total_rewards"][1] == pytest.approx(compute_trajectory_reward(ordered[1])["total_reward"])


class TestVarianceTrackerHook:
    def test_does_not_mutate_input_history(self):
        history = {"dependency": [0.0, 0.0]}
        result = route_trajectory_reward_to_variance_tracker("dependency", [1.0, 1.0, 1.0], history)
        assert history == {"dependency": [0.0, 0.0]}  # untouched
        assert result["updated_variance_history"]["dependency"] == [0.0, 0.0, 0.0]

    def test_flags_downsample_after_enough_consecutive_zero_variance_rounds(self):
        history: dict[str, list[float]] = {}
        result = None
        for _ in range(5):
            result = route_trajectory_reward_to_variance_tracker(
                "dependency", [0.5, 0.5, 0.5], history
            )
            history = result["updated_variance_history"]
        assert result["variance"] == 0.0
        assert result["should_downsample"] is True

    def test_does_not_flag_downsample_when_variance_is_real(self):
        history: dict[str, list[float]] = {}
        result = None
        for _ in range(5):
            result = route_trajectory_reward_to_variance_tracker(
                "hybrid_capstone", [0.1, 0.9, 1.5], history
            )
            history = result["updated_variance_history"]
        assert result["variance"] > 0.0
        assert result["should_downsample"] is False
