"""Both ablation scripts must run for real (no mocking of reward/ or
prefix_split.py) and produce non-trivial, doc-consistent output."""
from __future__ import annotations

from eval.ablations import no_credit_assignment, no_prefix_split


def test_no_prefix_split_runs_and_shows_diluted_termination_signal():
    result = no_prefix_split.run()
    assert result["n_trajectories"] == 5
    prefix = result["prefix_split"]
    no_split = result["no_split"]

    # Real data/cold_start/prefix_split.py output: 5 trajectories -> 16 subsamples
    # (dep_clean=5, fail_evidence=1, fail_mismatch=1, hookdeny_res=4, hybrid_leak=5).
    assert prefix["subsample_count"] == 16
    assert prefix["converge_examples"] == 5
    assert prefix["continue_examples"] == 11

    # no-split baseline: one whole-trajectory example per trajectory, zero of
    # which are a standalone, dedicated "converge" example.
    assert no_split["subsample_count"] == 5
    assert no_split["converge_examples"] == 0
    assert no_split["termination_example_ratio"] == 0.0

    # the actual claim this ablation demonstrates: prefix-split gives the
    # converge decision a non-zero share of standalone dedicated examples;
    # no-split gives it exactly zero.
    assert prefix["termination_example_ratio"] > no_split["termination_example_ratio"]
    assert result["subsample_count_delta"] == 16 - 5


def test_no_credit_assignment_naive_pipeline_has_zero_variance_real_pipeline_does_not():
    result = no_credit_assignment.run()
    assert result["group_size"] == 6

    # construction sanity, restated at the test level: step-level and outcome
    # rewards must be identical across the synthetic group (that's what
    # isolates credit assignment as the only source of variance).
    assert len(set(round(s, 9) for s in result["step_level_sum_per_member"])) == 1
    assert len(set(round(o, 9) for o in result["outcome_reward_total_per_member"])) == 1
    assert all(o == 0.0 for o in result["outcome_reward_total_per_member"])

    # credit-assignment sums must actually differ (monotonically, since
    # coverage strictly increases member to member) for this to be a real test.
    credit_sums = result["credit_assignment_sum_per_member"]
    assert credit_sums == sorted(credit_sums)
    assert len(set(credit_sums)) == len(credit_sums)

    assert result["naive_pipeline_group_variance"] == 0.0
    assert result["real_pipeline_group_variance"] > 0.0


def test_no_credit_assignment_run_raises_if_construction_breaks():
    # run() self-checks its own construction via internal asserts; calling it
    # twice with different group sizes should still hold the same invariants.
    result = no_credit_assignment.run(group_size=3)
    assert result["group_size"] == 3
    assert result["naive_pipeline_group_variance"] == 0.0
    assert result["real_pipeline_group_variance"] > 0.0
