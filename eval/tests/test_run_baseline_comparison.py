"""``eval/run_baseline_comparison.py`` — directional sanity only (this repo
has no real trained checkpoints, so we assert ordering, not the design
doc's exact percentages)."""
from __future__ import annotations

import json

from eval.run_baseline_comparison import DOC_TIER_ENDPOINTS, run_comparison


def test_comparison_covers_all_four_doc_tiers():
    comparison = run_comparison(include_perfect=False)
    assert len(comparison["tiers"]) == len(DOC_TIER_ENDPOINTS) == 4
    assert set(comparison["tiers"]) == {
        "zero_shot_like",
        "sft_like",
        "sft_grpo_like",
        "opus_reference_placeholder",
    }
    assert comparison["n_cases"] == 5
    json.dumps(comparison, ensure_ascii=False)  # must be JSON-serializable


def test_comparison_table_has_both_doc_sections():
    comparison = run_comparison()
    table = comparison["comparison_table"]
    assert "user_level" in table
    assert "system_internal" in table
    assert "route_accuracy" in table["user_level"]
    assert "termination_decision_accuracy" in table["system_internal"]


def test_perfect_endpoint_clearly_outscores_zero_shot_like_directionally():
    comparison = run_comparison(include_perfect=True)
    route_row = comparison["comparison_table"]["user_level"]["route_accuracy"]
    perfect_score = route_row["perfect"]
    zero_shot_score = route_row["zero_shot_like"]
    assert perfect_score is not None and zero_shot_score is not None
    assert perfect_score > zero_shot_score
    assert perfect_score == 1.0  # perfect is 100%-by-construction (see model_endpoints.py)


def test_sft_grpo_like_scores_at_least_as_well_as_zero_shot_like():
    comparison = run_comparison()
    route_row = comparison["comparison_table"]["user_level"]["route_accuracy"]
    assert (route_row["sft_grpo_like"] or 0.0) >= (route_row["zero_shot_like"] or 0.0)
