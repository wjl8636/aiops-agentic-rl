from __future__ import annotations

import common
import pytest
import split
from conftest import load_all_fixtures, load_fixture


def _synthetic_pool(n_per_stratum: int = 20) -> list[dict]:
    """构造一个比 5 条 fixture 大得多的合成池，每个 stratum 都有 n_per_stratum 条，
    用来测试「分层比例在容忍度内」这种在极小样本下没法有效验证的性质。"""
    templates = {
        "kind:dependency": load_fixture("dependency_1"),
        "kind:resource": load_fixture("resource_1"),
        "kind:deploy_regression": load_fixture("deploy_regression_1"),
        "hybrid": load_fixture("hybrid_1"),
        "low_confidence": load_fixture("low_confidence_1"),
    }
    pool = []
    for stratum, template in templates.items():
        for i in range(n_per_stratum):
            item = dict(template)
            item = {**item, "alertname": f"{template['alertname']}__{i}"}
            pool.append(item)
    return pool


def test_infer_kind_matches_expected_bucket_on_fixtures():
    assert common.infer_kind(load_fixture("dependency_1")) == "dependency"
    assert common.infer_kind(load_fixture("resource_1")) == "resource"
    assert common.infer_kind(load_fixture("deploy_regression_1")) == "deploy_regression"


def test_infer_hybrid_and_low_confidence_flags():
    assert common.infer_hybrid(load_fixture("hybrid_1")) is True
    assert common.infer_hybrid(load_fixture("dependency_1")) is False

    assert common.infer_low_confidence(load_fixture("low_confidence_1")) is True
    assert common.infer_low_confidence(load_fixture("dependency_1")) is False


def test_stratum_key_priority_low_confidence_over_hybrid_over_kind():
    assert split.common.stratum_key(load_fixture("low_confidence_1")) == "low_confidence"
    assert split.common.stratum_key(load_fixture("hybrid_1")) == "hybrid"
    assert split.common.stratum_key(load_fixture("dependency_1")) == "kind:dependency"


def test_stratum_key_respects_explicit_fields_over_heuristics():
    """如果上游（比如 sibling 生成任务）已经显式标了 kind/hybrid/low_confidence 字段，
    要优先信任显式字段，不要被关键词启发式覆盖掉。"""
    alert = {
        "alertname": "X",
        "service": "recommendation",
        "kind": "config",
        "hybrid": False,
        "low_confidence": False,
        "annotations": {"summary": "内存持续上涨 OOM 发版 commit", "description": "看起来像资源型+发版回归"},
    }
    assert common.infer_kind(alert) == "config"
    assert common.stratum_key(alert) == "kind:config"


def test_strata_report_on_fixtures_covers_five_named_strata():
    fixtures = load_all_fixtures()
    relevant = [
        fixtures["dependency_1"],
        fixtures["resource_1"],
        fixtures["deploy_regression_1"],
        fixtures["hybrid_1"],
        fixtures["low_confidence_1"],
    ]
    report = split.strata_report(relevant)
    assert report == {
        "hybrid": 1,
        "kind:dependency": 1,
        "kind:deploy_regression": 1,
        "kind:resource": 1,
        "low_confidence": 1,
    }


def test_stratified_split_never_drops_or_duplicates_small_fixture_pool():
    fixtures = load_all_fixtures()
    alerts = list(fixtures.values())

    result = split.stratified_split(alerts, held_out_ratio=0.2, seed=7)
    held_out, train_pool = result["held_out"], result["train_pool"]

    assert len(held_out) + len(train_pool) == len(alerts)

    # 不丢不重：用 id() 而不是 == 比较，因为部分 fixture 内容可能巧合相等
    all_ids = {id(a) for a in alerts}
    out_ids = {id(a) for a in held_out} | {id(a) for a in train_pool}
    assert out_ids == all_ids
    assert len(held_out) + len(train_pool) == len(list(held_out) + list(train_pool))


def test_stratified_split_is_reproducible_with_same_seed():
    alerts = _synthetic_pool(n_per_stratum=15)
    r1 = split.stratified_split(alerts, held_out_ratio=0.2, seed=123)
    r2 = split.stratified_split(alerts, held_out_ratio=0.2, seed=123)
    assert [a["alertname"] for a in r1["held_out"]] == [a["alertname"] for a in r2["held_out"]]
    assert [a["alertname"] for a in r1["train_pool"]] == [a["alertname"] for a in r2["train_pool"]]


def test_stratified_split_preserves_proportions_within_tolerance_on_larger_pool():
    """在每个 stratum 有 30 条的合成池上验证：整体 held_out 占比接近配置的 ratio，
    且每个 stratum 内部也接近这个比例（分层抽样的核心保证）。"""
    n_per_stratum = 30
    ratio = 0.2
    alerts = _synthetic_pool(n_per_stratum=n_per_stratum)

    result = split.stratified_split(alerts, held_out_ratio=ratio, seed=99)
    held_out, train_pool = result["held_out"], result["train_pool"]

    assert len(held_out) + len(train_pool) == len(alerts)

    overall_ratio = len(held_out) / len(alerts)
    assert abs(overall_ratio - ratio) < 0.02

    held_out_by_stratum = split.strata_report(held_out)
    for stratum, count in held_out_by_stratum.items():
        expected = n_per_stratum * ratio
        assert abs(count - expected) <= 1, f"{stratum} held_out count {count} 偏离预期 {expected} 太多"


def test_stratified_split_never_drops_or_duplicates_on_larger_pool():
    alerts = _synthetic_pool(n_per_stratum=25)
    result = split.stratified_split(alerts, held_out_ratio=0.3, seed=5)
    held_out, train_pool = result["held_out"], result["train_pool"]

    combined_names = sorted(a["alertname"] for a in held_out) + sorted(a["alertname"] for a in train_pool)
    original_names = sorted(a["alertname"] for a in alerts)
    assert sorted(combined_names) == original_names
    assert len(combined_names) == len(alerts)


def test_stratified_split_ratio_is_configurable_not_hardcoded_to_500():
    """确认比例可配置：n=25 的小池子换一个 ratio（0.4）依然能正确切分，不要求 n=500。"""
    alerts = _synthetic_pool(n_per_stratum=5)  # 25 条
    result = split.stratified_split(alerts, held_out_ratio=0.4, seed=1)
    assert len(result["held_out"]) + len(result["train_pool"]) == 25
    # 5 档，每档 5 条，held_out_ratio=0.4 → 每档 round(5*0.4)=2 条held_out，共10条
    assert len(result["held_out"]) == 10


def test_stratified_split_rejects_invalid_ratio():
    alerts = _synthetic_pool(n_per_stratum=2)
    with pytest.raises(ValueError):
        split.stratified_split(alerts, held_out_ratio=1.5)
