from __future__ import annotations

import pytest

import common
import dedup
from conftest import load_fixture


def test_canonical_json_is_key_order_insensitive():
    a = load_fixture("exact_dup_a")
    b = load_fixture("exact_dup_b")
    # 文件里字段顺序完全不同（包括嵌套 labels 内部），canonical_json 应该忽略顺序差异。
    assert dedup.canonical_json(a) == dedup.canonical_json(b)
    assert dedup.alert_md5(a) == dedup.alert_md5(b)


def test_md5_dedup_drops_exact_duplicates_keeps_first_seen():
    a = load_fixture("exact_dup_a")
    b = load_fixture("exact_dup_b")  # 语义上跟 a 完全一样，只是 key 顺序不同
    other = load_fixture("dependency_1")

    kept, dropped = dedup.md5_dedup([a, other, b])

    assert kept == [a, other]  # 保留先出现的一条（a），砍掉后出现的重复（b）
    assert dropped == [b]


def test_md5_dedup_keeps_distinct_alerts():
    a = load_fixture("dependency_1")
    b = load_fixture("resource_1")
    c = load_fixture("deploy_regression_1")

    kept, dropped = dedup.md5_dedup([a, b, c])

    assert kept == [a, b, c]
    assert dropped == []


def test_lexical_fallback_backend_used_when_forced():
    backend = dedup.load_embedding_backend(force_fallback=True)
    assert backend.name == "lexical-fallback"
    assert backend.encode is None


def test_char_ngram_jaccard_basic_properties():
    assert dedup._char_ngram_jaccard("abc", "abc") == 1.0
    assert dedup._char_ngram_jaccard("abc", "xyz") == 0.0
    assert dedup._char_ngram_jaccard("", "") == 1.0


def test_semantic_dedup_fallback_drops_near_duplicate_above_threshold():
    """近义改写（措辞微调，实测词法相似度 ~0.94 > 0.92）应被判定语义重复。"""
    near_a = load_fixture("semantic_near_dup_a")
    near_b = load_fixture("semantic_near_dup_b")
    backend = dedup.load_embedding_backend(force_fallback=True)

    kept, dropped = dedup.semantic_dedup([near_a, near_b], backend=backend)

    assert kept == [near_a]
    assert dropped == [near_b]


def test_semantic_dedup_fallback_keeps_clearly_distinct_alerts():
    """完全不同话题的告警（实测词法相似度 ~0.18-0.21，远低于 0.92）应该都保留。"""
    a = load_fixture("dependency_1")
    b = load_fixture("resource_1")
    c = load_fixture("semantic_distinct")
    backend = dedup.load_embedding_backend(force_fallback=True)

    kept, dropped = dedup.semantic_dedup([a, b, c], backend=backend)

    assert kept == [a, b, c]
    assert dropped == []


def test_semantic_dedup_mixed_batch_fallback():
    """混合批次：near-dup 对里砍掉第二条，明显不同的话题全部保留。"""
    near_a = load_fixture("semantic_near_dup_a")
    near_b = load_fixture("semantic_near_dup_b")
    distinct = load_fixture("semantic_distinct")
    backend = dedup.load_embedding_backend(force_fallback=True)

    kept, dropped = dedup.semantic_dedup([near_a, distinct, near_b], backend=backend)

    assert kept == [near_a, distinct]
    assert dropped == [near_b]


def test_dedup_pipeline_end_to_end_fallback():
    exact_a = load_fixture("exact_dup_a")
    exact_b = load_fixture("exact_dup_b")  # md5 精确重复
    near_a = load_fixture("semantic_near_dup_a")
    near_b = load_fixture("semantic_near_dup_b")  # 语义近似重复
    distinct = load_fixture("semantic_distinct")

    backend = dedup.load_embedding_backend(force_fallback=True)
    result = dedup.dedup_pipeline(
        [exact_a, near_a, exact_b, near_b, distinct], backend=backend
    )

    assert result["backend_used"] == "lexical-fallback"
    assert result["kept"] == [exact_a, near_a, distinct]
    assert result["dropped_md5_exact"] == [exact_b]
    assert result["dropped_semantic"] == [near_b]


# --- 真实 BGE 路径（可用时才跑；本环境系统 python3 没装 sentence-transformers，会被跳过） ---


def test_semantic_dedup_real_bge_backend_if_available():
    pytest.importorskip("sentence_transformers")
    backend = dedup.load_embedding_backend()
    if backend.name != "bge-real":
        pytest.skip("sentence-transformers 已安装但模型加载失败（大概率是无网络/无缓存），跳过真实路径用例")

    near_a = load_fixture("semantic_near_dup_a")
    near_b = load_fixture("semantic_near_dup_b")
    distinct = load_fixture("semantic_distinct")

    kept, dropped = dedup.semantic_dedup([near_a, near_b, distinct], backend=backend)

    assert kept == [near_a, distinct]
    assert dropped == [near_b]
