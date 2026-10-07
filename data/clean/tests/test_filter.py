from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import filter as filter_mod
from conftest import load_fixture


def _ingest_real_fingerprint_available() -> bool:
    """探测子模块的真实 alert_fingerprint 是否可 import（用于挑一条跳过/断言的逻辑）。"""
    repo_root = Path(__file__).resolve().parents[3]
    aiops_agent_dir = repo_root / "AIops-agent"
    path_str = str(aiops_agent_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        import agent.integrations.ingest  # noqa: F401

        return True
    except Exception:
        return False


def _reference_fingerprint(alert: dict) -> str:
    """跟 AIops-agent/agent/integrations/ingest.py::alert_fingerprint 完全一致的独立重实现，
    专门用来在测试里跟 filter.py 的结果对拍，不共享代码路径（避免「测试和被测代码抄同一段逻辑,
    抄错了也测不出来」）。"""
    name = alert.get("alertname") or alert.get("name") or ""
    labels = alert.get("labels", {}) or {}
    service = alert.get("service") or labels.get("service") or labels.get("job") or ""
    severity = labels.get("severity", "")
    key = f"{name}|{service}|{severity}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 指纹一致性：filter.py 用的指纹函数（无论走真实 import 还是本地回退）跟 ingest.py 的真实算法一致
# ---------------------------------------------------------------------------


def test_fingerprint_fn_matches_reference_on_real_alerts():
    dep = load_fixture("dependency_1")
    res = load_fixture("resource_1")
    fp_fn = filter_mod.get_fingerprint_fn()

    assert fp_fn(dep) == _reference_fingerprint(dep)
    assert fp_fn(res) == _reference_fingerprint(res)


def test_fingerprint_fn_matches_reference_on_real_ingest_alerts():
    """跟仓库里真实的 AIops-agent/alerts/s1.json 对拍（如果子模块存在）。"""
    repo_root = Path(__file__).resolve().parents[3]
    s1_path = repo_root / "AIops-agent" / "alerts" / "s1.json"
    if not s1_path.is_file():
        return  # 子模块未 checkout 时跳过，不让测试因为环境缺文件而失败
    import json

    alert = json.loads(s1_path.read_text(encoding="utf-8"))
    fp_fn = filter_mod.get_fingerprint_fn()
    assert fp_fn(alert) == _reference_fingerprint(alert)


def test_fallback_fingerprint_matches_reference():
    dep = load_fixture("dependency_1")
    assert filter_mod._fallback_alert_fingerprint(dep) == _reference_fingerprint(dep)


def test_real_and_fallback_fingerprint_agree_when_both_available():
    """如果真的能从子模块 import 到 ingest.alert_fingerprint，它跟本地回退实现必须逐字节一致
    （这正是模块 docstring 承诺的『两处实现必须保持字节级一致』）。"""
    if not _ingest_real_fingerprint_available():
        return
    from agent.integrations.ingest import alert_fingerprint as real_fp

    for name in ["dependency_1", "resource_1", "hybrid_1", "low_confidence_1"]:
        alert = load_fixture(name)
        assert real_fp(alert) == filter_mod._fallback_alert_fingerprint(alert)


# ---------------------------------------------------------------------------
# 幂等去重压掉的重复项：指纹碰撞过滤
# ---------------------------------------------------------------------------


def test_filter_fingerprint_collisions_drops_second_occurrence():
    a = load_fixture("fingerprint_collision_a")
    b = load_fixture("fingerprint_collision_b")  # 同 alertname|service|severity，内容不同
    other = load_fixture("dependency_1")

    assert filter_mod.get_fingerprint_fn()(a) == filter_mod.get_fingerprint_fn()(b)

    kept, dropped = filter_mod.filter_fingerprint_collisions([a, other, b])

    assert kept == [a, other]
    assert dropped == [b]


def test_filter_fingerprint_collisions_keeps_distinct_fingerprints():
    a = load_fixture("dependency_1")
    b = load_fixture("resource_1")
    c = load_fixture("deploy_regression_1")

    kept, dropped = filter_mod.filter_fingerprint_collisions([a, b, c])

    assert kept == [a, b, c]
    assert dropped == []


# ---------------------------------------------------------------------------
# 不可复现类过滤（启发式代理）
# ---------------------------------------------------------------------------


def test_is_likely_unreproducible_flags_external_network_alert():
    alert = load_fixture("unreproducible_external_net")
    flagged, reasons = filter_mod.is_likely_unreproducible(alert)
    assert flagged is True
    assert len(reasons) >= 1


def test_is_likely_unreproducible_keeps_alert_with_clear_fixture_hook():
    alert = load_fixture("reproducible_with_hook")
    flagged, reasons = filter_mod.is_likely_unreproducible(alert)
    assert flagged is False
    assert reasons == []


def test_is_likely_unreproducible_keeps_known_service_normal_alerts():
    for name in ["dependency_1", "resource_1", "deploy_regression_1", "hybrid_1"]:
        alert = load_fixture(name)
        flagged, _ = filter_mod.is_likely_unreproducible(alert)
        assert flagged is False, f"{name} 不应被判定为不可复现"


def test_filter_unreproducible_pipeline_separates_flagged_and_kept():
    good = load_fixture("dependency_1")
    bad = load_fixture("unreproducible_external_net")

    kept, dropped = filter_mod.filter_unreproducible([good, bad])

    assert kept == [good]
    assert len(dropped) == 1
    dropped_alert, reasons = dropped[0]
    assert dropped_alert == bad
    assert len(reasons) >= 1


# ---------------------------------------------------------------------------
# 组合管线
# ---------------------------------------------------------------------------


def test_quality_filter_pipeline_end_to_end():
    good = load_fixture("dependency_1")
    unrepro = load_fixture("unreproducible_external_net")
    fp_a = load_fixture("fingerprint_collision_a")
    fp_b = load_fixture("fingerprint_collision_b")

    result = filter_mod.quality_filter_pipeline([good, unrepro, fp_a, fp_b])

    assert result["kept"] == [good, fp_a]
    assert [a for a, _ in result["dropped_unreproducible"]] == [unrepro]
    assert result["dropped_fingerprint_collision"] == [fp_b]
