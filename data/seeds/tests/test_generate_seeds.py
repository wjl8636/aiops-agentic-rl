"""对 `data/seeds/generate_seeds.py` 的校验测试。

覆盖三件事（对应任务要求）：
  1. 每条生成的告警都满足真实 alert schema 的必填字段。
  2. `kind`-隐含字段（scenario_hint 里标注的类别 / flagd flag / 历史工单 id）
     与「是哪条生成路径产出的」保持一致——用生成器自己的数据表做白盒核对，
     不是靠脆弱的关键词猜测。
  3. 不存在两条字节级完全相同的种子。

同时对磁盘上已经跑出来的 `data/seeds/generated/*.json`（脚本默认产出的 30 条真实样本）
做一遍同样的校验，确保「文档承诺的产物」和「代码实际产出」一致。
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

SEEDS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SEEDS_DIR))

import generate_seeds as gs  # noqa: E402

# 历史工单反演路现在默认会尝试真的走子进程查询 Milvus（见 `_search_historical_tickets_via_milvus()`）。
# 测试套件不应该依赖一个真的跑起来的 Milvus 实例才能跑——模块级把这个函数换成一个恒返回
# 空列表的假实现，让 `_milvus_candidate_records()` 恒为空池，`generate_historical_reverse_derivation()`
# 因此稳定退回静态兜底 `HISTORICAL_INCIDENT_SUMMARIES`，跟改造前的行为完全一致、确定性可复现。
# 需要测试「真实检索路径」本身的用例，会在各自的测试函数里用 `monkeypatch` 局部覆盖
# `gs._milvus_candidate_records`（而不是这里的底层桥接函数），测试结束后自动恢复。
gs._search_historical_tickets_via_milvus = lambda *args, **kwargs: []

REQUIRED_TOP_FIELDS = {"alertname", "service", "labels", "annotations", "startsAt", "scenario_hint"}
REQUIRED_LABEL_FIELDS = {"service", "severity", "job"}
REQUIRED_ANNOTATION_FIELDS = {"summary", "description"}
VALID_SEVERITIES = {"critical", "warning", "info"}
ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

GENERATED_DIR = SEEDS_DIR / "generated"


# ---------------------------------------------------------------------------
# 内存态生成结果（不依赖磁盘产物，独立可跑）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return gs.generate_seeds(count=30, seed=42)


def _assert_schema_valid(alert: dict) -> None:
    missing = REQUIRED_TOP_FIELDS - alert.keys()
    assert not missing, f"缺少必填字段: {missing}"

    assert isinstance(alert["alertname"], str) and alert["alertname"]
    assert isinstance(alert["service"], str) and alert["service"]
    assert isinstance(alert["scenario_hint"], str) and alert["scenario_hint"]

    labels = alert["labels"]
    assert REQUIRED_LABEL_FIELDS.issubset(labels.keys()), f"labels 缺字段: {labels}"
    assert labels["service"] == alert["service"], "labels.service 必须与顶层 service 一致"
    assert labels["severity"] in VALID_SEVERITIES, f"未知 severity: {labels['severity']}"
    assert isinstance(labels["job"], str) and labels["job"]

    annotations = alert["annotations"]
    assert REQUIRED_ANNOTATION_FIELDS.issubset(annotations.keys()), f"annotations 缺字段: {annotations}"
    assert isinstance(annotations["summary"], str) and annotations["summary"]
    assert isinstance(annotations["description"], str) and len(annotations["description"]) > 10

    assert ISO8601_RE.match(alert["startsAt"]), f"startsAt 不是 ISO8601 UTC: {alert['startsAt']}"

    # alert 本身不应该带 kind 字段——kind 是 Diagnosis 的输出字段，不是告警输入字段
    # （见 AIops-agent/agent/core/schema.py）。这里显式守住，防止误把内部标注写进真实 schema。
    assert "kind" not in alert


def test_required_field_schema(manifest):
    assert len(manifest) == 30
    for entry in manifest:
        _assert_schema_valid(entry["alert"])


def test_intended_kind_is_valid_enum_value(manifest):
    for entry in manifest:
        assert entry["intended_kind"] in gs.KIND_VALUES


def test_no_two_seeds_byte_identical(manifest):
    serialized = [
        json.dumps(entry["alert"], ensure_ascii=False, sort_keys=True) for entry in manifest
    ]
    assert len(serialized) == len(set(serialized)), "存在两条字节级完全相同的种子"


def test_generation_is_deterministic_for_fixed_seed():
    m1 = gs.generate_seeds(count=30, seed=42)
    m2 = gs.generate_seeds(count=30, seed=42)
    s1 = [json.dumps(e["alert"], sort_keys=True, ensure_ascii=False) for e in m1]
    s2 = [json.dumps(e["alert"], sort_keys=True, ensure_ascii=False) for e in m2]
    assert s1 == s2


# ---------------------------------------------------------------------------
# kind-隐含字段一致性：白盒核对，比对生成器自己的数据表
# ---------------------------------------------------------------------------


_PARAM_CATEGORY_TO_KIND = {
    "dependency": "dependency",
    "resource_cpu": "resource",
    "queue_backlog": "resource",
    "deploy_regression": "deploy_regression",
}


def test_parametrized_scenario_hint_matches_intended_kind(manifest):
    hint_re = re.compile(r"\[参数化扩增/(\w+)\]")
    seen = 0
    for entry in manifest:
        if entry["generation_path"] != "parametrized":
            continue
        seen += 1
        m = hint_re.search(entry["alert"]["scenario_hint"])
        assert m, f"参数化扩增场景缺少可解析的 scenario_hint 标注: {entry['alert']['scenario_hint']}"
        category = m.group(1)
        assert category in _PARAM_CATEGORY_TO_KIND, f"未知参数化类别: {category}"
        assert _PARAM_CATEGORY_TO_KIND[category] == entry["intended_kind"], (
            f"参数化扩增类别 {category} 应对应 kind={_PARAM_CATEGORY_TO_KIND[category]}，"
            f"但 manifest 里标的是 {entry['intended_kind']}"
        )
    assert seen > 0


def test_flagd_single_scenario_matches_source_spec(manifest):
    single_specs = {s["flag"]: s for s in gs._flagd_single_specs()}
    hint_re = re.compile(r"\[flagd组合/单点\] flag=(\w+)")
    seen = 0
    for entry in manifest:
        if entry["generation_path"] != "flagd_combination":
            continue
        m = hint_re.search(entry["alert"]["scenario_hint"])
        if not m:
            continue  # 这是一条多点组合，交给下一个测试处理
        seen += 1
        flag = m.group(1)
        assert flag in gs.CONFIRMED_FLAGD_FLAGS, f"scenario_hint 引用了未经核实的 flag: {flag}"
        assert flag in single_specs, f"flag {flag} 不在单点组合的 spec 表里"
        assert single_specs[flag]["kind"] == entry["intended_kind"]
    assert seen > 0


def test_flagd_multi_scenario_matches_source_spec(manifest):
    # 用完整候选池（手写语义组合 + itertools.combinations 机械遍历过滤后的组合），
    # 跟 generate_flagd_combinations() 内部用的池子保持一致——见 _flagd_multi_combo_pool()。
    multi_specs = gs._flagd_multi_combo_pool()
    hint_re = re.compile(r"\[flagd组合/多点\] flags=([\w+]+)")
    seen = 0
    for entry in manifest:
        if entry["generation_path"] != "flagd_combination":
            continue
        m = hint_re.search(entry["alert"]["scenario_hint"])
        if not m:
            continue
        seen += 1
        flags = tuple(m.group(1).split("+"))
        for f in flags:
            assert f in gs.CONFIRMED_FLAGD_FLAGS, f"scenario_hint 引用了未经核实的 flag: {f}"
        matching = [spec for spec in multi_specs if tuple(spec["flags"]) == flags]
        assert matching, f"组合 {flags} 不在多点组合的 spec 表里"
        assert matching[0]["kind"] == entry["intended_kind"]
    assert seen > 0


def test_historical_reverse_derivation_matches_source_ticket(manifest):
    records_by_id = {r["id"]: r for r in gs.HISTORICAL_INCIDENT_SUMMARIES}
    hint_re = re.compile(r"source_ticket=([\w-]+)")
    seen = 0
    for entry in manifest:
        if entry["generation_path"] != "historical_reverse_derivation":
            continue
        seen += 1
        m = hint_re.search(entry["alert"]["scenario_hint"])
        assert m, "历史工单反演场景缺少 source_ticket 标注"
        ticket_id = m.group(1)
        assert ticket_id in records_by_id, f"未知历史工单 id: {ticket_id}"
        assert records_by_id[ticket_id]["kind"] == entry["intended_kind"]
        assert records_by_id[ticket_id]["service"] == entry["alert"]["service"]
    assert seen > 0


# ---------------------------------------------------------------------------
# 历史工单反演路的真实 Milvus 检索集成：真实检索路径（mocked，不需要真跑起来的 Milvus）、
# 混合/完全退回静态兜底路径、格式兼容性。
# 顶部已经把 `gs._search_historical_tickets_via_milvus` 换成恒返回 `[]` 的假实现，这里
# 各测试再按需用 `monkeypatch` 局部覆盖更上层的 `gs._milvus_candidate_records`，测试结束
# 自动还原，不影响其它测试用到的 `manifest` fixture（它依然稳定退回静态兜底）。
# ---------------------------------------------------------------------------


def _fake_milvus_record(idx: int, service: str, kind: str, severity: str = "warning") -> dict:
    """构造一条跟 `_milvus_hit_to_ticket_record()` 输出同构的假「真实检索池」记录。"""
    job = gs._SERVICE_JOB_FALLBACK[service]
    return {
        "id": f"milvus-{idx:04d}-{service}-{kind}",
        "summary": f"[mock] Milvus 检索出的历史工单 #{idx}：{service} 服务的 {kind} 类故障叙事。",
        "kind": kind,
        "service": service,
        "job": job,
        "severity": severity,
    }


def test_historical_reverse_derivation_prefers_real_pool_when_available(monkeypatch):
    """真实检索池条数 >= n 时，应该全部从真实池取，不触碰静态兜底列表。"""
    fake_pool = [
        _fake_milvus_record(0, "recommendation", "dependency"),
        _fake_milvus_record(1, "payment", "resource"),
        _fake_milvus_record(2, "cart", "deploy_regression", severity="critical"),
        _fake_milvus_record(3, "quote", "config"),
    ]
    monkeypatch.setattr(gs, "_milvus_candidate_records", lambda: list(fake_pool))

    rng = random.Random(7)
    results = gs.generate_historical_reverse_derivation(4, rng)
    assert len(results) == 4

    fake_by_id = {r["id"]: r for r in fake_pool}
    hint_re = re.compile(r"\[历史工单反演/(\w+)\] source_ticket=([\w-]+)")
    for alert, kind in results:
        m = hint_re.search(alert["scenario_hint"])
        assert m, alert["scenario_hint"]
        source, ticket_id = m.group(1), m.group(2)
        assert source == "milvus", "真实池够大时不应该退回 static"
        assert ticket_id in fake_by_id
        assert fake_by_id[ticket_id]["kind"] == kind
        assert fake_by_id[ticket_id]["service"] == alert["service"]
        # 真实检索出的记录也必须满足跟静态记录同样的 alert schema 约束
        _assert_schema_valid(alert)


def test_historical_reverse_derivation_mixes_real_and_static_when_pool_insufficient(monkeypatch):
    """真实池非空但比 n 小时：真实池全部用上，差额从静态兜底补齐（不是「有真实数据就完全
    不用静态」，也不是「静态数据完全不用真实池」）。"""
    fake_pool = [
        _fake_milvus_record(0, "ad", "resource"),
        _fake_milvus_record(1, "email", "config"),
    ]
    monkeypatch.setattr(gs, "_milvus_candidate_records", lambda: list(fake_pool))

    rng = random.Random(11)
    n = 6
    results = gs.generate_historical_reverse_derivation(n, rng)
    assert len(results) == n

    hint_re = re.compile(r"\[历史工单反演/(\w+)\] source_ticket=([\w-]+)")
    sources = []
    for alert, _kind in results:
        m = hint_re.search(alert["scenario_hint"])
        assert m
        sources.append(m.group(1))

    assert sources.count("milvus") == len(fake_pool), "真实池的记录应该全部被用上"
    assert sources.count("static") == n - len(fake_pool), "差额应该从静态兜底补齐"


def test_historical_reverse_derivation_falls_back_to_static_when_pool_empty(monkeypatch):
    """真实池为空（Milvus 不可用/检索不到有效记录）时，行为应该跟改造前完全一致：
    完全退回 `HISTORICAL_INCIDENT_SUMMARIES`。"""
    monkeypatch.setattr(gs, "_milvus_candidate_records", lambda: [])

    rng = random.Random(13)
    results = gs.generate_historical_reverse_derivation(5, rng)
    assert len(results) == 5

    records_by_id = {r["id"]: r for r in gs.HISTORICAL_INCIDENT_SUMMARIES}
    hint_re = re.compile(r"\[历史工单反演/(\w+)\] source_ticket=([\w-]+)")
    for alert, kind in results:
        m = hint_re.search(alert["scenario_hint"])
        assert m
        source, ticket_id = m.group(1), m.group(2)
        assert source == "static"
        assert ticket_id in records_by_id
        assert records_by_id[ticket_id]["kind"] == kind


def test_search_historical_tickets_via_milvus_returns_empty_when_venv_missing(monkeypatch):
    """找不到 `AIops-agent/.venv/bin/python3` 时，桥接函数应该直接返回空列表，不抛异常。"""
    monkeypatch.setattr(gs, "_aiops_agent_venv_python", lambda: None)
    assert gs._search_historical_tickets_via_milvus(["随便一个查询"]) == []


def test_search_historical_tickets_via_milvus_returns_empty_on_subprocess_error(monkeypatch):
    """子进程调用抛异常（比如权限问题/解释器不存在）时也应该降级为空列表，不向上传播异常。"""
    monkeypatch.setattr(gs, "_aiops_agent_venv_python", lambda: Path("/nonexistent/python3"))

    def _boom(*args, **kwargs):
        raise OSError("simulated subprocess failure")

    monkeypatch.setattr(gs.subprocess, "run", _boom)
    assert gs._search_historical_tickets_via_milvus(["随便一个查询"]) == []


def test_milvus_candidate_records_dedupes_and_filters_invalid_hits(monkeypatch):
    """`_milvus_candidate_records()` 应该：
      1. 丢弃 kind 不在 KIND_VALUES 里的脏记录；
      2. 丢弃 service 不在已知 11 个 otel-demo 服务 + kafka 枚举里的记录（比如集合里遗留的
         "unknown" 低置信度记录，或畸形 service 字符串）；
      3. 对 (service, kind, summary 前缀) 相同的命中去重，只留一条。
    """
    raw_hits = [
        {"suspect_service": "recommendation", "kind": "dependency", "summary": "同一条工单被两条查询各命中一次" * 1, "route": "feishu_online_op"},
        {"suspect_service": "recommendation", "kind": "dependency", "summary": "同一条工单被两条查询各命中一次", "route": "feishu_online_op"},
        {"suspect_service": "unknown", "kind": "resource", "summary": "低置信度兜底遗留记录", "route": "feishu_low_confidence"},
        {"suspect_service": "kafka（消费者：fraud-detection/accounting）", "kind": "resource", "summary": "畸形 service 字符串的遗留记录", "route": "feishu_online_op"},
        {"suspect_service": "ad", "kind": "not_a_real_kind", "summary": "kind 不合法的脏记录", "route": "feishu_online_op"},
        {"suspect_service": "ad", "kind": "resource", "summary": "", "route": "feishu_online_op"},
        {"suspect_service": "payment", "kind": "deploy_regression", "summary": "一条干净有效的记录", "route": "auto_remediated_and_code_fix_pr"},
    ]
    monkeypatch.setattr(gs, "_search_historical_tickets_via_milvus", lambda *a, **k: raw_hits)

    pool = gs._milvus_candidate_records()
    services = {r["service"] for r in pool}
    assert services == {"recommendation", "payment"}
    assert len(pool) == 2  # 去重后的 recommendation 一条 + payment 一条
    payment_rec = next(r for r in pool if r["service"] == "payment")
    assert payment_rec["kind"] == "deploy_regression"
    assert payment_rec["severity"] == "critical"  # deploy_regression 按启发式规则应判 critical
    assert payment_rec["job"] == gs._SERVICE_JOB_FALLBACK["payment"]


# ---------------------------------------------------------------------------
# 定点补充：generate_pure_code_fix_and_info_only() / append_seeds()
# ---------------------------------------------------------------------------


def test_generate_pure_code_fix_and_info_only_schema_and_labels():
    results = gs.generate_pure_code_fix_and_info_only()
    assert len(results) == 2
    for alert, kind in results:
        _assert_schema_valid(alert)
        assert kind in gs.KIND_VALUES

    code_fix_alert, _ = results[0]
    info_only_alert, _ = results[1]
    assert "pure_code_fix_ranking" in code_fix_alert["scenario_hint"]
    assert code_fix_alert["service"] == "recommendation"
    assert "pure_info_only" in info_only_alert["scenario_hint"]
    assert info_only_alert["labels"]["severity"] == "info"


def test_generate_pure_code_fix_and_info_only_is_deterministic():
    r1 = gs.generate_pure_code_fix_and_info_only()
    r2 = gs.generate_pure_code_fix_and_info_only()
    s1 = [json.dumps(a, sort_keys=True, ensure_ascii=False) for a, _ in r1]
    s2 = [json.dumps(a, sort_keys=True, ensure_ascii=False) for a, _ in r2]
    assert s1 == s2


def test_append_seeds_continues_numbering_and_does_not_touch_existing_files(tmp_path):
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    first_batch = gs.generate_seeds(count=3, seed=1)
    gs.write_seeds(first_batch, out_dir)
    existing_files_before = {p.name for p in out_dir.glob("seed_*.json")}

    new_entries = [
        {"alert": alert, "generation_path": "parametrized", "intended_kind": kind}
        for alert, kind in gs.generate_pure_code_fix_and_info_only()
    ]
    written = gs.append_seeds(new_entries, out_dir)

    assert written == ["seed_0004_parametrized_deploy_regression.json", "seed_0005_parametrized_resource.json"]
    # 已有的种子文件必须原样保留，一个字节都不能被 append_seeds 碰到。
    assert existing_files_before.issubset({p.name for p in out_dir.glob("seed_*.json")})

    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 5
    assert manifest[-2]["file"] == "seed_0004_parametrized_deploy_regression.json"
    assert manifest[-1]["file"] == "seed_0005_parametrized_resource.json"

    on_disk = {p.name for p in out_dir.glob("seed_*.json")}
    assert on_disk == {r["file"] for r in manifest}


# ---------------------------------------------------------------------------
# 2026-09 补充：generate_pure_code_fix_and_info_only_v2()（第二批 code_fix/info_only）
# ---------------------------------------------------------------------------


def test_generate_pure_code_fix_and_info_only_v2_schema_and_labels():
    results = gs.generate_pure_code_fix_and_info_only_v2()
    assert len(results) == 5
    for alert, kind in results:
        _assert_schema_valid(alert)
        assert kind in gs.KIND_VALUES

    dedupe_alert, ranking_alert, cart_alert, payment_alert, shipping_alert = [a for a, _ in results]

    assert "pure_code_fix_dedupe" in dedupe_alert["scenario_hint"]
    assert dedupe_alert["service"] == "recommendation"

    assert "pure_code_fix_ranking" in ranking_alert["scenario_hint"]
    assert ranking_alert["service"] == "recommendation"

    for info_alert, expected_service in (
        (cart_alert, "cart"),
        (payment_alert, "payment"),
        (shipping_alert, "shipping"),
    ):
        assert "pure_info_only" in info_alert["scenario_hint"]
        assert info_alert["service"] == expected_service

    kinds = [kind for _, kind in results]
    assert kinds == [
        "deploy_regression",
        "deploy_regression",
        "dependency",
        "config",
        "deploy_regression",
    ]


def test_generate_pure_code_fix_and_info_only_v2_is_deterministic():
    r1 = gs.generate_pure_code_fix_and_info_only_v2()
    r2 = gs.generate_pure_code_fix_and_info_only_v2()
    s1 = [json.dumps(a, sort_keys=True, ensure_ascii=False) for a, _ in r1]
    s2 = [json.dumps(a, sort_keys=True, ensure_ascii=False) for a, _ in r2]
    assert s1 == s2


def test_append_seeds_v2_continues_numbering_after_v1(tmp_path):
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    first_batch = gs.generate_seeds(count=3, seed=1)
    gs.write_seeds(first_batch, out_dir)

    v1_entries = [
        {"alert": alert, "generation_path": "parametrized", "intended_kind": kind}
        for alert, kind in gs.generate_pure_code_fix_and_info_only()
    ]
    gs.append_seeds(v1_entries, out_dir)
    existing_files_before = {p.name for p in out_dir.glob("seed_*.json")}

    v2_entries = [
        {"alert": alert, "generation_path": "parametrized", "intended_kind": kind}
        for alert, kind in gs.generate_pure_code_fix_and_info_only_v2()
    ]
    written = gs.append_seeds(v2_entries, out_dir)

    assert written == [
        "seed_0006_parametrized_deploy_regression.json",
        "seed_0007_parametrized_deploy_regression.json",
        "seed_0008_parametrized_dependency.json",
        "seed_0009_parametrized_config.json",
        "seed_0010_parametrized_deploy_regression.json",
    ]
    # v1 + 原有种子文件必须原样保留，一个字节都不能被第二次 append_seeds 碰到。
    assert existing_files_before.issubset({p.name for p in out_dir.glob("seed_*.json")})

    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 10
    on_disk = {p.name for p in out_dir.glob("seed_*.json")}
    assert on_disk == {r["file"] for r in manifest}


def test_all_three_generation_paths_are_present(manifest):
    paths = {entry["generation_path"] for entry in manifest}
    assert paths == {"parametrized", "flagd_combination", "historical_reverse_derivation"}


# ---------------------------------------------------------------------------
# 2026-09 补充：generate_historical_reverse_derivation_supplement_v1()
# （手写补 historical_reverse_derivation 路径，定点填补 config kind 空白）
# ---------------------------------------------------------------------------


def test_generate_historical_reverse_derivation_supplement_v1_schema_and_kinds():
    results = gs.generate_historical_reverse_derivation_supplement_v1()
    assert len(results) == 5
    for alert, kind in results:
        _assert_schema_valid(alert)
        assert kind in gs.KIND_VALUES

    kinds = [kind for _, kind in results]
    assert kinds == ["config", "config", "resource", "resource", "deploy_regression"]

    for alert, _ in results:
        assert alert["scenario_hint"].startswith("[历史工单反演/static]")


def test_generate_historical_reverse_derivation_supplement_v1_config_seeds_use_image_slow_load_flag():
    results = gs.generate_historical_reverse_derivation_supplement_v1()
    config_alerts = [alert for alert, kind in results if kind == "config"]
    assert len(config_alerts) == 2
    for alert in config_alerts:
        assert alert["service"] == "frontend"
        assert "flag=imageSlowLoad" in alert["scenario_hint"]


def test_generate_historical_reverse_derivation_supplement_v1_resource_seeds_use_kafka_flag():
    results = gs.generate_historical_reverse_derivation_supplement_v1()
    resource_alerts = [alert for alert, kind in results if kind == "resource"]
    assert len(resource_alerts) == 2
    for alert in resource_alerts:
        assert alert["service"] == "kafka"
        assert "flag=kafkaQueueProblems" in alert["scenario_hint"]


def test_generate_historical_reverse_derivation_supplement_v1_leak_seed_has_no_flag_marker():
    """recommendation 内存泄漏没有对应的真实 flagd flag（机制是换容器 fixture），这条种子的
    scenario_hint 必须让 fault_injection.parse_flags() 解析不出任何 flag 名（用真实的
    parse_flags() 校验，而不是脆弱的字面 "flag=" 子串匹配——hint 里的说明文字本身就会提到
    "flag=" 这个词，字面子串匹配会把说明文字误判成标注）。解析不出 flag 才会落到
    fault_injection.py::_inject_historical_fault() 的内存/OOM 关键词兜底分支，正确路由到
    真实泄漏 fixture；如果不小心写成能被 parse_flags() 解析出来的假 flag 名，
    activate_flags() 会对不存在的 flag 名静默跳过，反而绕过了关键词兜底、变成完全不注入。"""
    cold_start_tests_dir = Path(__file__).resolve().parent.parent.parent / "cold_start"
    if str(cold_start_tests_dir) not in sys.path:
        sys.path.insert(0, str(cold_start_tests_dir))
    import fault_injection as fi  # noqa: E402

    results = gs.generate_historical_reverse_derivation_supplement_v1()
    leak_alerts = [alert for alert, kind in results if kind == "deploy_regression"]
    assert len(leak_alerts) == 1
    leak_alert = leak_alerts[0]
    assert leak_alert["service"] == "recommendation"
    assert fi.parse_flags(leak_alert["scenario_hint"]) == []
    text = leak_alert["annotations"]["summary"] + leak_alert["annotations"]["description"]
    assert any(k in text for k in fi._HIST_LEAK_KEYWORDS)
    assert not any(k in text for k in fi._HIST_CERT_KEYWORDS)


def test_generate_historical_reverse_derivation_supplement_v1_is_deterministic():
    r1 = gs.generate_historical_reverse_derivation_supplement_v1()
    r2 = gs.generate_historical_reverse_derivation_supplement_v1()
    s1 = [json.dumps(a, sort_keys=True, ensure_ascii=False) for a, _ in r1]
    s2 = [json.dumps(a, sort_keys=True, ensure_ascii=False) for a, _ in r2]
    assert s1 == s2


def test_generate_historical_reverse_derivation_supplement_v1_no_two_seeds_byte_identical():
    results = gs.generate_historical_reverse_derivation_supplement_v1()
    serialized = [json.dumps(alert, ensure_ascii=False, sort_keys=True) for alert, _ in results]
    assert len(serialized) == len(set(serialized))


def test_append_seeds_historical_supplement_v1_continues_numbering(tmp_path):
    out_dir = tmp_path / "generated"
    out_dir.mkdir()
    first_batch = gs.generate_seeds(count=3, seed=1)
    gs.write_seeds(first_batch, out_dir)
    existing_files_before = {p.name for p in out_dir.glob("seed_*.json")}

    entries = [
        {"alert": alert, "generation_path": "historical_reverse_derivation", "intended_kind": kind}
        for alert, kind in gs.generate_historical_reverse_derivation_supplement_v1()
    ]
    written = gs.append_seeds(entries, out_dir)

    assert written == [
        "seed_0004_historical_reverse_derivation_config.json",
        "seed_0005_historical_reverse_derivation_config.json",
        "seed_0006_historical_reverse_derivation_resource.json",
        "seed_0007_historical_reverse_derivation_resource.json",
        "seed_0008_historical_reverse_derivation_deploy_regression.json",
    ]
    # 已有种子文件必须原样保留，一个字节都不能被 append_seeds 碰到。
    assert existing_files_before.issubset({p.name for p in out_dir.glob("seed_*.json")})

    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 8
    on_disk = {p.name for p in out_dir.glob("seed_*.json")}
    assert on_disk == {r["file"] for r in manifest}


# ---------------------------------------------------------------------------
# flagd 组合的机械遍历（itertools.combinations）：数量、语义过滤规则、格式兼容性。
# ---------------------------------------------------------------------------


def test_mechanical_combo_count_matches_filtered_combinatorics():
    """C(11,2)=55、C(11,3)=165 是文档承诺过的全量；过滤掉互斥对/过量 hard-down 组合后
    应该分别剩 51 条（55 - 4 个互斥对）和 129 条（165 - 36 条命中互斥对或 3 个 hard-down
    的组合），总计 180 条——真的比之前手写的 4 条多了两个数量级。
    """
    mech = gs._flagd_mechanical_combo_specs()
    r2 = [s for s in mech if len(s["flags"]) == 2]
    r3 = [s for s in mech if len(s["flags"]) == 3]
    assert len(r2) == 55 - 4
    assert len(r3) == 165 - 36
    assert len(mech) == 180


def test_mechanical_combo_pool_is_additive_not_destructive():
    """新的机械遍历是在原有 4 条手写组合基础上的补充，不能把它们挤掉。"""
    hand_written = gs._flagd_multi_combo_specs()
    pool = gs._flagd_multi_combo_pool()
    assert len(hand_written) == 4
    hand_written_flag_tuples = {tuple(s["flags"]) for s in hand_written}
    pool_flag_tuples = {tuple(s["flags"]) for s in pool}
    assert hand_written_flag_tuples.issubset(pool_flag_tuples)
    assert len(pool) == len(hand_written) + len(gs._flagd_mechanical_combo_specs())


@pytest.mark.parametrize(
    "flags",
    [
        ("paymentFailure", "paymentUnreachable"),
        ("adFailure", "adHighCpu"),
        ("adFailure", "adManualGc"),
        ("loadGeneratorFloodHomepage", "imageSlowLoad"),
        # 互斥对被第三个 flag 扩展成三元组，同样应该被排除。
        ("paymentFailure", "paymentUnreachable", "cartFailure"),
        ("kafkaQueueProblems", "adFailure", "adHighCpu"),
        # 3 个 hard-down flag 同时出现（"环境被同时炸了三次"），即便两两之间都不在
        # _MUTUALLY_EXCLUSIVE_FLAG_PAIRS 里，也应该被排除。
        ("adFailure", "cartFailure", "paymentUnreachable"),
    ],
)
def test_semantically_invalid_combos_are_excluded(flags):
    assert gs._is_semantically_valid_combo(flags) is False
    mech = gs._flagd_mechanical_combo_specs()
    assert not any(frozenset(s["flags"]) == frozenset(flags) for s in mech), (
        f"不合理组合 {flags} 本该被过滤掉，却出现在机械遍历结果里"
    )


@pytest.mark.parametrize(
    "flags",
    [
        ("adHighCpu", "adManualGc"),  # 同服务复合资源故障，跟手写组合里的先例一致，合理
        ("cartFailure", "paymentUnreachable"),  # 跟手写组合里的先例一致，合理
        ("productCatalogFailure", "kafkaQueueProblems"),  # 不同服务、语义不冲突，合理
    ],
)
def test_semantically_valid_combos_are_kept(flags):
    assert gs._is_semantically_valid_combo(flags) is True
    mech = gs._flagd_mechanical_combo_specs()
    # itertools.combinations 按 CONFIRMED_FLAGD_FLAGS 的字典插入顺序遍历，不保证跟测试
    # 参数里手写的 flag 顺序一致，用 frozenset 比较组合本身，不比较元素顺序。
    assert any(frozenset(s["flags"]) == frozenset(flags) for s in mech), (
        f"合理组合 {flags} 应该出现在机械遍历结果里，却被漏掉了"
    )


def test_mechanical_combo_specs_share_schema_with_hand_written_specs():
    """机械遍历产出的 spec 字典必须跟手写 `_flagd_multi_combo_specs()` 同构，
    这样 `generate_flagd_combinations()` 才能无差别地消费两者。"""
    hand_written_keys = {frozenset(s.keys()) for s in gs._flagd_multi_combo_specs()}
    mech = gs._flagd_mechanical_combo_specs()
    assert mech, "机械遍历不应该产出空列表"
    for spec in mech:
        assert frozenset(spec.keys()) in hand_written_keys
        assert spec["kind"] in gs.KIND_VALUES
        assert 2 <= len(spec["flags"]) <= 3
        assert all(f in gs.CONFIRMED_FLAGD_FLAGS for f in spec["flags"])
        assert spec["service"] and spec["job"]
        assert spec["alertname"] and spec["summary"] and spec["desc"]


def test_mechanical_combo_kind_uses_resource_over_dependency_precedent():
    """已有先例：kafkaQueueProblems（resource）+ recommendationCacheFailure（dependency）
    在手写组合 `_flagd_multi_combo_specs()` 里被判定 kind="resource"。机械遍历对同一种
    kind 冲突应该给出跟这条先例一致的结论。"""
    assert gs._infer_combo_kind(("kafkaQueueProblems", "recommendationCacheFailure")) == "resource"


def test_generate_flagd_combinations_can_exceed_old_nine_pattern_pool():
    """扩容后候选池（5 单点 + 184 多点）应该远大于旧的 9 条手写 pattern，采样 n 大一些
    时，flagd_combination 路径产出的组合应该出现明显比旧版本更丰富的 flag 组合分布。"""
    rng = random.Random(123)
    results = gs.generate_flagd_combinations(120, rng)
    assert len(results) == 120

    def _combo_signature(alert: dict) -> str:
        m = re.search(r"\[flagd组合/(?:单点|多点)\] (?:flag|flags)=([\w+]+)", alert["scenario_hint"])
        assert m
        return m.group(1)

    distinct_combos = {_combo_signature(alert) for alert, _kind in results}
    # 旧版本候选池只有 9 条 base pattern；新版本至少应该看到几十种不同组合。
    assert len(distinct_combos) > 9 * 2


# ---------------------------------------------------------------------------
# 磁盘产物校验：磁盘上跑出来的种子文件数量应该跟 `_manifest.json` 记录的一致
# （具体条数由 generate_seeds.py --count 决定，不是固定 30——仓库可能提交的是
# 30 条 demo 规模，也可能是真实复现时扩到的 700 条，这条测试对两者都应该成立）。
# ---------------------------------------------------------------------------


def _on_disk_alert_files() -> list[Path]:
    if not GENERATED_DIR.exists():
        return []
    return sorted(p for p in GENERATED_DIR.glob("seed_*.json"))


@pytest.mark.skipif(not GENERATED_DIR.exists(), reason="data/seeds/generated 尚未生成，先跑一次 generate_seeds.py")
def test_on_disk_seed_files_match_schema_and_are_unique():
    files = _on_disk_alert_files()
    manifest_path = GENERATED_DIR / "_manifest.json"
    assert manifest_path.exists()
    expected_count = len(json.loads(manifest_path.read_text(encoding="utf-8")))
    assert len(files) == expected_count, f"预期 {expected_count} 个种子文件，实际 {len(files)} 个"

    raw_contents = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        raw_contents.append(raw)
        alert = json.loads(raw)
        _assert_schema_valid(alert)

    assert len(raw_contents) == len(set(raw_contents)), "磁盘上存在两个字节级完全相同的种子文件"


@pytest.mark.skipif(not GENERATED_DIR.exists(), reason="data/seeds/generated 尚未生成，先跑一次 generate_seeds.py")
def test_on_disk_manifest_matches_files():
    manifest_path = GENERATED_DIR / "_manifest.json"
    assert manifest_path.exists()
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_on_disk = {p.name for p in _on_disk_alert_files()}
    manifest_files = {r["file"] for r in records}
    assert files_on_disk == manifest_files
    for r in records:
        assert r["intended_kind"] in gs.KIND_VALUES
        assert r["generation_path"] in {
            "parametrized",
            "flagd_combination",
            "historical_reverse_derivation",
        } or r["generation_path"].startswith("v9_aug_")
