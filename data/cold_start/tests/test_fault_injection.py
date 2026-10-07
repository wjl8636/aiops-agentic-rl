from __future__ import annotations

import json

import fault_injection as fi


def _write_config(path, flags):
    path.write_text(json.dumps({"flags": flags}), encoding="utf-8")


def test_parse_flags_single():
    hint = "[参数化扩增/dependency] flag=paymentFailure service=payment severity=critical：说明"
    assert fi.parse_flags(hint) == ["paymentFailure"]


def test_parse_flags_combination():
    hint = "[flagd组合/多点] flags=adHighCpu+adManualGc：多个 flagd 故障开关同时注入"
    assert fi.parse_flags(hint) == ["adHighCpu", "adManualGc"]


def test_parse_flags_absent_for_non_flagd_scenarios():
    hint = "[参数化扩增/resource_cpu] threshold=90% severity=warning：验证资源型 CPU 场景"
    assert fi.parse_flags(hint) == []


def test_activate_flags_and_reset(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(
        config_path,
        {
            "cartFailure": {"defaultVariant": "off", "variants": {"on": True, "off": False}},
            "adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}},
        },
    )

    touched = fi.activate_flags(config_path, ["cartFailure"])
    assert touched == ["cartFailure"]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["cartFailure"]["defaultVariant"] == "on"
    assert cfg["flags"]["adHighCpu"]["defaultVariant"] == "off"

    fi.reset_all_flags(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["cartFailure"]["defaultVariant"] == "off"
    assert cfg["flags"]["adHighCpu"]["defaultVariant"] == "off"


def test_activate_flags_skips_unknown_flag_names(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(
        config_path,
        {"cartFailure": {"defaultVariant": "off", "variants": {"on": True, "off": False}}},
    )
    touched = fi.activate_flags(config_path, ["notARealFlag"])
    assert touched == []


def test_activate_flags_picks_most_severe_non_boolean_variant(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(
        config_path,
        {
            "paymentFailure": {
                "defaultVariant": "off",
                "variants": {"100%": 1, "90%": 0.95, "off": 0},
            },
            "imageSlowLoad": {
                "defaultVariant": "off",
                "variants": {"10sec": 10, "5sec": 5, "off": 0},
            },
        },
    )
    touched = fi.activate_flags(config_path, ["paymentFailure", "imageSlowLoad"])
    assert set(touched) == {"paymentFailure", "imageSlowLoad"}
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["paymentFailure"]["defaultVariant"] == "100%"
    assert cfg["flags"]["imageSlowLoad"]["defaultVariant"] == "10sec"


def test_inject_alert_fault_noop_without_config_path():
    alert = {"scenario_hint": "flag=paymentFailure service=payment"}
    assert fi.inject_alert_fault(alert, None) == []


def test_inject_alert_fault_noop_when_hint_has_no_flag(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    alert = {"scenario_hint": "threshold=90% severity=warning"}
    assert fi.inject_alert_fault(alert, config_path) == []


def test_inject_alert_fault_sets_flag(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    monkeypatch.setattr(fi.time, "sleep", lambda _seconds: None)
    alert = {"scenario_hint": "flag=adHighCpu service=ad severity=warning"}
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["adHighCpu"]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["adHighCpu"]["defaultVariant"] == "on"


def test_inject_alert_fault_handles_percentage_style_flag(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(
        config_path,
        {"paymentFailure": {"defaultVariant": "off", "variants": {"100%": 1, "90%": 0.95, "off": 0}}},
    )
    monkeypatch.setattr(fi.time, "sleep", lambda _seconds: None)
    alert = {"scenario_hint": "flag=paymentFailure service=payment severity=critical"}
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["paymentFailure"]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["paymentFailure"]["defaultVariant"] == "100%"


def test_inject_alert_fault_resource_cpu_tag_sets_ad_high_cpu(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    monkeypatch.setattr(fi.time, "sleep", lambda _seconds: None)
    alert = {
        "service": "ad",
        "scenario_hint": "[参数化扩增/resource_cpu] threshold=92% severity=critical：验证资源型 CPU 场景",
    }
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["adHighCpu"]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["adHighCpu"]["defaultVariant"] == "on"


def test_inject_alert_fault_queue_backlog_tag_sets_kafka_flag(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(
        config_path,
        {"kafkaQueueProblems": {"defaultVariant": "off", "variants": {"on": 100, "off": 0}}},
    )
    monkeypatch.setattr(fi.time, "sleep", lambda _seconds: None)
    alert = {
        "service": "kafka",
        "scenario_hint": "[参数化扩增/queue_backlog] topic=orders lag=9800 severity=warning：验证队列积压场景",
    }
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["kafkaQueueProblems"]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["kafkaQueueProblems"]["defaultVariant"] == "on"


def test_inject_alert_fault_deploy_regression_tag_calls_leak_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    calls = []
    monkeypatch.setattr(fi, "start_recommendation_leak_fixture", lambda: calls.append("start") or ["recommendation"])
    alert = {
        "service": "recommendation",
        "scenario_hint": "[参数化扩增/deploy_regression] 泄漏机制=无界 list append（unbounded_list_append）：内存泄漏 capstone",
    }
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["recommendation"]
    assert calls == ["start"]


def test_inject_alert_fault_pure_code_fix_ranking_tag_calls_ranking_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    calls = []
    monkeypatch.setattr(
        fi, "start_recommendation_ranking_fixture", lambda: calls.append("start") or ["recommendation"]
    )
    alert = {
        "service": "recommendation",
        "scenario_hint": "[参数化扩增/pure_code_fix_ranking] fixture_port=18080：排序方向写反，纯代码问题",
    }
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["recommendation"]
    assert calls == ["start"]


def test_inject_alert_fault_pure_code_fix_dedupe_tag_calls_dedupe_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    calls = []
    monkeypatch.setattr(
        fi, "start_recommendation_dedupe_fixture", lambda: calls.append("start") or ["recommendation"]
    )
    alert = {
        "service": "recommendation",
        "scenario_hint": "[参数化扩增/pure_code_fix_dedupe] fixture_port=18081：去重失效，纯代码问题",
    }
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["recommendation"]
    assert calls == ["start"]


def test_start_recommendation_dedupe_fixture_skips_when_image_missing(monkeypatch):
    def _fake_run(cmd, **kwargs):
        class _R:
            returncode = 1

        return _R()

    monkeypatch.setattr(fi.subprocess, "run", _fake_run)
    assert fi.start_recommendation_dedupe_fixture() == []


def test_start_recommendation_dedupe_fixture_publishes_dedupe_host_port(monkeypatch):
    """去重 bug 跟排序 bug 一样要从 host curl 观测，必须发布到它自己的固定端口（18081）。"""
    calls = []

    class _OkImageInspect:
        returncode = 0

    class _NoBackup:
        returncode = 1

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _OkImageInspect()
        if cmd[:2] == ["docker", "inspect"]:
            return _NoBackup()

        class _Ok:
            returncode = 0

        return _Ok()

    monkeypatch.setattr(fi.subprocess, "run", _fake_run)
    touched = fi.start_recommendation_dedupe_fixture()
    assert touched == ["recommendation"]
    run_cmd = calls[-1]
    assert "-p" in run_cmd
    assert f"{fi.RECOMMENDATION_DEDUPE_HOST_PORT}:8080" in run_cmd
    assert run_cmd[-1] == fi.RECOMMENDATION_DEDUPE_IMAGE


def test_inject_alert_fault_pure_info_only_tag_is_noop(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    alert = {
        "service": "recommendation",
        "scenario_hint": "[参数化扩增/pure_info_only] 服务当前健康，不注入任何故障",
    }
    assert fi.inject_alert_fault(alert, config_path) == []


def test_start_recommendation_ranking_fixture_skips_when_image_missing(monkeypatch):
    def _fake_run(cmd, **kwargs):
        class _R:
            returncode = 1

        return _R()

    monkeypatch.setattr(fi.subprocess, "run", _fake_run)
    assert fi.start_recommendation_ranking_fixture() == []


def test_start_recommendation_fixture_publishes_port_when_requested(monkeypatch):
    """跟内存泄漏 fixture 不同：排序 bug 要从 host curl 观测，必须发布端口。"""
    calls = []

    class _OkImageInspect:
        returncode = 0

    class _NoBackup:
        returncode = 1

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _OkImageInspect()
        if cmd[:2] == ["docker", "inspect"]:
            return _NoBackup()

        class _Ok:
            returncode = 0

        return _Ok()

    monkeypatch.setattr(fi.subprocess, "run", _fake_run)
    touched = fi._start_recommendation_fixture(
        "recommendation-ranking-fixture:test", {"PORT": "8080"}, publish_port=18080
    )
    assert touched == ["recommendation"]
    run_cmd = calls[-1]
    assert "-p" in run_cmd
    assert run_cmd[run_cmd.index("-p") + 1] == "18080:8080"


def test_inject_alert_fault_historical_cert_scenario_is_noop(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    alert = {
        "service": "checkout",
        "annotations": {
            "summary": "历史工单反演：hist-2026-07-checkout-cert-rotation-mismatch",
            "description": "checkout 调用的一个内部服务证书轮换后，调用方信任链配置没有同步更新，导致下单请求间歇性握手失败。",
        },
        "scenario_hint": "[历史工单反演/stub] source_ticket=hist-2026-07-checkout-cert-rotation-mismatch：从手写历史工单摘要样例反向抽取的告警",
    }
    assert fi.inject_alert_fault(alert, config_path) == []


def test_inject_alert_fault_historical_leak_on_recommendation_calls_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    calls = []
    monkeypatch.setattr(fi, "start_recommendation_leak_fixture", lambda: calls.append("start") or ["recommendation"])
    alert = {
        "service": "recommendation",
        "annotations": {
            "summary": "历史工单反演：hist-2026-01-cache-no-ttl",
            "description": "recommendation 团队为临时提升缓存命中率手动加了一个全局缓存字典，上线时忘了配 TTL，服务内存逐步爬升到被 OOMKilled。",
        },
        "scenario_hint": "[历史工单反演/stub] source_ticket=hist-2026-01-cache-no-ttl：从手写历史工单摘要样例反向抽取的告警",
    }
    assert fi.inject_alert_fault(alert, config_path) == ["recommendation"]
    assert calls == ["start"]


def test_inject_alert_fault_historical_leak_on_checkout_is_noop(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    alert = {
        "service": "checkout",
        "annotations": {
            "summary": "历史工单反演：hist-2026-04-checkout-session-cache-no-ttl",
            "description": "checkout 团队为减少重复计算加了一个会话级缓存字典，同样忘了配 TTL，最终把 checkout 内存吃满触发重启。",
        },
        "scenario_hint": "[历史工单反演/stub] source_ticket=hist-2026-04-checkout-session-cache-no-ttl：从手写历史工单摘要样例反向抽取的告警",
    }
    assert fi.inject_alert_fault(alert, config_path) == []


def test_inject_alert_fault_historical_dependency_on_cart_sets_cart_failure(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"cartFailure": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    monkeypatch.setattr(fi.time, "sleep", lambda _seconds: None)
    alert = {
        "service": "cart",
        "annotations": {
            "summary": "历史工单反演：hist-2025-08-cart-batch-query-timeout",
            "description": "cart 团队上线了一个购物车对账批处理脚本，脚本里的下游查询忘记设超时，占满 cart 的处理线程池，checkout 调用 cart 接口大面积超时。",
        },
        "scenario_hint": "[历史工单反演/stub] source_ticket=hist-2025-08-cart-batch-query-timeout：从手写历史工单摘要样例反向抽取的告警",
    }
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["cartFailure"]
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["cartFailure"]["defaultVariant"] == "on"


def test_inject_alert_fault_historical_queue_on_kafka_sets_kafka_flag(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(
        config_path,
        {"kafkaQueueProblems": {"defaultVariant": "off", "variants": {"on": 100, "off": 0}}},
    )
    monkeypatch.setattr(fi.time, "sleep", lambda _seconds: None)
    alert = {
        "service": "kafka",
        "annotations": {
            "summary": "历史工单反演：hist-2025-07-shipping-consumer-scale-down",
            "description": "一次容量规划失误把物流更新消费者组的副本数从 5 缩到了 2，生产速率不变但消费能力骤降，consumer lag 迅速堆积。",
        },
        "scenario_hint": "[历史工单反演/stub] source_ticket=hist-2025-07-shipping-consumer-scale-down：从手写历史工单摘要样例反向抽取的告警",
    }
    touched = fi.inject_alert_fault(alert, config_path)
    assert touched == ["kafkaQueueProblems"]


def test_inject_alert_fault_historical_cpu_on_frontend_is_noop(tmp_path):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"adHighCpu": {"defaultVariant": "off", "variants": {"on": True, "off": False}}})
    alert = {
        "service": "frontend",
        "annotations": {
            "summary": "历史工单反演：hist-2025-10-frontend-gc-config-change",
            "description": "frontend 的一次运行时参数配置变更间接改变了内部缓存清理频率，变更后 CPU 使用率持续处于高位，页面响应 p99 明显上升。",
        },
        "scenario_hint": "[历史工单反演/stub] source_ticket=hist-2025-10-frontend-gc-config-change：从手写历史工单摘要样例反向抽取的告警",
    }
    assert fi.inject_alert_fault(alert, config_path) == []


def test_reset_all_resets_flags_and_stops_leak_fixture(tmp_path, monkeypatch):
    config_path = tmp_path / "demo.flagd.json"
    _write_config(config_path, {"cartFailure": {"defaultVariant": "on", "variants": {"on": True, "off": False}}})
    calls = []
    monkeypatch.setattr(fi, "stop_recommendation_leak_fixture", lambda: calls.append("stop"))
    fi.reset_all(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["flags"]["cartFailure"]["defaultVariant"] == "off"
    assert calls == ["stop"]
