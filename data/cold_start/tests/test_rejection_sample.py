from __future__ import annotations

import collect_trajectories as ct
import rejection_sample as rs

REPO_GROUND_TRUTH_DIR = rs.HERE / "ground_truth"


def test_pass_and_fail_separation_on_hand_designed_mocks(tmp_path):
    """核心验收：5 条手写 mock 轨迹里，3 条该过、2 条该拒，且拒绝原因命中我们设计的那个坑。"""
    ct.mock_mode(tmp_path)  # 现写现测，不依赖磁盘上残留的产物
    results = rs.run(tmp_path, REPO_GROUND_TRUTH_DIR)

    assert results.keys() == {
        "mock_dep_clean",
        "mock_hookdeny_res",
        "mock_hybrid_leak",
        "mock_fail_evidence",
        "mock_fail_mismatch",
    }

    for alert_id in ("mock_dep_clean", "mock_hookdeny_res", "mock_hybrid_leak"):
        assert results[alert_id]["keep"] is True, results[alert_id]["reasons"]

    assert results["mock_fail_evidence"]["keep"] is False
    assert any("evidence" in r for r in results["mock_fail_evidence"]["reasons"])

    assert results["mock_fail_mismatch"]["keep"] is False
    reasons_text = " ".join(results["mock_fail_mismatch"]["reasons"])
    assert "remediation_type" in reasons_text or "suspect_service" in reasons_text or "route" in reasons_text


def test_missing_ground_truth_is_rejected(tmp_path):
    ct.mock_mode(tmp_path)
    empty_gt_dir = tmp_path / "no_such_gt_dir"
    results = rs.run(tmp_path, empty_gt_dir)
    for alert_id, verdict in results.items():
        assert verdict["keep"] is False
        assert "ground truth" in verdict["reasons"][0] or "ground_truth" in verdict["reasons"][0]


def test_evidence_traceability_rule_directly():
    steps = [
        {"tool_name": "Bash", "tool_input": {"command": "docker stats --no-stream ad"}, "observation": "ad CPU% 99.8"},
    ]
    assert rs.is_evidence_traceable("docker stats 显示 ad CPU% 99.8 打满", steps) is True
    assert rs.is_evidence_traceable("kubectl top 显示 payment 内存 900MiB 打满", steps) is False


def test_executed_actions_ledger_check_catches_fabricated_action():
    steps = [
        {
            "tool_name": "Bash",
            "tool_input": {"command": "docker restart ad"},
            "hook_decision": "allow",
            "hook_action": "restart_instance",
            "hook_target": "ad",
        }
    ]
    real_ledger_actions = [{"action": "restart_instance", "target": "ad", "command": "docker restart ad"}]
    fabricated_actions = real_ledger_actions + [
        {"action": "scale_resources", "target": "ad", "command": "docker update --cpus 2 ad"}
    ]

    assert rs.executed_actions_match_ledger(steps, real_ledger_actions) is True
    assert rs.executed_actions_match_ledger(steps, fabricated_actions) is False
    assert rs.executed_actions_match_ledger(steps, []) is False


def _hybrid_online_op_gt(expect_also_code_fix: bool = True) -> dict:
    return {
        "remediation_type": "online_op",
        "suspect_service": "recommendation",
        "expect_route": "auto_remediated_and_code_fix_pr",
        "expect_also_code_fix": expect_also_code_fix,
    }


def _hybrid_online_op_traj(also_code_fix: bool) -> dict:
    """`remediation_type=='online_op' and also_code_fix=True` 的真实模式轨迹：route 因为
    `_infer_route_real()` 没跑代码修复 Agent 而如实是 None（不是 mock 的确定值）。"""
    steps = [
        {
            "tool_name": "Bash",
            "tool_input": {"command": "docker restart recommendation"},
            "observation": "docker stats: recommendation 内存从 900MiB 降到 100MiB",
            "hook_decision": "allow",
            "hook_action": "restart_instance",
            "hook_target": "recommendation",
        }
    ]
    diag = {
        "remediation_type": "online_op",
        "suspect_service": "recommendation",
        "also_code_fix": also_code_fix,
        "evidence": ["docker stats 显示 recommendation 内存从 900MiB 降到 100MiB"],
        "executed_actions": [
            {"action": "restart_instance", "target": "recommendation", "command": "docker restart recommendation"}
        ],
    }
    return {"final_diagnosis": diag, "steps": steps, "route": None}


def test_route_none_with_correct_also_code_fix_is_kept():
    """route 依赖未跑的代码修复 Agent、如实留空为 None；only also_code_fix 判对时才该 keep=True。"""
    traj = _hybrid_online_op_traj(also_code_fix=True)
    gt = _hybrid_online_op_gt(expect_also_code_fix=True)
    verdict = rs.check_trajectory(traj, gt)
    assert verdict["keep"] is True, verdict["reasons"]


def test_route_none_with_wrong_also_code_fix_is_rejected():
    """同一场景，若诊断把 also_code_fix 判错（该转代码修复却判 False），必须被拒绝。"""
    traj = _hybrid_online_op_traj(also_code_fix=False)
    gt = _hybrid_online_op_gt(expect_also_code_fix=True)
    verdict = rs.check_trajectory(traj, gt)
    assert verdict["keep"] is False
    assert any("also_code_fix" in r for r in verdict["reasons"])


def test_pure_code_fix_route_none_is_kept_when_diagnosis_is_otherwise_correct():
    """纯 remediation_type=='code_fix' 场景：route 同样如实是 None，
    不该再被 route 精确匹配卡死——只要其余 4 项都对就该 keep=True。"""
    steps = [
        {
            "tool_name": "Bash",
            "tool_input": {"command": "git show abc123 -- cart_batch.py"},
            "observation": "cart_batch.py 增加超时保护 timeout=5s，修复 commit abc123",
        }
    ]
    diag = {
        "remediation_type": "code_fix",
        "suspect_service": "cart",
        "evidence": ["cart_batch.py 缺少超时保护，修复 commit abc123 增加 timeout=5s"],
        "executed_actions": [],
    }
    traj = {"final_diagnosis": diag, "steps": steps, "route": None}
    gt = {
        "remediation_type": "code_fix",
        "suspect_service": "cart",
        "expect_route": "feishu_fix_unverified",
    }
    verdict = rs.check_trajectory(traj, gt)
    assert verdict["keep"] is True, verdict["reasons"]


# ---------------------------------------------------------------------------
# kind 校验（2026-09-08 数据治理新增，见 docs/数据增强方案.md §4.5/§7.2/§7.3）
# ---------------------------------------------------------------------------


def _kind_check_traj(kind: str) -> dict:
    return {
        "final_diagnosis": {
            "kind": kind,
            "remediation_type": "online_op",
            "suspect_service": "kafka",
            "evidence": ["kafka_consumer_records_lag 达 15000"],
            "executed_actions": [],
        },
        "steps": [
            {
                "tool_name": "Bash",
                "tool_input": {"command": "kafka-consumer-groups.sh --describe --group g"},
                "observation": "kafka_consumer_records_lag 15000 当前积压",
            }
        ],
        "route": "feishu_online_op",
    }


def _kafka_gt(expect_kind_any_of: list[str]) -> dict:
    return {
        "remediation_type": "online_op",
        "suspect_service": "kafka",
        "expect_route": "feishu_online_op",
        "expect_kind_any_of": expect_kind_any_of,
    }


def test_kind_in_any_of_set_is_kept():
    """kafka 积压轨迹 kind=resource，GT expect_kind_any_of=[resource] -> keep。"""
    verdict = rs.check_trajectory(_kind_check_traj("resource"), _kafka_gt(["resource"]))
    assert verdict["keep"] is True, verdict["reasons"]


def test_kind_not_in_any_of_set_is_rejected():
    """同一轨迹被教师标成 config（v8 实际发生的噪声标签），必须被拒绝。"""
    verdict = rs.check_trajectory(_kind_check_traj("config"), _kafka_gt(["resource"]))
    assert verdict["keep"] is False
    assert any("kind" in r for r in verdict["reasons"])


def test_kind_any_of_accepts_every_member():
    """eval 口径多值集合（如 s1 的 [dependency, config]）里每个成员都该过。"""
    for kind in ("dependency", "config"):
        verdict = rs.check_trajectory(_kind_check_traj(kind), _kafka_gt(["dependency", "config"]))
        assert verdict["keep"] is True, (kind, verdict["reasons"])


def test_kind_check_skipped_when_gt_has_no_kind_field():
    """GT 还没回填 expect_kind_any_of 时（旧标注），kind 校验整体跳过、不误伤。"""
    gt = _kafka_gt(["resource"])
    del gt["expect_kind_any_of"]
    verdict = rs.check_trajectory(_kind_check_traj("deploy_regression"), gt)
    assert verdict["keep"] is True, verdict["reasons"]


def test_expect_kind_single_value_is_accepted_as_one_element_set():
    """兼容 `expect_kind` 单值写法（等价于长度 1 的 any_of 集合）。"""
    gt = _kafka_gt(["resource"])
    del gt["expect_kind_any_of"]
    gt["expect_kind"] = "resource"
    ok = rs.check_trajectory(_kind_check_traj("resource"), gt)
    bad = rs.check_trajectory(_kind_check_traj("resource"), {**gt, "expect_kind": "config"})
    assert ok["keep"] is True, ok["reasons"]
    assert bad["keep"] is False



# ===========================================================================
# v9 扩展校验（docs/数据增强方案.md §7.2 + §4 各家族验收规则）——2026-09 Phase C
# ===========================================================================


def _v9_base_diag(**overrides) -> dict:
    """一条能通过既有 6 项校验的基础诊断（kafka lag 场景），供扩展校验测试覆写。"""
    diag = {
        "summary": "kafkaQueueProblems 开关注入消费延迟，消费积压持续增长。",
        "kind": "resource",
        "suspect_service": "kafka",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "建议人工复位开关；已发飞书卡片交人工处理。",
        "confidence": 0.7,
        "evidence": [
            "OFREP 查询 kafkaQueueProblems 返回 variant=on",
            "docker logs fraud-detection 显示消费者 sleeping 1 second",
        ],
        "also_code_fix": False,
        "executed_actions": [],
    }
    diag.update(overrides)
    return diag


def _v9_base_traj(diag: dict, steps: list | None = None) -> dict:
    if steps is None:
        steps = [
            {"tool_name": "Bash", "tool_input": {"command": "curl -s -X POST http://localhost:8016/ofrep/v1/evaluate/flags/kafkaQueueProblems"},
             "hook_decision": "passthrough", "hook_action": None, "hook_target": None,
             "observation": '{"value":100,"key":"kafkaQueueProblems","variant":"on"}'},
            {"tool_name": "Bash", "tool_input": {"command": "docker logs fraud-detection --tail 50"},
             "hook_decision": "passthrough", "hook_action": None, "hook_target": None,
             "observation": "sleeping 1 second before replying"},
        ]
    return {
        "alert": {}, "alert_id": "t", "steps": steps,
        "final_diagnosis": diag, "ground_truth": None,
        "route": "feishu_online_op", "post_action_health": None, "meta": {},
    }


def _v9_kafka_gt(**extra) -> dict:
    gt = {
        "remediation_type": "online_op",
        "suspect_service": "kafka",
        "expect_route": "feishu_online_op",
        "expect_kind_any_of": ["resource"],
    }
    gt.update(extra)
    return gt


def test_confidence_range_pass_and_fail():
    gt = _v9_kafka_gt(expect_confidence_range=[0.6, 0.8])
    assert rs.check_trajectory(_v9_base_traj(_v9_base_diag()), gt)["keep"] is True
    bad = rs.check_trajectory(_v9_base_traj(_v9_base_diag(confidence=0.9)), gt)
    assert bad["keep"] is False and any("confidence" in r for r in bad["reasons"])
    # B1b 低置信卡口：0.59 过、0.6 拒
    gt_low = _v9_kafka_gt(expect_confidence_range=[0.0, 0.59])
    assert rs.check_trajectory(_v9_base_traj(_v9_base_diag(confidence=0.59)), gt_low)["keep"] is True
    assert rs.check_trajectory(_v9_base_traj(_v9_base_diag(confidence=0.6)), gt_low)["keep"] is False


def test_max_steps_counts_non_malformed_steps():
    traj = _v9_base_traj(_v9_base_diag())
    gt = _v9_kafka_gt(expect_max_steps=2)
    assert rs.check_trajectory(traj, gt)["keep"] is True
    gt8 = _v9_kafka_gt(expect_max_steps=8)
    assert rs.check_trajectory(traj, gt8)["keep"] is True
    gt1 = _v9_kafka_gt(expect_max_steps=1)
    assert rs.check_trajectory(traj, gt1)["keep"] is False
    # malformed 步不计入步数
    traj_m = _v9_base_traj(_v9_base_diag(), steps=traj["steps"] + [
        {"tool_name": "StructuredOutput", "tool_input": {"__unparsedToolInput": "烂 JSON"},
         "hook_decision": "passthrough", "hook_action": None, "hook_target": None, "observation": ""},
    ])
    assert rs.check_trajectory(traj_m, _v9_kafka_gt(expect_max_steps=2))["keep"] is True


def test_suspect_file_any_of():
    gt = {
        "remediation_type": "code_fix", "suspect_service": "recommendation",
        "expect_route": "code_fix_pr", "expect_kind_any_of": ["deploy_regression"],
        "expect_suspect_file_any_of": ["pagination.py", "src/pagination.py"],
    }
    diag = _v9_base_diag(
        kind="deploy_regression", suspect_service="recommendation",
        remediation_type="code_fix", suspect_file_hint="pagination.py",
        remediation_detail="分页 off-by-one 需改代码修复。",
    )
    traj = _v9_base_traj(diag)
    traj["route"] = None  # code_fix 依赖路由的真实模式留空
    assert rs.check_trajectory(traj, gt)["keep"] is True, rs.check_trajectory(traj, gt)["reasons"]
    wrong = _v9_base_traj(_v9_base_diag(**{**diag, "suspect_file_hint": "ranking.py"}))
    wrong["route"] = None
    verdict = rs.check_trajectory(wrong, gt)
    assert verdict["keep"] is False and any("suspect_file_hint" in r for r in verdict["reasons"])


def test_negative_claim_markers_and_object():
    gt = _v9_kafka_gt(
        expect_negative_claim={"object": "paymentFailure",
                               "markers": ["未观测", "off", "不成立"]}
    )
    ok_diag = _v9_base_diag(
        summary="kafka 积压为真；paymentFailure 开关当前为 off，未观测到支付失败。",
    )
    assert rs.check_trajectory(_v9_base_traj(ok_diag), gt)["keep"] is True
    # 有否定措辞但没提对象
    no_obj = _v9_base_traj(_v9_base_diag(summary="kafka 积压为真；另一个症状不成立。"))
    v1 = rs.check_trajectory(no_obj, gt)
    assert v1["keep"] is False and any("否定性" in r for r in v1["reasons"])
    # 完全没有否定措辞
    v2 = rs.check_trajectory(_v9_base_traj(_v9_base_diag(summary="kafka 积压为真。")), gt)
    assert v2["keep"] is False


def test_evidence_groups_require_both_sources():
    groups = [["ofrep", "8016", "kafkaQueueProblems"], ["docker logs", "jaeger", "日志", "fraud-detection"]]
    gt = _v9_kafka_gt(expect_evidence_groups=groups)
    # 两条 evidence 各覆盖一组 -> 过
    assert rs.check_trajectory(_v9_base_traj(_v9_base_diag()), gt)["keep"] is True
    # 只剩 OFREP 证据（缺第二信源旁证）-> 拒
    only_flag = _v9_base_diag(evidence=["OFREP 查询 kafkaQueueProblems 返回 variant=on"])
    v = rs.check_trajectory(_v9_base_traj(only_flag), gt)
    assert v["keep"] is False and any("信源" in r for r in v["reasons"])


def test_detail_contains_any():
    gt = _v9_kafka_gt(expect_detail_contains_any=["人工", "飞书", "手动"])
    assert rs.check_trajectory(_v9_base_traj(_v9_base_diag()), gt)["keep"] is True
    bad = _v9_base_traj(_v9_base_diag(remediation_detail="建议通过 flagd 控制台复位开关。"))
    v = rs.check_trajectory(bad, gt)
    assert v["keep"] is False and any("remediation_detail" in r for r in v["reasons"])


def test_remediation_any_of_and_route_any_of():
    # B1b：remediation_type 在低置信降级下不确定，列表放行
    gt = _v9_kafka_gt(
        expect_remediation_any_of=["online_op", "info_only"],
        expect_route="feishu_low_confidence",
        expect_confidence_range=[0.0, 0.59],
    )
    traj_lo = _v9_base_traj(_v9_base_diag(confidence=0.5, remediation_type="info_only"))
    traj_lo["route"] = "feishu_low_confidence"
    assert rs.check_trajectory(traj_lo, gt)["keep"] is True, rs.check_trajectory(traj_lo, gt)["reasons"]
    traj_out = _v9_base_traj(_v9_base_diag(confidence=0.5, remediation_type="code_fix"))
    traj_out["route"] = "feishu_low_confidence"
    assert rs.check_trajectory(traj_out, gt)["keep"] is False
    # route_any_of：任一命中即过
    gt2 = _v9_kafka_gt()
    del gt2["expect_route"]
    gt2["expect_route_any_of"] = ["feishu_online_op", "auto_remediated"]
    assert rs.check_trajectory(_v9_base_traj(_v9_base_diag()), gt2)["keep"] is True
    gt2["expect_route_any_of"] = ["auto_remediated"]
    assert rs.check_trajectory(_v9_base_traj(_v9_base_diag()), gt2)["keep"] is False


def test_v9_checks_skipped_when_gt_fields_absent():
    """所有扩展字段都不在 GT 里时，行为与旧版 6 项校验完全一致（向后兼容）。"""
    verdict = rs.check_trajectory(_v9_base_traj(_v9_base_diag()), _v9_kafka_gt())
    assert verdict["keep"] is True and verdict["reasons"] == []
