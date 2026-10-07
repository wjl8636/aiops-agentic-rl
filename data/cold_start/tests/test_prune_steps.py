from __future__ import annotations

import prune_steps as ps


def _bash(cmd: str, obs: str = "ok", **extra) -> dict:
    step = {"tool_name": "Bash", "tool_input": {"command": cmd}, "observation": obs}
    step.update(extra)
    return step


def _diag(evidence: list[str] | None = None, executed: list[dict] | None = None) -> dict:
    return {
        "summary": "test",
        "kind": "resource",
        "suspect_service": "kafka",
        "remediation_type": "online_op",
        "remediation_detail": "test",
        "confidence": 0.7,
        "evidence": evidence if evidence is not None else ["docker stats 显示 kafka CPU 99%"],
        "executed_actions": executed or [],
    }


def test_malformed_structured_output_is_pruned_but_success_kept():
    steps = [
        _bash("docker stats kafka", "kafka CPU 99%"),
        {
            "tool_name": "StructuredOutput",
            "tool_input": {"__unparsedToolInput": "{summary: 坏 JSON ..."},
            "observation": "Error parsing tool input",
        },
        {
            "tool_name": "StructuredOutput",
            "tool_input": {"summary": "好", "kind": "resource"},
            "observation": "Structured output provided successfully",
        },
    ]
    traj = {"steps": steps, "final_diagnosis": _diag(), "route": "feishu_online_op", "meta": {}}
    out = ps.prune_trajectory(traj)
    assert len(out["steps"]) == 2
    assert out["steps"][-1]["observation"] == "Structured output provided successfully"
    assert out["meta"]["pruning"]["removed"][0]["reason"] == "malformed-structured-output"
    # 原 dict 不被就地修改
    assert len(traj["steps"]) == 3


def test_taskstop_and_find_root_are_noise():
    steps = [
        _bash("docker ps", "kafka up"),
        {"tool_name": "TaskStop", "tool_input": {}, "observation": ""},
        _bash('find / -maxdepth 4 -iname "deploys.log" 2>/dev/null', "/a/b/deploys.log"),
        _bash("cat deploys.log | tail -5", "deploy history here"),
    ]
    traj = {"steps": steps, "final_diagnosis": _diag(), "route": None, "meta": {}}
    out = ps.prune_trajectory(traj)
    tools = [s["tool_name"] for s in out["steps"]]
    assert "TaskStop" not in tools
    cmds = [s["tool_input"].get("command", "") for s in out["steps"]]
    assert not any(c.startswith("find /") for c in cmds)


def test_marked_repeat_is_pruned():
    steps = [
        _bash("curl localhost:8016/ofrep/v1/evaluate/flags/adFailure", '{"value":true}'),
        _bash(
            "curl localhost:8016/ofrep/v1/evaluate/flags/adFailure",
            '{"value":true}',
            is_repeat_no_new_info=True,
        ),
    ]
    traj = {
        "steps": steps,
        "final_diagnosis": _diag(evidence=["ofrep 查询 adFailure 返回 value=true"]),
        "route": None,
        "meta": {},
    }
    out = ps.prune_trajectory(traj)
    assert len(out["steps"]) == 1


def test_skeleton_repeat_param_variant_with_empty_obs_is_pruned():
    """`docker logs X --tail 100` 之后 `docker logs X --since 30m` 拿到空输出：R1 的
    exact-key 口径抓不住（命令串不同），R4 的骨架归组要抓住。"""
    steps = [
        _bash("docker logs recommendation --tail 100 2>&1 | tail -100", "WARN some errors"),
        _bash("docker logs recommendation --since 30m -t 2>&1 | tail -80", "(Bash completed with no output)"),
        _bash("docker logs recommendation --since 2h -t 2>&1 | tail -60", "(Bash completed with no output)"),
    ]
    traj = {
        "steps": steps,
        "final_diagnosis": _diag(evidence=["docker logs recommendation 出现 WARN some errors"]),
        "route": None,
        "meta": {},
    }
    out = ps.prune_trajectory(traj)
    assert len(out["steps"]) == 1
    reasons = [r["reason"] for r in out["meta"]["pruning"]["removed"]]
    assert all(r.startswith("skeleton-repeat:docker-logs:recommendation") for r in reasons)


def test_failed_query_then_successful_retry_prunes_the_failed_one():
    """R5 失败查询重试：同骨架先失败（空观测）后成功——删失败那次，保留成功的那次。"""
    steps = [
        _bash("curl -s localhost:16686/jaeger/ui/api/traces?service=ad", "(Bash completed with no output)"),
        _bash("curl -s localhost:16686/jaeger/ui/api/traces?service=ad", "traceID=abc123 span=GetAds ERROR"),
    ]
    traj = {"steps": steps, "final_diagnosis": _diag(["traceID=abc123 的 GetAds span 报错"]), "route": None, "meta": {}}
    out = ps.prune_trajectory(traj)
    assert len(out["steps"]) == 1
    assert "abc123" in out["steps"][0]["observation"]
    assert out["meta"]["pruning"]["removed"][0]["reason"].startswith("failed-query-retry:")


def test_failed_query_that_is_the_only_one_of_its_kind_is_kept():
    """R5 边界：某骨架只查了一次且拿到空结果（没有成功兄弟步）——这是「查了→没有」的
    事实本身（可能被 evidence 引用），保留。"""
    steps = [
        _bash("curl -s localhost:16686/jaeger/ui/api/traces?service=ad", "(Bash completed with no output)"),
        _bash("docker stats ad", "ad CPU 1.2%"),
    ]
    traj = {"steps": steps, "final_diagnosis": _diag(["docker stats 显示 ad CPU 1.2%"]), "route": None, "meta": {}}
    out = ps.prune_trajectory(traj)
    assert len(out["steps"]) == 2


def test_skeleton_repeat_with_genuinely_new_observation_tokens_is_kept():
    """同骨架、且这次观测带来了真正的新事实 token（不是时间戳/哈希这类重跑噪声）：
    不该被剪——docker stats 在 docker ps 之后拿到具体数值是新信息。"""
    steps = [
        _bash("docker stats ad --no-stream", "ad CPU 99.8% Mem 400MiB"),
        _bash("docker stats ad", "ad CPU 1.2% Mem 88MiB 新窗口采样"),
    ]
    traj = {"steps": steps, "final_diagnosis": _diag(["docker stats ad CPU 99.8%"]), "route": None, "meta": {}}
    out = ps.prune_trajectory(traj)
    assert len(out["steps"]) == 2


def test_different_promql_queries_are_not_merged():
    """不同 PromQL（不同 service/标签）是不同的查询骨架，不能互相剪掉。
    （v8 实际数据里踩过的坑：query 截断到引号会把所有查询并成同一个骨架。）"""
    steps = [
        _bash(
            "curl -s --data-urlencode 'query=traces_span_metrics_calls_total{service_name=\"payment\"}' localhost:9090/api/v1/query",
            "payment result",
        ),
        _bash(
            "curl -s --data-urlencode 'query=traces_span_metrics_calls_total{service_name=\"checkout\"}' localhost:9090/api/v1/query",
            "checkout result",
        ),
    ]
    traj = {"steps": steps, "final_diagnosis": _diag(), "route": None, "meta": {}}
    out = ps.prune_trajectory(traj)
    assert len(out["steps"]) == 2


def test_allow_step_is_never_pruned():
    """hook 放行过的处置动作（台账步骤）即使命中重复规则也不能剪，否则台账校验会炸。"""
    allow_step = _bash(
        "docker restart kafka",
        "kafka restarted",
        hook_decision="allow",
        hook_action="restart_instance",
        hook_target="kafka",
        is_repeat_no_new_info=True,  # 即使被标了重复也不能剪
    )
    steps = [_bash("docker ps", "kafka up"), allow_step]
    diag = _diag(
        executed=[{"action": "restart_instance", "target": "kafka", "command": "docker restart kafka"}]
    )
    traj = {"steps": steps, "final_diagnosis": diag, "route": "auto_remediated", "meta": {}}
    out = ps.prune_trajectory(traj)
    assert any(s.get("hook_decision") == "allow" for s in out["steps"])


def test_evidence_cited_step_is_restored_by_verification_net():
    """安全网：候选步骤的观测是某条 evidence 的唯一出处时，修剪必须被回退。"""
    steps = [
        # 第 1 步与 evidence 的事实 token 重合 < 2（观测里没有 uniquevalue77），追不上
        _bash("docker stats kafka", "container up 2 weeks"),
        _bash("docker stats kafka --no-stream", "kafka uniquevalue77 CPU 99.8%", is_repeat_no_new_info=True),
    ]
    traj = {
        "steps": steps,
        "final_diagnosis": _diag(evidence=["uniquevalue77 显示 kafka CPU 99.8%"]),
        "route": None,
        "meta": {},
    }
    out = ps.prune_trajectory(traj)
    # 第 2 步虽是标记重复，但它是 evidence 的唯一出处 -> 被安全网回退
    assert len(out["steps"]) == 2
    assert out["meta"]["pruning"]["restored_by_verification"]


def test_pruned_trajectory_still_passes_verification():
    """修剪结果自检：evidence 可追溯 + 台账一致（对多条真实 v8 轨迹的冒烟由 CLI 承担）。"""
    steps = [
        _bash("docker stats kafka", "kafka CPU 99%"),
        _bash("docker stats kafka", "kafka CPU 99%", is_repeat_no_new_info=True),
        {"tool_name": "StructuredOutput", "tool_input": {"summary": "ok"}, "observation": "ok"},
    ]
    traj = {"steps": steps, "final_diagnosis": _diag(), "route": None, "meta": {}}
    out = ps.prune_trajectory(traj)
    assert ps.verify_pruned(out, out["steps"]) == []
