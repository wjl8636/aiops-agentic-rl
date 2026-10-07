"""收集冷启轨迹（design doc §4.3）。

两种模式：
- 真实模式（默认）：对 `--alerts-dir` 下每条告警，真实跑一遍 AIops-agent 的故障诊断处置 Agent
  （真实 Opus API 调用 + 真实 hook/白名单），把完整的「每一步工具调用+观测+hook 判定」记录成
  一条轨迹，写到 `data/cold_start/trajectories/<alert_id>.json`。
  **本次会话不会真的跑这条路径**（需要真实 API key + kind/docker 环境），但代码本身是可运行的、
  可 import 的；用 `--check-env` 可以只做「AIops-agent 环境是否可 import」的检查，不发起任何调用。
- `--mock` 模式：生成 3~5 条完全合成但 schema 合法的轨迹，写到
  `data/cold_start/trajectories/mock/*.json`，本次会话里真的跑一遍并校验产物。

## 真实模式怎么拿到「每一步」轨迹（关键设计选择，任务里要求写清楚）
`AIops-agent/agent/agents/diagnose.py::diagnose()` 只是薄薄调用了
`AIops-agent/agent/core/sdk_runner.py::run_sdk()`，而 `run_sdk()` 只在消息流里挑
`ResultMessage`（最终结果），中间的 `AssistantMessage`（模型的每次工具调用）和
`UserMessage`（工具执行返回的 `ToolResultBlock`）都被丢弃了——所以拿不到 step 级轨迹。

本文件选择方案 (a)：不走 `diagnose()`/`run_sdk()`，而是在 `_run_diagnose_capturing_steps()`
里直接调用 `claude_agent_sdk.query()`，复刻 `sdk_runner.run_sdk()` 里 `ClaudeAgentOptions` 的
构造方式（同样的 allowed_tools / hook / json_schema / skills 参数），但保留每一条中间消息：
- `AssistantMessage.content` 里的 `ToolUseBlock(id, name, input)` = 模型发起的一次工具调用；
- 随后的 `UserMessage.content` 里同 `tool_use_id` 的 `ToolResultBlock(content, is_error)`
  = 这次调用的观测结果（工具执行由 SDK/CLI 在本地完成，Bash/Read/Grep/Glob 都是内置工具）。
把两者按 `tool_use_id` 配对，就还原出「调用了什么 -> 观测到了什么」的完整 step 序列。
真正的 PreToolUse hook 判定（allow/deny，命中白名单的 action/target）不会随消息流回传，
所以额外用 `_make_recording_hook()` 包一层：包装函数在调用真实的 `guard_diagnose_ops` 之前，
自己也调一次 `agent.core.remediation.classify_command()`（判定逻辑与 hook 内部完全一致，
只是提前把结构化结果记下来），再原样转发调用/返回值给真正的 hook——不改变任何放行/拦截行为，
只是"偷看"一眼判定结果。

这套消息类型（AssistantMessage / UserMessage / ToolUseBlock / ToolResultBlock / ResultMessage）
是根据 claude-agent-sdk-python 的公开源码（`src/claude_agent_sdk/types.py`）确认的，但本机沙箱
没有安装 `claude_agent_sdk`（也不应该在这次会话里装/跑，会触发真实 API 调用），所以这条真实
路径本身**没有在本次会话里被执行验证过**——上线前务必先在装好 SDK + 有 API key 的机器上跑
`--check-env` 确认能 import，再拿一条真告警跑通再批量跑。

## 与 `reward/trajectory.py`（并行任务）的字段对齐
输出的每条轨迹是一个 dict，字段名故意跟任务说明里给的 `Trajectory`/`Step` 大致形状对齐：
`alert, steps:[{tool_name, tool_input, hook_decision, hook_action, hook_target, observation,
signal_types_covered, is_repeat_no_new_info}], final_diagnosis, ground_truth, route,
post_action_health`。这里不 `import reward.*`（怕并行任务此刻还没建好那个模块），只是保证
字段名字/形状一致，方便以后 reward 模块直接消费这份 JSON。`ground_truth` 在这一步永远是
`None`——它是人工标注产物，由 `rejection_sample.py` 从 `data/cold_start/ground_truth/<id>.json`
读取后再做比对，不在采集阶段就编进轨迹里。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import signals  # noqa: E402  (本目录内的共享信号分类工具)
import fault_injection  # noqa: E402  (本目录内：跑 Agent 前注入 scenario_hint 里的 flagd 故障)

REPO_ROOT = HERE.parent.parent  # .../aiops-agentic-rl
AIOPS_AGENT_DIR = REPO_ROOT / "AIops-agent"
DEFAULT_ALERTS_DIR = AIOPS_AGENT_DIR / "alerts"
DEFAULT_TRAJ_DIR = HERE / "trajectories"
DEFAULT_MOCK_DIR = DEFAULT_TRAJ_DIR / "mock"


def _ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def _load_aiops_core():
    """导入 AIops-agent 里「不需要 claude_agent_sdk」的那部分（config/schema/remediation/hooks）。

    这几个模块本身没有 import claude_agent_sdk（只有 sdk_runner.py / diagnose.py 才需要），
    所以即使这台机器没装 SDK，也能拿到真实的 Diagnosis pydantic 模型 + 真实的白名单判定逻辑
    ——mock 模式用它们来生成/校验「schema 真的合法」「hook 判定真的是白名单代码给出的」，
    而不是自己再写一份近似的规则。
    """
    _ensure_aiops_agent_on_path()
    from agent import config  # type: ignore
    from agent.core import remediation  # type: ignore
    from agent.core import schema  # type: ignore
    from agent.core.hooks import guard_diagnose_ops  # type: ignore

    return config, remediation, schema, guard_diagnose_ops


def check_env() -> dict[str, Any]:
    """只读检查：AIops-agent 环境能不能被 import。不发起任何真实调用，也不需要 API key。"""
    _ensure_aiops_agent_on_path()
    result: dict[str, Any] = {
        "aiops_agent_dir": str(AIOPS_AGENT_DIR),
        "aiops_agent_dir_exists": AIOPS_AGENT_DIR.is_dir(),
        "core_importable": False,  # config/schema/remediation/hooks（真实模式+mock模式都要用）
        "claude_agent_sdk_importable": False,  # 只有真实模式的 query() 才需要
        "sdk_runner_importable": False,  # agent.core.sdk_runner / agent.agents.diagnose
        "errors": [],
    }
    try:
        _load_aiops_core()
        result["core_importable"] = True
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"core import failed: {exc!r}")

    try:
        import claude_agent_sdk  # noqa: F401

        result["claude_agent_sdk_importable"] = True
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"claude_agent_sdk import failed: {exc!r}")

    try:
        from agent.core import sdk_runner  # noqa: F401

        result["sdk_runner_importable"] = True
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"sdk_runner import failed: {exc!r}")
    return result


# --------------------------------------------------------------------------
# 真实模式：真的跑一遍 AIops-agent，保留每一条中间消息。
# --------------------------------------------------------------------------

def _stringify_tool_result(content: Any) -> str:
    """把 ToolResultBlock.content（str 或 list[dict] 两种可能形态）拍平成字符串观测。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


#: v9 Phase C（2026-09）采集侧追加的系统提示词片段：kind 分类口径 + 矛盾信号交叉验证。
#: 为什么只在采集侧加、不改 AIops-agent 的 prompts.py：AIops-agent 仓库只读；且这两条是
#: 「数据生成的标注口径对齐」——docs/数据增强方案.md §7.3 的 kind 决策表（对齐 eval 的
#: expected.json）。v8 的教训（Phase A 治理淘汰 9 条）：教师对 flagd 注入的队列积压一律标
#: config（按注入手段分类），与 s3 期望的 resource（按症状机制分类）确定性分歧——不引导
#: 教师就会复刻同样的噪声标签，B1a（本轮最关键家族）会整家族被拒。观测本身仍是真实工具
#: 输出，只是分类口径对齐目标 taxonomy；SFT 训练正是要让底座模型把这套口径内化进权重。
_KIND_TAXONOMY_GUIDANCE = """

## kind 字段分类口径（按故障的症状机制归类，不是按注入/触发手段归类）
- 队列/消息积压、消费延迟、CPU/内存/容量耗尽类症状 → `resource`（即使该故障是功能开关注入的，积压本身就是资源/容量类症状）
- 服务间依赖调用失败/超时/不可达 → `dependency`
- 发版引入的代码回归：内存泄漏、无界数据结构、业务逻辑 bug（排序/去重/计数/分页/计价等）→ `deploy_regression`
- 纯配置项/功能开关取值错误（开关只改变行为参数，不落入上述症状机制，如图片加载限速档位）→ `config`

## 矛盾信号的交叉验证纪律
若某个信号源的读数与告警或其它证据矛盾（例如 Prometheus 指标缺失、冻结为 0 或显示正常，但业务侧症状确实存在），不要单凭这一个读数否定告警——换第二信源交叉验证（日志 / 调用链 / OFREP 实时开关值），把两路证据都写进 evidence 再下结论。
"""


def _llm_env_overrides_real_teacher(config_mod) -> dict[str, str]:
    """构造塞进 claude CLI 子进程的环境变量，把教师请求钉死在 config.LLM_* 指定的后端上。

    与 `AIops-agent/agent/core/sdk_runner.py::_llm_env_overrides()` 同一机制（`ClaudeAgentOptions`
    的 env= 是子进程环境合并的最后一层，无条件覆盖 inherited_env 与 settings.json 的 env 块），
    但有一处关键差异：**保留 ANTHROPIC_CUSTOM_HEADERS**。sdk_runner 版本会把它清空（防真实公司
    代理的 SSO Cookie 泄给本地 vLLM）；v9 采集的教师恰恰是真实代理（AIOPS_LLM_BASE_URL 映射
    ANTHROPIC_BASE_URL），该代理鉴权依赖这个 Cookie header，清空会直接 401/无登录信息。

    为什么这里必须显式传 env=：本脚本 `setting_sources=["project"]` + cwd=AIops-agent 仓库根，
    会加载嵌套 checkout 的 `.claude/settings.json`——它的 env 块把 ANTHROPIC_BASE_URL 钉死在
    `http://localhost:8000`（本地 vLLM，云端机器已关机、端口已死）。不覆盖它，采集会全部连接
    失败（好在失败是响亮的，不会静默漏到别的模型）。

    config_mod：`_load_aiops_core()` 返回的 agent.config 模块（本模块不在顶层 import 它，
    跟 `_load_aiops_core()` 的懒加载约定一致）。
    """
    import os

    return {
        "ANTHROPIC_BASE_URL": config_mod.LLM_BASE_URL,
        "ANTHROPIC_API_KEY": config_mod.LLM_API_KEY,
        "ANTHROPIC_AUTH_TOKEN": config_mod.LLM_AUTH_TOKEN,
        "ANTHROPIC_MODEL": config_mod.MODEL,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": config_mod.MODEL,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": config_mod.MODEL,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": config_mod.MODEL,
        "ANTHROPIC_SMALL_FAST_MODEL": config_mod.MODEL,
        # 真实代理鉴权用的 SSO Cookie header：原样透传（见 docstring，与 sdk_runner 的差异点）。
        "ANTHROPIC_CUSTOM_HEADERS": os.environ.get("ANTHROPIC_CUSTOM_HEADERS", ""),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def _write_teacher_settings_file(env: dict[str, str]) -> str:
    """把 `_llm_env_overrides_real_teacher()` 的 env 块写进一个临时 settings JSON，返回路径。

    为什么除了 `ClaudeAgentOptions(env=...)` 还要这份文件：实测（2026-09-08，CLI 2.1.139）
    CLI 进程在应用 `setting_sources=["project"]` 加载的项目 settings.json 时，会把其 env 块
    应用到自己 process.env 之上——**文件型 settings 的 env 压过 spawn 时传入的进程 env**
    （嵌套 checkout 的 `.claude/settings.json` 把 ANTHROPIC_BASE_URL 钉死在已死的
    localhost:8000，options.env 拦不住，表现为 ConnectionRefused 长重试）。而 `--settings`
    传入的程序化 settings 优先级高于一切文件型 settings，是唯一可靠的覆盖层。两处都传：
    env= 管子进程初始环境，settings 文件负责压过项目 settings.json。
    """
    import json
    import tempfile

    d = tempfile.mkdtemp(prefix="aiops_teacher_settings_")
    p = Path(d) / "settings.json"
    p.write_text(json.dumps({"env": env}, indent=2), encoding="utf-8")
    return str(p)


def _make_recording_hook(inner_hook, remediation_mod, sink: dict[str, dict[str, Any]]):
    """包一层 hook：转发调用给真正的 guard_diagnose_ops，但把它的判定结果顺手记一份下来。

    hook_decision 直接读 `inner_hook` 真正返回的 `permissionDecision`（allow/deny/其余视为
    passthrough）——不能只看 `remediation.classify_command()`，因为 guard_diagnose_ops 内部
    在 classify_command 判 passthrough 之后，还有一层 `_READONLY_DENY`/`_WRITE_FALLBACK_DENY`
    兜底拦截（比如 `docker exec` 没命中白名单但明显是高风险写操作），只看 classify_command
    会漏掉这一层，把这类命令误判成 passthrough。action/target 只在 allow 时才有意义，
    直接复用 classify_command 的结果（allow 分支下两者判定是一致的）。
    这层包装不改变任何放行/拦截行为，只在真正决策之后把结构化结果记下来。
    """

    async def _hook(input_data, tool_use_id, context):  # noqa: ANN001
        result = await inner_hook(input_data, tool_use_id, context)
        decision_raw = (result.get("hookSpecificOutput") or {}).get("permissionDecision")
        decision = decision_raw if decision_raw in ("allow", "deny") else "passthrough"
        action = target = None
        if input_data.get("tool_name") == "Bash":
            # classify_command 在 allow 和「命中白名单但目标未授权」的 deny 两种情形下都会
            # 带出 action/target（后者是「本来想做什么，但目标不在授权清单里」），只有落到
            # _READONLY_DENY/_WRITE_FALLBACK_DENY 兜底拦截或 passthrough 时才没有意义的 action/target。
            verdict = remediation_mod.classify_command(str(input_data.get("tool_input", {}).get("command", "")))
            if verdict.get("decision") in ("allow", "deny"):
                action, target = verdict.get("action"), verdict.get("target")
        sink[tool_use_id] = {"decision": decision, "action": action, "target": target}
        return result

    return _hook


async def _run_diagnose_capturing_steps(alert: dict[str, Any]) -> dict[str, Any]:
    """真实模式核心：复刻 sdk_runner.run_sdk() 的 options 构造，但保留每条中间消息。

    没有在本次会话执行过（见模块 docstring）；claude_agent_sdk 只在函数内部 import，
    这样 --mock / --check-env 路径完全不需要装这个包。
    """
    from claude_agent_sdk import (  # 延迟 import：mock/check-env 路径不需要装这个包
        AssistantMessage,
        ClaudeAgentOptions,
        HookMatcher,
        ResultMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )

    config, remediation, schema, guard_diagnose_ops = _load_aiops_core()
    from agent.agents import prompts  # type: ignore
    from agent.integrations import rag_tool  # type: ignore

    remediation.reset_ledger()
    # scenario_hint 是数据生成阶段的分层标注（写着 flag=xxx service=xxx），真实告警不会有
    # 这个字段；喂给诊断 Agent 会泄漏答案，且真实推理时也不存在，训练出来的行为不泛化。
    alert_for_prompt = {k: v for k, v in alert.items() if k != "scenario_hint"}
    prompt = prompts.diagnose_prompt(alert_for_prompt)

    allowed = ["Bash", "Read", "Grep", "Glob"]
    mcp_servers = None
    if config.RAG_ENABLED:
        allowed.append(rag_tool.TOOL_NAME)
        mcp_servers = {"aiops_rag": rag_tool.server}

    hook_sink: dict[str, dict[str, Any]] = {}
    recording_hook = _make_recording_hook(guard_diagnose_ops, remediation, hook_sink)

    # v9（2026-09）：教师 LLM 后端双重钉死——env= 管子进程初始环境，程序化 settings 文件
    # 压过嵌套 checkout .claude/settings.json 的 env 块（ANTHROPIC_BASE_URL=localhost:8000，
    # 已死；文件型 settings 的 env 实测压过 spawn 时的进程 env，见 _write_teacher_settings_file）。
    teacher_env = _llm_env_overrides_real_teacher(config)
    teacher_settings_path = _write_teacher_settings_file(teacher_env)

    opts = ClaudeAgentOptions(
        cwd=str(config.REPO_ROOT),
        model=config.MODEL,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            # v9 Phase C：在 AIops-agent 原始诊断提示词之上追加 kind 分类口径 + 交叉验证
            # 纪律（见 _KIND_TAXONOMY_GUIDANCE 的说明——修 v8 遗留的 kind 标签噪声）。
            "append": prompts.diagnose_append() + _KIND_TAXONOMY_GUIDANCE,
        },
        allowed_tools=allowed,
        permission_mode="acceptEdits",
        disallowed_tools=["Bash(rm -rf *)", "Bash(git push --force *)"],
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[recording_hook])]},
        setting_sources=["project"],
        max_turns=config.DIAGNOSE_MAX_TURNS,
        output_format={"type": "json_schema", "schema": schema.DIAGNOSIS_JSON_SCHEMA},
        mcp_servers=mcp_servers,
        skills="all",
        env=teacher_env,
        settings=teacher_settings_path,
    )

    pending_calls: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    result_text: Optional[str] = None
    structured: Any = None
    degraded: Optional[str] = None
    usage: dict[str, int] = {}
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    model_usage: Any = None
    duration_ms: Optional[int] = None

    def _accumulate_usage(u: Optional[dict]) -> None:
        """跟 sdk_runner._accumulate_usage 同一口径：cache 感知地累加 token 用量。"""
        if not u:
            return
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            v = u.get(k)
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + int(v)

    try:
        async with asyncio.timeout(config.DIAGNOSE_TIMEOUT_S):
            async for msg in query(prompt=prompt, options=opts):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            pending_calls[block.id] = {"tool_name": block.name, "tool_input": block.input}
                elif isinstance(msg, UserMessage):
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            call = pending_calls.pop(block.tool_use_id, None)
                            if call is None:
                                continue
                            hook_info = hook_sink.get(block.tool_use_id, {})
                            steps.append(
                                {
                                    "tool_name": call["tool_name"],
                                    "tool_input": call["tool_input"],
                                    "hook_decision": hook_info.get("decision", "passthrough"),
                                    "hook_action": hook_info.get("action"),
                                    "hook_target": hook_info.get("target"),
                                    "observation": _stringify_tool_result(block.content),
                                }
                            )
                elif isinstance(msg, ResultMessage):
                    _accumulate_usage(msg.usage)
                    if msg.total_cost_usd is not None:
                        cost_usd = msg.total_cost_usd
                    num_turns = msg.num_turns
                    model_usage = msg.model_usage
                    duration_ms = msg.duration_ms
                    if msg.subtype == "success":
                        result_text = msg.result
                        structured = msg.structured_output
                    else:
                        degraded = msg.subtype
                        result_text = msg.result
                        structured = msg.structured_output
    except asyncio.TimeoutError:
        # DIAGNOSE_TIMEOUT_S 到点：已经录下的 steps 保留，标成降级而不是整条丢弃。
        degraded = "timeout"
    except Exception as exc:  # noqa: BLE001
        # claude_agent_sdk 撞到 max_turns（或子进程其它硬错误）时不会产出一条正常的
        # ResultMessage，而是直接从 query() 抛异常（ProcessError/ResultError）——这里必须接住，
        # 否则 real_mode() 遍历一批告警时，一条撞上限就会让后面所有告警都跑不到。
        # 已经录下的 steps 仍然是真实、有信用分配价值的轨迹前缀，不因为没收敛就整条丢弃
        # （reward/reward_router.py 的 max_turns_truncation_penalty 就是为这种截断轨迹设计的）。
        degraded = f"sdk_error:{type(exc).__name__}"
        result_text = str(exc)[:500]

    signals.annotate_steps(steps)

    final_diagnosis: Optional[dict[str, Any]] = None
    diag_obj = None
    if structured is not None:
        try:
            diag_obj = schema.Diagnosis.model_validate(structured)
            diag_obj.executed_actions = [schema.ExecutedAction(**e) for e in remediation.get_ledger()]
            final_diagnosis = diag_obj.model_dump(by_alias=True)
        except Exception:  # noqa: BLE001  校验失败也不崩：留一份兜底降级记录
            diag_obj = schema.Diagnosis.fallback(
                f"structured_output 校验失败，raw={str(result_text)[:200]!r}"
            )
            final_diagnosis = diag_obj.model_dump(by_alias=True)
            degraded = degraded or "schema_validation_failed"

    # route：能确定就填真值，不能确定（本脚本没跑代码修复 Agent，无法知道验证结果）就如实留 None，
    # 而不是像旧版一样对所有轨迹都硬编码 None——那样会让 rejection_sample.py 的
    # `traj["route"] == gt["expect_route"]` 判定对所有真实轨迹恒假，跟诊断质量无关地把它们全部拒绝。
    # 见 `_infer_route_real()` 的说明。
    route = _infer_route_real(diag_obj) if diag_obj is not None else None

    return {
        "alert": alert,
        "alert_id": alert.get("alertname") or alert.get("id") or "unknown",
        "steps": steps,
        "final_diagnosis": final_diagnosis,
        "ground_truth": None,  # 人工标注产物，采集阶段不填，见模块 docstring
        "route": route,
        "post_action_health": None,  # 真实模式若要填，需要在线上跑完 60s 观测窗口，另行采集
        "meta": {
            "source": "real",
            "degraded": degraded,
            "raw": result_text,
            # v9（2026-09）：教师调用的真实 API 计量（ResultMessage.usage / total_cost_usd /
            # model_usage），供「确认教师真是 Opus」与成本记账（docs/数据增强方案.md §6.2：
            # 每条轨迹 meta.cost_usd 记账、累计花费写进采集进度文件）。
            "teacher_model": config.MODEL,
            "usage": usage,
            "cost_usd": cost_usd,
            "num_turns": num_turns,
            "duration_ms": duration_ms,
            "model_usage": model_usage,
        },
    }


def real_mode(alerts_dir: Path, out_dir: Path, sample: Optional[int] = None) -> list[Path]:
    """真实模式入口：对 alerts_dir 下每条告警跑一遍，写轨迹到 out_dir。

    调用前请先跑 `--check-env` 确认 claude_agent_sdk 能 import、且真的有 API key/网络/
    kind 或 docker 环境——本函数不做这层校验，也不会在没有真实环境时降级成假数据。

    `sample` 限制只跑前 N 条（按文件名排序取前 N，保证同一目录下多次跑取的是同一批，
    方便先跑 5~10 条小批量验证再放量到 200 条）；不传则跑目录下全部告警。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    alert_files = sorted(alerts_dir.glob("*.json"))
    if sample is not None:
        alert_files = alert_files[:sample]

    flagd_config = fault_injection.discover_flagd_config()
    if flagd_config is None:
        print("[real] 没找到活的 flagd 容器/配置，本轮不注入 scenario_hint 里的 flag（跑的是环境当前的实际状态）。")

    for f in alert_files:
        alert_id = f.stem
        try:
            alert = json.loads(f.read_text(encoding="utf-8"))
            injected = fault_injection.inject_alert_fault(alert, flagd_config)
            if injected:
                print(f"[real] {alert_id} 注入故障：{injected}")
            try:
                traj = asyncio.run(_run_diagnose_capturing_steps(alert))
            finally:
                if flagd_config is not None:
                    # reset_all 同时复位 flagd 开关和（如果触发过）recommendation 内存泄漏
                    # fixture 容器，两者互相独立、都是安全的无条件调用。
                    fault_injection.reset_all(flagd_config)
        except Exception as exc:  # noqa: BLE001
            # 一条告警的意外失败（比如告警文件本身坏了）不该让批次里剩下的告警都跑不到——
            # _run_diagnose_capturing_steps 内部已经接住了 SDK 层面的 max_turns/timeout，
            # 这里是给它没接住的意外情况兜底，打日志继续下一条。
            print(f"[real] {alert_id} 失败，跳过继续下一条：{type(exc).__name__}: {exc}")
            continue
        traj["alert_id"] = alert_id
        out_path = out_dir / f"{alert_id}.json"
        out_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_path)
        print(f"[real] {alert_id} 完成（{len(traj.get('steps') or [])} 步，degraded={traj.get('meta', {}).get('degraded')}）")
    return written


# --------------------------------------------------------------------------
# mock 模式：手写但 schema 真实合法的合成轨迹。
# --------------------------------------------------------------------------

def _step(
    tool_name: str,
    tool_input: dict[str, Any],
    observation: str,
    *,
    hook_decision: str = "passthrough",
    hook_action: Optional[str] = None,
    hook_target: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "hook_decision": hook_decision,
        "hook_action": hook_action,
        "hook_target": hook_target,
        "observation": observation,
    }


def _hook_verdict_for_bash(guard_diagnose_ops, remediation_mod, command: str) -> dict[str, Any]:
    """mock 轨迹里 Bash 步骤的 hook 判定，直接跑一遍真实的 `guard_diagnose_ops` hook
    （而不是只调 remediation.classify_command）——因为 hook 里还有一层
    `_READONLY_DENY`/`_WRITE_FALLBACK_DENY` 兜底拦截（比如 `docker exec` 这种没命中白名单
    但明显是高风险写操作的命令），只看 classify_command 会把这类命令误判成 passthrough。
    这样 mock 数据里的 hook_decision/action/target 跟真实系统在同一条命令下会给出的判定完全一致。
    """
    input_data = {"tool_name": "Bash", "tool_input": {"command": command}, "hook_event_name": "PreToolUse"}
    result = asyncio.run(guard_diagnose_ops(input_data, "mock-tool-use-id", {}))
    decision_raw = (result.get("hookSpecificOutput") or {}).get("permissionDecision")
    decision = decision_raw if decision_raw in ("allow", "deny") else "passthrough"
    verdict = remediation_mod.classify_command(command)
    action = target = None
    if verdict.get("decision") in ("allow", "deny"):
        action, target = verdict.get("action"), verdict.get("target")
    return {"hook_decision": decision, "hook_action": action, "hook_target": target}


def build_mock_trajectories() -> list[dict[str, Any]]:
    """手写 5 条 schema 合法的合成轨迹，覆盖：
    1. mock_dep_clean       依赖故障，干净收敛，line 里含一次白名单命中的自动止血（allow）
    2. mock_hookdeny_res    资源型故障，先踩一次 hook deny，再改用白名单形态的命令拿到 allow
    3. mock_hybrid_leak     混合根因（内存泄漏 capstone），also_code_fix=true
    4. mock_fail_evidence   人为设计成「evidence 编造」——给 rejection_sample.py 当反例
    5. mock_fail_mismatch   人为设计成「suspect_service/remediation_type 跟人工标注不一致」——反例
    """
    _, remediation, schema, guard_diagnose_ops = _load_aiops_core()

    trajectories: list[dict[str, Any]] = []

    # --- 1. 依赖故障，干净收敛 --------------------------------------------------
    alert1 = {
        "alertname": "MockHighErrorRate",
        "service": "product-catalog",
        "labels": {"service": "product-catalog", "severity": "critical", "job": "product-catalog"},
        "annotations": {
            "summary": "product-catalog GetProduct 5xx 飙升",
            "description": "flagd productCatalogFailure 开关翻转后，product-catalog 依赖调用失败率飙升，"
            "上游 recommendation/frontend 大量报错。",
        },
        "startsAt": "2026-06-20T10:00:00Z",
    }
    restart_cmd = "docker restart product-catalog"
    steps1 = [
        _step("Bash", {"command": "docker ps"}, "product-catalog   Up 2 hours (unhealthy)\nrecommendation    Up 3 hours"),
        _step(
            "Bash",
            {"command": "curl -s http://localhost:9090/api/v1/query?query=up{job=\"product-catalog\"}"},
            'Prometheus: up{job="product-catalog"} = 0，过去 5 分钟内错误率从 1% 跳升到 62%。',
        ),
        _step(
            "Bash",
            {"command": "curl -s http://localhost:16686/api/traces?service=product-catalog&limit=5"},
            "Jaeger: 最近 5 条 trace 全部在 GetProduct span 返回 UNAVAILABLE，"
            "下游未再往前传播，故障止在 product-catalog 自身。",
        ),
        _step("Bash", {"command": "docker logs --tail 50 product-catalog"}, "ERROR flagd: feature flag "
              "'productCatalogFailure' evaluated to true, injecting synthetic 5xx"),
        # product-catalog 不在低风险自动处置白名单内（AIOPS_REMEDIATION_TARGETS 默认只放
        # recommendation/ad/frontend/cart/checkout/currency/payment/shipping/quote/email）——
        # 想自动重启会被真实 hook 拦下，只能改发飞书卡片交人工重启。观测文本如实记录「被拒绝」，
        # 不能写成执行成功（hook_decision 由下面真实调一遍 guard_diagnose_ops 得到，是 "deny"）。
        _step(
            "Bash",
            {"command": restart_cmd},
            "PreToolUse hook 拒绝执行：低风险操作 restart_instance 命中，但目标 product-catalog "
            "不在可自动处置白名单内，需改发飞书卡片交人工重启。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, restart_cmd),
        ),
    ]
    diag1 = {
        "summary": "flagd productCatalogFailure 开关注入的合成依赖故障，需重启 product-catalog 清除注入状态。",
        "kind": "dependency",
        "suspect_service": "product-catalog",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "建议重启 product-catalog 清除注入状态；product-catalog 不在自动处置白名单内，已改发飞书卡片交人工执行。",
        "confidence": 0.88,
        "evidence": [
            'Prometheus up{job="product-catalog"}=0，错误率从 1% 跳升到 62%',
            "Jaeger 最近 5 条 trace 全部在 GetProduct span 返回 UNAVAILABLE",
            "docker logs 显示 flagd productCatalogFailure 开关命中，注入合成 5xx",
        ],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_dep_clean", alert1, steps1, diag1, schema, remediation))

    # --- 2. 资源型故障，先 deny 再 allow -----------------------------------------
    alert2 = {
        "alertname": "MockAdCpuThrottled",
        "service": "ad",
        "labels": {"service": "ad", "severity": "warning", "job": "ad"},
        "annotations": {
            "summary": "ad 服务 CPU 打满，p99 延迟飙升",
            "description": "ad 服务容器 CPU throttling 持续 100%，AdRequest p99 从 80ms 涨到 1.2s。",
        },
        "startsAt": "2026-06-20T11:00:00Z",
    }
    scale_cmd_bad = "docker exec -it ad top"  # 模型先试了个会被拦的写法（进容器）
    scale_cmd_good = "docker update --cpus 2 --memory 1024m ad"
    steps2 = [
        _step("Bash", {"command": "docker stats --no-stream ad"}, "ad   CPU% 99.8   MEM 480MiB / 512MiB"),
        _step(
            "Bash",
            {"command": "curl -s http://localhost:9090/api/v1/query?query=rate(container_cpu_usage_seconds_total{name=\"ad\"}[1m])"},
            "Prometheus: ad 容器 CPU throttling 比例过去 10 分钟持续 100%。",
        ),
        _step(
            "Bash",
            {"command": scale_cmd_bad},
            "PreToolUse hook 拒绝执行：docker exec 属于进容器高风险操作，不在低风险白名单内。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, scale_cmd_bad),
        ),
        _step(
            "Bash",
            {"command": scale_cmd_good},
            "ad 容器资源上限已提升为 --cpus 2 --memory 1024m。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, scale_cmd_good),
        ),
    ]
    diag2 = {
        "summary": "ad 服务 CPU 打满导致延迟飙升，抬高资源上限后恢复。",
        "kind": "resource",
        "suspect_service": "ad",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "已临时抬高 ad 容器 CPU/内存上限缓解打满。",
        "confidence": 0.82,
        "evidence": [
            "docker stats 显示 ad 容器 CPU% 99.8",
            "Prometheus container_cpu_usage_seconds_total 显示 ad 持续 100% throttling",
        ],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_hookdeny_res", alert2, steps2, diag2, schema, remediation))

    # --- 3. 混合根因（内存泄漏 capstone）------------------------------------------
    alert3 = {
        "alertname": "MockRecommendationMemoryLeak",
        "service": "recommendation",
        "labels": {"service": "recommendation", "severity": "critical", "job": "recommendation"},
        "annotations": {
            "summary": "recommendation 内存单调上升且需先止血再改代码根治",
            "description": "recommendation 容器内存持续上涨、OOMKilled 次数飙升，怀疑最近一次发版引入了"
            "无界缓存；业务要求先扩容/重启止血，再触发代码修复 Agent 提 PR 根治。",
        },
        "startsAt": "2026-06-20T12:00:00Z",
    }
    restart3 = "docker restart recommendation"
    steps3 = [
        _step("Bash", {"command": "docker stats --no-stream recommendation"}, "recommendation   MEM 940MiB / 1024MiB（持续上涨）"),
        _step("Bash", {"command": "git -C /repo/recommendation log --oneline -5"},
              "abc1234 feat: cache popular items in-memory for latency\n"
              "d4e5f67 chore: bump grpc\n..."),
        _step("Bash", {"command": "git -C /repo/recommendation show abc1234 --stat"},
              "abc1234 修改 src/recommendation_server.py：新增全局 dict `_POPULAR_CACHE`，"
              "无过期/无容量上限，每次请求都会写入新 key。"),
        _step("Bash", {"command": "docker logs --tail 30 recommendation"}, "OOMKilled 事件 x3（过去 1 小时），"
              "MemoryError: cannot allocate memory"),
        _step(
            "Bash",
            {"command": restart3},
            "recommendation 已重启，内存回落到 120MiB。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, restart3),
        ),
    ]
    diag3 = {
        "summary": "commit abc1234 引入的无界内存缓存导致 OOM，已重启止血，需代码修复根治。",
        "kind": "deploy_regression",
        "suspect_service": "recommendation",
        "suspect_repo": "recommendation",
        "suspect_commit_hint": "abc1234",
        "suspect_file_hint": "src/recommendation_server.py",
        "remediation_type": "online_op",
        "remediation_detail": "已滚动重启 recommendation 清空内存，同时需要修复 abc1234 引入的无界缓存。",
        "confidence": 0.9,
        "evidence": [
            "docker stats 显示 recommendation 内存持续上涨至 940MiB/1024MiB",
            "git show abc1234 显示新增无过期/无容量上限的全局 dict 缓存",
            "docker logs 显示过去 1 小时 OOMKilled 事件 x3",
        ],
        "also_code_fix": True,
    }
    trajectories.append(_assemble("mock_hybrid_leak", alert3, steps3, diag3, schema, remediation))

    # --- 4. 反例：evidence 编造（rejection_sample.py 应该拒绝）----------------------
    alert4 = {
        "alertname": "MockFabricatedEvidence",
        "service": "checkout",
        "labels": {"service": "checkout", "severity": "critical", "job": "checkout"},
        "annotations": {"summary": "checkout 错误率上升", "description": "checkout 服务错误率上升。"},
        "startsAt": "2026-06-20T13:00:00Z",
    }
    steps4 = [
        _step("Bash", {"command": "docker ps"}, "checkout   Up 1 hour (healthy)"),
    ]
    diag4 = {
        "summary": "checkout 数据库连接池耗尽。",
        "kind": "resource",
        "suspect_service": "checkout",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "已重启 checkout。",
        "confidence": 0.7,
        # 编造证据：这句话在 steps4 的任何 observation 里都找不到出处（真实 observation 只说容器 healthy）
        "evidence": ["kubectl top 显示 checkout 数据库连接池占用 100%，触发连接超时"],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_fail_evidence", alert4, steps4, diag4, schema, remediation))

    # --- 5. 反例：结论跟人工标注不一致（rejection_sample.py 应该拒绝）----------------
    alert5 = {
        "alertname": "MockMismatchedConclusion",
        "service": "currency",
        "labels": {"service": "currency", "severity": "warning", "job": "currency"},
        "annotations": {"summary": "currency 延迟上升", "description": "currency 服务延迟上升。"},
        "startsAt": "2026-06-20T14:00:00Z",
    }
    steps5 = [
        _step("Bash", {"command": "docker stats --no-stream currency"}, "currency   CPU% 97.0"),
    ]
    diag5 = {
        "summary": "currency 服务判定为 info_only，无需处置。",
        "kind": "resource",
        # 人工标注（见 ground_truth/mock_fail_mismatch.json）期望 suspect_service 命中 currency
        # 且 remediation_type=online_op；这里故意写成不一致的结论，用于验证拒绝采样能正确拒绝。
        "suspect_service": "payment",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "info_only",
        "remediation_detail": "无需处置。",
        "confidence": 0.75,
        "evidence": ["docker stats 显示 currency CPU% 97.0"],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_fail_mismatch", alert5, steps5, diag5, schema, remediation))

    return trajectories


def _assemble(
    alert_id: str,
    alert: dict[str, Any],
    steps: list[dict[str, Any]],
    diag_fields: dict[str, Any],
    schema_mod,
    remediation_mod,
) -> dict[str, Any]:
    """把 (alert, steps, 手写 diagnosis 字段) 拼成一条完整轨迹 dict。

    用真实的 `agent.core.schema.Diagnosis` 校验 diag_fields——这保证 mock 数据不是
    随手编的字典，而是真的能通过 AIops-agent 生产代码里那份 pydantic 校验的合法 Diagnosis。
    `executed_actions` 从 steps 里 hook_decision=="allow" 的那些步骤反推（跟真实系统一致：
    executed_actions 不是模型自己填的，是从「真的被放行执行」的记录回填的）。
    """
    signals.annotate_steps(steps)

    executed_actions = [
        {
            "action": s["hook_action"],
            "target": s["hook_target"],
            "command": s["tool_input"].get("command", ""),
            "ts": "2026-06-20T00:00:00Z",
        }
        for s in steps
        if s["hook_decision"] == "allow"
    ]
    diag = schema_mod.Diagnosis.model_validate(diag_fields)
    diag.executed_actions = [schema_mod.ExecutedAction(**e) for e in executed_actions]

    route = _infer_route(diag)
    post_health = None
    if diag.remediation_type == "online_op" or executed_actions:
        post_health = {
            "checked": True,
            "healthy": True,
            "metric": "container_memory_usage_bytes" if diag.kind == "deploy_regression" else "http_error_rate",
            "window_s": 60,
        }

    return {
        "alert": alert,
        "alert_id": alert_id,
        "steps": steps,
        "final_diagnosis": diag.model_dump(by_alias=True),
        "ground_truth": None,
        "route": route,
        "post_action_health": post_health,
        "meta": {"source": "mock", "degraded": None, "raw": None},
    }


def _infer_route(diag) -> str:
    """按 AIops-agent/agent/run.py 里 _run_inner() 的同一套判定逻辑，从 Diagnosis 推出 route。

    这里只是复刻 run.py 的分支结构给 mock 数据打标（不 import run.py，因为它会连带
    拉进 code_fix.py -> claude_agent_sdk），跟真实编排层保持语义一致但不产生依赖。
    """
    if diag.confidence < 0.6:
        return "feishu_low_confidence"
    if diag.remediation_type == "online_op":
        auto_done = bool(diag.executed_actions)
        if diag.also_code_fix:
            return "auto_remediated_and_code_fix_pr" if auto_done else "online_op_and_code_fix_pr"
        return "auto_remediated" if auto_done else "feishu_online_op"
    if diag.remediation_type == "code_fix":
        return "code_fix_pr"
    return "info_only"


def _infer_route_real(diag) -> Optional[str]:
    """真实模式专用：跟 `_infer_route()` 同一套判定表，但如实处理「本脚本没跑代码修复 Agent」
    这个信息缺口。

    `_infer_route()` 是给 `build_mock_trajectories()` 手写数据打标签用的：那些 mock 场景里
    `remediation_type=="code_fix"` 或 `also_code_fix=True` 时，作者故意把数据构造成「代码修复
    一定验证通过」，所以它对这两种情形无条件返回 `code_fix_pr` /
    `auto_remediated_and_code_fix_pr` / `online_op_and_code_fix_pr`，从不检查
    `FixResult.verified`。但在 `run.py::_run_inner()` 里，这两种情形的真实 route 恰恰取决于
    代码修复 Agent 跑完之后 `fix.verified` 是 True 还是 False（True → `code_fix_pr` /
    `*_and_code_fix_pr`；False → `feishu_fix_unverified` / 退回 `auto_remediated`/
    `feishu_online_op`）——而 `_run_diagnose_capturing_steps()` 明确只跑诊断 Agent、不跑代码
    修复 Agent（见模块 docstring），所以这条信息在真实模式的轨迹里根本不存在。
    与其编造一个「假设总是验证通过」的乐观值，这里如实返回 None：
    - 置信度低于阈值的分支不受影响（逃生口发生在代码修复分支之前，`_infer_route()` 在这个
      分支上的判定跟是否跑代码修复 Agent 无关，可以放心直接复用）；
    - 其余不触碰代码修复分支的场景（`remediation_type` 是 `online_op`（且
      `also_code_fix=False`）或 `info_only`）route 是完全确定的，同样直接复用 `_infer_route()`。
    """
    if diag.confidence < 0.6:
        # 低置信度逃生口在 run.py 里发生在代码修复分支之前，跟是否跑过代码修复 Agent 无关，
        # 必须先判——否则会把「本该确定的 feishu_low_confidence」误判成「不确定」。
        return _infer_route(diag)
    if diag.remediation_type == "code_fix":
        return None
    if diag.remediation_type == "online_op" and diag.also_code_fix:
        return None
    return _infer_route(diag)


def mock_mode(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for traj in build_mock_trajectories():
        out_path = out_dir / f"{traj['alert_id']}.json"
        out_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="收集冷启轨迹（真实模式 / --mock 合成模式）")
    parser.add_argument("--mock", action="store_true", help="生成合成轨迹而不是真实调用 Agent")
    parser.add_argument("--check-env", action="store_true", help="只检查 AIops-agent 环境是否可 import，不发起任何调用")
    parser.add_argument("--alerts-dir", type=Path, default=DEFAULT_ALERTS_DIR, help="真实模式：告警 JSON 所在目录")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认真实模式写 trajectories/，mock 模式写 trajectories/mock/）")
    parser.add_argument("--sample", type=int, default=None, help="真实模式：只跑目录下前 N 条告警（不传则跑全部）")
    args = parser.parse_args()

    if args.check_env:
        print(json.dumps(check_env(), ensure_ascii=False, indent=2))
        return

    if args.mock:
        out_dir = args.out_dir or DEFAULT_MOCK_DIR
        written = mock_mode(out_dir)
        print(f"[mock] 写入 {len(written)} 条合成轨迹到 {out_dir}")
        for p in written:
            print(f"  - {p}")
        return

    out_dir = args.out_dir or DEFAULT_TRAJ_DIR
    scope = f"前 {args.sample} 条" if args.sample is not None else "全部"
    print(
        "[real] 即将对 "
        f"{args.alerts_dir} 下的{scope}告警发起真实 Agent 调用（需要真实 API key + docker/k8s 环境）。"
    )
    written = real_mode(args.alerts_dir, out_dir, sample=args.sample)
    print(f"[real] 写入 {len(written)} 条轨迹到 {out_dir}")


if __name__ == "__main__":
    main()
