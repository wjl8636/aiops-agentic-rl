"""把 `prefix_split.py` 的输出转成 LLaMA-Factory 的 SFT 训练格式。

## sharegpt 格式约定
1. sharegpt 格式的对话是 `{"from": "human"/"gpt"/"function_call"/"observation", "value": ...}` 的列表，
   顺序约束是「human/observation 在奇数位，gpt/function_call 在偶数位」（工具调用多轮对话规范）。
2. 顶层每条样本除了 `"conversations"`，还可以带可选的 `"system"`（系统提示词）和
   `"tools"`（工具schema的字符串描述）两个字段。
3. `dataset_info.json` 里对应的注册项形如
   `{"formatting": "sharegpt", "columns": {"messages": "conversations", "system": "system", "tools": "tools"}}`
   （`"tags"` 覆盖 role/content 别名是可选的，本文件用的都是默认 tag 名，不需要覆盖）。
这部分是**核实过的**，见本文件同目录下生成的 `dataset_info_snippet.json`。

## 我做的一个没法反查官方文档、需要以后验证的假设（明确标出来）
LLaMA-Factory 的 sharegpt 工具调用格式是给「多轮、真的执行了工具」的对话设计的
（`function_call` 后面接一条 `observation`，再接 `gpt` 的最终回复）。但 `prefix_split.py`
产出的每条子样本被设计成「只对最后一步计算 loss」——如果照抄多轮格式，把 prefix_steps 也
铺成 function_call/observation 轮次放进同一条 conversations 里，LLaMA-Factory 的 sharegpt SFT
默认会对**每一轮** `gpt`/`function_call` 都计算 loss（不会只挑最后一轮），这样就违反了
渐进式前缀拆分「只学最后一步」的设计初衷。
**假设/选择**：为了让「只对最后一步计算 loss」在 LLaMA-Factory 默认训练脚本下就成立
（不用改 LLaMA-Factory 的 mask 逻辑），本文件把每条子样本压成**单轮对话**——`prefix_steps`
的历史被拍平成一段文本，塞进唯一的 `human` 轮里（当作"已执行步骤"叙述），`conversations`
里只留一条 assistant 轮（`function_call`：预测下一步工具调用；`gpt`：预测最终 Diagnosis）。
这样这条样本里唯一的一轮 assistant 输出就是训练目标，天然只对它计算 loss，不需要额外配置。
代价是放弃了 LLaMA-Factory 原生「多轮工具调用」的样子（不追求视觉上像一次真实的多轮 Agent
对话），换来"loss 目标可控"这个更重要的约束。**这是本文件对 LLaMA-Factory 数据格式的一个
产品设计假设，不是从官方文档验证来的，SFT 任务接手时如果发现 LLaMA-Factory 版本支持
"loss_mask"/"only last turn"这类原生开关，可以改回多轮铺开的写法，本文件的假设请务必复核。**
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
AIOPS_AGENT_DIR = REPO_ROOT / "AIops-agent"

DEFAULT_IN_PATH = HERE / "prefix_subsamples.mock.json"
DEFAULT_OUT_JSONL = HERE / "aiops_cold_start_sft.mock.jsonl"
DEFAULT_DATASET_INFO_SNIPPET = HERE / "dataset_info_snippet.json"
DATASET_NAME = "aiops_cold_start_prefix_split"

# 诊断处置 Agent 真实开放的工具面就是 Bash/Read/Grep/Glob（见 AIops-agent/agent/agents/diagnose.py
# 的 allowed_tools）。这里给出的 JSON schema 是本文件自己写的简化版占位 schema——AIops-agent
# 用的是 Claude Code 内置工具，仓库里没有一份现成的、给 LLaMA-Factory "tools" 字段用的 JSON schema
# 文本，所以这份是按 Claude Code 文档里 Bash/Read/Grep/Glob 的公开参数形状手写的近似，
# 不是从仓库任何文件抓出来的，SFT 接手时如需更精确的工具 schema 请对照 Claude Code 官方文档核对。
_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "Bash",
        "description": "执行一条只读观测/低风险处置 Bash 命令（受 PreToolUse hook 白名单硬约束）。",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "要执行的 shell 命令"}},
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "读取一个文件的内容（如 deploys.log）。",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Grep",
        "description": "在文件/目录里按正则搜索内容。",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "Glob",
        "description": "按 glob 模式匹配文件路径。",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]
TOOLS_JSON_STR = json.dumps(_TOOLS_SCHEMA, ensure_ascii=False)


def _ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def _load_prompt_fns():
    """复用 AIops-agent 真实的提示词函数（不需要 claude_agent_sdk，跟 collect_trajectories.py
    里 `_load_aiops_core()` 同样的道理），保证「human 轮的告警描述」跟生产提示词一字不差。
    拿不到时退化成一个简化版占位提示词（比如这台机器没有 AIops-agent 子模块），不阻塞转换流程。
    """
    try:
        _ensure_aiops_agent_on_path()
        from agent.agents import prompts  # type: ignore

        return prompts.diagnose_prompt, prompts.diagnose_append
    except Exception:  # noqa: BLE001

        def _fallback_prompt(alert: dict[str, Any]) -> str:
            return f"请诊断以下告警：\n```json\n{json.dumps(alert, ensure_ascii=False, indent=2)}\n```\n"

        def _fallback_append() -> str:
            return "你是 AIOps 故障诊断处置 Agent，只读排查后输出结构化 Diagnosis。"

        return _fallback_prompt, _fallback_append


_diagnose_prompt, _diagnose_append = _load_prompt_fns()
SYSTEM_PROMPT = _diagnose_append()


def _render_prior_steps_block(prefix_steps: list[dict[str, Any]]) -> str:
    if not prefix_steps:
        return "（尚未执行任何工具调用。）"
    lines = ["已执行步骤："]
    for idx, step in enumerate(prefix_steps, start=1):
        hook_note = ""
        if step.get("hook_decision") == "deny":
            hook_note = "  [被 hook 拦截：未真正执行]"
        elif step.get("hook_decision") == "allow":
            hook_note = f"  [hook 已放行自动处置：{step.get('hook_action')}→{step.get('hook_target')}]"
        lines.append(
            f"{idx}. 工具: {step.get('tool_name')}  参数: {json.dumps(step.get('tool_input', {}), ensure_ascii=False)}{hook_note}\n"
            f"   观测: {step.get('observation', '')}"
        )
    return "\n".join(lines)


def _render_human(record: dict[str, Any]) -> str:
    base = _diagnose_prompt(record.get("alert") or {})
    prior = _render_prior_steps_block(record.get("prefix_steps") or [])
    if record["target_type"] == "tool_call":
        ask = "请给出下一步要调用的工具（只输出一个 JSON 对象：{\"tool_name\": ..., \"tool_input\": {...}}，不要多余文字）。"
    else:
        ask = "现有观测是否已经足够收敛？请输出最终结构化 Diagnosis JSON（严格符合 Diagnosis schema）。"
    return f"{base}\n{prior}\n\n{ask}"


def _render_target_turn(record: dict[str, Any]) -> dict[str, str]:
    if record["target_type"] == "tool_call":
        target = record["target"]
        value = json.dumps({"name": target["tool_name"], "arguments": target["tool_input"]}, ensure_ascii=False)
        return {"from": "function_call", "value": value}
    # target_type == "diagnosis"：最终结构化结论，作为 assistant 的最终回复
    value = json.dumps(record["target"], ensure_ascii=False)
    return {"from": "gpt", "value": value}


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    """单条 prefix_split 子样本 -> 单条 LLaMA-Factory sharegpt 格式样本（见模块 docstring 的单轮设计假设）。"""
    human_turn = {"from": "human", "value": _render_human(record)}
    target_turn = _render_target_turn(record)
    return {
        "conversations": [human_turn, target_turn],
        "system": SYSTEM_PROMPT,
        "tools": TOOLS_JSON_STR,
    }


def convert_all(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [convert_record(r) for r in records]


DATASET_INFO_SNIPPET: dict[str, Any] = {
    DATASET_NAME: {
        "file_name": DEFAULT_OUT_JSONL.name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
        # 用的都是 sharegpt 默认 tag 名（from/value/human/gpt/function_call/observation），
        # 不需要覆盖 "tags"；见模块 docstring 的核实说明。
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="把 prefix_split.py 的输出转成 LLaMA-Factory sharegpt jsonl")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN_PATH)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--dataset-info-out", type=Path, default=DEFAULT_DATASET_INFO_SNIPPET)
    args = parser.parse_args()

    records = json.loads(args.in_path.read_text(encoding="utf-8"))
    converted = convert_all(records)

    with args.out_path.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    snippet = dict(DATASET_INFO_SNIPPET)
    snippet[DATASET_NAME]["file_name"] = args.out_path.name
    args.dataset_info_out.write_text(json.dumps(snippet, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[to_llamafactory_format] {len(converted)} 条样本 -> {args.out_path}")
    print(f"[to_llamafactory_format] dataset_info.json 注册片段 -> {args.dataset_info_out}")
    print(f"[to_llamafactory_format] 把下面这段合并进 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps(snippet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
