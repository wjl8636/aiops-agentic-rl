"""共享的规则式信号分类工具，供 `collect_trajectories.py` 给每一步观测打「覆盖了哪类信号」
和「是否对同一目标重复调用且无新信息」两个标注。

对应设计文档 §5.1（step-level dense reward 的判定输入）与 §5.3 案例四（细粒度信用分配要用
「本轮观测覆盖了几类信号」）——这两处都要用同一套规则，所以单独抽成这个模块，
`collect_trajectories.py` 在真实模式/mock 模式都调用它，保证标注口径一致。

注意：这里只做「打标注」，不做 reward 数值计算——数值计算属于 `reward/` 模块的职责，
本文件只产出 reward 模块消费得到的中间特征（signal_types_covered / is_repeat_no_new_info）。
"""
from __future__ import annotations

import json
import re
from typing import Any

# 信号种类 -> 命中该信号的正则关键字（在 "tool_name + tool_input + observation" 拼成的文本上匹配）。
# 六类信号对应设计文档反复提到的「Prometheus / Jaeger / logs / kubectl 状态 / git 发版历史 / RAG 历史工单」
# 三源交叉验证原则的扩展版（拆成六类以便细粒度信用分配按 kind 差异化加权）。
SIGNAL_KEYWORDS: dict[str, list[str]] = {
    "prometheus": [r"prometheus", r":9090\b", r"/api/v1/query"],
    "jaeger": [r"jaeger", r":16686\b", r"/api/traces"],
    "logs": [r"kubectl\s+logs", r"docker\s+logs", r"\.log\b", r"deploys\.log"],
    "kubectl_status": [
        r"kubectl\s+(get|describe|top)\b",
        r"docker\s+(ps|stats|inspect)\b",
    ],
    "git_history": [r"git\s+(log|show|diff|blame)\b"],
    "rag_history": [r"search_past_incidents", r"历史工单", r"past_incidents"],
}


def classify_signal_types(tool_name: str, tool_input: dict[str, Any], observation: str) -> list[str]:
    """规则匹配：这一步的调用+观测覆盖了哪些信号类型。返回值按 SIGNAL_KEYWORDS 的键顺序排列。"""
    text = f"{tool_name} {json.dumps(tool_input, ensure_ascii=False)} {observation}".lower()
    covered = []
    for sig, patterns in SIGNAL_KEYWORDS.items():
        if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
            covered.append(sig)
    return covered


def normalize_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """把一次工具调用规整成可比较的 key（同一命令的等价写法应该归一到同一个 key）。"""
    if tool_name == "Bash":
        command = str(tool_input.get("command", "")).strip()
        command = re.sub(r"\s+", " ", command)
        return f"Bash:{command}"
    return f"{tool_name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"


def annotate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """就地给每个 step 补上 signal_types_covered / is_repeat_no_new_info，并原样返回。

    「重复无新信息」的判定规则（保持简单、显式可复核）：
    这一步的调用 key 之前已经出现过，且这次覆盖到的信号种类集合并没有比之前任何一次
    同 key 调用更大（即没有带来新的信号维度）——直接判为重复。
    """
    seen_max_signals: dict[str, set[str]] = {}
    for step in steps:
        tool_name = step.get("tool_name", "")
        tool_input = step.get("tool_input", {}) or {}
        observation = step.get("observation", "") or ""

        covered = classify_signal_types(tool_name, tool_input, observation)
        step["signal_types_covered"] = covered

        key = normalize_call(tool_name, tool_input)
        prev_signals = seen_max_signals.get(key)
        if prev_signals is not None and set(covered) <= prev_signals:
            step["is_repeat_no_new_info"] = True
        else:
            step["is_repeat_no_new_info"] = False
        seen_max_signals[key] = prev_signals | set(covered) if prev_signals else set(covered)
    return steps
