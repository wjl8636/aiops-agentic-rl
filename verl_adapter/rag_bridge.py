#!/usr/bin/env python3
"""Mac 侧 RAG 桥：一次性 CLI 包装 ``AIops-agent`` 的 ``search_past_incidents``。

## 谁在调用本脚本（执行拓扑）
GRPO 训练跑在 GPU 训练机上，真实故障环境（docker 栈 + Milvus + BGE 模型）在
Mac 上。GPU 机上 ``verl_adapter/rollout_worker.py::ssh_tunnel_mcp_executor``
经 SSH 反向隧道（GPU 机的 ``localhost:2222``）把本脚本当作一条远程命令执行：

    ssh -p 2222 -- <mac_user>@localhost \
        /path/to/aiops-agentic-rl/AIops-agent/.venv/bin/python \
        /path/to/aiops-agentic-rl/verl_adapter/rag_bridge.py '<json args>'

stdout 输出单行 JSON 结果（形状 = ``rag_tool.search_past_incidents`` 的真实
返回：``{"content": [{"type": "text", "text": ...}], "is_error": bool}``，
``observation_packer.pack_mcp_result`` 直接消费这个形状）。失败走 stderr +
非 0 exit code，由调用方转成错误观测——本脚本自己**绝不抛裸异常到 stdout**。

## 为什么 import 嵌套子模块那份 AIops-agent，而不是根目录 checkout
仓库自包含：换一台机器 clone 本仓库就能跑（不需要"旁边恰好有另一份
checkout"）。RAG 路径需要的配置（``RAG_ENABLED``/``MILVUS_URI``/
``EMBEDDING_MODEL``/``EMBEDDING_DIM``/``RAG_TOP_K``）在 ``agent/config.py``
的默认值与根目录 checkout 的 ``.env`` 完全一致，而 ``.env`` 只影响 LLM
路由（``AIOPS_MODEL``/``AIOPS_LLM_*``），本桥不经过那条路径。

## ``.handler`` 这个细节（已实测核实，2026-09-08）
``rag_tool.search_past_incidents`` 被 ``claude_agent_sdk.tool`` 装饰器包成了
``SdkMcpTool`` 描述符对象——**不是可直接调用的函数**（直接 call 会
``TypeError: 'SdkMcpTool' object is not callable``），原始 async 函数挂在
``.handler`` 属性上（``dir()`` 实测：``annotations/description/handler/
input_schema/name``）。本脚本按"callable 就直接调，否则取 ``.handler``"的
顺序做版本兼容，两条路都走不通时报错退出。

## 用哪个 python 跑
必须用装了 ``claude_agent_sdk`` + ``pymilvus`` + ``sentence_transformers``
的解释器。本仓库嵌套的 ``AIops-agent/.venv`` 已验证三者齐全（2026-09-08
实测）；调用方（GPU 机的 ``ssh_tunnel_mcp_executor``）通过
``AIOPS_MAC_PYTHON`` 环境变量指定，默认值就是那个 venv。

## 已实测的开销（2026-09-08，Mac 本机冷调用）
BGE 模型（``~/.cache/modelscope/BAAI/bge-small-zh-v1.5``，``memory.py::
_resolve_model_path`` 期望的路径）加载 + Milvus collection load + 向量检索，
端到端约 9~10 秒。每次调用都是全新进程（一次性 CLI），所以这是**每次调用
都要付的冷启动成本**，不是只有第一次。GRPO rollout 的节奏（每条轨迹 8~12
轮工具调用、其中 RAG 约 1 次）下可以接受；如果以后成为瓶颈，再考虑改成
Mac 常驻的本地服务——今天不做（一次性 CLI 无状态、无生命周期管理，先求
正确）。

## 已验证 / 判断的分层
- [VERIFIED 2026-09-08] 端到端链路在本机实测跑通：BGE 定位 + 加载 +
  Milvus 检索返回真实工单（相似度 0.69 的 kafka 积压工单），耗时 ~9s。
- [VERIFIED] ``SdkMcpTool`` 不可直接调用、原始函数在 ``.handler``（见上）。
- [JUDGMENT] 一次性 CLI vs 常驻服务：选前者（简单、无状态），代价是每次
  ~9s 冷启动（见上）。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
AIOPS_AGENT_DIR = REPO_ROOT / "AIops-agent"


def _load_search_fn() -> Callable[[dict], Any]:
    """Import ``rag_tool.search_past_incidents`` and unwrap the SDK decorator.

    ``claude_agent_sdk.tool`` turns the function into an ``SdkMcpTool``
    descriptor (not directly callable); the original async function lives on
    its ``.handler`` attribute — see module docstring. Both orders are tried
    so an SDK version that keeps the function plain-callable also works.
    """
    if AIOPS_AGENT_DIR.is_dir():
        p = str(AIOPS_AGENT_DIR)
        if p not in sys.path:
            sys.path.insert(0, p)
    from agent.integrations import rag_tool  # noqa: PLC0415 (deliberate late import: keeps CLI arg errors fast)

    fn = rag_tool.search_past_incidents
    if callable(fn):
        return fn
    handler = getattr(fn, "handler", None)
    if callable(handler):
        return handler
    raise RuntimeError(
        "rag_bridge: rag_tool.search_past_incidents is neither callable nor exposes a callable .handler "
        f"(type={type(fn).__name__}) — claude_agent_sdk 的 tool 装饰器行为可能变了，请核对 SDK 版本"
    )


async def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} '<json args dict, e.g. {{\"query\": \"...\"}}>'", file=sys.stderr)
        return 2
    try:
        args = json.loads(argv[1])
        if not isinstance(args, dict):
            raise ValueError(f"args must be a JSON object, got {type(args).__name__}")
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"rag_bridge: bad args JSON: {exc}", file=sys.stderr)
        return 2

    search_fn = _load_search_fn()
    try:
        result = await search_fn(args)
    except Exception as exc:  # noqa: BLE001 (CLI boundary: failure -> stderr + exit code, never a traceback on stdout)
        print(f"rag_bridge: search_past_incidents failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # 单行 JSON 到 stdout（BGE/Milvus 的进度条和 deprecation 告警都走 stderr，
    # 不会污染协议输出——2026-09-08 实测确认）。
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv)))
