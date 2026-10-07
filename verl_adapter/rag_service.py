#!/usr/bin/env python3
"""Mac 侧常驻 RAG 服务：启动时加载一次 BGE + Milvus collection，之后每次
查询 ~0.1-0.5s——消除 ``rag_bridge.py`` 一次性 CLI 每调用 ~10s 的冷启动
（Python+torch import ~3-5s、BGE 加载 ~1-2s、Milvus 建连+collection load
~2-3s，真正的 embed+检索 <1s，冷启动占比 >90%）。

## 部署形态（跟反向隧道同一个管理模式）
screen 会话常驻跑在本 Mac 上（restart-loop 包装，崩了 5 秒后自动拉起）：

    screen -dmS rag_service bash -c \
      'while true; do cd <repo> && AIops-agent/.venv/bin/python \
       verl_adapter/rag_service.py >> ~/tunnel_logs/rag_service.log 2>&1; \
       sleep 5; done'

GPU 训练机上的 ``rollout_worker.ssh_tunnel_mcp_executor`` 经隧道
``ssh ... "curl -s -m <t> -X POST http://localhost:<port>/search -d <json>"``
调用本服务；服务不可达时该 executor 自动回退到一次性 bridge（见其
docstring），所以本服务死掉只损失速度、不中断训练。

## HTTP 协议
- ``GET /healthz`` -> ``{"ok": true, "model": ..., "milvus_uri": ...}``（liveness）
- ``POST /search`` body ``{"query": "...", "top_k": 5?}`` ->
  **跟 ``rag_tool.search_past_incidents`` 完全相同的 MCP 形状**：
  ``{"content": [{"type": "text", "text": ...}], "is_error": bool}``。
  应用层失败（Milvus 挂了等）也返回 200 + ``is_error`` body——调用方
  （executor）把它原样交给模型当错误观测，**不触发**它的 bridge 回退
  （回退只留给连接层失败：服务没起/崩了；Milvus 挂了 bridge 也一样挂，
  回退没有意义）。请求 JSON 不合法返回 400（executor 侧的 bug，不是
  运行环境问题）。

## 复用与依赖
- 检索逻辑零复制：``from verl_adapter.rag_bridge import _load_search_fn``
  （含 SdkMcpTool.handler 解包 + 嵌套 AIops-agent 的 sys.path 自举），
  启动时加载一次，之后每次请求直接调用。
- 纯标准库 HTTP（``ThreadingHTTPServer``），不引 fastapi/uvicorn——
  单端点 localhost 服务，标准库足够，也避免给 venv 加依赖。
- 并发：ThreadingHTTPServer 每请求一线程；BGE encode 与 Milvus 检索
  加一把模块级锁串行化（单次 <1s，串行已远快于此前每次 10s 的冷启动，
  且避开 sentence-transformers 并发 encode 的线程安全隐患）。
- 跑它的解释器：嵌套 ``AIops-agent/.venv``（claude_agent_sdk + pymilvus
  + sentence_transformers 齐全，2026-09-08 验证）。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl_adapter.rag_bridge import _load_search_fn  # noqa: E402 (needs the sys.path bootstrap above)

DEFAULT_PORT = 18765
_PORT_ENV = "AIOPS_RAG_SERVICE_PORT"

#: 启动时加载一次（含 BGE 模型 + Milvus collection load 的全部冷启动成本）。
_search_fn: Callable[[dict], Any] | None = None
_search_lock = threading.Lock()


def _init_search_fn() -> None:
    """[2026-09-08 修复] 此前这里只拿到 handler 引用就认为"预热完成"——但
    ``rag_tool.search_past_incidents`` 对 BGE/Milvus 的加载是**懒加载**
    （在 ``memory._get_model()``/``memory.search_tickets()`` 内部，第一次
    真实调用才触发），本函数原来只是拿到了一个可调用对象，没有真的触发那次
    加载。实测后果：服务 `/healthz` 立刻返回正常，但**第一个真实 `/search`
    请求**要在原本以为"预热完"的基础上再付一次完整冷启动（BGE 加载+
    Milvus 建连，实测 15s+），跟"model warm, ready"这句日志矛盾。

    修法：启动时真的发一次非空查询（走 BGE embed + Milvus 检索的完整真实
    路径），逼出这次加载，让日志"ready"字面意义上属实。查询内容任选（结果
    不使用，只是触发路径）；失败也不阻塞启动（比如 Milvus 暂时没起来）——
    交给第一次真实请求去报错/重试，比服务干脆起不来更安全。
    """
    global _search_fn
    _search_fn = _load_search_fn()
    if not callable(_search_fn):
        raise RuntimeError("rag_service: _load_search_fn returned a non-callable")
    try:
        asyncio.run(_search_fn({"query": "服务启动预热探测"}))
    except Exception as exc:  # noqa: BLE001 — warm-up failure must not block startup, see docstring above
        print(f"[rag_service] warm-up query failed (will retry lazily on first real request): {exc}", flush=True)


class _RagHandler(BaseHTTPRequestHandler):
    """协议见模块 docstring。所有输出单行 JSON，跟 bridge/rag_tool 同形。"""

    def _respond_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server 的命名约定)
        if self.path == "/healthz":
            self._respond_json(200, {"ok": True})
        else:
            self._respond_json(404, {"ok": False, "error": f"unknown path {self.path!r}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/search":
            self._respond_json(404, {"is_error": True, "content": [{"type": "text", "text": f"unknown path {self.path!r}"}]})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            args = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(args, dict):
                raise ValueError(f"body must be a JSON object, got {type(args).__name__}")
        except (json.JSONDecodeError, ValueError) as exc:
            self._respond_json(400, {"is_error": True, "content": [{"type": "text", "text": f"bad request body: {exc}"}]})
            return
        if _search_fn is None:
            self._respond_json(500, {"is_error": True, "content": [{"type": "text", "text": "rag_service: search fn not initialized"}]})
            return
        try:
            with _search_lock:
                result = asyncio.run(_search_fn(args))
        except Exception as exc:  # noqa: BLE001 (应用层失败 -> is_error body，见模块 docstring 的 200 约定)
            result = {"is_error": True, "content": [{"type": "text", "text": f"rag_service search failed: {type(exc).__name__}: {exc}"}]}
        self._respond_json(200, result)

    def log_message(self, fmt: str, *args: Any) -> None:  # 默认写 stderr 的访问日志改写 stdout（screen 日志统一收口）
        print(f"[rag_service] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    port = int(os.environ.get(_PORT_ENV) or DEFAULT_PORT)
    _init_search_fn()
    server = ThreadingHTTPServer(("127.0.0.1", port), _RagHandler)
    print(f"[rag_service] listening on 127.0.0.1:{port} (env {_PORT_ENV} overrides; model warm, ready)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
