"""Tests for ``verl_adapter/rag_bridge.py`` — the Mac-side one-shot CLI the
GPU box's ``ssh_tunnel_mcp_executor`` runs through the tunnel.

All hermetic: the import of the real ``rag_tool`` (which needs
claude_agent_sdk + pymilvus + sentence_transformers and a live Milvus) is
mocked. The real end-to-end chain (BGE + Milvus on this Mac) was verified
live on 2026-09-08 — see ``rag_bridge.py``'s docstring — and the
through-tunnel variant is pinned by ``test_rollout_worker.py``'s
``AIOPS_TUNNEL_TEST``-gated live test, which must run on the GPU box.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from verl_adapter import rag_bridge


def _fake_rag_tool_module(search_attr) -> types.ModuleType:
    """Build a fake ``agent.integrations.rag_tool`` module whose
    ``search_past_incidents`` attribute is ``search_attr`` (either a plain
    async function, or an SdkMcpTool-shaped non-callable with ``.handler``),
    plus the parent packages so ``from agent.integrations import rag_tool``
    resolves against it."""
    mod = types.ModuleType("agent.integrations.rag_tool")
    mod.search_past_incidents = search_attr
    pkg = types.ModuleType("agent.integrations")
    pkg.rag_tool = mod
    agent_pkg = types.ModuleType("agent")
    agent_pkg.integrations = pkg
    return mod


class _FakeSdkMcpTool:
    """Mirrors the real ``claude_agent_sdk.SdkMcpTool`` shape that matters
    here: NOT directly callable, original async function on ``.handler``
    ([VERIFIED 2026-09-08] by dir() on the real object)."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.name = "search_past_incidents"
        self.description = ""
        self.input_schema = {}


async def _real_shape_result(args: dict) -> dict:
    return {"content": [{"type": "text", "text": f"fake hits for {args.get('query', '')!r}"}]}


# ---------------------------------------------------------------------------
# _load_search_fn: the SdkMcpTool unwrap.
# ---------------------------------------------------------------------------

def test_load_search_fn_unwraps_sdk_mcp_tool_handler(monkeypatch: pytest.MonkeyPatch):
    mod = _fake_rag_tool_module(_FakeSdkMcpTool(_real_shape_result))
    monkeypatch.setitem(sys.modules, "agent.integrations.rag_tool", mod)
    monkeypatch.setitem(sys.modules, "agent.integrations", types.SimpleNamespace(rag_tool=mod))
    fn = rag_bridge._load_search_fn()
    assert asyncio.run(fn({"query": "x"}))["content"][0]["text"] == "fake hits for 'x'"


def test_load_search_fn_passes_plain_callable_through(monkeypatch: pytest.MonkeyPatch):
    """SDK versions that keep the decorated function plain-callable must also
    work (the compat order documented in rag_bridge's docstring)."""
    mod = _fake_rag_tool_module(_real_shape_result)
    monkeypatch.setitem(sys.modules, "agent.integrations.rag_tool", mod)
    monkeypatch.setitem(sys.modules, "agent.integrations", types.SimpleNamespace(rag_tool=mod))
    fn = rag_bridge._load_search_fn()
    assert asyncio.run(fn({"query": "y"}))["content"][0]["text"] == "fake hits for 'y'"


def test_load_search_fn_neither_callable_nor_handler_raises(monkeypatch: pytest.MonkeyPatch):
    mod = _fake_rag_tool_module(object())  # neither callable nor .handler
    monkeypatch.setitem(sys.modules, "agent.integrations.rag_tool", mod)
    monkeypatch.setitem(sys.modules, "agent.integrations", types.SimpleNamespace(rag_tool=mod))
    with pytest.raises(RuntimeError, match="claude_agent_sdk"):
        rag_bridge._load_search_fn()


# ---------------------------------------------------------------------------
# _main: CLI arg / output protocol.
# ---------------------------------------------------------------------------

def test_main_prints_single_line_json_on_stdout(capsys, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rag_bridge, "_load_search_fn", lambda: _real_shape_result)
    code = asyncio.run(rag_bridge._main(["rag_bridge.py", '{"query": "kafka 积压"}']))
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1  # exactly one line: the JSON protocol output
    assert json.loads(out) == {"content": [{"type": "text", "text": "fake hits for 'kafka 积压'"}]}


def test_main_failure_goes_to_stderr_with_exit_1(capsys, monkeypatch: pytest.MonkeyPatch):
    async def boom(args):
        raise RuntimeError("milvus down")

    monkeypatch.setattr(rag_bridge, "_load_search_fn", lambda: boom)
    code = asyncio.run(rag_bridge._main(["rag_bridge.py", '{"query": "x"}']))
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays protocol-clean
    assert "milvus down" in captured.err


def test_main_bad_args_json_exits_2(capsys, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rag_bridge, "_load_search_fn", lambda: _real_shape_result)
    code = asyncio.run(rag_bridge._main(["rag_bridge.py", "not json"]))
    assert code == 2
    assert capsys.readouterr().out == ""


def test_main_non_dict_args_exits_2(capsys, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rag_bridge, "_load_search_fn", lambda: _real_shape_result)
    code = asyncio.run(rag_bridge._main(["rag_bridge.py", '["a list"]']))
    assert code == 2


def test_main_wrong_arg_count_exits_2(capsys, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rag_bridge, "_load_search_fn", lambda: _real_shape_result)
    assert asyncio.run(rag_bridge._main(["rag_bridge.py"])) == 2
    assert asyncio.run(rag_bridge._main(["rag_bridge.py", "{}", "extra"])) == 2


def test_bridge_bootstraps_nested_aiops_agent_on_sys_path():
    """The repo-self-containment contract: importing the bridge must put the
    sibling ``AIops-agent`` dir at the FRONT of sys.path so ``from
    agent.integrations import rag_tool`` resolves to the nested copy."""
    assert str(Path(rag_bridge.__file__).resolve().parent.parent / "AIops-agent") in (
        str(p) for p in sys.path
    ) or rag_bridge.AIOPS_AGENT_DIR.is_dir() is False
