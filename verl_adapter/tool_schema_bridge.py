"""Bridges the AIops-agent diagnose-agent tool surface (raw Claude Agent SDK
builtin tools ``Bash``/``Read``/``Grep``/``Glob`` + the RAG MCP tool
``search_past_incidents``) into veRL's rollout tool-calling schema format.

## What format does veRL actually expect? (documented source, not guessed)
veRL is NOT installed in this environment (per task constraints — do not
try to install it). This module's schema shape was determined by fetching
the real public source of ``volcengine/verl`` on 2026-08-30 via WebFetch.

Confirmed from that source read: veRL describes tools with OpenAI's
function-calling schema (``OpenAIFunctionToolSchema``)::

    {"type": "function",
     "function": {"name": ..., "description": ..., "strict": bool,
                  "parameters": {"type": "object", "properties": {...},
                                 "required": [...]}}}

and a ``BaseTool`` subclass implements an async lifecycle —
``create(instance_id, **kwargs) -> (instance_id, ToolResponse)`` /
``execute(instance_id, parameters, **kwargs) -> (ToolResponse, reward, metrics)``
/ ``calc_reward(instance_id, **kwargs) -> float`` /
``release(instance_id, **kwargs) -> None`` — where ``ToolResponse`` is
``{"text": ..., "image": [...], "video": [...]}`` (this project only ever
uses ``text``). ``ToolAgentLoop`` drives the multi-turn loop: sample an
assistant turn -> parse ``tool_calls`` out of it via a pluggable
``ToolParser`` -> dispatch each call through the create/execute/release
lifecycle (or a plain callable, for a stateless "FunctionTool") -> append
one tool-response message per call back into the running message list,
tagged with a ``tool_call_id`` -> loop back into another generation call.
This maps directly onto what this project needs: one OpenAI-function-
calling schema per tool, and a create/execute/release contract per call —
see ``rollout_worker.py::RealHookRoutedTool`` for the concrete duck-typed
implementation of that contract that routes through the real hook.

## Why this bridge stays deliberately "thin" (per task, load-bearing)
The diagnose agent (``AIops-agent/agent/agents/diagnose.py`` +
``agent/agents/prompts.py::diagnose_append()``) does NOT give the model N
narrow per-CLI tool schemas — it gives the model ONE generic ``Bash`` tool
(a free-form shell-command string) and tells it, in the system prompt,
which CLIs to run (``kubectl``/``docker``/``curl``/``git``/``rg``/Skills).
There is no rich "kubectl_get_pod(namespace, pod)"-shaped schema to
translate. Being honest about that here means the schemas below stay
deliberately generic (mirroring the SDK's own built-in tool shapes), not an
invented rich API — this module is mostly a pass-through/wrapper, exactly
as the task describes it.

## [VERIFIED 2026-09-07] ``tool_config.yaml``'s real load-time shape
Confirmed by SSH'ing into the GPU training box (host alias ``aiops-gpu``,
real ``verl`` source vendored at ``/root/autodl-tmp/code/verl/``) and
reading ``verl/tools/tool_registry.py::initialize_tools_from_config()``
directly (not inferred this time — this supersedes the old "flagged
assumption #1" below):
  - The YAML's top-level key is ``tools:`` (real code iterates
    ``for tool_config in tools_config.tools:``) — it is NOT a bare list at
    the document root. ``build_tool_config_entries()`` below still returns
    a bare *list* of per-tool entries (unchanged, and still consumed as a
    list by ``build_verl_tool_schemas()``/tests); it is the *file generator*
    (``generate_tool_config.py``) that wraps that list under a ``tools:``
    key before writing ``tool_config.yaml`` to disk.
  - Each entry's ``config`` dict MUST carry a ``type`` key (real code does
    ``tool_type = ToolType(tool_config.config.type)``, and ``ToolType`` is
    currently a single-member enum whose only value is ``"native"``).
    ``build_tool_config_entries()`` below sets ``"type": "native"`` on every
    entry's ``config`` dict for this reason — this was a real gap (config
    was previously missing the key entirely) that is now fixed at the
    source function, not patched onto the generated YAML after the fact.
  - The per-entry ``{"class_name", "config", "tool_schema"}`` outer shape
    itself (how ``class_name`` resolves to a ``BaseTool`` subclass) was
    already right; only the missing ``config.type`` key was the bug.

## Flagged assumptions — correct against the real installed veRL package
  1. The exact ``ToolParser`` format string (e.g. "hermes"/"qwen"/...) that
     ``rollout_config.multi_turn.format`` would need for a Qwen3.5 base
     model was not determined here — left as a TODO for whoever wires this
     against a real installed veRL + LLaMA-Factory-SFT'd Qwen3.5 checkpoint.
  2. ``Read``/``Grep``/``Glob``'s parameter schemas below are a best-effort
     mirror of Claude Code's builtin tool shapes (``file_path``/``pattern``/
     etc.), not pulled from a public SDK-exposed JSON schema (the SDK does
     not expose one to callers for its builtins) — treat these three as
     approximate; ``Bash``'s schema (the tool this whole project actually
     exercises for remediation) and ``search_past_incidents``'s schema (read
     straight off ``rag_tool.py``'s ``@tool(...)`` decorator) are the two
     that matter and are grounded in real source, not guessed.
"""
from __future__ import annotations

from typing import Any

#: Raw AIops-agent diagnose-agent tool surface. Source:
#: data/cold_start/collect_trajectories.py's `allowed = ["Bash", "Read", "Grep", "Glob"]`
#: (which mirrors AIops-agent/agent/agents/diagnose.py's allowed_tools wiring).
AIOPS_AGENT_BUILTIN_TOOLS: tuple[str, ...] = ("Bash", "Read", "Grep", "Glob")

#: AIops-agent/agent/integrations/rag_tool.py::TOOL_NAME — the in-process
#: MCP tool mounted only on the diagnose agent (server key "aiops_rag" +
#: tool name "search_past_incidents", per the SDK's mcp naming convention).
RAG_TOOL_NAME = "mcp__aiops_rag__search_past_incidents"


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    """Build one ``OpenAIFunctionToolSchema``-shaped dict (see module
    docstring for the source confirming this shape)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
            "strict": False,
        },
    }


def bash_tool_schema() -> dict[str, Any]:
    """Deliberately generic — see module docstring: the real ``Bash`` tool
    takes one free-form shell-command string; the prompt (not this schema)
    is what tells the model which CLIs to run, and the real PreToolUse hook
    (not this schema) is what actually gates execution."""
    return _tool_schema(
        "Bash",
        "Execute a shell command. Read-only observability commands (kubectl get/"
        "describe/top/logs, docker ps/stats/inspect/logs, curl GET, git log/show/"
        "diff/blame, rg/grep/cat/ls) are unrestricted. Low-risk remediation "
        "commands are auto-executed ONLY if they match the AIops-agent "
        "remediation.py whitelist AND target an authorized instance — every "
        "other command (including a whitelisted action against an "
        "unauthorized target) is denied by the real PreToolUse hook before it "
        "ever runs.",
        {"command": {"type": "string", "description": "The shell command to run."}},
        ["command"],
    )


def read_tool_schema() -> dict[str, Any]:
    """Approximate mirror of the SDK's builtin ``Read`` tool (see module
    docstring's flagged assumption #3)."""
    return _tool_schema(
        "Read",
        "Read a file from the local filesystem (read-only).",
        {"file_path": {"type": "string", "description": "Absolute path to the file to read."}},
        ["file_path"],
    )


def grep_tool_schema() -> dict[str, Any]:
    """Approximate mirror of the SDK's builtin ``Grep`` tool (see module
    docstring's flagged assumption #3)."""
    return _tool_schema(
        "Grep",
        "Search file contents with a regex pattern (read-only).",
        {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search (optional)."},
        },
        ["pattern"],
    )


def glob_tool_schema() -> dict[str, Any]:
    """Approximate mirror of the SDK's builtin ``Glob`` tool (see module
    docstring's flagged assumption #3)."""
    return _tool_schema(
        "Glob",
        "Find files matching a glob pattern (read-only).",
        {"pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."}},
        ["pattern"],
    )


def search_past_incidents_tool_schema() -> dict[str, Any]:
    """Derived directly from
    ``AIops-agent/agent/integrations/rag_tool.py``'s
    ``@tool("search_past_incidents", <description>, {"query": str, "top_k": int})``
    decorator — real parameter names/types/description, not guessed."""
    return _tool_schema(
        RAG_TOOL_NAME,
        "语义检索历史相似的诊断工单（根因/处置经验）。当你想参考过去是否处理过类似故障、"
        "或想借鉴历史根因与处置方式时调用。传入描述当前故障的自然语言查询（如告警症状、"
        "可疑服务、现象），返回最相似的若干历史工单摘要。",
        {
            "query": {"type": "string", "description": "描述当前故障的自然语言查询。"},
            "top_k": {"type": "integer", "description": "返回的历史工单条数。"},
        },
        ["query"],
    )


_SCHEMA_BUILDERS = {
    "Bash": bash_tool_schema,
    "Read": read_tool_schema,
    "Grep": grep_tool_schema,
    "Glob": glob_tool_schema,
    RAG_TOOL_NAME: search_past_incidents_tool_schema,
}


def build_verl_tool_schemas(include_rag: bool = True) -> list[dict[str, Any]]:
    """The full list of ``OpenAIFunctionToolSchema``-shaped dicts for the
    diagnose agent's tool surface, in a stable order (builtins first, RAG
    tool last). Set ``include_rag=False`` to mirror a config where
    ``AIOPS_RAG_ENABLED=0`` (see ``agent/config.py::RAG_ENABLED``).
    """
    names = list(AIOPS_AGENT_BUILTIN_TOOLS) + ([RAG_TOOL_NAME] if include_rag else [])
    return [_SCHEMA_BUILDERS[n]() for n in names]


def build_tool_config_entries(
    include_rag: bool = True,
    class_name: str = "verl_adapter.rollout_worker.RealHookRoutedTool",
) -> list[dict[str, Any]]:
    """Shaped like veRL's ``tool_config.yaml`` list entries — see the module
    docstring's "[VERIFIED 2026-09-07]" section for how this shape was
    confirmed against the real ``verl/tools/tool_registry.py`` source
    (``initialize_tools_from_config()``): each entry is
    ``{"class_name", "config", "tool_schema"}``, and ``config`` MUST include
    ``"type": "native"`` (``ToolType`` is a single-member enum whose only
    value is ``"native"``; the loader does
    ``ToolType(tool_config.config.type)`` and raises/KeyErrors without it).

    Returns a bare *list* (not wrapped in a ``tools:`` key) — that wrapping
    happens one layer up, in ``generate_tool_config.py``, which is also
    the only place that writes the real ``tool_config.yaml`` file to disk.

    ``class_name`` defaults to this project's own duck-typed ``BaseTool``
    mirror (``rollout_worker.RealHookRoutedTool``, which routes every call
    through the real ``guard_diagnose_ops`` hook) — swap for whatever the
    real ``tool_config.yaml`` loader expects once veRL is actually vendored.
    """
    return [
        {
            "class_name": class_name,
            "config": {"type": "native", "tool_name": schema["function"]["name"]},
            "tool_schema": schema,
        }
        for schema in build_verl_tool_schemas(include_rag=include_rag)
    ]
