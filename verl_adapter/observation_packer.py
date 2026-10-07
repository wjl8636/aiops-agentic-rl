"""Converts a raw tool execution result into the exact text convention
``reward.trajectory.Step.observation`` already follows in this repo.

That convention was NOT invented here — it is read off two existing,
already-shipped sources and matched exactly:

1. ``data/cold_start/collect_trajectories.py::_stringify_tool_result`` — the
   function that flattens a real Claude Agent SDK ``ToolResultBlock.content``
   (which is either a plain ``str``, or a ``list[dict]`` of MCP-style
   content blocks such as ``{"type": "text", "text": ...}``) into the
   string that becomes ``Step.observation``. ``stringify_tool_result_content``
   below is a byte-for-byte port of that function's logic (duplicated, not
   imported, for the same "depend on public contracts not private helpers"
   reason as ``_repo_paths.py``).
2. ``data/cold_start/trajectories/mock/*.json`` — every ``observation``
   field in those fixtures is a **plain narrative/output string**, never a
   ``"stdout=...\\nstderr=...\\nexit_code=..."``-style wrapper, and a
   **denied** Bash call's observation is the hook's own denial-reason text
   verbatim (see e.g. ``mock_dep_clean.json``'s last step: the command was
   never executed, so the "observation" is
   ``"PreToolUse hook 拒绝执行：低风险操作 restart_instance 命中..."``, not
   empty output). ``pack_hook_denied_result`` / ``pack_bash_exec_result``
   below match that shape.
"""
from __future__ import annotations

import json
from typing import Any


def stringify_tool_result_content(content: Any) -> str:
    """Flatten a ``ToolResultBlock.content``-shaped value into plain text.

    Exact port of ``collect_trajectories.py::_stringify_tool_result``:
    - ``str`` -> returned as-is.
    - ``list[dict]`` (MCP-style content blocks, e.g. what
      ``rag_tool.py::search_past_incidents`` returns as
      ``{"content": [{"type": "text", "text": ...}]}``) -> each ``{"type":
      "text", "text": ...}`` block's text is taken; any other block shape
      is JSON-dumped verbatim (defensive: we've never seen a non-text
      block in this project, but silently dropping data would be worse).
      Joined with ``"\\n"``.
    - anything else (dict, None, ...) -> JSON-dumped, or ``""`` for ``None``.
    """
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


def pack_mcp_result(mcp_result: dict[str, Any] | None) -> str:
    """Pack an MCP tool's raw return dict (e.g.
    ``{"content": [{"type": "text", "text": ...}], "is_error": bool}`` —
    the exact shape ``rag_tool.py::search_past_incidents`` returns) into a
    ``Step.observation`` string. Delegates straight to
    ``stringify_tool_result_content`` on the ``content`` list, since that is
    the same shape a Claude Agent SDK ``ToolResultBlock.content`` takes for
    an MCP tool call.
    """
    return stringify_tool_result_content((mcp_result or {}).get("content", []))


def pack_bash_exec_result(stdout: str, stderr: str = "", exit_code: int = 0) -> str:
    """Pack a raw ``(stdout, stderr, exit_code)`` execution result — the
    shape a subprocess-style Bash executor (real or mocked) produces —
    into a ``Step.observation`` string.

    Convention (matches every non-denied Bash step in the mock fixtures):
    plain output text, stdout first then stderr if both are non-empty, with
    NO ``"stdout=...\\n"``-style label wrapper. A failing command
    (``exit_code != 0``) still surfaces its stderr/stdout rather than being
    swallowed — the model (and the reward module's evidence-traceability
    check) needs to see failures, not silence. Only appends an explicit
    ``(exit_code=...)`` marker when there is genuinely nothing else to show,
    or when the command failed, so a passing command's plain-text output
    stays undecorated (matching the fixtures exactly).

    NOTE: a call denied by the PreToolUse hook must NOT go through this
    function — it was never executed, so it has no stdout/stderr/exit_code.
    Use ``pack_hook_denied_result`` for that branch instead.
    """
    parts = [p for p in (stdout.rstrip("\n") if stdout else "", stderr.rstrip("\n") if stderr else "") if p]
    if not parts:
        return f"(no output; exit_code={exit_code})"
    text = "\n".join(parts)
    if exit_code != 0:
        text += f"\n(exit_code={exit_code})"
    return text


def pack_hook_denied_result(reason: str) -> str:
    """Pack a PreToolUse hook **deny** verdict into a ``Step.observation``
    string: the hook's own denial-reason text, verbatim. This matches how
    the real Claude Agent SDK behaves when a ``PreToolUse`` hook denies a
    call (the command is never executed; the ``ToolResultBlock`` the model
    sees carries the denial reason as its content) and how every denied
    step in the mock fixtures is written (see module docstring).
    """
    return reason
