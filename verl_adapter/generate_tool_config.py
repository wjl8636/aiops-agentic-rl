"""Generates the real ``verl_adapter/tool_config.yaml`` file that
``grpo/verl_config/aiops_grpo.yaml``'s
``actor_rollout_ref.rollout.multi_turn.tool_config_path`` points at.

## Why this is a generator, not a hand-written YAML file
``tool_schema_bridge.py::build_tool_config_entries()`` is the single source
of truth for "what tools does the diagnose agent expose, and what does each
one's veRL-facing config/schema look like" — it already has its own
docstrings tracking what is [VERIFIED] against real veRL source vs a
best-effort mirror of the Claude Agent SDK's builtin tool shapes (see that
module's module docstring). Hand-writing a static ``tool_config.yaml`` next
to it would create a second copy of that information that silently drifts
the moment the tool surface changes (e.g. a new MCP tool gets mounted, or a
schema's ``description``/``parameters`` gets edited). Instead, this script
is the *only* place that turns that Python source of truth into the on-disk
YAML veRL actually loads — regenerate the file by re-running it, never edit
``tool_config.yaml`` by hand:

    python3 -m verl_adapter.generate_tool_config

## [VERIFIED 2026-09-07] top-level shape required by the real veRL loader
Confirmed by reading ``verl/tools/tool_registry.py::initialize_tools_from_config()``
on the GPU training box's vendored veRL checkout (see
``tool_schema_bridge.py``'s module docstring for the full account): the
loader does ``for tool_config in tools_config.tools:`` — i.e. the YAML
document's top-level key MUST be ``tools:``, wrapping a list of entries.
``build_tool_config_entries()`` itself still returns a bare list (that
function is also unit-tested standalone as a list); this script is what
wraps that list under ``{"tools": [...]}`` before writing it to disk.

## ``include_rag`` choice: True, matching the real diagnose agent's default
``AIops-agent/agent/config.py::RAG_ENABLED = os.getenv("AIOPS_RAG_ENABLED",
"1") == "1"`` — i.e. the real diagnose agent mounts the
``search_past_incidents`` RAG tool by default (only ``AIOPS_RAG_ENABLED=0``
turns it off). This generator mirrors that default (``include_rag=True``)
so the generated tool surface matches what the real agent actually exposes
out of the box. If the real deployment sets ``AIOPS_RAG_ENABLED=0``,
re-run this script with ``--no-rag`` to regenerate a matching 4-tool
(Bash/Read/Grep/Glob only) ``tool_config.yaml``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from verl_adapter.tool_schema_bridge import build_tool_config_entries

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_PATH = HERE / "tool_config.yaml"


def build_tool_config_document(include_rag: bool = True) -> dict[str, Any]:
    """The full ``{"tools": [...]}`` document veRL's real YAML loader
    expects — see this module's docstring for why the ``tools:`` wrapper key
    is required and where ``[...]`` itself comes from."""
    return {"tools": build_tool_config_entries(include_rag=include_rag)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 verl_adapter/tool_config.yaml（veRL 的 multi_turn.tool_config_path 指向的真实文件）"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--no-rag",
        dest="include_rag",
        action="store_false",
        default=True,
        help="不含 search_past_incidents RAG 工具（对应真实部署里 AIOPS_RAG_ENABLED=0 的情形）。默认包含，"
        "因为 AIops-agent/agent/config.py::RAG_ENABLED 默认就是开的。",
    )
    args = parser.parse_args()

    document = build_tool_config_document(include_rag=args.include_rag)

    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(document, f, allow_unicode=True, sort_keys=False)

    tool_names = [entry["config"]["tool_name"] for entry in document["tools"]]
    print(f"[generate_tool_config] {len(tool_names)} 条工具 -> {args.out}")
    print(f"[generate_tool_config] 工具: {tool_names}")


if __name__ == "__main__":
    main()
