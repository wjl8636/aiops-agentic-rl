"""veRL <-> Claude Agent SDK protocol adapter for the AIOps diagnosis agent.

Per the design doc (§5.1 "GRPO rollout 参数与 KL 约束") and the transcript's
"veRL 怎么对接 Claude Agent SDK 的工具协议" Q&A: veRL owns GRPO rollout
sampling, but it speaks its own multi-turn tool-calling protocol, not the
Claude Agent SDK's ``ToolUseBlock``/``ToolResultBlock``/hook protocol. This
package is the adapter layer that sits between the two, built in two steps
(risk reduction, per the transcript):

1. ``mock_tools.py`` — canned, deterministic tool observations (no real
   docker/k8s/API calls) driving the SAME real hook/whitelist logic
   (``AIops-agent/agent/core/hooks.py::guard_diagnose_ops`` +
   ``agent/core/remediation.py::classify_command``) that production uses.
   This is the path actually run and tested in this repo right now.
2. ``rollout_worker.py`` — the real-environment code path: an SSH
   reverse-tunnel ``bash_executor`` (GPU training box -> the Mac that
   hosts the docker-compose fault environment), Read/Grep/Glob mapped
   onto remote ``cat``/``grep -rE``/``find`` commands through the same
   seam, and a ``BaseTool``-shaped wrapper matching veRL's real
   create/execute/release calling convention (verified against the
   vendored source on the GPU box, and exercised live through the tunnel
   2026-09-07 — see that module's docstring; a full veRL-driven rollout
   is still pending).

Supporting modules: ``tool_schema_bridge.py`` (SDK tool surface -> veRL
tool schema), ``observation_packer.py`` (tool result -> ``Step.observation``
text), ``namespace_isolation.py`` (per-worker namespace/lock isolation for
concurrent GRPO rollout workers).

Every module here is designed to produce/consume ``reward.trajectory.Step``/
``Trajectory`` with zero adaptation glue — see ``reward/trajectory.py`` for
that contract.
"""
from __future__ import annotations
