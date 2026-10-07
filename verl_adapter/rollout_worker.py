"""Step 2 of the transcript's two-step risk-reduction plan (§十): the
real-environment rollout code path.

The counterpart of ``mock_tools.py`` (the canned path this repo actually
runs and tests): everything here is real, structured, importable Python
with NO veRL / torch / claude_agent_sdk import, and the parts that need a
live veRL loop or a live fault environment are isolated behind small,
explicit, duck-typed seams (``ToolCallSource``, ``bash_executor``,
``mcp_executor``, ``post_action_health_reader``) so they can be swapped
without touching the rest of this module.

## Execution topology (why there is an SSH tunnel executor at all)
GRPO/veRL rollouts run on a rented cloud GPU box that is itself a nested
container without ``CAP_NET_ADMIN`` — it cannot run the docker-compose OTel
Demo fault environment (custom networks). That environment lives on the
Mac. A persistent reverse tunnel (the Mac runs ``ssh -N -R 2222:localhost:22
aiops-gpu`` in an auto-restarting loop) makes the GPU box's
``localhost:2222`` forward to the Mac's sshd, so on the GPU box
``ssh -p 2222 <mac_user>@localhost "<cmd>"`` executes on the Mac.
``ssh_tunnel_bash_executor`` below is the ``bash_executor`` seam
implementation for exactly that setup (endpoint overridable via
``AIOPS_TUNNEL_PORT``/``AIOPS_TUNNEL_USER``/``AIOPS_TUNNEL_HOST``).
NOTE: the tunnel's listener lives on the GPU box — on the Mac itself,
``localhost:2222`` is colima's ssh mux, NOT this tunnel (a real port
collision found while verifying; "testing the tunnel" from the Mac by
ssh'ing to localhost:2222 would talk to the wrong daemon).

## [VERIFIED 2026-09-07] veRL's real BaseTool calling convention
Read directly off the vendored source on the GPU training box (host alias
``aiops-gpu``, checkout at ``/root/autodl-tmp/code/verl/``), not from the
old WebFetch-of-GitHub-main notes:
  - ``verl/tools/base_tool.py::BaseTool.create`` is
    ``async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]``
    — it returns a 2-tuple; when ``instance_id`` is None it generates a
    fresh ``uuid4()``; the second element is an (empty) ``ToolResponse``.
  - ``verl/experimental/agent_loop/tool_agent_loop.py`` (the BaseTool
    dispatch branch, ≈lines 464-488) calls::

        instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
        tool_execution_response, tool_reward, res = await tool.execute(
            instance_id, tool_args, agent_data=agent_data)
        ...
        await tool.release(instance_id)

    i.e. ``create()`` is invoked with the ``create_kwargs`` KEYWORD (never a
    positional ``instance_id``), its return is unpacked as a 2-tuple, and
    ``execute()`` receives ``agent_data`` as a keyword on every call;
    ``release()`` is called positionally. ``RealHookRoutedTool`` below now
    matches this exactly (an earlier draft returned a bare ``instance_id``
    str and expected the caller to pass one in — fixed 2026-09-07).

## [VERIFIED 2026-09-07] the SSH tunnel path itself
From the GPU box (via ``ssh aiops-gpu``), a ``subprocess.run`` argv-list
call of ``["ssh", "-p", "2222", "--", "<mac_user>@localhost",
"cd <dir> && <cmd>"]`` was run against the live tunnel: it executed on the
Mac (Darwin / correct ``whoami``), a command mixing single and double
quotes arrived byte-exact, the ``cd``-prefix cwd handling worked, and
``docker ps`` / Prometheus ``:9090`` on the Mac answered (OTel Demo
containers listed). The full new code path was then exercised ON the GPU
box against the live tunnel: ``execute_tool_call`` for Bash (hook-denied
command never ran; read-only ``docker ps`` listed OTel containers) and for
Read/Grep/Glob (real files under the Mac's repo checkout were read,
grepped, globbed), plus the whole ``RealHookRoutedTool`` lifecycle
(``create(create_kwargs=...)`` -> ``execute(..., agent_data=...)`` ->
``release``), and ``verl_adapter/tests/test_rollout_worker.py`` in full
with ``AIOPS_TUNNEL_TEST=1`` (32/32 passed, including the three live
checks). That live run also caught and fixed a real bug: a cwd-less Glob
(``find . -path ...``) executes from the remote login default dir — the
Mac's entire ``$HOME`` — and times out; ``cwd`` is now threaded from
``execute_tool_call``/``RealHookRoutedTool``/``run_real_rollout`` down to
the executor (see ``_run_executor``), so production wiring must pass the
agent workspace root as ``cwd``. Mocked unit tests plus the opt-in live
checks live in ``verl_adapter/tests/test_rollout_worker.py``
(``AIOPS_TUNNEL_TEST=1``, meant to run on the GPU box where the tunnel's
listener lives).

## Read/Grep/Glob are now implemented (were NotImplementedError before)
They map to remote commands and run through the SAME ``bash_executor`` seam
as Bash — so wiring ``ssh_tunnel_bash_executor`` as the executor on the GPU
box routes all four tools through the tunnel at once:
  - Read ``{"file_path"}``        -> ``cat <file_path>``
  - Grep ``{"pattern", "path"?}`` -> ``grep -rE -- <pattern> <path|.>``
  - Glob ``{"pattern"}``          -> ``find . -path './<pattern>' -print``
Field names are copied from ``data/cold_start/to_llamafactory_format.py``'s
``_TOOLS_SCHEMA`` (the Claude Agent SDK builtin shapes this project's
cold-start data was built against), not invented here. The real hook only
ever gates ``Bash`` calls (``hooks.py::_extract_command`` returns ``""``
for any other ``tool_name``, so ``guard_diagnose_ops`` returns ``{}``),
which is why these three always come back ``passthrough`` — consistent with
``execute_tool_call`` only special-casing deny for Bash.

## What this module reuses from ``data/cold_start/collect_trajectories.py``
That module already solved "how do you reconstruct a step-by-step
trajectory from a live Claude Agent SDK rollout": pair each
``AssistantMessage``'s ``ToolUseBlock`` with the matching (same
``tool_use_id``) ``UserMessage``'s ``ToolResultBlock``, and wrap the real
``guard_diagnose_ops`` hook to record its decision as a side effect without
changing its allow/deny behavior (``_make_recording_hook``). This module
reuses that exact PATTERN — ``StepAccumulator`` below pairs
``SampledToolCall``/``ToolResult`` by ``tool_use_id`` the same way, and
``route_through_real_hook`` peeks at the real hook's decision the same way
``_make_recording_hook`` does — expressed with this module's own dataclasses
instead of ``claude_agent_sdk``'s types, because that package cannot be
imported in this environment/session.

## What is verified vs. still assumed about veRL's API surface
  1. [VERIFIED 2026-09-07] ``BaseTool``'s create/execute/release/calc_reward
     signatures and the ``ToolAgentLoop`` dispatch shape — read off the
     vendored source at ``/root/autodl-tmp/code/verl/`` on ``aiops-gpu``
     (see the "[VERIFIED 2026-09-07] veRL's real BaseTool calling
     convention" section above for the exact code).
  2. ``SampledToolCall`` below models "one sampled tool_call" with this
     project's own dataclass (``tool_use_id``/``tool_name``/``tool_input``)
     rather than importing veRL's ``OpenAIFunctionToolCall`` — same shape,
     zero import dependency on the (unvendored-here) package.
  3. ``ToolCallSource`` is a stand-in for "whatever hands us the next
     sampled tool_call from veRL's rollout" (a real integration would feed
     from ``ToolAgentLoop``'s generation step); here it is just a
     duck-typed async callable, so this module has NO veRL import at all.
  4. ``RealHookRoutedTool`` mirrors ``verl.tools.base_tool.BaseTool``'s
     contract signature-for-signature (verified, see 1) but does NOT
     literally subclass it (veRL is not importable in this dev
     environment) — once rollout actually runs on ``aiops-gpu``, either
     keep this duck-typed class as the ``class_name`` in
     ``tool_schema_bridge.build_tool_config_entries()`` or make it inherit
     from the real ``BaseTool`` there.
  5. STILL NOT VERIFIED: no end-to-end veRL rollout has been driven through
     ``RealHookRoutedTool`` on the GPU box yet (no ``ToolAgentLoop`` run
     against it, no ``tool_config.yaml`` load in a real training job) — the
     class is contract-aligned with the vendored source, but the real loop
     has not executed it even once. The exact ``ToolParser`` format string
     for the SFT'd checkpoint is likewise still unknown (see
     ``tool_schema_bridge.py``'s flagged assumptions).

## The one non-negotiable invariant (per task's ground-truth facts)
Every sampled tool_call — allow, deny, or passthrough — MUST be routed
through the real ``agent.core.hooks.guard_diagnose_ops`` before any
execution happens. ``route_through_real_hook`` / ``execute_tool_call`` below
never skip or re-derive an approximation of that decision; they call the
real function every time.
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from reward.trajectory import Step, Trajectory
from verl_adapter import _repo_paths, observation_packer
from verl_adapter.tool_schema_bridge import RAG_TOOL_NAME


# --------------------------------------------------------------------------
# Message shapes (this module's own stand-ins for the SDK's ToolUseBlock /
# ToolResultBlock — see module docstring assumption #2).
# --------------------------------------------------------------------------

@dataclass
class SampledToolCall:
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class ToolResult:
    tool_use_id: str
    content: Any  # str, or list[{"type": "text", "text": ...}] (MCP shape)
    is_error: bool = False


# --------------------------------------------------------------------------
# Real (but pluggable) executors. These are genuine, callable implementations
# — no live tunnel / docker / k8s environment is required merely to import
# or unit-test them (tests mock ``subprocess.run``; the live checks are
# opt-in, see tests/test_rollout_worker.py's AIOPS_TUNNEL_TEST gate).
# --------------------------------------------------------------------------

def default_subprocess_bash_executor(
    command: str, cwd: Optional[str] = None, timeout_s: float = 30.0
) -> tuple[str, str, int]:
    """Real local subprocess execution of a Bash command — this is exactly
    how the real Claude Agent SDK's own builtin ``Bash`` tool behaves (runs
    the command locally via a shell); there is no separate "AIops-agent
    execution backend" beyond that. Only called for a tool_call whose hook
    verdict was NOT "deny" (see ``execute_tool_call``).

    Kept as the DEFAULT executor so this module stays hermetic (unit tests
    and local runs never need a tunnel); the GPU rollout wiring passes
    ``bash_executor=ssh_tunnel_bash_executor`` instead.
    """
    proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
    return proc.stdout, proc.stderr, proc.returncode


# --------------------------------------------------------------------------
# SSH reverse-tunnel executor: run commands on the Mac (where the real
# docker-compose OTel Demo fault environment lives) from the GPU training
# box (where veRL rollouts run) — see the module docstring's "Execution
# topology" section.
# --------------------------------------------------------------------------

#: Env var names for the tunnel endpoint. Port/user/host are deployment
#: facts that can change without a code change, hence env-overridable
#: (the constants below are placeholders, NOT the real deployed values —
#: ``AIOPS_TUNNEL_USER`` etc. MUST be exported before running against the
#: actual Mac's persistent ``ssh -N -R 2222:localhost:22 aiops-gpu`` loop).
AIOPS_TUNNEL_PORT_ENV = "AIOPS_TUNNEL_PORT"
AIOPS_TUNNEL_USER_ENV = "AIOPS_TUNNEL_USER"
AIOPS_TUNNEL_HOST_ENV = "AIOPS_TUNNEL_HOST"
_TUNNEL_DEFAULT_PORT = "2222"
_TUNNEL_DEFAULT_USER = "changeme"
_TUNNEL_DEFAULT_HOST = "localhost"

#: Env var names for the Mac-side (environment-host) endpoints the RAG bridge
#: needs. Like the tunnel vars above: deployment facts, env-overridable, read
#: per-call (not cached at import) so tests can monkeypatch. The constants
#: below are placeholders, NOT the real deployed values — ``AIOPS_MAC_PYTHON``
#: / ``AIOPS_MAC_REPO_ROOT`` MUST be exported before running against the
#: actual deployment: the nested ``AIops-agent/.venv`` (verified 2026-09-08 to
#: have claude_agent_sdk + pymilvus + sentence_transformers) and this repo's
#: root on the Mac.
AIOPS_MAC_PYTHON_ENV = "AIOPS_MAC_PYTHON"
AIOPS_MAC_REPO_ROOT_ENV = "AIOPS_MAC_REPO_ROOT"
AIOPS_MAC_BRIDGE_PATH_ENV = "AIOPS_MAC_BRIDGE_PATH"
#: Port of the Mac-resident RAG service (``verl_adapter/rag_service.py``，
#: the PRIMARY tier of ``ssh_tunnel_mcp_executor``; same default on both
#: sides — the service reads it too).
AIOPS_RAG_SERVICE_PORT_ENV = "AIOPS_RAG_SERVICE_PORT"
_RAG_SERVICE_DEFAULT_PORT = "18765"
#: Working directory tool commands run in on the Mac (the agent workspace root
#: where ``deploys.log`` and ``workspace/<svc>/`` live — the ROOT checkout of
#: AIops-agent, NOT this repo's nested submodule copy: the live docker env and
#: the cloned fix-target repos are mounted there). Read by
#: ``RealHookRoutedTool.__init__`` when the tool_config entry carries no cwd.
AIOPS_MAC_AGENT_DIR_ENV = "AIOPS_MAC_AGENT_DIR"
_MAC_DEFAULT_PYTHON = "/path/to/aiops-agentic-rl/AIops-agent/.venv/bin/python"
_MAC_DEFAULT_REPO_ROOT = "/path/to/aiops-agentic-rl"
_MAC_DEFAULT_BRIDGE_PATH = "verl_adapter/rag_bridge.py"


def tunnel_ssh_argv() -> list[str]:
    """Leading argv for one tunnel ssh invocation — everything up to (but
    not including) the remote command itself, e.g.
    ``["ssh", "-p", "2222", "--", "<mac_user>@localhost"]``.

    Reads ``AIOPS_TUNNEL_PORT`` / ``AIOPS_TUNNEL_USER`` /
    ``AIOPS_TUNNEL_HOST`` on every call (not cached at import time) so
    tests can monkeypatch the environment and a mid-process change is
    honored. ``--`` marks end-of-options so a remote command that happens
    to start with ``-`` can never be parsed as an ssh option.
    """
    port = os.environ.get(AIOPS_TUNNEL_PORT_ENV) or _TUNNEL_DEFAULT_PORT
    user = os.environ.get(AIOPS_TUNNEL_USER_ENV) or _TUNNEL_DEFAULT_USER
    host = os.environ.get(AIOPS_TUNNEL_HOST_ENV) or _TUNNEL_DEFAULT_HOST
    return ["ssh", "-p", port, "--", f"{user}@{host}"]


def ssh_tunnel_bash_executor(
    command: str, cwd: Optional[str] = None, timeout_s: float = 30.0
) -> tuple[str, str, int]:
    """Execute ``command`` on the environment host (the Mac) through the
    reverse SSH tunnel — same signature and ``(stdout, stderr, exit_code)``
    return shape as ``default_subprocess_bash_executor``, so it drops into
    the same ``bash_executor`` seam (and thereby also serves the
    Read/Grep/Glob remote-command mapping in ``execute_tool_call``).

    How quoting stays correct (the subtle part — this is what the mixed-
    quote tests in tests/test_rollout_worker.py pin down):
      1. ``subprocess.run`` is called with an argv LIST and no
         ``shell=True``, so NO local shell ever parses the command — it
         reaches ssh as one byte-exact argv element.
      2. ssh passes that string to the remote login shell verbatim. The
         command therefore gets parsed by exactly ONE shell — the remote
         one — which is precisely what a full shell-command string (what
         the model's Bash tool emits) expects. Mixed single/double quotes
         survive byte-for-byte; verified live through the tunnel on
         2026-09-07 (see module docstring).
      3. ``cwd`` is prepended as ``cd <cwd> && <command>`` (never assumed
         to be the remote default cwd), with ``cwd`` shlex-quoted so paths
         containing spaces or quotes survive the remote shell intact.

    A dead tunnel surfaces the same way any failing remote command does:
    ssh exits non-zero (255) with the reason on stderr, and
    ``execute_tool_call`` packs that into the step observation rather than
    raising — the model sees the failure and can react. Only a genuine
    timeout raises (``subprocess.TimeoutExpired``), same as the local
    default executor.
    """
    remote_command = f"cd {shlex.quote(cwd)} && {command}" if cwd else command
    argv = [*tunnel_ssh_argv(), remote_command]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    return proc.stdout, proc.stderr, proc.returncode


def _mcp_error_observation(text: str) -> dict[str, Any]:
    """The ``is_error``-marked MCP result every failure path of
    ``ssh_tunnel_mcp_executor`` returns — NEVER an exception (see that
    function's docstring for why)."""
    return {"is_error": True, "content": [{"type": "text", "text": text}]}


async def ssh_tunnel_mcp_executor(
    tool_name: str,
    tool_input: dict[str, Any],
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Execute an MCP tool call on the environment host (the Mac) through the
    reverse SSH tunnel — the ``McpExecutor`` counterpart of
    ``ssh_tunnel_bash_executor``.

    Only the RAG tool (``mcp__aiops_rag__search_past_incidents`` — the single
    MCP tool this project registers, see ``tool_schema_bridge.RAG_TOOL_NAME``)
    is wired, in two tiers:

    1. **Primary**: ``curl`` the Mac-resident ``verl_adapter/rag_service.py``
       (loaded BGE + Milvus collection once at startup, ~0.1-0.5s per query —
       see that module's docstring for the deployment/protocol details).
    2. **Fallback**: if curl fails at the CONNECTION level (service not
       running / crashed / port wrong — curl exits non-zero), fall back to a
       one-shot execution of ``verl_adapter/rag_bridge.py`` (~10s per call:
       the full cold start is paid every time, see its docstring). This keeps
       a 20h unattended GRPO run CORRECT if the service dies mid-training —
       speed degrades, rollouts don't start failing. [JUDGMENT: fallback on
       connection-level failure only; an app-level ``is_error`` body from the
       service (Milvus down etc.) is returned as-is WITHOUT fallback — the
       bridge hits the same Milvus and would fail the same way.]

    Error semantics ([JUDGMENT], same rationale as the Bash executor): any
    failure — dead tunnel (ssh exit 255), both tiers failing, non-JSON
    stdout, timeout — returns ``{"is_error": true, "content": [...]}`` and
    **never raises**: a failed tool call must become an error observation the
    MODEL sees and can react to, not an exception that kills the whole
    rollout. Timeouts ``kill()`` the ssh process.

    Async plumbing: ``asyncio.create_subprocess_exec`` (never a blocking
    ``subprocess.run``) because this executor is awaited inside veRL's
    async tool loop — a blocking call would stall every concurrent rollout.
    """
    if tool_name != RAG_TOOL_NAME:
        return _mcp_error_observation(
            f"ssh_tunnel_mcp_executor: no bridge wired for {tool_name!r} (only {RAG_TOOL_NAME})"
        )

    args_json = json.dumps(tool_input or {}, ensure_ascii=False)
    port = os.environ.get(AIOPS_RAG_SERVICE_PORT_ENV) or _RAG_SERVICE_DEFAULT_PORT

    # One remote command string, one remote shell parse — same quoting
    # discipline as ssh_tunnel_bash_executor: argv list (no local shell) +
    # shlex.quote per argument so the JSON (Chinese text, quotes, spaces)
    # survives byte-exact to the Mac-side argv.
    curl_command = (
        f"curl -s -m {int(timeout_s)} -X POST http://localhost:{port}/search"
        f" -H {shlex.quote('Content-Type: application/json')} -d {shlex.quote(args_json)}"
    )
    result = await _ssh_run_remote_json(curl_command, timeout_s=timeout_s, what="rag_service curl")
    if result is not None:
        return result

    # Connection-level failure -> one-shot bridge fallback (see docstring).
    python = os.environ.get(AIOPS_MAC_PYTHON_ENV) or _MAC_DEFAULT_PYTHON
    repo_root = os.environ.get(AIOPS_MAC_REPO_ROOT_ENV) or _MAC_DEFAULT_REPO_ROOT
    bridge_rel = os.environ.get(AIOPS_MAC_BRIDGE_PATH_ENV) or _MAC_DEFAULT_BRIDGE_PATH
    bridge_abs = os.path.join(repo_root, bridge_rel)
    bridge_command = f"{shlex.quote(python)} {shlex.quote(bridge_abs)} {shlex.quote(args_json)}"
    result = await _ssh_run_remote_json(bridge_command, timeout_s=timeout_s, what="rag_bridge fallback")
    if result is not None:
        return result
    return _mcp_error_observation(
        "rag_service unreachable AND rag_bridge fallback failed "
        f"(see stderr details in the two preceding observations' logs; tunnel port/host: "
        f"{os.environ.get(AIOPS_TUNNEL_PORT_ENV) or _TUNNEL_DEFAULT_PORT})"
    )


async def _ssh_run_remote_json(
    remote_command: str, timeout_s: float, what: str
) -> Optional[dict[str, Any]]:
    """Run ``remote_command`` through the tunnel, expect a single-line JSON
    dict on stdout (the MCP result shape). Returns ``None`` for CONNECTION-
    level failure (non-zero exit / timeout) so the caller can fall back;
    returns the parsed dict on success (including app-level ``is_error``
    bodies, which are NOT connection failures); returns an error-shaped dict
    (not None) when stdout parses but has the wrong shape — that is a
    protocol bug on the Mac side, not something a fallback would fix."""
    argv = [*tunnel_ssh_argv(), remote_command]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        print(f"[ssh_tunnel_mcp_executor] {what}: timed out after {timeout_s}s", flush=True)
        return None
    stdout, stderr = out_b.decode("utf-8", errors="replace").strip(), err_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        print(f"[ssh_tunnel_mcp_executor] {what}: exited {proc.returncode}: {stderr[:200]}", flush=True)
        return None
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return _mcp_error_observation(f"{what}: stdout is not JSON: {stdout[:200]!r}")
    if not isinstance(parsed, dict) or "content" not in parsed:
        return _mcp_error_observation(f"{what}: result has unexpected shape: {str(parsed)[:200]!r}")
    return parsed


#: Duck-typed seam: given a shell command string, return (stdout, stderr, exit_code).
#: This one-arg call is the MINIMUM shape every executor must accept; the
#: two real executors above additionally accept the optional ``cwd`` /
#: ``timeout_s`` keywords, and ``_run_executor`` below threads ``cwd``
#: through whenever one is configured (see its docstring for why that
#: matters — a live-tunnel run proved a cwd-less Glob scans the remote
#: login default dir, i.e. the Mac's entire $HOME, and times out).
BashExecutor = Callable[[str], tuple[str, str, int]]


def _run_executor(
    bash_executor: BashExecutor, command: str, cwd: Optional[str] = None
) -> tuple[str, str, int]:
    """Call ``bash_executor`` with ``command``, threading ``cwd`` through
    only when one is actually configured.

    The conditional matters for duck-typing: executors written against the
    one-arg ``BashExecutor`` minimum shape (e.g. test doubles like
    ``lambda cmd: ("", "", 0)``) keep working verbatim as long as no cwd is
    requested; once a cwd IS requested the executor gets it as the
    ``cwd=`` keyword — the shape both real executors
    (``default_subprocess_bash_executor`` / ``ssh_tunnel_bash_executor``)
    define, mapping it onto ``cd <cwd> && <command>`` / ``subprocess.run(cwd=...)``.
    """
    if cwd:
        return bash_executor(command, cwd=cwd)
    return bash_executor(command)

#: Duck-typed seam: given (tool_name, tool_input) for an MCP tool call
#: (e.g. "mcp__aiops_rag__search_past_incidents"), return the raw MCP
#: result dict (``{"content": [...], "is_error": bool}`` shape — see
#: AIops-agent/agent/integrations/rag_tool.py). Awaitable because the real
#: implementation does a Milvus query.
McpExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------------
# The non-negotiable seam: route every sampled tool_call through the REAL
# PreToolUse hook before any execution.
# --------------------------------------------------------------------------

async def route_through_real_hook(tool_call: SampledToolCall) -> dict[str, Any]:
    """Route ``tool_call`` through the REAL ``guard_diagnose_ops`` hook.

    Mirrors ``_make_recording_hook``'s sink shape in
    ``collect_trajectories.py``: the hook's own ``permissionDecision`` is
    authoritative for ``decision`` (never re-derived from
    ``classify_command`` alone — see that function's docstring on why:
    ``guard_diagnose_ops`` has an extra ``_READONLY_DENY``/
    ``_WRITE_FALLBACK_DENY`` fallback layer beyond the whitelist).
    ``action``/``target`` are only meaningful for Bash calls, and are
    filled from ``classify_command`` (consistent with the hook's own
    allow/deny branches, which call ``classify_command`` internally too).

    Returns ``{"decision": "allow"|"deny"|"passthrough", "action": str|None,
    "target": str|None, "reason": str|None}``.
    """
    remediation, guard_diagnose_ops = _repo_paths.load_real_hook_modules()
    input_data = {
        "tool_name": tool_call.tool_name,
        "tool_input": tool_call.tool_input,
        "hook_event_name": "PreToolUse",
    }
    result = await guard_diagnose_ops(input_data, tool_call.tool_use_id, {})
    hook_output = result.get("hookSpecificOutput") or {}
    decision_raw = hook_output.get("permissionDecision")
    decision = decision_raw if decision_raw in ("allow", "deny") else "passthrough"
    action = target = None
    if tool_call.tool_name == "Bash":
        verdict = remediation.classify_command(str(tool_call.tool_input.get("command", "")))
        if verdict.get("decision") in ("allow", "deny"):
            action, target = verdict.get("action"), verdict.get("target")
    return {
        "decision": decision,
        "action": action,
        "target": target,
        "reason": hook_output.get("permissionDecisionReason"),
    }


# --------------------------------------------------------------------------
# Read/Grep/Glob -> remote-command mapping. Field names are copied from
# data/cold_start/to_llamafactory_format.py's _TOOLS_SCHEMA (the Claude
# Agent SDK builtin shapes this project's cold-start data was built
# against): Read takes {"file_path"}, Grep takes {"pattern", "path"?},
# Glob takes {"pattern"}. All three are read-only by construction (cat /
# grep -r / find with shlex-quoted arguments), and the real hook never
# gates them (it only extracts a command for tool_name == "Bash" — see
# hooks.py::_extract_command), so they always execute as "passthrough".
# --------------------------------------------------------------------------

def _read_remote_command(tool_input: dict[str, Any]) -> str:
    """Read -> remote ``cat <file_path>``. The diagnose prompt's intended
    targets are workspace-rooted files like ``deploys.log`` and cloned repo
    sources under ``workspace/<svc>/`` (see
    ``AIops-agent/agent/agents/prompts.py::diagnose_append``); paths are
    interpreted relative to the executor's cwd, exactly like the Bash
    ``cat`` the prompt tells the model to run.
    """
    file_path = str(tool_input.get("file_path", ""))
    return f"cat {shlex.quote(file_path)}"


def _grep_remote_command(tool_input: dict[str, Any]) -> str:
    """Grep -> remote ``grep -rE -- <pattern> <path>``. ``pattern`` is
    treated as an extended regex (grep -E) to match the prompt's own
    ``rg``/``grep`` usage; ``path`` defaults to ``.`` (the executor's cwd —
    the agent workspace root where ``deploys.log`` and ``workspace/`` live)
    because the schema marks it optional. ``--`` guards against patterns
    that start with ``-`` being read as grep options. Plain ``grep`` (not
    ``rg``) because it is universally present on the environment host.
    """
    pattern = str(tool_input.get("pattern", ""))
    path = tool_input.get("path")
    target = shlex.quote(str(path)) if path else "."
    return f"grep -rE -- {shlex.quote(pattern)} {target}"


def _glob_remote_command(tool_input: dict[str, Any]) -> str:
    """Glob -> remote ``find . -path './<pattern>' -print``. ``find``'s own
    matching is used instead of a shell glob so behavior is identical under
    the remote zsh/bash/sh and never depends on shell options; a leading
    ``./`` on the pattern is normalized because ``find`` compares against
    ``./``-prefixed paths. Semantics worth knowing: patterns are relative
    to the executor's cwd, and ``*`` in ``find -path`` also matches ``/``
    (so ``**`` behaves like a recursive glob). A pattern that matches
    nothing simply produces empty stdout with exit code 0 — same as the
    real Glob tool.
    """
    pattern = str(tool_input.get("pattern", ""))
    normalized = pattern[2:] if pattern.startswith("./") else pattern
    return f"find . -path {shlex.quote('./' + normalized)} -print"


_READ_GREP_GLOB_COMMAND_BUILDERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "Read": _read_remote_command,
    "Grep": _grep_remote_command,
    "Glob": _glob_remote_command,
}


async def execute_tool_call(
    tool_call: SampledToolCall,
    bash_executor: BashExecutor = default_subprocess_bash_executor,
    mcp_executor: Optional[McpExecutor] = None,
    cwd: Optional[str] = None,
) -> tuple[ToolResult, dict[str, Any]]:
    """Route ``tool_call`` through the REAL hook, then — only if allowed or
    passthrough — actually execute it. Packs the result via
    ``observation_packer`` so the returned ``ToolResult.content`` already
    matches ``Step.observation``'s convention.

    ``cwd`` is the working directory ALL tool commands run in (Bash and the
    Read/Grep/Glob remote commands alike) — in production the agent
    workspace root on the environment host (where ``deploys.log`` and
    ``workspace/<svc>/`` live). Never assume the executor's default cwd is
    right: with ``ssh_tunnel_bash_executor`` the remote default is the
    login shell's $HOME on the Mac, and a cwd-less Glob
    (``find . -path ...``) from there scans the entire home directory and
    times out — found by actually running it through the live tunnel
    (2026-09-07). ``None`` means no cwd control (local executor: inherit
    the process cwd; tunnel executor: remote login default).

    NEVER executes a Bash call the hook denied: the returned observation
    for a deny is the hook's own reason text (``pack_hook_denied_result``),
    matching how the real SDK behaves when ``PreToolUse`` denies a call —
    the command genuinely never runs.
    """
    verdict = await route_through_real_hook(tool_call)

    if tool_call.tool_name == "Bash" and verdict["decision"] == "deny":
        content = observation_packer.pack_hook_denied_result(verdict["reason"] or "denied by PreToolUse hook")
        return ToolResult(tool_call.tool_use_id, content, is_error=True), verdict

    if tool_call.tool_name == "Bash":
        stdout, stderr, exit_code = _run_executor(
            bash_executor, str(tool_call.tool_input.get("command", "")), cwd=cwd
        )
        content = observation_packer.pack_bash_exec_result(stdout, stderr, exit_code)
        return ToolResult(tool_call.tool_use_id, content, is_error=exit_code != 0), verdict

    if tool_call.tool_name.startswith("mcp__"):
        if mcp_executor is None:
            raise RuntimeError(f"execute_tool_call: no mcp_executor configured for {tool_call.tool_name!r}")
        raw = await mcp_executor(tool_call.tool_name, tool_call.tool_input)
        content = observation_packer.pack_mcp_result(raw)
        return ToolResult(tool_call.tool_use_id, content, is_error=bool((raw or {}).get("is_error"))), verdict

    # Read/Grep/Glob: map the tool_input onto an equivalent remote command
    # (cat / grep -rE / find, see the builders above) and run it through
    # the SAME bash_executor seam as Bash — one seam wires all four tools
    # through the tunnel. Their results are "bash-like output" (stdout /
    # stderr / exit code), so they reuse pack_bash_exec_result directly;
    # the hook never denies these (it only gates Bash), so the verdict is
    # always "passthrough" and execution always proceeds.
    command_builder = _READ_GREP_GLOB_COMMAND_BUILDERS.get(tool_call.tool_name)
    if command_builder is not None:
        stdout, stderr, exit_code = _run_executor(bash_executor, command_builder(tool_call.tool_input), cwd=cwd)
        content = observation_packer.pack_bash_exec_result(stdout, stderr, exit_code)
        return ToolResult(tool_call.tool_use_id, content, is_error=exit_code != 0), verdict

    raise NotImplementedError(
        f"execute_tool_call: no real executor wired for tool_name={tool_call.tool_name!r} "
        "(supported: Bash, mcp__*, Read, Grep, Glob)."
    )


# --------------------------------------------------------------------------
# Step accumulation -> Trajectory, reusing the tool_use_id pairing PATTERN
# from collect_trajectories.py (expressed with this module's own types).
# --------------------------------------------------------------------------

class StepAccumulator:
    """Mirrors ``collect_trajectories.py::_run_diagnose_capturing_steps``'s
    ``pending_calls``/``steps`` pairing-by-``tool_use_id`` pattern, in terms
    of this module's ``SampledToolCall``/``ToolResult``/hook-verdict shapes.
    """

    def __init__(self) -> None:
        self._raw_steps: list[dict[str, Any]] = []

    def record(self, tool_call: SampledToolCall, result: ToolResult, hook_verdict: dict[str, Any]) -> None:
        self._raw_steps.append(
            {
                "tool_name": tool_call.tool_name,
                "tool_input": tool_call.tool_input,
                "hook_decision": hook_verdict["decision"],
                "hook_action": hook_verdict["action"],
                "hook_target": hook_verdict["target"],
                "observation": observation_packer.stringify_tool_result_content(result.content),
            }
        )

    def build_trajectory(
        self,
        alert: dict[str, Any],
        final_diagnosis: dict[str, Any],
        ground_truth: dict[str, Any],
        route: str,
        post_action_health: Optional[dict[str, Any]] = None,
    ) -> Trajectory:
        """Tag signals (via ``data/cold_start/signals.py``, the single
        shared home for that rule-based logic) and assemble the final
        ``reward.trajectory.Trajectory`` — the contract this whole module
        exists to satisfy.
        """
        signals_mod = _repo_paths.load_signals_module()
        signals_mod.annotate_steps(self._raw_steps)
        steps = [
            Step(
                tool_name=s["tool_name"],
                tool_input=s["tool_input"],
                hook_decision=s["hook_decision"],
                observation=s["observation"],
                hook_action=s["hook_action"],
                hook_target=s["hook_target"],
                signal_types_covered=s["signal_types_covered"],
                is_repeat_no_new_info=s["is_repeat_no_new_info"],
            )
            for s in self._raw_steps
        ]
        return Trajectory(
            alert=alert,
            steps=steps,
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth,
            route=route,
            post_action_health=post_action_health,
        )


#: Duck-typed seam (assumption #3): given the running message history so
#: far, return the next SampledToolCall the model sampled, or None when the
#: model produced a final answer instead of another tool call. A real
#: integration wires this to veRL's ToolAgentLoop generation step.
ToolCallSource = Callable[[list[dict[str, Any]]], Awaitable[Optional[SampledToolCall]]]


async def run_real_rollout(
    alert: dict[str, Any],
    next_tool_call: ToolCallSource,
    final_diagnosis_reader: Callable[[], Awaitable[dict[str, Any]]],
    ground_truth: dict[str, Any],
    route: str,
    post_action_health_reader: Optional[Callable[[], Awaitable[Optional[dict[str, Any]]]]] = None,
    bash_executor: BashExecutor = default_subprocess_bash_executor,
    mcp_executor: Optional[McpExecutor] = None,
    cwd: Optional[str] = None,
    max_turns: int = 12,
) -> Trajectory:
    """End-to-end real-mode rollout orchestrator.

    *** NOT YET DRIVEN END-TO-END *** (needs a real veRL loop feeding
    ``next_tool_call`` and a real health probe for
    ``post_action_health_reader``; the ``bash_executor`` side is ready —
    pass ``ssh_tunnel_bash_executor`` on the GPU box, or the local default
    for a same-machine environment). Structure mirrors
    ``collect_trajectories.py``'s real-mode function, generalized away from
    ``claude_agent_sdk``'s exact message types.

    ``max_turns=12`` matches doc §5.1's GRPO rollout cap ("最长交互 12 轮，
    超过 12 轮还没收敛的截断给负奖") — penalizing a truncated rollout is the
    caller's (``grpo/reward_router.py``'s) job, not this function's; we
    simply stop accumulating steps at the cap and hand back whatever
    trajectory exists so far.

    ``post_action_health_reader`` MUST read a real signal (health endpoint,
    ``kubectl get pod`` status, a Prometheus metric) — see
    ``Trajectory.post_action_health``'s docstring: this can never be
    derived from the model's own self-reported text.
    """
    accumulator = StepAccumulator()
    history: list[dict[str, Any]] = []
    for _ in range(max_turns):
        tool_call = await next_tool_call(history)
        if tool_call is None:
            break
        result, verdict = await execute_tool_call(
            tool_call, bash_executor=bash_executor, mcp_executor=mcp_executor, cwd=cwd
        )
        accumulator.record(tool_call, result, verdict)
        history.append({"tool_call": tool_call, "result": result})

    final_diagnosis = await final_diagnosis_reader()
    post_action_health = await post_action_health_reader() if post_action_health_reader is not None else None

    return accumulator.build_trajectory(alert, final_diagnosis, ground_truth, route, post_action_health)


# --------------------------------------------------------------------------
# BaseTool-shaped wrapper (assumption #4) — the concrete `class_name` target
# tool_schema_bridge.build_tool_config_entries() defaults to.
# --------------------------------------------------------------------------


class ToolResponseShim:
    """Duck-typed stand-in for ``verl.tools.schemas.ToolResponse``.

    [2026-09-09 实测修复, 云端 aiops-gpu 真实 GRPO rollout 复现] Earlier drafts of
    ``RealHookRoutedTool.create``/``execute`` returned plain ``{"text": ...}``
    dicts here, reasoning (wrongly) that the real pydantic ``ToolResponse``
    "is not importable" in this no-veRL sandbox and that the loop only reads
    it by dict subscript. Once real multi-turn tool calls actually reached
    execution for the first time (after fixing the two upstream veRL config
    bugs — ``rollout.agent.default_agent_loop`` defaulting to
    ``single_turn_agent``, and ``multi_turn.format`` defaulting to
    ``hermes`` instead of ``qwen3_coder``), the real
    ``ToolAgentLoop._call_tool`` (``verl/experimental/agent_loop/
    tool_agent_loop.py:490``) crashed every single call with
    ``AttributeError: 'dict' object has no attribute 'text'`` — it reads
    ``tool_execution_response.text`` by ATTRIBUTE, never by subscript. This
    class is the minimal attribute-compatible fix: it exposes exactly the
    ``text``/``image``/``video`` attributes the real ``ToolResponse`` model
    has, keeping this module's deliberate zero-veRL-import policy (see
    module docstring) intact — no ``from verl.tools.schemas import
    ToolResponse`` needed, so this file still imports fine in the no-veRL
    dev sandbox / CI.
    """

    __slots__ = ("text", "image", "video")

    def __init__(
        self,
        text: Optional[str] = None,
        image: Optional[list[Any]] = None,
        video: Optional[list[Any]] = None,
    ) -> None:
        self.text = text
        self.image = image
        self.video = video

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ToolResponseShim):
            return (self.text, self.image, self.video) == (other.text, other.image, other.video)
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"ToolResponseShim(text={self.text!r}, image={self.image!r}, video={self.video!r})"


class RealHookRoutedTool:
    """Duck-typed mirror of ``verl.tools.base_tool.BaseTool``'s contract
    (create/execute/release/calc_reward — [VERIFIED 2026-09-07] against the
    vendored source at ``/root/autodl-tmp/code/verl/`` on ``aiops-gpu``;
    see the module docstring for the exact dispatch code
    ``ToolAgentLoop`` runs). Does NOT literally subclass the real
    ``verl.tools.base_tool.BaseTool`` (veRL is not importable in this dev
    environment) — on the GPU box either use this class as-is as the
    ``class_name`` in ``tool_schema_bridge.build_tool_config_entries()`` or
    make it inherit from the real ``BaseTool`` there.

    Every ``execute()`` call routes through ``execute_tool_call`` above, so
    the real hook gate is exercised on every veRL-sampled tool_call, exactly
    per the task's non-negotiable invariant.
    """

    def __init__(
        self,
        config: dict[str, Any],
        tool_schema: Any = None,
        *,
        bash_executor: BashExecutor = ssh_tunnel_bash_executor,
        mcp_executor: Optional[McpExecutor] = ssh_tunnel_mcp_executor,
        cwd: Optional[str] = None,
    ) -> None:
        """[VERIFIED 2026-09-08] veRL's tool registry instantiates this class
        as ``tool_cls(config=..., tool_schema=...)`` (``verl/tools/
        tool_registry.py::initialize_tools_from_config`` — ``config`` is the
        tool_config.yaml entry's ``config`` dict, e.g. ``{"type": "native",
        "tool_name": "Bash"}``, and ``tool_schema`` the validated
        ``OpenAIFunctionToolSchema``) — this signature matches that call
        exactly, which the previous ``(tool_name, ...)`` shape did not (it
        would have raised TypeError at instantiation).

        Executor defaults are the TUNNEL implementations because veRL passes
        nothing else — when instantiated from ``tool_config.yaml`` on the GPU
        training box, these defaults ARE the wiring. Hermetic/local callers
        (tests, Mac-side debugging) pass explicit executors instead, exactly
        like the existing tests do; ``execute_tool_call``'s own module-level
        default remains the local subprocess executor, so importing this
        module never requires a tunnel.

        ``cwd`` resolution order: explicit argument > the tool_config entry's
        ``config["cwd"]`` > ``AIOPS_MAC_AGENT_DIR`` env var > ``None`` (no
        cwd control — see ``execute_tool_call``'s docstring for why
        production should always set one; the GPU-side launcher exports
        ``AIOPS_MAC_AGENT_DIR`` pointing at the Mac's ROOT AIops-agent
        checkout, where ``deploys.log``/``workspace/<svc>/`` live).
        """
        tool_name = (config or {}).get("tool_name")
        if not tool_name:
            raise ValueError(
                f"RealHookRoutedTool: config dict must carry a 'tool_name' entry (got {config!r}) "
                "— this is how verl's tool_config.yaml passes the tool's name"
            )
        self.name = tool_name
        self.tool_schema = tool_schema
        self._bash_executor = bash_executor
        self._mcp_executor = mcp_executor
        self._cwd = cwd or (config or {}).get("cwd") or os.environ.get(AIOPS_MAC_AGENT_DIR_ENV) or None

    async def create(
        self, instance_id: Optional[str] = None, **kwargs: Any
    ) -> tuple[str, ToolResponseShim]:
        """Mirror of ``BaseTool.create``'s exact contract
        (``verl/tools/base_tool.py``, verified 2026-09-07):

        - The real ``ToolAgentLoop`` calls this as
          ``instance_id, _ = await tool.create(create_kwargs={...})`` — the
          ``create_kwargs`` keyword lands in ``**kwargs`` and is ignored
          (this tool keeps no per-instance state, exactly like the base
          class's own default body).
        - Returns a 2-tuple ``(instance_id, tool_creation_response)``. When
          ``instance_id`` is None a fresh ``uuid4()`` is generated (never a
          shared ``"<tool>-instance"`` string — parallel rollout workers
          would otherwise collide). The second element is discarded by the
          real loop anyway, but mirrors the base class's empty
          ``ToolResponse()`` via ``ToolResponseShim`` (see that class's
          docstring for why this is an attribute-compatible shim rather than
          a plain dict — a plain dict crashed real rollouts, 2026-09-09).
        """
        resolved = instance_id if instance_id is not None else str(uuid.uuid4())
        return resolved, ToolResponseShim(text="")

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs: Any,
    ) -> tuple[ToolResponseShim, float, dict[str, Any]]:
        """Returns ``(tool_response, tool_reward_score, tool_metrics)`` —
        the exact shape ``verl.tools.base_tool.BaseTool.execute`` documents.
        ``agent_data`` is passed by ``ToolAgentLoop`` on every call
        (``await tool.execute(instance_id, tool_args, agent_data=agent_data)``)
        and is accepted-but-unused here, matching how the real ``BaseTool``
        subclasses that don't need it still receive it via ``**kwargs``
        without a signature mismatch.

        ``tool_response`` is a ``ToolResponseShim`` (NOT a plain dict) —
        [2026-09-09 实测修复] the real ``ToolAgentLoop._call_tool``
        (``tool_agent_loop.py:490``) reads ``tool_execution_response.text``
        by attribute; a plain ``{"text": ...}`` dict crashed every real
        tool call with ``AttributeError: 'dict' object has no attribute
        'text'`` once tool calls actually reached execution for the first
        time in a real cloud rollout. See ``ToolResponseShim``'s docstring.

        ``tool_reward_score`` is always ``0.0``: this project's step-level
        reward is computed by ``reward/step_reward.py`` from the assembled
        ``Trajectory`` afterwards, not per-call inside the tool itself —
        see that module's docstring for why (it needs cross-step context
        like "is this a repeat call" that a single ``execute()`` call
        doesn't have).
        """
        tool_call = SampledToolCall(tool_use_id=instance_id, tool_name=self.name, tool_input=parameters)
        result, verdict = await execute_tool_call(
            tool_call, bash_executor=self._bash_executor, mcp_executor=self._mcp_executor, cwd=self._cwd
        )
        text = observation_packer.stringify_tool_result_content(result.content)
        tool_response = ToolResponseShim(text=text)
        tool_metrics = {
            "hook_decision": verdict["decision"],
            "hook_action": verdict["action"],
            "hook_target": verdict["target"],
        }
        return tool_response, 0.0, tool_metrics

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0  # see execute()'s docstring: reward/ owns step-level scoring, not this shim

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        return None
