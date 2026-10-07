"""veRL custom_reward_function adapter: the per-SAMPLE ``compute_score`` real
veRL calls, wrapping ``grpo/reward_router.py``'s per-GROUP composition.

WHY THIS MODULE EXISTS
----------------------
``grpo/verl_config/aiops_grpo.yaml``'s ``reward.custom_reward_function``
originally pointed at ``grpo/reward_router.py::compute_group_rewards`` — a
whole-GROUP signature (``trajectories: list[Trajectory] -> dict``), chosen so
``group_reward_variance`` could be computed. Real veRL, however, calls the
custom reward function once PER SAMPLE. This module is the thin adapter that
yaml's comment anticipated: veRL keeps calling a per-sample
``compute_score(data_source, solution_str, ground_truth, extra_info)``, this
module reconstructs that one rollout's ``reward.trajectory.Trajectory`` from
``solution_str``, scores it via
``grpo.reward_router.compute_trajectory_reward``, and moves the group-level
variance computation into an in-process accumulator (``_GroupAccumulator``).

Everything below is grouped into three trust tiers, per this repo's
convention: [VERIFIED] facts checked against real source (file paths + line
numbers given), [JUDGMENT] engineering calls this module owns, and KNOWN
APPROXIMATIONS that are documented, not hidden.

[VERIFIED] — checked against the real veRL checkout on the training machine
on 2026-09-07:

1. Call convention — ``verl/workers/reward_manager/naive.py``:
   - L131-135: ``self.compute_score(data_source=data_source,
     solution_str=response_str, ground_truth=ground_truth,
     extra_info=extra_info)`` — keyword names are exactly these four.
   - L112-115: ``solution_str`` is ``tokenizer.decode(valid_response_ids,
     skip_special_tokens=True)`` over the RESPONSE segment only.
   - L121: ``ground_truth`` is ``data_item.non_tensor_batch["reward_model"]
     ["ground_truth"]`` — i.e. exactly the dict ``grpo/to_verl_dataset.py``
     wrote into ``reward_model.ground_truth`` (``route``/``remediation_type``
     /``suspect_service``; ``kind`` is currently absent from every annotation
     file, so ``Trajectory.ground_truth.get("kind")`` being ``None`` is the
     expected state, not an error).
   - L123-127: ``extra_info`` is the dataset row's static ``extra_info``
     (``split``/``index``/``alert_id``/``alert``) PLUS two runtime-injected
     keys: ``extra_info["num_turns"]`` and ``extra_info["rollout_reward_scores"]``
     (from ``AsyncRolloutRequest.reward_scores``; our
     ``verl_adapter/rollout_worker.py::RealHookRoutedTool.execute`` always
     returns ``0.0`` as the tool reward score, so that key is useless to us
     and this module never reads it). There is NO structured step-by-step
     trajectory in ``extra_info`` — the only faithful record is the
     ``solution_str`` text itself.
   - L145-152: the return value may be a plain float OR a dict carrying a
     ``"score"`` key (extra keys land in ``reward_extra_info``); this module
     returns a plain float.
   - L102 (``for i in range(len(data))``): samples are scored serially, in
     one process — the factual basis for the in-process group accumulator
     below.
2. Function loading — ``verl/trainer/ppo/reward.py::get_custom_reward_fn``
   (L53-83) -> ``verl/utils/import_utils.py::load_extern_object`` (L248) ->
   ``load_module`` (L151-191): the ``path:`` value is a FILE PATH loaded via
   ``importlib.util.spec_from_file_location`` under a synthetic module name —
   NO package context, and NOT registered in ``sys.modules`` (registration
   only happens when an explicit ``module_name`` is passed, which
   ``get_custom_reward_fn`` doesn't). Two consequences, both handled by the
   bootstrap block below: (a) the sibling ``reward``/``grpo`` imports need
   the repo root on ``sys.path``; (b) stdlib ``dataclasses`` dereferences
   ``sys.modules[cls.__module__]`` (``None`` here) when resolving the string
   annotations this module uses — verified empirically to crash
   ``@dataclass`` definition on Python 3.9.6 AND 3.11 under exactly that
   loading mode, so the module self-registers a shim in ``sys.modules``.
   ``path:`` itself stays relative to the trainer's CWD (``load_module``'s
   ``os.path.exists`` check).
3. What is inside ``solution_str`` — ``verl/experimental/agent_loop/
   tool_agent_loop.py``: tool responses become ``{"role": "tool", ...}``
   messages (L337-340) merged into the response region by
   ``ct_merge_non_assistant_msg`` (L377-385); responses are appended in the
   same order as the sampled calls (L317: ``enumerate(responses)`` over tasks
   created in call order); a rollout truncated by turn/budget limits
   (L281-286) can end right after a tool call with no observation.
   So ``solution_str`` = every assistant-generated turn (with its
   ``<tool_call>`` markup) + the interleaved, chat-template-rendered tool
   observations + the final assistant turn, all decoded with
   ``skip_special_tokens=True``.
4. Tool-call text format (``qwen3_xml``) — ``vllm/tool_parsers/__init__.py``
   L177-178 maps the name ``"qwen3_xml"`` to the
   ``qwen3_engine_tool_parser`` module; ``qwen3_engine_tool_parser.py`` is
   ``class Qwen3EngineToolParser(Qwen3ParserToolAdapter)``; the grammar lives
   in ``vllm/parser/qwen3.py`` (read in full)::

       <tool_call>
       <function=NAME>
       <parameter=key>value
       </parameter>
       </function>
       </tool_call>

   with optional ``<think>...</think>`` reasoning blocks, parameter regex
   ``r"<\\s*parameter\\s*=\\s*([^>]*)>(.*?)(?:<\\s*/\\s*parameter\\s*>|
   (?=<\\s*parameter\\s*=))"`` (DOTALL), each parameter value having exactly
   one leading and one trailing newline trimmed (``_trim_wrapping_newlines``),
   and string-only argument values (``tool_args_json=False``). The parser
   below ports that regex and trimming verbatim rather than re-inventing a
   "Qwen-like" format.
5. Tool-response text format — the real model's chat template
   ``/root/autodl-tmp/models/qwen3.5-9b-text-only/chat_template.jinja``
   (read in full): tool messages render as user-role turns
   ``<|im_start|>user\\n<tool_response>\\n{content}\\n</tool_response><|im_end|>\\n``
   (consecutive tool messages share one user header); assistant tool calls
   render as ``<tool_call>\\n<function=NAME>\\n<parameter=arg>\\nvalue\\n
   </parameter>\\n</function>\\n</tool_call>``.
6. Token survival (checked empirically with the real tokenizer):
   ``<tool_call>``/``</tool_call>``/``<tool_response>``/``</tool_response>``/
   ``<think>``/``</think>`` are added tokens with ``special=False`` — they
   SURVIVE ``skip_special_tokens=True`` decoding and are therefore present in
   ``solution_str``. ``<|im_start|>``/``<|im_end|>`` ARE special tokens and
   get stripped, leaving bare ``assistant\\n``/``user\\n`` role-word lines
   between turns (the parser strips those).
7. Route table — ``AIops-agent/agent/run.py::_run_inner`` (L68-135, read in
   full) and its established mirror
   ``data/cold_start/collect_trajectories.py::_infer_route`` (L676-691):
   confidence < 0.6 -> ``feishu_low_confidence``; ``online_op`` ->
   ``auto_remediated``/``feishu_online_op`` by ``executed_actions``, with the
   ``also_code_fix`` hybrid branching to ``*_and_code_fix_pr``;
   ``code_fix`` -> ``code_fix_pr``/``feishu_fix_unverified``; else
   ``info_only``. ``_infer_route()``'s confidence constant 0.6 matches
   ``agent/config.py::CONFIDENCE_THRESHOLD``'s default.
8. Hook recompute — ``AIops-agent/agent/core/remediation.py::classify_command``
   (L133-160) is a pure function of the command string +
   ``config.REMEDIATION_ALLOWED_TARGETS``; this module re-invokes it per Bash
   step, exactly the reuse convention already proven in
   ``verl_adapter/rollout_worker.py::route_through_real_hook`` (L146-166).

KNOWN APPROXIMATIONS (documented, deliberate — do NOT "fix" silently):
- Hook decisions are recomputed with ``classify_command`` ONLY.
  ``agent/core/hooks.py::guard_diagnose_ops`` has a further
  ``_READONLY_DENY``/``_WRITE_FALLBACK_DENY`` fallback layer (hooks.py
  L27-51) that this recompute cannot see: e.g. ``docker exec -it ad top``
  is classified ``passthrough`` by ``classify_command`` but DENIED by the
  real hook. Such steps reconstruct as ``passthrough`` instead of ``deny``.
  Rationale: ``classify_command`` is pure (``guard_diagnose_ops`` also has a
  ledger side effect on allow), and this is the same boundary
  ``route_through_real_hook``'s docstring already documents. Also note
  ``classify_command`` reads env-driven config (``AIOPS_REMEDIATION_ENABLED``
  / ``AIOPS_REMEDIATION_TARGETS``): if the reward process runs with
  different env than the rollout did, the recompute diverges.
- ``route`` uses the OPTIMISTIC default (assume the code-fix agent's
  verification passed -> ``code_fix_pr`` / ``*_and_code_fix_pr``), not the
  honest ``None`` that ``collect_trajectories.py::_infer_route_real``
  returns when the code-fix agent never ran. Why: ``Trajectory.route`` is a
  required ``str`` constrained to ``ALL_ROUTES`` — the reward data contract
  cannot represent ``None``, so we adopt ``_infer_route()``'s optimistic
  table. Consequence, per ``data/cold_start/ground_truth/_annotation_report.md``
  2026-09-07 "发现B": in real ``run.py`` an UNVERIFIED hybrid fix silently
  degrades to ``auto_remediated``/``feishu_online_op`` — this adapter would
  instead report ``auto_remediated_and_code_fix_pr``, potentially
  over-crediting ``route_match_reward`` in exactly those cases.
- ``skipped_duplicate`` is never produced: it requires the alert-fingerprint
  cooldown state, which a rollout does not have.
- ``post_action_health`` is always ``None`` (orchestration-layer concern,
  outside rollout scope); ``outcome_reward.stop_loss_reward`` then scores
  online_op trajectories as "not recovered" (0.0) by design.
- ``tool_call_schema_valid`` is always ``True`` for reconstructed steps:
  only tool calls that PARSED could become steps, so the -schema-valid
  signal can never fire here (malformed ``<tool_call>`` blocks are dropped
  and counted in ``ParsedRollout.n_dropped_malformed_calls``).
- Text-only reconstruction has inherent ambiguities: a tool observation
  that itself echoes ``<tool_response>`` markup would split wrongly, and a
  rollout truncated mid-``<tool_call>`` leaves an unclosed tag that is
  treated as final-turn text (yielding a schema-fallback diagnosis).

[JUDGMENT] — engineering calls this module owns:
- The k-th parsed tool call is paired with the k-th ``<tool_response>``
  block in global document order (valid for veRL's per-turn
  PROCESSING_TOOLS flow where all responses are appended before the next
  assistant turn; a call with no following response — truncation or
  ``max_parallel_calls`` cutoff — gets ``""``).
- ``Step.observation`` = the ``<tool_response>`` block content with the
  template's wrapping newlines stripped — i.e. exactly the (possibly
  veRL-truncated) text the model saw in context.
- The final ``Diagnosis`` is the LAST balanced JSON object in the final
  non-tool text segment; it is then validated with the REAL
  ``agent.core.schema.Diagnosis`` pydantic model. On extraction or
  validation failure the module follows the repo's existing schema-fallback
  convention (see ``collect_trajectories.py::_run_diagnose_capturing_steps``
  and ``reward/step_reward.py::schema_fallback_unrecovered_penalty``):
  ``Diagnosis.fallback(...)`` + sibling key
  ``final_diagnosis["schema_fallback_triggered"] = True`` (-> the -0.2
  unrecovered-fallback penalty, and confidence 0 -> ``feishu_low_confidence``
  route).
- ``executed_actions`` is rebuilt from the reconstructed steps whose
  recomputed ``hook_decision == "allow"`` — the same "code-level record, not
  model self-report" semantics as ``collect_trajectories.py::_assemble``
  (L641-652), which mirrors the real hook ledger backfill.
- ``_GroupAccumulator`` collects per-``extra_info["index"]`` (one GRPO
  group = 6 rollouts of the same dataset row share one ``index``) and calls
  ``compute_group_rewards`` once a bucket reaches ``DEFAULT_GROUP_SIZE``
  (= 6, mirroring ``aiops_grpo.yaml``'s ``actor_rollout_ref.rollout.n``;
  the shape test pins the two together). *** This accumulator ASSUMES the
  same process receives all members of a group, sequentially — that
  assumption is NOT verified against a real veRL training run. ``naive.py``'s
  serial per-sample loop (fact 1 above) makes it plausible, but whether ray
  ever shards one group's reward batch across workers/processes is unknown.
  This is the same uncertainty class as the yaml's flagged "哪种 hook 点真实
  存在" question: plausible wiring, unverified end-to-end. ***
- The scenario-type key fed into
  ``route_trajectory_reward_to_variance_tracker`` is
  ``ground_truth.get("kind") or "unknown"`` — doc §5.3 案例四's granularity
  is the fault kind, but current annotations carry no ``kind``, so history
  lands in one ``"unknown"`` bucket until annotations add it.

Importability: this module imports only stdlib + this repo's pure ``reward``/
``grpo`` packages at module level (no torch/transformers/vllm), and defers
the ``AIops-agent`` imports (``agent.core.remediation``/``agent.core.schema``
— pydantic-only) and ``data/cold_start/signals.py`` to first use, so it stays
importable and unit-testable in this dependency-light dev environment.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# sys.path bootstrap. veRL loads this file BY PATH with no package context
# (see module docstring fact 2), so the sibling ``reward``/``grpo`` imports
# below only work if the repo root is on sys.path. When imported normally as
# ``grpo.verl_reward_adapter`` (tests) this is a no-op.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# veRL's ``load_module`` also does NOT register the loaded module in
# ``sys.modules`` (it only registers when an explicit ``module_name`` is
# passed — ``get_custom_reward_fn`` doesn't pass one). Stdlib ``dataclasses``
# resolves string annotations (this module uses ``from __future__ import
# annotations``) through ``sys.modules[cls.__module__].__dict__`` and
# dereferences ``None`` when the module is unregistered — verified
# empirically to crash ``@dataclass`` definition on BOTH this dev machine
# (Python 3.9.6) and the training machine's env (Python 3.11), under exactly
# veRL's unregistered-``spec_from_file_location`` loading mode. Register a
# module object for ourselves so class definition works in that mode. The
# snapshot copy below is sufficient (and stays honest): the only consumers
# of ``sys.modules[__name__]`` during class creation resolve typing-ish
# identifiers (ClassVar/InitVar/KW_ONLY) that tolerate absence; the real
# classes/functions keep living in this module's own globals either way.
if __name__ not in sys.modules:
    import types as _types

    _self_shim = _types.ModuleType(__name__)
    _self_shim.__dict__.update(globals())
    sys.modules[__name__] = _self_shim

from grpo.reward_router import (  # noqa: E402  (after the sys.path bootstrap above)
    compute_group_rewards,
    compute_trajectory_reward,
    route_trajectory_reward_to_variance_tracker,
)
from reward.trajectory import Step, Trajectory  # noqa: E402

__all__ = [
    "DEFAULT_GROUP_SIZE",
    "ROUTE_CONFIDENCE_THRESHOLD",
    "ParsedRollout",
    "compute_score",
    "extract_last_json_object",
    "infer_route_optimistic",
    "parse_solution_str",
    "parse_tool_call_block",
    "recompute_hook_verdict",
    "reset_group_accumulator",
    "trajectory_from_solution_str",
]

#: doc §5.1 group_size = 6; must equal ``aiops_grpo.yaml``'s
#: ``actor_rollout_ref.rollout.n`` (pinned together by
#: ``grpo/tests/test_verl_config.py`` so the two can't drift).
DEFAULT_GROUP_SIZE = 6

#: Mirrors ``collect_trajectories.py::_infer_route``'s constant, which itself
#: matches ``agent/config.py::CONFIDENCE_THRESHOLD``'s default (0.6). Env
#: overrides of the real config are deliberately NOT read here — the route
#: table this module mirrors hardcodes the default too.
ROUTE_CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# qwen3_xml text format — regexes ported from the real parser source
# (see module docstring facts 4-5).
# ---------------------------------------------------------------------------

#: One ``<tool_call>...</tool_call>`` (assistant-generated) or
#: ``<tool_response>...</tool_response>`` (env-appended) block, non-greedy so
#: multiple blocks in one text split correctly.
_TOOL_EVENT_RE = re.compile(
    r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>", re.DOTALL
)

#: Reasoning blocks (Qwen3 ``<think>``); removed before final-answer parsing.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

#: ``<function=NAME>`` inside a tool_call block (parser state TOOL_NAME ends
#: at the closing angle bracket).
_FUNCTION_NAME_RE = re.compile(r"<\s*function\s*=\s*([^<>]+?)\s*>")

#: VERBATIM port of vLLM's ``_PARAM_RE`` (vllm/parser/qwen3.py), including
#: the lookahead alternative close (a parameter may be closed by the NEXT
#: parameter's opening tag rather than by its own ``</parameter>``).
_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)


def _trim_wrapping_newlines(value: str) -> str:
    """VERBATIM port of vLLM's ``_trim_wrapping_newlines``: strip one leading
    and one trailing newline (the Qwen3 template markup around parameter
    values), not arbitrary whitespace.
    """
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


@dataclass
class ParsedRollout:
    """The structured view of one decoded ``solution_str``.

    Attributes:
        tool_calls: ordered ``{"tool_name": str, "tool_input": dict}`` entries
            for every tool_call block that parsed.
        observations: ordered observation texts; the k-th entry belongs to
            the k-th tool call (``""`` when the rollout ended before that
            call's response was appended — truncation /
            ``max_parallel_calls`` cutoff).
        final_text: the last non-tool text segment (think blocks and
            role-word lines stripped) — where the final Diagnosis JSON lives.
        n_dropped_malformed_calls: tool_call blocks that existed but did not
            yield a function name (dropped, per the ``tool_call_schema_valid``
            approximation in the module docstring).
    """

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_text: str = ""
    n_dropped_malformed_calls: int = 0


def parse_tool_call_block(block_text: str) -> Optional[dict[str, Any]]:
    """Parse one ``<tool_call>...</tool_call>`` block's INNER text into
    ``{"tool_name": ..., "tool_input": {...}}``, or ``None`` if the block has
    no ``<function=NAME>`` (a malformed call the vLLM parser would have
    dropped). Regex/trimming ported verbatim from ``vllm/parser/qwen3.py``
    (see module docstring fact 4) so this accepts exactly the text the real
    parser accepts.
    """
    name_match = _FUNCTION_NAME_RE.search(block_text)
    if name_match is None:
        return None
    tool_input: dict[str, Any] = {}
    for param_match in _PARAM_RE.finditer(block_text):
        param_name = param_match.group(1).strip()
        tool_input[param_name] = _trim_wrapping_newlines(param_match.group(2))
    return {"tool_name": name_match.group(1), "tool_input": tool_input}


def _strip_reasoning_and_role_words(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (and an unclosed trailing
    ``<think>`` — everything after it is reasoning with no answer), then drop
    the bare ``assistant``/``user`` role-word lines left behind when
    ``<|im_start|>``/``<|im_end|>`` get stripped from the decode (module
    docstring fact 6).
    """
    text = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    lines = [ln for ln in text.split("\n") if ln.strip() not in ("assistant", "user")]
    return "\n".join(lines).strip()


def parse_solution_str(solution_str: str) -> ParsedRollout:
    """Split one decoded rollout text into ordered tool calls, observations,
    and the final assistant text. Pairing rule (k-th call <-> k-th response
    in global order) and truncation semantics are [JUDGMENT] items documented
    in the module docstring.
    """
    parsed = ParsedRollout()
    events = list(_TOOL_EVENT_RE.finditer(solution_str or ""))
    last_end = 0
    for match in events:
        block = match.group(0)
        if block.startswith("<tool_call>"):
            call = parse_tool_call_block(block)
            if call is None:
                parsed.n_dropped_malformed_calls += 1
            else:
                parsed.tool_calls.append(call)
        else:  # <tool_response>
            # Template renders "\n<tool_response>\n" + content + "\n</tool_response>";
            # strip the wrapping newlines so the observation is exactly the text
            # the tool returned (modulo veRL's own truncation markers).
            parsed.observations.append(block[len("<tool_response>") : -len("</tool_response>")].strip("\n"))
        last_end = match.end()

    if len(parsed.observations) > len(parsed.tool_calls):
        # More responses than calls should not happen in veRL's flow; keep the
        # prefix-aligned pairing and let the trailing extras be dropped rather
        # than silently mis-aligning earlier pairs.
        parsed.observations = parsed.observations[: len(parsed.tool_calls)]
    while len(parsed.observations) < len(parsed.tool_calls):
        parsed.observations.append("")

    tail = solution_str[last_end:] if events else (solution_str or "")
    parsed.final_text = _strip_reasoning_and_role_words(tail)
    return parsed


def extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    """Return the LAST balanced top-level ``{...}`` object in ``text`` that
    ``json.loads`` parses into a dict, or ``None``. String-literal-aware
    brace matching, so braces inside JSON string values don't confuse the
    scan. [JUDGMENT] "last" because the final assistant turn may contain
    prose and fenced code blocks around the actual Diagnosis JSON, and the
    Diagnosis is the last thing a converged rollout emits.
    """
    candidates: list[str] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------------------
# Real AIops-agent imports (lazy: pydantic-only, no claude_agent_sdk needed —
# same import surface ``collect_trajectories.py::_load_aiops_core`` uses).
# ---------------------------------------------------------------------------

_REAL_MODULES: Optional[tuple[Any, Any, Any]] = None
_REAL_MODULES_LOCK = threading.Lock()


def _load_real_modules() -> tuple[Any, Any, Any]:
    """Lazy, cached import of the REAL ``agent.core.remediation`` /
    ``agent.core.schema`` and ``data/cold_start/signals`` modules — the exact
    import surface ``verl_adapter/_repo_paths.py`` already established for
    this repo (duplicated here, not imported, because veRL loads this file
    without package context and ``grpo`` should not depend on
    ``verl_adapter``'s private helpers).
    """
    global _REAL_MODULES
    with _REAL_MODULES_LOCK:
        if _REAL_MODULES is not None:
            return _REAL_MODULES
        aiops_dir = _REPO_ROOT / "AIops-agent"
        cold_start_dir = _REPO_ROOT / "data" / "cold_start"
        for p in (str(aiops_dir), str(cold_start_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from agent.core import remediation, schema  # type: ignore

            import signals  # type: ignore  (data/cold_start/signals.py)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "grpo.verl_reward_adapter needs the real AIops-agent core modules "
                "(agent.core.remediation / agent.core.schema) and "
                "data/cold_start/signals.py to reconstruct hook verdicts and the "
                f"Diagnosis; import failed from {_REPO_ROOT}: {exc!r}"
            ) from exc
        _REAL_MODULES = (remediation, schema, signals)
        return _REAL_MODULES


def recompute_hook_verdict(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Recompute this step's PreToolUse verdict by re-invoking the REAL
    ``agent.core.remediation.classify_command`` — the pure-function reuse
    convention proven in ``verl_adapter/rollout_worker.py::
    route_through_real_hook`` (including its action/target filling rule: they
    are only meaningful when the verdict is allow/deny).

    Non-Bash tools (Read/Grep/Glob/``mcp__*__search_past_incidents``) are
    read-only and outside the whitelist's jurisdiction -> ``passthrough``.

    KNOWN APPROXIMATION (module docstring): this misses ``guard_diagnose_ops``
    's ``_READONLY_DENY``/``_WRITE_FALLBACK_DENY`` fallback layer — e.g.
    ``docker exec ...`` reconstructs as ``passthrough`` though the real hook
    denies it.
    """
    if tool_name != "Bash":
        return {"decision": "passthrough", "action": None, "target": None}
    remediation, _, _ = _load_real_modules()
    command = str((tool_input or {}).get("command", ""))
    verdict = remediation.classify_command(command)
    decision = verdict.get("decision")
    action = target = None
    if decision in ("allow", "deny"):
        action, target = verdict.get("action"), verdict.get("target")
    return {"decision": decision, "action": action, "target": target}


def _parse_final_diagnosis(final_text: str) -> tuple[Any, bool]:
    """Extract + validate the final ``Diagnosis`` from the final text segment.

    Returns ``(diag, schema_fallback_triggered)`` where ``diag`` is a REAL
    ``agent.core.schema.Diagnosis``. Follows the repo's existing
    schema-fallback convention (``collect_trajectories.py::
    _run_diagnose_capturing_steps`` L277-287): any extraction/validation
    failure becomes ``Diagnosis.fallback(...)`` with the sibling flag set —
    which ``reward/step_reward.py::schema_fallback_unrecovered_penalty``
    prices at -0.2, and whose confidence 0 routes to
    ``feishu_low_confidence``.
    """
    _, schema, _ = _load_real_modules()
    obj = extract_last_json_object(final_text)
    if obj is None:
        return (
            schema.Diagnosis.fallback(
                f"final turn contained no parseable Diagnosis JSON; raw tail={final_text[-200:]!r}"
            ),
            True,
        )
    try:
        return schema.Diagnosis.model_validate(obj), False
    except Exception as exc:  # noqa: BLE001  same degrade-not-crash convention
        return (
            schema.Diagnosis.fallback(
                f"Diagnosis schema validation failed: {exc}; raw={str(obj)[:200]!r}"
            ),
            True,
        )


def infer_route_optimistic(final_diagnosis: dict[str, Any]) -> str:
    """Route inference over the SAME table as
    ``data/cold_start/collect_trajectories.py::_infer_route`` (itself the
    mirror of ``AIops-agent/agent/run.py::_run_inner``, verified L68-135):
    confidence < 0.6 -> ``feishu_low_confidence``; ``online_op`` ->
    ``auto_remediated``/``feishu_online_op`` by ``executed_actions`` (hybrid
    ``also_code_fix`` -> ``*_and_code_fix_pr``); ``code_fix`` ->
    ``code_fix_pr``; else ``info_only``.

    OPTIMISTIC on purpose (module docstring): assumes the code-fix agent's
    verification passed, because ``Trajectory.route`` is a required ``str``
    in ``ALL_ROUTES`` and cannot carry ``_infer_route_real()``'s honest
    ``None``. ``skipped_duplicate`` is unreachable here (needs the
    fingerprint-cooldown state a rollout does not have).
    """
    confidence = final_diagnosis.get("confidence")
    if confidence is None or float(confidence) < ROUTE_CONFIDENCE_THRESHOLD:
        return "feishu_low_confidence"
    remediation_type = final_diagnosis.get("remediation_type")
    auto_done = bool(final_diagnosis.get("executed_actions"))
    if remediation_type == "online_op":
        if final_diagnosis.get("also_code_fix"):
            return "auto_remediated_and_code_fix_pr" if auto_done else "online_op_and_code_fix_pr"
        return "auto_remediated" if auto_done else "feishu_online_op"
    if remediation_type == "code_fix":
        return "code_fix_pr"
    return "info_only"


def trajectory_from_solution_str(
    solution_str: str,
    ground_truth: Optional[dict[str, Any]] = None,
    extra_info: Optional[dict[str, Any]] = None,
) -> Trajectory:
    """Reconstruct one full ``reward.trajectory.Trajectory`` from the decoded
    rollout text — the per-sample inverse of what the rollout layer wrote
    into the response segment. Field-by-field provenance:

    - ``alert``: ``extra_info["alert"]`` (written by
      ``grpo/to_verl_dataset.py``; ``scenario_hint`` already stripped there).
    - ``steps``: parsed (tool_call, tool_response) pairs; hook verdicts
      RE-COMPUTED via the real ``classify_command``; signal coverage /
      repeat flags via the real ``data/cold_start/signals.py::annotate_steps``
      (same reuse as ``verl_adapter/rollout_worker.py::StepAccumulator.
      build_trajectory``).
    - ``final_diagnosis``: last JSON object in the final text, validated by
      the real ``Diagnosis`` model (fallback + sibling flag on failure);
      ``executed_actions`` rebuilt from allow-verdict steps (code-level
      record, not model self-report).
    - ``ground_truth``: passed through from veRL (``kind`` absent -> ``None``
      is expected).
    - ``route``: optimistic inference (see ``infer_route_optimistic``).
    - ``post_action_health``: always ``None`` (module docstring).
    """
    remediation, schema, signals_mod = _load_real_modules()
    parsed = parse_solution_str(solution_str)

    raw_steps: list[dict[str, Any]] = []
    for call, observation in zip(parsed.tool_calls, parsed.observations):
        verdict = recompute_hook_verdict(call["tool_name"], call["tool_input"])
        raw_steps.append(
            {
                "tool_name": call["tool_name"],
                "tool_input": call["tool_input"],
                "hook_decision": verdict["decision"],
                "hook_action": verdict["action"],
                "hook_target": verdict["target"],
                "observation": observation,
            }
        )
    signals_mod.annotate_steps(raw_steps)
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
        for s in raw_steps
    ]

    diag, schema_fallback = _parse_final_diagnosis(parsed.final_text)
    # executed_actions: rebuilt from the steps the (recomputed) hook ALLOWED —
    # the same code-level-record semantics as collect_trajectories.py::_assemble
    # (L641-652) and the real ledger backfill; never the model's self-report.
    diag.executed_actions = [
        schema.ExecutedAction(
            action=s["hook_action"] or "",
            target=s["hook_target"] or "",
            command=str(s["tool_input"].get("command", "")),
        )
        for s in raw_steps
        if s["hook_decision"] == "allow"
    ]
    final_diagnosis = diag.model_dump(by_alias=True)
    final_diagnosis["schema_fallback_triggered"] = schema_fallback

    extra = extra_info if isinstance(extra_info, dict) else {}
    alert = extra.get("alert") if isinstance(extra.get("alert"), dict) else {}

    return Trajectory(
        alert=alert,
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=dict(ground_truth) if isinstance(ground_truth, dict) else {},
        route=infer_route_optimistic(final_diagnosis),
        post_action_health=None,
    )


# ---------------------------------------------------------------------------
# In-process group accumulator: the group-level half of the adapter.
# ---------------------------------------------------------------------------


class _GroupAccumulator:
    """Collects per-sample ``Trajectory`` objects keyed by
    ``extra_info["index"]`` and, each time a bucket reaches ``group_size``,
    calls ``grpo.reward_router.compute_group_rewards`` on the whole group and
    feeds the variance into ``route_trajectory_reward_to_variance_tracker``.

    *** UNVERIFIED ASSUMPTION (module docstring): the same process receives
    all members of one GRPO group, sequentially — plausible given
    ``naive.py``'s serial per-sample loop, but never confirmed against a real
    (possibly ray-sharded) veRL training run. If that assumption fails, a
    bucket never reaches ``group_size`` and no variance is computed (the
    per-sample score returned by ``compute_score`` is unaffected). ***
    """

    def __init__(self, group_size: int = DEFAULT_GROUP_SIZE) -> None:
        self.group_size = group_size
        self._pending: dict[Any, list[Trajectory]] = defaultdict(list)
        self._last_group_result: dict[Any, dict[str, Any]] = {}
        #: scenario_type -> list of group variances, oldest-first (the state
        #: ``route_trajectory_reward_to_variance_tracker`` updates).
        self.variance_history: dict[str, list[float]] = {}
        self.last_tracker_result: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()

    def add(self, index: Any, traj: Trajectory) -> Optional[dict[str, Any]]:
        """Add one rollout to its ``index`` bucket; returns the
        ``compute_group_rewards`` result when this addition completed a
        group, else ``None``.
        """
        with self._lock:
            bucket = self._pending[index]
            bucket.append(traj)
            if len(bucket) < self.group_size:
                return None
            trajectories = self._pending.pop(index)
        return self._flush(index, trajectories)

    def _flush(self, index: Any, trajectories: list[Trajectory]) -> dict[str, Any]:
        result = compute_group_rewards(trajectories)
        with self._lock:
            self._last_group_result[index] = result
            # [JUDGMENT] scenario granularity: doc §5.3 案例四 wants the fault
            # kind; current annotations carry no `kind`, so until they do the
            # history lands in one "unknown" bucket.
            scenario_type = str(trajectories[0].ground_truth.get("kind") or "unknown") if trajectories else "unknown"
            tracker = route_trajectory_reward_to_variance_tracker(
                scenario_type, result["total_rewards"], self.variance_history
            )
            self.variance_history = tracker["updated_variance_history"]
            self.last_tracker_result = tracker
        return result

    def pending(self, index: Any) -> list[Trajectory]:
        with self._lock:
            return list(self._pending.get(index, []))

    def last_group_result(self, index: Any) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._last_group_result.get(index)

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._last_group_result.clear()
            self.variance_history = {}
            self.last_tracker_result = None


#: Module-level singleton ``compute_score`` feeds. Trainers/tests can reset
#: it between epochs/runs via ``reset_group_accumulator()``.
_GROUP_ACCUMULATOR = _GroupAccumulator()


def reset_group_accumulator() -> None:
    """Clear the module-level group accumulator (between epochs / tests)."""
    _GROUP_ACCUMULATOR.reset()


# ---------------------------------------------------------------------------
# The per-sample entry point real veRL calls.
# ---------------------------------------------------------------------------


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Optional[dict[str, Any]] = None,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    """veRL's per-sample custom reward function — the exact keyword signature
    ``NaiveRewardManager.__call__`` invokes (module docstring fact 1).

    ``data_source`` is accepted and ignored: our dataset writes a single
    ``"aiops_diagnosis"`` source and this adapter does not branch on it
    (unlike veRL's ``default_compute_score``, which dispatches per source).
    ``extra_info["num_turns"]`` / ``extra_info["rollout_reward_scores"]`` are
    likewise accepted but not relied upon (the latter is always 0.0 from
    ``RealHookRoutedTool.execute``).

    Returns the trajectory's ``compute_trajectory_reward`` total as a plain
    float (naive.py L145-152 also accepts a dict with a ``"score"`` key; we
    keep the return simple). Side effect: the reconstructed ``Trajectory`` is
    added to the group accumulator under ``extra_info["index"]`` (when
    present), so that every completed group of ``DEFAULT_GROUP_SIZE`` gets
    its ``compute_group_rewards`` variance computed in-process.
    """
    traj = trajectory_from_solution_str(solution_str, ground_truth, extra_info)
    breakdown = compute_trajectory_reward(traj)
    index = extra_info.get("index") if isinstance(extra_info, dict) else None
    if index is not None:
        _GROUP_ACCUMULATOR.add(index, traj)
    return float(breakdown["total_reward"])
