"""Unit tests for ``grpo/verl_reward_adapter.py`` — the per-sample
``compute_score`` adapter between real veRL's calling convention and
``grpo/reward_router.py``'s group-level composition.

The ``solution_str`` fixtures below are hand-written to the REAL qwen3_xml
multi-turn text format, byte-shaped like what ``NaiveRewardManager`` decodes:
``<tool_call>\\n<function=NAME>\\n<parameter=k>\\nv\\n</parameter>\\n</function>\\n</tool_call>``
blocks for the model's calls, ``user\\n<tool_response>\\n...\\n</tool_response>``
blocks for the env-appended observations (bare ``assistant``/``user`` role-word
lines left behind by the stripped ``<|im_start|>``/``<|im_end|>``, and
``<think>`` blocks that survive ``skip_special_tokens=True``). See the adapter
module's docstring for the real-source citations (vllm/parser/qwen3.py, the
qwen3.5-9b chat_template.jinja, verl naive.py / tool_agent_loop.py — all read
on the training machine, not guessed).

What each test group proves:

- ``TestSingleSampleEquivalence``: one hand-written rollout text, scored via
  ``compute_score``, gives EXACTLY the same reward as a ``Trajectory``
  independently constructed through the established real-hook path
  (``verl_adapter.rollout_worker.StepAccumulator`` +
  ``route_through_real_hook``, which runs the FULL ``guard_diagnose_ops``
  including the deny-fallback layers — so the equivalence also validates the
  ``classify_command``-only recompute for every command in the fixture).
  The expected total is additionally hand-computed from the doc's calibrated
  magnitudes and pinned, so a silent change in any reward term fails loudly.
- ``TestGroupAccumulator``: per-sample ``compute_score`` calls sharing one
  ``extra_info["index"]`` accumulate in-process; the 6th call triggers
  ``compute_group_rewards`` with the right totals/variance and feeds the
  variance tracker; a second full group appends a second variance entry.
- ``TestRouteInference``: the optimistic route table (mirrors
  ``collect_trajectories.py::_infer_route`` / ``run.py::_run_inner``), both
  as plain-dict unit cases and end-to-end through a real rollout text.
- ``TestHookRecompute``: ``classify_command`` recompute branches (allow /
  deny-by-target / passthrough / non-Bash), equivalence with the real hook
  outside its fallback layer, and the DOCUMENTED approximation where the
  fallback layer diverges (``docker exec`` -> real hook denies, recompute
  says passthrough).
- ``TestQwen3XmlParsing`` / ``TestFinalDiagnosisExtraction``: the parser's
  format coverage and the schema-fallback convention.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import pytest

from grpo.reward_router import compute_trajectory_reward
from grpo.verl_reward_adapter import (
    DEFAULT_GROUP_SIZE,
    _GROUP_ACCUMULATOR,
    _load_real_modules,
    compute_score,
    extract_last_json_object,
    infer_route_optimistic,
    parse_solution_str,
    parse_tool_call_block,
    recompute_hook_verdict,
    reset_group_accumulator,
    trajectory_from_solution_str,
)
from reward.anti_hacking import group_reward_variance
from reward.trajectory import Trajectory
from verl_adapter.rollout_worker import (
    SampledToolCall,
    StepAccumulator,
    ToolResult,
    route_through_real_hook,
)

# ---------------------------------------------------------------------------
# Fixtures: ground truth, alert, the "good" diagnosis, and a hand-written
# qwen3_xml multi-turn rollout text containing one passthrough Bash, one
# prometheus Bash, one Read, one DENIED Bash (non-whitelisted target), one
# ALLOWED Bash (whitelisted target), and a final Diagnosis JSON.
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    "route": "auto_remediated",
    "remediation_type": "online_op",
    "suspect_service": "ad",
    "kind": "resource",
}

ALERT = {
    "alertname": "MockAdCpuThrottled",
    "service": "ad",
    "labels": {"service": "ad", "severity": "warning", "job": "ad"},
    "annotations": {
        "summary": "ad 服务 CPU 打满，p99 延迟飙升",
        "description": "ad 服务容器 CPU throttling 持续 100%，AdRequest p99 从 80ms 涨到 1.2s。",
    },
    "startsAt": "2026-06-20T11:00:00Z",
}

GOOD_DIAG_FIELDS = {
    "summary": "ad 服务 CPU 打满导致延迟飙升，抬高资源上限后恢复。",
    "kind": "resource",
    "suspect_service": "ad",
    "suspect_repo": None,
    "suspect_commit_hint": None,
    "suspect_file_hint": None,
    "remediation_type": "online_op",
    "remediation_detail": "已临时抬高 ad 容器 CPU/内存上限缓解打满。",
    "confidence": 0.82,
    "evidence": [
        "ad   CPU% 99.8   MEM 480MiB / 512MiB",
        "Prometheus: ad 容器 CPU throttling 比例过去 10 分钟持续 100%。",
    ],
    "also_code_fix": False,
}

#: The same calls, hand-written (NOT produced by the adapter's parser), for
#: building the independently-constructed expected Trajectory.
EXPECTED_CALLS = [
    {"tool_name": "Bash", "tool_input": {"command": "docker stats --no-stream ad"}},
    {
        "tool_name": "Bash",
        "tool_input": {
            "command": (
                'curl -s http://localhost:9090/api/v1/query'
                '?query=rate(container_cpu_usage_seconds_total{name="ad"}[1m])'
            )
        },
    },
    {"tool_name": "Read", "tool_input": {"file_path": "/repo/ad/src/main.py"}},
    {"tool_name": "Bash", "tool_input": {"command": "docker restart product-catalog"}},
    {"tool_name": "Bash", "tool_input": {"command": "docker update --cpus 2 --memory 1024m ad"}},
]

EXPECTED_OBSERVATIONS = [
    "ad   CPU% 99.8   MEM 480MiB / 512MiB",
    "Prometheus: ad 容器 CPU throttling 比例过去 10 分钟持续 100%。",
    "async def handle(request):\n    return process(request)",
    (
        "PreToolUse hook 拒绝执行：低风险操作 restart_instance 命中，但目标 product-catalog "
        "不在可自动处置白名单内，需改发飞书卡片交人工重启。"
    ),
    "ad 容器资源上限已提升为 --cpus 2 --memory 1024m。",
]


def _tool_call_xml(name: str, **params: str) -> str:
    inner = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{inner}</function>\n</tool_call>"


def _tool_response_xml(obs: str) -> str:
    return f"user\n<tool_response>\n{obs}\n</tool_response>\n"


def _turns_as_xml(calls: list[dict], observations: list[str]) -> list[tuple[Optional[str], str, str]]:
    out = []
    for call, obs in zip(calls, observations):
        xml = _tool_call_xml(call["tool_name"], **call["tool_input"])
        out.append((None, xml, obs))
    return out


def _render_solution_str(turns, final_text: str, *, think: bool = True) -> str:
    """Render a decoded ``solution_str`` byte-shaped like the real chat
    template's surviving text: ``<think>`` blocks, bare role-word lines, one
    newline after each turn's closing tag (the ``\\n`` that follows the
    stripped ``<|im_end|>``)."""
    parts: list[str] = []
    if think:
        parts.append("<think>\n需要先确认容器状态与资源水位。\n</think>\n\n")
    for i, (_prose, call_xml, obs) in enumerate(turns):
        if i > 0:
            parts.append("assistant\n")
            if think:
                parts.append("<think>\n继续排查。\n</think>\n\n")
        parts.append(call_xml)
        parts.append("\n")  # newline surviving the stripped <|im_end|>
        parts.append(_tool_response_xml(obs))
    parts.append("assistant\n")
    if think:
        parts.append("<think>\n结论已明确。\n</think>\n\n")
    parts.append(final_text)
    return "".join(parts)


def good_solution_str() -> str:
    turns = _turns_as_xml(EXPECTED_CALLS, EXPECTED_OBSERVATIONS)
    return _render_solution_str(turns, json.dumps(GOOD_DIAG_FIELDS, ensure_ascii=False, indent=2))


def degraded_solution_str() -> str:
    """Same tool turns, but the final turn is prose without any JSON — the
    schema-fallback path."""
    turns = _turns_as_xml(EXPECTED_CALLS, EXPECTED_OBSERVATIONS)
    return _render_solution_str(turns, "经排查，ad 服务 CPU 打满，已抬高资源上限缓解，无需进一步动作。")


# ---------------------------------------------------------------------------
# TestSingleSampleEquivalence
# ---------------------------------------------------------------------------


class TestSingleSampleEquivalence:
    def _expected_trajectory(self) -> Trajectory:
        """Independent construction: the established rollout-layer path
        (``StepAccumulator`` + the FULL real hook via
        ``route_through_real_hook``), NOT the adapter under test."""
        _, schema, _ = _load_real_modules()
        accumulator = StepAccumulator()
        for i, (call, obs) in enumerate(zip(EXPECTED_CALLS, EXPECTED_OBSERVATIONS)):
            tool_call = SampledToolCall(
                tool_use_id=f"expected-{i}", tool_name=call["tool_name"], tool_input=call["tool_input"]
            )
            verdict = asyncio.run(route_through_real_hook(tool_call))
            accumulator.record(tool_call, ToolResult(tool_call.tool_use_id, obs), verdict)
        diag = schema.Diagnosis.model_validate(GOOD_DIAG_FIELDS)
        diag.executed_actions = [
            schema.ExecutedAction(
                action="scale_resources", target="ad", command="docker update --cpus 2 --memory 1024m ad"
            )
        ]
        return accumulator.build_trajectory(
            alert=ALERT,
            final_diagnosis=diag.model_dump(by_alias=True),
            ground_truth=dict(GROUND_TRUTH),
            route="auto_remediated",
        )

    def test_compute_score_matches_independently_built_trajectory(self):
        reset_group_accumulator()
        expected = self._expected_trajectory()
        expected_breakdown = compute_trajectory_reward(expected)

        extra_info = {"split": "train", "index": 1234, "alert_id": "mock-ad", "alert": ALERT}
        score = compute_score(
            data_source="aiops_diagnosis",
            solution_str=good_solution_str(),
            ground_truth=GROUND_TRUTH,
            extra_info=extra_info,
        )
        assert isinstance(score, float)
        assert score == pytest.approx(expected_breakdown["total_reward"])
        assert score == pytest.approx(3.70)  # hand-computed, see docstring below

        traj = trajectory_from_solution_str(good_solution_str(), GROUND_TRUTH, extra_info)
        adapter_breakdown = compute_trajectory_reward(traj)
        assert adapter_breakdown["per_step_rewards"] == pytest.approx(expected_breakdown["per_step_rewards"])

    def test_hand_computed_total(self):
        """Pin the hand-computed total so silent changes in any reward term
        fail loudly. Arithmetic (kind=resource weights: kubectl_status 1.5,
        prometheus 1.3; credit is MARGINAL — each newly-covered signal type
        is paid its weight exactly once, on the step that first covered it):

        step rewards (schema +0.05 each; advances +0.10 on new signal type;
        deny -0.15; allow +0.10):
          S1 docker stats   : 0.15 + credit 1.5 (new kubectl_status)  = 1.65
          S2 prometheus curl: 0.15 + credit 1.3 (new prometheus)      = 1.45
          S3 Read           : 0.05 + credit 0                         = 0.05
          S4 denied restart : -0.10 + credit 0                        = -0.10
          S5 allowed update : 0.15 + credit 0                         = 0.15
        terminal (on S5): root cause 0.3 (service+kind+remediation_type all
        match) + route 0.2 (auto_remediated == gt) + stop-loss 0.0
        (post_action_health None -> "not recovered") + dual-source 0 (2 signal
        types) + turn/truncation 0 = +0.50 -> S5 becomes 0.65.
        Total = 1.65+1.45+0.05-0.10+0.65 = 3.70.
        """
        reset_group_accumulator()
        score = compute_score(
            "aiops_diagnosis", good_solution_str(), GROUND_TRUTH, {"index": 4321, "alert": ALERT}
        )
        assert score == pytest.approx(3.70)

    def test_reconstructed_trajectory_fields(self):
        extra_info = {"split": "train", "index": 77, "alert_id": "mock-ad", "alert": ALERT}
        traj = trajectory_from_solution_str(good_solution_str(), GROUND_TRUTH, extra_info)

        assert [s.tool_name for s in traj.steps] == ["Bash", "Bash", "Read", "Bash", "Bash"]
        assert [s.hook_decision for s in traj.steps] == ["passthrough", "passthrough", "passthrough", "deny", "allow"]
        allow_step, deny_step = traj.steps[4], traj.steps[3]
        assert (allow_step.hook_action, allow_step.hook_target) == ("scale_resources", "ad")
        assert (deny_step.hook_action, deny_step.hook_target) == ("restart_instance", "product-catalog")
        assert traj.steps[0].observation == "ad   CPU% 99.8   MEM 480MiB / 512MiB"
        assert traj.steps[3].observation == EXPECTED_OBSERVATIONS[3]
        assert traj.steps[0].signal_types_covered == ["kubectl_status"]
        assert traj.steps[1].signal_types_covered == ["prometheus"]
        assert all(not s.is_repeat_no_new_info for s in traj.steps)
        assert traj.route == "auto_remediated"
        assert traj.alert == ALERT
        assert traj.post_action_health is None
        assert traj.final_diagnosis["schema_fallback_triggered"] is False
        assert traj.final_diagnosis["executed_actions"] == [
            {
                "action": "scale_resources",
                "target": "ad",
                "command": "docker update --cpus 2 --memory 1024m ad",
                "ts": None,
            }
        ]


# ---------------------------------------------------------------------------
# TestGroupAccumulator
# ---------------------------------------------------------------------------


class TestGroupAccumulator:
    def test_flushes_exactly_at_group_size(self):
        reset_group_accumulator()
        extra_info = {"split": "train", "index": 42, "alert_id": "mock-ad", "alert": ALERT}
        scores = []
        for _ in range(DEFAULT_GROUP_SIZE - 1):
            scores.append(compute_score("aiops_diagnosis", good_solution_str(), GROUND_TRUTH, extra_info))
        assert _GROUP_ACCUMULATOR.last_group_result(42) is None
        assert len(_GROUP_ACCUMULATOR.pending(42)) == DEFAULT_GROUP_SIZE - 1

        # 6th (degraded) sample completes the group -> compute_group_rewards fires
        scores.append(compute_score("aiops_diagnosis", degraded_solution_str(), GROUND_TRUTH, extra_info))
        result = _GROUP_ACCUMULATOR.last_group_result(42)
        assert result is not None
        assert result["group_size"] == DEFAULT_GROUP_SIZE
        assert result["total_rewards"] == pytest.approx(scores)
        assert result["group_reward_variance"] == pytest.approx(group_reward_variance(scores))
        assert result["group_reward_variance"] > 0
        assert _GROUP_ACCUMULATOR.pending(42) == []
        # variance tracker fed under the kind-based scenario key
        assert _GROUP_ACCUMULATOR.variance_history["resource"] == pytest.approx(
            [result["group_reward_variance"]]
        )
        assert _GROUP_ACCUMULATOR.last_tracker_result["scenario_type"] == "resource"
        assert _GROUP_ACCUMULATOR.last_tracker_result["should_downsample"] is False

        # second epoch: another full group appends a second variance entry
        for _ in range(DEFAULT_GROUP_SIZE):
            compute_score("aiops_diagnosis", good_solution_str(), GROUND_TRUTH, extra_info)
        assert len(_GROUP_ACCUMULATOR.variance_history["resource"]) == 2

    def test_scenario_key_defaults_to_unknown_without_kind(self):
        reset_group_accumulator()
        gt = {"route": "auto_remediated", "remediation_type": "online_op", "suspect_service": "ad"}
        extra_info = {"index": 7, "alert": ALERT}
        for _ in range(DEFAULT_GROUP_SIZE):
            compute_score("aiops_diagnosis", good_solution_str(), gt, extra_info)
        assert _GROUP_ACCUMULATOR.last_tracker_result["scenario_type"] == "unknown"
        assert "unknown" in _GROUP_ACCUMULATOR.variance_history

    def test_missing_index_skips_accumulation_but_still_scores(self):
        reset_group_accumulator()
        score = compute_score("aiops_diagnosis", good_solution_str(), GROUND_TRUTH, {"alert": ALERT})
        assert isinstance(score, float)
        assert score == pytest.approx(3.70)


# ---------------------------------------------------------------------------
# TestRouteInference
# ---------------------------------------------------------------------------


class TestRouteInference:
    @pytest.mark.parametrize(
        "diag,expected",
        [
            (
                {"confidence": 0.55, "remediation_type": "online_op", "executed_actions": [{}], "also_code_fix": True},
                "feishu_low_confidence",
            ),
            (
                {"confidence": 0.82, "remediation_type": "online_op", "executed_actions": [{}], "also_code_fix": True},
                "auto_remediated_and_code_fix_pr",
            ),
            (
                {"confidence": 0.82, "remediation_type": "online_op", "executed_actions": [], "also_code_fix": True},
                "online_op_and_code_fix_pr",
            ),
            (
                {"confidence": 0.82, "remediation_type": "online_op", "executed_actions": [{}], "also_code_fix": False},
                "auto_remediated",
            ),
            (
                {"confidence": 0.82, "remediation_type": "online_op", "executed_actions": [], "also_code_fix": False},
                "feishu_online_op",
            ),
            ({"confidence": 0.82, "remediation_type": "code_fix", "executed_actions": []}, "code_fix_pr"),
            ({"confidence": 0.82, "remediation_type": "info_only", "executed_actions": []}, "info_only"),
            ({"confidence": None, "remediation_type": "info_only"}, "feishu_low_confidence"),
        ],
    )
    def test_optimistic_route_table(self, diag, expected):
        assert infer_route_optimistic(diag) == expected

    def test_hybrid_with_allow_end_to_end(self):
        """also_code_fix=True + an allowed Bash step -> the OPTIMISTIC
        ``auto_remediated_and_code_fix_pr`` (assumes fix.verified; the real
        ``run.py`` could degrade to ``auto_remediated`` — documented
        approximation, see the adapter's module docstring)."""
        diag_fields = {
            **GOOD_DIAG_FIELDS,
            "also_code_fix": True,
            "suspect_repo": "ad",
            "suspect_commit_hint": "abc1234",
            "suspect_file_hint": "src/main.py",
        }
        sol = _render_solution_str(
            _turns_as_xml(EXPECTED_CALLS, EXPECTED_OBSERVATIONS), json.dumps(diag_fields, ensure_ascii=False)
        )
        traj = trajectory_from_solution_str(sol, GROUND_TRUTH, {"index": 99, "alert": ALERT})
        assert traj.route == "auto_remediated_and_code_fix_pr"
        executed = traj.final_diagnosis["executed_actions"]
        assert executed and executed[0]["action"] == "scale_resources"

    def test_hybrid_without_allow_end_to_end(self):
        diag_fields = {**GOOD_DIAG_FIELDS, "also_code_fix": True}
        sol = _render_solution_str(
            _turns_as_xml(EXPECTED_CALLS[:4], EXPECTED_OBSERVATIONS[:4]), json.dumps(diag_fields, ensure_ascii=False)
        )
        traj = trajectory_from_solution_str(sol, GROUND_TRUTH, {"index": 98, "alert": ALERT})
        assert traj.final_diagnosis["executed_actions"] == []
        assert traj.route == "online_op_and_code_fix_pr"


# ---------------------------------------------------------------------------
# TestHookRecompute
# ---------------------------------------------------------------------------


class TestHookRecompute:
    @pytest.mark.parametrize(
        "command,expected_decision,expected_action,expected_target",
        [
            ("docker update --cpus 2 --memory 1024m ad", "allow", "scale_resources", "ad"),
            ("docker restart ad", "allow", "restart_instance", "ad"),
            ("docker restart product-catalog", "deny", "restart_instance", "product-catalog"),
            ("docker stats --no-stream ad", "passthrough", None, None),
        ],
    )
    def test_recompute_branches_and_real_hook_equivalence(
        self, command, expected_decision, expected_action, expected_target
    ):
        """For commands OUTSIDE the hook's fallback deny layer, the
        ``classify_command``-only recompute must agree with the FULL real
        hook (``route_through_real_hook`` runs ``guard_diagnose_ops``
        end-to-end) on decision, action, and target."""
        recomputed = recompute_hook_verdict("Bash", {"command": command})
        assert recomputed == {"decision": expected_decision, "action": expected_action, "target": expected_target}

        real = asyncio.run(
            route_through_real_hook(SampledToolCall("t", "Bash", {"command": command}))
        )
        assert real["decision"] == recomputed["decision"]
        assert real["action"] == recomputed["action"]
        assert real["target"] == recomputed["target"]

    def test_non_bash_tools_are_passthrough(self):
        for tool_name, tool_input in [
            ("Read", {"file_path": "/repo/ad/src/main.py"}),
            ("Grep", {"pattern": "cache", "path": "/repo"}),
            ("mcp__aiops_rag__search_past_incidents", {"query": "ad cpu"}),
        ]:
            assert recompute_hook_verdict(tool_name, tool_input) == {
                "decision": "passthrough",
                "action": None,
                "target": None,
            }

    def test_fallback_deny_layer_is_a_documented_approximation(self):
        """``docker exec ...`` is passthrough under ``classify_command`` but
        DENIED by the real hook's ``_READONLY_DENY`` layer. The adapter
        reconstructs it as passthrough — a documented known approximation
        (adapter module docstring), pinned here so any change to it is a
        deliberate decision, not an accident."""
        recomputed = recompute_hook_verdict("Bash", {"command": "docker exec -it ad top"})
        assert recomputed["decision"] == "passthrough"
        real = asyncio.run(
            route_through_real_hook(SampledToolCall("t", "Bash", {"command": "docker exec -it ad top"}))
        )
        assert real["decision"] == "deny"


# ---------------------------------------------------------------------------
# TestQwen3XmlParsing
# ---------------------------------------------------------------------------


class TestQwen3XmlParsing:
    def test_canonical_tool_call_block(self):
        block = (
            "<tool_call>\n<function=Bash>\n<parameter=command>\ndocker ps\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        assert parse_tool_call_block(block) == {"tool_name": "Bash", "tool_input": {"command": "docker ps"}}

    def test_multi_parameter_block(self):
        block = (
            "<tool_call>\n<function=Read>\n<parameter=file_path>\n/a/b.py\n</parameter>\n"
            "<parameter=offset>\n10\n</parameter>\n</function>\n</tool_call>"
        )
        assert parse_tool_call_block(block) == {
            "tool_name": "Read",
            "tool_input": {"file_path": "/a/b.py", "offset": "10"},
        }

    def test_whitespace_tolerant_parameter_tags(self):
        """vLLM's own ``_PARAM_RE`` tolerates ``< parameter = k >`` /
        ``</ parameter >`` spacing; the port keeps that tolerance."""
        block = (
            "<tool_call>\n<function=Bash>\n< parameter = command >\ndocker ps\n"
            "</ parameter >\n</function>\n</tool_call>"
        )
        assert parse_tool_call_block(block) == {"tool_name": "Bash", "tool_input": {"command": "docker ps"}}

    def test_param_value_keeps_inner_newlines_trims_wrapping_ones(self):
        block = (
            "<tool_call>\n<function=Bash>\n<parameter=script>\nline1\nline2\n"
            "</parameter>\n</function>\n</tool_call>"
        )
        assert parse_tool_call_block(block)["tool_input"]["script"] == "line1\nline2"

    def test_malformed_block_without_function_returns_none(self):
        assert parse_tool_call_block("<tool_call>\n</tool_call>") is None

    def test_full_parse_pairs_calls_and_responses_in_order(self):
        parsed = parse_solution_str(good_solution_str())
        assert parsed.tool_calls == EXPECTED_CALLS
        assert parsed.observations == EXPECTED_OBSERVATIONS
        assert parsed.n_dropped_malformed_calls == 0
        assert "suspect_service" in parsed.final_text

    def test_multiple_calls_in_one_turn_pair_in_order(self):
        calls_xml = (
            _tool_call_xml("Bash", command="docker stats --no-stream ad")
            + "\n"
            + _tool_call_xml("Read", file_path="/repo/ad/src/main.py")
        )
        sol = (
            calls_xml
            + "\nuser\n<tool_response>\nobs-1\n</tool_response>\n"
            + "<tool_response>\nobs-2\n</tool_response>\nassistant\n最终结论。"
        )
        parsed = parse_solution_str(sol)
        assert [c["tool_name"] for c in parsed.tool_calls] == ["Bash", "Read"]
        assert parsed.observations == ["obs-1", "obs-2"]
        assert parsed.final_text == "最终结论。"

    def test_truncated_rollout_missing_observation_and_final(self):
        sol = _tool_call_xml("Bash", command="docker stats --no-stream ad")
        parsed = parse_solution_str(sol)
        assert len(parsed.tool_calls) == 1
        assert parsed.observations == [""]
        assert parsed.final_text == ""
        traj = trajectory_from_solution_str(sol, GROUND_TRUTH, {"index": 55, "alert": ALERT})
        assert traj.steps[0].observation == ""
        # no final text -> schema fallback convention kicks in
        assert traj.final_diagnosis["schema_fallback_triggered"] is True
        assert traj.route == "feishu_low_confidence"

    def test_think_and_role_words_stripped_from_final_text(self):
        sol = 'assistant\n<think>\n推理中\n</think>\n\n最终结论：\n{"confidence": 0.9}'
        parsed = parse_solution_str(sol)
        assert parsed.final_text == '最终结论：\n{"confidence": 0.9}'

    def test_unclosed_think_block_drops_everything_after(self):
        parsed = parse_solution_str("<think>\n没写完的推理")
        assert parsed.final_text == ""
        assert parsed.tool_calls == []


# ---------------------------------------------------------------------------
# TestFinalDiagnosisExtraction
# ---------------------------------------------------------------------------


class TestFinalDiagnosisExtraction:
    def test_fenced_json(self):
        text = '结论如下：\n```json\n{"confidence": 0.9, "remediation_type": "info_only"}\n```'
        assert extract_last_json_object(text) == {"confidence": 0.9, "remediation_type": "info_only"}

    def test_prose_around_json_and_braces_inside_strings(self):
        text = '经排查 {"summary": "重启 {ad} 容器后恢复"} 以上。'
        assert extract_last_json_object(text) == {"summary": "重启 {ad} 容器后恢复"}

    def test_last_of_multiple_objects_wins(self):
        text = '{"first": 1} 然后 {"second": 2}'
        assert extract_last_json_object(text) == {"second": 2}

    def test_no_json_returns_none(self):
        assert extract_last_json_object("纯文本结论，没有 JSON。") is None

    def test_unparseable_final_text_triggers_schema_fallback_convention(self):
        traj = trajectory_from_solution_str(degraded_solution_str(), GROUND_TRUTH, {"index": 5, "alert": ALERT})
        assert traj.final_diagnosis["schema_fallback_triggered"] is True
        assert traj.final_diagnosis["confidence"] == 0.0
        assert traj.route == "feishu_low_confidence"
        # NOTE (existing reward/ semantics, observed not changed here): the
        # fallback Diagnosis's synthetic evidence string is by construction
        # untraceable to any observation, so the -0.5 fabrication penalty
        # fires ON TOP of the -0.2 schema-fallback penalty. Hand-computed
        # under MARGINAL credit assignment (kind=config weights kubectl_status
        # 0.8, prometheus 0.7; each signal type paid once on its first-cover
        # step): step+credit per-step totals 0.95 + 0.85 + 0.05 + -0.10 + -0.55
        # = 1.20, terminal outcome 0 -> total 1.20.
        assert compute_trajectory_reward(traj)["total_reward"] == pytest.approx(1.20)

    def test_no_tool_calls_direct_answer(self):
        sol = "<think>\n直接给结论\n</think>\n\n" + json.dumps(GOOD_DIAG_FIELDS, ensure_ascii=False)
        traj = trajectory_from_solution_str(sol, GROUND_TRUTH, None)
        assert traj.steps == []
        assert traj.route == "feishu_online_op"  # online_op, no executed_actions, no also_code_fix
        breakdown = compute_trajectory_reward(traj)
        assert len(breakdown["per_step_rewards"]) == 1  # zero-step synthetic element


def test_module_loadable_by_file_path_in_isolated_interpreter():
    """veRL loads the custom reward function BY FILE PATH with no package
    context (verified: ``verl/utils/import_utils.py::load_module`` ->
    ``importlib.util.spec_from_file_location``, unregistered in
    ``sys.modules``). Simulate exactly that in a clean subprocess — neutral
    cwd (so the repo root is NOT on ``sys.path``) and ``PYTHONPATH`` cleared
    — and confirm ``compute_score`` still works. This guards both of the
    module's bootstrap defenses: the repo-root ``sys.path`` insert (without
    which the sibling ``reward``/``grpo`` imports fail) and the
    ``sys.modules`` self-registration (without which ``@dataclass``
    definition crashes — verified to reproduce on Python 3.9 and 3.11
    under exactly veRL's loading mode)."""
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    adapter_path = Path(__file__).resolve().parents[1] / "verl_reward_adapter.py"
    script = (
        "import importlib.util, sys\n"
        "path = sys.argv[1]\n"
        "assert not any(p.endswith('aiops-agentic-rl') for p in sys.path), sys.path\n"
        "spec = importlib.util.spec_from_file_location('custom_module_test', path)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)  # exactly veRL's load_module (unregistered)\n"
        "assert 'custom_module_test' in sys.modules  # the module self-registered\n"
        "sol = (\n"
        "    '<tool_call>\\n<function=Bash>\\n<parameter=command>\\ndocker stats --no-stream ad\\n"
        "</parameter>\\n</function>\\n</tool_call>\\n'\n"
        "    'user\\n<tool_response>\\nad CPU% 99.8\\n</tool_response>\\n'\n"
        "    'assistant\\n'\n"
        "    '{\"summary\": \"s\", \"kind\": \"resource\", \"suspect_service\": \"ad\", "
        "\"remediation_type\": \"info_only\", \"remediation_detail\": \"d\", "
        "\"confidence\": 0.9, \"evidence\": [\"ad CPU% 99.8\"]}'\n"
        ")\n"
        "score = mod.compute_score(\n"
        "    data_source='aiops_diagnosis', solution_str=sol,\n"
        "    ground_truth={}, extra_info={'index': 1, 'alert': {}},\n"
        ")\n"
        "assert isinstance(score, float), repr(score)\n"
        "print(score)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", script, str(adapter_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tempfile.gettempdir(),
        env=env,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"
    assert float(proc.stdout.strip()) == pytest.approx(1.45)
