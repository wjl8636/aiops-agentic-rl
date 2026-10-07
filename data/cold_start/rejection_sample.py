"""拒绝采样过滤器（design doc §4.3；2026-09-08 起含 kind 校验；2026-09 v9 Phase C 起含 §7.2 扩展
schema 的 8 项新校验，见 `docs/数据增强方案.md` §4 各家族验收规则）。

对 `--traj-dir` 下每条收集到的轨迹，按 alert_id 去 `--gt-dir` 找对应的人工标注 ground truth
（`{"remediation_type":..., "suspect_service":..., "expect_route":...,
"expect_kind_any_of": [...]}`），只保留同时满足：
  1. `Diagnosis.remediation_type` 与人工标注一致（v9：或落在 `expect_remediation_any_of` 内）
  2. `Diagnosis.suspect_service` 与人工标注一致（子串匹配，口径对齐 AIops-agent 自带的
     `eval/expected.json` 里 `suspect_service_contains` 的写法）
  3. route 判定与人工标注一致（v9：或落在 `expect_route_any_of` 内）——例外：当期望路由属于
     「依赖代码修复 Agent 验证结果」
     的路由（`code_fix_pr`/`feishu_fix_unverified`/`auto_remediated_and_code_fix_pr`/
     `online_op_and_code_fix_pr`）且轨迹里的 `route` 确实是 None 时（真实模式下
     `collect_trajectories.py::_infer_route_real()` 没跑代码修复 Agent、不知道
     `fix.verified`，如实留空），不对 route 做精确匹配，只校验（当
     `remediation_type=="online_op"` 时）`also_code_fix` 判断是否与人工标注一致；
     若轨迹的 `route` 不是 None（例如 mock 数据用 `_infer_route()` 假设「代码修复一定验证
     通过」算出的确定值），仍然按精确匹配处理
  4. `Diagnosis.evidence` 里每一条声明都能在轨迹的工具调用观测里找到出处
  5. `executed_actions` 与轨迹里 hook 真正放行（`hook_decision=="allow"`）的记录一致
  6. `Diagnosis.kind` 落在 `expect_kind_any_of` 集合内（2026-09-08 新增，修 kind 0.68 的
     直接一刀；GT 缺该字段时跳过本校验，向后兼容旧标注）。集合口径对齐
     `AIops-agent/eval/expected.json` 的 `kind_any_of` 写法：机制上只有唯一正确答案的
     （kafka 积压=resource、内存泄漏/逻辑 bug fixture=deploy_regression）单值严格；
     eval 对同机制场景明确接受多个 kind 的（如 s1 的 [dependency, config]）按 eval 口径放行，
     列表首位是 `docs/数据增强方案.md` §7.3 决策表的 canonical 值。

v9 新增（§7.2 扩展 schema；全部「GT 有字段才启用、没有就跳过」，向后兼容）：
  7. `expect_confidence_range [lo, hi]`：confidence 必须落在区间内（A1/B1a/B1b/B2/B3/C1/D 校准卡口）
  8. `expect_max_steps N`：剔除 malformed 步后的步数 ≤ N（C1 ≤8 / A1 ≤10）
  9. `expect_suspect_file_any_of [...]`：suspect_file_hint 命中任一文件名（子串、大小写不敏感，
     对齐 eval 的 expect_changed_files_any_of）——D 家族 code_fix 交接字段
 10. `expect_negative_claim {"object":..., "markers":[...]}`：summary/evidence 必须出现 markers
     里至少一条否定性/如实措辞，且（若指定 object）提及该对象——A1 噪声措辞 / A2 对不存在的
     第二症状如实否定 / B1b 未能定位措辞 / B3 已自愈措辞
 11. `expect_evidence_groups [[tok...],[tok...]]`：每个 token 组都被至少一条 evidence 覆盖
     ——B1a 用它卡「flag 证据（OFREP）与至少一路非 Prometheus 旁证并存」
 12. `expect_detail_contains_any [...]`：remediation_detail 必含任一措辞——A3 的「交人工」语义
 13. `expect_remediation_any_of` / `expect_route_any_of`：列表式放行（B1b 低置信降级下
     remediation_type 本就不确定）

的轨迹。design doc 原话是「跟 reward 模块复用同一套可追溯性校验逻辑」，但这里**不 import
`reward/*`**（并行任务、这次会话跑的时候可能还没建好那个模块），而是自己重新实现一份——
逻辑刻意写得很直白（正则抓「有信息量的标识符/数字」再算重合度），保证不需要看 reward 模块
的实现就能独立复核对不对。

证据可追溯性判定方法：把 evidence 字符串和每一步的 `observation + tool_input` 都跑一遍
`_significant_tokens()`（抓大小写字母数字标识符/百分数/版本号这类「事实性 token」，
中文自然语言部分故意不参与比对——它们是转述用的连接词，比不比对不影响事实性判断），
只要某一步跟这条 evidence 重合的「事实性 token」数量达到 `min(2, evidence 自身 token 数)`，
就认为这条 evidence 有出处。这个规则足够简单、可以肉眼复核，也确实能把「编造的证据」
（跟任何一步都没有事实重合）和「真实转述的证据」（跟某一步共享服务名/百分比/commit id
这类事实 token）区分开——但它只是一个启发式，不追求语义级别的严谨。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_TRAJ_DIR = HERE / "trajectories" / "mock"
DEFAULT_GT_DIR = HERE / "ground_truth"

# 抓「有信息量的标识符」：字母数字开头，后面允许字母数字/下划线/点/百分号/短横线延续。
# 故意不匹配纯中文——中文部分是转述连接词，事实性内容（服务名/指标/commit id/百分比）
# 在这份合成数据里都是用 ASCII/数字写的，用它们做重合度判定既简单又足够准。
_IDENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%-]*")

# 这些 route 的真实取值依赖代码修复 Agent 跑完之后的 `fix.verified`（True/False），而本脚本
# （`collect_trajectories.py::_infer_route_real()`）不跑代码修复 Agent，如实把 route 留空
# （None）。对这些 route，不能拿 None 去跟 ground truth 里预设的 expect_route 做精确匹配
# ——那样只会让「诊断质量再好」的样本也因为一个从未被计算过的字段被无差别拒绝。
_CODE_FIX_DEPENDENT_ROUTES = {
    "code_fix_pr",
    "feishu_fix_unverified",
    "auto_remediated_and_code_fix_pr",
    "online_op_and_code_fix_pr",
}


def _significant_tokens(text: str) -> set[str]:
    return {t.lower() for t in _IDENT_RE.findall(text or "") if len(t) >= 2}


def is_evidence_traceable(evidence: str, steps: list[dict[str, Any]]) -> bool:
    """这条 evidence 字符串能不能在某一步的观测/调用里找到事实性出处。"""
    ev_tokens = _significant_tokens(evidence)
    if not ev_tokens:
        # evidence 里没有任何可比对的标识符（纯自然语言断言）——退化成子串包含检查。
        stripped = evidence.strip()
        return bool(stripped) and any(stripped in (s.get("observation", "") or "") for s in steps)
    need = min(2, len(ev_tokens))
    for step in steps:
        haystack = f"{step.get('observation', '')} {json.dumps(step.get('tool_input', {}), ensure_ascii=False)}"
        hay_tokens = _significant_tokens(haystack)
        if len(ev_tokens & hay_tokens) >= need:
            return True
    return False


def evidence_traceability_ok(diagnosis: dict[str, Any], steps: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    bad = [ev for ev in diagnosis.get("evidence", []) if not is_evidence_traceable(ev, steps)]
    return (len(bad) == 0, bad)


def executed_actions_match_ledger(steps: list[dict[str, Any]], executed_actions: list[dict[str, Any]]) -> bool:
    """`Diagnosis.executed_actions` 必须跟轨迹里真正被 hook 放行（allow）的那些步骤一一对应。

    只信「hook 真的放行过」这份台账，不信 Diagnosis 里 executed_actions 字段本身
    （模型完全可能在 executed_actions 里多写/少写/编一个从没被放行过的操作）。
    """
    ledger = {
        (s.get("hook_action"), s.get("hook_target"), (s.get("tool_input") or {}).get("command", ""))
        for s in steps
        if s.get("hook_decision") == "allow"
    }
    claimed = {(a.get("action"), a.get("target"), a.get("command", "")) for a in executed_actions}
    return ledger == claimed


def suspect_service_matches(actual: str, expected: str) -> bool:
    """口径对齐 AIops-agent/eval/expected.json 的 `suspect_service_contains`：子串、大小写不敏感。"""
    return bool(expected) and expected.strip().lower() in (actual or "").strip().lower()


def expect_kind_set(gt: dict[str, Any]) -> list[str] | None:
    """从 GT 里取期望 kind 集合。兼容两种写法：`expect_kind_any_of`（列表，本仓库标准）
    和 `expect_kind`（单值，等价于长度为 1 的集合）。都没有时返回 None（跳过 kind 校验，
    向后兼容还没回填的旧标注/mock）。"""
    any_of = gt.get("expect_kind_any_of")
    if isinstance(any_of, list) and any_of:
        return [str(k) for k in any_of]
    single = gt.get("expect_kind")
    if isinstance(single, str) and single:
        return [single]
    return None


# ---------------------------------------------------------------------------
# 2026-09 v9 扩展校验（docs/数据增强方案.md §7.2 + §4 各家族验收规则）。
# 全部遵循同一约定：GT 里没有对应字段就跳过该校验（向后兼容），有就严格执行。
# ---------------------------------------------------------------------------


def _confidence_in_range(diag: dict[str, Any], rng: Any) -> tuple[bool, str]:
    lo, hi = float(rng[0]), float(rng[1])
    conf = diag.get("confidence")
    ok = isinstance(conf, (int, float)) and lo <= float(conf) <= hi
    return ok, f"confidence={conf!r} 不在期望区间 [{lo}, {hi}]"


def _suspect_file_hits(diag: dict[str, Any], files: list[str]) -> tuple[bool, str]:
    """`suspect_file_hint` 命中任一期望文件名（子串、大小写不敏感）——口径对齐
    `AIops-agent/eval/expected.json` 的 `expect_changed_files_any_of`。"""
    hint = (diag.get("suspect_file_hint") or "")
    hit = next((f for f in files if f.lower() in hint.lower()), None)
    return hit is not None, f"suspect_file_hint={hint!r} 未命中 expect_suspect_file_any_of={files!r}"


def _steps_within_max(traj: dict[str, Any], max_steps: int) -> tuple[bool, str]:
    """步数上限按「剔除 malformed 步骤后的步数」计——跟 prefix_split 的口径一致
    （malformed 步本来就不该成为目标）。不 import prune_steps（它反向 import 本模块，
    虽然函数内延迟 import 不会真的成环，但内联这两行更简单）。"""

    def _malformed(step: dict[str, Any]) -> bool:
        return "__unparsedToolInput" in json.dumps(step.get("tool_input") or {}, ensure_ascii=False)

    n = len([s for s in (traj.get("steps") or []) if not _malformed(s)])
    return n <= max_steps, f"步数 {n} 超过 expect_max_steps={max_steps}"


def _negative_claim_present(diag: dict[str, Any], claim: dict[str, Any]) -> tuple[bool, str]:
    """summary/evidence 里出现否定性措辞（且提及 GT 指定的对象，若有）。

    A2（部分证据不外推）：markers 如「未观测到/不成立/off」+ object=被声称但不存在的 flag/服务名。
    A1（空观测如实上报）：markers 如「噪声/误报/无活跃」，object=None。
    B1b（证据拿不到）：markers 如「未能定位/无法确认」，object=None。
    B3（陈旧已自愈）：markers 如「已恢复/已自愈」，object=None。
    """
    markers = [str(m) for m in (claim.get("markers") or [])]
    obj = claim.get("object")
    text = " ".join(
        [str(diag.get("summary") or "")] + [str(e) for e in (diag.get("evidence") or [])]
    )
    low = text.lower()
    hit_marker = next((m for m in markers if m.lower() in low), None)
    if hit_marker is None:
        return False, f"summary/evidence 缺少否定性/如实措辞（markers={markers!r} 一个都没出现）"
    if obj and str(obj).lower() not in low:
        return False, f"否定性措辞存在（{hit_marker!r}）但未提及期望对象 {obj!r}"
    return True, ""


def _evidence_groups_covered(diag: dict[str, Any], groups: list[list[str]]) -> tuple[bool, str]:
    """每个 token 组都必须被至少一条 evidence 覆盖（大小写不敏感子串）。

    B1a 用它卡「evidence 必须同时含 flag 证据（OFREP/开关名）和至少一路非 Prometheus
    旁证（docker logs/Jaeger/日志等）」：group1=flag 证据 token，group2=第二信源 token。
    """
    evidences = [str(e) for e in (diag.get("evidence") or [])]
    low_ev = [e.lower() for e in evidences]
    missing = []
    for gi, group in enumerate(groups):
        tokens = [str(t) for t in group]
        if not any(any(t.lower() in ev for t in tokens) for ev in low_ev):
            missing.append((gi, tokens))
    if missing:
        return False, f"evidence 未覆盖的信源组: {missing!r}"
    return True, ""


def _detail_contains_any(diag: dict[str, Any], needles: list[str]) -> tuple[bool, str]:
    detail = str(diag.get("remediation_detail") or "")
    hit = next((n for n in needles if n.lower() in detail.lower()), None)
    return hit is not None, f"remediation_detail 未包含 expect_detail_contains_any={needles!r} 中任何一个"


def check_trajectory(traj: dict[str, Any], gt: dict[str, Any]) -> dict[str, Any]:
    """对一条轨迹跑全部 6 项拒绝采样条件，返回 {"keep": bool, "reasons": [...]}。

    `reasons` 为空列表 <=> keep=True；非空列表里每一条都是一个独立的拒绝原因
    （故意不做「命中第一条就短路返回」，方便调试时一次看到所有没过的条件）。
    """
    diag = traj.get("final_diagnosis") or {}
    steps = traj.get("steps") or []
    reasons: list[str] = []

    kind_any_of = expect_kind_set(gt)
    if kind_any_of is not None:
        actual_kind = (diag.get("kind") or "").strip()
        if actual_kind not in kind_any_of:
            reasons.append(
                f"kind 不在期望集合: got={actual_kind!r} expect_any_of={kind_any_of!r}"
            )

    # remediation_type：精确匹配（默认）或 expect_remediation_any_of 任一命中（B1b 低置信家族
    # 的 remediation_type 本身不确定，置信度才是验收锚点）
    remediation_any_of = gt.get("expect_remediation_any_of")
    if isinstance(remediation_any_of, list) and remediation_any_of:
        if diag.get("remediation_type") not in remediation_any_of:
            reasons.append(
                f"remediation_type 不一致: got={diag.get('remediation_type')!r}"
                f" expect_any_of={remediation_any_of!r}"
            )
    elif diag.get("remediation_type") != gt.get("remediation_type"):
        reasons.append(
            f"remediation_type 不一致: got={diag.get('remediation_type')!r} expect={gt.get('remediation_type')!r}"
        )

    # confidence 区间（v9 新增）
    conf_range = gt.get("expect_confidence_range")
    if isinstance(conf_range, list) and len(conf_range) == 2:
        ok, why = _confidence_in_range(diag, conf_range)
        if not ok:
            reasons.append(f"confidence 越界: {why}")

    # 步数上限（v9 新增，C1/A1 效率家族）
    max_steps = gt.get("expect_max_steps")
    if isinstance(max_steps, int):
        ok, why = _steps_within_max(traj, max_steps)
        if not ok:
            reasons.append(why)

    # suspect_file_hint 命中（v9 新增，code_fix 家族交接字段）
    suspect_files = gt.get("expect_suspect_file_any_of")
    if isinstance(suspect_files, list) and suspect_files:
        ok, why = _suspect_file_hits(diag, [str(f) for f in suspect_files])
        if not ok:
            reasons.append(why)

    # 否定性/如实措辞（v9 新增，A1/A2/B1b/B3 反幻觉家族）
    negative_claim = gt.get("expect_negative_claim")
    if isinstance(negative_claim, dict) and negative_claim.get("markers"):
        ok, why = _negative_claim_present(diag, negative_claim)
        if not ok:
            reasons.append(f"缺少否定性/如实陈述: {why}")

    # evidence 信源组覆盖（v9 新增，B1a 交叉验证家族：flag 证据 + 非 Prometheus 旁证并存）
    evidence_groups = gt.get("expect_evidence_groups")
    if isinstance(evidence_groups, list) and evidence_groups:
        ok, why = _evidence_groups_covered(diag, evidence_groups)
        if not ok:
            reasons.append(f"evidence 信源覆盖不足: {why}")

    # remediation_detail 必含措辞（v9 新增，A3 台账纪律：交人工语义）
    detail_any = gt.get("expect_detail_contains_any")
    if isinstance(detail_any, list) and detail_any:
        ok, why = _detail_contains_any(diag, [str(n) for n in detail_any])
        if not ok:
            reasons.append(why)

    if not suspect_service_matches(diag.get("suspect_service", ""), gt.get("suspect_service", "")):
        reasons.append(
            f"suspect_service 不一致: got={diag.get('suspect_service')!r} expect_contains={gt.get('suspect_service')!r}"
        )

    # route：精确匹配（默认）/ expect_route_any_of 任一命中（eval 的 expected.json 同款写法）。
    # code_fix 依赖路由 + route=None 的例外逻辑对两种写法同样生效。
    expect_route = gt.get("expect_route")
    route_any_of = gt.get("expect_route_any_of")
    actual_route = traj.get("route")
    code_fix_dependent = (
        expect_route in _CODE_FIX_DEPENDENT_ROUTES
        or (isinstance(route_any_of, list) and any(r in _CODE_FIX_DEPENDENT_ROUTES for r in route_any_of))
    )
    if code_fix_dependent and actual_route is None:
        # route 依赖代码修复 Agent 验证结果，且本条轨迹的 route 确实是「诚实留空」的 None
        # （真实模式下 `collect_trajectories.py::_infer_route_real()` 没跑代码修复 Agent，
        # 不知道 `fix.verified`，如实返回 None）——不能拿 None 去跟 expect_route 精确匹配，
        # 否则只要诊断牵出代码修复就会被无差别拒绝，跟诊断质量无关。改为校验诊断 Agent
        # 有没有正确判定「这个要不要转代码修复」。
        # 注意：如果 actual_route 不是 None（例如 mock 模式的 `_infer_route()` 故意假设
        # 「代码修复一定验证通过」、给出一个确定值），说明这条轨迹的 route 是可比对的，
        # 走下面 else 分支做精确匹配，不落入这个例外。
        if gt.get("remediation_type") == "online_op":
            expect_also_code_fix = gt.get("expect_also_code_fix")
            if diag.get("also_code_fix") != expect_also_code_fix:
                reasons.append(
                    f"also_code_fix 判断错误: got={diag.get('also_code_fix')!r} expect={expect_also_code_fix!r}"
                )
    elif isinstance(route_any_of, list) and route_any_of and expect_route is None:
        if actual_route not in route_any_of:
            reasons.append(f"route 不一致: got={actual_route!r} expect_any_of={route_any_of!r}")
    elif actual_route != expect_route:
        reasons.append(f"route 不一致: got={actual_route!r} expect={expect_route!r}")

    ok, bad_evidence = evidence_traceability_ok(diag, steps)
    if not ok:
        reasons.append(f"evidence 无法追溯到工具观测出处: {bad_evidence!r}")

    if not executed_actions_match_ledger(steps, diag.get("executed_actions") or []):
        reasons.append("executed_actions 与轨迹里 hook 放行台账不一致")

    return {"keep": len(reasons) == 0, "reasons": reasons}


def run(traj_dir: Path, gt_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for f in sorted(Path(traj_dir).glob("*.json")):
        traj = json.loads(f.read_text(encoding="utf-8"))
        alert_id = traj.get("alert_id") or f.stem
        gt_path = Path(gt_dir) / f"{alert_id}.json"
        if not gt_path.exists():
            results[alert_id] = {"keep": False, "reasons": [f"未找到人工标注 ground truth: {gt_path}"]}
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        results[alert_id] = check_trajectory(traj, gt)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="对冷启轨迹跑拒绝采样过滤")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR, help="轨迹 JSON 所在目录")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR, help="人工标注 ground truth 所在目录")
    parser.add_argument("--copy-kept-to", type=Path, default=None, help="可选：把通过的轨迹复制到这个目录")
    args = parser.parse_args()

    results = run(args.traj_dir, args.gt_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    kept = [k for k, v in results.items() if v["keep"]]
    rejected = [k for k, v in results.items() if not v["keep"]]
    total = len(results)
    print(f"\n通过 {len(kept)}/{total}: {kept}")
    print(f"拒绝 {len(rejected)}/{total}: {rejected}")

    if args.copy_kept_to:
        import shutil

        args.copy_kept_to.mkdir(parents=True, exist_ok=True)
        for alert_id in kept:
            src = Path(args.traj_dir) / f"{alert_id}.json"
            shutil.copy(src, args.copy_kept_to / src.name)
        print(f"已把 {len(kept)} 条通过的轨迹复制到 {args.copy_kept_to}")


if __name__ == "__main__":
    main()
