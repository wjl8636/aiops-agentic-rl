"""步级修剪（docs/数据增强方案.md §4.3 C2 / §4.5 / §6.3 第 4 步）。

对 `--traj-dir` 下的每条轨迹做步级修剪，把修剪后的副本写到 `--out-dir`（原文件不动）。
v8 实测均值 13.4 步/条，目标 ≤11 步；每删一步同时减少多条前缀子样本的训练噪声。

修剪规则（按优先级，全部只针对「只读/无副作用」步骤）：

  R1 marked-repeat：`signals.annotate_steps` 采集时标记的 `is_repeat_no_new_info=true` 步骤
     （同一调用 key 重复、信号覆盖没有扩大）。
  R2 noise：TaskStop / KillShell / KillBash 等中断类工具调用；`find /` 全盘扫描类 Bash 命令
     （采集环境 cwd 固定，全盘 find 只会产出巨大噪声观测）。
  R3 malformed：`StructuredOutput` 工具调用里带 `__unparsedToolInput` 的 schema/JSON 解析
     失败重试步（SDK 在工具输入不合法时把这个原始字符串挂在 `__unparsedToolInput` 键上，
     这正是 v8 里 seed_0381 第 9 步混成训练目标的那类步骤）——保留成功的那一次。
  R4 skeleton-repeat：对 Bash 步骤按「命令骨架」（见 `_bash_skeleton`）归组；同骨架的后续
     调用若 ① 覆盖的信号类型没有比先前同骨架调用更多（复用 `signals.classify_signal_types`
     的口径），且 ② 观测里的事实性 token（复用 `rejection_sample._significant_tokens` 的
     口径）没有超出先前同骨架观测的并集——即「同一件事再查了一遍、什么新东西都没拿到」
     （典型形态：`docker logs X --tail 100` 之后又 `docker logs X --since 30m` 拿到空输出；
     同一 PromQL 再查一遍拿到空结果集）。R4 比 R1 敏感：R1 的 key 是完整命令字符串，
     `--tail 100` vs `--since 30m` 这种参数级变体会漏掉，R4 把它们归到同一骨架。

安全网（设计文档 C2 第 4 条「不过就回退该步」）：
  * `hook_decision=="allow"` 的步骤（真正执行过的处置动作）永远不修剪——`executed_actions`
    台账一致性依赖它们；
  * 成功的 `StructuredOutput` 终止步永远不修剪；
  * 每一步候选修剪都立即复验 `rejection_sample.evidence_traceability_ok`（evidence 每一条
    仍能在剩余步骤的观测里找到出处）+ `rejection_sample.executed_actions_match_ledger`
    （台账仍一致），复验不过就回退该步——保证修剪不会把 evidence 引用的观测步骤删掉。

每个修剪后轨迹的 `meta` 里追加 `pruning` 字段记录删了哪些步、为什么（原步序号 1-based +
原因），保证修剪可追溯、可回滚（原文件未动，重跑本脚本可复现）。

用法：
    python3 prune_steps.py --traj-dir data/cold_start/trajectories_accepted_v8 \
        --out-dir data/cold_start/trajectories_governed_v9
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import rejection_sample as rs
from signals import classify_signal_types

HERE = Path(__file__).resolve().parent
DEFAULT_TRAJ_DIR = HERE / "trajectories_accepted_v8"
DEFAULT_OUT_DIR = HERE / "trajectories_governed_v9"

#: 中断/终止类工具调用，对诊断证据链没有任何贡献
_NOISE_TOOL_NAMES = {"TaskStop", "KillShell", "KillBash", "Task"}

#: `find / ...` 全盘扫描（命令以 find / 开头，或作为管道/复合命令的一段出现；
#: find 前面必须是行首/空白/命令分隔符，避免误伤 myfind 之类的普通单词）
_FIND_ROOT_RE = re.compile(r"(?:^|[\s;&|])find\s+/")

#: 「空观测」的固定样板文案（SDK/采集层写的，不是模型产出的信息）——做事实 token
#: 比对前先归一成空串，否则 "(Bash completed with no output)" 自带的英文 token
#: 会让「重复查询且拿到空输出」这一最典型的 R4 形态漏判。
_NO_OUTPUT_OBS_RE = re.compile(r"^\((?:Bash completed with no output|No content)\)$|^Exit code \d+$|^---$")

#: 空结果集的 JSON 响应（Jaeger/Prometheus 的 "查了但没有数据"）——同样视为无信息。
_EMPTY_RESULT_OBS_RE = re.compile(r'"(?:data|result|traces)"\s*:\s*\[\s*\]|"(?:total|errors)"\s*:\s*0\b')

#: 观测里不构成「新信息」的 token：时间戳片段（00.131695275z / 2026-08-18t08 / 12:34:56）、
#: 长十六进制哈希（容器/镜像/trace id）、IP 地址。同一查询换个时刻重跑，观测里新出现的
#: 全是这类 token——它们是噪声，不是新事实。
_NOISE_TOKEN_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}t\d{2}"  # 2026-08-18t08（ISO 时间戳被 _IDENT_RE 切开后的碎片）
    r"|^\d{2}:\d{2}:\d{2}"  # 12:34:56
    r"|^\d+(\.\d+)?z$"  # 00.131695275z
    r"|^[0-9a-f]{12,}$"  # sha256 / 容器 id / trace id 片段
    r"|^\d{1,3}(\.\d{1,3}){3}$"  # IP
)


def _is_no_info_observation(observation: str) -> bool:
    """这条观测是不是「查了但什么都没拿到」（空样板/空结果集/纯分隔符）。"""
    obs = (observation or "").strip()
    if not obs:
        return True
    return bool(_NO_OUTPUT_OBS_RE.match(obs) or _EMPTY_RESULT_OBS_RE.search(obs))


def _obs_facts(observation: str) -> set[str]:
    """观测里的事实性 token（空样板/空结果集归一成空集；时间戳/哈希/IP 这类
    「重跑必然变化的噪声 token」过滤掉，剩下的才是判断有没有新信息的依据）。"""
    if _is_no_info_observation(observation):
        return set()
    return {
        t
        for t in rs._significant_tokens(observation)
        if not _NOISE_TOKEN_RE.match(t)
    }


def is_malformed_step(step: dict[str, Any]) -> bool:
    """StructuredOutput 的 schema/JSON 解析失败重试步：SDK 会把解析不出的原始输入挂在
    `__unparsedToolInput` 键上。识别方式就是找这个标记键（不猜语义）。"""
    return "__unparsedToolInput" in json.dumps(step.get("tool_input") or {}, ensure_ascii=False)


def is_noise_step(step: dict[str, Any]) -> bool:
    if step.get("tool_name") in _NOISE_TOOL_NAMES:
        return True
    cmd = str((step.get("tool_input") or {}).get("command", "") or "")
    return bool(_FIND_ROOT_RE.search(" " + cmd.strip() + " ;"))


def _strip_cd_prefix(cmd: str) -> str:
    """`cd <dir> && rest` / `cd <dir>; rest` -> `rest`（同一目录下的变体命令归到一起）。"""
    m = re.match(r"^\s*cd\s+\S+\s*(&&|;)\s*", cmd)
    return cmd[m.end():] if m else cmd


def _bash_skeleton(cmd: str) -> str:
    """把一条 Bash 命令归到它的「查询骨架」：骨架相同 = 在查同一件事。

    只对采集环境里实际出现过的规整命令族做结构化归组（docker logs/ps/stats/inspect、
    flagd OFREP、Jaeger、Prometheus），其余命令（git/grep/python 内联等）退化为
    「去掉 cd 前缀后的近exact 匹配」——宁可少剪不可错剪。
    """
    c = _strip_cd_prefix(cmd.strip())

    m = re.search(r"docker\s+logs\s+([A-Za-z0-9_.-]+)", c)
    if m:
        return f"docker-logs:{m.group(1)}"

    m = re.search(r"docker\s+(?:ps|stats|inspect)\b", c)
    if m:
        f = re.search(r"name=([A-Za-z0-9_.-]+)", c)
        if f:
            return f"docker-status:{f.group(1)}"
        # docker stats/inspect <container> 的第一个非 flag 参数；docker ps 无参时归 all
        toks = [t for t in c.split() if not t.startswith("-")]
        verb_idx = next((i for i, t in enumerate(toks) if t in ("ps", "stats", "inspect")), -1)
        container = toks[verb_idx + 1] if 0 <= verb_idx + 1 < len(toks) else "all"
        if container.startswith(("{{", "'{{", '"{{')):
            container = "all"
        return f"docker-status:{container}"

    m = re.search(r"ofrep/v1/evaluate/flags/([A-Za-z0-9_.-]+)", c)
    if m:
        return f"ofrep:{m.group(1)}"

    if ":16686" in c:
        m = re.search(r":16686(/\S+)", c)
        return f"jaeger:{m.group(1) if m else '*'}"

    if ":9090" in c:
        # --data-urlencode 'query=<expr>' 或 URL 里的 query=<expr>，取完整 PromQL 表达式
        m = re.search(r"query=([^'&\s]+)", c)
        return f"prom:{m.group(1) if m else '*'}"

    return "cmd:" + re.sub(r"\s+", " ", c)


def _prune_candidates(steps: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """返回 [(step_index_0based, reason), ...] 候选修剪列表（安全网复验前）。"""
    candidates: list[tuple[int, str]] = []
    # skeleton -> (信号类型并集, 观测事实 token 并集)
    seen: dict[str, tuple[set[str], set[str]]] = {}
    # 先扫一遍：每个 skeleton 有没有「拿到过数据」的观测（供 R5 判断失败重试）
    skeleton_has_data: dict[str, bool] = {}
    for step in steps:
        if step.get("tool_name") != "Bash":
            continue
        sk = _bash_skeleton(str((step.get("tool_input") or {}).get("command", "") or ""))
        if _obs_facts(step.get("observation") or ""):
            skeleton_has_data[sk] = True

    for i, step in enumerate(steps):
        if step.get("hook_decision") == "allow":
            continue  # 台账步骤，永不修剪
        if step.get("tool_name") == "StructuredOutput" and not is_malformed_step(step):
            continue  # 成功的终止输出步，永不修剪

        if is_malformed_step(step):
            candidates.append((i, "malformed-structured-output"))
            continue
        if is_noise_step(step):
            candidates.append((i, "noise"))
            continue
        if step.get("is_repeat_no_new_info"):
            candidates.append((i, "marked-repeat"))
            continue

        if step.get("tool_name") == "Bash":
            cmd = str((step.get("tool_input") or {}).get("command", "") or "")
            skeleton = _bash_skeleton(cmd)
            sigs = set(classify_signal_types("Bash", step.get("tool_input") or {}, step.get("observation") or ""))
            obs_tokens = _obs_facts(step.get("observation") or "")
            # R6 纯噪声步：既没覆盖任何一类信号源（Prometheus/Jaeger/logs/容器状态/
            # git 历史/RAG），观测里也没有 ≥2 个事实 token（典型的：cd 失败的源码
            # 探索 grep、ls workspace/ 目录列表、date 回显）——对证据链零贡献。
            # 注意「查了某信号源但结果为空」不算噪声（有信号覆盖，R4/R5 管），
            # 「源码探索找到了东西」也不算（有事实 token）。
            if not sigs and len(obs_tokens) < 2:
                candidates.append((i, "no-signal-no-fact"))
                continue
            prev_sigs, prev_tokens = seen.get(skeleton, (set(), set()))
            if skeleton in seen and sigs <= prev_sigs and obs_tokens <= prev_tokens:
                candidates.append((i, f"skeleton-repeat:{skeleton}"))
                continue
            # R5 失败查询重试：这一步「查了但什么都没拿到」，且同一骨架后续（或先前）
            # 有拿到数据的调用——失败的那次是纯噪声，保留成功的那次即可。
            # （跟 R3「删 malformed 重试、保留成功的那次」同一逻辑的 Bash 版。）
            if not obs_tokens and skeleton_has_data.get(skeleton):
                candidates.append((i, f"failed-query-retry:{skeleton}"))
                continue
            if skeleton not in seen:
                seen[skeleton] = (sigs, obs_tokens)
            else:
                seen[skeleton] = (prev_sigs | sigs, prev_tokens | obs_tokens)

    return candidates


def verify_pruned(traj: dict[str, Any], pruned_steps: list[dict[str, Any]]) -> list[str]:
    """修剪后的轨迹是否仍满足 evidence 可追溯 + 台账一致。返回失败原因列表（空=通过）。"""
    reasons: list[str] = []
    diag = traj.get("final_diagnosis") or {}
    ok, bad = rs.evidence_traceability_ok(diag, pruned_steps)
    if not ok:
        reasons.append(f"evidence 无法追溯: {bad!r}")
    if not rs.executed_actions_match_ledger(pruned_steps, diag.get("executed_actions") or []):
        reasons.append("executed_actions 台账不一致")
    return reasons


def prune_trajectory(traj: dict[str, Any]) -> dict[str, Any]:
    """修剪一条轨迹，返回带 `meta.pruning` 记录的新轨迹（原 dict 不动）。"""
    steps = traj.get("steps") or []
    candidates = dict(_prune_candidates(steps))

    kept_idx = [i for i in range(len(steps)) if i not in candidates]

    def _steps_with(restored: list[int]) -> list[dict[str, Any]]:
        return [steps[i] for i in sorted(kept_idx + restored)]

    # 贪心回退（设计文档 C2 第 4 条「不过就回退该步」）：先全量应用候选修剪并复验；
    # 复验不过时按候选逆序逐个回退，直到复验通过（回退最少够用的步数）。
    # verify_pruned 返回失败原因列表，空列表 = 通过。
    to_restore: list[int] = []
    if candidates and verify_pruned(traj, _steps_with(to_restore)):
        for cand_idx in sorted(candidates, reverse=True):
            to_restore.append(cand_idx)
            if not verify_pruned(traj, _steps_with(to_restore)):
                break

    final_idx = sorted(kept_idx + to_restore)
    removed_idx = sorted(set(candidates) - set(to_restore))
    removed_records = []
    for i in removed_idx:
        removed_records.append(
            {
                "step_index_1based": i + 1,
                "tool_name": steps[i].get("tool_name"),
                "reason": candidates[i],
                "command_snippet": str((steps[i].get("tool_input") or {}).get("command", ""))[:160],
            }
        )

    out = {k: v for k, v in traj.items() if k != "steps"}
    out["steps"] = [steps[i] for i in final_idx]
    meta = dict(traj.get("meta") or {})
    meta["pruning"] = {
        "steps_before": len(steps),
        "steps_after": len(final_idx),
        "removed": removed_records,
        "restored_by_verification": [
            {"step_index_1based": i + 1, "reason": candidates[i]} for i in sorted(to_restore)
        ],
    }
    out["meta"] = meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="对冷启轨迹做步级修剪（C2）")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for f in sorted(args.traj_dir.glob("*.json")):
        traj = json.loads(f.read_text(encoding="utf-8"))
        pruned = prune_trajectory(traj)
        (args.out_dir / f.name).write_text(
            json.dumps(pruned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        m = pruned["meta"]["pruning"]
        stats.append(
            {
                "alert_id": traj.get("alert_id") or f.stem,
                "steps_before": m["steps_before"],
                "steps_after": m["steps_after"],
                "removed": len(m["removed"]),
                "restored_by_verification": len(m["restored_by_verification"]),
            }
        )

    total_before = sum(s["steps_before"] for s in stats)
    total_after = sum(s["steps_after"] for s in stats)
    total_removed = sum(s["removed"] for s in stats)
    total_restored = sum(s["restored_by_verification"] for s in stats)
    print(f"[prune_steps] {len(stats)} 条轨迹: {total_before} 步 -> {total_after} 步"
          f"（修剪 {total_removed} 步，安全网回退 {total_restored} 步）")
    if stats:
        print(f"[prune_steps] 均值 {total_before / len(stats):.2f} -> {total_after / len(stats):.2f} 步/条")
    print(f"[prune_steps] 写入 {args.out_dir}")


if __name__ == "__main__":
    main()
