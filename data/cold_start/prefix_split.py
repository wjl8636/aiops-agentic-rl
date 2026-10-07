"""渐进式前缀拆分（design doc §4.4；2026-09-08 起跳过 malformed 步骤，见
`docs/数据增强方案.md` §4.5-3「malformed 目标剔除：prefix_split 跳过 __unparsedToolInput
步骤（规则化，不再靠人工发现）」）。

一条完整轨迹有 N 个「工具调用」步骤（`trajectory["steps"]`，长度 N），外加一个终止动作——
输出最终 `Diagnosis`。design doc 把这个终止动作算作「第 N 步」（"120 条完整轨迹平均 8.2 步
工具调用，只有最后 1 步是输出 Diagnosis 收敛"），所以一条 N 步轨迹恰好拆出 N 条训练子样本：
  子样本 i（i = 1..N-1）：上下文 = 前 i-1 步的调用+观测，目标 = 第 i 步的工具调用（tool_call）
  子样本 N（最后一条）  ：上下文 = 全部 N 步的调用+观测，目标 = 最终 Diagnosis（diagnosis）
每个子样本只对「目标」这一项计算 loss（上下文部分只是喂进去的历史，不参与 loss）——
这正是解决"反复调 kubectl 不收敛"问题的关键：模型在每一个中间节点都单独学一次
「现在证据够不够收敛」，而不是只在整条轨迹的最后一步学一次。

malformed 步骤（`StructuredOutput` 带 `__unparsedToolInput` 的 schema/JSON 解析失败重试）
在拆分前先整步剔除（既不做目标、也不进前缀上下文）——把这种步骤当训练目标等于教模型
输出不合法 JSON（v8 里 seed_0381 第 9 步的教训）。正常管线里 `prune_steps.py` 已经在
拆分前把这类步骤从轨迹里删掉了，这里是第二道防线，保证喂进本脚本的任何轨迹都不会
再把 malformed 步骤漏成目标。

一条 N 步轨迹 -> N 条子样本，跟设计文档「120 条完整轨迹（平均 8.2 步）拆完约 990 条子样本」
的比例吻合（120 * 8.2 ≈ 984 ≈ 990）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_TRAJ_DIR = HERE / "trajectories" / "mock"
DEFAULT_OUT_PATH = HERE / "prefix_subsamples.mock.json"


def _is_malformed_step(step: dict[str, Any]) -> bool:
    """StructuredOutput 的解析失败重试步：SDK 把解析不出的原始输入挂在 `__unparsedToolInput`。"""
    return "__unparsedToolInput" in json.dumps(step.get("tool_input") or {}, ensure_ascii=False)


def split_trajectory(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """把一条轨迹拆成 N 条前缀子样本（N = len(steps)）。

    每条子样本形状：
      {
        "alert_id": ..., "alert": {...},
        "step_index": i,          # 1-indexed，这是「第几个训练子样本」
        "total_steps": N,
        "prefix_steps": [...],    # 喂给模型的历史（不计 loss）
        "target_type": "tool_call" | "diagnosis",
        "target": {...},          # 计 loss 的目标
      }
    """
    steps = [s for s in (traj.get("steps") or []) if not _is_malformed_step(s)]
    n = len(steps)
    alert = traj.get("alert")
    alert_id = traj.get("alert_id")
    subsamples: list[dict[str, Any]] = []

    for i in range(1, n + 1):
        if i < n:
            prefix_steps = steps[0 : i - 1]  # 决策第 i 步动作时，前面 i-1 步的观测已经到手
            target_type = "tool_call"
            target = {"tool_name": steps[i - 1]["tool_name"], "tool_input": steps[i - 1]["tool_input"]}
        else:
            prefix_steps = steps[0:n]  # 最后一条：全部 N 步的完整观测都已到手
            target_type = "diagnosis"
            target = traj.get("final_diagnosis")

        subsamples.append(
            {
                "alert_id": alert_id,
                "alert": alert,
                "step_index": i,
                "total_steps": n,
                "prefix_steps": prefix_steps,
                "target_type": target_type,
                "target": target,
            }
        )
    return subsamples


def split_directory(traj_dir: Path) -> list[dict[str, Any]]:
    all_subsamples: list[dict[str, Any]] = []
    for f in sorted(Path(traj_dir).glob("*.json")):
        traj = json.loads(f.read_text(encoding="utf-8"))
        all_subsamples.extend(split_trajectory(traj))
    return all_subsamples


def main() -> None:
    parser = argparse.ArgumentParser(description="把冷启轨迹按前缀拆成 step-level 训练子样本")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    subsamples = split_directory(args.traj_dir)
    args.out.write_text(json.dumps(subsamples, ensure_ascii=False, indent=2), encoding="utf-8")

    n_traj = len(list(Path(args.traj_dir).glob("*.json")))
    n_diag_targets = sum(1 for s in subsamples if s["target_type"] == "diagnosis")
    print(f"[prefix_split] {n_traj} 条轨迹 -> {len(subsamples)} 条子样本（其中 {n_diag_targets} 条是 diagnosis 收敛目标）")
    print(f"[prefix_split] 写入 {args.out}")


if __name__ == "__main__":
    main()
