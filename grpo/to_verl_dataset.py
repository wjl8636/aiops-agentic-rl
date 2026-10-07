"""把 `data/clean/split/{train,held_out}` 的告警 + `data/cold_start/ground_truth`
的人工标注，转成 veRL GRPO 训练/评测要读的 `train.jsonl`/`held_out.jsonl`。

## 这一步为什么需要，且为什么本仓库之前一直没有
`grpo/train_grpo.sh` 里已经有 `data.train_batch_size`/`data.max_prompt_length`/
`data.max_response_length` 这几个 CLI 覆盖项，但**从未设置** `data.train_files`/
`data.val_files`——`docs/复现指南.md` 第 6 节列的两个已知适配点（reward 函数签名、
vLLM/sglang 冲突）都不是这个缺口。veRL 吃的是它自己能加载的数据集文件，不是本仓库
散落的告警 JSON（`data/clean/split/train/*.json`）+ ground truth JSON
（`data/cold_start/ground_truth/*.json`）两份文件对不上号地摆在那——这个脚本补上
这道转换，`train_grpo.sh` 引用它的默认输出路径。

## 每行的 schema：真实 veRL 惯例
顶层字段是
`data_source`/`prompt`/`ability`/`reward_model`/`extra_info`；`prompt` 是
`[{"role": ..., "content": ...}]` 的 chat message 列表；`reward_model` 是
`{"style": "rule", "ground_truth": ...}`。多轮工具调用场景（跟本
项目 `actor_rollout_ref.rollout.multi_turn.enable=true` 的设置一致）：
`prompt` 是 `[system, user]` 两条消息；`extra_info` 除了常规的 split/index，
还有 `need_tools_kwargs`/`tools_kwargs` 字段——但那是给 gsm8k 示例里那个会
**内联算分**的 `calc_gsm8k_reward` 工具用的（`tools_kwargs` 塞的是这个工具的
`create_kwargs`）。本项目的工具（Bash/Read/Grep/Glob，见
`verl_adapter/tool_schema_bridge.py`）不这样工作——`rollout_worker.py::
RealHookRoutedTool.execute()` 每次固定回 `tool_reward_score=0.0`，真正的
reward 由 `grpo/reward_router.py` 在整条 rollout 结束后统一算（该模块
docstring 解释了为什么）。所以本脚本**不产出** `tools_kwargs`/
`need_tools_kwargs`——这是本项目工具面跟 gsm8k 示例的真实差异，不是遗漏。
  - 加载端同样用 WebFetch 核对了 `verl/utils/dataset/rl_dataset.py`：
    `RLHFDataset` 读 `data.train_files`/`data.val_files` 时按文件名后缀分支，
    `.parquet` 走 `datasets.load_dataset("parquet", ...)`，`.json`/`.jsonl` 走
    `datasets.load_dataset("json", ...)`——两者都支持，不是只认 parquet。本仓库
    开发环境没装 `pandas`/`pyarrow`（`requirements.txt` 里明确不下场型依赖），
    所以本脚本产出 `.jsonl`（纯标准库 `json` 就能写、能测），真实训练机器上两种
    格式对 veRL 是等价的。

## ground_truth 字段名的一处必须做的映射（否则 reward 函数拿不到值）
`data/cold_start/ground_truth/*.json` 人工标注用的键是 `expect_route`（跟
`_annotation_report.md` 的表头一致，是标注阶段自己起的名字），但
`reward/trajectory.py::Trajectory.ground_truth` 和
`reward/outcome_reward.py::route_match_reward()` 读的键是 `route`
（`traj.ground_truth.get("route")`）——不是 `expect_route`。本脚本在写进
`reward_model.ground_truth` 之前做这一步改名；漏掉这一步不会报错，只会让
`route_match_reward` 对每一条 rollout 都静默拿到 `None` 从而永远给 0 分，是
一个容易被忽略但会让一整层 reward 信号失效的坑。

`expect_also_code_fix` 字段目前没有任何 reward 函数读取它
（`reward/outcome_reward.py::_handoff_applicable()` 判断 `also_code_fix` 只看
模型自己输出的 `final_diagnosis["also_code_fix"]`，不看 ground truth）——本脚本
如实丢弃，不塞进 `reward_model.ground_truth`（塞了也没人读，只会让 payload 变大、
还可能被误当作"这个字段真的会被用来打分"）。

`kind`（`reward/outcome_reward.py::ROOT_CAUSE_FIELDS` 三个字段之一）目前**全部**
ground_truth 标注文件都没有这个字段（无论是本脚本要转换的、还是已有的 29+45 条）
——这是标注数据本身的既有缺口，不是本脚本引入的新问题，如实反映现状，不在这里
编造一个 `kind` 值。

## 告警喂给模型前必须剔除 `scenario_hint`
`scenario_hint` 是数据生成阶段写的分层标注（`data/cold_start/collect_trajectories.py`
第 194-196 行同样的处理），真实告警不会有这个字段，`fault_injection.py` 模块
docstring 也明确写了"喂给模型的 prompt 里已经剔除了 scenario_hint 字段"——本脚本
在构造 prompt 和 `extra_info["alert"]` 时都统一剔除，保证训练时模型看到的、和
`extra_info` 里存的元数据一致，不会有一份剔除一份没剔除的不一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
AIOPS_AGENT_DIR = REPO_ROOT / "AIops-agent"

DEFAULT_ALERTS_DIR = REPO_ROOT / "data" / "clean" / "split" / "train"
DEFAULT_GROUND_TRUTH_DIR = REPO_ROOT / "data" / "cold_start" / "ground_truth"
DEFAULT_OUT = HERE / "verl_data" / "train.jsonl"

DATA_SOURCE = "aiops_diagnosis"
ABILITY = "aiops_diagnosis"

#: `data/cold_start/ground_truth/*.json` 人工标注用的键 -> `Trajectory.ground_truth`
#: 真实读取的键。只映射 reward 代码实际会读的三个字段（见模块 docstring）。
_GT_KEY_MAP = {
    "expect_route": "route",
    "remediation_type": "remediation_type",
    "suspect_service": "suspect_service",
    # 目前标注文件里没有这三个，但 ROOT_CAUSE_FIELDS/HANDOFF_FIELDS 认得，未来标全了
    # 直接透传，不需要改这个脚本。
    "kind": "kind",
    "suspect_repo": "suspect_repo",
    "suspect_commit_hint": "suspect_commit_hint",
    "suspect_file_hint": "suspect_file_hint",
}


def _ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def _load_prompt_fns():
    """复用 AIops-agent 真实的提示词函数——跟 `data/cold_start/to_llamafactory_format.py`
    同样的道理：保证 GRPO rollout 里模型看到的提示词跟 SFT 训练时、跟真实
    `diagnose.py` 生产环境里一字不差，三者共用同一套 prompt 是训练闭环成立的前提
    （SFT 学的分布、GRPO 探索的分布、真实部署时吃到的分布，指的必须是同一个分布）。
    拿不到时（比如这台机器没有 AIops-agent 子模块）退化成一个简化版占位提示词，
    不阻塞转换流程,但绝不能把这份占位数据当真用于真实训练。
    """
    try:
        _ensure_aiops_agent_on_path()
        from agent.agents import prompts  # type: ignore

        return prompts.diagnose_prompt, prompts.diagnose_append
    except Exception:  # noqa: BLE001

        def _fallback_prompt(alert: dict[str, Any]) -> str:
            return f"请诊断以下告警：\n```json\n{json.dumps(alert, ensure_ascii=False, indent=2)}\n```\n"

        def _fallback_append() -> str:
            return "你是 AIOps 故障诊断处置 Agent，只读排查后输出结构化 Diagnosis。"

        return _fallback_prompt, _fallback_append


_diagnose_prompt, _diagnose_append = _load_prompt_fns()
SYSTEM_PROMPT = _diagnose_append()


def strip_scenario_hint(alert: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in alert.items() if k != "scenario_hint"}


def map_ground_truth(gt: dict[str, Any]) -> dict[str, Any]:
    """`expect_route` -> `route`；丢弃 `expect_also_code_fix`（没有 reward 函数读它，
    见模块 docstring）；其余已知键透传，未知键原样保留（宁可多带一点，不要在这里
    悄悄吃掉标注文件里将来可能新增的字段）。
    """
    out: dict[str, Any] = {}
    for key, value in gt.items():
        if key == "expect_also_code_fix":
            continue
        out[_GT_KEY_MAP.get(key, key)] = value
    return out


def build_prompt_messages(alert: dict[str, Any]) -> list[dict[str, str]]:
    alert_for_prompt = strip_scenario_hint(alert)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _diagnose_prompt(alert_for_prompt)},
    ]


def build_row(alert_id: str, alert: dict[str, Any], ground_truth: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    return {
        "data_source": DATA_SOURCE,
        "prompt": build_prompt_messages(alert),
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": map_ground_truth(ground_truth)},
        "extra_info": {
            "split": split,
            "index": index,
            "alert_id": alert_id,
            "alert": strip_scenario_hint(alert),
        },
    }


def convert(alerts_dir: Path, ground_truth_dir: Path, split: str) -> tuple[list[dict[str, Any]], list[str]]:
    """返回 (rows, skipped_alert_ids)。缺 ground truth 的告警**不静默丢弃**——
    调用方（`main()`）必须把 `skipped_alert_ids` 完整打印出来，不能只报个数字。
    """
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    alert_files = sorted(alerts_dir.glob("*.json"))
    for idx, alert_path in enumerate(alert_files):
        alert_id = alert_path.stem
        gt_path = ground_truth_dir / f"{alert_id}.json"
        if not gt_path.is_file():
            skipped.append(alert_id)
            continue
        alert = json.loads(alert_path.read_text(encoding="utf-8"))
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        rows.append(build_row(alert_id, alert, ground_truth, idx, split))
    return rows, skipped


def write_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把告警 + ground truth 转成 veRL GRPO 能读的 jsonl 数据集"
    )
    parser.add_argument("--alerts-dir", type=Path, default=DEFAULT_ALERTS_DIR)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--split", type=str, default="train", help="写进 extra_info.split 的标签")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows, skipped = convert(args.alerts_dir, args.ground_truth_dir, args.split)
    write_jsonl(rows, args.out)

    print(f"[to_verl_dataset] {len(rows)} 条 -> {args.out}")
    if skipped:
        print(
            f"[to_verl_dataset] 跳过 {len(skipped)} 条（{args.alerts_dir} 下没有对应的 "
            f"ground truth 文件，不能参与 reward 计算，未写入输出）："
        )
        for alert_id in skipped:
            print(f"  - {alert_id}")


if __name__ == "__main__":
    main()
