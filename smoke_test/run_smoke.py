"""端到端小规模烟雾测试 —— 全程走真实代码，不需要 GPU/真实 API/真实 docker/k8s。

串联的是设计文档 4-5 章描述的完整链路，每一步都是本仓库真实存在的模块/脚本，
不是摆样子的伪代码：

    data/seeds/generated (30条真实种子)
      -> data/clean 三件套 (质量过滤 -> 双层去重 -> 分层切分)
      -> data/cold_start/collect_trajectories.py --mock (mock 轨迹)
      -> data/cold_start/rejection_sample.py (拒绝采样)
      -> data/cold_start/prefix_split.py (渐进式前缀拆分)
      -> data/cold_start/to_llamafactory_format.py (SFT 数据格式转换)
      -> reward/ + grpo/reward_router.py (三层 reward 组合计算 + 组内方差)
      -> eval/run_baseline_comparison.py (mock 端点三档对比报告)

跑通即代表：数据构造->冷启动->reward 计算->GRPO 分组->评测这条主链路在
接口层面是通的，不代表任何真实训练效果——见 docs/README.md 的「真实 vs 目标」说明。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "smoke_test" / "out"
PY = sys.executable


def _step(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, **kwargs)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"[run_smoke] 命令失败(exit={result.returncode}): {' '.join(cmd)}")
    return result


def step_seeds() -> list[dict]:
    """复用已经真实生成的 30 条种子（data/seeds/generated/），不重新生成——
    生成器本身的正确性由 data/seeds/tests/ 保证，这里只验证下游能不能消费它的产物。"""
    _step("1. 种子数据（复用 data/seeds/generated/ 的真实产物）")
    seed_dir = REPO_ROOT / "data" / "seeds" / "generated"
    files = sorted(seed_dir.glob("*.json"))
    files = [f for f in files if f.name != "_manifest.json"]
    if not files:
        raise SystemExit(
            f"[run_smoke] {seed_dir} 下没有种子 JSON，先跑一遍 "
            "`python3 data/seeds/generate_seeds.py` 生成种子"
        )
    alerts = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    print(f"加载 {len(alerts)} 条真实种子告警")
    return alerts


def step_clean(alerts: list[dict]) -> dict:
    _step("2. 清洗切分（data/clean: filter -> dedup -> split）")
    sys.path.insert(0, str(REPO_ROOT / "data" / "clean"))
    import filter as clean_filter  # noqa: E402
    import dedup as clean_dedup  # noqa: E402
    import split as clean_split  # noqa: E402

    filtered = clean_filter.quality_filter_pipeline(alerts)
    print(
        f"质量过滤: 保留 {len(filtered['kept'])}, "
        f"剔除不可复现 {len(filtered['dropped_unreproducible'])}, "
        f"剔除指纹碰撞 {len(filtered['dropped_fingerprint_collision'])}"
    )

    deduped = clean_dedup.dedup_pipeline(filtered["kept"])
    print(
        f"双层去重(backend={deduped['backend_used']}): 保留 {len(deduped['kept'])}, "
        f"MD5精确去重掉 {len(deduped['dropped_md5_exact'])}, "
        f"语义去重掉 {len(deduped['dropped_semantic'])}"
    )

    split_result = clean_split.stratified_split(deduped["kept"])
    strata = clean_split.strata_report(deduped["kept"])
    print(
        f"分层切分: held_out={len(split_result['held_out'])}, "
        f"train_pool={len(split_result['train_pool'])}, 分层分布={strata}"
    )
    return split_result


def step_cold_start() -> Path:
    _step("3. 冷启动轨迹采集（--mock 模式，真实调用 collect_trajectories.py）")
    traj_dir = OUT_DIR / "trajectories" / "mock"
    _run([PY, "data/cold_start/collect_trajectories.py", "--mock", "--out-dir", str(traj_dir)])
    n = len(list(traj_dir.glob("*.json")))
    print(f"生成 {n} 条 mock 轨迹于 {traj_dir}")
    return traj_dir


def step_rejection_sample(traj_dir: Path) -> Path:
    _step("4. 拒绝采样（rejection_sample.py）")
    kept_dir = OUT_DIR / "trajectories" / "kept"
    gt_dir = REPO_ROOT / "data" / "cold_start" / "ground_truth"
    _run(
        [
            PY,
            "data/cold_start/rejection_sample.py",
            "--traj-dir",
            str(traj_dir),
            "--gt-dir",
            str(gt_dir),
            "--copy-kept-to",
            str(kept_dir),
        ]
    )
    n = len(list(kept_dir.glob("*.json"))) if kept_dir.exists() else 0
    print(f"拒绝采样后保留 {n} 条轨迹于 {kept_dir}")
    return kept_dir


def step_prefix_split(kept_dir: Path) -> Path:
    _step("5. 渐进式前缀拆分（prefix_split.py）")
    out_path = OUT_DIR / "prefix_subsamples.json"
    _run([PY, "data/cold_start/prefix_split.py", "--traj-dir", str(kept_dir), "--out", str(out_path)])
    subsamples = json.loads(out_path.read_text(encoding="utf-8"))
    n_diag = sum(1 for s in subsamples if s.get("target_type") == "diagnosis")
    print(f"拆出 {len(subsamples)} 条子样本，其中 {n_diag} 条收敛(diagnosis)目标")
    return out_path


def step_llamafactory_format(subsamples_path: Path) -> Path:
    _step("6. 转 LLaMA-Factory sharegpt 格式（to_llamafactory_format.py）")
    out_jsonl = OUT_DIR / "aiops_cold_start_sft.jsonl"
    dataset_info = OUT_DIR / "dataset_info_snippet.json"
    _run(
        [
            PY,
            "data/cold_start/to_llamafactory_format.py",
            "--in",
            str(subsamples_path),
            "--out",
            str(out_jsonl),
            "--dataset-info-out",
            str(dataset_info),
        ]
    )
    n_lines = sum(1 for _ in out_jsonl.open(encoding="utf-8"))
    print(f"写出 {n_lines} 条 sharegpt 格式样本于 {out_jsonl}")
    return out_jsonl


def step_reward_and_grpo_grouping() -> None:
    _step("7. Reward 计算 + GRPO 分组（reward/ + grpo/reward_router.py，对真实 5 条 mock 轨迹）")
    sys.path.insert(0, str(REPO_ROOT))
    from reward.trajectory import Step, Trajectory  # noqa: E402
    from grpo.reward_router import compute_trajectory_reward, compute_group_rewards  # noqa: E402

    traj_dir = REPO_ROOT / "data" / "cold_start" / "trajectories" / "mock"
    trajectories = []
    for f in sorted(traj_dir.glob("*.json")):
        raw = json.loads(f.read_text(encoding="utf-8"))
        steps = [Step(**{k: v for k, v in s.items() if k in Step.__dataclass_fields__}) for s in raw["steps"]]
        health = raw.get("post_action_health")
        post_health = (
            {"healthy_at_60s": bool(health.get("healthy", False)), "new_alert_triggered": False}
            if health
            else None
        )
        traj = Trajectory(
            alert=raw["alert"],
            steps=steps,
            final_diagnosis=raw["final_diagnosis"],
            ground_truth=raw.get("ground_truth") or {},
            route=raw["route"],
            post_action_health=post_health,
        )
        result = compute_trajectory_reward(traj)
        print(f"  {f.stem}: total_reward={result['total_reward']:.3f} (route={raw['route']})")
        trajectories.append(traj)

    group = compute_group_rewards(trajectories)
    print(
        f"GRPO 组内方差(group_reward_variance) = {group['group_reward_variance']:.4f}"
        f"（{group['group_size']} 条轨迹一组）"
    )


def step_eval_baseline_comparison() -> None:
    _step("8. 评测 harness 三档 baseline 对比（eval/run_baseline_comparison.py）")
    report_path = OUT_DIR / "baseline_comparison.json"
    _run(
        [
            PY,
            "-m",
            "eval.run_baseline_comparison",
            "--include-perfect",
            "--reports-dir",
            str(OUT_DIR),
            "--out",
            str(report_path),
        ]
    )
    json.loads(report_path.read_text(encoding="utf-8"))  # 确认报告是合法 JSON
    print(f"报告已写入 {report_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alerts = step_seeds()
    step_clean(alerts)
    traj_dir = step_cold_start()
    kept_dir = step_rejection_sample(traj_dir)
    subsamples_path = step_prefix_split(kept_dir)
    step_llamafactory_format(subsamples_path)
    step_reward_and_grpo_grouping()
    step_eval_baseline_comparison()
    print("\n" + "=" * 8 + " 烟雾测试全部步骤跑通 " + "=" * 8)
    print(f"中间产物见 {OUT_DIR}")


if __name__ == "__main__":
    main()
