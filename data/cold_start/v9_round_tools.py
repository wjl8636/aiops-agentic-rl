"""v9 Phase C 轮次工具：按当前 GT 重新验收全部轮次轨迹 + 选择重采名单 + 家族统计。

用法（AIops-agent/.venv/bin/python3）：
    python v9_round_tools.py refresh     # 用当前 GT 重判全部 round 轨迹，更新 state 的 verdict
    python v9_round_tools.py stats       # 打印家族级验收统计 + 数据门自查
    python v9_round_tools.py retry-list  # 打印需要重采的种子（B1a 未过 / 配额风险 / ERROR）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rejection_sample as rs  # noqa: E402

GT_DIR = HERE / "ground_truth"
STATE_JSON = HERE / "_collection_v9_state.json"
RAW_ROOT = HERE / "trajectories_v9_raw"
AUG_MAP = HERE.parent.parent / "data" / "seeds" / "generated" / "_v9_augmentation_map.json"

# 数据门（方案 §8.1）的家族配额
FAMILY_QUOTA = {"B1a": 6, "B1b": 3, "A1": 5, "D1": 3, "C1": 6}


def load_aug() -> dict[str, dict[str, Any]]:
    return json.loads(AUG_MAP.read_text())


def load_state() -> dict[str, Any]:
    return json.loads(STATE_JSON.read_text())


def all_round_traj_dirs() -> list[Path]:
    return sorted(p for p in RAW_ROOT.glob("round*") if p.is_dir())


def best_verdicts() -> dict[str, dict[str, Any]]:
    """对每个种子取「所有轮次里最好的一次」：ACCEPTED 优先（取第一个 ACCEPTED 的轮次），
    否则取最后一轮的 REJECTED/ERROR 记录。"""
    state = load_state()
    best: dict[str, dict[str, Any]] = {}
    for r in state["runs"]:
        stem = r["stem"]
        if stem not in best or (best[stem].get("verdict") != "ACCEPTED" and r.get("verdict") == "ACCEPTED"):
            best[stem] = r
    return best


def refresh() -> None:
    """用当前 GT 重新验收全部轮次轨迹，把每个 run 记录的 verdict/reasons 刷新为当前口径。"""
    state = load_state()
    n = 0
    for r in state["runs"]:
        stem = r["stem"]
        gt_path = GT_DIR / f"{stem}.json"
        if not gt_path.exists():
            continue
        gt = json.loads(gt_path.read_text())
        # 找该 run 对应轮次的轨迹文件（round 编号在记录里）
        traj_path = RAW_ROOT / f"round{r.get('round', 1)}" / f"{stem}.json"
        if not traj_path.exists():
            continue
        traj = json.loads(traj_path.read_text())
        v = rs.check_trajectory(traj, gt)
        r["verdict"] = "ACCEPTED" if v["keep"] else "REJECTED"
        r["reject_reasons"] = v["reasons"]
        if traj.get("meta", {}).get("degraded"):
            r["degraded"] = traj["meta"]["degraded"]
        n += 1
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(f"[refresh] 重新验收 {n} 条 run 记录")
    stats()


def stats() -> None:
    aug = load_aug()
    best = best_verdicts()
    fam_stat: dict[str, dict[str, int]] = {}
    for stem, info in aug.items():
        fam = info["family"]
        fam_stat.setdefault(fam, {"seeds": 0, "accepted": 0, "rejected": 0, "error": 0, "pending": 0})
        fam_stat[fam]["seeds"] += 1
        r = best.get(stem)
        if r is None:
            fam_stat[fam]["pending"] += 1
        elif r.get("verdict") == "ACCEPTED":
            fam_stat[fam]["accepted"] += 1
        elif r.get("verdict") == "ERROR":
            fam_stat[fam]["error"] += 1
        else:
            fam_stat[fam]["rejected"] += 1

    print(f"{'家族':<5} {'种子':>4} {'过':>3} {'拒':>3} {'错':>3} {'未跑':>4}  配额")
    tot_acc = tot_run = 0
    for fam, s in fam_stat.items():
        quota = FAMILY_QUOTA.get(fam)
        qtxt = f"需≥{quota} {'✓' if s['accepted'] >= quota else '✗'}" if quota else "-"
        print(f"{fam:<5} {s['seeds']:>4} {s['accepted']:>3} {s['rejected']:>3} {s['error']:>3} {s['pending']:>4}  {qtxt}")
        tot_acc += s["accepted"]
        tot_run += s["accepted"] + s["rejected"] + s["error"]
    state = load_state()
    print(f"\n验收率（已跑 {tot_run} 条中通过 {tot_acc} 条）: {tot_acc / tot_run * 100 if tot_run else 0:.0f}%（门：≥55%）")
    print(f"累计花费: ${state.get('total_cost_usd', 0):.2f}（预算 $60 / 停机线 $50）")


def retry_list() -> None:
    """重采名单：ERROR 的必重采；REJECTED 的 B1a 按方案重采（3 轮）；其它家族 REJECTED
    若威胁配额也列出。"""
    aug = load_aug()
    best = best_verdicts()
    errors, b1a_failed, quota_risk = [], [], []
    fam_acc: dict[str, int] = {}
    for stem, r in best.items():
        fam = aug[stem]["family"]
        if r.get("verdict") == "ACCEPTED":
            fam_acc[fam] = fam_acc.get(fam, 0) + 1
    for stem, info in aug.items():
        r = best.get(stem)
        if r is None or r.get("verdict") == "ACCEPTED":
            continue
        fam = info["family"]
        if r.get("verdict") == "ERROR":
            errors.append(stem)
        elif fam == "B1a":
            rounds_used = sum(1 for x in load_state()["runs"] if x["stem"] == stem)
            if rounds_used < 3:
                b1a_failed.append(stem)
        else:
            quota = FAMILY_QUOTA.get(fam, 0)
            if fam_acc.get(fam, 0) < quota:
                quota_risk.append(stem)
    print("ERROR 重采:", errors)
    print("B1a 未过且轮次<3:", b1a_failed)
    print("配额风险家族的未过种子:", quota_risk)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stats"
    {"refresh": refresh, "stats": stats, "retry-list": retry_list}[cmd]()
