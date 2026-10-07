"""数据清洗第三步：分层切分（对应 4.2 节「分层切分」）。

设计文档目标：干净种子池约 500 条，按「四类根因 + 混合根因 + 低置信度」五档分层，抽 100 条
（≈20%）作 held-out，剩余进训练池；OOD 50 条是单独构造的全新故障类型（TLS 证书过期 / DNS
污染 / 慢查询 / GC 长暂停），**不从主种子池抽样产生**。

## 分层键怎么算

用 `common.stratum_key()`，互斥优先级：**低置信度 > 混合根因 > 四类根因**
（dependency / resource / deploy_regression / config）。

判断取舍（这是本模块一个明确的设计决定，写在这里防止之后有人以为是 bug）：
- schema.py 定义的 4 个 `kind` 取值里，`"config"` 在真实种子告警的措辞里几乎不会被
  关键词命中（它更像是诊断 Agent 侧的兜底/降级 kind，`Diagnosis.fallback()` 用的就是
  `kind="config"`，不是种子告警文本本身会自然写出来的措辞）。所以在真实 500 条种子池上，
  实际观察到的非空分层数常常是 5 档而不是理论上限的 6 档（4 类根因 + hybrid + low_confidence），
  跟设计文档「五档分层」的表述吻合——这不是本模块强制产出恰好 5 个桶，而是数据分布自然呈现出
  这个数字，本模块只保证「有多少非空桶就分多少层」，不假设固定层数。
- 优先级顺序（低置信度 > 混合根因 > 单一 kind）意味着一条同时具备「低置信度」和「混合根因」
  信号的告警会被归到 `low_confidence` 桶，不会被同时计两次——分层必须是互斥划分，不然
  `stratified_split()` 的「不丢不重」保证就没法成立。

## OOD 边界（不在本模块职责内）

OOD 50 条来自 `data/ood_fixtures/`（由另一个任务产出），是全新故障类型，不从主种子池抽样，
因此本模块的 `stratified_split()` 只处理 `held_out` / `train_pool` 两块。评测阶段会把
OOD 池单独拼进评测集，跟这里的切分逻辑没有耦合。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common  # noqa: E402

DEFAULT_HELD_OUT_RATIO = 0.2  # 对齐设计文档的 100/500 ≈ 20%


def stratify(alerts: list[dict], key_fn: Callable[[dict], str] = common.stratum_key) -> dict[str, list[int]]:
    """按 key_fn 分桶，返回 {stratum_key: [alerts 里的下标, ...]}。

    用下标而不是拷贝对象，方便后面精确切割、并在切完后断言「不丢不重」。
    """
    buckets: dict[str, list[int]] = {}
    for idx, a in enumerate(alerts):
        k = key_fn(a)
        buckets.setdefault(k, []).append(idx)
    return buckets


def stratified_split(
    alerts: list[dict],
    held_out_ratio: float = DEFAULT_HELD_OUT_RATIO,
    seed: int = 42,
    key_fn: Callable[[dict], str] = common.stratum_key,
) -> dict[str, list[dict]]:
    """按 stratum 分层抽样切成 {"held_out": [...], "train_pool": [...]}。

    - `held_out_ratio` 可配置（设计文档目标是 100/500=20%，但本 dev 环境样本量小，
      比例做成参数而不是硬编码只在 n=500 时才成立，方便小样本场景下测试/迭代）。
    - 每个 stratum 内部先用固定 `seed` 洗牌再切，保证可复现；每个 stratum 内
      `held_out` 数量 = `round(len(stratum) * held_out_ratio)`。小样本 stratum 四舍五入后
      可能出现 held_out=0（或者该 stratum 全部进 held_out），这是分层抽样在小样本下的固有
      取舍，不是 bug——真实 500 条种子池上每个 stratum 都有几十条，这个问题不明显。
    - 保证不丢不重：每条输入恰好被分到 held_out 或 train_pool 之一，用 assert 在返回前自检。
    """
    if not 0.0 <= held_out_ratio <= 1.0:
        raise ValueError(f"held_out_ratio must be in [0, 1], got {held_out_ratio}")

    buckets = stratify(alerts, key_fn=key_fn)
    rng = random.Random(seed)

    held_out_idx: list[int] = []
    train_idx: list[int] = []
    for k in sorted(buckets):  # 排序保证同一 seed 下切分结果确定，不依赖 dict 插入顺序之外的因素
        idxs = list(buckets[k])
        rng.shuffle(idxs)
        n_held = round(len(idxs) * held_out_ratio)
        held_out_idx.extend(idxs[:n_held])
        train_idx.extend(idxs[n_held:])

    held_out = [alerts[i] for i in held_out_idx]
    train_pool = [alerts[i] for i in train_idx]

    assert len(held_out) + len(train_pool) == len(alerts), "held_out + train_pool 数量应等于输入总数"
    assert len(set(held_out_idx) | set(train_idx)) == len(alerts), "不应该有下标被漏掉"
    assert len(set(held_out_idx) & set(train_idx)) == 0, "不应该有下标被分到两边"

    return {"held_out": held_out, "train_pool": train_pool}


def strata_report(alerts: list[dict], key_fn: Callable[[dict], str] = common.stratum_key) -> dict[str, int]:
    """各 stratum 的告警数量，供人工核对分层是否符合预期比例（例如混合根因/低置信度占比）。"""
    buckets = stratify(alerts, key_fn=key_fn)
    return {k: len(v) for k, v in sorted(buckets.items())}


def main() -> None:
    """真实 CLI：读 --in-dir 下的种子告警，跑分层切分，写到 --out-dir/train 和 --out-dir/held_out。

    docs/复现指南.md 第 3 节给的正是 `python3 -m data.clean.split --in-dir ... --out-dir ...`
    这条命令、第 4 节引用的 `data/clean/split/train` 就是这里的 `--out-dir/train`——此前
    `__main__` 只是拿两条手写 demo 打印一下 strata_report，并不会真的读写目录。
    """
    import argparse
    import logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, help="输入目录，读取其中的 seed_*.json")
    parser.add_argument("--out-dir", required=True, help="输出根目录，会在其下建 train/ 和 held_out/")
    parser.add_argument("--held-out-ratio", type=float, default=DEFAULT_HELD_OUT_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    pairs = common.read_alerts_dir(args.in_dir)
    logger.info("读取 %d 条种子告警自 %s", len(pairs), args.in_dir)

    alerts = [a for _, a in pairs]
    logger.info("分层统计：%s", strata_report(alerts))

    result = stratified_split(alerts, held_out_ratio=args.held_out_ratio, seed=args.seed)

    out_dir = Path(args.out_dir)
    for split_name, out_key in (("train", "train_pool"), ("held_out", "held_out")):
        wanted_ids = {id(a) for a in result[out_key]}
        split_pairs = [(fname, a) for fname, a in pairs if id(a) in wanted_ids]
        common.write_alerts_dir(split_pairs, out_dir / split_name)
        logger.info("%s: %d 条 -> %s", split_name, len(split_pairs), out_dir / split_name)


if __name__ == "__main__":
    main()
