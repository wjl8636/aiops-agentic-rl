"""数据清洗第二步：质量过滤（对应 4.2 节「质量过滤」，目标砍掉约 120 条）。

两类过滤：

## 1. 不可复现类过滤（启发式代理 —— 请读完这段再用）

设计文档原话：「剔除无法在 kind/docker 环境里稳定复现的场景（比如依赖外部网络的偶发故障）」。
本 dev 环境**没有真实 kind 集群 / docker 环境**可以实际跑一遍
`AIops-agent/scripts/inject.sh <scenario>` 去验证某条告警是否真的能被复现出来——那个脚本
本身也是硬编码 s1~s4 四个已知场景到具体 flagd flag / 镜像 tag 的映射（见该脚本），并不是一个
能对任意新告警通用的复现性判定器。

所以这里用**结构化启发式**代理这一步，而不是真的做复现性验证：
- `service` 解析不出来，或解析出来但不在已知 fixture 服务集合里 → 没有可挂的注入钩子；
- 描述文本命中「外部网络依赖」或「非确定性时序/偶发」关键词，且文本里找不到任何明确的注入钩子
  线索（flagd flag 名 / commit / 发版 / 镜像等）→ 大概率是没法在本地 fixture 里稳定复现的
  偶发故障。

**诚实声明**：这是启发式代理，不是真实复现性验证，precision/recall 都没有在真实环境标定过。
真实产线的这一步应该是对每条候选种子告警实际跑一遍
`AIOPS_BACKEND=docker ./scripts/inject.sh <scenario>`（或 kind 集群版本），观察注入后
Prometheus 指标 / 健康检查是否出现预期异常，用那个真实结果替换/校正这里的启发式判断。这个
heuristic 只在本 dev 环境里当占位，方便先把管线跑通。

## 2. 幂等去重压掉的重复项

复用 `AIops-agent/agent/integrations/ingest.py` 的真实 `alert_fingerprint()`：
`fingerprint = sha256(f"{alertname}|{service}|{severity}")[:16]`。真实系统的
`seen_recently()` 会在 `DEDUP_TTL_SECONDS` 冷却窗口内，同一指纹的第二条告警直接跳过不处理
（route=`skipped_duplicate`，参照 `alerts/s5_dup.json` 的场景设计）。这些告警在真实系统里
从来不会真的产出一条独立的诊断轨迹，留在训练池里只会占位置、不提供新信息，所以清洗阶段就按
同一套指纹算法过滤掉。

本模块优先直接从 `AIops-agent` 子模块 import 真实的 `alert_fingerprint()`（`ingest.py`
只依赖 `agent.config`，没有 pydantic / claude-agent-sdk 这类重依赖，正常环境下 import 应该
能成功）；只有子模块不可达或 import 失败时，才回退到本文件内逐字节重实现的
`_fallback_alert_fingerprint()`，并在日志里明确说明走了哪条路径。**两处实现必须保持字节级
一致**——改一处务必同步改另一处，否则「指纹过滤」和真实系统的 `seen_recently()` 口径会对不上。
"""
from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. 不可复现类过滤（启发式代理，见模块 docstring）
# ---------------------------------------------------------------------------

EXTERNAL_DEP_PATTERNS = [
    r"外部网络",
    r"公网",
    r"第三方(?:服务|接口|API)",
    r"运营商",
    r"卫星链路",
    r"跨机房专线",
    r"互联网(?:出口|链路)",
]
NONDETERMINISTIC_TIMING_PATTERNS = [
    r"偶发",
    r"随机时间",
    r"不定期",
    r"时不时",
    r"看运气",
    r"时好时坏",
]
# 出现这些线索，说明这条告警至少有个明确的注入/复现钩子，不应被判定为不可复现。
FIXTURE_HOOK_PATTERNS = [r"flagd", r"flag", r"故障注入", r"inject", r"commit", r"发版", r"部署", r"镜像"]


def is_likely_unreproducible(alert: dict) -> tuple[bool, list[str]]:
    """规则启发式判断这条告警是否『大概率没法在本地 fixture 里稳定复现』。

    返回 (是否判定为不可复现, 命中原因列表)。见模块顶部 docstring 的诚实声明——
    这不是真实复现性验证，只是结构化信号代理。
    """
    reasons: list[str] = []
    service = common.resolve_service(alert)
    if not service or service.lower() == "unknown":
        reasons.append("缺失可解析的 service，没有可挂的注入钩子")
    elif service.lower() not in common.KNOWN_FIXTURE_SERVICES:
        reasons.append(f"service={service!r} 不在已知 fixture 服务集合内")

    text = common.alert_text(alert)
    has_hook = any(re.search(p, text, re.IGNORECASE) for p in FIXTURE_HOOK_PATTERNS)
    if not has_hook:
        if any(re.search(p, text) for p in EXTERNAL_DEP_PATTERNS):
            reasons.append("描述涉及外部网络依赖，且未见明确注入钩子（flagd flag / commit / 发版）")
        if any(re.search(p, text) for p in NONDETERMINISTIC_TIMING_PATTERNS):
            reasons.append("描述涉及非确定性时序/偶发现象，且未见明确注入钩子")

    return (len(reasons) > 0, reasons)


def filter_unreproducible(alerts: list[dict]) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    """按 is_likely_unreproducible() 过滤。返回 (kept, [(dropped_alert, reasons), ...])。"""
    kept: list[dict] = []
    dropped: list[tuple[dict, list[str]]] = []
    for a in alerts:
        flagged, reasons = is_likely_unreproducible(a)
        if flagged:
            dropped.append((a, reasons))
        else:
            kept.append(a)
    return kept, dropped


# ---------------------------------------------------------------------------
# 2. 幂等去重指纹过滤 —— 复用 ingest.py::alert_fingerprint()
# ---------------------------------------------------------------------------


def _load_real_fingerprint_fn() -> Optional[Callable[[dict], str]]:
    """尝试直接从子模块 AIops-agent 导入真实的 alert_fingerprint()。失败返回 None（不抛异常）。"""
    repo_root = Path(__file__).resolve().parents[2]
    aiops_agent_dir = repo_root / "AIops-agent"
    if not aiops_agent_dir.is_dir():
        logger.warning("找不到 AIops-agent 子模块目录 (%s)，回退到本地重实现的指纹函数", aiops_agent_dir)
        return None

    path_str = str(aiops_agent_dir)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        from agent.integrations.ingest import alert_fingerprint  # type: ignore

        return alert_fingerprint
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "无法从 AIops-agent 子模块导入真实 alert_fingerprint()（%s: %s），"
            "回退到本地重实现（_fallback_alert_fingerprint，逻辑需与 ingest.py 保持字节级一致）",
            type(exc).__name__,
            exc,
        )
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass
        return None


def _fallback_alert_fingerprint(alert: dict) -> str:
    """与 AIops-agent/agent/integrations/ingest.py::alert_fingerprint 逐字节保持一致的重实现。

    只在无法从子模块 import 时使用（例如子模块未 checkout、或路径布局变化导致 import 失败）。
    任何改动都必须同步改那边的原始实现，否则这里的『幂等去重代理过滤』就会跟真实系统的
    `seen_recently()` 口径不一致。
    """
    name = alert.get("alertname") or alert.get("name") or ""
    labels = alert.get("labels", {}) or {}
    service = alert.get("service") or labels.get("service") or labels.get("job") or ""
    severity = labels.get("severity", "")
    key = f"{name}|{service}|{severity}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


_fingerprint_fn: Optional[Callable[[dict], str]] = None
_fingerprint_source = "uninitialized"


def get_fingerprint_fn() -> Callable[[dict], str]:
    """获取指纹函数（懒加载 + 缓存），并记录到底用了「导入真实实现」还是「本地重实现」。"""
    global _fingerprint_fn, _fingerprint_source
    if _fingerprint_fn is None:
        real = _load_real_fingerprint_fn()
        if real is not None:
            _fingerprint_fn = real
            _fingerprint_source = "AIops-agent.agent.integrations.ingest.alert_fingerprint (imported)"
        else:
            _fingerprint_fn = _fallback_alert_fingerprint
            _fingerprint_source = "data.clean.filter._fallback_alert_fingerprint (reimplemented)"
        logger.info("指纹函数来源：%s", _fingerprint_source)
    return _fingerprint_fn


_DEFAULT_DEDUP_TTL_SECONDS = 30 * 60  # 与 AIops-agent/agent/config.py 的默认值保持一致


def _load_dedup_ttl_seconds() -> int:
    """尽量从 AIops-agent 子模块读真实的 DEDUP_TTL_SECONDS，读不到就退回默认值 1800。"""
    try:
        from agent import config  # type: ignore

        return int(config.DEDUP_TTL_SECONDS)
    except Exception:  # noqa: BLE001
        return _DEFAULT_DEDUP_TTL_SECONDS


def _parse_starts_at_epoch(alert: dict) -> Optional[float]:
    """把 alert['startsAt']（ISO8601 UTC）解析成 epoch 秒；解析不出来返回 None。"""
    raw = alert.get("startsAt")
    if not raw:
        return None
    try:
        import datetime

        dt = datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc
        )
        return dt.timestamp()
    except ValueError:
        return None


def filter_fingerprint_collisions(
    alerts: list[dict], fingerprint_fn: Optional[Callable[[dict], str]] = None
) -> tuple[list[dict], list[dict]]:
    """指纹碰撞过滤：跟同一指纹「最近一次出现」的时间差在 `DEDUP_TTL_SECONDS` 冷却窗口内的，
    判定为「会被真实系统 seen_recently() 压掉的重复项」。

    真实系统按到达时的 wall-clock 时间判定冷却窗口，这里用每条告警自己的 `startsAt` 代替
    「到达时间」——训练数据是批量构造的，`startsAt` 就是这批告警各自模拟发生的时刻。
    没有 `startsAt` 或解析失败的告警，退化为原来的全局去重（保守处理，不假设它不重复）。
    """
    fp_fn = fingerprint_fn or get_fingerprint_fn()
    ttl = _load_dedup_ttl_seconds()
    last_seen_at: dict[str, float] = {}
    kept: list[dict] = []
    dropped: list[dict] = []
    for a in alerts:
        fp = fp_fn(a)
        ts = _parse_starts_at_epoch(a)
        last = last_seen_at.get(fp)
        if last is not None and (ts is None or abs(ts - last) <= ttl):
            dropped.append(a)
        else:
            if ts is not None:
                last_seen_at[fp] = ts
            elif fp not in last_seen_at:
                last_seen_at[fp] = 0.0
            kept.append(a)
    return kept, dropped


# ---------------------------------------------------------------------------
# 组合管线
# ---------------------------------------------------------------------------


def quality_filter_pipeline(alerts: list[dict]) -> dict:
    """完整质量过滤管线：先砍不可复现类，再砍指纹碰撞的重复项。"""
    kept1, dropped_unrepro = filter_unreproducible(alerts)
    kept2, dropped_fp = filter_fingerprint_collisions(kept1)
    return {
        "kept": kept2,
        "dropped_unreproducible": dropped_unrepro,
        "dropped_fingerprint_collision": dropped_fp,
    }


def main() -> None:
    """真实 CLI：读 --in-dir 下的种子告警，跑质量过滤，把幸存的写到 --out-dir。

    docs/复现指南.md 第 3 节给的正是 `python3 -m data.clean.filter --in-dir ... --out-dir ...`
    这条命令——这个 main() 就是那条命令真正要执行的逻辑（此前只有一段打印指纹来源的
    smoke test，没有真的读写目录，指南里的命令实际上什么也不会产出）。
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, help="输入目录，读取其中的 seed_*.json")
    parser.add_argument("--out-dir", required=True, help="输出目录，写入幸存的 seed_*.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    get_fingerprint_fn()
    logger.info("fingerprint fn source in this environment: %s", _fingerprint_source)

    pairs = common.read_alerts_dir(args.in_dir)
    logger.info("读取 %d 条种子告警自 %s", len(pairs), args.in_dir)

    alerts = [a for _, a in pairs]
    result = quality_filter_pipeline(alerts)

    kept_ids = {id(a) for a in result["kept"]}
    kept_pairs = [(fname, a) for fname, a in pairs if id(a) in kept_ids]
    common.write_alerts_dir(kept_pairs, args.out_dir)

    logger.info(
        "质量过滤完成：保留 %d 条，砍掉不可复现 %d 条、指纹碰撞 %d 条 -> 写入 %s",
        len(kept_pairs),
        len(result["dropped_unreproducible"]),
        len(result["dropped_fingerprint_collision"]),
        args.out_dir,
    )


if __name__ == "__main__":
    main()
