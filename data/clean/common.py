"""data/clean 内部共享的告警字段解析与结构化信号推断工具。

被 dedup.py / filter.py / split.py 复用，避免三个模块各自重复实现「怎么从一条告警 dict
里取 service / 拼文本 / 猜 kind」这类基础逻辑。

字段口径对照：
- service 解析优先级：alert.service -> labels.service -> labels.job，这样 filter.py 复用真实指纹函数时，
  「不可复现过滤」和「指纹去重过滤」两步对 service 的理解不会打架。
- kind 四个取值：dependency / resource / deploy_regression / config。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

KIND_VALUES = ("dependency", "resource", "deploy_regression", "config")

# 已知能挂上本地 fixture（flagd flag / docker 容器 / kind Deployment）的服务集合。
# 处置目标服务：recommendation, ad, frontend, cart, checkout, currency, payment,
# shipping, quote, email。
# 另有两个「诊断目标」依赖型服务（product-catalog 是被 flagd 开关的目标；
# kafka 是队列积压场景的目标）。
# 生产环境应该直接从 agent.config 读取这份集合；这里手写一份是为了 data/clean 不必
# 强依赖 pydantic / claude-agent-sdk 就能跑基础的结构化过滤逻辑。
KNOWN_FIXTURE_SERVICES = {
    "recommendation",
    "ad",
    "adservice",
    "frontend",
    "cart",
    "checkout",
    "currency",
    "payment",
    "shipping",
    "quote",
    "email",
    "product-catalog",
    "productcatalog",
    "kafka",
}


def resolve_service(alert: dict) -> str:
    """取 service，优先级跟 ingest.py::alert_fingerprint 保持一致：
    alert['service'] -> labels.service -> labels.job -> ''。"""
    labels = alert.get("labels", {}) or {}
    service = alert.get("service") or labels.get("service") or labels.get("job") or ""
    return service.strip()


def alert_text(alert: dict) -> str:
    """拼 annotations.summary + annotations.description，作为语义/关键词匹配的文本来源。"""
    ann = alert.get("annotations", {}) or {}
    summary = (ann.get("summary") or "").strip()
    description = (ann.get("description") or "").strip()
    return f"{summary} {description}".strip()


# --- 混合根因 / 低置信度 / kind 的启发式判断 -------------------------------------------------
#
# 三者都遵循同一个原则：优先信任显式字段（万一上游生成脚本，比如 sibling 任务的种子扩增，
# 已经标好了 hybrid / low_confidence / kind），显式字段缺失时才退到关键词启发式兜底。
# 这是数据切分阶段的粗筛，不是诊断 Agent 的真实判定——真实判定要等模型跑完 Diagnosis 才有,
# 这里只服务于 split.py 的分层抽样，务必在使用处保持这层认知。

_HYBRID_PATTERNS = [
    r"混合根因",
    r"also_code_fix",
    r"既要.*也要",
    r"先.*(?:止血|止损).*(?:再|同时).*(?:根治|改代码|code_fix)",
    r"同时(?:触发|需要).*(?:代码修复|code_fix)",
]

_LOW_CONFIDENCE_PATTERNS = [
    r"证据不足",
    r"没有明确",
    r"没有指名",
    r"不清楚具体",
    r"无法定位",
    r"感觉有点",
    r"大约.*左右",
    r"间歇性.*但",
]

# kind 关键词打分表：命中越多某一类的关键词，越可能属于该 kind。
_KIND_KEYWORD_MAP: dict[str, list[str]] = {
    "resource": [r"cpu", r"内存", r"memory", r"oom", r"扩容", r"资源(?:不足|打满)"],
    "deploy_regression": [r"发版", r"上线", r"部署", r"commit", r"镜像", r"回归", r"deploy"],
    "dependency": [r"依赖", r"上游", r"下游", r"调用失败", r"5xx", r"错误率", r"flagd", r"队列", r"积压", r"kafka"],
    "config": [r"配置(?:开关|中心|变更)", r"flag(?:d)?\s*(?:开关|key)", r"功能开关"],
}


def infer_hybrid(alert: dict) -> bool:
    """告警是否带有『混合根因』信号（同时需要 online_op 止血 + code_fix 根治）。"""
    explicit = alert.get("hybrid")
    if isinstance(explicit, bool):
        return explicit
    text = f"{alert_text(alert)} {alert.get('scenario_hint', '')}"
    return any(re.search(p, text, re.IGNORECASE) for p in _HYBRID_PATTERNS)


def infer_low_confidence(alert: dict) -> bool:
    """告警是否『低置信度』（service 不明确 / 描述模糊没有可量化信号）。"""
    explicit = alert.get("low_confidence")
    if isinstance(explicit, bool):
        return explicit
    service = resolve_service(alert)
    if not service or service.lower() == "unknown":
        return True
    text = alert_text(alert)
    return any(re.search(p, text) for p in _LOW_CONFIDENCE_PATTERNS)


def infer_kind(alert: dict) -> str:
    """推断四类根因之一（dependency/resource/deploy_regression/config）。

    优先信任显式且合法的 `kind` 字段；否则用关键词打分，取命中最多的一类；
    全无命中时兜底为 "dependency"（真实 alerts/*.json 种子池里出现最多的类别，
    经验兜底，不是理论上的默认值）。
    """
    explicit = alert.get("kind")
    if explicit in KIND_VALUES:
        return explicit

    haystack = " ".join(
        [
            alert_text(alert),
            str(alert.get("alertname", "")),
            str(alert.get("scenario_hint", "")),
        ]
    ).lower()
    scores = {k: sum(1 for p in pats if re.search(p, haystack)) for k, pats in _KIND_KEYWORD_MAP.items()}
    best_kind, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "dependency"
    return best_kind


def stratum_key(alert: dict) -> str:
    """五档分层键，互斥优先级：低置信度 > 混合根因 > 四类根因。

    判断取舍（设计决定，见 data/clean/split.py 模块 docstring 的进一步说明）：
    - 低置信度信号最强、最不该被 kind 掩盖——它本身就意味着 kind 推断不可靠。
    - 混合根因次之——它同时具备多类信号，若按单一 kind 分层会丢失「混合」这个
      对训练最重要的属性。
    - 剩下的才按四类根因（dependency/resource/deploy_regression/config）分层。
    """
    if infer_low_confidence(alert):
        return "low_confidence"
    if infer_hybrid(alert):
        return "hybrid"
    return f"kind:{infer_kind(alert)}"


# ---------------------------------------------------------------------------
# 目录级 I/O：filter.py / dedup.py / split.py 的 CLI 都靠这两个函数串起来。
# 文件名沿用输入时的原名（不重新编号），保证一条告警从 seeds/generated 一路流到
# clean/split 都能用文件名对得上是同一条，方便审计每一步砍掉了哪些。
# ---------------------------------------------------------------------------


def read_alerts_dir(in_dir: str | Path) -> list[tuple[str, dict]]:
    """按文件名排序读取一个目录下所有 `seed_*.json`，返回 [(文件名, alert dict), ...]。"""
    in_dir = Path(in_dir)
    return [
        (p.name, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(in_dir.glob("seed_*.json"))
    ]


def write_alerts_dir(pairs: list[tuple[str, dict]], out_dir: str | Path) -> None:
    """把 [(文件名, alert dict), ...] 写回一个目录，文件名沿用输入时的原名。

    重跑防陈旧文件：写入前先清掉 `out_dir` 里已存在、但不在本次 `pairs` 里的
    `seed_*.json`——否则对同一个 out-dir 重跑（比如换了阈值/换了 embedding 后端
    重新 dedup）时，上一次跑剩下的文件会跟这次的新产出混在一起，产出目录里的
    文件数会比本次真实保留的条数多，且没有任何报错提示，统计口径会悄悄错掉。
    只清 `seed_*.json`（跟 `read_alerts_dir` 的 glob 对齐），不动 `_manifest.json`
    等旁路文件。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {fname for fname, _ in pairs}
    for stale in out_dir.glob("seed_*.json"):
        if stale.name not in keep_names:
            stale.unlink()
    for fname, alert in pairs:
        with (out_dir / fname).open("w", encoding="utf-8") as f:
            json.dump(alert, f, ensure_ascii=False, indent=2)
            f.write("\n")
