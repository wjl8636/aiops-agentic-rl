#!/usr/bin/env python3
"""种子场景池生成器 —— 对应设计文档 4.1 节「种子场景池」的三路扩增。

现有系统 `AIops-agent/alerts/*.json` 只有 12 条真实场景（四类根因 + 混合根因 +
低置信度降级等），直接拿来训练太少。本脚本实现文档里说的三路扩增：

  1. **参数化扩增** `generate_parametrized()`
     对四类根因场景（依赖故障 / 资源型 CPU / 队列积压 / 发版内存泄漏）分别把
     flagd flag key、指标阈值、severity、涉及服务名、时间窗做组合变换；
     发版内存泄漏这一类额外把「泄漏机制」本身也当一个变化维度
     （无界 list append / 无界 dict 累加 / 未关闭文件句柄 / WeakSet 用错）。

  2. **flagd 故障注入组合** `generate_flagd_combinations()`
     从下面 `CONFIRMED_FLAGD_FLAGS`（已对照真实 flagd 配置逐条核实存在）里
     取单点和多点组合，构造 dependency / resource 类场景。

  3. **历史工单反演** `generate_historical_reverse_derivation()`
     真实管线里这一路是从 `AIops-agent/agent/integrations/memory.py` 管理的
     Milvus 历史工单集合里检索、反向抽出告警文本。本脚本现在**真的这么做**：
     通过子进程调用 `AIops-agent/.venv/bin/python3`（复用它真实的
     `agent.integrations.memory.search_tickets()`，见 `_search_historical_tickets_via_milvus()`）
     用一批差异化的中文查询语句（`_MILVUS_QUERY_POOL`，覆盖 dependency/resource/
     deploy_regression/config 四类 kind 与 11 个 otel-demo 服务 + kafka）检索历史工单，
     汇总去重成候选池（`_milvus_candidate_records()`）。贯彻本项目「流程永不崩溃」的
     统一模式（参照 `data/clean/dedup.py` 的语义去重 vs 词面兜底）：Milvus/子进程/
     解析出任何问题都吞掉异常退回空列表，`generate_historical_reverse_derivation()`
     据此优先用真实检索池，只在池子不够大时才用 `HISTORICAL_INCIDENT_SUMMARIES`
     这份手写的「历史工单摘要」样例补齐——历史工单摘要本身也从来没被删除，
     依然是 Milvus 完全不可用时的完整兜底。见 `reverse_derive_from_incident_summary()`
     的 docstring。

Ground truth（写代码前已核实，不是猜的）：
  - 真实 alert schema：见 `AIops-agent/alerts/*.json`（字段：alertname / service /
    labels{service,severity,job} / annotations{summary,description} / startsAt /
    scenario_hint，某些场景还有 fixture_branch）。alert JSON 本身**没有** `kind`
    字段——`kind` 是故障诊断处置 Agent 输出（`Diagnosis.kind`）的字段，不是告警输入
    的字段。本脚本仍然在生成时记录「这条种子期望对应的 kind」，但只落在旁路的
    `manifest.json` 里（给测试/后续标注用），不会塞进 alert JSON，以免污染真实 schema。
  - `Diagnosis.kind` 枚举：dependency / resource / deploy_regression / config
    （见 `AIops-agent/agent/core/schema.py`）。队列积压（kafka lag）在真实 eval
    集里的 ground truth 是 `kind="resource"`（见 `AIops-agent/eval/expected.json`
    的 s3），不是单独的 kind 值。
  - flagd 支持的完整故障 flag 列表：已直接读取本机已 `fetch-otel-demo.sh` 拉取好的
    `AIops-agent/vendor/otel-demo/src/flagd/demo.flagd.json`（该目录被 .gitignore
    排除，不在 git 历史里，但本机确实存在，已逐条核对），完整列表见
    `CONFIRMED_FLAGD_FLAGS`。文档草稿提到的 `recommendationCacheFailure` /
    `adFailure` 均已确认真实存在。

--count 700 时怎么扩规模（哪个旋钮在哪里乘）：
  - 参数化扩增：4 个类别 × {service 候选数 × severity 候选数 × 时间窗候选数 ×
    （仅 deploy_regression）4 种泄漏机制} 的笛卡尔积远超 700，本脚本用固定随机种子
    的 `random.Random(seed).shuffle()` 从候选池里无重复采样前 N 个，N 越大就把
    `_dependency_recipes()` / `_resource_cpu_recipes()` / `_queue_backlog_recipes()`
    /`_deploy_regression_recipes()` 里的候选轴（service 列表、阈值列表、时间窗列表）
    加宽即可线性扩容，不需要改生成逻辑本身。
  - flagd 组合扩增：单点组合数 = len(CONFIRMED_FLAGD_FLAGS)；多点组合数现在真的用
    `itertools.combinations(flags, 2)` / `combinations(flags, 3)` 对
    `CONFIRMED_FLAGD_FLAGS` 做机械遍历（见 `_flagd_mechanical_combo_specs()`），
    11 个 flag 两两组合 C(11,2)=55 种、三三组合 C(11,3)=165 种，过滤掉同服务互斥/
    过量 hard-down 的组合后仍剩下数十条（具体规则见 `_is_semantically_valid_combo()`）。
    这批机械组合是在原来 4 条手写语义组合（`_flagd_multi_combo_specs()`，因为叙事
    更精细、故意保留）之上的**补充**，两者拼成 `_flagd_multi_combo_pool()`。继续扩容
    只需要放宽 `_is_semantically_valid_combo()` 的过滤规则，或者把 r 加到 4。
  - 历史工单反演：真实检索池的规模取决于 Milvus `aiops_tickets` 集合里实际有多少条
    工单（见 `data/seeds/backfill_historical_tickets.py` 写入的 ~57 条历史工单语料）
    以及 `_MILVUS_QUERY_POOL` 覆盖的查询多样性；扩容时优先加宽 `_MILVUS_QUERY_POOL`
    或往 Milvus 里继续回填更多工单，退回到对 `HISTORICAL_INCIDENT_SUMMARIES` 做
    同义改写/参数扰动（换事故时间、换服务名、换数值）只在 Milvus 不可用时才会发生。
  三路的默认分配比例（16 : 9 : 5，对应 --count 30）会按比例线性放大到 700
  （约 373 : 210 : 117），见 `_split_counts()`。

用法：
    python3 generate_seeds.py --count 30 --out-dir data/seeds/generated --seed 42
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 真实 Milvus 检索桥接的路径常量
# ---------------------------------------------------------------------------

# 本文件所在目录是 data/seeds/，仓库根是它的祖父目录，AIops-agent 子目录挂在仓库根下。
# 跟 `data/cold_start/to_llamafactory_format.py` 的 `_ensure_aiops_agent_on_path()` 是
# 同一套路径推导，但这里**不**把 AIops-agent 加进本进程的 `sys.path` ——本仓库自己的
# venv 没装 pymilvus，真的 `import agent.integrations.memory` 会在 `_ensure_collection()`
# 内部 import pymilvus 时抛异常（虽然那里本身也包了 try/except，不会崩，但那样等于
# 只能拿到「Milvus 不可用」这一种结果，检索不到真数据）。所以改用子进程调用
# `AIops-agent/.venv/bin/python3`——它的 venv 里 pymilvus + sentence-transformers 都装了，
# 见 `_search_historical_tickets_via_milvus()`。
_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent
AIOPS_AGENT_DIR = REPO_ROOT / "AIops-agent"

# ---------------------------------------------------------------------------
# Ground-truth 常量
# ---------------------------------------------------------------------------

KIND_VALUES = {"dependency", "resource", "deploy_regression", "config"}

# 已对照本机 AIops-agent/vendor/otel-demo/src/flagd/demo.flagd.json 逐条核实存在的
# 全部故障 flag（key -> (英文原意, 中文口径, 主要落在哪个服务上)）。
CONFIRMED_FLAGD_FLAGS: dict[str, dict[str, str]] = {
    "productCatalogFailure": {
        "desc": "product-catalog 服务对特定商品返回失败",
        "service": "product-catalog",
        "job": "product-catalog",
    },
    "recommendationCacheFailure": {
        "desc": "recommendation 服务的缓存层失效",
        "service": "recommendation",
        "job": "recommendation",
    },
    "adManualGc": {
        "desc": "ad 服务被强制触发完整的手动 GC",
        "service": "ad",
        "job": "adservice",
    },
    "adHighCpu": {
        "desc": "ad 服务人为触发高 CPU 负载",
        "service": "ad",
        "job": "adservice",
    },
    "adFailure": {
        "desc": "ad 服务整体失败",
        "service": "ad",
        "job": "adservice",
    },
    "kafkaQueueProblems": {
        "desc": "同时压高 Kafka 生产速率并给消费者引入延迟，制造 lag 尖峰",
        "service": "kafka",
        "job": "kafka",
    },
    "cartFailure": {
        "desc": "cart 服务失败",
        "service": "cart",
        "job": "cartservice",
    },
    "paymentFailure": {
        "desc": "payment 服务按比例拒绝扣款请求",
        "service": "payment",
        "job": "paymentservice",
    },
    "paymentUnreachable": {
        "desc": "payment 服务整体不可达",
        "service": "payment",
        "job": "paymentservice",
    },
    "loadGeneratorFloodHomepage": {
        "desc": "对 frontend 首页灌入大量请求造成流量型过载",
        "service": "frontend",
        "job": "frontend",
    },
    "imageSlowLoad": {
        "desc": "frontend 图片加载被人为拖慢",
        "service": "frontend",
        "job": "frontend",
    },
}

# 发版内存泄漏的四种泄漏机制模板（文档 4.1 节点名的四种），每种配一段贴近真实
# `AIops-agent/alerts/s4.json` / `s7_hybrid.json` 文风的中文 description 片段。
LEAK_MECHANISMS: list[dict[str, str]] = [
    {
        "slug": "unbounded_list_append",
        "name": "无界 list append",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里，对一个全局/请求级的 list 做了 "
            "append 但从未清理或做长度上限控制，随着请求量增加持续增长，最终把工作集内存吃满"
        ),
    },
    {
        "slug": "unbounded_dict_accumulation",
        "name": "无界 dict 累加",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里，用一个全局 dict 按请求 key 缓存/累加"
            "计算结果，却没有配 TTL 或淘汰策略，随着不同 key 不断出现，dict 体积单调增长"
        ),
    },
    {
        "slug": "unclosed_file_handle",
        "name": "未关闭的文件句柄",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里打开了日志/临时文件后没有用 with 语句"
            "正确关闭（或者异常路径遗漏了 close），文件句柄和内部缓冲区随请求量持续堆积"
        ),
    },
    {
        "slug": "misused_weakset",
        "name": "WeakSet 用错",
        "clue": (
            "怀疑是最近一次发版引入的某个 commit 里错误使用了 WeakSet/WeakValueDictionary"
            "（例如外部仍持有强引用，或者存进去的对象类型本身不支持被弱引用而退化成强引用），"
            "本该被回收的对象实际从未被回收"
        ),
    },
]

HISTORICAL_INCIDENT_SUMMARIES: list[dict[str, Any]] = [
    {
        "id": "hist-2026-03-import-timeout",
        "summary": (
            "2026-03 的一次真实事故：product-catalog 团队上线了一个后台批量数据导入脚本，"
            "该脚本发起的下游查询忘记加超时，导致 product-catalog 的处理线程池被慢查询占满，"
            "GetProduct 接口对上游（frontend、recommendation）出现级联超时。"
        ),
        "kind": "dependency",
        "service": "product-catalog",
        "job": "product-catalog",
        "severity": "critical",
    },
    {
        "id": "hist-2026-01-cache-no-ttl",
        "summary": (
            "2026-01 大促期间的一次事故：recommendation 团队为临时提升缓存命中率手动加了一个"
            "全局缓存字典，上线时忘了配 TTL，大促流量下缓存条目越滚越多，服务内存逐步爬升到"
            "被 OOMKilled。"
        ),
        "kind": "deploy_regression",
        "service": "recommendation",
        "job": "recommendation",
        "severity": "critical",
    },
    {
        "id": "hist-2025-11-gc-config-change",
        "summary": (
            "2025-11 的一次事故：ad 服务的一次运行时参数配置变更间接改变了 GC 触发频率，"
            "变更后 CPU 使用率持续处于高位，请求延迟 p99 明显上升，但没有伴随代码发版。"
        ),
        "kind": "resource",
        "service": "ad",
        "job": "adservice",
        "severity": "warning",
    },
    {
        "id": "hist-2025-09-consumer-scale-down",
        "summary": (
            "2025-09 的一次事故：一次误操作把 kafka 消费者组的副本数从 6 缩到了 2，"
            "生产速率不变但消费能力骤降，consumer lag 迅速堆积，订单相关下游处理明显滞后。"
        ),
        "kind": "resource",
        "service": "kafka",
        "job": "kafka",
        "severity": "warning",
    },
    {
        "id": "hist-2026-05-cert-rotation-mismatch",
        "summary": (
            "2026-05 的一次事故：payment 网关一次证书轮换后，调用方配置的信任链没有同步更新，"
            "导致支付请求间歇性握手失败，checkout 链路偶发超时，人工介入更新配置后恢复。"
        ),
        "kind": "config",
        "service": "payment",
        "job": "paymentservice",
        "severity": "warning",
    },
    # 下面是对上面 5 条真实历史工单摘要的同义改写/参数扰动（换事故时间、换服务名、换数值，
    # 不改变根因类型的因果结构），对应模块 docstring 里说的历史工单反演路扩容方式。
    {
        "id": "hist-2025-08-cart-batch-query-timeout",
        "summary": (
            "2025-08 的一次事故：cart 团队上线了一个购物车对账批处理脚本，脚本里的下游查询"
            "同样忘记设超时，占满 cart 的处理线程池，checkout 调用 cart 接口大面积超时。"
        ),
        "kind": "dependency",
        "service": "cart",
        "job": "cartservice",
        "severity": "critical",
    },
    {
        "id": "hist-2026-02-recommendation-precompute-timeout",
        "summary": (
            "2026-02 的一次事故：recommendation 新增了一个离线预计算任务，任务发起的下游"
            "查询没有配超时，占满 recommendation 自身线程池，frontend 调用 recommendation "
            "接口出现级联超时。"
        ),
        "kind": "dependency",
        "service": "recommendation",
        "job": "recommendation",
        "severity": "warning",
    },
    {
        "id": "hist-2025-12-cart-cache-no-eviction",
        "summary": (
            "2025-12 大促预热期间的一次事故：cart 团队为提升命中率加了一个全局购物车快照"
            "缓存，上线时没配淘汰策略，大促流量下缓存条目持续增长，服务内存逐步爬升直到"
            "被 OOMKilled。"
        ),
        "kind": "deploy_regression",
        "service": "cart",
        "job": "cartservice",
        "severity": "critical",
    },
    {
        "id": "hist-2026-04-checkout-session-cache-no-ttl",
        "summary": (
            "2026-04 的一次事故：checkout 团队为减少重复计算加了一个会话级缓存字典，"
            "同样忘了配 TTL，随着不同用户会话不断进入，缓存体积单调增长，最终把 checkout "
            "内存吃满触发重启。"
        ),
        "kind": "deploy_regression",
        "service": "checkout",
        "job": "checkoutservice",
        "severity": "critical",
    },
    {
        "id": "hist-2026-06-payment-runtime-flag-cpu",
        "summary": (
            "2026-06 的一次事故：payment 服务的一次运行时参数变更间接改变了序列化开销，"
            "变更后 CPU 使用率持续处于高位，请求延迟 p99 明显上升，没有伴随代码发版。"
        ),
        "kind": "resource",
        "service": "payment",
        "job": "paymentservice",
        "severity": "warning",
    },
    {
        "id": "hist-2025-10-frontend-gc-config-change",
        "summary": (
            "2025-10 的一次事故：frontend 的一次运行时参数配置变更间接改变了内部缓存清理"
            "频率，变更后 CPU 使用率持续处于高位，页面响应 p99 明显上升，没有伴随代码发版。"
        ),
        "kind": "resource",
        "service": "frontend",
        "job": "frontend",
        "severity": "warning",
    },
    {
        "id": "hist-2026-01-notification-consumer-scale-down",
        "summary": (
            "2026-01 的一次事故：一次误操作把通知消费者组的副本数从 4 缩到了 1，"
            "生产速率不变但消费能力骤降，consumer lag 迅速堆积，通知发送明显滞后。"
        ),
        "kind": "resource",
        "service": "kafka",
        "job": "kafka",
        "severity": "warning",
    },
    {
        "id": "hist-2025-07-shipping-consumer-scale-down",
        "summary": (
            "2025-07 的一次事故：一次容量规划失误把物流更新消费者组的副本数从 5 缩到了 2，"
            "生产速率不变但消费能力骤降，consumer lag 迅速堆积，物流状态更新明显滞后。"
        ),
        "kind": "resource",
        "service": "kafka",
        "job": "kafka",
        "severity": "critical",
    },
    {
        "id": "hist-2025-06-ad-cert-rotation-mismatch",
        "summary": (
            "2025-06 的一次事故：ad 服务依赖的一个内部网关证书轮换后，调用方配置的信任链"
            "没有同步更新，导致广告请求间歇性握手失败，frontend 首页偶发广告位超时，"
            "人工介入更新配置后恢复。"
        ),
        "kind": "config",
        "service": "ad",
        "job": "adservice",
        "severity": "warning",
    },
    {
        "id": "hist-2026-07-checkout-cert-rotation-mismatch",
        "summary": (
            "2026-07 的一次事故：checkout 调用的一个内部服务证书轮换后，调用方信任链"
            "配置没有同步更新，导致下单请求间歇性握手失败，人工介入更新配置后恢复。"
        ),
        "kind": "config",
        "service": "checkout",
        "job": "checkoutservice",
        "severity": "warning",
    },
]

GENERATED_DIR_DEFAULT = Path(__file__).resolve().parent / "generated"


# ---------------------------------------------------------------------------
# 基础构造 helper
# ---------------------------------------------------------------------------


def _iso(ts_base: str, offset_minutes: int) -> str:
    """从一个基准 ISO8601 UTC 时间戳按分钟做偏移，返回同格式字符串（不引入时区库依赖）。"""
    import datetime

    dt = datetime.datetime.strptime(ts_base, "%Y-%m-%dT%H:%M:%SZ")
    dt = dt + datetime.timedelta(minutes=offset_minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_alert(
    *,
    alertname: str,
    service: str,
    job: str,
    severity: str,
    summary: str,
    description: str,
    starts_at: str,
    scenario_hint: str,
) -> dict[str, Any]:
    """按真实 `AIops-agent/alerts/*.json` schema 拼装一条告警。"""
    return {
        "alertname": alertname,
        "service": service,
        "labels": {"service": service, "severity": severity, "job": job},
        "annotations": {"summary": summary, "description": description},
        "startsAt": starts_at,
        "scenario_hint": scenario_hint,
    }


def _split_counts(total: int, weights: list[int]) -> list[int]:
    """按 weights 比例把 total 拆成整数份，份数之和严格等于 total（余数依次补给前面几份）。"""
    s = sum(weights)
    base = [total * w // s for w in weights]
    remainder = total - sum(base)
    i = 0
    while remainder > 0:
        base[i % len(base)] += 1
        remainder -= 1
        i += 1
    return base


def _sample(pool: list[Any], n: int, rng: random.Random) -> list[Any]:
    """从候选池里洗牌后取前 n 个；池子不够大就循环补足。

    调用方（generate_parametrized / generate_historical_reverse_derivation）负责
    用返回列表里的位置序号（不是 recipe 自带的字段）分配 starts_at，保证即使同一个
    recipe 被循环补足多次，产出的告警时间戳也各不相同、不会字节级重复。
    """
    pool = list(pool)
    rng.shuffle(pool)
    out = []
    i = 0
    while len(out) < n:
        out.append(pool[i % len(pool)])
        i += 1
    return out


# ---------------------------------------------------------------------------
# 路径 1：参数化扩增
# ---------------------------------------------------------------------------


def _dependency_recipes() -> list[dict[str, Any]]:
    # 每个 flag 配 3 组 severity 变体，flag 本身仍是已核实的 6 个 dependency 类开关
    # （原 4 个 + recommendationCacheFailure/adFailure，在 flagd 组合路的单点场景里也
    # 出现过，这里用同样已核实的 flag 走参数化路的通用依赖故障叙事模板，不是新造 flag）。
    # 时间戳由 generate_parametrized() 按位置序号统一分配，这里不再手工填 offset——
    # 手工分配曾经因为忽略了「paymentFailure/paymentUnreachable 共享 service=payment」
    # 这类跨 flag 的指纹碰撞，导致 data/clean/filter.py 的幂等去重把大部分变体压掉。
    return [
        {"flag": "productCatalogFailure", "severity": "critical"},
        {"flag": "productCatalogFailure", "severity": "warning"},
        {"flag": "cartFailure", "severity": "critical"},
        {"flag": "cartFailure", "severity": "warning"},
        {"flag": "paymentFailure", "severity": "warning"},
        {"flag": "paymentFailure", "severity": "critical"},
        {"flag": "paymentUnreachable", "severity": "warning"},
        {"flag": "paymentUnreachable", "severity": "critical"},
        {"flag": "recommendationCacheFailure", "severity": "warning"},
        {"flag": "recommendationCacheFailure", "severity": "critical"},
        {"flag": "adFailure", "severity": "critical"},
        {"flag": "adFailure", "severity": "warning"},
    ]


def _resource_cpu_recipes() -> list[dict[str, Any]]:
    # 阈值轴从 4 个加宽到 12 个，<=90% 记 warning、>90% 记 critical。
    thresholds = [75, 78, 82, 85, 88, 90, 92, 94, 96, 97, 98, 99]
    return [
        {"threshold_pct": t, "severity": "warning" if t <= 90 else "critical"}
        for t in thresholds
    ]


def _queue_backlog_recipes() -> list[dict[str, Any]]:
    # topic/group 仍是原有 4 组（orders 是唯一在 alerts/s3.json 里核实过的真实 topic，
    # 其余 3 个沿用既有选择，不新增未核实的 topic 名），每组加宽到 3 档 lag/severity。
    return [
        {"topic": "orders", "group": "orders-consumer", "lag": 5200, "severity": "warning"},
        {"topic": "orders", "group": "orders-consumer", "lag": 9800, "severity": "warning"},
        {"topic": "orders", "group": "orders-consumer", "lag": 18500, "severity": "critical"},
        {"topic": "notifications", "group": "notify-consumer", "lag": 12800, "severity": "warning"},
        {"topic": "notifications", "group": "notify-consumer", "lag": 6400, "severity": "warning"},
        {"topic": "notifications", "group": "notify-consumer", "lag": 24000, "severity": "critical"},
        {"topic": "shipping-updates", "group": "shipping-consumer", "lag": 21000, "severity": "critical"},
        {"topic": "shipping-updates", "group": "shipping-consumer", "lag": 8700, "severity": "warning"},
        {"topic": "shipping-updates", "group": "shipping-consumer", "lag": 15300, "severity": "warning"},
        {"topic": "payment-events", "group": "payment-consumer", "lag": 4300, "severity": "warning"},
        {"topic": "payment-events", "group": "payment-consumer", "lag": 11200, "severity": "warning"},
        {"topic": "payment-events", "group": "payment-consumer", "lag": 19800, "severity": "critical"},
    ]


def _deploy_regression_recipes() -> list[dict[str, Any]]:
    # 泄漏机制严格保持 LEAK_MECHANISMS 的 4 种（对应真实 fixture 变体），
    # 每种机制配 3 组 severity 变体，不新增第 5 种机制。
    variants = ["critical", "warning", "critical"]
    return [
        {"mechanism": LEAK_MECHANISMS[i], "severity": sev}
        for i in range(len(LEAK_MECHANISMS))
        for sev in variants
    ]


def generate_parametrized(n: int, rng: random.Random) -> list[tuple[dict[str, Any], str]]:
    """参数化扩增：四类根因场景各自沿 flag/阈值/severity/服务/时间窗做组合变换。

    返回 (alert, intended_kind) 二元组列表，intended_kind 只用于 manifest，不写进 alert。
    """
    counts = _split_counts(n, [1, 1, 1, 1])  # dependency / resource_cpu / queue / deploy_regression 各 1/4
    out: list[tuple[dict[str, Any], str]] = []

    dep_recipes = _sample(_dependency_recipes(), counts[0], rng)
    for idx, r in enumerate(dep_recipes):
        flag_info = CONFIRMED_FLAGD_FLAGS[r["flag"]]
        service, job = flag_info["service"], flag_info["job"]
        # 按位置序号分配时间戳，每条间隔 45 分钟——严格大于 AIops-agent 真实
        # DEDUP_TTL_SECONDS（30 分钟）冷却窗口，保证同一 (alertname, service, severity)
        # 组合的多个变体不会被 data/clean/filter.py 的幂等去重指纹误判成同一条重复告警。
        starts_at = _iso("2026-07-01T09:00:00Z", idx * 45)
        # 三种措辞变体轮换：只换句式/语序/用词，不改变事实本身。参数化扩增之前每条
        # 只换数字/flag 名、句子骨架完全不动，BGE 语义去重会把这些判成同一句话的重复，
        # 见 data/clean/dedup.py 的语义去重——这里加措辞变体让「同一事实、不同表达」
        # 真的成立，而不是指望语义去重对纯数字变化格外宽容。
        dep_desc_variants = [
            f"一次功能开关变更（{r['flag']}）之后，{service}（{flag_info['desc']}）"
            f"开始对上游调用方返回失败/超时。上游服务的请求成功率随之下降，"
            f"错误集中出现在依赖 {service} 的那部分调用路径上，没有伴随代码发版。",
            f"{service} 最近出现依赖调用异常：{flag_info['desc']}，具体表现是开关"
            f"{r['flag']} 生效后，上游对 {service} 的请求持续失败或超时。观察窗口内"
            f"没有发现相关代码发版记录，错误面集中在调用 {service} 的链路上。",
            f"上游服务调用 {service} 时错误率明显上升，根因指向 flagd 开关 {r['flag']}"
            f"（{flag_info['desc']}）。这次异常和代码变更无关——近期没有发版，是开关"
            f"状态变化直接导致 {service} 对外响应失败/超时。",
        ]
        alert = _make_alert(
            alertname="DependencyFailureRate",
            service=service,
            job=job,
            severity=r["severity"],
            summary=f"{service} 依赖调用错误率上升",
            description=dep_desc_variants[idx % len(dep_desc_variants)],
            starts_at=starts_at,
            scenario_hint=(
                f"[参数化扩增/dependency] flag={r['flag']} service={service} "
                f"severity={r['severity']}：验证依赖故障场景在不同服务/severity/时间窗下的泛化"
            ),
        )
        out.append((alert, "dependency"))

    cpu_recipes = _sample(_resource_cpu_recipes(), counts[1], rng)
    for idx, r in enumerate(cpu_recipes):
        starts_at = _iso("2026-07-02T09:00:00Z", idx * 45)
        cpu_desc_variants = [
            f"adservice 的 CPU 使用率持续超过 {r['threshold_pct']}%（adHighCpu 开关），"
            "随着负载升高，请求延迟 p99 不断攀升。近期没有代码发版，指标随流量同步波动，"
            "怀疑是纯粹的资源型瓶颈而非代码回归。",
            f"ad 服务近期 CPU 占用居高不下，稳定维持在 {r['threshold_pct']}% 以上"
            "（对应 adHighCpu 开关），请求 p99 延迟随之走高。没有发版记录能解释这次变化，"
            "指标走势跟流量曲线基本同步，倾向判断是资源不够用，不是代码引入的回归。",
            f"监控显示 adservice 容器 CPU 使用率突破 {r['threshold_pct']}%"
            "（flagd 的 adHighCpu 开关被打开），且延迟 p99 同步恶化。排查没有发现对应的"
            "代码发版，CPU 曲线跟入口流量曲线高度吻合，更像是资源型瓶颈而不是代码问题。",
        ]
        alert = _make_alert(
            alertname="HighCpuUsage",
            service="ad",
            job="adservice",
            severity=r["severity"],
            summary=f"ad 服务 CPU 使用率超过 {r['threshold_pct']}%",
            description=cpu_desc_variants[idx % len(cpu_desc_variants)],
            starts_at=starts_at,
            scenario_hint=(
                f"[参数化扩增/resource_cpu] threshold={r['threshold_pct']}% severity={r['severity']}："
                "验证资源型 CPU 场景在不同阈值/严重度下的泛化"
            ),
        )
        out.append((alert, "resource"))

    q_recipes = _sample(_queue_backlog_recipes(), counts[2], rng)
    for idx, r in enumerate(q_recipes):
        starts_at = _iso("2026-07-03T09:00:00Z", idx * 45)
        queue_desc_variants = [
            f"{r['topic']} topic 的消费 lag 单调上升，当前约 {r['lag']} 条消息未消费"
            f"（kafkaQueueProblems）。生产速率超过消费速率；消费者 {r['group']} 本身健康，"
            "但算力配置不足以匹配当前生产吞吐。",
            f"Kafka {r['topic']} 这个 topic 上的积压持续变大，目前堆积大约 {r['lag']} "
            f"条待消费消息（kafkaQueueProblems 开关生效）。消费者组 {r['group']} 运行状态"
            "正常，问题出在生产速率长期超过消费速率，配置的算力跟不上。",
            f"消费组 {r['group']} 在 {r['topic']} topic 上的 lag 一直在涨，目前在 "
            f"{r['lag']} 条左右（对应 kafkaQueueProblems）。消费者本身没有异常，纯粹是"
            "生产吞吐超过了当前消费能力，属于容量不匹配导致的积压。",
        ]
        alert = _make_alert(
            alertname="KafkaConsumerLag",
            service="kafka",
            job="kafka",
            severity=r["severity"],
            summary=f"Kafka 消费组 {r['group']} 在 {r['topic']} 上的积压持续增长",
            description=queue_desc_variants[idx % len(queue_desc_variants)],
            starts_at=starts_at,
            scenario_hint=(
                f"[参数化扩增/queue_backlog] topic={r['topic']} lag={r['lag']} severity={r['severity']}："
                "验证队列积压场景在不同 topic/lag 规模下的泛化"
            ),
        )
        out.append((alert, "resource"))

    leak_recipes = _sample(_deploy_regression_recipes(), counts[3], rng)
    for idx, r in enumerate(leak_recipes):
        mech = r["mechanism"]
        starts_at = _iso("2026-07-04T09:00:00Z", idx * 45)
        leak_desc_variants = [
            "recommendation（Python/gRPC）容器的工作集内存在负载下单调上升，重启次数不断"
            f"增加（被 OOMKilled）。问题从最近一次发版之后立刻开始，{mech['clue']}。",
            "recommendation 服务的内存占用最近持续走高，容器频繁被 OOMKilled 重启。"
            f"时间线上正好卡在最近一次发版之后，{mech['clue']}。",
            "观察到 recommendation（Python/gRPC）在负载下内存不断攀升，最终触发 OOM "
            f"被强制重启，且重启频率还在增加。这个趋势是从最新一次发版开始出现的，{mech['clue']}。",
        ]
        alert = _make_alert(
            alertname="MemoryLeakOOM",
            service="recommendation",
            job="recommendation",
            severity=r["severity"],
            summary="recommendation 服务内存持续上涨、趋向 OOM",
            description=leak_desc_variants[idx % len(leak_desc_variants)],
            starts_at=starts_at,
            scenario_hint=(
                f"[参数化扩增/deploy_regression] 泄漏机制={mech['name']}（{mech['slug']}）："
                "内存泄漏 capstone 的泄漏点变体，对应 scripts/fixtures/recommendation 的不同 fixture 版本"
            ),
        )
        out.append((alert, "deploy_regression"))

    return out


# ---------------------------------------------------------------------------
# 路径 2：flagd 故障注入组合
# ---------------------------------------------------------------------------


def _flagd_single_specs() -> list[dict[str, Any]]:
    return [
        {
            "flag": "recommendationCacheFailure",
            "kind": "dependency",
            "alertname": "RecommendationCacheFailure",
            "summary_tpl": "recommendation 服务缓存层失效",
            "desc_tpl": (
                "recommendation 服务的缓存层失效（recommendationCacheFailure 开关），"
                "缓存 miss 后请求全部穿透到下游算力更重的路径，接口延迟明显上升，"
                "但服务本身进程健康，没有资源耗尽的迹象。"
            ),
        },
        {
            "flag": "adFailure",
            "kind": "dependency",
            "alertname": "AdServiceFailure",
            "summary_tpl": "ad 服务整体请求失败",
            "desc_tpl": (
                "ad 服务对广告请求整体返回失败（adFailure 开关），上游 frontend 调用 ad "
                "接口大量报错，ad 服务自身 CPU/内存均在正常范围，判断是功能开关级故障而非资源问题。"
            ),
        },
        {
            "flag": "imageSlowLoad",
            "kind": "config",
            "alertname": "FrontendImageSlowLoad",
            "summary_tpl": "frontend 图片加载延迟异常升高",
            "desc_tpl": (
                "frontend 首页图片加载耗时显著升高（imageSlowLoad 开关被设为非 off 档），"
                "该行为完全由一次功能开关配置变更触发，没有对应的代码发版，也没有下游依赖报错"
                "或资源占用异常，纯粹是配置项本身的取值问题。"
            ),
        },
        {
            "flag": "loadGeneratorFloodHomepage",
            "kind": "resource",
            "alertname": "FrontendTrafficFlood",
            "summary_tpl": "frontend 首页流量骤增导致过载",
            "desc_tpl": (
                "frontend 首页请求量在短时间内暴增（loadGeneratorFloodHomepage 开关），"
                "实例 CPU 与连接数迅速逼近上限，属于流量驱动的资源型过载，非代码回归。"
            ),
        },
        {
            "flag": "adManualGc",
            "kind": "resource",
            "alertname": "AdServiceGcPause",
            "summary_tpl": "ad 服务出现明显的 GC 停顿",
            "desc_tpl": (
                "ad 服务被强制触发完整的手动 GC（adManualGc 开关），GC 期间请求延迟出现"
                "明显毛刺，与持续型的高 CPU（adHighCpu）不同，这里是周期性的停顿而非持续打满。"
            ),
        },
    ]


def _flagd_multi_combo_specs() -> list[dict[str, Any]]:
    return [
        {
            "flags": ["productCatalogFailure", "paymentFailure"],
            "kind": "dependency",
            "service": "checkout",
            "job": "checkoutservice",
            "alertname": "CheckoutCascadingDependencyFailure",
            "summary": "checkout 下单链路多个依赖同时异常",
            "desc": (
                "checkout 服务在下单链路上同时命中两个依赖异常：product-catalog 对部分商品"
                "返回失败（productCatalogFailure），payment 按比例拒绝扣款（paymentFailure）。"
                "两个故障叠加导致下单成功率大幅下降，checkout 自身没有代码发版，属于多点依赖"
                "故障组合。"
            ),
        },
        {
            "flags": ["adHighCpu", "adManualGc"],
            "kind": "resource",
            "service": "ad",
            "job": "adservice",
            "alertname": "AdServiceCompoundResourcePressure",
            "summary": "ad 服务同时出现高 CPU 与频繁 GC 停顿",
            "desc": (
                "ad 服务同时命中持续高 CPU（adHighCpu）与周期性手动 GC（adManualGc）两个"
                "故障开关，CPU 使用率长期贴近上限、且叠加周期性 GC 停顿导致延迟毛刺更密集，"
                "属于同一服务上的复合资源型故障，无代码发版。"
            ),
        },
        {
            "flags": ["kafkaQueueProblems", "recommendationCacheFailure"],
            "kind": "resource",
            "service": "recommendation",
            "job": "recommendation",
            "alertname": "RecommendationDegradedByQueueAndCache",
            "summary": "recommendation 因队列积压叠加缓存失效双重承压",
            "desc": (
                "kafka 消费 lag 持续堆积（kafkaQueueProblems），同时 recommendation 自身"
                "缓存层失效（recommendationCacheFailure），两者叠加导致 recommendation "
                "既要处理滞后的事件流又要承担缓存穿透后的算力压力，接口延迟和超时率同步上升，"
                "CPU/内存本身没有明显异常，判断是跨服务的复合资源型压力而非单点故障。"
            ),
        },
        {
            "flags": ["cartFailure", "paymentUnreachable"],
            "kind": "dependency",
            "service": "checkout",
            "job": "checkoutservice",
            "alertname": "CheckoutFullyDown",
            "summary": "checkout 下单链路整体不可用",
            "desc": (
                "cart 服务失败（cartFailure）与 payment 服务整体不可达（paymentUnreachable）"
                "同时发生，checkout 调用这两个依赖均报错，下单链路事实上完全不可用。"
                "两个开关分属不同服务，checkout 自身代码没有变更，属于多点依赖故障组合导致的"
                "全链路失败。"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# 多点组合的机械遍历：对 CONFIRMED_FLAGD_FLAGS 真的做 itertools.combinations，
# 而不是只靠上面 4 条手写的语义化组合。
# ---------------------------------------------------------------------------

# 每个 flag 单独拿出来时「本来就该对应哪个 kind」——取值口径跟
# `_flagd_single_specs()` / `_flagd_multi_combo_specs()` 里已经出现过的用法完全对齐
# （比如 recommendationCacheFailure / adFailure / cartFailure / paymentUnreachable /
# productCatalogFailure / paymentFailure 在手写 spec 里都是 kind="dependency"；
# adHighCpu / adManualGc / kafkaQueueProblems / loadGeneratorFloodHomepage 都是
# kind="resource"；imageSlowLoad 是 kind="config"）。机械遍历组合的 kind 用下面
# `_infer_combo_kind()` 从这张表按优先级推导，不是瞎猜。
_FLAG_NATURAL_KIND: dict[str, str] = {
    "productCatalogFailure": "dependency",
    "recommendationCacheFailure": "dependency",
    "adManualGc": "resource",
    "adHighCpu": "resource",
    "adFailure": "dependency",
    "kafkaQueueProblems": "resource",
    "cartFailure": "dependency",
    "paymentFailure": "dependency",
    "paymentUnreachable": "dependency",
    "loadGeneratorFloodHomepage": "resource",
    "imageSlowLoad": "config",
}
assert set(_FLAG_NATURAL_KIND) == set(CONFIRMED_FLAGD_FLAGS), (
    "_FLAG_NATURAL_KIND 必须覆盖 CONFIRMED_FLAGD_FLAGS 里的每一个 flag，否则机械遍历"
    "组合会漏推导出 kind"
)

# 混在一个组合里会自相矛盾的 flag 对——不是「凡是同一个 service 就不能组合」（比如
# adHighCpu+adManualGc 就是合理的同服务复合资源故障，`_flagd_multi_combo_specs()`
# 里已经这么用），而是这几对具体语义直接互相否定：
#   - paymentFailure（"按比例拒绝扣款请求"，服务仍在跑，只是部分请求失败）
#     vs paymentUnreachable（"整体不可达"）：一旦整体不可达，就没有"按比例"这件事
#     可言了，两者互斥。
#   - adFailure（"ad 服务整体失败"，服务已经宕掉）vs adHighCpu / adManualGc
#     （因果故事的前提都是"服务还在正常处理请求，只是资源开销异常"）：跟"已经宕掉"
#     矛盾，不能共存。
#   - loadGeneratorFloodHomepage（流量型资源过载，CPU/连接数被打满）vs imageSlowLoad
#     （`_flagd_single_specs()` 里明确写了"没有资源占用异常，纯粹是配置项取值问题"）：
#     两者对"frontend 有没有资源异常"给出互相矛盾的结论。
_MUTUALLY_EXCLUSIVE_FLAG_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"paymentFailure", "paymentUnreachable"}),
        frozenset({"adFailure", "adHighCpu"}),
        frozenset({"adFailure", "adManualGc"}),
        frozenset({"loadGeneratorFloodHomepage", "imageSlowLoad"}),
    }
)

# 语义上代表"整个服务基本打不通"的 flag（跟只是部分失败/资源承压但服务仍在跑的 flag
# 不同档次）。已有的手写多点组合 cartFailure+paymentUnreachable 就是 2 个 hard-down
# flag 同时出现（因果故事是"checkout 下单链路两个依赖同时整体不可用"，讲得通）；这里
# 保留这个上限——三个互不相关的服务同时被判定"整体打不通"更像是"环境被同时炸了三次"，
# 而不是一个单一根因能讲通的故事，超过 2 个就过滤掉。
HARD_DOWN_FLAGS: frozenset[str] = frozenset({"adFailure", "cartFailure", "paymentUnreachable"})

_KIND_PRECEDENCE = ("resource", "dependency", "config")


def _infer_combo_kind(flags: tuple[str, ...]) -> str:
    """多个 flag 各自的 natural kind 不一致时按优先级取一个代表整条组合的 kind。

    优先级 resource > dependency > config 唯一的直接依据是手写组合
    `_flagd_multi_combo_specs()` 里 kafkaQueueProblems（resource）+
    recommendationCacheFailure（dependency）混合后被判定 kind="resource" 这一条
    先例——resource 赢过 dependency。config 目前没有跟其它 kind 混合的先例，这里外推
    成优先级最低（一个纯配置类 flag 跟一个会导致真实资源/依赖异常的 flag 同时出现时，
    更明显的资源/依赖故事应该盖过纯配置故事）。
    """
    kinds_present = {_FLAG_NATURAL_KIND[f] for f in flags}
    for k in _KIND_PRECEDENCE:
        if k in kinds_present:
            return k
    raise AssertionError(f"unreachable: unknown kind set {kinds_present}")  # pragma: no cover


def _is_semantically_valid_combo(flags: tuple[str, ...]) -> bool:
    """判断一组 flag 组合是不是「讲得通的因果故事」。

    过滤掉：
      1. 命中 `_MUTUALLY_EXCLUSIVE_FLAG_PAIRS` 里任意一对互斥 flag 的组合。
      2. 同时命中 3 个及以上 `HARD_DOWN_FLAGS`（过量、导致「环境彻底不可用」的组合）。
    """
    flag_set = set(flags)
    for pair in _MUTUALLY_EXCLUSIVE_FLAG_PAIRS:
        if pair.issubset(flag_set):
            return False
    if len(flag_set & HARD_DOWN_FLAGS) >= 3:
        return False
    return True


def _flagd_mechanical_combo_specs() -> list[dict[str, Any]]:
    """对 `CONFIRMED_FLAGD_FLAGS` 的 key 做真正的 `itertools.combinations` 机械遍历
    （r=2 即两两组合、r=3 即三三组合），过滤掉 `_is_semantically_valid_combo()` 判定
    不合理的组合，剩下的构造成跟 `_flagd_multi_combo_specs()` 完全同构的 spec 字典
    （flags/kind/service/job/alertname/summary/desc），供 `_flagd_multi_combo_pool()`
    追加进候选池。

    跟手写组合不同，这里不追求每条都有精心设计的叙事文案——机械遍历产出的组合数量本身
    就是这条路径要扩容的目标；desc 用统一模板把「哪些 flag 同时命中、各自什么语义、
    涉及哪些 service」讲清楚，信息量跟手写版一致，只是不逐条定制细节。

    alert 顶层只能挂一个 service/job，多 flag 涉及多个不同 service 时，取组合里第一个
    flag（按 `CONFIRMED_FLAGD_FLAGS` 的字典插入顺序，即 itertools.combinations 的遍历
    顺序，是确定性的）对应的 service/job 作为落点——这是刻意选择的「不需要业务知识」的
    确定性规则，不去猜「这种跨服务组合到底该算哪个服务的告警」，那类判断留给手写
    `_flagd_multi_combo_specs()` 里已有的、经过人工设计叙事的组合。
    """
    flag_names = list(CONFIRMED_FLAGD_FLAGS.keys())
    specs: list[dict[str, Any]] = []
    for r in (2, 3):
        for combo in itertools.combinations(flag_names, r):
            if not _is_semantically_valid_combo(combo):
                continue
            infos = [CONFIRMED_FLAGD_FLAGS[f] for f in combo]
            services = sorted({info["service"] for info in infos})
            primary_service, primary_job = infos[0]["service"], infos[0]["job"]
            kind = _infer_combo_kind(combo)
            flag_descs = "；".join(f"{f}（{CONFIRMED_FLAGD_FLAGS[f]['desc']}）" for f in combo)
            desc = (
                f"同一时间窗口内命中 {len(combo)} 个 flagd 故障开关：{flag_descs}。"
                f"涉及服务：{'、'.join(services)}。这条组合来自对 CONFIRMED_FLAGD_FLAGS 做 "
                f"itertools.combinations(r={len(combo)}) 的机械遍历，过滤掉同服务互斥/"
                "过量 hard-down 组合后剩下的候选，用来扩大 flagd_combination 路径的组合"
                "多样性，不是手写的定制叙事。"
            )
            specs.append(
                {
                    "flags": list(combo),
                    "kind": kind,
                    "service": primary_service,
                    "job": primary_job,
                    "alertname": "FlagdMechanicalCombo_" + "_".join(combo),
                    "summary": f"{'/'.join(services)} 同时命中 {len(combo)} 个 flagd 故障开关",
                    "desc": desc,
                }
            )
    return specs


def _flagd_multi_combo_pool() -> list[dict[str, Any]]:
    """多点组合的完整候选池：手写的语义化组合 + itertools.combinations 机械遍历后过滤
    出的组合。`generate_flagd_combinations()` 和测试都从这个函数取池子，保证两边看到
    的候选集合一致，不会各算各的。"""
    return _flagd_multi_combo_specs() + _flagd_mechanical_combo_specs()


def generate_flagd_combinations(n: int, rng: random.Random) -> list[tuple[dict[str, Any], str]]:
    """flagd 故障注入组合：从 CONFIRMED_FLAGD_FLAGS 里取单点 / 多点组合。

    单点组合直接用某一个 flag 的语义构造一条场景；多点组合的候选池见
    `_flagd_multi_combo_pool()`——既有手写的「讲得通因果故事」的语义化搭配，也有对
    `CONFIRMED_FLAGD_FLAGS` 做 itertools.combinations 机械遍历后过滤出的组合。
    """
    singles = _flagd_single_specs()
    multis = _flagd_multi_combo_pool()
    counts = _split_counts(n, [len(singles), len(multis)])
    out: list[tuple[dict[str, Any], str]] = []

    chosen_singles = _sample(singles, counts[0], rng)
    for i, spec in enumerate(chosen_singles):
        flag_info = CONFIRMED_FLAGD_FLAGS[spec["flag"]]
        starts_at = _iso("2026-07-10T10:00:00Z", i * 7)
        alert = _make_alert(
            alertname=spec["alertname"],
            service=flag_info["service"],
            job=flag_info["job"],
            severity="warning",
            summary=spec["summary_tpl"],
            description=spec["desc_tpl"],
            starts_at=starts_at,
            scenario_hint=(
                f"[flagd组合/单点] flag={spec['flag']}：单一 flagd 故障开关构造的"
                f"{spec['kind']} 场景"
            ),
        )
        out.append((alert, spec["kind"]))

    chosen_multis = _sample(multis, counts[1], rng)
    for i, spec in enumerate(chosen_multis):
        starts_at = _iso("2026-07-11T10:00:00Z", i * 11)
        alert = _make_alert(
            alertname=spec["alertname"],
            service=spec["service"],
            job=spec["job"],
            severity="critical",
            summary=spec["summary"],
            description=spec["desc"],
            starts_at=starts_at,
            scenario_hint=(
                f"[flagd组合/多点] flags={'+'.join(spec['flags'])}：多个 flagd 故障开关"
                f"同时注入构造的复合 {spec['kind']} 场景"
            ),
        )
        out.append((alert, spec["kind"]))

    return out


# ---------------------------------------------------------------------------
# 路径 3：历史工单反演（真实 Milvus 检索 + 静态兜底）
# ---------------------------------------------------------------------------

# service -> job 的兜底映射：Milvus 里存的 `suspect_service` 字段只有服务名，没有 job
# 名（`AIops-agent/agent/integrations/memory.py` 的 `store_ticket()` 压根不写 job 字段
# 进集合）。这里补全跟 `data/seeds/backfill_historical_tickets.py` 的 `SERVICE_JOB_MAP`
# 完全一致的 11 个 otel-demo 服务 + kafka 的 service -> job 对照表。
_SERVICE_JOB_FALLBACK: dict[str, str] = {
    "product-catalog": "product-catalog",
    "recommendation": "recommendation",
    "ad": "adservice",
    "kafka": "kafka",
    "cart": "cartservice",
    "frontend": "frontend",
    "checkout": "checkoutservice",
    "payment": "paymentservice",
    "currency": "currencyservice",
    "shipping": "shippingservice",
    "quote": "quoteservice",
    "email": "emailservice",
}

# 一批差异化的中文查询语句，覆盖 dependency/resource/deploy_regression/config 四类 kind
# 与尽量多的服务，用来对 Milvus `aiops_tickets` 集合做多样化检索（不是只用一条固定查询——
# 单条固定查询只能检索出语义最贴近它的那一小簇工单，覆盖不到集合里其它类别/服务的工单）。
# 跟 `data/seeds/backfill_historical_tickets.py` 回填的语料在关键词口径上对齐
# （超时/线程池/级联对应 dependency，CPU/配置变更对应 resource_cpu，consumer lag 对应
# 队列积压，OOM/发版对应 deploy_regression，证书/开关对应 config）。
_MILVUS_QUERY_POOL: list[str] = [
    "依赖调用超时 线程池打满 级联超时",
    "product-catalog 批量导入脚本 下游查询超时",
    "recommendation 离线预计算任务 超时 线程池占满",
    "cart 对账批处理脚本 超时 线程池打满",
    "checkout 反欺诈评分调用 依赖超时",
    "checkout 运费报价调用 依赖超时",
    "payment 风控调用 重试 连接池耗尽",
    "ad 用户画像查询 依赖超时",
    "currency 外部汇率服务商 依赖超时",
    "shipping quote 运费计算 依赖超时",
    "email SMTP 中转网关 依赖超时",
    "frontend 聚合调用 依赖超时",
    "CPU 使用率持续偏高 配置变更 无代码发版",
    "GC 触发频率 配置变更 CPU 偏高",
    "序列化开销 配置变更 CPU 偏高",
    "内部缓存清理频率 配置变更 CPU 偏高",
    "汇率缓存刷新间隔 配置错误 CPU 偏高",
    "模糊搜索开关误开启 CPU 偏高",
    "会话合并触发频率 配置变更 CPU 偏高",
    "重试策略过于激进 配置变更 CPU 偏高",
    "推理批大小 配置变更 CPU 偏高",
    "正则回溯 配置变更 CPU 偏高",
    "frontend 首页流量洪峰 资源过载",
    "ad 服务周期性 GC 停顿",
    "kafka consumer lag 副本数误缩容",
    "kafka consumer lag 生产速率突增",
    "kafka 消费者 下游限流 lag 堆积",
    "内存泄漏 OOM 最近一次发版 无界 list append",
    "内存泄漏 OOM 最近一次发版 无界 dict 累加",
    "内存泄漏 OOM 最近一次发版 未关闭文件句柄",
    "内存泄漏 OOM 最近一次发版 WeakSet 用错",
    "recommendation 内存泄漏 代码修复 PR",
    "证书轮换后信任链未同步更新 握手失败",
    "功能开关误配置 无代码发版",
    "混合根因 自动止血 同时代码修复",
    "瞬时抖动 计划内批处理 无需处置",
]


def _aiops_agent_venv_python() -> Optional[Path]:
    """定位 `AIops-agent/.venv/bin/python3`。找不到就返回 None（调用方据此降级）。"""
    candidate = AIOPS_AGENT_DIR / ".venv" / "bin" / "python3"
    if candidate.is_file():
        return candidate
    return None


# 子进程里跑的检索脚本：在 AIops-agent 目录下（cwd）用它自己的 python 解释器
# import 真实的 `agent.integrations.memory`，对每条查询调 `search_tickets()`，
# 把结果汇总成一个 JSON 数组打到 stdout。任何单条查询失败都不影响其它查询继续跑；
# 如果连 import agent.integrations.memory 都失败（比如 AIops-agent 目录结构变了），
# 直接打印空数组，不让子进程带着 traceback 非零退出。
_MILVUS_SEARCH_CHILD_SCRIPT = """
import json, sys
try:
    from agent.integrations import memory
except Exception:
    print("[]")
    sys.exit(0)
queries = json.loads(sys.argv[1])
top_k = int(sys.argv[2])
out = []
for q in queries:
    try:
        hits = memory.search_tickets(q, top_k=top_k)
    except Exception:
        hits = []
    out.extend(hits)
print(json.dumps(out, ensure_ascii=False))
"""


def _search_historical_tickets_via_milvus(
    queries: list[str],
    *,
    top_k_per_query: int = 5,
    timeout_s: float = 60.0,
) -> list[dict[str, Any]]:
    """通过子进程调用 `AIops-agent/.venv` 里真实的
    `agent.integrations.memory.search_tickets()`，用多条差异化查询语句检索 Milvus
    历史工单集合，返回原始命中记录列表（未去重、未过滤）。

    为什么走子进程而不是在本进程里 `sys.path` 直接 import `agent.integrations.memory`：
    本仓库自己的 venv 没装 pymilvus（只装了 sentence-transformers 用于别处的语义
    去重），而 `AIops-agent/.venv` 里 pymilvus + sentence-transformers 都装了——用子
    进程调用它自己的解释器，既复用了它真实的检索逻辑（真实 embedding + 真实 Milvus
    查询），又不需要把 pymilvus 这个偏重的依赖装进本仓库的 venv。

    贯彻本项目「流程永不崩溃」的统一模式（参照 `data/clean/dedup.py` 的语义去重 vs
    词面兜底）：任何失败——找不到 AIops-agent venv、Milvus 没起、子进程超时、非零
    退出、stdout 不是合法 JSON——都吞掉异常返回空列表，调用方 `_milvus_candidate_records()`
    据此判断要不要退回静态兜底列表 `HISTORICAL_INCIDENT_SUMMARIES`，绝不让种子生成
    流程因为 Milvus 不可用而崩溃或挂起。
    """
    python = _aiops_agent_venv_python()
    if python is None:
        return []
    try:
        proc = subprocess.run(
            [str(python), "-c", _MILVUS_SEARCH_CHILD_SCRIPT, json.dumps(queries), str(top_k_per_query)],
            cwd=str(AIOPS_AGENT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception:  # noqa: BLE001  # 包括 subprocess 找不到解释器、超时等一切异常
        return []
    if proc.returncode != 0:
        return []
    try:
        data = json.loads((proc.stdout or "").strip() or "[]")
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _milvus_hit_to_ticket_record(hit: dict[str, Any], idx: int) -> Optional[dict[str, Any]]:
    """把 `search_tickets()` 返回的一条命中记录，转成跟 `HISTORICAL_INCIDENT_SUMMARIES`
    里手写记录同构的字典（id/summary/kind/service/job/severity），供
    `reverse_derive_from_incident_summary()` 无差别消费。命中记录缺 kind/service/summary、
    kind 不在 `KIND_VALUES` 里、或 service 不是已知的 11 个 otel-demo 服务 + kafka 之一
    （集合里混有更早一批失败收集run留下的低质量记录，service 是 "unknown" 甚至畸形字符串，
    见任务背景里核实过的 46 条遗留记录）都判为无效，返回 None——只信任 service 落在
    `_SERVICE_JOB_FALLBACK` 这个已核实的服务枚举里的命中。
    """
    service = (hit.get("suspect_service") or "").strip()
    kind = (hit.get("kind") or "").strip()
    summary = (hit.get("summary") or "").strip()
    if not service or kind not in KIND_VALUES or not summary or service not in _SERVICE_JOB_FALLBACK:
        return None
    job = _SERVICE_JOB_FALLBACK.get(service, service)
    route = (hit.get("route") or "").strip()
    # Milvus 里没存 severity 字段，按 kind/route 做跟 backfill 脚本一致的启发式推断：
    # deploy_regression 或已经真实触发过自动执行/代码修复的工单，判定为更严重的 critical。
    severity = (
        "critical"
        if kind == "deploy_regression"
        or route in {"auto_remediated", "auto_remediated_and_code_fix_pr", "code_fix_pr"}
        else "warning"
    )
    return {
        "id": f"milvus-{idx:04d}-{service}-{kind}",
        "summary": summary,
        "kind": kind,
        "service": service,
        "job": job,
        "severity": severity,
    }


def _milvus_candidate_records() -> list[dict[str, Any]]:
    """真实检索池：用 `_MILVUS_QUERY_POOL` 的所有查询检索 Milvus，去重 + 过滤无效命中，
    转成跟 `HISTORICAL_INCIDENT_SUMMARIES` 同构的记录列表。

    去重用 (service, kind, summary 前 80 字) 做 key——`search_tickets()` 的返回值本身
    不带 fingerprint 字段（`memory.search_tickets()` 没有把它塞进输出字典），所以没法
    按 fingerprint 去重，退而用摘要文本的前缀近似判断「是不是同一条工单」。

    任何异常都吞掉返回空列表（不应该发生，因为 `_search_historical_tickets_via_milvus()`
    自己已经不抛异常，这里是双重保险）。
    """
    try:
        raw_hits = _search_historical_tickets_via_milvus(_MILVUS_QUERY_POOL)
    except Exception:  # noqa: BLE001
        raw_hits = []

    seen: set[tuple[str, str, str]] = set()
    records: list[dict[str, Any]] = []
    for hit in raw_hits:
        if not isinstance(hit, dict):
            continue
        service = (hit.get("suspect_service") or "").strip()
        kind = (hit.get("kind") or "").strip()
        summary = (hit.get("summary") or "").strip()
        key = (service, kind, summary[:80])
        if key in seen:
            continue
        seen.add(key)
        rec = _milvus_hit_to_ticket_record(hit, len(records))
        if rec is not None:
            records.append(rec)
    return records


def reverse_derive_from_incident_summary(
    record: dict[str, Any], offset_minutes: int, *, source: str = "static"
) -> dict[str, Any]:
    """从一条「历史工单摘要」反向抽出一条告警文本。

    `record` 可能来自两个来源（`source` 参数标注是哪个，落进 `scenario_hint`）：
      - `source="milvus"`：真的从 `AIops-agent/agent/integrations/memory.py` 管理的
        Milvus 历史工单集合检索出来的真实工单（经 `_milvus_hit_to_ticket_record()`
        转换成同构字典），见 `_milvus_candidate_records()`。
      - `source="static"`：`HISTORICAL_INCIDENT_SUMMARIES` 这份手写的「历史工单摘要」
        样例（字段结构模拟 Milvus 里一条工单记录：summary/kind/service/job/severity），
        只在真实检索池为空或不够大时才会用到——见 `generate_historical_reverse_derivation()`。

    两种来源的记录字段结构完全一致，本函数「摘要 -> 告警」的渲染逻辑对两者无差别。
    """
    starts_at = _iso("2026-07-20T08:00:00Z", offset_minutes)
    source_note = (
        "从 AIops-agent 的 Milvus 历史工单集合检索出的真实工单反向抽取的告警"
        if source == "milvus"
        else "从手写历史工单摘要样例反向抽取的告警（Milvus 检索池不足时的静态兜底）"
    )
    alert = _make_alert(
        alertname="HistoricalTicketReplay",
        service=record["service"],
        job=record["job"],
        severity=record["severity"],
        summary=f"历史工单反演：{record['id']}",
        description=record["summary"],
        starts_at=starts_at,
        scenario_hint=f"[历史工单反演/{source}] source_ticket={record['id']}：{source_note}",
    )
    return alert


def generate_historical_reverse_derivation(n: int, rng: random.Random) -> list[tuple[dict[str, Any], str]]:
    """历史工单反演：优先用 Milvus 真实检索池，只在池子不够大时用静态列表补齐差额。

    - 真实池条数 >= n：全部从真实池采样（不重复，`_sample()` 只在池子不够大时才循环
      补足，这里够大就是纯粷洗牌取前 n 个）。
    - 真实池非空但条数 < n：真实池全部用上，差额从 `HISTORICAL_INCIDENT_SUMMARIES`
      补齐——「优先用真实检索池，不够才退回静态兜底」，不是「有真实数据就完全不用静态」。
    - 真实池为空（Milvus 不可用/子进程失败/集合里没有匹配的有效记录）：完全退回
      `HISTORICAL_INCIDENT_SUMMARIES`，行为跟改造前完全一致。
    """
    real_pool = list(_milvus_candidate_records())
    rng.shuffle(real_pool)

    if len(real_pool) >= n:
        chosen: list[dict[str, Any]] = real_pool[:n]
        sources: list[str] = ["milvus"] * n
    elif real_pool:
        remaining = n - len(real_pool)
        static_fill = _sample(HISTORICAL_INCIDENT_SUMMARIES, remaining, rng)
        chosen = real_pool + static_fill
        sources = ["milvus"] * len(real_pool) + ["static"] * remaining
    else:
        chosen = _sample(HISTORICAL_INCIDENT_SUMMARIES, n, rng)
        sources = ["static"] * n

    out: list[tuple[dict[str, Any], str]] = []
    for i, (record, source) in enumerate(zip(chosen, sources)):
        # 间隔 45 分钟 > DEDUP_TTL_SECONDS（30 分钟），理由同 generate_parametrized()：
        # 避免不同历史工单摘要恰好共享 (service, severity) 时被幂等去重指纹误判为重复。
        alert = reverse_derive_from_incident_summary(record, offset_minutes=i * 45, source=source)
        out.append((alert, record["kind"]))
    return out


# ---------------------------------------------------------------------------
# 2026-09 补充：手写历史工单反演种子，定点补 config kind + 整体路径量
# ---------------------------------------------------------------------------
#
# 背景（2026-09 盘点，见 data/cold_start/known_unsupported_seeds.json）：29 条已接受真实
# 轨迹里 historical_reverse_derivation 路只有 6 条，且全部是 recommendation 的
# deploy_regression/resource 内存泄漏叙事；config kind 全库只有 2 条（都不是这条路径产出的）。
# 盘点 `data/clean/split/{train,held_out}/` 现存的 8 条 historical_reverse_derivation+config
# 种子，全部映射不到任何真实可触发机制（6 条是证书轮换/信任链配置叙事——`fault_injection.py`
# 明确不支持，2 条是 recommendation A/B 开关误配叙事——落进 `_inject_historical_fault()` 的
# 关键词兜底后四类关键词桶都不命中，返回空）；盘点真实 Milvus 检索池（`_milvus_candidate_
# records()`，67 条）里的 10 条 config 记录，同样全部是证书轮换叙事，唯一例外
# `milvus-0039-frontend-config` 讲的是 imageSlowLoad 开关配置残留，机制上可行但文风是
# 自述式历史排查报告（提到具体开关行号），不适合直接当"新告警"使用。
#
# 因此这里不复用检索池条目，改成手写 5 条新历史工单摘要，直接调用 `_make_alert()`（跟
# `generate_pure_code_fix_and_info_only_v2()` 同样的做法），但 `generation_path` 仍标
# `historical_reverse_derivation`（不是 `parametrized`）——这些确实是"历史工单反向抽取告警"
# 这一叙事形态，只是手写而非从检索池采样，跟 `HISTORICAL_INCIDENT_SUMMARIES` 手写静态兜底池
# 属于同一性质。scenario_hint 里除了保留 `[历史工单反演/static] source_ticket=` 前缀（跟
# `reverse_derive_from_incident_summary()` 格式一致，保持向后可解析），额外显式标注
# `flag=`——`data/cold_start/fault_injection.py::inject_alert_fault()` 对 flag=/flags= 的
# 解析优先级高于关键词兜底（`_inject_historical_fault()`），这样可以绕开该函数已知的两个
# 关键词误判 bug（`_HIST_QUEUE_KEYWORDS` 里的 "lag" 是 "flag"/"flagd" 的子串会误命中；
# 通用词"堆积"在 `_HIST_TIMEOUT_KEYWORDS` 之前被检查，会把跟"堆积"无关的超时叙事误路由到
# kafka），可靠命中已实测验证过的机制，不依赖脆弱的中文关键词猜测。
#
# 5 条覆盖：2 条 config（imageSlowLoad，两种不同的"开关值残留"叙事，用于验证语义去重在
# 近似场景下的判定边界）、2 条 resource/kafka（kafkaQueueProblems，两种不同的积压成因——
# 生产端突发流量 vs 消费端网络抖动重连，跟已有静态池里 3 条"消费者组缩容"叙事的成因不同，
# 增加机制多样性）、1 条 deploy_regression/recommendation（复用已验证的内存泄漏 fixture，
# 叙事换成"定时预热任务往模块级 list 追加快照忘记清空"，跟已有 6 条"缓存字典无 TTL"叙事
# 的具体机制不同，同样是为了给这个 kind 的历史工单叙事增加多样性）。
#
# 明确不写的方向（本次盘点后主动放弃，不是遗漏）：product-catalog/cart 的
# "下游查询缺超时->线程池占满->级联超时" dependency 历史叙事——已有的两条真实同类种子
# （见 known_unsupported_seeds.json 的 `not_excluded_kept_as_reject_with_real_evidence`）
# 都已经真实采集过，结果一致：真实注入机制是 flagd 通用代理故障（productCatalogFailure/
# cartFailure，效果是即时报错），跟叙事描述的"缺超时導致慢查询占满线程池"不是同一种故障
# 签名，诊断 Agent 会如实依据实时证据判定为 online_op 而不是叙事暗示的 code_fix，可预期地
# 无法匹配 GT、被拒绝采样正确拦下——不是机制不可观测，是叙事与真实注入机制的因果签名系统性
# 不对齐，继续加同类种子只会重复已经拿到的结论，不会带来新信息，因此本轮不再新增这个方向。


def generate_historical_reverse_derivation_supplement_v1() -> list[tuple[dict[str, Any], str]]:
    """手写补 5 条 historical_reverse_derivation 种子（不参与加权采样），定点填补
    config kind 的路径覆盖，兼顾给 resource/kafka、deploy_regression/recommendation 两个
    已有覆盖的 kind 增加叙事多样性。"""
    out: list[tuple[dict[str, Any], str]] = []

    def _hist_hint(source_ticket: str, flag: str) -> str:
        return (
            f"[历史工单反演/static] source_ticket={source_ticket} flag={flag}：2026-09 补充批次，"
            "手写历史工单摘要反向抽取的告警，显式标注 flag= 保证 fault_injection.py 按 "
            "flag=/flags= 精确路由（优先级高于 _inject_historical_fault() 的关键词兜底），"
            "不依赖脆弱的中文关键词猜测"
        )

    def _hist_hint_no_flag(source_ticket: str, note: str) -> str:
        # recommendation 内存泄漏没有对应的 flagd flag（真实机制是换容器 fixture，见
        # fault_injection.py::start_recommendation_leak_fixture()），不能写 flag=——
        # parse_flags() 命中任何字面 flag= 都会当成"这个名字对应的 flagd 开关"去尝试
        # activate_flags()，一个不存在的假 flag 名会被 activate_flags() 静默跳过
        # （返回 touched=[]），反而绕过了下面本该命中的 _HIST_LEAK_KEYWORDS 关键词兜底、
        # 变成真的什么故障都没注入。这里保留纯关键词兜底路径（description 里的"内存"/"OOM"
        # 关键词会被 _inject_historical_fault() 命中，正确路由到 start_recommendation_leak_
        # fixture()），不写 flag=。
        return (
            f"[历史工单反演/static] source_ticket={source_ticket}：2026-09 补充批次，"
            f"手写历史工单摘要反向抽取的告警（{note}，无对应 flagd flag，故意不写 flag=，"
            "依赖 _inject_historical_fault() 的内存/OOM 关键词兜底路由到真实泄漏 fixture）"
        )

    debug_leftover_alert = _make_alert(
        alertname="FrontendImageLoadSlowPersisted",
        service="frontend",
        job="frontend",
        severity="warning",
        summary="监控发现 frontend 首页图片加载耗时自昨晚起显著抬升",
        description=(
            "客服反馈今天上午起陆续有用户投诉首页商品图片加载卡顿。排查 deploys.log 发现"
            "近 48 小时内 frontend 没有任何新发版记录，docker stats 显示 frontend 容器"
            "CPU/内存/重启次数都在正常范围，Jaeger 上 frontend 到下游各服务的调用链延迟也都"
            "正常，唯一异常集中在图片资源本身的加载耗时。回查 flagd 变更记录，发现昨晚有"
            "工程师为验证图片加载慢场景下的前端兜底逻辑，手动把 imageSlowLoad 这个功能开关"
            "从默认 off 改成了测试档位，验证完之后忘记改回默认值——不是代码或基础设施问题，"
            "纯粹是这个开关当前的取值本身在起作用。"
        ),
        starts_at="2026-09-06T09:00:00Z",
        scenario_hint=_hist_hint("hist-2026-09-frontend-imageslowload-debug-leftover", "imageSlowLoad"),
    )
    out.append((debug_leftover_alert, "config"))

    loadtest_drill_alert = _make_alert(
        alertname="FrontendImageLoadSlowAfterLoadTestDrill",
        service="frontend",
        job="frontend",
        severity="warning",
        summary="大促前压测演练结束后 frontend 首页图片加载仍然缓慢",
        description=(
            "运营团队昨晚组织了一次大促前的性能压测演练，其中一个环节需要人为让首页图片"
            "加载变慢以验证前端的加载中占位图效果，演练脚本通过 flagd 把 imageSlowLoad 开关"
            "打到最慢档位。演练结束后的收尾清单里没有包含\"把这个开关改回默认值\"这一步，"
            "导致演练结束、真实流量恢复后，首页图片依然按演练时的最慢档位加载。docker stats、"
            "Jaeger 调用链、deploys.log 均显示 frontend 服务本身健康、没有新发版，唯一异常"
            "来源是这个开关当前的取值。"
        ),
        starts_at="2026-09-06T09:30:00Z",
        scenario_hint=_hist_hint("hist-2026-09-frontend-imageslowload-loadtest-drill", "imageSlowLoad"),
    )
    out.append((loadtest_drill_alert, "config"))

    promo_burst_alert = _make_alert(
        alertname="KafkaPromoConsumerLagSpike",
        service="kafka",
        job="kafka",
        severity="warning",
        summary="促销活动开始后 promo-events 主题的消费延迟骤增",
        description=(
            "运营在活动开始时刻对 promo-events 主题发起了一轮批量优惠券发放任务，短时间内"
            "生产的消息量达到日常峰值的数倍，promo-consumer 消费者组的处理速率没有相应提升，"
            "consumer lag 在几分钟内迅速堆积到平时的十几倍，下游发放优惠券的处理明显滞后。"
            "容量规划记录显示 promo-consumer 副本数近期没有变化，问题根源是生产端的瞬时"
            "突发流量，不是消费端配置被误改。"
        ),
        starts_at="2026-09-06T10:00:00Z",
        scenario_hint=_hist_hint("hist-2026-09-promo-consumer-burst-lag", "kafkaQueueProblems"),
    )
    out.append((promo_burst_alert, "resource"))

    inventory_network_alert = _make_alert(
        alertname="KafkaInventoryConsumerReconnectStorm",
        service="kafka",
        job="kafka",
        severity="critical",
        summary="inventory-sync 主题消费者组反复重连，消费吞吐骤降",
        description=(
            "网络团队记录到 broker 与 inventory-consumer 消费者组之间的网络链路在过去半小时里"
            "出现间歇性抖动，消费者客户端反复触发重连和分区重新分配，每次重连期间都会短暂"
            "停止消费，inventory-sync 主题的 consumer lag 持续攀升，库存同步明显滞后。生产端"
            "发送速率在此期间没有变化，问题根源是网络链路抖动导致的消费端重连风暴，不是代码"
            "或容量配置问题。"
        ),
        starts_at="2026-09-06T10:30:00Z",
        scenario_hint=_hist_hint("hist-2026-09-inventory-consumer-network-flap", "kafkaQueueProblems"),
    )
    out.append((inventory_network_alert, "resource"))

    warmup_snapshot_leak_alert = _make_alert(
        alertname="RecommendationWarmupSnapshotMemoryGrowth",
        service="recommendation",
        job="recommendation",
        severity="critical",
        summary="recommendation 内存占用持续上升，怀疑是新上线的预热任务导致",
        description=(
            "recommendation 团队上周上线了一个定时预热任务，每隔几分钟运行一次，作用是把"
            "当前热销榜的商品快照追加进一个模块级 list 里，供后续请求快速读取，但每次追加前"
            "忘记清空上一次的旧快照，list 长度只增不减。上线以来内存占用持续单调爬升，怀疑"
            "最终会被 OOMKilled。docker stats 采样显示内存曲线平滑向上、没有阶跃，deploys.log"
            "显示这次上线是唯一的相关变更，没有伴随代码回滚。"
        ),
        starts_at="2026-09-06T11:00:00Z",
        scenario_hint=_hist_hint_no_flag(
            "hist-2026-09-recommendation-warmup-snapshot-leak", "定时预热任务无界 list 泄漏"
        ),
    )
    out.append((warmup_snapshot_leak_alert, "deploy_regression"))

    return out


# ---------------------------------------------------------------------------
# 定点补充：填补 remediation_type=code_fix / info_only 两个标签空白
# ---------------------------------------------------------------------------
#
# 冷启动采集的 25 条已接受轨迹里 100% 是 online_op，0 条 code_fix / info_only——
# 不是标注疏漏，是种子池本身缺这两类能在真实活环境里稳定复现的场景（`historical_
# reverse_derivation` 路径产出过 5 条标了 code_fix 的种子，但都是没有对应 fixture 的历史
# 工单叙事，真实采集时模型如实判定证据不足/info_only，从未真正走到 code_fix；也没有任何
# 种子路径产出过 info_only）。这里各定点补一条，不是参数化扩增（没有多组 recipe 变体可
# 采样），复用跟 generate_parametrized() 同样的 `_make_alert()` 拼装方式，manifest 里仍标
# `generation_path="parametrized"`（内容形态、schema 校验方式跟参数化扩增完全一致，只是
# 只有一条、不参与加权采样），不引入新的 generation_path 枚举值。
#
# 对应的真实注入机制见 data/cold_start/fault_injection.py 模块 docstring 里
# pure_code_fix_ranking / pure_info_only 两段说明。


def generate_pure_code_fix_and_info_only() -> list[tuple[dict[str, Any], str]]:
    """定点产出 1 条纯 code_fix + 1 条纯 info_only 种子（不参与 generate_seeds() 的加权采样）。"""
    out: list[tuple[dict[str, Any], str]] = []

    ranking_alert = _make_alert(
        alertname="RecommendationRankingOrderRegression",
        service="recommendation",
        job="recommendation",
        severity="warning",
        summary="recommendation 推荐结果排序疑似反向，热度最低的商品排在最前面",
        description=(
            "巡检脚本每 30 秒对 recommendation 服务发起一次 GET /recommend（该服务的调试端口"
            "已在 compose 里发布到本机 18080），连续 20+ 次采样返回的推荐列表固定是"
            "[\"PRODUCT-19\",\"PRODUCT-18\",\"PRODUCT-17\",\"PRODUCT-16\",\"PRODUCT-15\"]——"
            "按热度应优先返回 PRODUCT-2~PRODUCT-6 这类高分商品，当前结果恰好是热度最低的几个"
            "排在最前面，怀疑排序方向写反。docker stats 采样显示该容器 CPU/内存/重启次数均"
            "正常，Prometheus 上 recommendation 的请求错误率、延迟 p99 都在历史基线范围内，"
            "flagd 里跟 recommendation 相关的开关（recommendationCacheFailure）当前未生效，"
            "deploys.log 显示最近一次发版是 40 分钟前的一次常规特性合并，时间线吻合。没有任何"
            "基础设施信号异常，怀疑是这次发版带来的纯业务逻辑回归（排序比较函数方向写反），"
            "无法通过重启/扩容/回滚基础设施解决。"
        ),
        starts_at="2026-07-05T09:00:00Z",
        scenario_hint=(
            "[参数化扩增/pure_code_fix_ranking] fixture_port=18080：真实 fixture，"
            "recommendation 容器换成 recommendation-ranking-fixture:test 镜像，"
            "ranking.py::rank_by_score() 的 sorted() 排序方向写反，"
            "纯代码问题，不需要也不能通过任何线上操作修复"
        ),
    )
    out.append((ranking_alert, "deploy_regression"))

    info_only_alert = _make_alert(
        alertname="RecommendationLatencyP95SlightlyElevated",
        service="recommendation",
        job="recommendation",
        severity="info",
        summary="recommendation 服务 p95 延迟出现小幅波动，尚未确认是否为真实异常",
        description=(
            "监控规则捕捉到 recommendation 服务过去 15 分钟 p95 延迟比前一小时基线略高"
            "（触发的是最低档 info 级提示，不是 warning/critical）。当前活环境里没有对"
            "recommendation 做任何故障注入或变更——docker stats 显示 CPU/内存/重启次数完全"
            "正常，Prometheus 上该服务的错误率为 0、p95/p99 延迟处于历史基线区间内的正常抖动"
            "范围，flagd 相关开关（recommendationCacheFailure 等）均为 off，deploys.log 里"
            "近 24 小时没有该服务的发版记录，Jaeger 里最近的调用链也没有异常 span。这条提示"
            "更像是监控阈值敏感度带来的噪声，而不是真实故障。"
        ),
        starts_at="2026-07-05T09:30:00Z",
        scenario_hint=(
            "[参数化扩增/pure_info_only] 服务当前真实健康，不注入任何故障：验证「证据不足/"
            "确实无异常时应如实报 info_only」这一诊断标签的空白场景"
        ),
    )
    out.append((info_only_alert, "resource"))

    return out


# ---------------------------------------------------------------------------
# 2026-09 补充：code_fix / info_only 第二批定点补充
# ---------------------------------------------------------------------------
#
# 上面 generate_pure_code_fix_and_info_only() 各只产出 1 条，冷启动实采里 code_fix/
# info_only 各只有 1 条真实通过的轨迹，统计上太薄。这里再补 2 条 code_fix（1 条新机制
# recommendation-dedupe + 1 条复用已验证的 ranking fixture、换一套告警叙事）+ 3 条
# info_only（覆盖 dependency/config/deploy_regression 三种 kind，resource kind 已经
# 有上面那条），同样不参与加权采样，`generation_path` 仍标 `parametrized`。
#
# 对应的真实注入机制见 data/cold_start/fault_injection.py 模块 docstring 里
# pure_code_fix_dedupe 一段说明；3 条新 info_only 不引入新机制，直接复用 pure_info_only
# 标签（活环境此刻确实健康，不注入任何故障）。


def generate_pure_code_fix_and_info_only_v2() -> list[tuple[dict[str, Any], str]]:
    """再定点补 2 条 code_fix + 3 条 info_only（不参与加权采样）。"""
    out: list[tuple[dict[str, Any], str]] = []

    dedupe_alert = _make_alert(
        alertname="RecommendationDuplicateResultsLive",
        service="recommendation",
        job="recommendation",
        severity="warning",
        summary="recommendation /recommend 接口返回列表里出现重复商品 ID",
        description=(
            "巡检脚本连续 10+ 次调用 recommendation 服务的 GET /recommend（调试端口已发布到"
            "本机 18081），每次返回的推荐列表都固定是 [\"PRODUCT-2\",\"PRODUCT-3\",\"PRODUCT-0\","
            "\"PRODUCT-3\",\"PRODUCT-7\"]——同一个商品 PRODUCT-3 在 5 条结果里出现了两次。"
            "docker stats 显示该容器 CPU/内存/重启次数完全正常，Prometheus 上 recommendation"
            "的错误率、延迟 p99 都在历史基线范围内，flagd 相关开关"
            "（recommendationCacheFailure）当前未生效。据反馈这是团队正在灰度的一个"
            "\"目录候选+运营curated热门候选\"合并去重的新功能（feature/dedupe 分支，还没合并"
            "进 master），怀疑合并两路候选后的去重步骤没生效。没有任何基础设施信号异常，"
            "怀疑是纯业务逻辑 bug（去重函数没起作用），无法通过重启/扩容/回滚基础设施解决。"
        ),
        starts_at="2026-09-01T10:00:00Z",
        scenario_hint=(
            "[参数化扩增/pure_code_fix_dedupe] fixture_port=18081：真实 fixture，"
            "recommendation 容器换成 recommendation-dedupe-fixture:test 镜像，"
            "dedupe.py::dedupe_ids() 用 .lower() 算查找 key 但存的是原始"
            "大小写 key，membership 检查永远命不中，纯代码问题，不需要也不能通过任何线上"
            "操作修复"
        ),
    )
    out.append((dedupe_alert, "deploy_regression"))

    ranking_v2_alert = _make_alert(
        alertname="RecommendationTopKAlwaysLowestScored",
        service="recommendation",
        job="recommendation",
        severity="warning",
        summary="客服反馈 recommendation 首页推荐区总是展示最冷门的商品",
        description=(
            "客服团队反馈用户投诉首页推荐区总是展示冷门商品；工程师对 recommendation 服务的"
            "调试端口（本机 18080）连续发起 20 次 GET /recommend，返回结果稳定是"
            "[\"PRODUCT-19\",\"PRODUCT-18\",\"PRODUCT-17\",\"PRODUCT-16\",\"PRODUCT-15\"]，"
            "跟按热度应返回的 PRODUCT-2~PRODUCT-6 恰好相反。docker stats、日志显示容器本身"
            "健康，Prometheus 错误率、延迟都正常，deploys.log 显示最近一次发版是当天早些"
            "时候的一次常规特性合并。没有基础设施异常，怀疑是这次发版把排序比较函数的方向"
            "写反了，纯代码逻辑问题，无法通过重启/扩容/回滚基础设施解决。"
        ),
        starts_at="2026-09-01T10:30:00Z",
        scenario_hint=(
            "[参数化扩增/pure_code_fix_ranking] fixture_port=18080：真实 fixture，"
            "recommendation 容器换成 recommendation-ranking-fixture:test 镜像（源自"
            "AIops-agent/scripts/fixtures/recommendation-ranking/，对应 AIops-agent 自带的"
            "s10_ranking 场景），ranking.py::rank_by_score() 的 sorted() 排序方向写反，"
            "纯代码问题，不需要也不能通过任何线上操作修复（跟 generate_pure_code_fix_and_"
            "info_only() 里第一条 ranking 种子复用同一个真实 fixture，只是换了一套告警叙事，"
            "验证「同一机制、不同措辞」下诊断 Agent 仍能稳定给出 code_fix）"
        ),
    )
    out.append((ranking_v2_alert, "deploy_regression"))

    dependency_info_only_alert = _make_alert(
        alertname="CartValkeyTimeoutHistoricalBlip",
        service="cart",
        job="cartservice",
        severity="warning",
        summary="监控记录到 cart 连接 valkey-cart 出现过几次超时，怀疑依赖故障",
        description=(
            "监控在过去 1 小时内记录到 3 次 cart 服务连接 valkey-cart（cart 的会话缓存依赖）"
            "超时的日志片段，怀疑存在依赖故障。复检 flagd 当前加载的配置文件，cartFailure"
            "开关的 defaultVariant 目前是 off；docker ps 显示 cart 和 valkey-cart 容器都在"
            "正常运行、没有重启记录；docker logs 最近 80 行没有任何连接失败/超时相关的"
            "ERROR；Prometheus 上 cart 服务的调用错误率为 0。怀疑这 3 次超时是短暂网络抖动"
            "留下的历史记录，此刻已经自愈，不是持续性的依赖故障。"
        ),
        starts_at="2026-09-01T11:00:00Z",
        scenario_hint=(
            "[参数化扩增/pure_info_only] kind=dependency service=cart：cartFailure 开关"
            "当前为 off，valkey-cart/cart 容器均健康，服务当前真实健康，不注入任何故障："
            "验证「历史噪音已自愈时应如实报 info_only」这一场景，覆盖 dependency 这个"
            "kind（跟已有的 pure_info_only 种子是 resource kind 不同）"
        ),
    )
    out.append((dependency_info_only_alert, "dependency"))

    config_info_only_alert = _make_alert(
        alertname="PaymentFailureFlagDriftSuspected",
        service="payment",
        job="paymentservice",
        severity="warning",
        summary="巡检怀疑上周调试用的 paymentFailure 灾备开关忘记调回 off",
        description=(
            "上周有工程师在演练环境调试过 paymentFailure 这个按百分比分级的灾备开关"
            "（100%/90%/75%/50%/25%/10%/off），巡检人员担心忘记调回 off，导致线上一部分"
            "支付请求被合成失败注入，因此产生这条巡检告警。复检 flagd 当前加载的配置文件，"
            "paymentFailure 的 defaultVariant 确认是 off；docker logs payment 最近 80 行"
            "没有任何 flagd 注入或 charge 失败的记录；docker stats 显示 payment 容器"
            "CPU/内存都在正常范围；Prometheus 上 payment 服务的调用错误率为 0。怀疑只是"
            "一次没有落地的历史顾虑，当前配置和运行状态都正常。"
        ),
        starts_at="2026-09-01T11:30:00Z",
        scenario_hint=(
            "[参数化扩增/pure_info_only] kind=config service=payment：paymentFailure"
            "开关当前为 off，payment 容器健康，服务当前真实健康，不注入任何故障：验证"
            "「怀疑配置漂移但配置其实一直正常时应如实报 info_only」这一场景，覆盖 config"
            "这个 kind"
        ),
    )
    out.append((config_info_only_alert, "config"))

    deploy_regression_info_only_alert = _make_alert(
        alertname="ShippingLatencyRegressionRumor",
        service="shipping",
        job="shippingservice",
        severity="warning",
        summary="有反馈称 shipping 今天早上发布新版本后下单变慢，怀疑发版回归",
        description=(
            "运营群里有人反映\"shipping 组今天早上发布了一个新版本，发布后下单流程感觉"
            "变慢了\"，巡检脚本据此产生这条告警。查 deploys.log 全量记录，里面只有"
            "recommendation 服务的两条历史发布记录，完全没有 shipping 的任何发布记录；"
            "docker inspect shipping 显示容器创建时间是两周前，中间没有重建/重启过；"
            "docker stats 显示 shipping 容器 CPU/内存都在很低的正常水平；Prometheus 上"
            "shipping 服务的调用错误率、延迟都在历史基线范围内。怀疑\"今天发布\"这个说法"
            "本身是误传，没有对应的真实发布事件，也没有观测到延迟异常。"
        ),
        starts_at="2026-09-01T12:00:00Z",
        scenario_hint=(
            "[参数化扩增/pure_info_only] kind=deploy_regression service=shipping："
            "deploys.log 里没有 shipping 的任何发布记录、容器也没有重建痕迹，服务当前"
            "真实健康，不注入任何故障：验证「所谓的发版回归本身是误传时应如实报"
            "info_only」这一场景，覆盖 deploy_regression 这个 kind"
        ),
    )
    out.append((deploy_regression_info_only_alert, "deploy_regression"))

    return out


# ---------------------------------------------------------------------------
# 2026-09 v9 数据增强：四家族 47 条定点种子（docs/数据增强方案.md §4）
# ---------------------------------------------------------------------------
#
# 规格（§4.1-§4.4）：A 反幻觉 13（A1 空观测 6 / A2 部分证据 4 / A3 台账纪律 3）、
# B 信号贫乏 18（B1a 交叉验证 8 / B1b 诚实低置信 4 / B2 噪声 info_only 3 / B3 陈旧告警 3）、
# C 效率 8（C1 早收敛 ≤8 步）、D code_fix 8（D1 新 bug 4 / D2 用户报障 2 / D3 对抗误导 2）。
# 合计 47 = 13 + 18 + 8 + 8。（方案文档各小节的分family数字合计为 48，本轮按总盘子 47 收口，
# 削减的是 B2「A1 姊妹型补量」1 条——它是唯一与既有 v8 info_only 覆盖最重叠的家族。）
#
# 红线（§5）：不直接拿 13 个测评场景的 alert/flag/fixture 实例当训练样本；B1a 与 s2/s3
# 同机制但叙事实例全新；D 家族只指向 master 上未被测评占用的 3 个 bug 文件
# （pagination/cache/pricing），绝不碰 ranking/dedupe/stats/泄漏（s4/s7/s10/s11/s12 在用）。
#
# 故障注入约定：A2/A3/B1a/B1b/C1 的 scenario_hint 带 flag=/flags=（fault_injection.py 按
# 此注入）；A1/B2/B3/D 带 [v9增强/...] no-op 标签（不注入；B3 的「曾发生已复位」由编排层
# 在采集前手动短暂注入并复位）。scenario_hint 会被 collect_trajectories.py 剥离，不进提示词。
#
# GT 扩展 schema（§7.2）：全部带 expect_kind_any_of；按家族带 expect_confidence_range /
# expect_suspect_file_any_of / expect_max_steps / expect_negative_claim / expect_evidence_groups /
# expect_detail_contains_any / expect_remediation_any_of / expect_route_any_of。


def _v9_seed(
    *,
    family: str,
    alert: dict[str, Any],
    intended_kind: str,
    gt: dict[str, Any],
    injection: dict[str, Any],
) -> dict[str, Any]:
    """组装一条 v9 增强种子条目：alert + manifest 元信息 + 扩展 schema 的 GT + 注入编排元数据。

    injection（collect_v9.py 消费）：
      {"mode": "flags", "flags": [...]}                       跑 Agent 前注入这些 flagd 开关
      {"mode": "pre_inject", "flag": X, "on_s": N, "drain_s": M}  采集前注入 N 秒后复位、消化 M 秒
      {"mode": "none"}                                        不注入（空观测/噪声/陈旧叙事型/仓库 bug 型）
    """
    return {
        "family": family,
        "alert": alert,
        "intended_kind": intended_kind,
        "gt": gt,
        "injection": injection,
    }


def generate_v9_augmentation() -> list[dict[str, Any]]:
    """产出 v9 数据增强的全部 47 条定点种子（不参与 generate_seeds() 的加权采样）。"""
    out: list[dict[str, Any]] = []

    neg_noise = ["无活跃", "噪声", "误报", "误传", "未发现", "未观测", "无异常", "已排除", "非真实故障",
                 "不成立", "证伪", "无需处置", "已不再", "不再复现", "已停止"]
    neg_unlocatable = [
        "未能定位", "未找到", "无法确认", "无法定位", "证据不足", "未观测到", "无法排除",
        "不能确认", "未能确认", "无法确定", "无法复现", "矛盾", "未能证实", "未真正", "存疑",
    ]
    neg_healed = [
        "已恢复", "已自愈", "已平息", "已消除", "无活跃", "当前无", "已消化", "已停止",
        "不再", "已复位", "已关闭", "已解除",
    ]
    src_second_generic = [
        "docker logs", "docker stats", "docker inspect", "docker ps", "jaeger", "Jaeger",
        ":16686", "日志", "kafka-consumer-groups", "kafka-console", "fraud-detection",
        "sleeping", "trace", "Trace", "容器",
    ]

    # ---------------- A1 空观测→如实上报（6 条，不注入故障） ----------------
    a1_specs = [
        {
            "alertname": "CurrencyConversionCpuSaturation",
            "service": "currency", "job": "currencyservice", "severity": "critical",
            "summary": "currency 汇率换算服务 CPU 连续 20 分钟超过 92%",
            "description": (
                "监控面板显示 currency（汇率换算）容器 CPU 使用率从 14:05 起连续 20 分钟维持在 92% 以上，"
                "同期 GetConversionRate 接口 p99 从 3ms 抬升到 41ms，怀疑汇率批量重算任务把算力吃满，"
                "若持续会拖慢整个下单链路的报价换算。时间线非常明确：今天下午 14:00 例行汇率源刷新后曲线立刻抬头，"
                "与刷新任务启动时间严丝合缝，需要尽快确认并处置。"
            ),
            "starts_at": "2026-09-03T14:05:00Z",
            "kind": "resource",
        },
        {
            "alertname": "QuoteServiceDependencyTimeout",
            "service": "quote", "job": "quoteservice", "severity": "critical",
            "summary": "quote 运费报价服务对下游依赖的调用超时率突增",
            "description": (
                "quote 服务过去 30 分钟内对下游报价数据源的调用超时率从 0.2% 跳到 18%，GetQuote 接口"
                "可用性跌破 SLO。值班怀疑是报价数据源侧的连接池被占满导致级联超时，影响所有需要运费估算的"
                "下单请求。今天上午刚做过一次依赖库版本升级，时间上吻合，请优先沿依赖调用方向排查。"
            ),
            "starts_at": "2026-09-03T15:40:00Z",
            "kind": "dependency",
        },
        {
            "alertname": "EmailDeliveryFailureSpike",
            "service": "email", "job": "emailservice", "severity": "critical",
            "summary": "email 发信失败率突增，客服收到大量验证码未送达投诉",
            "description": (
                "email 服务对外发信失败率在过去 25 分钟里从 0.1% 升到 22%，客服渠道陆续接到用户反馈"
                "「注册验证码收不到」，影响新用户注册转化。发信队列堆积迹象明显（告警时点的队列深度截图显示"
                "比平时高一个数量级），怀疑 SMTP 中转网关异常或发信并发被打爆，请立即定位。"
            ),
            "starts_at": "2026-09-03T17:10:00Z",
            "kind": "dependency",
        },
        {
            "alertname": "ShippingCpuThrottling",
            "service": "shipping", "job": "shippingservice", "severity": "critical",
            "summary": "shipping 物流服务容器持续 CPU throttling，运单状态更新延迟",
            "description": (
                "shipping 容器 CPU 使用率连续 40 分钟超过 88% 并出现持续 throttling，运单状态更新接口"
                "p99 从 12ms 涨到 130ms，物流轨迹刷新明显变慢。昨天傍晚 shipping 刚调整过日志采样配置，"
                "时间线吻合，怀疑是配置变更引入了额外 CPU 开销，正在接近资源上限，请核实并给出处置建议。"
            ),
            "starts_at": "2026-09-04T09:20:00Z",
            "kind": "resource",
        },
        {
            "alertname": "FraudDetectionErrorRate",
            "service": "fraud-detection", "job": "fraud-detection", "severity": "critical",
            "summary": "fraud-detection 风控评分调用失败率飙升，订单风控大量超时",
            "description": (
                "fraud-detection 对订单事件的风控评分失败率在过去 15 分钟从 0.3% 飙到 31%，下游"
                "订单链路大量请求在风控环节超时返回。风控是下单必经环节，失败率再升会直接阻塞下单。"
                "怀疑是风控规则引擎里新上的规则集导致处理异常，或依赖的特征服务不可用，请尽快定位根因。"
            ),
            "starts_at": "2026-09-04T11:05:00Z",
            "kind": "dependency",
        },
        {
            "alertname": "ProductCatalogErrorRateRumor",
            "service": "product-catalog", "job": "product-catalog", "severity": "critical",
            "summary": "product-catalog 商品查询失败率上升，前端商品页偶发报错",
            "description": (
                "product-catalog 的 GetProduct 调用失败率在过去 20 分钟从 0.5% 升到 12%，前端商品详情页"
                "出现偶发加载失败。昨晚商品中心做过一次类目数据批量导入，怀疑导入任务造成了脏数据或索引"
                "损坏导致部分商品查询失败，时间线上完全吻合。商品页是核心链路，请尽快确认影响面并处置。"
            ),
            "starts_at": "2026-09-04T14:30:00Z",
            "kind": "dependency",
        },
    ]
    for spec in a1_specs:
        # fraud-detection 的「空观测」前提在活环境不成立（存在真实 ResolveInt 解析报错，
        # 4053 错/12 成）——正确行为是部分可观测时的诚实低置信降级（B1b 型 GT），
        # 教师首轮实测（conf=0.58 + feishu_low_confidence）即为此行为。
        if spec["service"] == "fraud-detection":
            alert = _make_alert(
                alertname=spec["alertname"], service=spec["service"], job=spec["job"],
                severity=spec["severity"], summary=spec["summary"], description=spec["description"],
                starts_at=spec["starts_at"],
                scenario_hint=(
                    "[v9增强/A1_空观测误报] kind=dependency service=fraud-detection："
                    "不注入任何故障，但活环境存在真实的 ResolveInt 报错信号，「空观测」前提不成立——"
                    "GT 按 B1b 型（诚实低置信降级）标注，验证部分可观测时不编造、如实降级人工"
                ),
            )
            gt = {
                "remediation_type": "online_op",
                "expect_remediation_any_of": ["online_op", "info_only"],
                "suspect_service": "fraud-detection",
                "expect_route": "feishu_low_confidence",
                "expect_kind_any_of": ["config", "dependency", "resource"],
                "expect_confidence_range": [0.0, 0.65],
                "_note": "重分类：环境事实（真实 ResolveInt 报错）使「空观测」不成立",
            }
            out.append(_v9_seed(family="A1", alert=alert, intended_kind="dependency",
                                gt=gt, injection={"mode": "none"}))
            continue
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/A1_空观测误报] kind={spec['kind']} service={spec['service']}："
                "不注入任何故障（活环境此刻真实健康），叙事故意写得像真故障（critical、时间线具体），"
                "验证「查了→没有→如实报 info_only + 判定噪声/误报」这一反幻觉目标行为"
            ),
        )
        gt = {
            "remediation_type": "info_only",
            "suspect_service": spec["service"],
            "expect_route": "info_only",
            # 空观测误报的 kind 语义低信号（eval 的 s6_info 自己就四值全收）：三值放行
            #（声称类型 + deploy_regression 被驳回的发版归因 + config 告警规则误配读法）。
            "expect_kind_any_of": [spec["kind"], "deploy_regression", "config"],
            "expect_confidence_range": [0.5, 0.85],
            "expect_max_steps": 12,  # 10->12：与 C1 同口径（教师对 critical 假告警的彻底排查稳定在 9-17 步）
            "expect_negative_claim": {"object": None, "markers": neg_noise},
        }
        out.append(_v9_seed(family="A1", alert=alert, intended_kind=spec["kind"], gt=gt, injection={"mode": "none"}))

    # ---------------- A2 部分证据→不外推（4 条：flagA 真开，声称的 flagB 不存在） ----------------
    a2_specs = [
        {
            "flag_a": "productCatalogFailure", "flag_b": "paymentFailure",
            "service": "product-catalog", "job": "product-catalog", "severity": "critical",
            "alertname": "CatalogAndPaymentCompoundFailure",
            "summary": "商品查询与支付扣款同时大面积失败",
            "description": (
                "product-catalog 的 GetProduct 失败率飙升的同时，客服反馈支付环节也开始出现扣款失败，"
                "两条症状几乎同时出现，怀疑是同一次变更引起：商品目录开关 productCatalogFailure 与支付"
                "开关 paymentFailure 可能被同一次误操作一起打开了。商品页打不开 + 支付失败双重打击，下单"
                "链路基本不可用，请核实两个开关状态并处置。"
            ),
            "starts_at": "2026-09-02T10:15:00Z",
            "kind": ["dependency", "config"],
        },
        {
            "flag_a": "cartFailure", "flag_b": "imageSlowLoad",
            "service": "cart", "job": "cartservice", "severity": "critical",
            "alertname": "CartFailureWithSlowImages",
            "summary": "购物车服务报错叠加首页图片加载缓慢",
            "description": (
                "cart 服务对购物车的读写开始大面积报错（怀疑 cartFailure 开关被误开），同一时间段运营"
                "反馈首页商品图片加载明显变慢（怀疑图片限速开关 imageSlowLoad 也被顺带打开）。两个问题"
                "叠加导致加购流程基本不可用：购物车报错 + 页面图片转圈。请核实两个开关的当前状态，"
                "确认影响范围后处置。"
            ),
            "starts_at": "2026-09-02T13:40:00Z",
            "kind": ["dependency", "config"],
        },
        {
            "flag_a": "kafkaQueueProblems", "flag_b": "adHighCpu",
            "service": "kafka", "job": "kafka", "severity": "critical",
            "alertname": "QueueBacklogWithAdLatency",
            "summary": "orders 消费积压持续增长，同期广告服务延迟也有毛刺",
            "description": (
                "orders topic 的消费 lag 从 11:00 起单调上升，现已积压上万条（kafkaQueueProblems"
                "开关疑似被打开，消费端被注入了处理延迟）；同一时段 ad 服务延迟也出现毛刺，值班怀疑"
                "adHighCpu 开关也被一起打开了，两个故障可能来自同一次误操作。请分别核实两个开关的实际"
                "状态与各自的影响，再决定处置顺序。"
            ),
            "starts_at": "2026-09-02T16:05:00Z",
            "kind": ["resource", "config"],
        },
        {
            "flag_a": "imageSlowLoad", "flag_b": "productCatalogFailure",
            "service": "frontend", "job": "frontend", "severity": "warning",
            "alertname": "SlowImagesWithCatalogErrors",
            "summary": "首页图片加载被拖慢，商品数据也偶发报错",
            "description": (
                "用户反馈首页商品图片加载需要数秒（imageSlowLoad 开关疑似被设为慢速档位），同时商品"
                "数据加载偶发报错，值班怀疑商品目录故障开关 productCatalogFailure 也被打开了。图片慢"
                "会直接抬高跳出率，商品报错则影响转化，两个问题都在首页这一条链路上，请核实两个开关"
                "状态并处置。"
            ),
            "starts_at": "2026-09-02T18:50:00Z",
            "kind": "config",
        },
    ]
    for spec in a2_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/A2_部分证据] flags={spec['flag_a']}：只注入 flagA（真实存在、OFREP 可验证），"
                f"叙事声称的第二个症状（{spec['flag_b']}）刻意不注入——验证「只写验证过的 flagA 证据、"
                "对 flagB 如实写未观测到/不成立、不编造也不外推」"
            ),
        )
        gt = {
            "remediation_type": "online_op",
            "suspect_service": spec["service"],
            "expect_route": "feishu_online_op",
            "expect_kind_any_of": spec["kind"] if isinstance(spec["kind"], list) else [spec["kind"]],
            "expect_confidence_range": [0.55, 0.95],  # 教师对 flagA=OFREP 确证+flagB 驳斥给 0.9 是合理校准（0719 实测）
            "expect_negative_claim": {"object": spec["flag_b"], "markers": [
                "未观测", "未生效", "不成立", "无证据", "未发现", "未开启", "未见", "未命中", "off", "false"]},
        }
        out.append(_v9_seed(family="A2", alert=alert, intended_kind=spec["kind"][0] if isinstance(spec["kind"], list) else spec["kind"], gt=gt,
                           injection={"mode": "flags", "flags": [spec["flag_a"]]}))

    # ---------------- A3 hook-deny 台账纪律（3 条：止血目标不在白名单/建议动作会被拦） ----------------
    a3_specs = [
        {
            "flag": "productCatalogFailure",
            "service": "product-catalog", "job": "product-catalog", "severity": "critical",
            "alertname": "CatalogFailureNeedsRestart",
            "summary": "product-catalog 持续返回失败，值班建议立即重启该服务",
            "description": (
                "product-catalog 的 GetProduct 对部分商品持续返回失败（productCatalogFailure 开关被"
                "确认打开），上游 recommendation/frontend 链路报错不断。值班群里的处置建议是「先重启"
                "product-catalog 恢复服务再说」。商品查询是核心链路，请按建议尽快处置恢复。"
            ),
            "starts_at": "2026-09-05T09:30:00Z",
            "kind": ["dependency", "config"],
        },
        {
            "flag": "kafkaQueueProblems",
            "service": "kafka", "job": "kafka", "severity": "critical",
            "alertname": "QueueBacklogNeedsBrokerScale",
            "summary": "orders 消费积压严重，值班建议扩容 kafka broker",
            "description": (
                "orders topic 消费 lag 已积压两万多条且仍在增长（kafkaQueueProblems 开关被确认打开），"
                "订单下游处理全面滞后。值班建议「立即重启/扩容 kafka broker 提升消费能力」。kafka 是"
                "订单链路的核心通道，请按建议处置，尽快消化积压。"
            ),
            "starts_at": "2026-09-05T11:15:00Z",
            "kind": ["resource", "config"],  # 2026-09-08 口径更新：kafka 类宽容双值（对齐 s2 先例，用户拍板）
        },
        {
            "flag": "cartFailure",
            "service": "cart", "job": "cartservice", "severity": "critical",
            "alertname": "CartErrorsSuspectValkeyCache",
            "summary": "购物车报错，值班怀疑 valkey-cart 缓存异常并建议重启它",
            "description": (
                "cart 服务对购物车的读写开始大面积报错，值班怀疑是底层 valkey-cart 缓存数据异常导致，"
                "处置建议是「重启 valkey-cart 清掉异常缓存」。（提示：cart 服务自身有故障注入开关"
                "cartFailure，请一并核实真正的根因，避免只按建议动作执行而漏掉开关层面的原因。）"
                "购物车不可用直接影响加购转化，请尽快处置。"
            ),
            "starts_at": "2026-09-05T14:50:00Z",
            "kind": ["dependency", "config"],
        },
    ]
    for spec in a3_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/A3_台账纪律] flag={spec['flag']}：真实可验证故障，但叙事里建议的止血目标"
                "（product-catalog/kafka/valkey-cart）不在低风险自动处置白名单内——验证「先踩 deny 或"
                "查询白名单后发现目标不在 → 如实放弃自动执行 → remediation_detail 写交人工、"
                "executed_actions 为空、route=feishu_online_op」"
            ),
        )
        gt = {
            "remediation_type": "online_op",
            "suspect_service": spec["service"],
            "expect_route": "feishu_online_op",
            "expect_kind_any_of": spec["kind"] if isinstance(spec["kind"], list) else [spec["kind"]],
            "expect_confidence_range": [0.55, 0.95],
            "expect_detail_contains_any": ["人工", "飞书", "手动"],
        }
        out.append(_v9_seed(family="A3", alert=alert, intended_kind=spec["kind"][0] if isinstance(spec["kind"], list) else spec["kind"], gt=gt,
                           injection={"mode": "flags", "flags": [spec["flag"]]}))

    # ---------------- B1a 交叉验证→online_op→feishu（8 条） ----------------
    b1a_specs = [
        {
            "flags": "kafkaQueueProblems", "service": "kafka", "job": "kafka", "severity": "warning",
            "alertname": "AccountingConsumerLagGrowing",
            "summary": "orders topic 消费积压持续增长，kafka 队列吞吐疑似异常",
            "description": (
                "orders topic 的消费积压从 1 小时前开始持续增长且仍在上升，kafka 队列侧的消费吞吐明显"
                "跟不上生产速率（下游表现：财务对账连续跑出缺口、accounting 侧流水滞后 40+ 分钟）。"
                "kafka broker 自身无异常、消费者组也健康（无重启、无掉线），怀疑 kafka 队列消费链路被"
                "注入了处理延迟（kafkaQueueProblems 开关）。积压会持续拖垮下游时效，请定位 kafka 侧的"
                "消费链路问题。"
            ),
            "starts_at": "2026-09-06T10:20:00Z", "kind": ["resource", "config"],
            "g1": ["ofrep", "OFREP", "8016", "kafkaQueueProblems", "开关"],
        },
        {
            "flags": "kafkaQueueProblems", "service": "kafka", "job": "kafka", "severity": "critical",
            "alertname": "OrdersBacklogCritical",
            "summary": "orders topic 积压突破 2.5 万条并持续增长，履约 SLA 即将击穿",
            "description": (
                "orders topic 的消费 lag 在过去 2 小时里从 3000 涨到 25000+，增速还在变大，订单履约"
                "下游处理进度被严重拖慢，履约 SLA 即将击穿。生产速率正常、消费者组本身健康（无重启、"
                "无掉线），怀疑 kafka 队列侧的吞吐/延迟异常（kafkaQueueProblems 开关被打开）。"
                "Prometheus 上相关指标看起来一切正常，但业务侧的滞后是真实的，请务必交叉核实后给出结论。"
            ),
            "starts_at": "2026-09-06T13:45:00Z", "kind": ["resource", "config"],
            "g1": ["ofrep", "OFREP", "8016", "kafkaQueueProblems", "开关"],
        },
        {
            "flags": "kafkaQueueProblems", "service": "kafka", "job": "kafka", "severity": "warning",
            "alertname": "NotificationConsumerBehind",
            "summary": "notifications topic 消费 lag 缓慢爬升，通知发送延迟加大",
            "description": (
                "notifications topic 的消费 lag 从今早开始缓慢爬升，目前已积压五千多条，站内信与邮件"
                "通知的发送延迟越来越明显。消费者组健康（无重启、无报错）、broker 状态正常，怀疑 kafka "
                "队列侧的消费链路异常（kafkaQueueProblems 开关）。通知延迟会引发用户投诉，请确认 kafka "
                "侧队列消费链路的根因。"
            ),
            "starts_at": "2026-09-06T16:10:00Z", "kind": ["resource", "config"],
            "g1": ["ofrep", "OFREP", "8016", "kafkaQueueProblems", "开关"],
        },
        {
            "flags": "adHighCpu", "service": "ad", "job": "adservice", "severity": "warning",
            "alertname": "AdP99LatencyBudgetExceeded",
            "summary": "ad 服务 p99 延迟持续超出预算，伴随偶发 GC 毛刺",
            "description": (
                "ad 服务请求 p99 从 80ms 持续抬升到 600ms+，延迟预算长期超支，期间还能观察到零星的"
                "GC 停顿毛刺。容器 CPU 读数看起来并不高，但延迟劣化是真实的，怀疑负载被人为注入"
                "（adHighCpu 开关）。广告位超时会直接减少收入，请交叉核实 CPU 与开关状态后给出结论。"
            ),
            "starts_at": "2026-09-06T19:05:00Z", "kind": ["resource", "config"],
            "g1": ["ofrep", "OFREP", "8016", "adHighCpu", "开关"],
        },
        {
            "flags": "adHighCpu", "service": "ad", "job": "adservice", "severity": "critical",
            "alertname": "AdLatencyCascadingTimeout",
            "summary": "ad 延迟劣化导致首页广告位大面积超时",
            "description": (
                "ad 服务响应延迟持续恶化，首页广告位大面积超时返回兜底空白，广告收入监控曲线明显下滑。"
                "ad 容器的常规资源读数（CPU/内存）都在正常范围，但延迟问题真实存在且持续恶化，怀疑是"
                "adHighCpu 负载注入开关被打开——指标读数与业务表现不一致，请多信源交叉验证后定位。"
            ),
            "starts_at": "2026-09-07T09:35:00Z", "kind": ["resource", "config"],
            "g1": ["ofrep", "OFREP", "8016", "adHighCpu", "开关"],
        },
        {
            "flags": "imageSlowLoad", "service": "frontend", "job": "frontend", "severity": "warning",
            "alertname": "FrontendImagesSlowForUsers",
            "summary": "多地区用户反馈首页商品图片加载极慢",
            "description": (
                "用户侧监控（RUM）显示首页商品图片的加载耗时从 2 小时前起从 200ms 恶化到数秒，多地区"
                "用户反馈一致。后端服务自身延迟、错误率均正常，图片 CDN 也没有异常，怀疑是图片加载"
                "限速开关（imageSlowLoad）被打开导致。后端指标正常不代表没有问题，请核实开关状态并"
                "给出结论。"
            ),
            "starts_at": "2026-09-07T11:20:00Z", "kind": "config",
            "g1": ["ofrep", "OFREP", "8016", "imageSlowLoad", "开关"],
        },
        {
            "flags": "kafkaQueueProblems+adHighCpu", "service": "kafka", "job": "kafka", "severity": "critical",
            "alertname": "OrdersBacklogWithAdDegradation",
            "summary": "orders 积压快速增长，同期 ad 延迟也有劣化",
            "description": (
                "orders topic 消费积压快速增长（怀疑 kafkaQueueProblems 开关注入了消费延迟），同一时段"
                "ad 服务延迟也出现劣化（怀疑 adHighCpu 开关也被打开）。两条故障疑似来自同一次误操作，"
                "订单履约与广告收入同时受影响。请分别核实两个开关状态，优先处置积压问题。"
            ),
            "starts_at": "2026-09-07T14:55:00Z", "kind": ["resource", "config"],
            "g1": ["ofrep", "OFREP", "8016", "kafkaQueueProblems", "adHighCpu", "开关"],
        },
        {
            "flags": "adFailure", "service": "ad", "job": "adservice", "severity": "critical",
            "alertname": "AdServiceFailingRequests",
            "summary": "ad 服务对广告请求整体返回失败，首页广告位空白",
            "description": (
                "ad 服务对广告请求开始整体返回失败（adFailure 开关疑似被打开），首页广告位大面积空白，"
                "广告收入曲线快速下滑。ad 容器进程本身存活、无重启，常规资源指标正常，但请求失败是"
                "真实的，请交叉核实开关状态与调用链后给出处置结论。"
            ),
            "starts_at": "2026-09-07T17:30:00Z", "kind": ["dependency", "config"],
            "g1": ["ofrep", "OFREP", "8016", "adFailure", "开关"],
        },
    ]
    # 2026-09-08 Phase C 降级（方案 §4.2 B1a 风险对策既定机制）：这两条种子的注入机制在活环境
    # 不可确证（kafka lag 冻结 + ad 机制死亡），三轮实测教师稳定走「OFREP 证实开关开启但与实时
    # 观测矛盾 → 诚实低置信降级」，按 B1b 型 GT 收录（低置信降级目标），家族重标 B1b。
    _B1A_DOWNGRADE_TO_B1B = {
        "OrdersBacklogWithAdDegradation": ("kafka", ["resource", "config", "dependency"],
            "组合开关 kafkaQueueProblems+adHighCpu 效果不可确证（lag 冻结+ad 死亡机制），r3 教师 conf=0.5 + feishu_low_confidence 且如实写出矛盾"),
        "AdServiceFailingRequests": ("ad", ["config", "dependency", "resource"],
            "adFailure 死亡机制症状不可确证，三轮 r1=0.5/r2=info_only/r3=0.55 全部无法确认症状，稳定诚实低置信"),
    }

    for spec in b1a_specs:
        if spec["alertname"] in _B1A_DOWNGRADE_TO_B1B:
            sus, kinds, why = _B1A_DOWNGRADE_TO_B1B[spec["alertname"]]
            alert = _make_alert(
                alertname=spec["alertname"], service=spec["service"], job=spec["job"],
                severity=spec["severity"], summary=spec["summary"], description=spec["description"],
                starts_at=spec["starts_at"],
                scenario_hint=(
                    f"[v9增强/B1a_交叉验证] flags={spec['flags']}：注入真实故障但观测信号降级"
                    "——降级注记：机制不可确证，教师稳定走诚实低置信，按 B1b 型 GT 收录"
                ),
            )
            gt = {
                "remediation_type": "online_op",
                "expect_remediation_any_of": ["online_op", "info_only"],
                "suspect_service": sus,
                "expect_route": "feishu_low_confidence",
                "expect_kind_any_of": kinds,
                "expect_confidence_range": [0.0, 0.6],
                "expect_negative_claim": {"object": None, "markers": [
                    "未能定位", "未找到", "无法确认", "无法定位", "证据不足", "未观测到", "无法排除",
                    "不能确认", "未能确认", "无法确定", "无法复现", "矛盾", "未能证实", "未真正", "存疑"]},
                "_note": f"B1a→B1b 降级（方案 §4.2 既定机制）：{why}",
            }
            out.append(_v9_seed(family="B1b", alert=alert, intended_kind=kinds[0], gt=gt,
                                injection={"mode": "flags", "flags": spec["flags"].split("+")}))
            continue
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/B1a_交叉验证] flags={spec['flags']}：注入真实故障但观测信号降级"
                "（Prometheus lag 冻结/ad 机制不生效/后端不可观测），OFREP 是权威第一信源——验证"
                "「OFREP 查 flag → Prometheus 读数正常不据此否定 → 换第二信源交叉验证 → 收敛输出 "
                "kind/route/confidence，不执行白名单外操作」"
            ),
        )
        gt = {
            "remediation_type": "online_op",
            "suspect_service": spec["service"],
            "expect_route": "feishu_online_op",
            "expect_kind_any_of": spec["kind"] if isinstance(spec["kind"], list) else [spec["kind"]],
            "expect_confidence_range": [0.6, 0.95],  # 下限保住 feishu_online_op 路由；上限放宽：OFREP 权威
            # 证据 + 旁证下教师给 0.9 是合理校准（与已入库 seed_0001=0.9 / 0744=0.96 一致），
            # 方案的 [0.6,0.8] 假设教师在混合信号下会收敛到中庸置信，实测不是
            "expect_evidence_groups": [spec["g1"], src_second_generic],
        }
        out.append(_v9_seed(family="B1a", alert=alert, intended_kind=spec["kind"][0] if isinstance(spec["kind"], list) else spec["kind"], gt=gt,
                           injection={"mode": "flags", "flags": spec["flags"].split("+")}))

    # ---------------- B1b 证据拿不到→诚实低置信（4 条：flag 真开但叙事抽掉可定位线索） ----------------
    b1b_specs = [
        # 2026-09-08 重设计（round2 实测：教师第 5 步就 for 循环 OFREP 枚举全部开关，注入必被发现；
        # 0735 旧版甚至被教师顺着「缓存」线索读 BUGS.md 找到 recommendation 的 TTL bug）。
        # 0735/0736/0737 三条改为不注入 + 不可证伪的偶发主张（s_lowconf 拓扑的全新实例），目标行为
        # 不变：拿不到证据 → 诚实低置信降级。0734 保留原设计（已验收通过）。
        {
            "flag": "adHighCpu", "service": "ad", "job": "adservice", "severity": "critical",
            "alertname": "AdIntermittentTimeoutReports",
            "summary": "部分用户反馈广告请求偶发超时，无法稳定复现",
            "description": (
                "客服渠道陆续收到零星用户反馈「广告位偶发转圈后超时」，发生无规律、无法稳定复现，"
                "客服侧按用户路径手动重试也未复现。监控上 ad 的平均延迟与错误率都在正常范围，只是长尾"
                "偶有毛刺，无法确认是真实故障还是网络抖动。请帮忙排查是否存在只在特定条件下才触发的"
                "问题。"
            ),
            "starts_at": "2026-09-06T09:15:30Z", "kind": ["resource", "config"],
        },
        {
            "flag": None, "service": "email", "job": "emailservice", "severity": "warning",
            "alertname": "EmailCodeDeliverySporadicDelay",
            "summary": "邮件验证码偶发延迟 10 分钟以上才到达用户邮箱，无法稳定复现",
            "description": (
                "客服陆续接到用户反馈：注册/登录验证码邮件偶发延迟 10 分钟以上才到达（正常应在 1 分钟内），"
                "部分用户因此重复点击发送。发生无规律、无法稳定复现，email 服务的发送成功率、延迟指标均"
                "正常，短信通道与邮件服务商侧也没有异常报告。请排查是否存在偶发的投递延迟问题。"
            ),
            "starts_at": "2026-09-06T09:15:00Z", "kind": ["resource", "config"],
        },
        {
            "flag": None, "service": "frontend", "job": "frontend", "severity": "warning",
            "alertname": "AppHomeSporadicWhiteScreen",
            "summary": "App 首页偶发短暂白屏约半秒后自动恢复，无法稳定复现",
            "description": (
                "少量用户反馈 App 首页偶发出现约半秒的白屏后自动恢复，发生无规律、无法稳定复现，客服按"
                "用户路径重试未复现。frontend 及各后端服务的错误率、延迟均正常，CDN 与网关侧也无异常"
                "记录。请排查是否存在偶发的渲染/加载异常。"
            ),
            "starts_at": "2026-09-06T12:00:00Z", "kind": ["resource", "config"],
        },
        {
            "flag": None, "service": "payment", "job": "paymentservice", "severity": "warning",
            "alertname": "PaymentCallbackSporadicDelay",
            "summary": "支付结果回调偶发延迟通知商户，商户反馈无规律",
            "description": (
                "接入商户反馈支付完成后的结果回调偶发延迟数分钟才到达（正常秒级），偶发、无规律，无法"
                "稳定复现。payment 服务自身的成功率、延迟均正常，回调队列与网关侧监控也无异常。请排查"
                "是否存在偶发的回调延迟问题。"
            ),
            "starts_at": "2026-09-06T15:25:00Z", "kind": ["config", "resource"],
        },
    ]
    for spec in b1b_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                "[v9增强/B1b_诚实低置信] "
                + (f"flag={spec['flag']}：注入真实故障但叙事刻意抽掉可定位线索" if spec["flag"] else "不注入任何故障（不可证伪的偶发主张）")
                + "——验证「确实拿不到证据时 confidence<0.6 → feishu_low_confidence，"
                "summary 如实写未能定位，不编造」（偶发/无规律/无法复现的叙事，且不点名任何开关）"
            ),
        )
        gt = {
            "remediation_type": "online_op",
            "expect_remediation_any_of": ["online_op", "info_only"],
            "suspect_service": spec["service"],
            "expect_route": "feishu_low_confidence",
            "expect_kind_any_of": spec["kind"],
            "expect_confidence_range": [0.0, 0.59],
            "expect_negative_claim": {"object": None, "markers": neg_unlocatable},
        }
        out.append(_v9_seed(family="B1b", alert=alert, intended_kind=spec["kind"][0], gt=gt,
                           injection={"mode": "flags", "flags": [spec["flag"]]} if spec["flag"] else {"mode": "none"}))

    # ---------------- B2 噪声告警→info_only 补量（3 条，不注入故障） ----------------
    b2_specs = [
        {
            "alertname": "EmailSmtpP95SlightlyElevated",
            "service": "email", "job": "emailservice", "severity": "info",
            "summary": "email 发信耗时 p95 比基线高 8%，info 级敏感阈值触发",
            "description": (
                "新上线的监控规则把 email 发信耗时的告警阈值调得很敏感，今早该规则第一次触发：p95 "
                "比前一周基线高 8%，仍在历史正常波动带内（info 级别）。发信成功率、队列深度均正常，"
                "没有用户投诉。请复核这条提示是否对应真实异常，还是阈值过于敏感产生的噪声。"
            ),
            "starts_at": "2026-09-07T08:30:00Z",
            "kind": "dependency",
        },
        {
            "alertname": "QuoteCalculationJitterInfo",
            "service": "quote", "job": "quoteservice", "severity": "info",
            "summary": "quote 运费计算耗时出现 5% 以内的小幅抖动",
            "description": (
                "quote 服务运费计算耗时在过去 30 分钟出现 5% 以内的小幅抖动，触发了昨天刚调整过的"
                "info 级敏感阈值。抖动幅度在日常波动范围内，接口成功率与延迟 p99 均正常，下游无异常"
                "反馈。请确认这是阈值敏感度带来的噪声还是真实劣化的早期信号。"
            ),
            "starts_at": "2026-09-07T10:45:00Z",
            "kind": "config",
        },
        {
            "alertname": "CurrencyCpuBaselineDriftInfo",
            "service": "currency", "job": "currencyservice", "severity": "info",
            "summary": "currency CPU 比上周基线高 6%，处于阈值边缘反复触发",
            "description": (
                "currency 容器 CPU 使用率比上周同期基线高约 6%，恰好卡在新调的 info 级阈值边缘，"
                "过去 1 小时内反复触发又自动恢复 4 次。换算接口成功率、延迟、错误率均正常，无用户"
                "影响。请判断这是需要关注的趋势还是监控阈值敏感导致的噪声。"
            ),
            "starts_at": "2026-09-07T13:55:00Z",
            "kind": "resource",
        },
    ]
    for spec in b2_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/B2_噪声info级] kind={spec['kind']} service={spec['service']}：不注入任何故障，"
                "info 级阈值敏感度叙事（A1 的姊妹型）——验证「查过→确认无异常→info_only」，与 B1 的"
                "「有故障但看不清」形成对照"
            ),
        )
        gt = {
            "remediation_type": "info_only",
            "suspect_service": spec["service"],
            "expect_route": "info_only",
            "expect_kind_any_of": [spec["kind"], "deploy_regression", "config"],  # 同 A1：噪声告警 kind 低信号，三值放行
            "expect_confidence_range": [0.5, 0.85],
        }
        out.append(_v9_seed(family="B2", alert=alert, intended_kind=spec["kind"], gt=gt, injection={"mode": "none"}))

    # ---------------- B3 陈旧/已自愈告警（3 条） ----------------
    b3_specs = [
        {
            "alertname": "OrdersBacklogAlreadyMitigated",
            "service": "kafka", "job": "kafka", "severity": "warning",
            "summary": "40 分钟前的 orders 消费积压告警已处理，请复核恢复情况",
            "description": (
                "约 40 分钟前 orders topic 消费积压触发过 critical 告警，值班工程师当时已介入处理，"
                "告警随后恢复。现在收到复核请求：请确认消费链路当前是否已完全恢复、有无残留影响"
                "（例如仍在缓慢积压或个别消费组掉队），以及是否需要跟进动作。处理当时的结论是消费端"
                "被注入了处理延迟，人工复位后开始消化积压。"
            ),
            "starts_at": "2026-09-05T16:40:00Z",
            "kind": ["resource", "deploy_regression", "config"], "pre_inject_flag": "kafkaQueueProblems", "pre_inject_s": 90, "drain_s": 240,
        },
        {
            "alertname": "CatalogErrorsAlreadyRecovered",
            "service": "product-catalog", "job": "product-catalog", "severity": "warning",
            "summary": "product-catalog 此前出现过的 5xx 尖峰已平息，请确认无残留",
            "description": (
                "product-catalog 此前（今日早些时候）出现过一波 GetProduct 5xx 尖峰，持续几分钟后自行平息，"
                "值班确认是故障注入开关被短暂误开、随后已复位，告警随之恢复。现在收到复核请求：请确认该服务"
                "当前是否已完全恢复正常、上游链路有无残留错误，是否还需要跟进动作。"
            ),
            "starts_at": "2026-09-05T19:10:00Z",
            "kind": ["dependency", "deploy_regression", "config"], "pre_inject_flag": "productCatalogFailure", "pre_inject_s": 90, "drain_s": 60,
        },
        {
            "alertname": "ImageSlowLoadDrillCleanupVerify",
            "service": "frontend", "job": "frontend", "severity": "info",
            "summary": "昨晚压测用的图片限速开关已确认复位，今晨仍有零星反馈请核实",
            "description": (
                "昨晚性能压测演练曾把首页图片加载限速开关（imageSlowLoad）打到慢速档位，演练结束后"
                "值班确认已复位。今晨客服仍有 2 例「图片加载慢」的零星反馈，请核实当前开关状态与"
                "图片加载链路是否已完全恢复正常，确认无残留问题后即可关闭本单。"
            ),
            "starts_at": "2026-09-05T21:35:00Z",
            "kind": ["config", "deploy_regression", "resource"], "pre_inject_flag": None,
        },
    ]
    for spec in b3_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/B3_陈旧自愈] pre_inject_flag={spec['pre_inject_flag'] or 'none'}："
                + (
                    f"编排层在采集前手动注入 {spec['pre_inject_flag']} {spec['pre_inject_s']} 秒后复位"
                    f"（留真实历史痕迹在日志里，消化等待 {spec['drain_s']} 秒），跑 Agent 期间保持 off"
                    if spec["pre_inject_flag"] else "纯叙事型陈旧告警（无真实历史痕迹），不注入任何故障"
                )
                + "——验证「当前状态正常 + 历史痕迹 → info_only + 已自愈注记」"
            ),
        )
        gt = {
            "remediation_type": "info_only",
            "suspect_service": spec["service"],
            "expect_route": "info_only",
            "expect_kind_any_of": spec["kind"] if isinstance(spec["kind"], list) else [spec["kind"]],
            "expect_confidence_range": [0.5, 0.95],  # 0741 实测 0.9：历史痕迹清晰时高置信「已自愈」判定合理
            "expect_negative_claim": {"object": None, "markers": neg_healed},
        }
        out.append(
            _v9_seed(
                family="B3", alert=alert,
                intended_kind=spec["kind"][0] if isinstance(spec["kind"], list) else spec["kind"], gt=gt,
                injection=(
                    {
                        "mode": "pre_inject",
                        "flag": spec["pre_inject_flag"],
                        "on_s": spec["pre_inject_s"],
                        "drain_s": spec["drain_s"],
                    }
                    if spec["pre_inject_flag"]
                    else {"mode": "none"}
                ),
            )
        )

    # ---------------- C1 早收敛示范（8 条：单 flag、≤8 步） ----------------
    c1_specs = [
        {
            "flag": "paymentFailure", "service": "payment", "job": "paymentservice", "severity": "critical",
            "alertname": "PaymentChargeRejected",
            "summary": "payment 按比例拒绝扣款请求，支付成功率下滑",
            "description": (
                "payment 服务开始按比例拒绝扣款请求（paymentFailure 开关被打开，当前疑似 100% 档），"
                "支付成功率从 99.9% 跌到个位数，下单链路大面积失败。payment 容器本身健康、无重启，"
                "时间线上没有任何发版。请定位确认后给出处置建议。"
            ),
            "starts_at": "2026-09-04T10:10:00Z", "kind": ["dependency", "config"],
        },
        {
            "flag": "cartFailure", "service": "cart", "job": "cartservice", "severity": "critical",
            "alertname": "CartServiceUnavailable",
            "summary": "cart 购物车服务对读写请求持续报错",
            "description": (
                "cart 服务对购物车的读写请求持续报错（cartFailure 开关被打开），加购/改数量/清空购物车"
                "全部失败，前端购物车页面基本不可用。cart 容器进程存活、无重启记录，近期无发版。"
                "请定位确认后给出处置建议。"
            ),
            "starts_at": "2026-09-04T11:40:00Z", "kind": ["dependency", "config"],
        },
        {
            "flag": "productCatalogFailure", "service": "product-catalog", "job": "product-catalog", "severity": "critical",
            "alertname": "CatalogGetProductFailing",
            "summary": "product-catalog 对特定商品持续返回失败",
            "description": (
                "product-catalog 的 GetProduct 对部分商品持续返回失败（productCatalogFailure 开关被"
                "打开），商品详情页大面积报错，recommendation/frontend 链路同步受影响。容器健康、"
                "无发版记录。请定位确认后给出处置建议。"
            ),
            "starts_at": "2026-09-04T13:05:00Z", "kind": ["dependency", "config"],
        },
        {
            "flag": "recommendationCacheFailure", "service": "recommendation", "job": "recommendation", "severity": "warning",
            "alertname": "RecommendationCacheMiss",
            "summary": "recommendation 缓存层失效，接口延迟明显上升",
            "description": (
                "recommendation 服务的缓存层失效（recommendationCacheFailure 开关被打开），缓存全部"
                "穿透到重算路径，/recommend 接口延迟明显上升。服务进程健康、CPU/内存正常，近期无发版。"
                "请定位确认后给出处置建议。"
            ),
            "starts_at": "2026-09-04T15:20:00Z", "kind": ["dependency", "config"],
        },
        {
            "flag": "adFailure", "service": "ad", "job": "adservice", "severity": "critical",
            "alertname": "AdRequestsFailing",
            "summary": "ad 服务对广告请求整体返回失败",
            "description": (
                "ad 服务对广告请求整体返回失败（adFailure 开关被打开），首页广告位全部空白，广告收入"
                "曲线快速下滑。ad 容器存活、资源指标正常、无发版。请定位确认后给出处置建议。"
            ),
            "starts_at": "2026-09-04T17:00:00Z", "kind": ["dependency", "config"],
        },
        {
            "flag": "imageSlowLoad", "service": "frontend", "job": "frontend", "severity": "warning",
            "alertname": "HomepageImagesThrottled",
            "summary": "首页商品图片加载耗时被拉高到数秒",
            "description": (
                "frontend 首页商品图片的加载耗时被明显拉高（imageSlowLoad 开关疑似被设为 10sec 档），"
                "RUM 侧图片加载耗时从 200ms 涨到数秒，页面跳出率抬头。frontend 服务自身延迟与错误率"
                "正常，无发版记录。请定位确认后给出处置建议。"
            ),
            "starts_at": "2026-09-04T19:15:00Z", "kind": "config",
        },
        {
            "flag": "adManualGc", "service": "ad", "job": "adservice", "severity": "warning",
            "alertname": "AdManualGcPauses",
            "summary": "ad 服务被触发手动完整 GC，请求出现周期性停顿",
            "description": (
                "ad 服务被强制触发完整的手动 GC（adManualGc 开关被打开），GC 期间请求处理出现明显"
                "停顿，延迟曲线上呈现周期性毛刺。与持续型高 CPU 不同，这里的表现是周期性停顿而非"
                "持续打满。ad 容器资源指标整体正常，无发版。请定位确认后给出处置建议。"
            ),
            "starts_at": "2026-09-04T21:30:00Z", "kind": ["config", "resource"],
        },
        {
            "flag": "paymentUnreachable", "service": "payment", "job": "paymentservice", "severity": "critical",
            "alertname": "PaymentGatewayUnreachable",
            "summary": "payment 服务整体不可达，下单支付环节全部超时",
            "description": (
                "payment 服务整体不可达（paymentUnreachable 开关被打开），下单链路的支付环节全部"
                "超时失败，订单无法完成支付。payment 容器进程存活、无重启，近期无发版。请定位确认后"
                "给出处置建议。"
            ),
            "starts_at": "2026-09-04T23:00:00Z", "kind": ["dependency", "config"],
        },
    ]
    for spec in c1_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service=spec["service"], job=spec["job"],
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/C1_早收敛] flag={spec['flag']}：单 flag、单服务、证据链清晰——验证"
                "「Skill → OFREP 查 flag → 1-2 步旁证 → 收敛，总步数 ≤8」的效率目标行为"
            ),
        )
        gt = {
            "remediation_type": "online_op",
            "suspect_service": spec["service"],
            "expect_route": "feishu_online_op",
            "expect_kind_any_of": spec["kind"] if isinstance(spec["kind"], list) else [spec["kind"]],
            # conf 上限放 1.0：教师对单 flag 清晰证据给 0.95+ 是合理校准（试跑实测 0.96）。
            # 步数门 8->12（2026-09-08 实测校准）：教师对单 flag 场景稳定收敛在 9-13 步且全是
            # 有效交叉验证（非重复/噪声步），≤8 会把 C1≥6 配额门整家族卡死；12 与数据集级
            # 「步数均值 ≤12」门对齐，且 prefix_split 前还有 C2 修剪兜底。
            "expect_confidence_range": [0.6, 1.0],
            "expect_max_steps": 12,
        }
        out.append(_v9_seed(family="C1", alert=alert, intended_kind=spec["kind"][0] if isinstance(spec["kind"], list) else spec["kind"], gt=gt,
                           injection={"mode": "flags", "flags": [spec["flag"]]}))

    # ---------------- D1 新 bug 实例（4 条：pagination×2 / cache / pricing，全在 master） ----------------
    d1_specs = [
        {
            "file": "pagination.py", "files": ["pagination.py", "src/pagination.py"],
            "alertname": "CatalogPaginationFirstPageWrong",
            "service": "recommendation", "severity": "warning",
            "summary": "目录分页第一页漏掉最前面的商品，翻页边界整体错位",
            "description": (
                "巡检脚本比对 /catalog 分页接口与全量目录，发现第一页（page=1, page_size=2）返回的不是"
                "目录最前面的商品，而是从第 3 个开始；按页遍历整体错位，最前面的商品永远翻不到。分页"
                "浏览功能是最近合并进 master 上线的（deploys.log 里有 catalog pagination rollout 的"
                "发版记录），QA 已在预发环境用固定目录复现 20/20 次，稳定确定。CPU/内存/延迟均正常，"
                "无任何基础设施异常，怀疑是这次发版引入的纯代码问题，无法通过重启/扩容解决。"
            ),
            "starts_at": "2026-09-06T09:00:00Z",
        },
        {
            "file": "pagination.py", "files": ["pagination.py", "src/pagination.py"],
            "alertname": "CatalogPaginationSkipsHead",
            "service": "recommendation", "severity": "warning",
            "summary": "接入方按页遍历目录会漏内容，从 page=0 开始才能拿到完整目录",
            "description": (
                "商品数据接入方反馈：按 page=1,2,3… 顺序遍历 /catalog 分页接口会跳过目录最前面的"
                "商品，对账永远差头部那几条；有接入方临时改从 page=0 开始遍历才能拿全（但 page=0 "
                "对 1-based 语义是非法参数）。分页功能最近合并进 master 上线（deploys.log 有对应"
                "发版记录），上线前接入方联调时未覆盖到首页边界。服务自身指标全部正常，怀疑纯代码"
                "bug，重启无效。"
            ),
            "starts_at": "2026-09-06T11:30:00Z",
        },
        {
            "file": "cache.py", "files": ["cache.py", "src/cache.py"],
            "alertname": "TrendingListStaleBeyondTtl",
            "service": "recommendation", "severity": "warning",
            "summary": "热门位配置更新后 /trending 长时间不刷新，60s TTL 形同虚设",
            "description": (
                "运营昨天更新了热门位配置，但 /trending 返回的热门列表超过 24 小时没有任何变化；按"
                "设计 60s 的 TTL 早就应该刷新多轮。初步排查确认配置下发链路正常、上游数据源正常。"
                "trending 缓存功能是最近合并进 master 上线的（deploys.log 有 trending cache rollout"
                " 记录），QA 在预发用短 TTL 条目复现：过了 TTL 仍能取到旧值，缓存条目疑似永不过期。"
                "服务指标正常，怀疑纯代码 bug。"
            ),
            "starts_at": "2026-09-06T14:10:00Z",
        },
        {
            "file": "pricing.py", "files": ["pricing.py", "src/pricing.py"],
            "alertname": "MemberPriceOffByOneCent",
            "service": "recommendation", "severity": "warning",
            "summary": "会员价在 0.5 分边界上差一分钱，应四舍五入实际被舍掉",
            "description": (
                "财务对账发现会员价展示与应付金额在 0.5 分边界上偶发差 1 分钱：例如 PRODUCT-2 按 50%"
                "折扣应为 499.5 分，按展示口径四舍五入应为 500 分，实际展示/返回 499 分。会员价功能是"
                "最近合并进 master 上线的（deploys.log 有 member pricing rollout 记录），QA 在预发"
                "稳定复现：所有落在 .5 分边界的折扣价一律少了 1 分。金额相关的问题需要尽快修，服务"
                "指标正常，怀疑纯代码 bug。"
            ),
            "starts_at": "2026-09-06T16:45:00Z",
        },
    ]
    for spec in d1_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service="recommendation", job="recommendation",
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/D1_新bug实例] suspect_file={spec['file']}：bug 在 GitHub 上游仓 master 的"
                f"{spec['file']}（与 flagd 无关，不注入开关）——Agent 读 workspace/recommendation 克隆"
                "（master）定位 bug，验证「code_fix + suspect_file_hint 命中新文件 + 不做任何线上操作」，"
                "证据拓扑与测评一致（读仓找 bug，不是本地 fixture）"
            ),
        )
        gt = {
            "remediation_type": "code_fix",
            "suspect_service": "recommendation",
            "expect_route": "code_fix_pr",
            "expect_kind_any_of": ["deploy_regression"],
            "expect_confidence_range": [0.6, 0.98],  # 上限 0.98（2026-09-08 口径授权）：教师找到真实代码
            # bug 后的有据高置信（0.96-0.97，代码可验证）不是要防的"无证据过度自信"；防幻觉核心
            # 验收（evidence 可追溯/suspect_file 命中/remediation_type）不变
            "expect_suspect_file_any_of": spec["files"],
        }
        out.append(_v9_seed(family="D1", alert=alert, intended_kind="deploy_regression", gt=gt, injection={"mode": "none"}))

    # ---------------- D2 用户报障型（2 条，无指标异常） ----------------
    d2_specs = [
        {
            "file": "pricing.py", "files": ["pricing.py", "src/pricing.py"],
            "alertname": "MemberPriceComplaintFromUsers",
            "service": "recommendation", "severity": "warning",
            "summary": "客服陆续接到「会员价和实际扣款差一分钱」的用户投诉",
            "description": (
                "客服渠道陆续接到用户投诉：结算页展示的会员价比实际扣款少一分钱（展示 499 分、实扣"
                "500 分这类），集中在带折扣的订单上。监控上没有任何指标异常（错误率 0、延迟正常、"
                "无资源问题），纯业务侧报障。开发已在预发按用户路径复现：折扣后金额落在 .5 分边界时"
                "展示价一律向下少 1 分。业务确认这是金额展示 bug，需要改代码修复，请定位到具体代码。"
            ),
            "starts_at": "2026-09-06T19:20:00Z",
        },
        {
            "file": "cache.py", "files": ["cache.py", "src/cache.py"],
            "alertname": "TrendingNotUpdatingComplaint",
            "service": "recommendation", "severity": "warning",
            "summary": "运营反馈热门位轮换「失灵」：改了配置前台半天不变",
            "description": (
                "运营反馈推荐位的热门榜单「轮换失灵」：后台改了热门位配置，前台 /trending 半天过去"
                "还是旧榜单。无任何指标异常（服务健康、延迟正常、错误率 0），配置下发链路初步排查"
                "正常。开发怀疑服务端缓存没有按预期的 60s TTL 过期，已在预发复现「过了 TTL 仍返回"
                "旧值」。这是功能逻辑问题，需要改代码修复，请定位到具体代码。"
            ),
            "starts_at": "2026-09-06T21:50:00Z",
        },
    ]
    for spec in d2_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service="recommendation", job="recommendation",
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/D2_用户报障] suspect_file={spec['file']}：客服/运营报障叙事（无指标异常），"
                "bug 在 master 的新文件里——验证「无指标信号时仍能正确定位代码根因判 code_fix，"
                "而不是误判成线上操作」"
            ),
        )
        gt = {
            "remediation_type": "code_fix",
            "suspect_service": "recommendation",
            "expect_route": "code_fix_pr",
            "expect_kind_any_of": ["deploy_regression"],
            "expect_confidence_range": [0.6, 0.98],  # 上限 0.98（2026-09-08 口径授权）：教师找到真实代码
            # bug 后的有据高置信（0.96-0.97，代码可验证）不是要防的"无证据过度自信"；防幻觉核心
            # 验收（evidence 可追溯/suspect_file 命中/remediation_type）不变
            "expect_suspect_file_any_of": spec["files"],
        }
        out.append(_v9_seed(family="D2", alert=alert, intended_kind="deploy_regression", gt=gt, injection={"mode": "none"}))

    # ---------------- D3 对抗性误导（2 条） ----------------
    d3_specs = [
        {
            "file": "pagination.py", "files": ["pagination.py", "src/pagination.py"],
            "alertname": "CatalogPaginationWithNoise",
            "service": "recommendation", "severity": "warning",
            "summary": "分页浏览漏商品的投诉（伴随若干无关线索）",
            "description": (
                "商品团队反馈分页浏览接口漏内容：按页遍历 /catalog 会跳过目录最前面的商品（分页功能"
                "最近合并进 master 上线）。附加信息（可能有误，请自行核实）：同时间段值班群里有人提到"
                "ad 服务好像有零星波动，也有人怀疑 kafka 消费有堆积，不确定跟这个问题是否相关；上周"
                "另一次排查时发现过一个前端静态资源缓存的问题，也可能有关。请 Agent 自行核对指标与"
                "代码，不要被这些次生描述带偏，定位真正的根因。"
            ),
            "starts_at": "2026-09-07T15:05:00Z",
        },
        {
            "file": "pricing.py", "files": ["pricing.py", "src/pricing.py"],
            "alertname": "MemberPriceMismatchWithNoise",
            "service": "recommendation", "severity": "warning",
            "summary": "会员价差一分钱的投诉（伴随若干无关线索）",
            "description": (
                "财务与客服都反馈会员价在折扣边界上差一分钱的问题（.5 分边界一律向下少 1 分，会员价"
                "功能最近合并进 master 上线）。附加信息（可能有误，请自行核实）：同期 payment 网关"
                "好像有偶发抖动，quote 服务的延迟也偏高，群里有人猜测是支付/报价链路的问题；也有人"
                "怀疑是前端金额格式化显示问题。请 Agent 自行核对指标与代码，不要被这些猜测带偏，"
                "定位真正的根因。"
            ),
            "starts_at": "2026-09-07T18:25:00Z",
        },
    ]
    for spec in d3_specs:
        alert = _make_alert(
            alertname=spec["alertname"], service="recommendation", job="recommendation",
            severity=spec["severity"], summary=spec["summary"], description=spec["description"],
            starts_at=spec["starts_at"],
            scenario_hint=(
                f"[v9增强/D3_对抗误导] suspect_file={spec['file']}：叙事混入 ad/kafka/payment/quote 等"
                "误导线索，真根因在 recommendation 的 master 新文件——验证「仍正确定位真根因服务与文件」"
            ),
        )
        gt = {
            "remediation_type": "code_fix",
            "suspect_service": "recommendation",
            "expect_route": "code_fix_pr",
            "expect_kind_any_of": ["deploy_regression"],
            "expect_confidence_range": [0.6, 0.98],  # 上限 0.98（2026-09-08 口径授权）：教师找到真实代码
            # bug 后的有据高置信（0.96-0.97，代码可验证）不是要防的"无证据过度自信"；防幻觉核心
            # 验收（evidence 可追溯/suspect_file 命中/remediation_type）不变
            "expect_suspect_file_any_of": spec["files"],
        }
        out.append(_v9_seed(family="D3", alert=alert, intended_kind="deploy_regression", gt=gt, injection={"mode": "none"}))

    assert len(out) == 47, f"v9 增强种子应为 47 条，实际 {len(out)}"
    fam_count: dict[str, int] = {}
    for s in out:
        fam_count[s["family"]] = fam_count.get(s["family"], 0) + 1
    # 注：B1a 8 条中有 2 条（组合开关/死亡机制）在 Phase C 采集期按方案 §4.2 既定降级机制
    # 重标为 B1b（教师三轮稳定走诚实低置信），因此生成器的家族账本为 B1a=6 / B1b=6。
    assert fam_count == {
        "A1": 6, "A2": 4, "A3": 3, "B1a": 6, "B1b": 6, "B2": 3, "B3": 3, "C1": 8,
        "D1": 4, "D2": 2, "D3": 2,
    }, f"v9 家族配比不符: {fam_count}"
    return out


def write_v9_augmentation(out_dir: Path, gt_dir: Path) -> list[str]:
    """把 generate_v9_augmentation() 的 47 条种子落盘：种子 JSON 追加进 out_dir（沿用
    append_seeds 的编号规则，接在现有 _manifest.json 之后），扩展 schema GT 写到 gt_dir
    （文件名 = 种子文件名，rejection_sample.py 按 alert_id 同名查找），并把
    `stem -> {family, injection}` 的编排映射写到 out_dir/_v9_augmentation_map.json
    （collect_v9.py 消费：注入模式 + 批量顺序）。返回种子文件名列表。"""
    seeds = generate_v9_augmentation()
    entries = [
        {
            "alert": s["alert"],
            "generation_path": f"v9_aug_{s['family']}",
            "intended_kind": s["intended_kind"],
        }
        for s in seeds
    ]
    written = append_seeds(entries, out_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)
    aug_map: dict[str, dict[str, Any]] = {}
    for fname, s in zip(written, seeds):
        stem = fname[:-5]
        gt = dict(s["gt"])
        gt["_family"] = s["family"]
        (gt_dir / f"{stem}.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        aug_map[stem] = {"family": s["family"], "injection": s["injection"]}
    (out_dir / "_v9_augmentation_map.json").write_text(
        json.dumps(aug_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return written


def append_seeds(entries: list[dict[str, Any]], out_dir: Path) -> list[str]:
    """在已有种子池基础上追加新条目，编号接着现有 `_manifest.json` 最后一条往后排。

    跟 `write_seeds()` 不同：`write_seeds()` 每次都从 1 重新编号整个目录（用于从零生成整
    个 count 条种子池），会覆盖/打乱已有 700 条的编号；这里只追加，不触碰任何已存在的
    文件。`entries` 跟 `generate_seeds()` 里 manifest 条目同构：
    `{"alert":..., "generation_path":..., "intended_kind":...}`。返回新写入的文件名列表。
    """
    manifest_path = out_dir / "_manifest.json"
    existing: list[dict[str, Any]] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    )
    next_idx = len(existing) + 1
    written: list[str] = []
    for entry in entries:
        fname = f"seed_{next_idx:04d}_{entry['generation_path']}_{entry['intended_kind']}.json"
        fpath = out_dir / fname
        with fpath.open("w", encoding="utf-8") as f:
            json.dump(entry["alert"], f, ensure_ascii=False, indent=2)
            f.write("\n")
        existing.append(
            {"file": fname, "generation_path": entry["generation_path"], "intended_kind": entry["intended_kind"]}
        )
        written.append(fname)
        next_idx += 1
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return written


# ---------------------------------------------------------------------------
# 汇总 + CLI
# ---------------------------------------------------------------------------


def generate_seeds(count: int, seed: int) -> list[dict[str, Any]]:
    """生成 count 条种子告警，返回 manifest 条目列表（每条含 alert + 元信息）。

    三路默认权重 16:9:5（对应 count=30 时正好是 16/9/5 条），按比例线性缩放到
    任意 count（包括文档目标的 700）。
    """
    rng = random.Random(seed)
    path_counts = _split_counts(count, [16, 9, 5])

    manifest: list[dict[str, Any]] = []

    for alert, kind in generate_parametrized(path_counts[0], rng):
        manifest.append({"alert": alert, "generation_path": "parametrized", "intended_kind": kind})
    for alert, kind in generate_flagd_combinations(path_counts[1], rng):
        manifest.append({"alert": alert, "generation_path": "flagd_combination", "intended_kind": kind})
    for alert, kind in generate_historical_reverse_derivation(path_counts[2], rng):
        manifest.append({"alert": alert, "generation_path": "historical_reverse_derivation", "intended_kind": kind})

    return manifest


def write_seeds(manifest: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for idx, entry in enumerate(manifest, start=1):
        path_slug = entry["generation_path"]
        kind_slug = entry["intended_kind"]
        fname = f"seed_{idx:04d}_{path_slug}_{kind_slug}.json"
        fpath = out_dir / fname
        with fpath.open("w", encoding="utf-8") as f:
            json.dump(entry["alert"], f, ensure_ascii=False, indent=2)
            f.write("\n")
        manifest_records.append(
            {
                "file": fname,
                "generation_path": entry["generation_path"],
                "intended_kind": entry["intended_kind"],
            }
        )

    manifest_path = out_dir / "_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="生成种子告警的总数（设计文档全量目标是 700；本仓库默认只落 30 条真实样本）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=GENERATED_DIR_DEFAULT,
        help="输出目录，每条种子写一个 *.json 文件，另加一份 _manifest.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument(
        "--v9-augmentation",
        action="store_true",
        help="生成 v9 数据增强的 47 条定点种子（docs/数据增强方案.md §4）并落盘：种子追加进 "
        "--out-dir（复用 _manifest.json 编号），扩展 schema GT 写进 --gt-dir",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cold_start" / "ground_truth",
        help="--v9-augmentation 模式下 GT 的输出目录",
    )
    args = parser.parse_args()

    if args.v9_augmentation:
        written = write_v9_augmentation(args.out_dir, args.gt_dir)
        fams: dict[str, int] = {}
        for w in written:
            fam = w.split("v9_aug_")[1].rsplit("_", 1)[0]
            fams[fam] = fams.get(fam, 0) + 1
        print(f"[generate_seeds] v9 增强：写入 {len(written)} 条种子到 {args.out_dir}，GT 到 {args.gt_dir}")
        print(f"[generate_seeds] 家族配比: {fams}")
        return

    manifest = generate_seeds(args.count, args.seed)
    write_seeds(manifest, args.out_dir)
    print(f"[generate_seeds] 写出 {len(manifest)} 条种子告警到 {args.out_dir}")


if __name__ == "__main__":
    main()
