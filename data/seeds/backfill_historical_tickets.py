#!/usr/bin/env python3
"""一次性回填脚本：把一批真实合理的历史工单叙事写进 Milvus `aiops_tickets` 集合。

## 为什么需要这个脚本
`generate_seeds.py` 的历史工单反演路（`generate_historical_reverse_derivation`）本来
应该查询 `AIops-agent/agent/integrations/memory.py` 管理的 Milvus 历史工单集合，但集合里
几乎没有真实数据——之前跑真实冷启动采集时（`data/cold_start/collect_trajectories.py`
的 `real_mode()`）为了拿到完整中间步骤，绕开了 `AIops-agent/agent/run.py` 的完整编排，
从未走到 `memory.store_ticket()`，所以 Milvus 里基本是空的，或者只有极少量遗留测试数据
（实测：46 条，其中 28 条是 `route=feishu_low_confidence` 且 `suspect_service=unknown` 的
低置信度兜底记录，只覆盖 4 个服务，检索不出什么有意义的历史工单）。

这个脚本把一批**手写、内容真实合理**的历史工单叙事，通过 `AIops-agent` 真实的
`agent.integrations.memory.store_ticket()`（真实 embedding + 真实写入 Milvus，不是 mock）
灌进集合，让 `generate_seeds.py` 新增的真实检索路径有真正可检索的语料。

## 覆盖面设计原则（详见下面 `_dependency_tickets()` 等分类函数的注释）
每条工单都对齐这几条已核实的项目事实，不瞎编不可能出现的组合：
  - `kind` 枚举、`suspect_service` 取值：见 `AIops-agent/agent/core/schema.py` 的
    `Kind = Literal["dependency", "resource", "deploy_regression", "config"]`，
    以及 `generate_seeds.py` 顶部已核实的 11 个 otel-demo 服务 + kafka。
  - 线上自动处置白名单：`AIops-agent/agent/config.py` 的 `REMEDIATION_ALLOWED_TARGETS`
    默认是 `recommendation,ad,frontend,cart,checkout,currency,payment,shipping,quote,email`
    ——**product-catalog 和 kafka 不在里面**（有状态/未授权），所以这两个服务的工单
    route 只能是 `feishu_online_op` / `info_only` / `feishu_low_confidence` 一类，绝不能是
    `auto_remediated`（那意味着 hook 真的放行执行了一条白名单操作，product-catalog/kafka
    不可能出现在这条台账里）。
  - 代码修复 Agent 当前 MVP 是单仓（`AIops-agent/agent/agents/code_fix.py` 的
    `clone_service_repo` 文档："仓名默认取 config.GITHUB_REPO（当前 MVP 是单仓）"，
    `GITHUB_REPO` 默认 `"recommendation"`）——所以 `code_fix_pr` /
    `feishu_fix_unverified` / `online_op_and_code_fix_pr` /
    `auto_remediated_and_code_fix_pr` 这几个牵涉代码修复的 route，只给 `recommendation`
    服务的工单用，不给别的服务编"这个服务也能自动提 PR"的假故事。
  - route 与 remediation_type 的对应关系：完全对照 `AIops-agent/agent/run.py` 的编排分支
    （`_run_inner` 里 online_op / code_fix / info_only 三条分支），而不是随手编。
  - 下面 `_validate_tickets()` 会在写入前机械校验这些约束，任何一条违反直接抛异常，
    不会带着矛盾数据写进 Milvus。

## 用法
    AIops-agent/.venv/bin/python3 data/seeds/backfill_historical_tickets.py
    AIops-agent/.venv/bin/python3 data/seeds/backfill_historical_tickets.py --dry-run
必须用 `AIops-agent/.venv/bin/python3` 跑（它装了 pymilvus + sentence-transformers；
`aiops-agentic-rl` 自己的 venv 没装 pymilvus，见任务背景）。脚本内部会把
`AIops-agent` 目录加进 `sys.path` 以复用它的 `agent.integrations.memory`，
跟 `data/cold_start/to_llamafactory_format.py` 复用 `agent.agents.prompts` 是同一个模式。

## 幂等性
每条工单的 `fingerprint`（复用 `agent.integrations.ingest.alert_fingerprint()`，由
`alertname|service|severity` 算出）写入前会先跟集合里已有的 fingerprint 去重，重复跑
这个脚本不会插入两遍。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent  # .../aiops-agentic-rl
AIOPS_AGENT_DIR = REPO_ROOT / "AIops-agent"

sys.path.insert(0, str(_THIS_DIR))
import generate_seeds as gs  # noqa: E402  复用 KIND_VALUES，保证 kind 取值口径一致


def _ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


_ensure_aiops_agent_on_path()

# service -> job（otel-demo 真实的 k8s Deployment / docker 容器名）。前 6 个已经在
# generate_seeds.py 的 CONFIRMED_FLAGD_FLAGS 里出现过，后 6 个是同一套服务 slug 的
# 延伸（跟线上自动处置白名单用的服务名一致）。
SERVICE_JOB_MAP: dict[str, str] = {
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

# 线上自动处置白名单：默认镜像线上自动处置白名单的默认值。故意不做运行期 import
# （agent.config 需要 AIops-agent 依赖链，这里只是校验用的静态常量），但下面 main() 里会
# 在真的连上 AIops-agent 时用真实 config 值做一次交叉核对，
# 发现漂移就报错，而不是悄悄用一份过期的本地拷贝。
_DEFAULT_ALLOWED_ONLINE_OP = {
    "recommendation", "ad", "frontend", "cart", "checkout",
    "currency", "payment", "shipping", "quote", "email",
}
CODE_FIX_CAPABLE_SERVICE = "recommendation"  # 当前 MVP 单仓，只有它能真的提代码修复 PR

_CODE_FIX_ROUTES = {
    "code_fix_pr", "feishu_fix_unverified",
    "online_op_and_code_fix_pr", "auto_remediated_and_code_fix_pr",
}
_AUTO_EXECUTED_ROUTES = {"auto_remediated", "auto_remediated_and_code_fix_pr"}
_VALID_ROUTES = {
    "feishu_online_op", "auto_remediated", "code_fix_pr", "feishu_fix_unverified",
    "online_op_and_code_fix_pr", "auto_remediated_and_code_fix_pr", "info_only",
    "feishu_low_confidence",
}


# ---------------------------------------------------------------------------
# 分类 1：dependency —— 某服务自身处理慢/占满线程池，导致上游调用它时报依赖失败/超时。
# 叙事骨架跟 generate_seeds.py 现有 5 条真实历史工单摘要一致："上线了一个 XX 任务 -> 忘记
# 加超时 -> 占满线程池/连接池 -> 上游调用报错/超时"，每条的"什么任务"具体不同，不是简单换
# 服务名的空话。
# ---------------------------------------------------------------------------


def _dependency_tickets() -> list[dict[str, Any]]:
    return [
        {
            "id": "bf-dep-product-catalog-batch-import",
            "service": "product-catalog", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "product-catalog 团队上线了一个后台批量数据导入脚本，该脚本发起的下游查询"
                "忘记加超时，占满了 product-catalog 的处理线程池，GetProduct 接口对上游"
                "（frontend、recommendation）出现级联超时。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart product-catalog 释放被占满的线程池"
                "（product-catalog 不在自动处置白名单内，不能自动执行，需人工确认）；"
                "同时建议下线该批量导入脚本或给其下游查询补上超时。"
            ),
            "evidence": [
                "Jaeger: 慢 span 集中在 product-catalog 的 GetProduct，耗时随时间单调上升",
                "线程池监控: product-catalog 工作线程占用率长期贴近 100%",
                "deploys.log: product-catalog 最近一次发版新增了批量导入脚本",
            ],
        },
        {
            "id": "bf-dep-recommendation-precompute",
            "service": "recommendation", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "recommendation 新增了一个离线预计算任务，任务发起的下游查询没有配超时，"
                "占满 recommendation 自身线程池，frontend 调用 recommendation 接口出现"
                "级联超时。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart recommendation 清理被占满的线程池，止血生效；"
                "建议后续给离线预计算任务的下游查询补上超时并挪到独立进程运行。"
            ),
            "evidence": [
                "Prometheus: recommendation 的请求排队时长 p99 持续升高",
                "Jaeger: 慢 span 集中在离线预计算任务发起的下游查询",
            ],
        },
        {
            "id": "bf-dep-cart-batch-reconcile",
            "service": "cart", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "cart 团队上线了一个购物车对账批处理脚本，脚本里的下游查询同样忘记设超时，"
                "占满 cart 的处理线程池，checkout 调用 cart 接口大面积超时。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart cart，并建议下线对账脚本或改为"
                "异步批处理避开在线请求路径。"
            ),
            "evidence": [
                "Jaeger: checkout -> cart 的调用大面积超时",
                "线程池监控: cart 工作线程占用率长期贴近 100%",
            ],
        },
        {
            "id": "bf-dep-checkout-fraud-score",
            "service": "checkout", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "checkout 在下单流程里新增了一次对第三方反欺诈评分服务的调用，这次调用没有"
                "配超时，评分服务偶发性变慢直接拖慢整条下单链路，checkout 自身线程池被占满，"
                "前端下单按钮长时间无响应。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart checkout 清理被占满的线程池，止血生效；建议给"
                "反欺诈评分调用补上超时+熔断，超时后降级为不评分放行。"
            ),
            "evidence": [
                "Jaeger: checkout 下单链路慢 span 集中在反欺诈评分调用",
                "Prometheus: checkout 请求排队时长与反欺诈服务的响应延迟同步升高",
            ],
        },
        {
            "id": "bf-dep-checkout-shipping-quote",
            "service": "checkout", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "checkout 在下单转换阶段同步调用 shipping 获取运费报价，这次调用没有配"
                "超时；shipping 依赖的 quote 服务当时正处于一次局部限流故障期，checkout "
                "的调用线程被逐个卡死在等待响应上，最终把自己的线程池耗尽。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart checkout；同时建议把运费报价调用"
                "改为异步/带超时，超时后用缓存报价兜底而不是同步阻塞。"
            ),
            "evidence": [
                "Jaeger: checkout -> shipping -> quote 调用链上出现大量挂起 span",
                "线程池监控: checkout 工作线程占用率长期贴近 100%",
            ],
        },
        {
            "id": "bf-dep-payment-antifraud-pool",
            "service": "payment", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "payment 在扣款前调用内部反欺诈风控服务做校验，最近一次变更让这个调用在"
                "超时后会不断重试且没有配退避策略，风控服务短暂抖动期间 payment 的连接池"
                "被大量悬而未决的重试请求耗尽，扣款请求整体排队。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart payment 释放耗尽的连接池；建议"
                "给反欺诈调用加退避重试上限和熔断，避免抖动期间被重试请求打满连接池。"
            ),
            "evidence": [
                "Prometheus: payment 出向连接池使用率短时间冲高到上限",
                "Jaeger: payment 对反欺诈风控服务的调用重试次数明显异常",
            ],
        },
        {
            "id": "bf-dep-ad-user-profile-lookup",
            "service": "ad", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "ad 服务给广告排序新增了一次用户画像查询调用，这次调用没有配超时；画像"
                "服务在一次数据刷新期间响应变慢，ad 服务的请求处理协程大量堆积在等待画像"
                "查询返回上，广告位请求整体超时。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart ad 清理堆积的请求处理协程，止血生效；建议给"
                "画像查询补上超时，超时后跳过个性化排序直接走默认排序兜底。"
            ),
            "evidence": [
                "Jaeger: ad 服务慢 span 集中在用户画像查询",
                "Prometheus: ad 服务请求处理协程数持续攀升",
            ],
        },
        {
            "id": "bf-dep-currency-fx-provider",
            "service": "currency", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "currency 依赖外部汇率服务商拉取实时汇率，这次调用没有配超时；服务商一次"
                "维护窗口期间响应大幅变慢，currency 的请求处理线程逐个卡在等待汇率响应上，"
                "价格换算接口整体超时，checkout 结算页汇率展示报错。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart currency；建议给汇率拉取调用补上"
                "超时，超时后用上一次成功拉取到的汇率缓存值兜底。"
            ),
            "evidence": [
                "Jaeger: currency 慢 span 集中在外部汇率服务商调用",
                "外部依赖状态页: 汇率服务商当时处于维护窗口",
            ],
        },
        {
            "id": "bf-dep-shipping-quote-timeout",
            "service": "shipping", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "shipping 同步调用 quote 服务计算运费，这次调用没有配超时；quote 服务当时"
                "正遭遇局部限流，shipping 的请求处理线程逐个卡死在等待运费报价上，shipping "
                "接口对上游 checkout 出现级联超时。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart shipping 清理卡死的请求处理线程，止血生效；建议"
                "给运费报价调用补上超时+缓存兜底。"
            ),
            "evidence": [
                "Jaeger: shipping -> quote 调用链上出现大量挂起 span",
                "线程池监控: shipping 工作线程占用率长期贴近 100%",
            ],
        },
        {
            "id": "bf-dep-quote-rate-calc-grpc",
            "service": "quote", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "quote 服务内部调用一个独立部署的运费计算 gRPC 依赖，这次调用没有配超时；"
                "运费计算依赖一次版本升级引入了性能回退，响应显著变慢，quote 服务的请求"
                "处理协程大量堆积，quote 对上游 shipping 出现级联超时。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart quote；建议给运费计算 gRPC 调用"
                "补上超时，并联系运费计算依赖团队复核那次版本升级。"
            ),
            "evidence": [
                "Jaeger: quote 慢 span 集中在运费计算 gRPC 调用",
                "Prometheus: quote 服务请求处理协程数持续攀升",
            ],
        },
        {
            "id": "bf-dep-email-smtp-relay",
            "service": "email", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "email 服务在一次促销活动期间批量发送确认邮件，发送调用走的 SMTP 中转"
                "网关没有配超时；中转网关在流量高峰期间响应显著变慢，email 服务的发送"
                "协程大量堆积，邮件确认接口整体超时。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart email 清理堆积的发送协程，止血生效；建议给"
                "SMTP 中转调用补上超时，并把批量发送迁移到限速的异步队列而不是同步调用。"
            ),
            "evidence": [
                "Jaeger: email 慢 span 集中在 SMTP 中转调用",
                "Prometheus: email 服务发送协程数在促销活动期间持续攀升",
            ],
        },
        {
            "id": "bf-dep-frontend-aggregation",
            "service": "frontend", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "frontend 首页需要同时聚合 product-catalog 的商品列表和 recommendation 的"
                "推荐结果，其中对 product-catalog 那一路调用没有配超时；product-catalog "
                "偶发性变慢时，frontend 的页面渲染请求被这一条慢腿拖住，首页整体超时。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart frontend；建议把聚合调用改成"
                "带超时的并发调用，任意一路超时就用兜底数据（如缓存的商品列表）渲染页面。"
            ),
            "evidence": [
                "Jaeger: frontend 首页请求的慢 span 集中在对 product-catalog 的聚合调用",
                "Prometheus: frontend 首页请求 p99 延迟与 product-catalog 响应延迟同步升高",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 分类 2：resource（CPU / GC / 流量过载）—— 没有伴随代码发版，纯资源型瓶颈。
# ---------------------------------------------------------------------------


def _resource_cpu_tickets() -> list[dict[str, Any]]:
    return [
        {
            "id": "bf-res-ad-gc-config",
            "service": "ad", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "ad 服务的一次运行时参数配置变更间接改变了 GC 触发频率，变更后 CPU 使用率"
                "持续处于高位，请求延迟 p99 明显上升，但没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 kubectl set resources deploy/ad --limits=cpu=1500m 临时抬高"
                "CPU 上限缓解压力；建议把 GC 触发频率相关参数改回原值。"
            ),
            "evidence": [
                "Prometheus: ad 服务 CPU 使用率持续 >90%，与请求量曲线不同步",
                "deploys.log: 无对应代码发版记录，同一时间窗口有运行时参数配置变更",
            ],
        },
        {
            "id": "bf-res-payment-serialization-config",
            "service": "payment", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "payment 服务的一次运行时参数变更间接改变了序列化开销，变更后 CPU 使用率"
                "持续处于高位，请求延迟 p99 明显上升，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 kubectl set resources deploy/payment 抬高 CPU"
                "上限；建议核对序列化相关配置项并回滚到变更前的值。"
            ),
            "evidence": [
                "Prometheus: payment 服务 CPU 使用率持续偏高",
                "deploys.log: 无对应代码发版记录",
            ],
        },
        {
            "id": "bf-res-frontend-cache-cleanup-config",
            "service": "frontend", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "frontend 的一次运行时参数配置变更间接改变了内部缓存清理频率，变更后 CPU "
                "使用率持续处于高位，页面响应 p99 明显上升，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker update --cpus 2 frontend 临时扩容 CPU 上限缓解压力；"
                "建议把缓存清理频率参数改回原值。"
            ),
            "evidence": [
                "Prometheus: frontend CPU 使用率持续偏高，页面响应 p99 同步升高",
                "deploys.log: 无对应代码发版记录",
            ],
        },
        {
            "id": "bf-res-currency-fx-cache-interval",
            "service": "currency", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "currency 的汇率缓存刷新间隔被一次配置变更误设成了远小于原值的数字，导致"
                "服务陷入接近死循环式的高频轮询拉取汇率，CPU 使用率持续处于高位，没有伴随"
                "代码发版。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把汇率缓存刷新间隔配置改回原值；临时可执行"
                "docker update --cpus 1.5 currency 缓解压力。"
            ),
            "evidence": [
                "Prometheus: currency 服务 CPU 使用率持续偏高，出向请求频率异常升高",
                "配置变更记录: 汇率缓存刷新间隔被从分钟级误改成秒级",
            ],
        },
        {
            "id": "bf-res-product-catalog-fuzzy-search-flag",
            "service": "product-catalog", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "product-catalog 的一个模糊搜索功能开关被误打开（本意只在测试环境启用），"
                "该模式下每次查询的计算成本明显更高，CPU 使用率持续处于高位，没有伴随代码"
                "发版。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把模糊搜索开关改回关闭状态（product-catalog 不在"
                "自动处置白名单内，不能自动执行任何操作）。"
            ),
            "evidence": [
                "Prometheus: product-catalog CPU 使用率持续偏高，与请求量曲线不同步",
                "配置变更记录: 模糊搜索功能开关被打开",
            ],
        },
        {
            "id": "bf-res-cart-session-merge-config",
            "service": "cart", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "cart 的一次运行时参数配置变更让会话合并逻辑的触发频率明显升高，CPU 使用率"
                "持续处于高位，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker update --cpus 1.5 cart 临时扩容 CPU 上限缓解压力；"
                "建议把会话合并触发频率参数改回原值。"
            ),
            "evidence": [
                "Prometheus: cart 服务 CPU 使用率持续偏高",
                "deploys.log: 无对应代码发版记录",
            ],
        },
        {
            "id": "bf-res-checkout-retry-policy-config",
            "service": "checkout", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "checkout 的重试策略配置被调得过于激进（更短的重试间隔、更多的重试次数），"
                "CPU 使用率持续处于高位，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把重试策略配置改回原值；临时可执行"
                "kubectl set resources deploy/checkout 抬高 CPU 上限。"
            ),
            "evidence": [
                "Prometheus: checkout CPU 使用率持续偏高，出向重试请求数异常升高",
                "配置变更记录: 重试间隔/次数配置被调整",
            ],
        },
        {
            "id": "bf-res-recommendation-batch-size-config",
            "service": "recommendation", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "recommendation 的推理批大小配置被一次变更调低，导致单位时间内需要处理的"
                "批次数明显增多、单位请求的固定开销占比上升，CPU 使用率持续处于高位，没有"
                "伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker update --cpus 2 recommendation 临时扩容 CPU 上限缓解"
                "压力；建议把推理批大小配置改回原值。"
            ),
            "evidence": [
                "Prometheus: recommendation CPU 使用率持续偏高，批处理次数明显上升",
                "配置变更记录: 推理批大小参数被调低",
            ],
        },
        {
            "id": "bf-res-shipping-rate-algo-config",
            "service": "shipping", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "shipping 的运费计算算法配置被切换到了一个计算成本更高的模式，CPU 使用率"
                "持续处于高位，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把运费计算算法配置切回原模式；临时可执行"
                "docker update --cpus 1.5 shipping 缓解压力。"
            ),
            "evidence": [
                "Prometheus: shipping CPU 使用率持续偏高",
                "配置变更记录: 运费计算算法模式被切换",
            ],
        },
        {
            "id": "bf-res-quote-regex-config",
            "service": "quote", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "quote 服务的入参校验正则表达式配置被一次变更改成了一个存在回溯风险的写法，"
                "个别输入会触发正则回溯，CPU 使用率持续处于高位，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker update --cpus 1.5 quote 临时扩容 CPU 上限缓解压力；"
                "建议把入参校验正则配置改回原写法，避免回溯风险。"
            ),
            "evidence": [
                "Prometheus: quote CPU 使用率间歇性冲高，与特定请求模式相关",
                "配置变更记录: 入参校验正则表达式被修改",
            ],
        },
        {
            "id": "bf-res-email-template-engine-config",
            "service": "email", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "email 服务的模板渲染引擎配置被切换到了一个更慢的渲染模式，CPU 使用率"
                "持续处于高位，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把模板渲染引擎配置切回原模式；临时可执行"
                "kubectl set resources deploy/email 抬高 CPU 上限。"
            ),
            "evidence": [
                "Prometheus: email CPU 使用率持续偏高",
                "配置变更记录: 模板渲染引擎模式被切换",
            ],
        },
        {
            "id": "bf-res-frontend-traffic-flood",
            "service": "frontend", "kind": "resource",
            "alertname": "FrontendTrafficFlood",
            "summary": (
                "frontend 首页请求量在短时间内暴增（活动预热带来的流量洪峰），实例 CPU 与"
                "连接数迅速逼近上限，属于流量驱动的资源型过载，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker update --cpus 2 --memory 1024m frontend 临时扩容资源；"
                "建议评估是否需要提前扩容副本数应对后续同类流量峰值。"
            ),
            "evidence": [
                "Prometheus: frontend 入口流量在短时间内成倍增长",
                "Prometheus: frontend 实例 CPU/连接数同步逼近上限",
            ],
        },
        {
            "id": "bf-res-ad-manual-gc-pause",
            "service": "ad", "kind": "resource",
            "alertname": "AdServiceGcPause",
            "summary": (
                "ad 服务出现明显的周期性 GC 停顿，跟前面持续型高 CPU 不同，这次是每隔一段"
                "时间出现一次延迟毛刺而非持续打满，没有伴随代码发版。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工核对 JVM/运行时 GC 相关参数配置；临时可执行"
                "docker restart ad 重置一次 GC 周期。"
            ),
            "evidence": [
                "Prometheus: ad 服务延迟 p99 呈现周期性毛刺而非持续偏高",
                "运行时指标: GC 停顿次数和单次停顿时长同步升高",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 分类 3：resource（队列积压）—— kafka 消费 lag，suspect_service 固定是 kafka
# （kafka 是有状态组件，不在自动处置白名单内，route 只能走人工）。
# ---------------------------------------------------------------------------


def _queue_backlog_tickets() -> list[dict[str, Any]]:
    return [
        {
            "id": "bf-queue-orders-consumer-scaledown",
            "service": "kafka", "kind": "resource",
            "alertname": "KafkaConsumerLag",
            "summary": (
                "一次误操作把 orders topic 消费者组 orders-consumer 的副本数从 6 缩到了 2，"
                "生产速率不变但消费能力骤降，consumer lag 迅速堆积，订单相关下游处理明显"
                "滞后。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把 orders-consumer 副本数恢复到 6（kafka 是有状态"
                "组件，不在自动处置白名单内，不能自动扩容）。"
            ),
            "evidence": [
                "Kafka 监控: orders-consumer 消费者组副本数从 6 降到 2",
                "Kafka 监控: orders topic 上 consumer lag 单调上升",
            ],
        },
        {
            "id": "bf-queue-orders-flash-sale-burst",
            "service": "kafka", "kind": "resource",
            "alertname": "KafkaConsumerLag",
            "summary": (
                "一次未提前报备的限时秒杀活动带来订单事件生产速率的突增，orders topic 的"
                "consumer lag 迅速堆积，orders-consumer 本身运行正常，纯粹是生产吞吐短时"
                "超过了当前消费能力。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工临时扩容 orders-consumer 副本数应对突增生产速率；"
                "建议后续活动前提前报备以便预扩容。"
            ),
            "evidence": [
                "Kafka 监控: orders topic 生产速率短时间内成倍增长",
                "Kafka 监控: orders-consumer 消费者组本身健康，无重启/报错",
            ],
        },
        {
            "id": "bf-queue-notifications-consumer-scaledown",
            "service": "kafka", "kind": "resource",
            "alertname": "KafkaConsumerLag",
            "summary": (
                "一次误操作把通知消费者组 notify-consumer 的副本数从 4 缩到了 1，生产速率"
                "不变但消费能力骤降，consumer lag 迅速堆积，通知发送明显滞后。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把 notify-consumer 副本数恢复到 4。"
            ),
            "evidence": [
                "Kafka 监控: notify-consumer 消费者组副本数从 4 降到 1",
                "Kafka 监控: notifications topic 上 consumer lag 单调上升",
            ],
        },
        {
            "id": "bf-queue-shipping-updates-downstream-ratelimit",
            "service": "kafka", "kind": "resource",
            "alertname": "KafkaConsumerLag",
            "summary": (
                "shipping-updates topic 的消费者 shipping-consumer 处理速度明显变慢，"
                "排查发现是消费逻辑调用的下游物流状态更新 API 被对方限流，单条消息处理"
                "耗时被拉长，consumer lag 持续增长。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工联系物流状态更新 API 提供方核实限流阈值，评估是否"
                "需要申请提额或给 shipping-consumer 加批量提交降低调用频率。"
            ),
            "evidence": [
                "Kafka 监控: shipping-updates topic 上 consumer lag 持续增长",
                "外部依赖调用日志: 物流状态更新 API 返回大量限流响应",
            ],
        },
        {
            "id": "bf-queue-shipping-updates-consumer-scaledown",
            "service": "kafka", "kind": "resource",
            "alertname": "KafkaConsumerLag",
            "summary": (
                "一次容量规划失误把物流更新消费者组 shipping-consumer 的副本数从 5 缩到了 "
                "2，生产速率不变但消费能力骤降，consumer lag 迅速堆积，物流状态更新明显"
                "滞后。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把 shipping-consumer 副本数恢复到 5。"
            ),
            "evidence": [
                "Kafka 监控: shipping-consumer 消费者组副本数从 5 降到 2",
                "Kafka 监控: shipping-updates topic 上 consumer lag 单调上升",
            ],
        },
        {
            "id": "bf-queue-payment-events-settlement-burst",
            "service": "kafka", "kind": "resource",
            "alertname": "KafkaConsumerLag",
            "summary": (
                "一次批量结算任务在短时间内向 payment-events topic 灌入了远超日常水平的"
                "事件量，payment-consumer 本身运行正常，但生产速率短时间内远超消费能力，"
                "consumer lag 迅速堆积。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工临时扩容 payment-consumer 副本数消化积压；建议后续"
                "把批量结算任务改成限速分批写入，避免瞬时冲击。"
            ),
            "evidence": [
                "Kafka 监控: payment-events topic 生产速率短时间内成倍增长",
                "Kafka 监控: payment-consumer 消费者组本身健康，无重启/报错",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 分类 4：deploy_regression（内存泄漏）—— 必须"最近一次发版之后开始"。只有 recommendation
# 是当前单仓 MVP 里真的能走代码修复 Agent 提 PR 的服务，其余服务的泄漏工单只走
# online_op（重启止血），不editorial 编"这个服务也能自动提 PR"的假故事。
# ---------------------------------------------------------------------------


def _deploy_regression_tickets() -> list[dict[str, Any]]:
    mech = {m["slug"]: m for m in gs.LEAK_MECHANISMS}
    return [
        {
            "id": "bf-leak-recommendation-list-append-codefix",
            "service": "recommendation", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "recommendation（Python/gRPC）容器的工作集内存在负载下单调上升，重启次数"
                "不断增加（被 OOMKilled）。问题从最近一次发版之后立刻开始，"
                f"{mech['unbounded_list_append']['clue']}。"
            ),
            "remediation_type": "code_fix", "route": "code_fix_pr",
            "remediation_detail": (
                "代码修复 Agent 已定位到问题 commit，去掉了无界 list append 逻辑改为"
                "有界队列，build/test 通过后已提交 PR。"
            ),
            "evidence": [
                "容器指标: recommendation 工作集内存随请求量单调上升，多次 OOMKilled",
                "deploys.log: 内存爬升起点与最近一次发版时间点吻合",
            ],
        },
        {
            "id": "bf-leak-recommendation-dict-accum-unverified",
            "service": "recommendation", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "recommendation 服务的内存占用最近持续走高，容器频繁被 OOMKilled 重启。"
                f"时间线上正好卡在最近一次发版之后，{mech['unbounded_dict_accumulation']['clue']}。"
            ),
            "remediation_type": "code_fix", "route": "feishu_fix_unverified",
            "remediation_detail": (
                "代码修复 Agent 定位到疑似问题 commit 并尝试补丁，但改动后 build/test 未"
                "通过验证，已降级发飞书卡片交人工介入复核。"
            ),
            "evidence": [
                "容器指标: recommendation 工作集内存单调上升",
                "deploys.log: 最近一次发版新增了一个全局缓存字典",
            ],
        },
        {
            "id": "bf-leak-recommendation-file-handle-hybrid-auto",
            "service": "recommendation", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "观察到 recommendation（Python/gRPC）在负载下内存不断攀升，最终触发 OOM "
                "被强制重启，且重启频率还在增加。这个趋势是从最新一次发版开始出现的，"
                f"{mech['unclosed_file_handle']['clue']}。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated_and_code_fix_pr",
            "also_code_fix": True,
            "remediation_detail": (
                "已自动执行 docker restart recommendation 先行止血释放内存；同时判定为"
                "混合根因（also_code_fix），代码修复 Agent 已定位到未关闭的文件句柄并提交"
                "PR 根治，build/test 均通过。"
            ),
            "evidence": [
                "容器指标: recommendation 工作集内存单调上升，重启频率持续增加",
                "deploys.log: 最近一次发版新增了日志文件写入逻辑",
            ],
        },
        {
            "id": "bf-leak-recommendation-weakset-hybrid-manual",
            "service": "recommendation", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "recommendation 容器工作集内存在负载下持续上升，尚未触发 OOM 但趋势明显，"
                f"时间线卡在最近一次发版之后，{mech['misused_weakset']['clue']}。"
            ),
            "remediation_type": "online_op", "route": "online_op_and_code_fix_pr",
            "also_code_fix": True,
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart recommendation 先行止血（判定为"
                "混合根因，也触发了代码修复 Agent）；代码修复 Agent 已定位到 WeakSet 用错"
                "的问题 commit 并提交 PR，build/test 均通过。"
            ),
            "evidence": [
                "容器指标: recommendation 工作集内存持续上升，趋势明显但尚未 OOM",
                "deploys.log: 最近一次发版改动了缓存相关的引用管理逻辑",
            ],
        },
        {
            "id": "bf-leak-cart-file-handle",
            "service": "cart", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "cart 服务内存占用最近持续走高，容器频繁重启。时间线上正好卡在最近一次"
                f"发版之后，{mech['unclosed_file_handle']['clue']}。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart cart 先行止血释放内存（当前单仓 MVP 只有"
                "recommendation 有可修复的代码仓，cart 暂无法触发自动代码修复，需要人工"
                "另行安排代码复核）。"
            ),
            "evidence": [
                "容器指标: cart 工作集内存单调上升，多次重启",
                "deploys.log: 最近一次发版新增了日志/临时文件写入逻辑",
            ],
        },
        {
            "id": "bf-leak-checkout-weakset",
            "service": "checkout", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "checkout 服务内存占用最近持续走高，容器重启次数增加。时间线上正好卡在"
                f"最近一次发版之后，{mech['misused_weakset']['clue']}。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart checkout 先行止血；建议安排人工"
                "代码复核最近一次发版改动的缓存引用管理逻辑。"
            ),
            "evidence": [
                "容器指标: checkout 工作集内存单调上升",
                "deploys.log: 最近一次发版改动了会话缓存的引用管理逻辑",
            ],
        },
        {
            "id": "bf-leak-ad-dict-accumulation",
            "service": "ad", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "ad 服务内存占用最近持续走高，容器频繁重启。时间线上正好卡在最近一次"
                f"发版之后，{mech['unbounded_dict_accumulation']['clue']}（这次是给用户"
                "画像结果加的缓存字典）。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart ad 先行止血释放内存；建议安排人工代码复核最近"
                "一次发版新增的画像结果缓存字典逻辑。"
            ),
            "evidence": [
                "容器指标: ad 工作集内存单调上升，多次重启",
                "deploys.log: 最近一次发版新增了画像结果缓存字典",
            ],
        },
        {
            "id": "bf-leak-payment-list-append",
            "service": "payment", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "payment 服务内存占用最近持续走高，容器重启次数增加。时间线上正好卡在"
                f"最近一次发版之后，{mech['unbounded_list_append']['clue']}（这次是给"
                "失败重试的请求加的待重试列表）。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart payment 先行止血；建议安排人工"
                "代码复核最近一次发版新增的待重试请求列表逻辑。"
            ),
            "evidence": [
                "容器指标: payment 工作集内存单调上升",
                "deploys.log: 最近一次发版新增了失败重试待处理列表",
            ],
        },
        {
            "id": "bf-leak-frontend-session-cache",
            "service": "frontend", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "frontend 服务内存占用最近持续走高，容器频繁重启。时间线上正好卡在最近"
                f"一次发版之后，{mech['unbounded_dict_accumulation']['clue']}（这次是给"
                "会话状态加的缓存字典）。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart frontend 先行止血释放内存；建议安排人工代码"
                "复核最近一次发版新增的会话状态缓存字典逻辑。"
            ),
            "evidence": [
                "容器指标: frontend 工作集内存单调上升，多次重启",
                "deploys.log: 最近一次发版新增了会话状态缓存字典",
            ],
        },
        {
            "id": "bf-leak-currency-fx-cache-dict",
            "service": "currency", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "currency 服务内存占用最近持续走高，容器重启次数增加。时间线上正好卡在"
                f"最近一次发版之后，{mech['unbounded_dict_accumulation']['clue']}（这次是"
                "给汇率查询结果加的缓存字典）。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart currency 先行止血；建议安排人工"
                "代码复核最近一次发版新增的汇率结果缓存字典逻辑。"
            ),
            "evidence": [
                "容器指标: currency 工作集内存单调上升",
                "deploys.log: 最近一次发版新增了汇率结果缓存字典",
            ],
        },
        {
            "id": "bf-leak-shipping-label-list",
            "service": "shipping", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "shipping 服务内存占用最近持续走高，容器频繁重启。时间线上正好卡在最近"
                f"一次发版之后，{mech['unbounded_list_append']['clue']}（这次是给生成的"
                "运单标签加的记录列表）。"
            ),
            "remediation_type": "online_op", "route": "auto_remediated",
            "remediation_detail": (
                "已自动执行 docker restart shipping 先行止血释放内存；建议安排人工代码"
                "复核最近一次发版新增的运单标签记录列表逻辑。"
            ),
            "evidence": [
                "容器指标: shipping 工作集内存单调上升，多次重启",
                "deploys.log: 最近一次发版新增了运单标签记录列表",
            ],
        },
        {
            "id": "bf-leak-quote-request-cache-dict",
            "service": "quote", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "quote 服务内存占用最近持续走高，容器重启次数增加。时间线上正好卡在最近"
                f"一次发版之后，{mech['unbounded_dict_accumulation']['clue']}（这次是给"
                "运费报价请求结果加的缓存字典）。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart quote 先行止血；建议安排人工"
                "代码复核最近一次发版新增的报价结果缓存字典逻辑。"
            ),
            "evidence": [
                "容器指标: quote 工作集内存单调上升",
                "deploys.log: 最近一次发版新增了报价结果缓存字典",
            ],
        },
        {
            "id": "bf-leak-email-weakset",
            "service": "email", "kind": "deploy_regression",
            "alertname": "MemoryLeakOOM",
            "summary": (
                "email 服务内存占用最近持续走高，容器频繁重启。时间线上正好卡在最近一次"
                f"发版之后，{mech['misused_weakset']['clue']}（这次是给批量发送任务的"
                "去重追踪逻辑加的引用管理）。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工执行 docker restart email 先行止血；建议安排人工"
                "代码复核最近一次发版改动的去重追踪引用管理逻辑。"
            ),
            "evidence": [
                "容器指标: email 工作集内存单调上升",
                "deploys.log: 最近一次发版改动了批量发送去重追踪逻辑",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 分类 5：config —— 纯配置/证书类漂移，没有代码发版，也不是持续的资源打满。
# ---------------------------------------------------------------------------


def _config_tickets() -> list[dict[str, Any]]:
    return [
        {
            "id": "bf-config-payment-cert-rotation",
            "service": "payment", "kind": "config",
            "alertname": "ConfigCertMismatch",
            "summary": (
                "payment 网关一次证书轮换后，调用方配置的信任链没有同步更新，导致支付"
                "请求间歇性握手失败，checkout 链路偶发超时，人工介入更新配置后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工同步更新调用方信任链配置。",
            "evidence": [
                "TLS 握手日志: payment 网关间歇性握手失败，错误码指向证书链不匹配",
                "变更记录: payment 网关证书最近完成一次轮换",
            ],
        },
        {
            "id": "bf-config-ad-cert-rotation",
            "service": "ad", "kind": "config",
            "alertname": "ConfigCertMismatch",
            "summary": (
                "ad 服务依赖的一个内部网关证书轮换后，调用方配置的信任链没有同步更新，"
                "导致广告请求间歇性握手失败，frontend 首页偶发广告位超时，人工介入更新"
                "配置后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工同步更新调用方信任链配置。",
            "evidence": [
                "TLS 握手日志: ad 服务对内部网关的调用间歇性握手失败",
                "变更记录: 内部网关证书最近完成一次轮换",
            ],
        },
        {
            "id": "bf-config-checkout-cert-rotation",
            "service": "checkout", "kind": "config",
            "alertname": "ConfigCertMismatch",
            "summary": (
                "checkout 调用的一个内部服务证书轮换后，调用方信任链配置没有同步更新，"
                "导致下单请求间歇性握手失败，人工介入更新配置后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工同步更新调用方信任链配置。",
            "evidence": [
                "TLS 握手日志: checkout 对内部服务的调用间歇性握手失败",
                "变更记录: 该内部服务证书最近完成一次轮换",
            ],
        },
        {
            "id": "bf-config-currency-cert-rotation",
            "service": "currency", "kind": "config",
            "alertname": "ConfigCertMismatch",
            "summary": (
                "currency 依赖的外部汇率服务商证书轮换后，currency 侧配置的信任链没有"
                "同步更新，导致汇率拉取间歇性握手失败，人工介入更新配置后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工同步更新信任链配置。",
            "evidence": [
                "TLS 握手日志: currency 对外部汇率服务商的调用间歇性握手失败",
                "外部依赖公告: 汇率服务商证书完成一次轮换",
            ],
        },
        {
            "id": "bf-config-shipping-cert-rotation",
            "service": "shipping", "kind": "config",
            "alertname": "ConfigCertMismatch",
            "summary": (
                "shipping 依赖的内部服务证书轮换后，调用方信任链配置没有同步更新，导致"
                "运费查询间歇性握手失败，人工介入更新配置后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工同步更新调用方信任链配置。",
            "evidence": [
                "TLS 握手日志: shipping 对内部服务的调用间歇性握手失败",
                "变更记录: 该内部服务证书最近完成一次轮换",
            ],
        },
        {
            "id": "bf-config-quote-carrier-cert-rotation",
            "service": "quote", "kind": "config",
            "alertname": "ConfigCertMismatch",
            "summary": (
                "quote 依赖的第三方承运商网关证书轮换后，quote 侧信任链配置没有同步"
                "更新，导致报价请求间歇性握手失败，人工介入更新配置后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工同步更新信任链配置。",
            "evidence": [
                "TLS 握手日志: quote 对承运商网关的调用间歇性握手失败",
                "外部依赖公告: 承运商网关证书完成一次轮换",
            ],
        },
        {
            "id": "bf-config-email-smtp-tls-cert",
            "service": "email", "kind": "config",
            "alertname": "ConfigCertMismatch",
            "summary": (
                "email 依赖的 SMTP 中转网关一次 TLS 证书轮换后，email 侧信任链配置没有"
                "同步更新，导致部分邮件发送间歇性握手失败，人工介入更新配置后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工同步更新信任链配置。",
            "evidence": [
                "TLS 握手日志: email 对 SMTP 中转网关的调用间歇性握手失败",
                "变更记录: SMTP 中转网关证书最近完成一次轮换",
            ],
        },
        {
            "id": "bf-config-frontend-image-slow-load",
            "service": "frontend", "kind": "config",
            "alertname": "FrontendImageSlowLoad",
            "summary": (
                "frontend 首页图片加载耗时显著升高，排查确认该行为完全由一次功能开关"
                "配置变更触发，没有对应的代码发版，也没有下游依赖报错或资源占用异常，"
                "纯粹是配置项本身的取值问题，人工把开关改回原值后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工把图片加载相关功能开关改回原值。",
            "evidence": [
                "前端性能监控: 首页图片加载耗时显著升高",
                "配置变更记录: 图片加载相关功能开关被切换",
            ],
        },
        {
            "id": "bf-config-product-catalog-search-flag",
            "service": "product-catalog", "kind": "config",
            "alertname": "FeatureFlagMisconfig",
            "summary": (
                "product-catalog 的搜索相关性排序开关被误配置成了一个实验性档位（本意"
                "只在小流量测试环境启用），影响了部分商品的搜索排序结果，没有对应的代码"
                "发版，人工把开关改回原值后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": (
                "已发飞书卡片提示人工把搜索相关性排序开关改回原值（product-catalog 不在"
                "自动处置白名单内，不能自动执行）。"
            ),
            "evidence": [
                "业务监控: 部分商品搜索排序结果异常",
                "配置变更记录: 搜索相关性排序开关被切换到实验档位",
            ],
        },
        {
            "id": "bf-config-cart-discount-rule-flag",
            "service": "cart", "kind": "config",
            "alertname": "FeatureFlagMisconfig",
            "summary": (
                "cart 的一个折扣规则开关被误保留在开启状态（原计划是活动结束后关闭），"
                "导致活动结束后购物车仍按活动折扣计算价格，没有对应的代码发版，人工把"
                "开关关闭后恢复。"
            ),
            "remediation_type": "info_only", "route": "info_only",
            "remediation_detail": (
                "问题影响面小且已由业务方确认并关闭对应开关，此次只需知会，无需 Agent "
                "额外执行操作。"
            ),
            "evidence": [
                "业务监控: 活动结束后购物车折扣计算仍按活动价生效",
                "配置变更记录: 折扣规则开关未按计划在活动结束后关闭",
            ],
        },
        {
            "id": "bf-config-recommendation-ab-flag",
            "service": "recommendation", "kind": "config",
            "alertname": "FeatureFlagMisconfig",
            "summary": (
                "recommendation 的一个 A/B 测试流量分配开关被误配置成了全量导向实验组"
                "模型版本（本意只分配 5% 流量），没有对应的代码发版，人工把流量分配比例"
                "改回原值后恢复。"
            ),
            "remediation_type": "online_op", "route": "feishu_online_op",
            "remediation_detail": "已发飞书卡片提示人工把 A/B 测试流量分配比例改回原值。",
            "evidence": [
                "业务监控: 实验组模型版本的流量占比远超预期的 5%",
                "配置变更记录: A/B 测试流量分配开关被误改",
            ],
        },
    ]


# ---------------------------------------------------------------------------
# 分类 6：resource（信息类/低影响，用于覆盖 info_only route 的资源型场景）
# ---------------------------------------------------------------------------


def _info_only_tickets() -> list[dict[str, Any]]:
    return [
        {
            "id": "bf-info-product-catalog-transient-blip",
            "service": "product-catalog", "kind": "dependency",
            "alertname": "DependencyFailureRate",
            "summary": (
                "product-catalog 出现一次短暂的依赖调用错误率抬升，持续不到两分钟便自行"
                "恢复，排查确认是下游一次瞬时网络抖动导致，没有对应的代码发版或配置变更，"
                "无需任何处置。"
            ),
            "remediation_type": "info_only", "route": "info_only",
            "remediation_detail": "瞬时抖动已自行恢复，无需任何处置，仅记录知会。",
            "evidence": [
                "Prometheus: product-catalog 依赖调用错误率短暂抬升后自行回落",
                "网络监控: 同一时间窗口存在一次瞬时网络抖动",
            ],
        },
        {
            "id": "bf-info-ad-cpu-transient-blip",
            "service": "ad", "kind": "resource",
            "alertname": "HighCpuUsage",
            "summary": (
                "ad 服务出现一次短暂的 CPU 使用率抬升，持续约一分钟后自行回落，排查确认"
                "是一次计划内的定时批处理任务导致，没有对应的代码发版，无需任何处置。"
            ),
            "remediation_type": "info_only", "route": "info_only",
            "remediation_detail": "计划内批处理任务导致的短暂波动，无需任何处置，仅记录知会。",
            "evidence": [
                "Prometheus: ad 服务 CPU 使用率短暂抬升后自行回落",
                "任务调度日志: 同一时间窗口存在一次计划内定时批处理任务",
            ],
        },
    ]


def all_tickets() -> list[dict[str, Any]]:
    return (
        _dependency_tickets()
        + _resource_cpu_tickets()
        + _queue_backlog_tickets()
        + _deploy_regression_tickets()
        + _config_tickets()
        + _info_only_tickets()
    )


# ---------------------------------------------------------------------------
# 校验：写入前机械核对 kind/service/route/remediation_type 的组合不自相矛盾。
# ---------------------------------------------------------------------------


def _validate_tickets(tickets: list[dict[str, Any]], allowed_online_op: set[str]) -> None:
    seen_ids: set[str] = set()
    for t in tickets:
        tid = t["id"]
        assert tid not in seen_ids, f"重复的工单 id: {tid}"
        seen_ids.add(tid)

        assert t["kind"] in gs.KIND_VALUES, f"{tid}: 未知 kind {t['kind']!r}"
        assert t["service"] in SERVICE_JOB_MAP, f"{tid}: 未知 service {t['service']!r}"
        assert t["route"] in _VALID_ROUTES, f"{tid}: 未知 route {t['route']!r}"
        assert t["remediation_type"] in {"online_op", "code_fix", "info_only"}, (
            f"{tid}: 未知 remediation_type {t['remediation_type']!r}"
        )
        assert len(t["summary"]) > 20, f"{tid}: summary 太短，像是空洞占位内容"
        assert t["remediation_detail"], f"{tid}: remediation_detail 不能为空"
        assert t.get("evidence"), f"{tid}: evidence 不能为空"

        also_code_fix = t.get("also_code_fix", False)

        # 代码修复相关 route 只允许出现在唯一有真实仓库的服务上（单仓 MVP 约束）。
        if t["route"] in _CODE_FIX_ROUTES:
            assert t["service"] == CODE_FIX_CAPABLE_SERVICE, (
                f"{tid}: route={t['route']!r} 意味着触发了代码修复 Agent，但 service="
                f"{t['service']!r} 不是当前单仓 MVP 里唯一有真实仓库的 "
                f"{CODE_FIX_CAPABLE_SERVICE!r}"
            )

        # 已自动执行（auto_remediated 系列）意味着 hook 真的放行了一条白名单操作，
        # 目标必须在 ALLOWED_TARGETS 里（有状态组件/未授权目标一律只能走人工）。
        if t["route"] in _AUTO_EXECUTED_ROUTES:
            assert t["service"] in allowed_online_op, (
                f"{tid}: route={t['route']!r} 意味着已自动执行线上处置，但 service="
                f"{t['service']!r} 不在自动处置白名单 {sorted(allowed_online_op)} 内"
            )

        # route 与 remediation_type 的对应关系要跟 AIops-agent/agent/run.py 的编排分支一致。
        if t["route"] == "info_only":
            assert t["remediation_type"] == "info_only", tid
        if t["route"] in {"code_fix_pr", "feishu_fix_unverified"}:
            assert t["remediation_type"] == "code_fix", tid
        if t["route"] in {
            "feishu_online_op", "auto_remediated",
            "online_op_and_code_fix_pr", "auto_remediated_and_code_fix_pr",
        }:
            assert t["remediation_type"] == "online_op", tid
        if t["route"] in {"online_op_and_code_fix_pr", "auto_remediated_and_code_fix_pr"}:
            assert also_code_fix, f"{tid}: 混合根因 route 必须标注 also_code_fix=True"


# ---------------------------------------------------------------------------
# 写入 Milvus：复用 memory.store_ticket()，构造成它期望的 report dict 形状。
# ---------------------------------------------------------------------------


def _build_report(ticket: dict[str, Any]) -> dict[str, Any]:
    from agent.integrations import ingest  # type: ignore

    service = ticket["service"]
    job = SERVICE_JOB_MAP[service]
    severity = "critical" if (
        ticket["kind"] == "deploy_regression"
        or ticket["route"] in _CODE_FIX_ROUTES | _AUTO_EXECUTED_ROUTES
    ) else "warning"
    alert = {
        "alertname": ticket["alertname"],
        "service": service,
        "labels": {"service": service, "severity": severity, "job": job},
        "annotations": {"summary": ticket["summary"][:80], "description": ticket["summary"]},
    }
    fingerprint = ingest.alert_fingerprint(alert)
    diagnosis = {
        "summary": ticket["summary"],
        "kind": ticket["kind"],
        "suspect_service": service,
        "remediation_type": ticket["remediation_type"],
        "remediation_detail": ticket["remediation_detail"],
        "confidence": 0.9,
        "evidence": ticket["evidence"],
        "also_code_fix": ticket.get("also_code_fix", False),
    }
    return {
        "alert": alert,
        "diagnosis": diagnosis,
        "meta": {"fingerprint": fingerprint},
        "route": ticket["route"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只校验+打印，不真的写入 Milvus")
    args = parser.parse_args()

    tickets = all_tickets()

    try:
        from agent import config as agent_config  # type: ignore

        allowed_online_op = set(
            x.strip() for x in agent_config.REMEDIATION_ALLOWED_TARGETS.split(",") if x.strip()
        )
        if allowed_online_op != _DEFAULT_ALLOWED_ONLINE_OP:
            print(
                f"[backfill] 警告: 真实 REMEDIATION_ALLOWED_TARGETS={sorted(allowed_online_op)} "
                f"跟本脚本假设的默认值 {sorted(_DEFAULT_ALLOWED_ONLINE_OP)} 不一致，"
                "以真实配置为准做校验。"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[backfill] 无法导入 agent.config（{exc}），退回本脚本内置的默认白名单做校验。")
        allowed_online_op = _DEFAULT_ALLOWED_ONLINE_OP

    _validate_tickets(tickets, allowed_online_op)
    print(f"[backfill] 校验通过，共 {len(tickets)} 条历史工单待写入。")

    from collections import Counter

    print("  kind 分布:", dict(Counter(t["kind"] for t in tickets)))
    print("  service 分布:", dict(Counter(t["service"] for t in tickets)))
    print("  route 分布:", dict(Counter(t["route"] for t in tickets)))

    if args.dry_run:
        print("[backfill] --dry-run，不写入 Milvus。")
        return

    from agent.integrations import memory  # type: ignore

    col = memory._ensure_collection()
    if col is None:
        print("[backfill] Milvus 不可用（_ensure_collection() 返回 None），中止写入。")
        return

    existing_fps: set[str] = set()
    try:
        rows = col.query(expr="id >= 0", output_fields=["fingerprint"], limit=10000)
        existing_fps = {r.get("fingerprint", "") for r in rows}
    except Exception as exc:  # noqa: BLE001
        print(f"[backfill] 查询已有 fingerprint 失败（{exc}），跳过幂等去重检查。")

    inserted = 0
    skipped = 0
    for ticket in tickets:
        report = _build_report(ticket)
        fp = report["meta"]["fingerprint"]
        if fp in existing_fps:
            skipped += 1
            continue
        memory.store_ticket(report)
        existing_fps.add(fp)
        inserted += 1

    print(f"[backfill] 写入完成: 新插入 {inserted} 条，跳过（fingerprint 已存在）{skipped} 条。")

    try:
        col.load()
        print(f"[backfill] 集合当前总条数: {col.num_entities}")
    except Exception as exc:  # noqa: BLE001
        print(f"[backfill] 写入后查询总条数失败（{exc}），但插入本身已完成。")


if __name__ == "__main__":
    main()
