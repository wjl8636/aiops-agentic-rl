"""往活环境注入告警描述的故障，跑完诊断 Agent 后复位。

真实模式采集轨迹时，如果活环境里没有真正复现告警描述的故障（flagd 开关仍是默认 off，
或者 recommendation 容器本来就没有泄漏），诊断 Agent 只能在"当前其实什么都没发生"的环境里
另找根因，结果步数偏长、且最终诊断和 ground_truth 不匹配。这个模块从 alert 的 scenario_hint /
service / annotations 里推断出该用哪种真实机制（数据生成阶段写的标注，真实告警不会有；喂给
模型的 prompt 里已经剔除了 scenario_hint 字段，这里只在 orchestration 层内部用来决定注入哪个
故障），在跑 Agent 前把故障真的触发，跑完再全部复位，保证每条告警互不污染。

覆盖的场景/机制（均为实测验证过的真实机制，不是伪造观测数据）：
  - dependency 依赖型、flagd_combination 组合型：scenario_hint 里显式标注 flag=/flags=，
    直接翻 flagd 开关（productCatalogFailure / cartFailure / kafkaQueueProblems 等实测有效；
    adFailure / recommendationCacheFailure 等因为对应服务的 flagd 客户端在这个 live 环境里
    不响应，flag 本身照样打开，只是不产生可观测效果——见下方各常量旁的说明）。
  - resource_cpu 参数化扩增（[参数化扩增/resource_cpu] 标记，服务固定是 ad）：翻
    adHighCpu 开关。这是 otel-demo 官方设计的机制，flagd 自身正确把它置为 on（配置文件、
    OFREP 都验证过），但实测 ad 服务自己的 OpenFeature/gRPC 客户端从未反映这个状态
    变化——用 `docker exec ad sh -c 'cat /proc/1/task/*/comm'` 直接枚举容器内线程名，
    确认 CPULoad 的高负载线程从未被拉起，adFailure/adManualGc 交叉测试、多次重启都是
    同样结果。这是 ad 服务一侧的 bug，不是这里的设计缺陷，如实保留机制但标注不可观测。
  - queue_backlog 参数化扩增（[参数化扩增/queue_backlog] 标记，服务固定是 kafka）：翻
    kafkaQueueProblems 开关。实测有效：checkout（生产端，Go）打开后日志出现
    "overloading queue"/"Done with #100 messages"；fraud-detection（消费端，Kotlin）
    打开后日志出现 31+ 次 "sleeping 1 second"，复位后恢复正常。
  - deploy_regression 参数化扩增（[参数化扩增/deploy_regression] 标记，服务固定是
    recommendation）：真的把 recommendation 容器换成 AIops-agent/scripts/fixtures/
    recommendation 里那个会持续内存泄漏的 fixture 镜像（不是靠 flagd——官方自带的
    recommendationCacheFailure 开关实测跟 adHighCpu 一样，flagd 状态正确但 recommendation
    自己的 Python 客户端不响应，40+ 次请求、重启后都没有任何 cache hit/miss 日志）。实测
    （docker stats 8 组采样、每组间隔 15 秒）内存从 15.62MiB 单调爬升到 32.34MiB，复位后
    换回原容器、内存回落到 ~40MiB 附近。
  - pure_code_fix_ranking（[参数化扩增/pure_code_fix_ranking] 标记，服务固定是
    recommendation）：为填补"纯 code_fix、不牵涉线上止血"这一诊断标签的空白而新增。复用
    AIops-agent 自己原有设计（scripts/fixtures/recommendation-ranking/，对应
    AIops-agent/eval/expected.json 的 s10_ranking 场景）——把 recommendation 容器换成这个
    fixture 镜像（`docker build -t recommendation-ranking-fixture:test
    scripts/fixtures/recommendation-ranking/`），`ranking.py::rank_by_score()` 里
    `sorted(..., key=lambda pair: pair[1])` 排序方向写反（按分数从低到高，本该从高到低），
    是纯业务逻辑 bug，跟 CPU/内存/依赖/配置等基础设施信号完全无关，任何重启/扩容/回滚都
    不会改变排序结果——只能改代码。跟内存泄漏 fixture 不同，这个 bug 不靠 docker stats
    观测，需要真的从 host 侧 curl 该服务的 `/recommend` 端点看返回内容，因此这里额外把
    容器的 8080 端口发布到固定 host 端口 18080（`RECOMMENDATION_RANKING_HOST_PORT`）。
    实测验证：`curl http://localhost:18080/recommend` 真实返回
    `["PRODUCT-19","PRODUCT-18","PRODUCT-17","PRODUCT-16","PRODUCT-15"]`（应为
    `PRODUCT-2..6`，热度最高的排在前面），复位后换回原容器即恢复正常。
    另一个候选 fixture（recommendation-dedupe，对应 s11_dedupe）经实测排除：
    `get_recommendations()` 默认 `max_results=5` 会把 `dedupe.dedupe_ids()`
    产出的重复 id（PRODUCT-3/PRODUCT-7）截断在返回列表之外——bug 在源码里真实存在，
    但默认 HTTP 请求路径下不可观测，不满足"真实机制、不伪造观测数据"的要求，如实弃用。
  - pure_code_fix_dedupe（[参数化扩增/pure_code_fix_dedupe] 标记，服务固定是
    recommendation）：2026-09 补充，把上一条"排除"的 recommendation-dedupe 重新捡回来——
    不是重新发明机制，是发现"不可观测"这个结论只针对*默认 20 SKU 目录*成立。把
    `scripts/fixtures/recommendation-dedupe/recommendation_server.py` 里的 `CATALOG` 从
    20 个 SKU 缩到 4 个（没有改 `dedupe.py` 里的 bug 本身一个字），重复 id 在 combined 列表
    里的位置就从第 18+ 位提前到 max_results=5 的窗口内。实测（本地起容器 + curl 复测 3 次）：
    `curl http://localhost:18081/recommend` 稳定返回
    `{"recommendations":["PRODUCT-2","PRODUCT-3","PRODUCT-0","PRODUCT-3","PRODUCT-7"]}`——
    PRODUCT-3 每次都出现两次，确定性复现，不依赖任何并发时序。
    另外尝试过、但放弃的候选：recommendation-race-stats（对应 s12_race）——`stats.py::
    record_hit()` 的读-改-写确实没加锁，是真 bug，但只有在 `sys.setswitchinterval(1e-6)`
    这种人为调紧 GIL 切换间隔、且单线程内密集连续调用几百次的场景下才会稳定丢计数（它自己
    的 `test_stats.py` 就是这么测的）。改成通过真实 HTTP 服务器 + 外部并发 curl 复现时
    （每个请求只触发一次 `record_hit()`，大部分线程时间花在 socket I/O 而不是那两行临界区），
    即使把 `sys.setswitchinterval` 调到跟测试一样低、外部发几千个并发请求，实测复现率也只有
    约 1/4800（3 次 100~1000 并发的批次里只有 1 次批次丢了 1 次计数）——远达不到"干净、
    可靠复现"的门槛，如实弃用，不勉强凑数。
  - pure_info_only（[参数化扩增/pure_info_only] 标记）：为填补"info_only"标签的空白而
    新增,不注入任何故障——这类场景本身的"真实机制"就是"活环境此刻确实健康、告警本身
    该被判定为不需要动作"，注入等价于什么都不做，如实原样跳过（跟历史工单里判定为
    "陈旧告警"的分支同构）。
  - historical_reverse_derivation 历史工单场景：scenario_hint 只有 source_ticket=，没有
    flag=/明确类别标记，靠 service + annotations 里的关键词兜底判断该走上面哪条真实机制
    （证书轮换类 config 场景、checkout/cart 的内存泄漏、frontend/payment 的 CPU 场景目前
    没有任何能在几分钟内真实触发的机制，如实跳过，不伪造）。
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

_FLAG_RE = re.compile(r"\bflags?=([\w+]+)")


def parse_flags(scenario_hint: str) -> list[str]:
    """从 scenario_hint 里解析 flag 名。支持单个（flag=x）和组合（flags=a+b）两种写法。"""
    m = _FLAG_RE.search(scenario_hint or "")
    if not m:
        return []
    return [name for name in m.group(1).split("+") if name]


def discover_flagd_config() -> Optional[Path]:
    """从正在跑的 flagd 容器反查它挂载的配置文件路径，不猜测/硬编码哪个 checkout。"""
    try:
        out = subprocess.run(
            ["docker", "inspect", "flagd", "--format", "{{json .Mounts}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for mount in json.loads(out or "[]"):
        if mount.get("Destination") == "/etc/flagd":
            candidate = Path(mount["Source"]) / "demo.flagd.json"
            if candidate.exists():
                return candidate
    return None


def _load(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _save(config_path: Path, cfg: dict[str, Any]) -> None:
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _most_severe_variant(flag_def: dict[str, Any]) -> Optional[str]:
    """挑这个 flag 里"故障生效"的那个 variant。

    大多数 flag 是简单的 on/off；但也有按百分比（paymentFailure：100%/90%/.../off）或
    按时长（imageSlowLoad：10sec/5sec/off）分级的——没有字面意义上的 "on"，这时退化成
    "variants 里第一个不是 off 的键"（JSON 里这些 flag 定义本身就是按严重度从高到低排的，
    第一个即最严重）。
    """
    variants = flag_def.get("variants", {})
    if "on" in variants:
        return "on"
    for name in variants:
        if name != "off":
            return name
    return None


def activate_flags(config_path: Path, flag_names: list[str]) -> list[str]:
    """把 flag_names 里每个 flag 打到"故障生效"的 variant。

    返回实际改动了的 flag 名；配置里根本不存在、或只有 off 一个 variant 的名字直接跳过，
    不报错——数据集参数化扩增出的部分 flag 名可能是旧版本 otel-demo 才有的，版本升级后
    已不存在。
    """
    cfg = _load(config_path)
    flags = cfg.get("flags", {})
    touched = []
    for name in flag_names:
        flag_def = flags.get(name)
        if flag_def is None:
            continue
        variant = _most_severe_variant(flag_def)
        if variant is not None:
            flag_def["defaultVariant"] = variant
            touched.append(name)
    if touched:
        _save(config_path, cfg)
    return touched


def reset_all_flags(config_path: Path) -> None:
    """把配置里所有 flag 的 defaultVariant 复位成 off（跟 scripts/reset.sh 语义一致）。"""
    cfg = _load(config_path)
    for flag in cfg.get("flags", {}).values():
        if "off" in flag.get("variants", {}):
            flag["defaultVariant"] = "off"
    _save(config_path, cfg)


# 参数化扩增里没有 flag=/flags= 标注的三类场景，靠 scenario_hint 里固定的方括号标签识别
# （数据生成阶段写的标签，见 data/seeds/generate_seeds.py 的 generate_parametrized）。
_RESOURCE_CPU_TAG = "[参数化扩增/resource_cpu]"
_QUEUE_BACKLOG_TAG = "[参数化扩增/queue_backlog]"
_DEPLOY_REGRESSION_TAG = "[参数化扩增/deploy_regression]"
# 纯 code_fix / info_only 补充场景（见 generate_seeds.py::generate_pure_code_fix_and_info_only()）。
_PURE_CODE_FIX_RANKING_TAG = "[参数化扩增/pure_code_fix_ranking]"
_PURE_INFO_ONLY_TAG = "[参数化扩增/pure_info_only]"
# 2026-09 补充：第二个纯 code_fix fixture（见
# generate_seeds.py::generate_pure_code_fix_and_info_only_v2()）——复用
# scripts/fixtures/recommendation-dedupe/（对应 s11_dedupe），不是新发明的机制。
_PURE_CODE_FIX_DEDUPE_TAG = "[参数化扩增/pure_code_fix_dedupe]"
# 2026-09-06 修复：`generate_seeds.py::generate_historical_reverse_derivation()`
# （见 data/seeds/generate_seeds.py:1220）实际写的是
# `f"[历史工单反演/{source}] ..."`，`source` 取值只有 `milvus`（真实 Milvus 检索）或
# `static`（检索池不足时的静态兜底）两种，**从未**写过字面 `stub`——用
# `grep -roh "历史工单反演/[a-zA-Z]*" data/clean/split/train/*.json` 核对过全量种子，
# 只有 `/milvus`（36 条）和 `/static`（8 条），0 条 `/stub`。之前这里精确匹配
# `"[历史工单反演/stub]"` 是死代码，导致 `inject_alert_fault()` 对**所有**
# historical_reverse_derivation 种子都直接跳过真实注入（不只是本次新增的种子——
# 老的 29 条种子里 9 条 historical 种子同样受影响，`_annotation_report.md` 的
# "系统性注意事项 1" 已经记录了这个后果，但没有诊断出是这个 tag 拼写不匹配的 bug）。
# 改成前缀匹配，覆盖 milvus/static（以及以防将来真的出现 stub）三种来源。
_HISTORICAL_TAG_PREFIX = "[历史工单反演/"

# resource_cpu 固定打这个 flag（服务固定是 ad）。实测：flagd 本身正确置为 on，但 ad 服务
# 的客户端不响应，本环境里不产生真实可观测效果——见模块顶部说明。
RESOURCE_CPU_FLAG = "adHighCpu"

# queue_backlog 固定打这个 flag（服务固定是 kafka）。实测有效（checkout 生产端 + fraud-
# detection 消费端都有真实日志变化）。
QUEUE_BACKLOG_FLAG = "kafkaQueueProblems"

# recommendation 内存泄漏 fixture：docker image 名字、原容器名、备份容器名、所在网络。
RECOMMENDATION_LEAK_IMAGE = "recommendation-leak-fixture:test"
RECOMMENDATION_CONTAINER = "recommendation"
RECOMMENDATION_BACKUP_CONTAINER = "recommendation-original-backup"
RECOMMENDATION_NETWORK = "opentelemetry-demo"

# recommendation 排序反向 fixture（纯 code_fix，见 scripts/fixtures/recommendation-ranking/）：
# 复用同一套「停原容器→改名备份→起同名新容器」的换血逻辑（_start_recommendation_fixture），
# 但这个 bug 是应用层逻辑（返回顺序反了），docker stats 看不出异常，必须真的从 host 侧
# curl 服务本身的 /recommend 才能观测到，所以额外把容器端口发布到固定 host 端口。
RECOMMENDATION_RANKING_IMAGE = "recommendation-ranking-fixture:test"
RECOMMENDATION_RANKING_HOST_PORT = 18080

# recommendation 去重失效 fixture（纯 code_fix，见 scripts/fixtures/recommendation-dedupe/，
# 对应 AIops-agent 自带的 s11_dedupe 场景）：dedupe.py::dedupe_ids() 用 `.lower()` 算查找 key
# 但存的是原始大小写 key，导致查找 key 永远对不上存的 key——不只是大小写不同的重复，任何
# 重复（包括大小写完全相同的）都测不出来。这个 bug 本身早就在源码里真实存在，之前被判定
# "已验证测试环境里看不到症状"（见 known_unsupported_seeds.json）：默认 20 个 SKU 的目录下，
# 重复项排在 combined 列表的第 18+ 位，而 get_recommendations() 硬编码 max_results=5 会在
# 看到重复之前就把列表截断。这次没有改 dedupe.py 本身，只把
# recommendation-dedupe/recommendation_server.py 里的演示目录从 20 个 SKU 缩到 4 个
# （实测见下）——candidate 列表变短之后，同一个重复 id（PRODUCT-3）在默认 max_results=5
# 的窗口内就出现了两次，不需要动 bug 逻辑，纯粹是让本来就存在的 bug 变得可观测。
# 实测验证（本地起容器 + curl）：`curl http://localhost:18081/recommend` 连续多次请求
# 稳定返回 `{"recommendations": ["PRODUCT-2","PRODUCT-3","PRODUCT-0","PRODUCT-3","PRODUCT-7"]}`
# ——PRODUCT-3 出现两次，确定性复现（不像 recommendation-race-stats 那样依赖并发时序，
# 见模块 docstring 末尾"已尝试但放弃"的说明）。
RECOMMENDATION_DEDUPE_IMAGE = "recommendation-dedupe-fixture:test"
RECOMMENDATION_DEDUPE_HOST_PORT = 18081

# 2026-09 v9 数据增强（docs/数据增强方案.md §4）：A1/B2/B3/D 家族的 scenario_hint 用
# `[v9增强/...]` 标签显式声明「不注入任何 flagd 开关」——这些家族的真实机制分别是
#「活环境此刻确实健康（空观测/噪声/陈旧告警）」和「bug 已经在 GitHub 上游仓 master 上
# （workspace/recommendation 克隆里，与 flagd 无关）」。显式写成 no-op 分支而不是靠
# 落到函数末尾的隐式 return []，是为了：① 语义自文档化，hint 里的家族标签跟注入行为
# 一一对应；② 防止这些 hint 文本将来被无意改出 flag=/flags= 字样时误触发注入。
_V9_AUG_TAG_PREFIX = "[v9增强/"

# historical_reverse_derivation 没有显式类别标注，靠 service + 告警文案关键词兜底：
# 命中"证书"→ config 类场景，目前没有能在几分钟内真实触发的机制（会牵涉到真的操作
# 容器内的 TLS 证书/信任链配置），如实跳过；命中"内存"/"OOM"→ deploy_regression 类，
# 只有 recommendation 有现成的泄漏 fixture，checkout/cart 没有对应 fixture，如实跳过；
# 命中 CPU 关键词→ resource(CPU) 类，只有 ad 有对应 flag（且已知不可观测），
# frontend/payment 没有对应机制，如实跳过；命中队列/消费关键词→ resource(queue) 类，
# 统一走 kafkaQueueProblems；命中超时/线程池/级联关键词→ dependency 类，按 service 查表。
_HIST_CERT_KEYWORDS = ("证书",)
_HIST_LEAK_KEYWORDS = ("内存", "OOM")
_HIST_CPU_KEYWORDS = ("CPU",)
_HIST_QUEUE_KEYWORDS = ("消费", "lag", "堆积", "consumer")
_HIST_TIMEOUT_KEYWORDS = ("超时", "线程池", "级联")

_HIST_CPU_FLAGS = {"ad": RESOURCE_CPU_FLAG}
_HIST_DEPENDENCY_FLAGS = {
    "product-catalog": "productCatalogFailure",  # 实测有效（Go）
    "cart": "cartFailure",  # 实测有效（C#：EmptyCart 转去连 badhost:1234，真实报错）
    # recommendation 依赖型故障官方对应 recommendationCacheFailure，但跟 adHighCpu 一样
    # 已知不可观测——保留 flag 名，环境修复后可能自动生效。
    "recommendation": "recommendationCacheFailure",
}


def _start_recommendation_fixture(
    image: str,
    env: dict[str, str],
    *,
    extra_run_args: Optional[list[str]] = None,
    publish_port: Optional[int] = None,
) -> list[str]:
    """通用的 recommendation 容器换血逻辑：停原容器→改名备份→起同名新容器（跑 image）。

    被 start_recommendation_leak_fixture / start_recommendation_ranking_fixture 共享——
    换血/还原的操作序列跟具体是哪个 fixture 镜像无关，只有 image 名字、env、是否需要发布
    端口不同。`publish_port` 非 None 时把容器的 8080 端口发布到该固定 host 端口（应用层
    bug 需要真的从 host curl 才能观测，跟内存泄漏那种 docker stats 就能看到的场景不同）。

    镜像不存在、或已经存在同名备份容器（说明某个 fixture 已经在跑）时原样跳过，不做任何
    容器改动——沿用原 start_recommendation_leak_fixture 的容错语义。
    """
    check = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        timeout=10,
    )
    if check.returncode != 0:
        return []
    backup_exists = subprocess.run(
        ["docker", "inspect", RECOMMENDATION_BACKUP_CONTAINER],
        capture_output=True,
        timeout=10,
    )
    if backup_exists.returncode == 0:
        return []
    try:
        subprocess.run(
            ["docker", "stop", RECOMMENDATION_CONTAINER],
            capture_output=True,
            timeout=20,
            check=True,
        )
        subprocess.run(
            ["docker", "rename", RECOMMENDATION_CONTAINER, RECOMMENDATION_BACKUP_CONTAINER],
            capture_output=True,
            timeout=10,
            check=True,
        )
        env_args = []
        for k, v in env.items():
            env_args += ["-e", f"{k}={v}"]
        publish_args = ["-p", f"{publish_port}:8080"] if publish_port else []
        subprocess.run(
            [
                "docker", "run", "-d", "--name", RECOMMENDATION_CONTAINER,
                "--network", RECOMMENDATION_NETWORK,
                "--network-alias", RECOMMENDATION_CONTAINER,
                *(extra_run_args or []),
                *publish_args,
                *env_args,
                image,
            ],
            capture_output=True,
            timeout=20,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [RECOMMENDATION_CONTAINER]


def start_recommendation_leak_fixture() -> list[str]:
    """把活的 recommendation 容器换成真的会内存泄漏的 fixture 镜像。

    fixture 来自 scripts/fixtures/recommendation（LOAD_RPS 驱动的后台线程往一个
    永不清空的 list 里持续 append），需要先在该目录下 `docker build -t
    recommendation-leak-fixture:test .` 构建好镜像——镜像不存在时原样跳过，不做任何容器改动。

    原容器改名保留为 RECOMMENDATION_BACKUP_CONTAINER（不删除），供 stop_recommendation_leak_
    fixture 还原；已经存在同名备份（说明泄漏 fixture 已经在跑）时直接跳过，避免误伤真实备份。
    """
    return _start_recommendation_fixture(
        RECOMMENDATION_LEAK_IMAGE,
        {"PORT": "9001", "LOAD_RPS": "400"},
        extra_run_args=["--memory=1073741824", "--memory-swap=1073741824"],
    )


def start_recommendation_ranking_fixture() -> list[str]:
    """把活的 recommendation 容器换成排序方向写反的 fixture 镜像（纯 code_fix，无线上止血）。

    fixture 来自 scripts/fixtures/recommendation-ranking（`docker build -t
    recommendation-ranking-fixture:test scripts/fixtures/recommendation-ranking/`）——镜像
    不存在时原样跳过。跟内存泄漏 fixture 共用同一套换血/还原逻辑（stop_recommendation_
    leak_fixture 本身就是镜像无关的，可以直接复用做还原），区别只是这里额外把容器 8080
    端口发布到 RECOMMENDATION_RANKING_HOST_PORT，因为这个 bug 要靠 curl `/recommend`
    观测，不是 docker stats。
    """
    return _start_recommendation_fixture(
        RECOMMENDATION_RANKING_IMAGE,
        {"PORT": "8080", "LOAD_RPS": "0"},
        publish_port=RECOMMENDATION_RANKING_HOST_PORT,
    )


def start_recommendation_dedupe_fixture() -> list[str]:
    """把活的 recommendation 容器换成去重失效的 fixture 镜像（纯 code_fix，无线上止血）。

    fixture 来自 scripts/fixtures/recommendation-dedupe（`docker build -t
    recommendation-dedupe-fixture:test scripts/fixtures/recommendation-dedupe/`）——镜像
    不存在时原样跳过。跟 ranking fixture 一样，这个 bug 要靠 curl `/recommend` 观测，不是
    docker stats，所以把容器 8080 端口发布到 RECOMMENDATION_DEDUPE_HOST_PORT。
    """
    return _start_recommendation_fixture(
        RECOMMENDATION_DEDUPE_IMAGE,
        {"PORT": "8080", "LOAD_RPS": "0"},
        publish_port=RECOMMENDATION_DEDUPE_HOST_PORT,
    )


def stop_recommendation_leak_fixture() -> None:
    """把 recommendation 换回来：删掉 fixture 容器，把备份改回原名字再启动。

    没有备份容器（说明没真的注入过，或者已经复位过）时什么都不做——可以安全地在每次
    跑完 Agent 后无条件调用。
    """
    backup_exists = subprocess.run(
        ["docker", "inspect", RECOMMENDATION_BACKUP_CONTAINER],
        capture_output=True,
        timeout=10,
    )
    if backup_exists.returncode != 0:
        return
    subprocess.run(["docker", "stop", RECOMMENDATION_CONTAINER], capture_output=True, timeout=20)
    subprocess.run(["docker", "rm", RECOMMENDATION_CONTAINER], capture_output=True, timeout=10)
    subprocess.run(
        ["docker", "rename", RECOMMENDATION_BACKUP_CONTAINER, RECOMMENDATION_CONTAINER],
        capture_output=True,
        timeout=10,
    )
    subprocess.run(["docker", "start", RECOMMENDATION_CONTAINER], capture_output=True, timeout=20)


def _activate_and_wait(config_path: Path, flag_names: list[str]) -> list[str]:
    touched = activate_flags(config_path, flag_names)
    if touched:
        time.sleep(1.5)  # flagd 热加载配置文件有小延迟，给它一点时间再跑 Agent
    return touched


def _inject_historical_fault(alert: dict[str, Any], config_path: Path, service: str) -> list[str]:
    """historical_reverse_derivation 场景：没有 flag= 标注，靠 service + 告警文案关键词兜底。"""
    annotations = alert.get("annotations", {}) or {}
    text = f"{annotations.get('summary', '')} {annotations.get('description', '')}"
    if any(k in text for k in _HIST_CERT_KEYWORDS):
        return []  # config 类：证书轮换/信任链配置，没有能真实触发的机制
    if any(k in text for k in _HIST_LEAK_KEYWORDS):
        if service == "recommendation":
            return start_recommendation_leak_fixture()
        return []  # checkout/cart 的内存泄漏历史工单：没有对应 fixture
    if any(k in text for k in _HIST_CPU_KEYWORDS):
        flag = _HIST_CPU_FLAGS.get(service)
        if flag is None:
            return []  # frontend/payment 的 CPU 历史工单：没有对应机制
        return _activate_and_wait(config_path, [flag])
    if any(k in text for k in _HIST_QUEUE_KEYWORDS):
        return _activate_and_wait(config_path, [QUEUE_BACKLOG_FLAG])
    if any(k in text for k in _HIST_TIMEOUT_KEYWORDS):
        flag = _HIST_DEPENDENCY_FLAGS.get(service)
        if flag is None:
            return []
        return _activate_and_wait(config_path, [flag])
    return []


def inject_alert_fault(alert: dict[str, Any], config_path: Optional[Path]) -> list[str]:
    """按 alert 的 scenario_hint 触发对应的真实故障；解析不出场景或没有活配置就原样跳过。

    优先级：flag=/flags= 显式标注（dependency、flagd_combination）> 参数化扩增的固定标签
    （resource_cpu/queue_backlog/deploy_regression）> 历史工单反演的关键词兜底。
    """
    if config_path is None:
        return []
    hint = alert.get("scenario_hint", "") or ""

    flag_names = parse_flags(hint)
    if flag_names:
        return _activate_and_wait(config_path, flag_names)

    if _RESOURCE_CPU_TAG in hint:
        return _activate_and_wait(config_path, [RESOURCE_CPU_FLAG])

    if _QUEUE_BACKLOG_TAG in hint:
        return _activate_and_wait(config_path, [QUEUE_BACKLOG_FLAG])

    if _DEPLOY_REGRESSION_TAG in hint:
        return start_recommendation_leak_fixture()

    if _PURE_CODE_FIX_RANKING_TAG in hint:
        return start_recommendation_ranking_fixture()

    if _PURE_CODE_FIX_DEDUPE_TAG in hint:
        return start_recommendation_dedupe_fixture()

    if _PURE_INFO_ONLY_TAG in hint:
        return []  # 故意不注入任何故障：真实机制就是「活环境此刻确实健康」

    if _V9_AUG_TAG_PREFIX in hint:
        # v9 增强：A1（空观测误报）/B2（info 级噪声）/B3（陈旧告警）/D（bug 在 GitHub master
        # 的 workspace 克隆里）都不需要 flagd 注入。B3 的「曾有故障、现已复位」由编排层
        # （collect_v9.py）在采集前手动短暂注入并复位，跑 Agent 期间保持 off。
        return []

    if _HISTORICAL_TAG_PREFIX in hint:
        service = alert.get("service") or alert.get("labels", {}).get("service", "")
        return _inject_historical_fault(alert, config_path, service)

    return []


def reset_all(config_path: Path) -> None:
    """把这个模块可能改动过的一切都复位：flagd 开关全部置 off + recommendation 容器换回来。

    给 collect_trajectories.py 的 finally 块统一调用；两部分复位互相独立，即使之前只触发过
    其中一种机制，另一种的复位调用也是安全的空操作。
    """
    reset_all_flags(config_path)
    stop_recommendation_leak_fixture()
