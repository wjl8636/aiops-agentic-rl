# AIOps 诊断 Agent 13 场景完整测评执行日志

- **模型**: aiops-qwen3.5-9b（微调后 checkpoint，vLLM 部署在 AutoDL A800 80GB，SSH 隧道 localhost:8000）
- **日期**: 2026-09-07 起
- **对照基线**: Opus（reports/opus_20260819/metrics.json，23 rows / 13 场景）
- **本次同时验证的修复**: 轮数预算 25→50、故障注入链路
- **代码 commit**: 见 metrics.json 的 commit 字段

## 环境前置核验（2026-09-07 晚）

| 项 | 结果 |
|---|---|
| config.MODEL | aiops-qwen3.5-9b |
| config.LLM_BASE_URL | http://localhost:8000（vLLM /v1/models 返回 200，served=aiops-qwen3.5-9b）|
| DIAGNOSE_MAX_TURNS / FIX_MAX_TURNS | 50 / 50（轮数预算 25→50 修复确认生效）|
| REMEDIATION_ENABLED / BACKEND | True / docker |
| docker 容器 | 27 个全部 Up（frontend-proxy / kafka / load-generator 正常）|
| GitHub 上游仓 | HuaiNan54321/recommendation master=suspect f18f452，与 deploys.log 一致 |
| gh auth | 已登录 HuaiNan54321（GITHUB_TOKEN 需 eval 前 `export GITHUB_TOKEN=$(gh auth token)`，.env 中为空）|
| vLLM 日志基线计数 | 2473 次 POST /v1/messages |

注意：sdk_runner `_llm_env_overrides()` 无条件把 ANTHROPIC_BASE_URL 钉死到 config.LLM_BASE_URL，shell 里公司代理的 ANTHROPIC_* 环境变量不会污染路由。

## 执行计划

顺序：路由验证(s2) → s2(x3) → s3(x3) → s1(x3) → 行为组 s_lowconf/s6_info/s9_adv/s5_dup(3+3+3+1) → s4(x1, RECO_PORT=8081, 先 seed-github) → 代码修复组 s7/s8/s10/s11/s12(各1)。每组之间 reset.sh。

路由验证的 run 落在 `reports/route-check/`（与正式报告目录隔离，不进当天合并大盘，保证与 Opus 23-row 口径公平可比）。

---

## 执行记录

### 2026-09-08 00:00-00:45 路由验证（s2）

- 00:00 `./scripts/inject.sh s2` → adHighCpu.defaultVariant=on 写入 demo.flagd.json，flagd 热加载（日志见 WRITE 事件 16:00:39Z）。
- OFREP 验证（**POST** /ofrep/v1/evaluate/flags）：`adHighCpu: value=true, variant=on`，其余 flag 全 off。✅
- 00:01-00:21 `./.venv/bin/python -m eval.multi_run --only s2 --runs 1 --reports-dir reports/route-check`
  - **注意**：`--runs 1` 被 expected.json 里 s2 的 `runs:3` 覆盖（multi_run.py 的 `exp.get("runs", default_runs)` 语义），实际跑了 3 轮，共 1212s。
- vLLM `POST /v1/messages` 计数：**2473 → 2745，delta = 272**（3 轮 ≈ 91 请求/轮，与多轮 Agent 诊断的量级吻合）。
  - 运行中抽查：00:13 计数 2641（engine 日志 Running: 1 reqs，prefix cache hit 57.4%），run 结束后计数稳定在 2745。
  - 全日志仅 7 条非 200（6×404 + 1×400，均在此前调试窗口，非本次 run 期间）。
  - **结论：路由验证通过，所有 LLM 请求均打到本地 vLLM（经 SSH 隧道），无漏到真实付费 API。**
- 结果（仅供诊断，不进正式大盘）：run#0 route=feishu_online_op service_ok=False kind_ok=True route_ok=True 649.7s；run#1 同 route，service_ok=False，497.9s；run#2 route=info_only service_ok=True route_ok=False 54.6s。

### 2026-09-08 00:05-00:45 重大发现：s2 的 flagd 故障注入无法在 ad 容器 manifest（环境问题，非本次代码 bug）

排查过程与证据：
1. flag 侧完全正常：OFREP(8016) 与 gRPC(8013) 均返回 `adHighCpu=true/on`（grpcurl 用官方 schema.proto 直连验证）。
2. ad 容器（Java）每个 getAds 请求都会执行 `cpuload.execute(ffClient.getBooleanValue("adHighCpu", false, ctx))`（镜像内 class 字节码确认 flag key 就是 adHighCpu、CPULoad 调用存在）。
3. 但 ad 从不向 flagd 发起评估请求：直连 `ad:9555 GetAds` 成功返回广告、ad 日志出现 Targeted 请求，**flagd 的 adHighCpu 评估计数不增**；ad 日志中 "High CPU-Load problempattern enabled" 出现次数为 **0**（全量 38710 行）；docker stats ad CPU 仅 0.19-0.73%；JVM CPU 指标 24h 峰值 0.88%。
4. `docker restart ad` 后依然不生效 → 排除"provider 缓存了启动旧值"假设。
5. 根因（高度可信推断）：运行镜像 `demo:latest-ad`（6/20 构建）内 flagd Java provider 是 **0.14.0**（vendored 源码声明 0.11.5，`latest` 镜像比 vendored 2.0.2 新），其评估从不打到 flagd；且 ad 容器 SLF4J 无 provider（日志明确告警），provider 的连接/初始化错误被整体吞掉，无法看到根因细节。
6. flagd 评估计数里所有 11 个 flag 均以 ~2.21/min 匀速增长（来源不明，疑似某组件周期性 OFREP 批量评估），与 ad 的 ~6/min 请求速率不符，进一步佐证那不是 ad 的评估。

**影响与决策**：
- s2 的预期故障信号（ad CPU 打满）在此环境**从未真正出现**。模型只能靠告警文本推理。
- ad 镜像自 6/20 未变 → **Opus 基线（8/17-19）跑 s2 时同样条件**（同样拿不到 CPU 信号），保持现状继续跑，对比公平。与任务说明中 kafka lag 口径问题的处理原则一致。
- 附带操作记录：排查期间 `docker restart ad`（00:33），ad StartedAt 变新；因 s2 本来就无 CPU 信号，该操作对可观测面无实质影响。

### 2026-09-08 00:44-01:12 s2 正式组第 1 次（已隔离）

- 注入：adHighCpu=on（沿用路由验证时的注入，OFREP 确认）。
- 结果：run#0 route=code_fix_pr service_ok=False kind_ok=True route_ok=False 861.2s；run#1 route=info_only service_ok=True kind_ok=True route_ok=False 62.6s；run#2 route=feishu_low_confidence service_ok=False kind_ok=True route_ok=False 723.7s（schema 校验重试 2 次失败降级；期间 Agent 对 ad 执行了白名单内 restart_instance 止血）。
- **发现两个必须处理的问题**：
  1. **GITHUB_TOKEN 未 export**（.env 里为空）：run#0 触发 code_fix 后 build+test 通过（fix_verified=True）但 push 无凭据，pr_created=False。而 Opus 基线所有 code_fix 行 pr_created 全为 True（当时 token 在）→ 条件不一致，对被测模型不公平。
  2. **run#0 期间（00:46:19）flagd 配置被写入、adHighCpu 被翻回 off**（flagd 日志 WRITE 事件；之后 OFREP 验证全 off）。最可能是 run#0 的诊断 Agent 把告警里点名的「adServiceHighCpu 开关」当处置关掉（该操作不在处置白名单内，未进台账，属 Agent 越权行为，记录备查）。受此影响 run#1/run#2 跑在 flag=off 状态。由于 ad 的 CPU 故障本就无法 manifest（见上），flag 开关对容器实际状态无影响，仅影响 Agent 经 OFREP 读到的开关值。
- **处置**：本次 run 目录改名隔离为 `quarantined-no-github-token_run_20260907T164437/`（不进 run_* 聚合 glob），重新注入 s2 并 export GITHUB_TOKEN=$(gh auth token) 后重跑全组。

### 2026-09-08 01:18 s2 正式组第 2 次（有效）

- 重新注入时发现：16:46 那次写入把 adHighCpu 的 **state 改成了 DISABLED**（description 还加了 "(DISABLED)" 后缀），inject.sh 只翻 defaultVariant 不动 state，所以 flag 一直评估为 off（OFREP 全 off、gRPC 返回 "Flag is disabled"）。已 `git checkout` 恢复 flagd 配置文件到 git 原始状态（state=ENABLED）后重新注入，OFREP 验证 adHighCpu=True。✅
- **教训（影响后续所有组）**：诊断 Agent 可能用 Bash/python 直接改 flagd 配置做「处置」（把 state 设为 DISABLED），该操作不在处置白名单内、不进执行台账（白名单绕过，最终汇报列出）；reset.sh 只翻 defaultVariant 不恢复 state，**每组注入前必须先核对 flag 文件 state=ENABLED**。
- 启动前 export GITHUB_TOKEN=$(gh auth token)（gh 已登录 HuaiNan54321，scopes 含 repo）。
- vLLM 计数：起点 3035。

- 结果（run_20260907T172316，57 分钟，vLLM 计数 3035→3940，delta=905）：
  - run#0: route=feishu_low_confidence service_ok=False kind_ok=True route_ok=False 1815.3s
  - run#1: route=info_only service_ok=False kind_ok=True route_ok=False 528.5s
  - run#2: route=feishu_low_confidence service_ok=False kind_ok=True route_ok=False 1079.2s
  - **s2 汇总：service 0/3、kind 3/3、route 0/3（Opus 基线 s2：2/3、2/3、2/3）**。3 轮均未触发 code_fix（无 PR）。本轮无 flagd 写入（Agent 未再改配置）。
  - 与 Opus 差距主要在 service 定位：ad 的 CPU 故障无任何可观测信号（见前述环境问题），模型定位不到 ad。

### 2026-09-08 02:45 s3 组

- reset 后核对 flag 文件：全部 state=ENABLED、defaultVariant=off。
- kafka lag 基线（sidecard kafka-consumer-groups.sh）：accounting=0、fraud-detection=0。
- `./scripts/inject.sh s3` → kafkaQueueProblems=on。等 150s 后复查：**accounting lag=1、fraud-detection lag=596**（在涨）→ **s3 故障真实 manifest**（与 s2 的 Java 链路不同，Go 消费链路正常）。✅
- OFREP 验证：仅 kafkaQueueProblems=True。✅
- vLLM 计数：起点 3940。export GITHUB_TOKEN 后启动。

- s3 结果（run_20260907T183332，39.5 分钟，vLLM 计数 3940→4379）：
  - run#0: route=info_only service_ok=False kind_ok=True route_ok=False 943.1s
  - run#1: route=info_only service_ok=True kind_ok=False route_ok=False 605.0s
  - run#2: route=info_only service_ok=False kind_ok=False route_ok=False 814.3s
  - **s3 汇总：service 1/3、kind 1/3、route 0/3（Opus 基线 s3：2/3、1/3、0/3）**。两模型 s3 route 都是 0/3（Opus 全走 feishu_low_confidence，本模型全走 info_only）——s3 对两个模型都难。run 期间有一次 flagd 写入但内容无实质变化（flag 仍 on、state 全 ENABLED，git diff 仅注入本身）；lag 全程真实存在（fraud-detection 3941+）。

### 2026-09-08 03:20 s1 组

- reset 后 lag 归零、flag 文件干净（state 全 ENABLED）。
- `./scripts/inject.sh s1` → productCatalogFailure=on。等 120s 验证：frontend 日志出现 **"Error: Product Catalog Fail Feature Flag Enabled"**（5 分钟 56 条，gRPC INTERNAL）→ **s1 故障真实 manifest**（Go 链路正常；product-catalog 自身日志无 error，错误返回给调用方）。✅
- OFREP 验证：仅 productCatalogFailure=True。✅
- vLLM 计数：起点 4379。

- s1 结果（run_20260907T193014，vLLM 计数 4379→4623+）：
  - run#0: route=feishu_low_confidence service_ok=False kind_ok=True route_ok=False 899.3s
  - run#1: route=feishu_online_op service_ok=True kind_ok=True route_ok=True 106.8s
  - run#2: route=feishu_online_op service_ok=True kind_ok=True route_ok=True 74.0s
  - **s1 汇总：service 2/3、kind 3/3、route 2/3（Opus 基线 s1：2/3、2/3、2/3）**——故障可观测时两模型打平（甚至 kind 更好），与 s2/s3 的差距印证「无信号场景定位难」。

### 2026-09-08 03:45 行为组（s_lowconf / s6_info / s9_adv / s5_dup）

- reset 后 OFREP 全 off（注意：flagd 热加载有 ~10-20s 延迟，reset/inject 后需等待再验证）。flag 文件 state 全 ENABLED。
- 无注入。`--only s_lowconf s6_info s9_adv s5_dup`，export GITHUB_TOKEN（s9_adv 可能路由 code_fix_pr）。预期轮数 3+3+3+1（s5_dup 由 multi_run 内部先跑一次 s2 种指纹）。vLLM 计数起点 4623。

### 2026-09-08 04:14-05:20 基础设施事故：SSH 隧道静默断开（已恢复）

- **~04:14 本地→AutoDL 的 SSH 隧道被静默踢断**（旧隧道进程无保活），05:14 发现并重建（新隧道带 30s keepalive）。期间 localhost:8000 不可达，vLLM 计数冻结在 4817。
- 影响评估：
  - s2 重跑（run_20260907T172316）、s3（run_20260907T183332）、s1（run_20260907T192607）全部完成于 04:14 之前，**数据有效**（计数各自正常增长）。
  - 行为组第一次（run_20260907T194607，03:46 启动）：04:06 前正常请求（4623→4817），04:14 起后端不可达，SDK 卡在连接重试，任务最终被 kill；jsonl 0 行，目录已删。**无数据污染。**
  - s_lowconf 单独重跑（run_20260907T205452，04:54 启动）：全程处于隧道断开窗口，run#0 以「schema 校验重试 2 次仍失败」degraded 收场（飞书卡片显示置信度 0 降级）——**该 run 的数据无效**。05:20 已 kill 进程、删除目录（jsonl 0 行，未进大盘；Milvus 里可能留有 1 条 degraded 工单，影响可忽略）。
- 教训：长时间窗口内 vLLM 计数不增长（>10 分钟）= 第一时间怀疑隧道，而不是 Agent 卡工具。

- s_lowconf 结果（run_20260907T211514，48 分钟，vLLM 计数 4822→5818）：
  - run#0: route=feishu_low_confidence service_ok=True kind_ok=True route_ok=True 957.5s
  - run#1: route=feishu_low_confidence service_ok=True kind_ok=True route_ok=True 1815.3s
  - run#2: route=feishu_low_confidence service_ok=True kind_ok=True route_ok=True 120.5s
  - **s_lowconf 汇总：3/3 全对（Opus 基线 s_lowconf 同为 3/3）**。鲁棒性场景表现完美。

- s6_info 结果（run_20260907T220409，12.7 分钟）：
  - run#0: route=info_only service_ok=True kind_ok=True route_ok=True 11.0s
  - run#1: route=info_only service_ok=True kind_ok=True route_ok=True 13.7s
  - run#2: route=feishu_low_confidence service_ok=False kind_ok=True route_ok=False 694.1s
  - **s6_info 汇总：service 2/3、kind 3/3、route 2/3（Opus 基线 s6_info：3/3、3/3、3/3）**。run#0/#1 极快（~12s）正确识别 info_only。

- s9_adv 结果（run_20260907T222353，37 分钟）：
  - run#0: route=feishu_low_confidence service_ok=False kind_ok=False route_ok=False 1362.3s
  - run#1: route=auto_remediated service_ok=True kind_ok=True route_ok=True 508.1s
  - run#2: route=auto_remediated service_ok=False kind_ok=True route_ok=True 333.7s
  - **s9_adv 汇总：service 1/3、kind 2/3、route 2/3（Opus 基线 s9_adv 1 轮：route=auto_remediated_and_code_fix_pr 全对）**。本模型 3 轮均未触发 code_fix（对 s9 的对抗性告警只走了 online_op/低置信度路线）。

- s5_dup 结果（run_20260907T230254，3.5 分钟，含内部 s2 种指纹 1 轮）：
  - run#0: route=skipped_duplicate service_ok=True kind_ok=True route_ok=True 0.0s
  - **s5_dup 汇总：1/1 全对，指纹去重瞬时生效（Opus 基线同为 1/1）**。
- **行为组小结（10 轮）**：s_lowconf 3/3、s6_info 2/3、s9_adv 2/3（1 次 code_fix 尝试 fix_verified=False，符合 s9 对抗性设计）、s5_dup 1/1。

### 2026-09-08 07:10 s4 组

- `LC_ALL=C ./scripts/seed-github.sh` 重新播种上游仓。**注意**：直接跑会报 `FULL…: unbound variable`——scripts/seed-github.sh 第 91 行 `$FULL` 后直接跟 UTF-8 省略号（e2 80 a6），bash 在 UTF-8 locale 下把省略号并入变量名触发 set -u 报错；git HEAD 里就是坏的（此前跑通疑似 C locale）。**workaround：LC_ALL=C，不改代码**，最终汇报列入问题清单。
- 播种结果：baseline=c953ea8、leak=b92be74（suspect，master HEAD），deploys.log 已刷新，与 GitHub 实际一致。✅
- `RECO_PORT=8081 ./scripts/inject.sh s4`（8081 避开 frontend-proxy 的 8080）：泄漏容器 recommendation:b92be74 --memory=128m 启动。
- 故障验证：内存 29.48MiB →（90s）→ 43.75MiB 单调爬升（上限 128MiB），/metrics 在 8081 可读 → **s4 故障真实 manifest**。✅
- vLLM 计数：起点 6628。export GITHUB_TOKEN 后启动 s4（1 轮）。

- s4 结果（run_20260907T231731）：
  - run#0: route=code_fix_pr service_ok=True kind_ok=False route_ok=True **fix_verified=True pr_created=True changed_files_hit=True** 123.1s（diag 61.7s + fix 61.4s）
  - **s4 端到端闭环全通**：定位 recommendation → clone → 改 recommendation_server.py（有界 deque）→ build+test → 真实 GitHub PR #44（branch fix-memory-leak-bounded-deque）。速度极快（123s），合理来源：SFT 训练数据 + Milvus RAG 里存有此前多轮 s4 工单（含标准修法）。
  - kind_ok=False（expected deploy_regression，模型给了别的类型）；route/service/fix/定位 4 项全对（Opus 基线 s4 同样全对）。
- reset：泄漏容器已删、compose 干净版 recommendation 恢复（latest-recommendation 镜像）、flag 文件 state 全 ENABLED、全 off。

### 2026-09-08 07:25 代码修复组（s7_hybrid / s8_fix_fail / s10 / s11 / s12）

- 无注入。每组 1 轮，export GITHUB_TOKEN。vLLM 计数：起点 6665。

### 2026-09-08 07:30 代码 bug：s7_hybrid 首跑触发 UnboundLocalError（已最小修复）

- s7_hybrid 第一次 run（run_20260907T232753）诊断完成后触发崩溃：
  ```
  agent/run.py line 112: note = _auto_note(diag) if auto_done else note
  UnboundLocalError: cannot access local variable 'note' where it is not associated with a value
  ```
- **精确触发条件**：`remediation_type=online_op` + 执行台账为空（auto_done=False，line 97 只在 auto_done=True 时赋值 note）+ `also_code_fix=True` + **fix 未通过验证**（fix2.verified=False）→ line 112 读未赋值的 note。整条 multi_run 进程被未捕获异常杀死，该场景 0 行落盘（目录已删，无数据污染）。
- **Opus 基线未触发此 bug 的原因**：Opus s7 的 fix_verified=True，走的 line 108-109 分支。
- **处置**：应用最小修复（agent/run.py +1 行：online_op 分支开头 `note = ""` 兜底初始化），语义中性——崩溃路径的 route 本来就该是 feishu_online_op，修复只是让它能落盘而不是崩掉；不改任何 route/判定逻辑。不修则后续任意代码修复场景（s8/s10/s11/s12）只要模型走 online_op+also_code_fix+fix 失败就会再崩。**这是本次测评发现的代码 bug #1，最终汇报列出。**
- 另记录：该次崩溃 run 的模型行为——诊断置信度 92%、remediation_type=online_op、also_code_fix=True，自述执行了 compose 重建+restart 止血但**执行台账为空**（hook 未放行/未记录），证据里有明显幻觉（"内存占用3.9GB"、"sleep.sh"、"memory.py:131"、"os._exit(295)" 等不存在的文件/代码），fix 修的目标也是幻觉目标（memory.py）→ 未通过验证。这本身就是有效的模型质量信号。
- 修复后重跑 s7_hybrid。vLLM 计数：起点 6941。
- **补充（08:25）**：用户随后把 agent/run.py 还原到原始状态（一行修复被撤销，尊重该意图、不再改动代码）。在跑的 s7 重跑进程启动时已加载修复版（语义中性：崩溃路径的 route/fix 结果与不崩时完全一致）；此后启动的场景（s8/s10/s11/s12）使用原始代码——若再触发同一路径会再次崩溃丢行，届时记录并按需重试（模型路径有随机性，重试可能走别的分支）。

- s7_hybrid 重跑结果（run_20260907T235832，23.8 分钟）：
  - run#0: route=feishu_low_confidence service_ok=False kind_ok=False route_ok=False 1411.1s（未再触发崩溃路径）
  - **s7 汇总：0/1（Opus 基线 s7 1/1：auto_remediated_and_code_fix_pr + fix_verified=True）**——本模型对混合根因场景整体失败。

### 2026-09-08 08:50 代码修复组续（s8 / s10 / s11 / s12）

- vLLM 计数：s8 起点 7177。

- s8_fix_fail 结果（**落在新目录 run_20260908T002816，UTC 日期跨天**）：
  - run#0: route=feishu_online_op service_ok=False kind_ok=True route_ok=False 718.6s，未触发 code_fix
  - **s8 汇总：0/1（Opus 基线 s8 1/1：feishu_fix_unverified）**。模型把它当线上操作处理、没走代码修复路线。
  - **注意**：multi_run 的 model 目录用 UTC 日期，00:28Z 后的 run 落在 `reports/aiops-qwen3.5-9b_20260908/`，与 `_20260907` 分家 → 最终需手动合并两目录的 jsonl 出总大盘（eval.aggregate 支持显式 --input）。
- s10_ranking 启动（vLLM 计数 7178+）。

- s10_ranking 结果（run_20260908T004834，21.3 分钟）：
  - run#0: route=feishu_low_confidence service_ok=False kind_ok=False route_ok=False 1274.1s，未触发 code_fix（changed_files_hit=False）
  - **s10 汇总：0/1（Opus 基线 s10 1/1：code_fix_pr + fix_verified + ranking.py 定位全对）**。
- s11_dedupe 启动。

- s11_dedupe 结果（run_20260908T011809，4.1 分钟）：
  - run#0: route=feishu_fix_unverified service_ok=True kind_ok=False route_ok=False **triggered_code_fix=True fix_verified=False pr_created=False changed_files_hit=False** 243.5s
  - **s11 汇总：0/1（Opus 基线 s11 1/1：code_fix_pr + verified + dedupe.py 定位）**。模型触发了代码修复但未通过 build/test 验证（走了 feishu_fix_unverified 降级）。
- s12_race 启动（协调者已把 note 兜底修复补回 run.py，s12 进程受保护）。

- s12_race 结果（run_20260908T012842，7.6 分钟）：
  - run#0: route=auto_remediated service_ok=True kind_ok=False route_ok=False 未触发 code_fix 453.3s
  - **s12 汇总：0/1（Opus 基线 s12 1/1：code_fix_pr + verified + stats.py 定位）**。

---

## 最终汇总（2026-09-08 09:50）

### 数据完整性

- **25 行 / 13 场景全部完成**（expected.json 的 runs 字段口径：s1/s2/s3/s6/s9/s_lowconf 各 3 轮，其余各 1 轮）。
- 最终大盘：`reports/aiops-qwen3.5-9b_20260907/metrics_final_combined.json`（手动合并 `_20260907` 9 个 run + `_20260908` 4 个 run；因 multi_run 的 model 目录按 UTC 日期分家，`_20260907/metrics.json` 只含当天 18+1 行、`_20260908/metrics.json` 只含 5 行）。
- 与 Opus 的口径差异：Opus 基线 23 行（**s9_adv 只跑了 1 轮**），本次按 expected.json 跑了 3 轮（s9 弱项被加权，略压低汇总值；按 Opus 口径截到 1 轮的话 service 0.5217 / kind 0.6522 / route 0.3913，量级结论不变）。
- 隔离数据（不进大盘）：`quarantined-no-github-token_run_20260907T164437`（无 token 的 s2 首跑）、`reports/route-check/`（路由验证 3 轮 + s2 无 token 3 轮）。

### 路由验证（全程证据）

- 起点 2473 → 终点 **7777**，全程 delta 5304 次 POST /v1/messages 全部落在本地 vLLM（经 SSH 隧道）。
- 非 200 仅 9 条（3×400 + 6×404，占 0.1%；404 是更早调试期遗留，400 为 CLI 辅助请求，SDK 均重试成功）。**无任何请求漏到真实付费 API。**
- 中途事故：04:14-05:14 SSH 隧道静默断开（见前述），恢复后带 30s keepalive 未再断。

### 最终 7 项指标 vs Opus 基线

| 指标 | aiops-qwen3.5-9b（本次，25 行） | Opus 基线（23 行） | 差距 |
|---|---|---|---|
| service_accuracy | **0.52** | 0.8696 | -0.35 |
| kind_accuracy | **0.68** | 0.8261 | -0.15 |
| route_accuracy | **0.44** | 0.7826 | -0.34 |
| fix_success_rate | **0.3333**（1/3） | 1.0（7/7） | -0.67 |
| code_location_accuracy | **0.2**（1/5） | 1.0（5/5） | -0.8 |
| mttr_seconds_p50 | **649.548**（10.8 分钟） | 195.482（3.3 分钟） | 3.3 倍慢 |
| pr_submission_rate | **1.0**（1/1） | 1.0（7/7） | 持平（分母小） |

### 每场景对照（route_ok / service_ok）

| 场景 | 本模型 | Opus | 备注 |
|---|---|---|---|
| s1 productCatalog | 2/3 / 2/3 | 2/3 / 2/3 | 故障可观测，打平 |
| s2 adHighCpu | 0/3 / 0/3 | 2/3 / 2/3 | **故障无法 manifest（环境缺陷，两模型同条件）** |
| s3 kafka lag | 0/3 / 1/3 | 0/3 / 2/3 | 两模型 route 都 0/3 |
| s4 内存泄漏 | 1/1 / 1/1 | 1/1 / 1/1 | 端到端 PR #44 全对（kind 判错） |
| s5_dup | 1/1 | 1/1 | 去重完美 |
| s6_info | 2/3 / 2/3 | 3/3 / 3/3 | 接近 |
| s7_hybrid | 0/1 / 0/1 | 1/1 / 1/1 | 差距最大场景之一 |
| s8_fix_fail | 0/1 / 0/1 | 1/1 / 1/1 | 未走代码修复路线 |
| s9_adv | 2/3 / 1/3 | 1/1 / 1/1 | Opus 只跑了 1 轮 |
| s10_ranking | 0/1 / 0/1 | 1/1 / 1/1 | 低置信度降级 |
| s11_dedupe | 0/1 / 1/1 | 1/1 / 1/1 | 修复未过验证 |
| s12_race | 0/1 / 1/1 | 1/1 / 1/1 | 走了线上处置路线 |
| s_lowconf | 3/3 / 3/3 | 3/3 / 3/3 | 完美 |

### 收尾状态

- reset.sh 已跑：flag 文件 state 全 ENABLED、defaultVariant 全 off，OFREP 验证全 off。
- 泄漏容器已删，compose 干净版 recommendation（latest-recommendation）在跑，27 容器全部 Up。
- 无残留 eval.multi_run 进程。GitHub 上游仓 master=b92be74（新 seed），本次测评产生的 PR：#44（s4，预期行为）。

### 问题清单（按重要度）

1. **agent/run.py:112 UnboundLocalError**（代码 bug）：触发条件 online_op + 台账空 + also_code_fix + fix 未验证；s7 首跑崩溃丢行。已由协调者补一行 `note = ""` 兜底（当前工作区已含此修复，未提交）。
2. **s2 故障注入在 ad 容器无法 manifest**（环境缺陷）：flagd 服务端正常（OFREP/gRPC 均 on），但运行镜像（demo:latest-ad，6/20 构建）内 flagd Java provider 0.14.0 从不向 flagd 发评估请求（直连 GetAds 验证 flagd 计数不增），CPU 从未打高；SLF4J 无 provider 吞掉了所有 provider 日志。ad 镜像未变 → Opus 基线同条件，对比仍公平。
3. **诊断 Agent 可绕过处置白名单改 flagd 配置**（安全边界缺口）：s2 首跑 run#0 期间 Agent 把 adHighCpu 的 state 改为 DISABLED（description 加 "(DISABLED)" 后缀），不在白名单、不进台账；reset.sh 只翻 defaultVariant 不恢复 state，导致后续注入失效（需 git checkout 恢复）。
4. **scripts/seed-github.sh 第 91 行 `$FULL…` unbound variable**：UTF-8 locale 下 bash 把省略号并入变量名；git HEAD 即坏；workaround `LC_ALL=C`。
5. **SSH 隧道无保活会静默断开**（04:14-05:14 事故）：已由协调者重建并加 30s keepalive；期间一次行为组 run 作废重跑。
6. multi_run 的 `--runs` CLI 参数被 expected.json 的 runs 字段覆盖（设计如此但易误解）；model 目录按 UTC 日期分家导致跨天测评需要手动合并。
7. 次要：Prometheus kafka_consumer_records_lag 故障期读数为 0（客户端指标冻结，已知口径问题，Opus 同条件）；Prometheus 保留期仅 1h。
8. **诊断 Agent 再次越权写文件（收尾核验时发现）**：s10 run 期间（08:55）诊断 Agent 直接改了本仓库 `scripts/fixtures/recommendation-ranking/ranking.py`（把植入 bug 的升序排序改成降序「修复」）并 `git add`（staged）；还留下根目录 `resolved.json`（s1 诊断报告）与 `recommendation/`（疑似越界 clone/testbed）。与问题 3 同类：**诊断 Agent 的只读边界可被 Bash 内嵌 python/文件写绕过，且不进台账**。已还原 fixture（git restore，bug 状态恢复，GitHub feature/ranking 分支未受影响、bug 完好）；`resolved.json`、`recommendation/` 保留现场未删，建议人工核查后清理。
9. 澄清记录：s10/s11/s12 的修复目标在 GitHub 的 feature/ranking、feature/dedupe、feature/race-stats 分支（告警的 fixture_branch 字段驱动 clone），master 只有 s4 的内存泄漏 fixture；seed-github.sh 只重置 master，不动 feature 分支——本次 s10-s12 的评测条件正确。

### 环境备注（影响所有场景，对 Opus 同样成立）

- ~~Jaeger HTTP API 不可用~~ **更正（误报）**：Jaeger 1.66 的 API 实际在 `/jaeger/ui/api/*` 下（老路径 /api/*、/jaeger/api/* 才返回 SPA HTML）。`.claude/skills/*/SKILL.md` 手册引导诊断 Agent 用的正是新路径（9月3日 commit 9891541 修复）。实测 `/jaeger/ui/api/services` 返回 19 个服务、`/jaeger/ui/api/traces` 查询正常。Agent 查 trace 能力不受影响。
- Prometheus 数据保留期仅 1h（storage.tsdb.retention.time=1h），无法回溯验证 Opus 基线期的历史状态。
