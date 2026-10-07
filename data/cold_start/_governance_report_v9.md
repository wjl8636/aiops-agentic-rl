# 旧数据治理报告（Phase A，v9 数据集准备）

> 执行日期：2026-09-08。依据：`docs/数据增强方案.md` §2.2（验收缺口 4 项中的 #1/#2/#3）、§4.3 C2（步修剪）、§4.5（跨类卫生治理）、§7.2/§7.3（GT 扩展 schema 与 kind 决策表）、§6.3（生成流程第 4-6 步）。
>
> 本轮治理只动 `aiops-agentic-rl` 仓库（AIops-agent 仓库只读，仅引用其 `eval/expected.json` 做口径对齐）。不启动任何 Opus 采集（那是 Phase C）。

## 0. 结论速览

| 治理项 | 结果 |
|---|---|
| kind-GT 回填 | 105 个 GT 文件（100 seed + 5 mock）补 `expect_kind_any_of`，依据 §7.3 决策表 + expected.json 同场景口径 |
| kind 校验接入验收 | `rejection_sample.py` 新增第 6 项校验（`check_trajectory`，GT 缺字段时跳过、向后兼容） |
| v8 全量重验收 | 35 条：26 通过、9 拒绝（全部且仅因 kind 错标，其余 5 项条件无一新增失败——确认 2026-09-05 的 route 口径修复工作正常） |
| 隔离/淘汰 | 10 条轨迹移入 `trajectories_quarantined_v9/`（1 条答案钥污染 + 9 条 kind 错标），v8 继承 25 条 |
| 污染全局扫描 | 553 个轨迹文件（12 个目录全量递归）：已验收池里只有 seed_0701（3 份字节相同副本）；另有 1 处方案文档未记录的原始归档污染（pilot_newtypes_v2 的 seed_0282，从未进验收池）；1 处复核后判为误报 |
| 步级修剪 | 25 条幸存轨迹 325 → 289 步，均值 13.00 → **11.56 步/条**；修剪后 25/25 重过全部 6 项验收、污染复检 0 命中 |
| 治理后产物 | `trajectories_governed_v9/`（25 条修剪后轨迹，每条 `meta.pruning` 记录删除明细）——Phase C 采集的新轨迹将与此目录合并组装 v9 |

## 1. kind-GT 回填（任务 1 上半）

### 1.1 口径（对齐依据）

字段名用 `docs/数据增强方案.md` §7.2 的 schema：`expect_kind_any_of`（列表）。判定规则（§7.3 决策表 + `AIops-agent/eval/expected.json` 同机制场景的 kind 定义）：

| 机制 | expect_kind_any_of | 对齐来源 |
|---|---|---|
| kafkaQueueProblems / 消费积压（含组合里的 kafka 主导场景） | `["resource"]` 严格 | s3 `kind: "resource"`（严格）+ queue-backlog 手册 |
| 内存泄漏（MemoryLeakOOM alertname / 无界 list / 缓存字典无 TTL / WeakSet / 文件句柄叙事） | `["deploy_regression"]` 严格 | s4 `kind: "deploy_regression"`（严格）；任务已知错标模式「MemoryLeakOOM 类应为 deploy_regression 不是 resource」 |
| 纯逻辑 bug fixture（ranking/dedupe/race 型） | `["deploy_regression"]` 严格 | s10/s11/s12 均严格 `deploy_regression` |
| 服务级失败 flag（productCatalogFailure / recommendationCacheFailure / cartFailure / paymentFailure / paymentUnreachable / adFailure） | `["dependency", "config"]` | s1（productCatalogFailure 场景）`kind_any_of: ["dependency", "config"]`；列表首位 `dependency` 是 §7.3 决策表的 canonical 值（「统一为 dependency」），第二位按 eval 自己的接受口径放行 |
| adHighCpu / loadGeneratorFloodHomepage 等 flag 驱动负载 | `["resource", "config"]` | s2（adHighCpu 场景）`kind_any_of: ["resource", "config"]` |
| imageSlowLoad / adManualGc 行为调参 flag；证书轮换 / A-B 开关误配等配置叙事 | `["config"]` | §7.3 决策表「行为调参 flag → config」 |
| 「参数配置变更 → CPU/GC 高位」叙事（0621/0625/0633/0635/0637/0643/0653/0588_res/0586 等） | `["resource", "config"]` | 症状=resource、根因=config，两类读法都合法；对齐 s2 对 flag 驱动负载的双值口径 |
| 「调用下游忘配超时/无退避 → 级联超时」叙事（0592_dep/0610/0611/0615/0618/0629/0631/0639/0649/0664/0665/0587_dep/0595_dep） | `["dependency", "deploy_regression"]` | 症状=依赖级联（家族名 dependency）、根因=调用方代码缺陷（s8/s9 对 code 缺陷场景接受 deploy_regression），无 eval 场景直接锚定，两值并放 |
| info_only 无故障种子（0702/0705/0706/0707） | 声称的类型：`["resource"]`/`["dependency"]`/`["config"]`/`["deploy_regression"]` | 与 `_annotation_report.md`「remediation_type 锚定告警声称类型」同一处理原则，scenario_hint 里显式声明了覆盖哪个 kind |
| 组合 flag | 主导机制决定（= 告警顶层 service 标签 / alertname 首位 flag / 叙事自述结论） | 与 GT `suspect_service` 的既有惯例一致（「统一取告警顶层 service 字段」） |

**关于服务级失败 flag 采用双值的说明**：§7.3 决策表括号里写了「s1 接受 dependency|config，统一为 dependency 消除训练内部分歧」。本轮回填选择 `["dependency", "config"]` 而非单值 `["dependency"]`，理由：(a) 任务给的对齐依据第一条就是「AIops-agent 的 eval/expected.json 里同场景的 kind 定义」，s1 的定义就是双值；(b) 方案文档预测「淘汰 3-6 条」、§7.1 预测「flagd/依赖 online_op 家族 v8 继承 ~19」——只有 config 继续被接受时这两个预测才成立（若按单值严格执行，v8 的 0001/0002/0004/0008/0010/0012/0377/0419/0525 共 9 条 kind=config 的轨迹会全部被拒，淘汰量变成 18 条，与文档自身的预测矛盾）；(c) canonical 值（dependency）放在列表首位以表达决策表的偏好。若后续想把口径收紧到单值，把 GT 里的列表改成单元素后重跑 `rejection_sample.py` 即可（脚本无需再改）。

### 1.2 回填范围与统计

- `data/cold_start/ground_truth/` 全部 **100 个 seed GT** + **5 个 mock GT**（mock 与 `build_mock_trajectories()` 的诊断 kind 对齐，保证 mock 测试在 kind 校验下仍 3 过 2 拒）。
- 其中 8 个 GT 对应旧种子池、当前仓库已无同名种子文件（seed_0585_dep-reg / 0586_res / 0587_dep / 0588_config / 0592_res / 0595_dep / 0598_res / 0599_res），其 expect_kind 按 `_annotation_report.md` 里记录的原始叙事判定，逐条依据已写进下方 §1.3 的映射表。
- 映射明细（机制 → 判定依据）已随每个 GT 文件的注释性结构自含（字段值本身），完整对照见本报告附录 A。

### 1.3 重验收结果（任务 1 下半）

命令：`python data/cold_start/rejection_sample.py --traj-dir data/cold_start/trajectories_accepted_v8 --gt-dir data/cold_start/ground_truth`

结果：**26/35 通过，9/35 拒绝**。9 条拒绝原因全部且仅是 kind 错标（remediation_type/suspect_service/route/evidence/台账 5 项无一新增失败——反向确认 2026-09-05 的 route=None 例外口径在 kind 校验接入后依然正确工作）：

| 被拒轨迹 | kind 实际值 | 期望集合 | 错标家族 |
|---|---|---|---|
| seed_0282 | resource | [deploy_regression] | MemoryLeakOOM→resource（§2.2-1 点名） |
| seed_0283 | resource | [deploy_regression] | 同上（§2.2-1 点名） |
| seed_0596 | resource | [deploy_regression] | 同上（§2.2-1 点名家族的第 4 例） |
| seed_0655 | resource | [deploy_regression] | 同上（§2.2-1 点名） |
| seed_0381 | config | [resource] | kafka 队列→config（§2.2-1 点名） |
| seed_0429 | config | [resource] | 同上（§2.2-1 点名） |
| seed_0495 | config | [resource] | 同上（叙事自述「复合资源型压力」，教师却走「flag=on 即 config」捷径） |
| seed_0703 | config | [deploy_regression] | 纯逻辑 bug fixture→config（s11 同实例，s11 严格 deploy_regression） |
| seed_0704 | config | [deploy_regression] | 同上（s10 同实例） |

**与方案文档预测的偏差（如实记录）**：文档预测「淘汰 3-6 条 kind 噪声轨迹」。实际 9 条。差集是 seed_0495（与 0381/0429 同属文档点名的「kafka 队列类标成 config」家族，只是 §2.2-1 举例时没点名它）和 seed_0703/0704（ranking/dedupe fixture 与 s10/s11 同实例、教师 kind=config，直接对应 s10/s11 测评 kind_ok=False 的训练来源；§7.1 曾按「剔除 0701 后继承 2 条」估算，与 §7.3 决策表冲突，本轮以决策表 + expected.json 为准执行）。

**家族级账本（对照 §7.1 的 v8 继承估算）**：

| 家族 | v8 实有 | 治理后继承 | §7.1 估算 | 偏差原因 |
|---|---|---|---|---|
| flagd/依赖 online_op | 19 | 16 | ~19 | 0381/0429/0495 kind 淘汰 |
| 混合根因 also_code_fix | 9 | 5 | ~8 | 0282/0283/0596/0655 kind 淘汰（文档 §7.1 的 ~8 与其 §2.2-1 自己点名的错标名单不自洽，以实际 kind 值为准） |
| 纯 code_fix | 3 | 0 | 2 | 0701 污染隔离 + 0703/0704 kind 淘汰；该家族由 Phase C D1(4)+D2(2)+D3(2) 重建，且新种子会用正确 kind 标注 |
| info_only/噪声 | 4 | 4 | ~4 | 一致 |
| **合计** | **35** | **25** | ~30-32 | 9 条 kind 淘汰 + 1 条污染隔离 |

kind 标签质量前后对比：v8 全量 35 条里 26 条 kind 落在新 GT 集合内（74%）；治理后 25/25（100%）。v8 kind 分布 config 20/35（57%）收敛为治理后 config 15/25（60%，其中 13 条是服务级失败/行为调参场景的合法 config，2 条是 0708/0709 imageSlowLoad 的合法 config）。

## 2. 答案钥污染扫描与隔离（任务 2）

### 2.1 扫描口径与范围

- 范围：`data/cold_start/` 下全部 12 个 `trajectories*` 目录（含 `trajectories/` 的全部子目录、各 accepted 池、raw 归档），共 **553 个轨迹 JSON**，命令（tool_input）与观测（observation）都查。
- 模式（§6.3 第 5 步终检口径 + §2.3 已知形态）：`eval/expected.json`、`eval/multi_run`、`multi_run`、`reports/`、`alerts/s\d+`、测评场景名（s10_ranking / s11_dedupe / s12_race / s7_hybrid / s8_fix_fail / s9_adv / s_lowconf / s5_dup / s6_info）、expected.json 专有字段名（expect_route / suspect_service_contains / kind_any_of）。

### 2.2 扫描结果

| 发现 | 位置 | 处置 |
|---|---|---|
| **seed_0701（已知案例，确认无扩大）** | 6 份字节相同副本：`trajectories_accepted_v5` / `trajectories_accepted_v6` / `trajectories_accepted_v8`（3 份已验收）+ `trajectories/code_fix_retry` / `real_run_merged_v4` / `real_run_merged_v5`（3 份原始归档）。污染步骤：第 13 步 find 命中 eval 文件（观测带出 expected.json / s10_ranking 字样）、第 14 步 `cat eval/expected.json | grep -A 20 s10_ranking`（观测可见 ./alerts/s10_ranking.json、suspect_service_contains 等答案钥内容）、第 15/16 步继续读 alerts/s10_ranking.json | 3 份已验收副本**整条移入隔离区**（三个池都移，防未来合并时复活）；3 份原始归档保留原位（归档不进数据集，manifest 里登记为禁止再验收，§6.3-5 终检会拦）。不做「修剪 13-16 步后保留」的方案（§4.5-1 推荐：其结论本身可能受答案钥影响） |
| **新发现：pilot_newtypes_v2 的 seed_0282** | 原始归档（meta.degraded=timeout、从未进任何验收池）。第 17 步 `find … -iname "expected.json"`、第 18/19 步用 python 直接打开 `eval/expected.json` 并打印 `d['s4']` / `d['s7_hybrid']` 等答案钥内容——比 0701 更直接 | 不移动（非验收池），在 manifest 登记：任何后续轮次不得从该归档验收此轨迹。v8 已验收的 0282 是另一次干净采集（扫描无命中，它因 kind 被淘汰是另一回事） |
| 误报 1 例 | pilot_newtypes_v1 的 seed_0283 第 7 步命中 `reports/`——人工复核为 `find` 列目录时观测里带出的普通路径片段，非读取测评报告 | 判定非污染，仅记录 |

**结论：已验收池里除了 seed_0701 没有其他答案钥污染**；污染行为集中在「教师找不到本地证据时去翻仓库」的场景（0701 的 s10 fixture 定位、pilot 0282 的 timeout 降级排查），这也印证了 §2.1 的判断：采集环境让教师能读到 `eval/expected.json` 是结构性入口（§8.4 建议 AIops-agent 侧隐藏答案钥，超出本轮范围）。

## 3. 全量步修剪（任务 3）

### 3.1 实现与规则

新脚本 `data/cold_start/prune_steps.py`（修剪后副本写 `trajectories_governed_v9/`，原文件不动，每条轨迹 `meta.pruning` 记录逐条删除原因与命令摘要，可复现可回滚）。规则：

| 规则 | 定义 | 本轮命中 |
|---|---|---|
| R1 marked-repeat | 采集时 `signals.annotate_steps` 标记的 `is_repeat_no_new_info=true` | 1 |
| R2 noise | TaskStop/KillShell/KillBash 等中断类工具；`find /` 全盘扫描 | 4 |
| R3 malformed | `StructuredOutput` 带 `__unparsedToolInput` 的 schema/JSON 解析失败重试步（**malformed 的识别方式**：SDK 把解析不出的原始输入挂在这个键上，找键即识别，不猜语义；保留成功的那次） | 0（v8 唯一实例在 seed_0381 第 9 步，该轨迹已因 kind 淘汰；规则保留给 Phase C 新数据） |
| R4 skeleton-repeat | 同「命令骨架」（docker-logs:容器 / docker-status:目标 / ofrep:flag / jaeger:URL / prom:完整PromQL / 其余去 cd 前缀近 exact）的后续调用，且 ① 信号类型无新增（signals 口径）② 观测事实 token 无新增（时间戳/长哈希/IP 这类重跑必变的噪声 token 先过滤） | 12 |
| R5 failed-query-retry | 「查了但什么都没拿到」（空观测/空结果集 JSON）且同骨架有成功兄弟步——删失败那次保留成功那次（R3 的 Bash 版） | 5 |
| R6 no-signal-no-fact | 既不覆盖任何信号源、观测里也没有 ≥2 个事实 token 的纯噪声步（cd 失败的源码 grep、`ls workspace/` 目录列表、`date` 回显） | 14 |

安全网（§4.3 C2 第 4 条「不过就回退该步」）：`hook_decision=allow` 的台账步骤与成功的 StructuredOutput 终止步永不修剪；每个候选修剪立即复验 evidence 可追溯（`rejection_sample.evidence_traceability_ok`）+ 台账一致（`executed_actions_match_ledger`），不过就回退。本轮**回退 0 步**（没有任何被修剪步骤是 evidence 的唯一出处——修剪目标本身就是无信息步）。

### 3.2 修剪前后步数（任务要求的记录）

| 口径 | 修剪前 | 修剪后 | 变化 |
|---|---|---|---|
| 25 条幸存轨迹总步数 | 325 | 289 | -36（-11.1%） |
| **均值（步/条）** | **13.00** | **11.56** | **-1.44** |
| 对照：v8 全量 35 条均值（含被隔离 10 条，它们均值 14.3 步） | 13.37 | — | — |
| 逐轨迹明细 | 见 `trajectories_governed_v9/*/meta.pruning`（如 0004: 22→19、0003: 17→11、0012: 20→16、0706: 6→4；最短 0708 保持 2 步） | | |

**与 ≤11 目标的差距（如实记录）**：规则化修剪的诚实下限是 11.56。剩余未修剪的步全部是「不同查询」（不同 PromQL 表达式 / 不同 Jaeger 参数 / docker ps→stats→inspect 的不同侧面 / 源码定位 / RAG），继续删就是删有信息的观测而不是噪声，会把 evidence 引用的观测删掉（安全网会大量回退）。≤11 目标隐含假设采集时的 repeat 标注能抓住大量重复——实际 v8 全量只有 4 个标记步（survivors 里只有 1 个），因为该标注的 key 是完整命令字符串，`--tail 100` vs `--since 30m` 这类参数级变体全部漏标（R4 的骨架归组正是为补这个洞）。补齐路径：Phase C 的 C1 家族本身就是 ≤8 步的新轨迹，v9 全集（25 旧 + ~40 新）的混合均值将低于 §8.1 的 ≤12 训练门；若仍想压旧数据，方向是改 `signals.py` 的 repeat 标注口径（影响 reward 模块共用语义，本轮未动）。

### 3.3 修剪后复验（全部通过）

- `rejection_sample.py` 对 `trajectories_governed_v9/` 全 6 项验收：**25/25 通过**（evidence 可追溯性在修剪后的步骤集上重算通过——没有 evidence 引用的观测被删）。
- 污染模式复扫：**0 命中**。
- 下游管线冒烟：`prefix_split.py` 对治理目录产出 289 条子样本（25 diagnosis + 264 tool_call，数量与步数一致），malformed 目标 0（`prefix_split.py` 已按 §4.5-3 加了跳过 `__unparsedToolInput` 步骤的第二道防线）。

## 4. 改动清单（全部在 aiops-agentic-rl 内）

| 文件 | 改动 |
|---|---|
| `data/cold_start/ground_truth/*.json`（105 个） | 新增 `expect_kind_any_of` 字段（100 seed + 5 mock；mock 与 build_mock_trajectories 的 kind 对齐） |
| `data/cold_start/ground_truth/_annotation_report.md` | 追加「2026-09-08 追加」节记录回填方法论 |
| `data/cold_start/rejection_sample.py` | 新增第 6 项 kind 校验（`expect_kind_set` 兼容 `expect_kind_any_of` 列表与 `expect_kind` 单值；GT 缺字段跳过） |
| `data/cold_start/prefix_split.py` | 拆分前剔除 malformed 步骤（§4.5-3 规则化） |
| `data/cold_start/prune_steps.py` | 新增：步级修剪脚本（R1-R6 + 安全网回退 + meta.pruning 记录） |
| `data/cold_start/tests/test_rejection_sample.py` | 新增 5 个 kind 校验测试 |
| `data/cold_start/tests/test_prune_steps.py` | 新增：11 个修剪规则/安全网测试 |
| `data/cold_start/tests/test_prefix_split.py` | 新增 1 个 malformed 跳过测试 |
| `data/cold_start/trajectories_quarantined_v9/` | 新增：隔离区（12 份文件 + `_quarantine_manifest.json` 逐条原因） |
| `data/cold_start/trajectories_governed_v9/` | 新增：治理后 25 条轨迹（修剪副本 + meta.pruning） |
| `data/cold_start/trajectories_accepted_v5/ v6/ v8/` | 移出 0701（3 份）与 9 条 kind 淘汰轨迹（v8）——历史 jsonl 产物未动 |

**测试**：`AIops-agent/.venv/bin/python -m pytest data/cold_start/tests/` 65/65 通过（repo 自带 `.venv` 缺 `dotenv`，16 个依赖 mock_mode 的测试在改动前后同样失败——失败集合 diff 为空，非本轮引入）。

## 5. 给 Phase C 的交接说明

1. 新采集轨迹直接并入 `trajectories_governed_v9/` 同口径处理：`rejection_sample.py`（含 kind）→ `prune_steps.py` → 污染终检（§6.3-5，模式清单见 §2.1）→ `prefix_split.py` → `to_llamafactory_format.py`。
2. 新种子 GT 一律带 `expect_kind_any_of`（§7.2 扩展 schema），决策表 §7.3 + 本报告 §1.1 的口径表。
3. v8 继承的纯 code_fix 家族已清零，D1/D2/D3（8 条）是该家族唯一来源。
4. `pilot_newtypes_v2` 归档确认有答案钥污染前科，复用前必须过终检。
5. 训练门（§8.1）自查：污染 0 ✓、kind 通过率 100%（25/25）✓、malformed 目标 0 ✓、步数均值 11.56（v8 继承部分；全集门 ≤12 依赖 Phase C 新轨迹混合）。

## 附录 A：expect_kind_any_of 完整映射

（`DEP=[dependency,config]`、`RES=[resource]`、`RES_CONF=[resource,config]`、`DR=[deploy_regression]`、`CONF=[config]`、`DEP_DR=[dependency,deploy_regression]`）

| alert_id | 机制 | 值 |
|---|---|---|
| seed_0001/0002/0003/0004/0008/0010/0012_parametrized_dependency | paymentUnreachable/paymentFailure/cartFailure/recommendationCacheFailure/productCatalogFailure/adFailure | DEP |
| seed_0095/0096_parametrized_resource | adHighCpu | RES_CONF |
| seed_0189/0191_parametrized_resource | kafka 消费积压 | RES |
| seed_0282/0283_parametrized_deploy_regression | MemoryLeakOOM | DR |
| seed_0375_flagd_combination_config | imageSlowLoad | CONF |
| seed_0376_flagd_combination_resource | loadGeneratorFloodHomepage | RES_CONF |
| seed_0377/0378_flagd_combination | adFailure / recommendationCacheFailure | DEP |
| seed_0379_flagd_combination_resource | adManualGc | CONF |
| seed_0381/0429_flagd_combination_resource | kafkaQueueProblems 主导组合 | RES |
| seed_0382/0384/0385_flagd_combination | productCatalogFailure / recommendationCacheFailure 主导组合 | DEP |
| seed_0388_flagd_combination_resource | adHighCpu 主导组合 | RES_CONF |
| seed_0395_flagd_combination_resource | adFailure 主导组合 | DEP |
| seed_0416/0431_flagd_combination_resource | adManualGc 主导组合 | [config, resource] |
| seed_0419/0525_flagd_combination_resource | productCatalogFailure 主导组合 | DEP |
| seed_0430_flagd_combination_dependency | cartFailure 主导组合 | DEP |
| seed_0449/0495_flagd_combination_resource | RecommendationDegradedByQueueAndCache（kafka lag 主导） | RES |
| seed_0482/0493_flagd_combination_resource | AdServiceCompoundResourcePressure（adHighCpu+adManualGc） | RES_CONF |
| seed_0494_flagd_combination_dependency | CheckoutFullyDown（cartFailure+paymentUnreachable） | DEP |
| seed_0585_hrd_deploy_regression | checkout 会话缓存无 TTL（内存泄漏叙事） | DR |
| seed_0585/0587_hrd_resource | kafka lag 陈旧告警 | RES |
| seed_0586_hrd_resource | GC 配置变更→资源问题（旧池，无种子文件） | RES_CONF |
| seed_0587/0595_hrd_dependency | cart/product-catalog 批量查询忘配超时（旧池） | DEP_DR |
| seed_0588_hrd_config | checkout 证书轮换（旧池） | CONF |
| seed_0588_hrd_resource | shipping 运费算法配置切换→CPU 高 | RES_CONF |
| seed_0591/0595_hrd_config/0600/0614/0658 | 证书轮换/信任链 | CONF |
| seed_0592_hrd_dependency | payment 风控调用无退避重试 | DEP_DR |
| seed_0592/0598_hrd_resource | consumer 误缩容积压（旧池） | RES |
| seed_0593/0626_hrd_resource | ad adHighCpu/GC 陈旧告警 | RES_CONF |
| seed_0594_hrd_resource | frontend 流量洪峰 | RES |
| seed_0596/0597/0598_dep/0620/0660/0655 | 内存泄漏叙事（重试列表/缓存字典无 TTL/WeakSet/会话缓存） | DR |
| seed_0599_hrd_resource | 运行时 flag→payment CPU（旧池） | RES_CONF |
| seed_0601_hrd_config | recommendation A/B 开关误配 | CONF |
| seed_0602_hrd_dependency | productCatalogFailure flag 驱动 5xx | DEP |
| seed_0605/0609/0617/0630/0712 | _seen_product_ids / 无界 list 内存泄漏 | DR |
| seed_0610/0611/0615/0618/0629/0631/0639/0649/0664/0665 | 调用下游忘配超时→级联超时 | DEP_DR |
| seed_0621/0625/0633/0635/0637/0643/0653 | 参数配置变更→CPU/GC 高位 | RES_CONF |
| seed_0641_hrd_deploy_regression | 排序 bug 传闻（声称类型；结论由 route=feishu_low_confidence 承载） | DR |
| seed_0642_hrd_resource | ad CPU 飙高（流量驱动） | RES |
| seed_0663/0666_hrd_resource | kafka 消费者副本缩容积压 | RES |
| seed_0701/0703/0704_parametrized | ranking/dedupe fixture 逻辑 bug（s10/s11 同实例） | DR |
| seed_0702/0705/0706/0707_parametrized | info_only（声称类型 resource/dependency/config/deploy_regression） | 对应单值 |
| seed_0708/0709_hrd_config | imageSlowLoad flag | CONF |
| seed_0710/0711_hrd_resource | kafka 突发/重连风暴 lag | RES |
| mock_dep_clean / mock_fail_evidence / mock_fail_mismatch / mock_hookdeny_res / mock_hybrid_leak | 与 build_mock_trajectories 诊断 kind 对齐 | [dependency]/[resource]/[resource]/[resource]/[deploy_regression] |
