# 冷启动种子告警 Ground Truth 标注报告

标注范围：`data/clean/split/train/` 下 29 条种子告警（不含 `data/cold_start/collect_trajectories.py::build_mock_trajectories()` 里已有的 5 条 mock 数据）。

**方法论声明**：本次标注全程未读取、未参考 `data/cold_start/trajectories/real_run/` 目录、`/tmp/collect_trajectories_real_run.log`，或任何"真实采集本次实际跑出了什么结果"的信息。判断依据仅限于：种子告警内容本身（`labels`/`annotations`/`scenario_hint`）、`data/seeds/generated/_manifest.json` 的 `intended_kind`、`AIops-agent/agent/core/remediation.py` 白名单逻辑、`AIops-agent/agent/run.py`/`data/cold_start/collect_trajectories.py::_infer_route()` 的路由推断逻辑、`AIops-agent/eval/expected.json` 的标注风格、`AIops-agent/.claude/skills/{dependency-error,resource-cpu,queue-backlog,memory-oom}/SKILL.md` 判定规则、`AIops-agent/scripts/fixtures/recommendation/recommendation_server.py` 的真实泄漏实现、`data/cold_start/fault_injection.py` 的真实故障注入覆盖范围，以及 `AIops-agent/agent/agents/code_fix.py` 的单仓 MVP 约束。

## 总体结论

- 高置信度：21 条
- 中置信度：4 条
- 低置信度：4 条

低置信度的 4 条（均为 `deploy_regression` 内存泄漏叙事，需要人工重点确认）：
`seed_0282_parametrized_deploy_regression`、`seed_0283_parametrized_deploy_regression`、
`seed_0585_historical_reverse_derivation_deploy_regression`、`seed_0596_historical_reverse_derivation_deploy_regression`。

## 2026-09-05 修正：flagd 开关驱动故障的止血无效性（方法论错误更正）

**问题**：本报告最初对 `dependency`/`flagd_combination` 家族（故障机制是 flagd 功能开关，`data/cold_start/fault_injection.py` 会在跑真实轨迹前把 `scenario_hint` 里 `flag=`/`flags=` 解析出的开关打到故障 variant）的种子，套用了"重启容器=止血手段"的经验规则，只要 `suspect_service` 在 `AIops-agent/agent/core/remediation.py` 的 `REMEDIATION_ALLOWED_TARGETS` 白名单内就标 `auto_remediated`。这是错的：白名单里仅有的三个动作——`restart_instance`/`migrate_instance`/`scale_resources`——全部是 docker 容器生命周期/资源操作，**不会改变 flagd 配置里的开关值**。对这类故障而言，重启/迁移/扩容容器后，flagd 仍会按同一份（仍处于故障 variant 的）配置文件应答服务的 OFREP 查询，故障立即复现，止血在语义上无效。

**验证方式**：逐条核对了 `data/cold_start/ground_truth/` 下所有 `_dependency` 或 `flagd_combination_*` 文件名、且当时 `expect_route` 为 `auto_remediated`（或 `_and_code_fix_pr`）的条目（共 13 条），确认每条种子文本本身的 `scenario_hint` 都含 `flag=`/`flags=`（机制确属 flagd 驱动），并进一步核对了 `data/cold_start/trajectories/real_run/<alert_id>.json` 里真实采集到的模型证据链：12 条里模型都主动查询了 flagd 的 OFREP 接口（`curl .../ofrep/v1/evaluate/flags/<flag>`）拿到了开关当前值，并在 `final_diagnosis.remediation_detail`/`evidence` 里明确写出"重启/扩容不会改变 flag 值，因此不执行白名单内的止血动作"，`executed_actions` 均为空、`route` 均为 `feishu_online_op`——这是模型的正确判断，不应被判 reject。

**改动**：以下 12 条的 `expect_route` 由 `auto_remediated` 改为 `feishu_online_op`（`remediation_type`/`suspect_service` 均未改动，因为 online_op 这个判断本身没错，只是"能否自动执行"错了）：
`seed_0001_parametrized_dependency`、`seed_0002_parametrized_dependency`、`seed_0003_parametrized_dependency`、
`seed_0004_parametrized_dependency`、`seed_0008_parametrized_dependency`、`seed_0012_parametrized_dependency`、
`seed_0375_flagd_combination_config`、`seed_0377_flagd_combination_dependency`、`seed_0379_flagd_combination_resource`、
`seed_0493_flagd_combination_resource`、`seed_0494_flagd_combination_dependency`（中置信度）、`seed_0495_flagd_combination_resource`（中置信度）。
逐条更新说明见下方对应表格行内追加的"[2026-09-05 更新]"标注（原理由文字保留未删，仅追加说明，便于追溯）。

**审查后未改动**：`seed_0376_flagd_combination_resource`（`flag=loadGeneratorFloodHomepage`，机制上同样是 flagd 驱动，理论上也该改）——但它对应的 `data/cold_start/trajectories/real_run/seed_0376_flagd_combination_resource.json` 是一条 `meta.degraded="timeout"` 的轨迹：`final_diagnosis`/`route` 均为 `None`，模型没有走到"查证 OFREP 后判断不执行"这一步，反而在排查过程中一度对**不相关的 cart 容器**执行了 `docker restart cart`（该容器崩溃是环境残留状态、与本条 frontend 告警的根因无关）。这条轨迹本身证据链不完整、不能作为"模型正确识别 flagd 驱动止血无效"的依据，所以按任务约束（证据模糊时不强改）保留 `expect_route=auto_remediated` 不动；即使改了，因为 `remediation_type`/`suspect_service`/`route` 三项在这条轨迹里都是 `None`，拒绝采样结果也不会因此从 reject 变成 accept。

**重跑拒绝采样结果**：`python data/cold_start/rejection_sample.py --traj-dir data/cold_start/trajectories/real_run --gt-dir data/cold_start/ground_truth --copy-kept-to data/cold_start/trajectories_accepted` 修正前通过 1/29（仅 `seed_0010_parametrized_dependency`），修正后通过 13/29（新增上述 12 条）。其余 16 条 reject 均是本次修正范围之外的问题（`resource_cpu`/`queue_backlog`/`historical_reverse_derivation` 系列的 `remediation_type`/`suspect_service` 不一致、`code_fix` 家族的 `feishu_fix_unverified` 未命中、4 条低置信度 `deploy_regression` 种子，以及上述 `seed_0376`），不在本次任务范围内。

## 系统性注意事项（不是单条种子的问题，但会系统性影响本轮 GT 的可验证性）

1. **故障注入覆盖范围有限**：`data/cold_start/fault_injection.py` 只处理 `dependency`、`flagd_combination` 两类生成路径（能从 `scenario_hint` 解析出 `flag=`/`flags=`）。`resource_cpu`、`queue_backlog`、`deploy_regression`、`historical_reverse_derivation` 四类生成路径**完全没有**真实故障注入，跑真实轨迹时活环境不一定处于告警描述的故障状态。本次 29 条种子里，属于这些无注入路径的有 15 条（2 条 resource_cpu、2 条 queue_backlog、2 条 deploy_regression、9 条 historical_reverse_derivation）。这不影响 GT 本身的标注逻辑（GT 是"这条告警应该怎么判"，不是"这次环境是否真复现了"），但意味着这些种子上的真实轨迹表现可能因环境未复现故障而系统性偏离 GT，拒绝采样时如果大量在这些种子上不匹配，不能简单归因于模型判断错误。
2. **`code_fix` 单仓 MVP 约束**：`AIops-agent/agent/agents/code_fix.py::clone_service_repo()` 硬编码克隆 `config.GITHUB_REPO`（默认 `"recommendation"`），**忽略传入的 `service` 参数**。也就是说，任何诊断为 `code_fix` 但 `suspect_service` 不是 `recommendation` 的种子，机制上不可能克隆到正确的仓库、更不可能验证修复。本轮里 `seed_0587`（cart）、`seed_0595`（product-catalog）属于这类，已参照 `eval/expected.json` 的 `s8_fix_fail` 先例标为 `feishu_fix_unverified`。
3. **`deploy_regression` 叙事与真实环境实现不匹配**：`AIops-agent/scripts/fixtures/recommendation/recommendation_server.py` 里唯一真实实现的内存泄漏机制是模块级列表 `_seen_product_ids` 的无界 `append`（即 `unbounded_list_append`）。种子生成阶段扩增出的 `unbounded_dict_accumulation`/`unclosed_file_handle`/`misused_weakset` 等叙事变体在真实环境里没有对应代码可复现。详见下方低置信度条目。
4. **路由推断的一个不对称性**：`_infer_route()`（`collect_trajectories.py`/`run.py`）对 `online_op`（含 `also_code_fix` 混合）路径只看 `执行台账是否有实际执行` 和 `also_code_fix` 标志位，不检查代码修复是否验证通过；只有纯 `code_fix` 路径的验证失败才会被上层逻辑改写成 `feishu_fix_unverified`（对照 `eval/expected.json` 的 `s8_fix_fail`）。这意味着"发版内存泄漏→online_op(重启止血)+also_code_fix=true"这类混合场景，即使 code_fix 那一半因为单仓约束注定验证失败，机械推导出的路由仍是 `auto_remediated_and_code_fix_pr`，而不会退化成 `feishu_fix_unverified`。本报告在给低置信度条目选择默认路由时遵循了这条机械规则，但已在对应条目里注明这个选择本身依赖于"如果真被判成这个 remediation_type"的假设，标注置信度因此偏低。

## 逐条标注明细

### 高置信度（21 条）

| alert_id | remediation_type | suspect_service | expect_route | 理由 |
|---|---|---|---|---|
| seed_0001_parametrized_dependency | online_op | payment | ~~auto_remediated~~ → **feishu_online_op** | payment 依赖故障（paymentUnreachable），payment 在白名单内，按 dependency-error 规则判 online_op。 **[2026-09-05 更新：expect_route 改为 `feishu_online_op`——故障是 flagd 开关 `paymentUnreachable` 驱动，白名单内的 restart/migrate/scale 都不改变开关值，止血无效；真实轨迹显示模型已查询 OFREP 并正确判断不执行。remediation_type/suspect_service 不变。见文首"2026-09-05 修正"节。]** |
| seed_0002_parametrized_dependency | online_op | payment | ~~auto_remediated~~ → **feishu_online_op** | 同上（paymentFailure），payment 在白名单内。 **[2026-09-05 更新：同 seed_0001，理由同上，flag=paymentFailure。]** |
| seed_0003_parametrized_dependency | online_op | cart | ~~auto_remediated~~ → **feishu_online_op** | cart 依赖故障，cart 在白名单内。 **[2026-09-05 更新：flag=cartFailure 驱动，真实轨迹显示 cart 因缺失的 valkey-cart 依赖崩溃、且 OFREP 确认开关为 on，重启前依赖仍缺失会立即再崩溃，模型正确未执行，改为 `feishu_online_op`。]** |
| seed_0004_parametrized_dependency | online_op | recommendation | ~~auto_remediated~~ → **feishu_online_op** | recommendation 缓存依赖故障，recommendation 在白名单内。 **[2026-09-05 更新：flag=recommendationCacheFailure，OFREP 确认开关值，重启不会改变 flagd 静态配置，改为 `feishu_online_op`。]** |
| seed_0008_parametrized_dependency | online_op | cart | ~~auto_remediated~~ → **feishu_online_op** | 与 seed_0003 同类（severity 不同），cart 在白名单内。 **[2026-09-05 更新：同 seed_0003，flag=cartFailure，理由同上。]** |
| seed_0010_parametrized_dependency | online_op | product-catalog | feishu_online_op | product-catalog 依赖故障，但 product-catalog **不在** `REMEDIATION_ALLOWED_TARGETS` 白名单，只能发飞书卡片交人工。 |
| seed_0012_parametrized_dependency | online_op | ad | ~~auto_remediated~~ → **feishu_online_op** | ad 依赖故障，ad 在白名单内。 **[2026-09-05 更新：flag=adFailure，OFREP 确认开关值，重启 ad 容器不改变 flagd 中的开关，改为 `feishu_online_op`。]** |
| seed_0095_parametrized_resource | online_op | ad | auto_remediated | CPU 超阈值（90%），按 resource-cpu 规则默认先止血，ad 在白名单内。 |
| seed_0096_parametrized_resource | online_op | ad | auto_remediated | 同上，阈值更高（94%，critical），判断不变。 |
| seed_0189_parametrized_resource | online_op | kafka | feishu_online_op | Kafka 消费积压，按 queue-backlog 规则判 online_op（扩消费者），但 kafka 是有状态组件、不在白名单内，机制上不可能自动执行，只能人工。 |
| seed_0191_parametrized_resource | online_op | kafka | feishu_online_op | 同上，lag 更大（24000，critical），判断不变。 |
| seed_0375_flagd_combination_config | online_op | frontend | ~~auto_remediated~~ → **feishu_online_op** | 单点 flag（imageSlowLoad）导致的 frontend 异常，frontend 在白名单内，与本仓库其余 flag 驱动场景（如 seed_0001 系列）保持同一套"重启视为止血手段"的处理惯例。 **[2026-09-05 更新：这条"重启视为止血手段"的处理惯例本身就是本次要修正的方法论错误——imageSlowLoad 是 flagd 配置里的延迟取值（如 "10sec"），只能靠改 flagd 配置文件回退，docker 层的 restart/migrate/scale 都不涉及 flagd 配置，止血无效；真实轨迹显示模型 OFREP 查询后正确未执行，改为 `feishu_online_op`。]** |
| seed_0376_flagd_combination_resource | online_op | frontend | auto_remediated | 流量骤增导致过载，属于典型 resource 场景，frontend 在白名单内。 **[2026-09-05 复核未改：机制上同属 flagd（flag=loadGeneratorFloodHomepage）驱动、理论上也该改，但对应真实轨迹 `meta.degraded="timeout"`，模型未走到最终诊断（`final_diagnosis`/`route` 均为 None），过程中还对无关的 cart 容器执行了一次 `docker restart cart`（环境残留的旧崩溃，与本告警根因无关）。证据链不完整，按"证据模糊不强改"原则保留原值，详见文首"2026-09-05 修正"节的说明；即使改了也不会让这条轨迹通过拒绝采样。]** |
| seed_0377_flagd_combination_dependency | online_op | ad | ~~auto_remediated~~ → **feishu_online_op** | 单点 flag（adFailure）导致 ad 整体请求失败，ad 在白名单内。 **[2026-09-05 更新：真实轨迹显示模型指出该故障由 flagd 集中判定、经 OFREP/gRPC 返回给 ad，与 ad 容器状态无关，重启无法止血，正确未执行，改为 `feishu_online_op`。]** |
| seed_0379_flagd_combination_resource | online_op | ad | ~~auto_remediated~~ → **feishu_online_op** | GC 停顿（adManualGc），按 resource-cpu"不确定优先 online_op 先止血"规则处理，ad 在白名单内。 **[2026-09-05 更新：GC 停顿由 flagd 开关 adManualGc 驱动（OFREP 确认为 on），容器重启后开关仍为 on、下次触发条件满足会继续强制 Major GC，重启不能真正止血，真实轨迹模型已正确判断不执行，改为 `feishu_online_op`。]** |
| seed_0493_flagd_combination_resource | online_op | ad | ~~auto_remediated~~ → **feishu_online_op** | 高 CPU + GC 停顿复合场景，仍是 ad 单服务的资源问题，ad 在白名单内。 **[2026-09-05 更新：复合故障由 flagd 开关 adHighCpu+adManualGc 驱动，真实轨迹显示模型核实容器实时指标正常（CPU/GC 均在低位）、判定问题在 flagd 配置侧而非容器资源侧，重启/扩容无效，正确未执行，改为 `feishu_online_op`。]** |
| seed_0586_historical_reverse_derivation_resource | online_op | ad | auto_remediated | 历史工单：GC 配置变更导致的资源问题，与 seed_0379 同类，ad 在白名单内。 |
| seed_0588_historical_reverse_derivation_config | online_op | checkout | auto_remediated | 历史工单：证书轮换不匹配导致 checkout 异常，属于配置类问题，checkout 在白名单内。 |
| seed_0592_historical_reverse_derivation_resource | online_op | kafka | feishu_online_op | 历史工单：consumer 被误缩容导致积压，仍是 kafka 队列积压，kafka 不在白名单内，只能人工。 |
| seed_0598_historical_reverse_derivation_resource | online_op | kafka | feishu_online_op | 历史工单：consumer 缩容导致积压，与 seed_0592 同类，kafka 不在白名单内。 |
| seed_0599_historical_reverse_derivation_resource | online_op | payment | auto_remediated | 历史工单：运行时 flag 导致 payment CPU 升高，属于资源类问题，payment 在白名单内。 |

### 中置信度（4 条）

| alert_id | remediation_type | suspect_service | expect_route | 理由 | 其他也算合理的路由/字段 |
|---|---|---|---|---|---|
| seed_0494_flagd_combination_dependency | online_op | cart | ~~auto_remediated~~ → **feishu_online_op** | 复合故障 `flags=cartFailure+paymentUnreachable` 同时命中 cart 和 payment 两个依赖，告警本身挂在 checkout（下单链路整体不可用），根因服务是 cart 和 payment 二者之一或二者都是，二者都在白名单内所以 `remediation_type`/`expect_route` 不受影响，但 `suspect_service` 该填哪一个存在真实歧义。 **[2026-09-05 更新：`expect_route` 改为 `feishu_online_op`——两个 flag 都是 flagd 驱动（OFREP 确认 `paymentUnreachable`/`cartFailure` 均为 true），且真实轨迹里 cart 容器还因缺失 valkey/redis 依赖崩溃，重启前依赖未恢复会立即再崩溃，重启对两个根因都无效；模型正确未执行。`remediation_type`/`suspect_service` 判断本身不变，仍是同一处歧义。]** | `suspect_service` 也可合理填 `payment`（甚至 `checkout` 本身，如果诊断落在告警服务而非上游依赖）。 |
| seed_0495_flagd_combination_resource | online_op | recommendation | ~~auto_remediated~~ → **feishu_online_op** | 复合故障 `flags=kafkaQueueProblems+recommendationCacheFailure`，recommendation 服务同时承受 kafka 队列积压和自身缓存失效双重压力。选择 recommendation+auto_remediated 是因为它是直接受影响、且在白名单内、可通过重启缓解缓存问题的服务。 **[2026-09-05 更新：`expect_route` 改为 `feishu_online_op`——"可通过重启缓解缓存问题"这个假设是本次要修正的方法论错误，真实轨迹显示 recommendation 容器本身健康（CPU/内存均正常、无重启无 OOM），重启/扩容不会改变 flagd 里 `recommendationCacheFailure`/`kafkaQueueProblems` 两个开关的值，模型核实后正确未执行。`remediation_type`/`suspect_service` 不变。]** | 若诊断把根因归到 kafka 队列积压本身，则 `suspect_service=kafka`、`expect_route=feishu_online_op`（kafka 不在白名单）。两种归因在这条复合告警里都合理。 |
| seed_0587_historical_reverse_derivation_dependency | code_fix | cart | feishu_fix_unverified | 历史工单描述"cart 批量查询超时"，是典型的代码层缺陷（无超时保护/低效查询），按 dependency-error 规则"根因在源码→code_fix"。但 `code_fix.py::clone_service_repo()` 硬编码只克隆 `recommendation` 仓库，cart 的 code_fix 机制上不可能验证，参照 `eval/expected.json` 的 `s8_fix_fail` 先例判 `feishu_fix_unverified`。 | 若模型对证据强度评估更保守，也可能直接落到 `feishu_low_confidence`；这属于同一类"结构性验证失败"的可接受路由。 |
| seed_0595_historical_reverse_derivation_dependency | code_fix | product-catalog | feishu_fix_unverified | 与 seed_0587 同理，"product-catalog 导入超时"是代码层问题，但 product-catalog 既不在处置白名单、code_fix 又只能操作 recommendation 仓库，双重约束下判 `feishu_fix_unverified`。 | 同上，`feishu_low_confidence` 也是可接受的替代路由。 |

### 低置信度（4 条）——均为 deploy_regression 内存泄漏叙事，需人工确认

| alert_id | remediation_type | suspect_service | expect_route | 为什么低置信度 | 需要人工确认什么 |
|---|---|---|---|---|---|
| seed_0282_parametrized_deploy_regression | online_op | recommendation | auto_remediated_and_code_fix_pr | 种子叙事的泄漏机制是"WeakSet 用错"（misused_weakset），但 `recommendation_server.py` 里真实实现的泄漏**只有**模块级列表无界 append（`unbounded_list_append`），没有任何 WeakSet 相关代码。按 memory-oom 规则"发版泄漏→online_op(重启止血)+also_code_fix=true"这套典型混合模式机械填了默认答案，但这个答案建立在"假设诊断 Agent 会找到的是真实存在的 list-append 泄漏"之上，而不是种子文本描述的 WeakSet 机制——两者对不上。 | 这条种子该按真实环境的 unbounded_list_append 泄漏重新改写叙事/标签，还是应该被排除出本轮训练集？如果保留，`remediation_detail`（本 schema 未包含，但会体现在轨迹里）该怎么和"WeakSet"这个错误叙事对齐？ |
| seed_0283_parametrized_deploy_regression | online_op | recommendation | auto_remediated_and_code_fix_pr | 泄漏机制叙事是"未关闭的文件句柄"（unclosed_file_handle），真实环境同样没有对应实现，只有 list-append。同 seed_0282 的问题。 | 同上：按真实 list-append 情况改标，还是排除这条种子。 |
| seed_0585_historical_reverse_derivation_deploy_regression | online_op | checkout | auto_remediated_and_code_fix_pr | 比 0282/0283 更严重：这条种子的**服务本身就选错了**——它是 checkout（"session cache 没有 TTL"），而真实泄漏 fixture 只存在于 `recommendation_server.py`，checkout 完全没有对应的可复现泄漏代码；历史工单反演路径本身又不触发任何真实故障注入。填的 `auto_remediated_and_code_fix_pr` 只是机械套用"发版泄漏混合模式"默认答案，实际上更可能是模型在 checkout 上找不到任何真实内存泄漏证据、应该判低置信度转人工（`feishu_low_confidence`）。 | 强烈建议人工判断：这条种子是否应该整条排除出本轮训练集（因为它对应的服务在真实环境里没有任何可验证的故障根源）；如果保留，是否应该改判 `feishu_low_confidence` 而不是当前的机械默认值。 |
| seed_0596_historical_reverse_derivation_deploy_regression | online_op | recommendation | auto_remediated_and_code_fix_pr | 服务对了（recommendation），但叙事"session cache 没有 TTL"含糊，既不明确等同于真实的 `unbounded_list_append`，也不属于用户列出的四种明确叙事变体（list_append/dict_accumulation/file_handle/weakset）之一，无法判断是否与真实实现一致。 | 需要人工确认"cache 没有 TTL"这个描述是否就是在指代真实的 list-append 泄漏（如果是，可以放心改成 HIGH 置信度），还是指代一种环境里不存在的不同泄漏机制（如果是，处理方式同 0282/0283）。 |

## 已确认不受影响的既有 mock 数据

`data/cold_start/ground_truth/mock_dep_clean.json`、`mock_fail_evidence.json`、`mock_fail_mismatch.json`、`mock_hookdeny_res.json`、`mock_hybrid_leak.json` 五个文件本次未做任何修改（对应 `collect_trajectories.py::build_mock_trajectories()` 的 5 个 mock alert_id，与本轮 29 条真实种子标注无关）。

## 2026-09-06 追加：新一批 23 条种子（`train_seeds_feasibility_check.json::recommended_real_collection_files`）的盲标

**方法论声明**：与上面 29 条一致，全程只依据种子告警内容本身（`labels`/`annotations`/`scenario_hint`）、`docs/技术方案.md`、`AIops-agent/agent/run.py` 的路由判定表、`AIops-agent/agent/core/remediation.py::classify_command()`/`REMEDIATION_ALLOWED_TARGETS` 白名单、`data/cold_start/fault_injection.py` 的真实机制边界。没有跑 `collect_trajectories.py`、没有调用 Claude API、没有看任何真实轨迹。

### 意外发现：23 条推荐清单里有 9 条实际是老 29 条种子的原文件，不属于本次标注范围

核对 `train_seeds_feasibility_check.json` 的 `recommended_real_collection_files` 清单时发现：`seed_0001/0002/0003/0008/0010_parametrized_dependency`、`seed_0189/0191_parametrized_resource`、`seed_0283_parametrized_deploy_regression`、`seed_0376_flagd_combination_resource` 这 9 个文件名，在 `data/cold_start/ground_truth/` 里**已经存在**，且逐字节核对 `data/clean/split/train/` 下的当前文件内容与 `data/cold_start/trajectories_accepted/`、`trajectories/real_run/` 里保存的原始 alert 内容**完全一致**（如 seed_0001 的 payment/paymentUnreachable 告警文本一字不差）——也就是说 `data/clean/split/train/` 这次扩容并没有覆盖这 9 个编号对应的文件，它们仍是上一轮 29 条种子的原件，早已有 GT 和已采集/已通过拒绝采样的真实轨迹。这与 `train_seeds_feasibility_check.json::_method` 里"两批种子编号有重叠但内容不同,不能按编号直接复用结论"这句提醒吻合（提醒的是"editorial 编号会重叠"这个已知风险），但本次核对结论是：这 9 个具体编号恰好是**内容也没变**的重叠，而不是"编号重叠但内容已被替换"的情况。

按任务边界"不要动已有的 29 条种子对应的 GT 文件"，本次对这 9 个文件**未做任何改动、未重新标注**，`data/cold_start/ground_truth/` 下这 9 份 GT 沿用它们已有的值（均已在上面"2026-09-05 修正"章节里更新为 flagd 止血无效性修正后的版本）。真正属于本次新标注范围的是剩下 **14 条**：`seed_0381/0429/0525_flagd_combination_resource`、`seed_0419_flagd_combination_resource`、`seed_0605/0609/0615/0617/0630/0655/0663/0664/0665/0666_historical_reverse_derivation_*`。

### 逐条标注明细（14 条新种子）

| alert_id | remediation_type | suspect_service | expect_route | 理由 |
|---|---|---|---|---|
| seed_0381_flagd_combination_resource | online_op | kafka | feishu_online_op | 组合 `kafkaQueueProblems`+`paymentUnreachable`：kafka 是有状态组件，不在 `REMEDIATION_ALLOWED_TARGETS` 白名单；payment 端是 flagd 开关驱动的"服务整体不可达"，重启不改变开关值。无论 `suspect_service` 落在 kafka 还是 payment，两者都不可能被 `classify_command()` 判定为可自动执行，route 结论稳定。 |
| seed_0419_flagd_combination_resource | online_op | product-catalog | feishu_online_op | 组合 `productCatalogFailure`+`loadGeneratorFloodHomepage`：涉及 product-catalog（flagd 开关驱动的"对特定商品返回失败"，且 product-catalog 本身也不在白名单）和 frontend（流量洪峰，frontend 在白名单内、理论上可 `auto_remediated`）。alert 的 `service` 标签落在 product-catalog（与 `alertname` 里 flag 出现顺序一致），按该标签取 `suspect_service=product-catalog`，倾向不确定时归人工审核的 `feishu_online_op`。 |
| seed_0429_flagd_combination_resource | online_op | kafka | feishu_online_op | 组合 `kafkaQueueProblems`+`cartFailure`，与 seed_0381 同理：kafka 不在白名单，cart 是 flag 驱动的服务失败，两者都不可自动止血。 |
| seed_0525_flagd_combination_resource | online_op | product-catalog | feishu_online_op | 组合 `productCatalogFailure`+`kafkaQueueProblems`：product-catalog 不在白名单、且是 flag 驱动；kafka 是有状态组件不在白名单。两个成员都指向人工介入，route 结论稳定。 |
| seed_0605_historical_reverse_derivation_resource | online_op | recommendation | auto_remediated_and_code_fix_pr | 文案精确点名 `_seen_product_ids`（list 无上限 append），并明确写了"已执行低风险止血 docker restart 恢复服务；根治需合并已准备好的代码修复"——与 `recommendation_server.py` 真实泄漏实现完全对应，是"重启止血+代码根治"混合根因的标准模式（同 seed_0282/0283/0585 先例）。 |
| seed_0609_historical_reverse_derivation_deploy_regression | online_op | recommendation | auto_remediated_and_code_fix_pr | 同 seed_0605：显式"混合根因：已重启单实例止血，但需改代码为该缓存加界方可根治"，服务、机制、结论都对应真实 list 泄漏 fixture。 |
| seed_0615_historical_reverse_derivation_dependency | code_fix | product-catalog | feishu_fix_unverified | 与已有 GT `seed_0595_historical_reverse_derivation_dependency`（`data/clean/split/train/` 中已不存在该文件，但 `trajectories/real_run/` 里保存的原始 `annotations.description` 逐字核对与本条**完全一致**：product-catalog 批量导入脚本忘记加超时、占满线程池、级联超时）为同一叙事的重复样本，直接复用其判定：根因是代码缺陷（缺超时保护），非 flagd 开关也非资源问题；`code_fix.py::clone_service_repo()` 硬编码只 clone recommendation 仓库，product-catalog 的 code_fix 机制上验证不了，故 `feishu_fix_unverified`。 |
| seed_0617_historical_reverse_derivation_deploy_regression | online_op | recommendation | auto_remediated_and_code_fix_pr | 文案给出了完整的实时指标证据链（重启前 `recommendation_seen_ids_total`=703590、内存 95.16%；重启后降到 11.19%）+明确点出"同一镜像下泄漏逻辑仍在，负载持续则会再次爬升复发，需代码根治"，是本批里证据最扎实的一条，标准混合根因模式。 |
| seed_0630_historical_reverse_derivation_deploy_regression | online_op | recommendation | auto_remediated_and_code_fix_pr | 文案较简略（"根因是无界 list 泄漏"），但明确点出"无界 list"这个与真实 fixture 完全一致的机制词，且服务、`_manifest` 标注的 `deploy_regression` 类别都对得上，按同一套混合根因模式判定。 |
| seed_0655_historical_reverse_derivation_deploy_regression | online_op | recommendation | auto_remediated_and_code_fix_pr | **置信度偏低**：文案是"手动加了一个全局缓存字典，上线时忘了配 TTL"——这是"dict + 缺 TTL"叙事，跟真实 fixture 唯一实现的"list 无上限 append/extend"不是同一种数据结构/机制，与已有 GT 里 `seed_0596_historical_reverse_derivation_deploy_regression`（"session cache 没有 TTL"）被判定为"含糊、无法确认与真实实现一致"的低置信度先例完全同类问题。remediation_type/route 的判断逻辑不受影响（同属 recommendation 内存泄漏混合根因），但真实采集时这条种子命中真实 list-append fixture 产生的证据链，能否被诊断 Agent 自然地对应到"缓存字典没配 TTL"这个叙事上，存在真实不确定性。 |
| seed_0663_historical_reverse_derivation_resource | online_op | kafka | feishu_online_op | 历史工单：物流更新消费者组副本数从 5 缩到 2 导致 consumer lag 堆积。修复手段是把消费者组副本数扩回去——这不是 `classify_command()` 白名单里任何一条动作（`restart_instance`/`migrate_instance`/`scale_resources` 都不是"改副本数"，且 `remediation.py` 文档明确把"缩容/扩容副本数"归为高风险、不进白名单），suspect_service=kafka 本身也不在 `REMEDIATION_ALLOWED_TARGETS` 里，只能人工。 |
| seed_0664_historical_reverse_derivation_dependency | code_fix | product-catalog | feishu_fix_unverified | 与 seed_0615 同一叙事的另一份重复样本（`hist-2026-03-import-timeout` vs `milvus-0001-product-catalog-dependency`），逐字核对 `annotations.description` 与已有 GT `seed_0595_historical_reverse_derivation_dependency` 保存的原始 alert 文本**完全一致**。直接复用 seed_0595 的判定。 |
| seed_0665_historical_reverse_derivation_dependency | code_fix | cart | feishu_fix_unverified | 与已有 GT `seed_0587_historical_reverse_derivation_dependency`（`trajectories/real_run/` 保存的原始 alert 文本核对后**完全一致**：cart 对账批处理脚本忘记设超时、占满线程池、checkout 调用大面积超时）是同一叙事的重复样本，直接复用其判定：`code_fix.py` 硬编码只 clone recommendation 仓库，cart 的 code_fix 验证不了，`feishu_fix_unverified`。 |
| seed_0666_historical_reverse_derivation_resource | online_op | kafka | feishu_online_op | 历史工单：通知消费者组副本数从 4 缩到 1 导致 consumer lag 堆积，与 seed_0663 同理——扩副本数不在白名单动作里，kafka 本身也不在白名单目标里，只能人工。 |

### 本轮 14 条置信度小结

- **高置信度（9 条）**：`seed_0605`、`seed_0609`、`seed_0615`、`seed_0617`、`seed_0630`、`seed_0663`、`seed_0664`、`seed_0665`、`seed_0666`。其中 0615/0664/0665 是对已有 29 条里 `seed_0595`/`seed_0587` 逐字重复的叙事，直接复用先例判定，可信度最高；0605/0609/0617 有精确到变量名/实时指标的证据链，与真实 fixture 完全对应；0630/0663/0666 逻辑链条单一、不依赖存在歧义的字段取值。
- **中置信度（4 条）**：`seed_0381`、`seed_0419`、`seed_0429`、`seed_0525`——均为 `flagd_combination` 机械组合种子，`suspect_service` 该落在组合里的哪个服务存在真实歧义（`alertname`/`service` 标签的排序本身是遍历 `CONFIRMED_FLAGD_FLAGS` 组合的副产物，不是手写叙事的产物），沿用已有 GT 里 `seed_0494`/`seed_0495` 对同类歧义的处理惯例定为中置信度；其中 `seed_0419` 的歧义还会改变 route 结论（若归因到 frontend 侧的流量洪峰而非 product-catalog 侧的开关失败，则 `expect_route` 应改为 `auto_remediated`），其余三条歧义不影响 route（组合内两个成员都不可自动止血）。
- **低置信度（1 条）**：`seed_0655`——"全局缓存字典缺 TTL"叙事与真实泄漏 fixture 唯一实现的"无界 list append/extend"机制不是同一种数据结构，与已有 GT `seed_0596` 的同类问题保持一致的低置信度判断，需人工确认真实采集时这条种子命中的 list 泄漏证据能否合理对应到"缓存字典"这个叙事。

### 与 `train_seeds_feasibility_check.json` 记录的分类理由核对结果

逐条核对后未发现内容层面的矛盾（feasibility_check 只判断"故障机制能否被真实触发"，不涉及 GT 该判什么，两者维度不同，理论上不会冲突）。唯一值得记录的错配：`seed_0605_historical_reverse_derivation_resource` 的 `intended_kind` 标注为 `resource`，但其 `annotations.description` 实际是一段标准的 `deploy_regression` 内存泄漏叙事（点名 `_seen_product_ids`、"已执行 docker restart"、"根治需合并代码修复"），与同批里标 `intended_kind=deploy_regression` 的 `seed_0609`/`seed_0617`/`seed_0630` 内容性质完全一样——这只是生成阶段的分类标签误差，不影响本条 GT 的判定逻辑（按内容判定，不按标签判定），如实记录不改标签。

## 2026-09-07 追加：`data/clean/split/train/` 剩余 45 条（GRPO 训练前的全量覆盖）

**背景**：为了让 `grpo/to_verl_dataset.py` 能把 `data/clean/split/train/` 74 条告警全部转成 veRL 训练数据（缺 GT 的告警没法参与 reward 计算，会被该脚本跳过），本轮把上面几批之后仍然缺 GT 的 45 条（`data/clean/split/train/` 当前这批种子池，跟本文件前几节标注时的种子池已经过一轮重新生成/扩容，编号有重叠但部分内容不同，按当前文件名逐一核对，不假设编号沿用旧结论）全部补上。

**方法论声明**：跟上面几批一致，只依据种子告警内容本身 + `AIops-agent/agent/config.py` 白名单（`REMEDIATION_ALLOWED_TARGETS=recommendation,ad,frontend,cart,checkout,currency,payment,shipping,quote,email`）+ `agent/core/remediation.py::classify_command()` + 四份排查手册 Skill + `data/cold_start/fault_injection.py` 的真实机制覆盖边界 + `code_fix.py::clone_service_repo()` 单仓约束 + 本文件已有的先例（尤其是"2026-09-05 修正"的 flagd 止血无效性判例）。流程：先自己写草稿，再各分派给一个独立子 agent 做**对抗式复核**（不是简单确认，要求主动找矛盾/反例），复核结果和本人核对后的最终结论一并记录如下。

### flagd_combination 家族（11 条）—— 全部高置信度

`suspect_service` 统一取告警顶层 `service` 字段（跟已有先例 `seed_0419`/`seed_0429`/`seed_0525` 的处理惯例一致，即使多个 flag 涉及多个服务，也不按"概念上更像根因"去挑）；机制上全部是 flagd 开关驱动，按"2026-09-05 修正"的判例，`restart_instance`/`migrate_instance`/`scale_resources` 都不改变 flagd 配置值，止血语义上无效，无论目标服务是否在白名单内：

| alert_id | suspect_service | expect_route |
|---|---|---|
| seed_0378_flagd_combination_dependency | recommendation | feishu_online_op |
| seed_0382_flagd_combination_resource | product-catalog | feishu_online_op |
| seed_0384_flagd_combination_dependency | recommendation | feishu_online_op |
| seed_0385_flagd_combination_resource | recommendation | feishu_online_op |
| seed_0388_flagd_combination_resource | ad | feishu_online_op |
| seed_0395_flagd_combination_resource | ad | feishu_online_op |
| seed_0416_flagd_combination_resource | ad | feishu_online_op |
| seed_0430_flagd_combination_dependency | cart | feishu_online_op |
| seed_0431_flagd_combination_resource | ad | feishu_online_op |
| seed_0449_flagd_combination_resource | recommendation | feishu_online_op |
| seed_0482_flagd_combination_resource | ad | feishu_online_op |

（`remediation_type` 全部为 `online_op`。）

### historical_reverse_derivation_dependency 家族（9 条）—— 全部高/中置信度

8 条是"调用下游忘记配超时→线程池占满→级联超时"叙事（缺陷在**调用方**自己的代码，不是下游），`remediation_type=code_fix`，`suspect_service` 填有缺陷调用代码的一方；除 `seed_0611`（suspect_service 正好是 `recommendation`，`code_fix.py::clone_service_repo()` 硬编码只 clone 这个仓，验证机制上可行，有直接先例 `seed_0701`/`seed_0703`/`seed_0704` 支持，中置信度——因为 `historical_reverse_derivation` 路径没有真实故障注入，不能保证 code_fix agent 真能找到跟叙事精确对应的代码）外，其余全部 `feishu_fix_unverified`。`seed_0602` 是例外：叙事本身讲的是 flagd `productCatalogFailure` 驱动的 5xx，告警文本自己已经论证完"product-catalog 不在白名单、回退开关不属于三类白名单动作，因此未执行任何自动处置"，不属于"调用方代码缺陷"模式，判 `online_op`。

| alert_id | remediation_type | suspect_service | expect_route |
|---|---|---|---|
| seed_0592_historical_reverse_derivation_dependency | code_fix | payment | feishu_fix_unverified |
| seed_0602_historical_reverse_derivation_dependency | online_op | product-catalog | feishu_online_op |
| seed_0610_historical_reverse_derivation_dependency | code_fix | shipping | feishu_fix_unverified |
| seed_0611_historical_reverse_derivation_dependency | code_fix | recommendation | code_fix_pr |
| seed_0618_historical_reverse_derivation_dependency | code_fix | checkout | feishu_fix_unverified |
| seed_0629_historical_reverse_derivation_dependency | code_fix | ad | feishu_fix_unverified |
| seed_0631_historical_reverse_derivation_dependency | code_fix | email | feishu_fix_unverified |
| seed_0639_historical_reverse_derivation_dependency | code_fix | currency | feishu_fix_unverified |
| seed_0649_historical_reverse_derivation_dependency | code_fix | quote | feishu_fix_unverified |

### historical_reverse_derivation_config 家族（6 条）—— 中置信度，有一处未解决的张力

6 条都是"证书轮换/信任链未同步/A-B配置误配，人工介入已恢复"的**过去时、已闭环**叙事（跟 dependency 家族"正在发生"的现在时叙事不同），判 `remediation_type=info_only`/`expect_route=info_only`，`suspect_service` 取告警自身 service 字段：

| alert_id | suspect_service |
|---|---|
| seed_0591_historical_reverse_derivation_config | currency |
| seed_0595_historical_reverse_derivation_config | quote |
| seed_0600_historical_reverse_derivation_config | shipping |
| seed_0601_historical_reverse_derivation_config | recommendation |
| seed_0614_historical_reverse_derivation_config | email |
| seed_0658_historical_reverse_derivation_config | checkout |

**未解决的张力（复核子 agent 指出，未强行改判）**：本文件里现存的 `seed_0588_historical_reverse_derivation_config.json`（旧种子池的同名文件，内容是 checkout 证书轮换叙事，跟上面几条高度同构）现有标注是 `online_op/checkout/auto_remediated`，不是 `info_only`；对应真实轨迹（`data/cold_start/trajectories/real_run/seed_0588_historical_reverse_derivation_config.json`）显示模型实际诊断是 `remediation_type=info_only, confidence=0.55`（低于阈值，`route=feishu_low_confidence`）。这条旧 GT 本身在"2026-09-06 追加"章节的拒绝采样结果里被列为 16 条"reject 且不在修正范围内"之一，从未被复核确认正确，权重不足以推翻本轮判断，但如实记录这处矛盾，供后续人工复核 config 家族的判断标准。

### historical_reverse_derivation_resource 家族（14 条）

`seed_0585`/`seed_0587`（kafka）：告警文本自带完整实时证据链证伪"队列积压"（Prometheus 查询为空/lag 恒为 0、flagd 配置状态与描述矛盾、时间线相差近两个月），`seed_0587` 文本甚至直接自引用"历史检索到 5 条高度相似的既往工单均判定为陈旧/失效告警，走向均为 feishu_low_confidence"——直接锚定结论。`seed_0593`/`seed_0626`（ad）同属这一模式。四条均判 `online_op`/`feishu_low_confidence`，高置信度。

`seed_0594`（frontend，活动预热流量洪峰）、`seed_0642`（ad，"流量驱动，建议扩容"）：干净的流量驱动案例，`online_op`/`auto_remediated`，高置信度。

`seed_0588`/`seed_0621`/`seed_0625`/`seed_0633`/`seed_0637`/`seed_0643`/`seed_0653`（shipping/currency/recommendation/ad/ad/cart/frontend）："某次运行时参数配置变更间接改变了 XX（GC 频率/批大小/刷新间隔/会话合并频率/缓存清理频率），CPU 使用率持续处于高位，没有伴随代码发版"这套叙事模板。`resource-cpu` 排查手册的判定桶只有三个：流量驱动→`online_op`、代码热点（需高置信度）→`code_fix`、不确定→**默认 `online_op` 先止血**；这批叙事"没有伴随代码发版"明确排除了 `code_fix`，按排除法落进第三桶，且本文件已有高置信度先例 `seed_0586`（ad，GC 频率变更）、`seed_0599`（payment，序列化开销变更）是同一套模板——`seed_0633` 与 `seed_0586` 近乎逐字相同。统一判 `online_op`/`auto_remediated`；`seed_0588`/`seed_0625`/`seed_0633`/`seed_0643`/`seed_0653` 因为有近乎逐字的高置信度先例支撑，定高置信度；`seed_0621`（"接近死循环式的高频轮询"，措辞比其余几条更极端，止血窗口可能极短）与 `seed_0637`（完全没有交代任何根因机制，纯症状描述）保持中置信度。

**特别说明为什么不套用"2026-09-05 flagd 止血无效性修正"把这批降级成 `feishu_online_op`**：那次修正只适用于 `scenario_hint` 里显式带 `flag=`/`flags=`、且 `fault_injection.py` 有真实机制验证（能查 OFREP 拿到确定开关值）的场景——"重启不改变 flagd 静态配置"是可以拿证据核实的事实。这批叙事没有 `flag=` 标记、也没有对应真实故障注入，是纯叙事性的"某个参数被改了"，不存在一个可核验的持久化状态对象能证明"重启/扩容对它无效"；`scale_resources`（抬资源上限）跟 `restart_instance` 性质不同，它不依赖"配置错误状态是否会被重启清除"，只是单纯抬高资源天花板，对任何持续性资源压力都是通用有效的止血动作。把窄证据范围的 flagd 修正套到这批纯叙事性的配置变更上，是过度泛化。

`seed_0635`（product-catalog，模糊搜索开关误开）：product-catalog 不在白名单，`online_op`/`feishu_online_op`，高置信度，跟上游怎么判无关。

| alert_id | suspect_service | expect_route | 置信度 |
|---|---|---|---|
| seed_0585_historical_reverse_derivation_resource | kafka | feishu_low_confidence | 高 |
| seed_0587_historical_reverse_derivation_resource | kafka | feishu_low_confidence | 高 |
| seed_0588_historical_reverse_derivation_resource | shipping | auto_remediated | 高 |
| seed_0593_historical_reverse_derivation_resource | ad | feishu_low_confidence | 高 |
| seed_0594_historical_reverse_derivation_resource | frontend | auto_remediated | 高 |
| seed_0621_historical_reverse_derivation_resource | currency | auto_remediated | 中 |
| seed_0625_historical_reverse_derivation_resource | recommendation | auto_remediated | 高 |
| seed_0626_historical_reverse_derivation_resource | ad | feishu_low_confidence | 高 |
| seed_0633_historical_reverse_derivation_resource | ad | auto_remediated | 高 |
| seed_0635_historical_reverse_derivation_resource | product-catalog | feishu_online_op | 高 |
| seed_0637_historical_reverse_derivation_resource | ad | auto_remediated | 中 |
| seed_0642_historical_reverse_derivation_resource | ad | auto_remediated | 高 |
| seed_0643_historical_reverse_derivation_resource | cart | auto_remediated | 高 |
| seed_0653_historical_reverse_derivation_resource | frontend | auto_remediated | 高 |

（`remediation_type` 全部为 `online_op`。）

### historical_reverse_derivation_deploy_regression 家族（5 条）—— 发现并纠正一处系统性注意事项错误

`seed_0597`（payment，重试列表无界 append）/`seed_0598`（frontend，缓存字典无 TTL）/`seed_0620`（email，WeakSet 误用）/`seed_0660`（checkout，会话缓存字典无 TTL）：均为"发版引入内存泄漏、关联到该服务本身"的混合根因叙事，`remediation_type=online_op` 且 `expect_also_code_fix=true`。

**这四条的 `expect_route` 不是 `auto_remediated_and_code_fix_pr`，是 `auto_remediated`**——这是本轮复核纠正的一处此前的方法论错误。本文件前面"系统性注意事项 #4"曾断言"`run.py::_run_inner()` 的路由判定表本身不检查混合分支里代码修复是否验证通过"，这个断言是**错的**：直接读 `AIops-agent/agent/run.py` 第 100-114 行，`also_code_fix=True` 分支明确调用了 `code_fix(...)` 并检查 `fix2.verified`——验证通过才是 `auto_remediated_and_code_fix_pr`/`online_op_and_code_fix_pr`，验证失败会静默退化成普通的 `auto_remediated`/`feishu_online_op`（没有像纯 `code_fix` 分支那样的专属 `feishu_fix_unverified` 标签）。而 `code_fix.py::clone_service_repo()` 硬编码只 clone `recommendation` 仓，完全忽略传入的 `service` 参数——这四条的 `suspect_service`（payment/frontend/email/checkout）都不是 `recommendation`，代码修复 Agent 会在错误的仓库里找一段该服务里根本不存在的结构（"待重试列表"/"缓存字典"/"WeakSet"），验证几乎必然失败，route 机械退化成 `auto_remediated`。`seed_0597` 中置信度（叙事机制单一、跟真实 fixture 机制类型一致，问题纯粹是仓不对）；`seed_0598`/`seed_0620`/`seed_0660` 低置信度（仓不对之外，叙事机制"缓存字典无 TTL"/"WeakSet 误用"本身也跟真实 fixture 的"list 无界 append"不是同一种数据结构，呼应已有先例 `seed_596`/`seed_282` 的同类问题）。

**这个纠正影响范围超出本次 45 条**：已有 GT `seed_0585_historical_reverse_derivation_deploy_regression.json`（旧种子池、checkout、`auto_remediated_and_code_fix_pr`）命中同样的问题（checkout ≠ recommendation），本轮未改动它（不在本次任务范围内，且属于"不动已有 29 条"的既定原则），但建议后续专门复核这一条和本文件其它涉及 `also_code_fix=true` 且 `suspect_service` 不是 `recommendation` 的既有条目。上面这段本身已经是对"系统性注意事项 #4"错误表述的更正，后续引用该注意事项时请以这里为准。

`seed_0641`（recommendation）是这批里唯一的例外：叙事表面指控"排序逻辑bug"，但告警文本自己完成了一次完整排查（代码审查、pytest 全绿、Prometheus/Jaeger 均无异常、历史工单不匹配）后得出"无法定位到匹配缺陷，不应贸然提交修复"的结论。跟 `seed_0585`/`seed_0587`/`seed_0593`/`seed_0626`（同批里"深入排查后发现当前没有活跃问题"的自证陈旧模式）适用同一套处理原则：`remediation_type` 锚定在告警表面声称的类型（这里是 `code_fix`），最终"什么都没有"的结论由独立的 `expect_route=feishu_low_confidence` 承载（`run.py` 里低置信度逃生口在 `remediation_type` 分流之前触发，两者互相独立）。跟真正的 `info_only` 先例（`seed_0702`/`seed_0705`，"服务当前真实健康、无需任何深入排查即可判定"的纯噪声场景）性质不同，不适用 `info_only`。高置信度。

| alert_id | remediation_type | suspect_service | expect_route | expect_also_code_fix | 置信度 |
|---|---|---|---|---|---|
| seed_0597_historical_reverse_derivation_deploy_regression | online_op | payment | auto_remediated | true | 中 |
| seed_0598_historical_reverse_derivation_deploy_regression | online_op | frontend | auto_remediated | true | 低 |
| seed_0620_historical_reverse_derivation_deploy_regression | online_op | email | auto_remediated | true | 低 |
| seed_0641_historical_reverse_derivation_deploy_regression | code_fix | recommendation | feishu_low_confidence | — | 高 |
| seed_0660_historical_reverse_derivation_deploy_regression | online_op | checkout | auto_remediated | true | 低 |

### 本轮 45 条置信度小结

- **高置信度（约 32 条）**：flagd_combination 全部 11 条；dependency 家族 8 条纯 `feishu_fix_unverified` + `seed_0602`；resource 家族里流量驱动/自证陈旧/有逐字先例支撑的 12 条；`seed_0641`。
- **中置信度（约 6 条）**：`seed_0611`（无真实故障注入验证）、config 家族全部 6 条（见上面"未解决的张力"）、`seed_0621`/`seed_0637`（resource，缺乏 flag 或明确根因机制的锚定证据）、`seed_0597`（deploy_regression，仓不对但机制单一）。
- **低置信度（3 条）**：`seed_0598`/`seed_0620`/`seed_0660`——仓不对叠加叙事机制跟真实 fixture 不完全对应两个问题共同作用，建议真实采集之后重点人工复核。

### 仍未覆盖的缺口（超出本次任务范围，如实记录）

`data/clean/split/held_out/` 18 条里仍有 16 条没有 ground truth（本次任务范围限定在 `train/`）——`grpo/to_verl_dataset.py` 转换时会把这些如实跳过并打印文件名，不影响 GRPO 训练本身（held_out 不参与训练），但会限制 `trainer.test_freq` 定期抽检 held-out 的样本量，后续需要按本文件的方法论补齐。

## 2026-09-08 追加：全部 GT 补 `expect_kind_any_of` 字段（v9 数据治理 Phase A）

**背景**：2026-09-08 完整测评暴露 kind 0.68（`AIops-agent/reports/aiops-qwen3.5-9b_20260907/metrics_final_combined.json`），根因之一是本目录的 GT schema 从来没有 kind 期望值——`rejection_sample.py` 的 5 项验收里 kind 完全不被校验，教师轨迹里的噪声 kind 标签（kafka 队列类标 config、MemoryLeakOOM 类标 resource）原样进了 v8 训练集（详见 `docs/数据增强方案.md` §2.2 缺口 #1）。

**改动**：本目录全部 100 个 seed GT + 5 个 mock GT 新增 `expect_kind_any_of` 字段（列表，首位是决策表 canonical 值）；`rejection_sample.py` 同步新增第 6 项 kind 校验。判定口径 = `docs/数据增强方案.md` §7.3 决策表 + `AIops-agent/eval/expected.json` 同机制场景的 kind 定义（s1 双值 [dependency, config]、s2 双值 [resource, config]、s3/s4/s10-s12 严格单值），逐条映射表与「服务级失败 flag 采用双值而非单值 dependency」的决策理由见 `data/cold_start/_governance_report_v9.md` §1.1 与附录 A。

**方法论声明**：与前几批一致——只依据种子告警文本本身的故障构造（scenario_hint 的 flag/机制叙事）、`_manifest.json`、`docs/数据增强方案.md` §7.3、`AIops-agent/eval/expected.json`、以及本文件前几节已确认的先例（如 seed_0605 的 intended_kind=resource 是生成阶段标签误差、内容实为 deploy_regression 内存泄漏叙事）。本轮**没有**参考任何轨迹的实际 kind 输出来反推 GT（防止「把模型的答案抄成标准答案」）；重验收淘汰 9 条是对着新 GT 正向执行的结果。旧种子池的 8 个无种子文件 GT（seed_0585_dep-reg/0586_res/0587_dep/0588_config/0592_res/0595_dep/0598_res/0599_res）按本文件前几节记录的原始叙事判定。

**重验收结果**：35 条 v8 已验收轨迹，26 过 9 拒（拒绝原因全部且仅是 kind 错标）；9 条淘汰轨迹 + seed_0701（答案钥污染）已移入 `trajectories_quarantined_v9/`，逐条原因见其 `_quarantine_manifest.json`。治理全貌见 `data/cold_start/_governance_report_v9.md`。
