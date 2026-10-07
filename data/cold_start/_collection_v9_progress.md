# v9 数据增强采集进度（Phase C）

- 更新时间：2026-09-08T09:50:40Z（全部时间 UTC）
- 教师模型：`claude-4.8-opus-google`；后端：`https://codewizllmproxy.devops.xiaohongshu.com/llmadapterproxy/v3/anthropic`
- 累计花费：**$58.77**（预算上限 $60，$50 停机汇报）
- 采集轮次统计：见下表（round 列）

| 种子 | 家族 | 轮次 | 步数 | degraded | 验收 | 拒绝原因 | cost_usd | 累计$ | 时间 |
|---|---|---|---|---|---|---|---|---|---|
| `seed_0744_v9_aug_C1_dependency` | C1 | 1 | 0 | sdk_error:Exception | ACCEPTED |  | 0.0000 | 0.00 | 2026-09-08T04:48:27Z |
| `seed_0744_v9_aug_C1_dependency` | C1 | 1 | 9 | - | ACCEPTED |  | 0.4853 | 0.49 | 2026-09-08T05:10:49Z |
| `seed_0713_v9_aug_A1_resource` | A1 | 1 | 17 | - | REJECTED | ['步数 17 超过 expect_max_steps=12'] | 0.7498 | 1.24 | 2026-09-08T05:16:14Z |
| `seed_0714_v9_aug_A1_dependency` | A1 | 1 | 9 | - | ACCEPTED |  | 0.5848 | 1.82 | 2026-09-08T05:18:57Z |
| `seed_0715_v9_aug_A1_dependency` | A1 | 1 | 11 | - | ACCEPTED |  | 0.5428 | 2.36 | 2026-09-08T05:20:44Z |
| `seed_0716_v9_aug_A1_resource` | A1 | 1 | 17 | sdk_error:Exception | REJECTED | ["kind 不在期望集合: got='' expect_any_of=['resource', 'deploy_regression', 'config']", "remediation_type 不一致: got=None expect | 1.1056 | 3.47 | 2026-09-08T05:22:56Z |
| `seed_0717_v9_aug_A1_dependency` | A1 | 1 | 21 | - | REJECTED | ['污染终检：step9 观测带出 eval alerts scenario_hint（答案相邻）+ step3 RAG 命中测评耦合工单，按 seed_0701 先例隔离'] | 1.7019 | 5.17 | 2026-09-08T05:26:46Z |
| `seed_0718_v9_aug_A1_dependency` | A1 | 1 | 8 | - | ACCEPTED |  | 0.5427 | 5.71 | 2026-09-08T05:32:27Z |
| `seed_0719_v9_aug_A2_dependency` | A2 | 1 | 7 | - | ACCEPTED |  | 0.9622 | 6.67 | 2026-09-08T05:34:04Z |
| `seed_0720_v9_aug_A2_dependency` | A2 | 1 | 10 | - | ACCEPTED |  | 0.4881 | 7.16 | 2026-09-08T05:35:45Z |
| `seed_0721_v9_aug_A2_resource` | A2 | 1 | 9 | - | ACCEPTED |  | 0.8027 | 7.97 | 2026-09-08T05:38:01Z |
| `seed_0722_v9_aug_A2_config` | A2 | 1 | 17 | - | ACCEPTED |  | 0.9629 | 8.93 | 2026-09-08T05:40:06Z |
| `seed_0723_v9_aug_A3_dependency` | A3 | 1 | 8 | - | ACCEPTED |  | 0.4993 | 9.43 | 2026-09-08T05:44:05Z |
| `seed_0724_v9_aug_A3_resource` | A3 | 1 | 9 | - | ACCEPTED |  | 0.6948 | 10.12 | 2026-09-08T05:45:28Z |
| `seed_0725_v9_aug_A3_dependency` | A3 | 1 | 23 | - | ACCEPTED |  | 0.6540 | 10.78 | 2026-09-08T05:47:38Z |
| `seed_0726_v9_aug_B1a_resource` | B1a | 1 | 9 | - | REJECTED | ['confidence 越界: confidence=0.55 不在期望区间 [0.6, 0.95]', "suspect_service 不一致: got='accounting' expect_contains='kafka'", " | 0.6679 | 11.44 | 2026-09-08T05:50:10Z |
| `seed_0727_v9_aug_B1a_resource` | B1a | 1 | 9 | - | REJECTED | ["suspect_service 不一致: got='fraud-detection' expect_contains='kafka'"] | 0.6319 | 12.08 | 2026-09-08T05:52:17Z |
| `seed_0728_v9_aug_B1a_resource` | B1a | 1 | 9 | - | REJECTED | ["suspect_service 不一致: got='fraud-detection' expect_contains='kafka'"] | 0.8113 | 12.89 | 2026-09-08T05:54:30Z |
| `seed_0729_v9_aug_B1a_resource` | B1a | 1 | 12 | - | ACCEPTED |  | 0.7037 | 13.59 | 2026-09-08T05:56:23Z |
| `seed_0724_v9_aug_A3_resource` | A3 | 1 | 11 | - | ACCEPTED |  | 0.7223 | 14.31 | 2026-09-08T06:02:45Z |
| `seed_0726_v9_aug_B1a_resource` | B1a | 1 | 9 | - | REJECTED | ['confidence 越界: confidence=0.55 不在期望区间 [0.6, 0.95]', "suspect_service 不一致: got='accounting' expect_contains='kafka'", " | 0.5599 | 14.87 | 2026-09-08T06:06:02Z |
| `seed_0727_v9_aug_B1a_resource` | B1a | 1 | 7 | - | REJECTED | ["suspect_service 不一致: got='fraud-detection' expect_contains='kafka'"] | 0.5259 | 15.40 | 2026-09-08T06:08:48Z |
| `seed_0728_v9_aug_B1a_resource` | B1a | 1 | 10 | - | REJECTED | ["suspect_service 不一致: got='fraud-detection' expect_contains='kafka'"] | 0.8180 | 16.22 | 2026-09-08T06:10:09Z |
| `seed_0729_v9_aug_B1a_resource` | B1a | 1 | 16 | - | ACCEPTED |  | 0.9813 | 17.20 | 2026-09-08T06:13:03Z |
| `seed_0730_v9_aug_B1a_resource` | B1a | 1 | 18 | - | REJECTED | ['confidence 越界: confidence=0.55 不在期望区间 [0.6, 0.95]', "route 不一致: got='feishu_low_confidence' expect='feishu_online_op'" | 1.0816 | 18.28 | 2026-09-08T06:16:51Z |
| `seed_0731_v9_aug_B1a_config` | B1a | 1 | 4 | - | ACCEPTED |  | 0.3443 | 18.62 | 2026-09-08T06:20:26Z |
| `seed_0732_v9_aug_B1a_resource` | B1a | 1 | 11 | - | REJECTED | ["suspect_service 不一致: got='flagd' expect_contains='kafka'", "route 不一致: got='info_only' expect='feishu_low_confidence'" | 0.9393 | 19.56 | 2026-09-08T06:21:21Z |
| `seed_0733_v9_aug_B1a_dependency` | B1a | 1 | 15 | - | ACCEPTED |  | 0.8372 | 20.40 | 2026-09-08T06:24:27Z |
| `seed_0734_v9_aug_B1b_resource` | B1b | 1 | 20 | - | ACCEPTED |  | 1.3402 | 21.74 | 2026-09-08T06:27:58Z |
| `seed_0735_v9_aug_B1b_resource` | B1b | 1 | 12 | - | REJECTED | ['confidence 越界: confidence=0.85 不在期望区间 [0.0, 0.59]', "缺少否定性/如实陈述: summary/evidence 缺少否定性/如实措辞（markers=['未能定位', '未找到', ' | 1.0836 | 22.83 | 2026-09-08T06:33:34Z |
| `seed_0736_v9_aug_B1b_resource` | B1b | 1 | 18 | - | REJECTED | ['confidence 越界: confidence=0.9 不在期望区间 [0.0, 0.59]', "suspect_service 不一致: got='fraud-detection' expect_contains='fronte | 1.0598 | 23.88 | 2026-09-08T06:36:12Z |
| `seed_0737_v9_aug_B1b_config` | B1b | 1 | 6 | - | REJECTED | ['confidence 越界: confidence=0.9 不在期望区间 [0.0, 0.59]', "缺少否定性/如实陈述: summary/evidence 缺少否定性/如实措辞（markers=['未能定位', '未找到', '无 | 0.4782 | 24.36 | 2026-09-08T06:40:10Z |
| `seed_0738_v9_aug_B2_dependency` | B2 | 1 | 7 | - | ACCEPTED |  | 0.4671 | 24.83 | 2026-09-08T06:41:36Z |
| `seed_0739_v9_aug_B2_config` | B2 | 1 | 11 | - | ACCEPTED |  | 0.7852 | 25.62 | 2026-09-08T06:42:40Z |
| `seed_0740_v9_aug_B2_resource` | B2 | 1 | 10 | - | ACCEPTED |  | 0.7797 | 26.40 | 2026-09-08T06:44:29Z |
| `seed_0741_v9_aug_B3_resource` | B3 | 1 | 9 | - | ACCEPTED |  | 0.6958 | 27.09 | 2026-09-08T06:46:43Z |
| `seed_0742_v9_aug_B3_dependency` | B3 | 1 | 18 | - | REJECTED | ["remediation_type 不一致: got='online_op' expect='info_only'", "route 不一致: got='feishu_online_op' expect='info_only'"] | 1.1920 | 28.28 | 2026-09-08T06:53:37Z |
| `seed_0743_v9_aug_B3_config` | B3 | 1 | 5 | - | ACCEPTED |  | 0.8591 | 29.14 | 2026-09-08T07:01:57Z |
| `seed_0745_v9_aug_C1_dependency` | C1 | 1 | 13 | - | REJECTED | ['步数 13 超过 expect_max_steps=12'] | 1.0146 | 30.16 | 2026-09-08T07:03:50Z |
| `seed_0746_v9_aug_C1_dependency` | C1 | 1 | 7 | - | ACCEPTED |  | 0.6385 | 30.80 | 2026-09-08T07:06:46Z |
| `seed_0747_v9_aug_C1_dependency` | C1 | 1 | 7 | - | ACCEPTED |  | 0.3493 | 31.14 | 2026-09-08T07:08:07Z |
| `seed_0748_v9_aug_C1_dependency` | C1 | 1 | 22 | sdk_error:Exception | REJECTED | ["kind 不在期望集合: got='' expect_any_of=['dependency', 'config']", "remediation_type 不一致: got=None expect='online_op'", 'con | 1.2928 | 32.44 | 2026-09-08T07:09:36Z |
| `seed_0749_v9_aug_C1_config` | C1 | 1 | 4 | - | ACCEPTED |  | 0.3718 | 32.81 | 2026-09-08T07:15:27Z |
| `seed_0750_v9_aug_C1_config` | C1 | 1 | 9 | - | ACCEPTED |  | 0.5305 | 33.34 | 2026-09-08T07:16:36Z |
| `seed_0751_v9_aug_C1_dependency` | C1 | 1 | 14 | - | REJECTED | ['步数 14 超过 expect_max_steps=12'] | 0.9490 | 34.29 | 2026-09-08T07:18:49Z |
| `seed_0752_v9_aug_D1_deploy_regression` | D1 | 1 | 7 | - | ACCEPTED |  | 0.4719 | 34.76 | 2026-09-08T07:21:36Z |
| `seed_0753_v9_aug_D1_deploy_regression` | D1 | 1 | 7 | - | ACCEPTED |  | 0.4875 | 35.25 | 2026-09-08T07:22:42Z |
| `seed_0754_v9_aug_D1_deploy_regression` | D1 | 1 | 12 | - | REJECTED | ["remediation_type 不一致: got='online_op' expect='code_fix'"] | 0.4489 | 35.70 | 2026-09-08T07:24:16Z |
| `seed_0755_v9_aug_D1_deploy_regression` | D1 | 1 | 10 | - | ACCEPTED |  | 0.5493 | 36.25 | 2026-09-08T07:26:26Z |
| `seed_0756_v9_aug_D2_deploy_regression` | D2 | 1 | 9 | - | ACCEPTED |  | 0.5126 | 36.76 | 2026-09-08T07:28:05Z |
| `seed_0757_v9_aug_D2_deploy_regression` | D2 | 1 | 6 | - | ACCEPTED |  | 0.3113 | 37.07 | 2026-09-08T07:29:49Z |
| `seed_0758_v9_aug_D3_deploy_regression` | D3 | 1 | 9 | - | ACCEPTED |  | 0.4801 | 37.55 | 2026-09-08T07:31:18Z |
| `seed_0759_v9_aug_D3_deploy_regression` | D3 | 1 | 10 | - | ACCEPTED |  | 0.3451 | 37.90 | 2026-09-08T07:32:54Z |
| `seed_0713_v9_aug_A1_resource` | A1 | 2 | 18 | sdk_error:Exception | REJECTED | ["kind 不在期望集合: got='' expect_any_of=['resource', 'deploy_regression', 'config']", "remediation_type 不一致: got=None expect | 1.2532 | 39.15 | 2026-09-08T07:40:39Z |
| `seed_0716_v9_aug_A1_resource` | A1 | 2 | 13 | - | REJECTED | ['步数 13 超过 expect_max_steps=12'] | 0.9081 | 40.06 | 2026-09-08T07:46:39Z |
| `seed_0726_v9_aug_B1a_resource` | B1a | 2 | 14 | - | REJECTED | ["suspect_service 不一致: got='accounting' expect_contains='kafka'"] | 0.8417 | 40.90 | 2026-09-08T07:49:17Z |
| `seed_0727_v9_aug_B1a_resource` | B1a | 2 | 7 | - | REJECTED | ["suspect_service 不一致: got='fraud-detection' expect_contains='kafka'"] | 0.5626 | 41.46 | 2026-09-08T07:52:32Z |
| `seed_0728_v9_aug_B1a_resource` | B1a | 2 | 9 | - | REJECTED | ["remediation_type 不一致: got='info_only' expect='online_op'", 'confidence 越界: confidence=0.35 不在期望区间 [0.6, 0.95]', "route | 0.4548 | 41.92 | 2026-09-08T07:54:07Z |
| `seed_0730_v9_aug_B1a_resource` | B1a | 2 | 15 | - | ACCEPTED |  | 0.9633 | 42.88 | 2026-09-08T07:56:19Z |
| `seed_0732_v9_aug_B1a_resource` | B1a | 2 | 12 | - | REJECTED | ["suspect_service 不一致: got='fraud-detection' expect_contains='kafka'", "route 不一致: got='feishu_online_op' expect='feishu | 1.0486 | 43.93 | 2026-09-08T07:59:11Z |
| `seed_0733_v9_aug_B1a_dependency` | B1a | 2 | 19 | - | REJECTED | ['confidence 越界: confidence=0.68 不在期望区间 [0.0, 0.6]', "route 不一致: got='info_only' expect='feishu_low_confidence'"] | 0.9232 | 44.85 | 2026-09-08T08:04:00Z |
| `seed_0735_v9_aug_B1b_resource` | B1b | 2 | 16 | - | REJECTED | ["kind 不在期望集合: got='deploy_regression' expect_any_of=['resource', 'config']", 'confidence 越界: confidence=0.86 不在期望区间 [0. | 0.9652 | 45.82 | 2026-09-08T08:07:51Z |
| `seed_0736_v9_aug_B1b_resource` | B1b | 2 | 13 | - | REJECTED | ['confidence 越界: confidence=0.72 不在期望区间 [0.0, 0.59]', "suspect_service 不一致: got='checkout' expect_contains='frontend'",  | 1.0182 | 46.83 | 2026-09-08T08:12:58Z |
| `seed_0737_v9_aug_B1b_config` | B1b | 2 | 9 | - | REJECTED | ['confidence 越界: confidence=0.9 不在期望区间 [0.0, 0.59]', "缺少否定性/如实陈述: summary/evidence 缺少否定性/如实措辞（markers=['未能定位', '未找到', '无 | 0.6149 | 47.45 | 2026-09-08T08:16:24Z |
| `seed_0745_v9_aug_C1_dependency` | C1 | 2 | 11 | - | ACCEPTED |  | 0.7586 | 48.21 | 2026-09-08T08:18:37Z |
| `seed_0748_v9_aug_C1_dependency` | C1 | 2 | 15 | - | REJECTED | ['confidence 越界: confidence=0.5 不在期望区间 [0.6, 1.0]', '步数 15 超过 expect_max_steps=12', "route 不一致: got='feishu_low_confiden | 1.0651 | 49.27 | 2026-09-08T08:20:27Z |
| `seed_0713_v9_aug_A1_resource` | A1 | 3 | 22 | - | REJECTED | ['步数 22 超过 expect_max_steps=12', "缺少否定性/如实陈述: summary/evidence 缺少否定性/如实措辞（markers=['无活跃', '噪声', '误报', '误传', '未发现', '未观测' | 1.2030 | 50.48 | 2026-09-08T08:37:52Z |
| `seed_0735_v9_aug_B1b_resource` | B1b | 3 | 18 | - | REJECTED | ['confidence 越界: confidence=0.72 不在期望区间 [0.0, 0.59]', "suspect_service 不一致: got='kafka' expect_contains='email'", "route | 1.2568 | 51.73 | 2026-09-08T09:09:23Z |
| `seed_0736_v9_aug_B1b_resource` | B1b | 3 | 22 | - | REJECTED | ["kind 不在期望集合: got='deploy_regression' expect_any_of=['resource', 'dependency', 'config']", 'confidence 越界: confidence=0 | 1.3129 | 53.05 | 2026-09-08T09:14:56Z |
| `seed_0737_v9_aug_B1b_config` | B1b | 3 | 17 | - | ACCEPTED |  | 1.1527 | 54.20 | 2026-09-08T09:20:31Z |
| `seed_0726_v9_aug_B1a_resource` | B1a | 3 | 8 | - | REJECTED | ["remediation_type 不一致: got='info_only' expect='online_op'", "route 不一致: got='info_only' expect='feishu_online_op'"] | 0.6816 | 54.88 | 2026-09-08T09:23:49Z |
| `seed_0727_v9_aug_B1a_resource` | B1a | 3 | 11 | - | REJECTED | ["suspect_service 不一致: got='fraud-detection' expect_contains='kafka'"] | 0.8696 | 55.75 | 2026-09-08T09:26:07Z |
| `seed_0728_v9_aug_B1a_resource` | B1a | 3 | 8 | - | REJECTED | ["remediation_type 不一致: got='info_only' expect='online_op'", 'confidence 越界: confidence=0.55 不在期望区间 [0.6, 0.95]', "route | 0.5374 | 56.29 | 2026-09-08T09:28:20Z |
| `seed_0732_v9_aug_B1a_resource` | B1a | 3 | 12 | - | ACCEPTED |  | 0.7508 | 57.04 | 2026-09-08T09:30:06Z |
| `seed_0733_v9_aug_B1a_dependency` | B1a | 3 | 18 | - | ACCEPTED |  | 0.9929 | 58.03 | 2026-09-08T09:32:44Z |
| `seed_0716_v9_aug_A1_resource` | A1 | 3 | 15 | - | REJECTED | ['步数 15 超过 expect_max_steps=12'] | 0.7384 | 58.77 | 2026-09-08T09:37:01Z |

**已验收通过 39 条 / 已拒绝 37 条**（未跑的不计）

---
## Phase C 采集最终汇总（2026-09-08）

- 采集总轮次：76 次真实 Opus API 调用（47 种子 × 1-4 轮）
- 总花费：**$58.77**（预算 $60 上限；$50 停机线触发后经协调员授权续跑保家族配额）
- 教师模型：claude-4.8-opus-google（真实代理路由；meta.usage 带 cache_creation/cache_read 字段、meta.cost_usd 实证）

| 家族 | 通过 | 拒绝 | 备注 |
|---|---|---|---|
| A1 | 3 | 3 | 0713 反模式超步数(17-22步)×3、0716 超步数(13-15步)、0717 污染隔离 |
| A2 | 4 | 0 | 全部通过（flagA 真开 + flagB 如实否定） |
| A3 | 3 | 0 | 全部通过（交人工语义 + 台账一致） |
| B1a | 3 | 3 | 教师对 kafka 类故障确定性把 suspect 标成消费者(fraud-detection/accounting)；0732/0733 按方案§4.2 降级为 B1b |
| B1b | 4 | 2 | 0734+0737+降级 0732/0733；教师用 OFREP 批量枚举开关，注入即被发现（重设计为不注入+不可证伪主张） |
| B2 | 3 | 0 | 全部通过 |
| B3 | 2 | 1 | 0742 教师把 pre-inject 新痕迹读成复发（叙事时延与痕迹时延不齐） |
| C1 | 6 | 2 | 步数门 8→12 校准后 6/8 |
| D1 | 3 | 1 | 0754 教师对 cache.py 给了 online_op 方案（方差，配额已满未重试） |
| D2 | 2 | 0 | 全部通过（bug 定位 pagination/cache/pricing 全命中） |
| D3 | 2 | 0 | 全部通过（对抗误导线索下仍正确指向真根因） |
| **合计** | **35** | **12** | 验收率 35/47 = 74% |

详见 `_v9_dataset_stats.md`（数据集统计 + 六道数据门）与 `_governance_report_v9.md`（Phase A 治理）。
