# v9 数据集统计（Phase C 组装产物）

- 轨迹总数：63（v8 治理基座 28 + v9 新增 35）
- 家族分布（轨迹数）：{"A1": 3, "A2": 4, "A3": 3, "B1a": 3, "B1b": 4, "B2": 3, "B3": 2, "C1": 6, "D1": 3, "D2": 2, "D3": 2, "v8_base": 28}
- remediation_type：{"online_op": 43, "info_only": 13, "code_fix": 7}
- route：{"feishu_online_op": 35, "info_only": 12, "None": 12, "feishu_low_confidence": 4}
- kind：{"config": 38, "deploy_regression": 14, "dependency": 4, "resource": 7}
- confidence：min=0.28 p50=0.88 max=0.97
- 步数：mean=10.59 min=2 p50=10 max=23（门：均值≤12）
- 前缀子样本：667 条（604 tool_call + 63 diagnosis）
- 最终 jsonl：aiops_cold_start_sft_v9.jsonl（667 行）

## 六道数据门
- [PASS] 污染扫描："0 命中"
- [PASS] 全量重验收(含kind)："63/63 全过"
- [PASS] malformed目标=0："0"
- [FAIL] 家族配额：{"B1a": "3/6", "B1b": "4/3", "A1": "3/5", "D1": "3/3", "C1": "6/6", "B1b+A1低置信目标": "4/5"}
- [PASS] 步数均值≤12："10.59"
- [PASS] 总验收率≥55%："35/47 = 74%"

## v9 新增轨迹明细（35 条，按家族）
- `seed_0714_v9_aug_A1_dependency` [A1] steps 9→9 conf=0.72 kind=deploy_regression route=info_only
- `seed_0715_v9_aug_A1_dependency` [A1] steps 11→10 conf=0.78 kind=config route=info_only
- `seed_0718_v9_aug_A1_dependency` [A1] steps 8→8 conf=0.62 kind=config route=info_only
- `seed_0719_v9_aug_A2_dependency` [A2] steps 7→7 conf=0.90 kind=config route=feishu_online_op
- `seed_0720_v9_aug_A2_dependency` [A2] steps 10→10 conf=0.82 kind=config route=feishu_online_op
- `seed_0721_v9_aug_A2_resource` [A2] steps 9→9 conf=0.72 kind=config route=feishu_online_op
- `seed_0722_v9_aug_A2_config` [A2] steps 17→16 conf=0.86 kind=config route=feishu_online_op
- `seed_0723_v9_aug_A3_dependency` [A3] steps 8→8 conf=0.95 kind=config route=feishu_online_op
- `seed_0724_v9_aug_A3_resource` [A3] steps 11→11 conf=0.78 kind=resource route=feishu_online_op
- `seed_0725_v9_aug_A3_dependency` [A3] steps 23→23 conf=0.95 kind=config route=feishu_online_op
- `seed_0729_v9_aug_B1a_resource` [B1a] steps 16→15 conf=0.75 kind=resource route=feishu_online_op
- `seed_0730_v9_aug_B1a_resource` [B1a] steps 15→15 conf=0.78 kind=resource route=feishu_online_op
- `seed_0731_v9_aug_B1a_config` [B1a] steps 4→4 conf=0.92 kind=config route=feishu_online_op
- `seed_0732_v9_aug_B1a_resource` [B1b] steps 12→12 conf=0.50 kind=resource route=feishu_low_confidence
- `seed_0733_v9_aug_B1a_dependency` [B1b] steps 15→13 conf=0.50 kind=config route=feishu_low_confidence
- `seed_0734_v9_aug_B1b_resource` [B1b] steps 20→20 conf=0.40 kind=config route=feishu_low_confidence
- `seed_0737_v9_aug_B1b_config` [B1b] steps 17→17 conf=0.28 kind=resource route=feishu_low_confidence
- `seed_0738_v9_aug_B2_dependency` [B2] steps 7→6 conf=0.85 kind=config route=info_only
- `seed_0739_v9_aug_B2_config` [B2] steps 11→11 conf=0.82 kind=config route=info_only
- `seed_0740_v9_aug_B2_resource` [B2] steps 10→10 conf=0.80 kind=config route=info_only
- `seed_0741_v9_aug_B3_resource` [B3] steps 9→9 conf=0.90 kind=resource route=info_only
- `seed_0743_v9_aug_B3_config` [B3] steps 5→5 conf=0.90 kind=config route=info_only
- `seed_0744_v9_aug_C1_dependency` [C1] steps 9→9 conf=0.96 kind=config route=feishu_online_op
- `seed_0745_v9_aug_C1_dependency` [C1] steps 11→11 conf=0.95 kind=config route=feishu_online_op
- `seed_0746_v9_aug_C1_dependency` [C1] steps 7→7 conf=0.95 kind=dependency route=feishu_online_op
- `seed_0747_v9_aug_C1_dependency` [C1] steps 7→7 conf=0.90 kind=config route=feishu_online_op
- `seed_0749_v9_aug_C1_config` [C1] steps 4→4 conf=0.95 kind=config route=feishu_online_op
- `seed_0750_v9_aug_C1_config` [C1] steps 9→9 conf=0.90 kind=config route=feishu_online_op
- `seed_0752_v9_aug_D1_deploy_regression` [D1] steps 7→7 conf=0.97 kind=deploy_regression route=None
- `seed_0753_v9_aug_D1_deploy_regression` [D1] steps 7→7 conf=0.97 kind=deploy_regression route=None
- `seed_0755_v9_aug_D1_deploy_regression` [D1] steps 10→9 conf=0.96 kind=deploy_regression route=None
- `seed_0756_v9_aug_D2_deploy_regression` [D2] steps 9→9 conf=0.97 kind=deploy_regression route=None
- `seed_0757_v9_aug_D2_deploy_regression` [D2] steps 6→6 conf=0.96 kind=deploy_regression route=None
- `seed_0758_v9_aug_D3_deploy_regression` [D3] steps 9→9 conf=0.96 kind=deploy_regression route=None
- `seed_0759_v9_aug_D3_deploy_regression` [D3] steps 10→10 conf=0.93 kind=deploy_regression route=None
