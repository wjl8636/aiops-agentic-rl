# AIOps 诊断 Agent · SFT + Agentic RL 后训练 — 使用指南

> **这是什么**：以 [AIops-agent](../AIops-agent/)（一个基于 Claude Agent SDK、已经端到端跑通的 AIOps 故障诊断修复 Agent）为地基，往模型训练方向再做一层——把其中的**故障诊断处置 Agent** 的决策底座，从外部 Claude Opus 换成一个自己训练的 Qwen3.5-9B + LoRA 小模型。整条链路走 Cold Start SFT → GRPO 两阶段，核心是把 AIops-agent 现有代码里的判定逻辑（hook allow/deny、schema 校验、白名单命中、健康检查回归）直接改造成零标注的 dense reward。本仓库通过 **git submodule** 引入 `AIops-agent/`，只读消费它的 SDK 接口和判定信号，不改它的核心代码（唯一的改动是给 `Diagnosis` 加了 3 个可选字段，见下文）。

> **本仓库的真实状态，请在读任何一份文档前先看这一段**：
> - **代码与测试**：数据构造全流程代码（种子生成 → 清洗去重 → 冷启轨迹采集 → 渐进式前缀拆分）、reward 三层设计的完整实现、veRL 协议适配层、SFT/GRPO 的配置文件、评测 harness——这些都有对应的单元测试（见下文「如何跑测试」）。本仓的 `eval/run_baseline_comparison.py` + `data/clean/split/held_out`（22 条）承担 GRPO 训练期间的 periodic held-out 验证，工具端点走 `eval/model_endpoints.py` 里的 mock 后端，`aiops-agentic-rl/eval/reports/` 下存放的就是这一档产物。
> - **训练与部署**：真实 Qwen3.5-9B LoRA **Cold Start SFT 训练**、**15-epoch GRPO 训练**（产出 checkpoint `aiops-qwen3.5-9b-grpo-step15`，单卡 A800-80GB，实测约 12 小时）、**LoRA 合并 + vLLM 服务化部署** + 接入 AIops-agent 的真实调用链路，都已在租卡环境跑通（环境版本坑、协议适配细节见 [`训练踩坑记录.md`](训练踩坑记录.md)）。
> - **数据规模**：冷启轨迹的真实采集（真调 Claude Opus API + 真实 docker 环境 + 场景化故障自动注入）v9 阶段 47 seed × 76 次 Opus API 调用，通过率 74%（35/47 accepted），加上 v8_base 保留 28 条 = **63 条通过轨迹**、实花 **$58.77**（预算 $60）；前缀拆分成 **667 条 SFT 样本**（604 tool_call + 63 diagnosis）；切分成 **train 74 / held_out 22**。
> - **下游 headline**：由 **AIops-agent 自己的 `eval/run.py`**、在它 13 个真实 docker-compose 故障场景上跑出，报告见 AIops-agent 独立 checkout 的 `reports/{aiops-qwen3.5-9b-base-zeroshot_20260910, aiops-qwen3.5-9b-v9_20260908, aiops-qwen3.5-9b-grpo-step15_20260910, opus_20260819}/`：**zero-shot 25 rows service/kind/route = 0.48/0.52/0.36；v9-SFT 37 rows = 0.5676/0.6757/0.4595；GRPO step15 25 rows = 0.80/0.72/0.64；Opus 23 rows 全线约 0.913**。

## 阅读顺序建议

1. **先看这份 README**，搞清楚仓库结构和「真实 vs 目标」的边界。
2. 如果你买了这个项目想真的把训练闭环跑出来，看 [`复现指南.md`](复现指南.md)——从租哪档 GPU、仓库自带的种子/冷启数据怎么用、要不要扩量，一直到真实 SFT → GRPO → 评测。

## 目录结构

```
aiops-agentic-rl/
├── AIops-agent/          # git submodule：已独立交付的 AIOps 故障诊断修复 Agent（只读消费，见下）
├── data/
│   ├── seeds/            # 种子告警三路扩增（参数化 / flagd 组合 / 历史工单反演），v9 阶段 47 条新种子 + v8_base 保留 28 条
│   ├── clean/            # 质量过滤 + 双层去重（MD5 精确 + BGE 语义）+ 分层切分（train 74 / held_out 22）
│   ├── cold_start/       # 冷启轨迹采集（真实/mock 双模式）+ 渐进式前缀拆分（63 轨迹 → 667 样本）+ LLaMA-Factory 格式转换
│   └── ood_fixtures/     # 4 条 alert stub (TLS 证书过期/DNS 污染/慢查询/GC 长暂停) 作为格式参考
├── reward/                # 三层 reward 设计的完整实现：step-level / outcome-level / 细粒度信用分配 + 四类防 hacking，53 个单测全过
├── verl_adapter/          # AIops-agent 工具协议 ↔ veRL rollout 协议的适配层，含可真跑的 mock 工具后端 + 未执行的真实环境路径
├── sft/                   # LLaMA-Factory 的 Cold Start SFT 配置（LoRA r=16/alpha=32，字段名已对照 LLaMA-Factory 真实源码核实）
├── grpo/                  # veRL 的 GRPO 配置 + reward 路由（把 reward/ 的三层组合成 GRPO 训练环真正消费的单一信号）
├── eval/                  # 评测 harness：本仓 held_out 22 条 + mock 模型端点（GRPO 训练期 periodic 验证）；机制层消融脚本 2 个（no_prefix_split / no_credit_assignment）
├── deploy/                # LoRA merge (merge_lora.py) + vLLM 服务化 (vllm_serve.sh) + AIOPS_MODEL 切换 (switch_aiops_model.md)，已在真实 GPU/checkpoint 上跑通并接入 AIops-agent
├── smoke_test/            # 端到端 mock 烟雾测试，已有 run_smoke.py/run_smoke.sh 和一份 out/ 产物
└── docs/                  # 你在这里
```

`deploy/` 里的三份产物都已经到位并真实跑通：`merge_lora.py`（LoRA 权重合并回 text-only 底座）、`vllm_serve.sh`（起 vLLM endpoint，已原生支持 Anthropic Messages API，不需要额外协议转译层）、`switch_aiops_model.md`（真实接入 AIops-agent 的操作记录）。`smoke_test/` 下已有 `run_smoke.py`/`run_smoke.sh` 和一份 `out/` 产物。

## 如何跑测试

**每个子目录的测试要分开跑，不要合在一次 `pytest` 调用里**。原因是好几个子目录各自有一份同名的 `tests/conftest.py`（比如 `data/clean/tests/conftest.py` 和 `data/cold_start/tests/conftest.py`），pytest 用文件路径去重导入模块时会把两份 `conftest` 当成同一个模块名，后加载的会覆盖先加载的，导致 `ImportError: cannot import name 'xxx' from 'conftest'`——这是本仓库和 AIops-agent 都存在的一个已知坑，不是 bug，是分开跑的约定。正确姿势：

```bash
pytest reward/tests
pytest data/seeds/tests
pytest data/clean/tests
pytest data/cold_start/tests
pytest verl_adapter/tests
pytest sft/tests
pytest grpo/tests
pytest eval/tests
pytest deploy/tests
```

九条命令跑完，本仓库写这份文档时的真实结果是 **439 个测试通过、5 个跳过**（4 条在 `verl_adapter/tests`，是依赖真实外部环境/包的用例——环境不满足时会跳过并走文档化的降级路径，不是失败；1 条在 `data/clean/tests`，是缺 `sentence_transformers` 时的 BGE 去重降级路径）。如果你想验证消融脚本或 mock 评测报告的数字，可以直接跑：

```bash
python3 -m eval.ablations.no_prefix_split
python3 -m eval.ablations.no_credit_assignment
python3 -m eval.run_baseline_comparison
```

这几个脚本不是测试，是会打印/落盘真实计算结果的小工具，`eval/reports/` 下已经有一份跑过的输出留档。

## 和 AIops-agent 的关系

`AIops-agent/` 是通过 `git submodule` 引入的独立仓库，本仓库不修改它的核心代码。唯一的一次改动是一条**纯新增**的提交——`Diagnosis` schema 加了 `suspect_repo` / `suspect_commit_hint` / `suspect_file_hint` 三个可选字段（默认 `None`，向后兼容），用来承载"诊断 Agent 把可疑仓库/commit/文件线索交接给代码修复 Agent"这个设计文档里设想的信号。这个提交只存在于本仓库这份 submodule 的本地副本里，没有推到 AIops-agent 的真实远程仓库。

本仓库对 AIops-agent 的消费方式是纯"读"——`reward/` 里的判定逻辑读的是 `AIops-agent/agent/core/hooks.py`/`remediation.py`/`schema.py` 本来就有的 allow/deny 结果和校验规则，不需要往 AIops-agent 里塞任何训练侧的钩子。
