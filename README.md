# AIOps Agentic RL — 诊断 Agent 的 SFT + GRPO 后训练

> **项目简介**：在自研 [AIops-agent](AIops-agent/)（一个基于 Claude Agent SDK、已端到端跑通的 AIOps 故障诊断修复 Agent，本仓库通过 git submodule 引入）之上，往**模型后训练**方向再做一层——用 **Cold Start SFT + GRPO 两段式后训练**，把「故障诊断处置 Agent」的决策底座从外部 Claude Opus 换成**自训 Qwen3.5-9B + LoRA**。核心思想是把 AIops-agent 生产环境里用来兜底安全的 **PreToolUse hook、白名单正则、Diagnosis JSON schema 校验零改造复用为 GRPO 训练的稠密奖励信号**，让开源小模型也能撑起同一套 Agent 系统的诊断决策。真实跑完：单卡 A800-80GB、15 epoch GRPO ≈ 12 小时，产出可在 AIops-agent 自己的 13 个真实 docker 故障场景上替换使用的 LoRA checkpoint。

本仓库通过 **git submodule**（`.gitmodules`）引入 [`AIops-agent/`](AIops-agent/)，克隆后需先 `git submodule update --init`（指针已钉在评测口径一致的 commit）。本仓库**只读**消费它的 SDK 接口和判定信号（hook allow/deny、schema 校验、白名单命中），不改其核心代码。它与 AIops-agent 的关系必须讲清楚：

- **AIops-agent（应用侧）**：告警进来，双 Agent（诊断 + 修复）用 Claude Opus 完成诊断 → 分流 → 自动止血或提 PR 的真实闭环。
- **本仓库（训练侧）**：只训**诊断 Agent** 的决策底座。**修复 Agent 不训**——代码修复需要跨文件、跨 build/test 的通用编程能力，9B 稠密模型撑不住；诊断 Agent 才是长轨迹 + 多信号 + 结构化输出 + 终止决策这一套 Agentic RL 的甜点。这个边界写死在代码里：`reward/outcome_reward.py` 的 `_assert_no_fix_result_leak` tripwire 运行时断言 `FixResult.verified` 绝不能进入诊断奖励。

**下游 headline（由 AIops-agent 自己的 `eval/run.py` 在 13 个真实 docker-compose 故障场景上跑出）**：

| 配置 | service | kind | route | 评测位置 |
|---|---|---|---|---|
| zero-shot（Qwen3.5-9B 底座，25 rows） | 0.48 | 0.52 | 0.36 | `AIops-agent/reports/aiops-qwen3.5-9b-base-zeroshot_20260910/` |
| v9-SFT（37 rows） | 0.5676 | 0.6757 | 0.4595 | `AIops-agent/reports/aiops-qwen3.5-9b-v9_20260908/` |
| **GRPO step15（25 rows）** | **0.80** | **0.72** | **0.64** | `AIops-agent/reports/aiops-qwen3.5-9b-grpo-step15_20260910/` |
| Opus 参照（23 rows） | ≈0.913 | ≈0.913 | ≈0.913 | `AIops-agent/reports/opus_20260819/` |

> GRPO 相对 v9-SFT：service +0.23、kind +0.04、route +0.19。**没有超过 Opus**——本项目要证明的是"开源小模型能撑同一套 Agent 系统"，9B 拿到 Opus 参照下这一档相对水平已足够生产替换的论证基础。三档 rows（37/25/23）是分次补跑造成的，主 headline 走均值口径。本仓 `eval/` 的 mock harness 只承担训练期 reward 调参的快速迭代，**所有对外披露的数字均来自 AIops-agent 的真环境 harness**。

## 核心亮点

- **两段式后训练路线**：Cold Start SFT（LLaMA-Factory + Qwen3.5-9B + LoRA r=16 α=32）→ GRPO Agentic RL（veRL 0.6.0 + vLLM colocate rollout + fsdp2 + LoRA）。不直接 RL from scratch——9B 底座工具调用格式与白名单命中不成熟，直接 RL 组内 advantage 接近零向量训不动。
- **零改造复用生产 hook 做稠密奖励**：`reward/step_reward.py` 里的 `guard_diagnose_ops`、`classify_command`、`DiagnosisToolCall` 直接从 AIops-agent `hooks/whitelist/schemas` import，训练侧不重写——生产 hook 迭代新规则，训练 reward 立即跟随，**训练/生产口径永远一致**。
- **三层 reward 设计 + 四类结构性反 hacking**：step_reward 7 条稠密信号（schema +0.05 / hook allow +0.10 / hook deny -0.15 / observation advances +0.10 / repeat no-new-info -0.15 / 末步 evidence fabrication -0.50 / schema fallback -0.20）；credit_assignment 按 kind × signal_type 权重表做**覆盖度加权求和**（不用 λ 指数衰减——GRPO 组内相对 advantage 场景下早期低成功率会梯度塌陷）；outcome_reward 结构化五项（stop_loss ±0.5/0/-0.3、root_cause 三字段各 +0.1、route +0.2、handoff 三字段各 +0.05）；反 hacking 四类防线全部**结构性**（双源覆盖门 -0.2、evidence 可追溯 -0.5、real-health-only、组方差降采样）。
- **veRL 适配层桥到真环境**：`RealHookRoutedTool` 注册 Bash / Read / Grep / Glob + `mcp__aiops_rag__search_past_incidents` 五类通用工具；SSH 反向隧道把 GPU 训练机的 rollout 桥回本地 Mac 的真实 docker OTel Demo；Milvus + BGE(bge-small-zh-v1.5) 记忆检索 RAG 接入训练。
- **冷启动数据真实采集**：真调 Claude Opus API 在真实 OTel Demo 上采集通过轨迹（v9 阶段 47 seed × 76 次 API 调用，验收率 74%，加 v8_base 保留合计 **63 条**通过轨迹，实花 **$58.77**），前缀拆分成 **667 条工具级 SFT 样本**（604 tool_call + 63 diagnosis），切分 train 74 / held_out 22。

## 目录结构

```
aiops-agentic-rl/
├── AIops-agent/              # git submodule：AIops-agent 应用侧仓库（只读消费，指针钉在评测口径一致的 commit）
├── data/
│   ├── seeds/                # 种子告警生成：generate_seeds.py 四路合成（OTel 场景参数化 + flagd 组合 + 历史工单反演 + 手写兜底）+ v9 定点行为增强（A/B/C/D 四家族）
│   ├── clean/                # 质量过滤 + 双层去重（MD5 精确 + BGE 语义 cosine≥0.92）+ 分层切分（train 74 / held_out 22）
│   ├── cold_start/           # 冷启轨迹采集（collect_trajectories.py，真实/mock 双模式）+ 拒绝采样 + 前缀拆分（63→667）+ LLaMA-Factory 格式转换
│   │   ├── trajectories_v9_final/   # 63 条真实通过轨迹
│   │   ├── aiops_cold_start_sft_v9.jsonl  # 667 条 SFT 样本
│   │   ├── _collection_v9_progress.md    # 采集账单与进度（$58.77 可溯源）
│   │   └── ground_truth/     # 人工标注
│   └── ood_fixtures/         # 4 条 OOD 告警 stub（TLS 过期/DNS 污染/慢查询/GC 长暂停）
├── reward/                   # 三层 reward 完整实现
│   ├── step_reward.py        # 每步 7 条稠密信号
│   ├── credit_assignment.py  # kind × signal_type 覆盖度加权
│   ├── outcome_reward.py     # 结构化五项 + _assert_no_fix_result_leak tripwire
│   ├── anti_hacking.py       # 四类结构性反 hacking 防线
│   └── trajectory.py         # Step / Trajectory 数据模型
├── verl_adapter/             # veRL 协议 ↔ 真实环境适配层
│   ├── rollout_worker.py     # RealHookRoutedTool + ssh_tunnel_bash_executor（真环境 rollout 路径）
│   ├── mock_tools.py         # 评测阶段 harness 回放
│   ├── rag_service.py        # Milvus + BGE 记忆检索服务（threading.Lock 串行化 + 启动预热）
│   ├── namespace_isolation.py# 命名派生（k8s/docker/milvus 三重）+ 文件锁
│   ├── tool_config.yaml      # 告诉 veRL 有哪些工具
│   └── start_rag_service.sh
├── sft/                      # Cold Start SFT
│   ├── train_sft.sh          # LLaMA-Factory 训练入口
│   ├── load_text_only.py     # Qwen3.5-9B 多模态 → text-only 剥离视觉塔
│   └── llamafactory_config/  # qwen3_5_9b_lora_sft.yaml（LoRA r=16 α=32 bf16 + 全线性层 12 proj）
├── grpo/                     # GRPO 训练
│   ├── train_grpo.sh         # veRL 训练入口（CLI dotted-override 调用方式，真机验证过）
│   ├── to_verl_dataset.py    # train/held_out → veRL 的 train.jsonl/held_out.jsonl
│   ├── verl_reward_adapter.py# veRL 自定义 reward 适配层
│   ├── reward_router.py      # 把 reward/ 三层组合成 GRPO 训练环消费的单一信号（per_step = step + credit，terminal 只加到末步）
│   └── verl_config/aiops_grpo.yaml  # 配置（字段标 [VERIFIED]/[ASSUMED]/[JUDGMENT] 状态）
├── eval/                     # 评测 harness（训练侧，mock 端点）
│   ├── run_eval.py           # 单端点评测
│   ├── run_baseline_comparison.py  # 三档对比（zero-shot/SFT/GRPO + Opus 参照骨架）
│   ├── model_endpoints.py    # 可插拔端点接口（真实使用时替换为 vLLM/Opus 适配器）
│   ├── metrics.py            # 指标
│   └── ablations/            # mechanism-level 消融：no_prefix_split.py / no_credit_assignment.py
├── deploy/                   # 部署
│   ├── merge_lora.py         # LoRA 合并回 text-only 底座
│   ├── vllm_serve.sh         # vLLM 服务化（原生 Anthropic Messages API 端点，已真机跑通接入 AIops-agent）
│   └── switch_aiops_model.md # AIOPS_MODEL 切换操作记录
├── smoke_test/               # 端到端 mock 烟雾测试（run_smoke.py / run_smoke.sh，全程真实代码、不需要 GPU/API/docker）
├── artifacts/                # 训练产物
│   ├── README.md             # 4 个 LoRA adapter 已公开上传 ModelScope（下载与加载示例）
│   ├── download_weights.sh   # 一键下载脚本
│   ├── curves/               # 训练曲线
│   └── eval_reports/         # AIops-agent 四档评测报告镜像
├── docs/                     # 文档（本仓库的详细说明在这里）
│   ├── README.md             # 使用指南（真实 vs 目标边界、测试方法）
│   ├── 复现指南.md           # 从租 GPU 到复现全部数字的完整指南
│   ├── 训练踩坑记录.md       # 7 个真实踩坑（torch/triton/fla 版本、veRL 默认值、SSH 隧道等）
│   └── 数据增强方案.md
├── requirements.txt          # 仅"数据 + reward + mock 评测"依赖；SFT/GRPO 大型框架依赖见 docs/复现指南.md
└── .gitmodules
```

## 环境依赖

**`requirements.txt` 只覆盖"无 GPU 也能装能跑"的部分**（数据构造 + reward 计算 + mock 烟雾测试 + 评测 harness）：

```text
pydantic>=2.0
jsonschema>=4.0
pytest>=7.0
sentence-transformers>=2.2      # BGE 语义去重（模型下不了时降级 hash 近似）
-r AIops-agent/requirements.txt  # submodule 本身的依赖
```

**真实 SFT / GRPO 训练与部署**所需的大型框架依赖**刻意不写进 requirements.txt**（版本对 CUDA/驱动强耦合，写死反而误导），只在租卡机器上按 [`docs/复现指南.md`](docs/复现指南.md) 安装：

| 组件 | 用途 | 版本要点 |
|---|---|---|
| torch / triton / flash-linear-attention | Qwen3.5 混合注意力底座 | 实测稳定组合 `torch==2.14.0` + `triton==3.8.0` + `fla==0.5.2`（线性注意力层强制 import fla） |
| LLaMA-Factory | Cold Start SFT | `git clone` + `pip install -e ".[torch,metrics]"` |
| veRL 0.6.0 | GRPO 训练框架 | `pip install -e .`（vllm extra） |
| vLLM | GRPO rollout + 服务化部署 | 需支持 Anthropic Messages API 的版本 |
| Qwen3.5-9B | 底座模型 | 需先 `sft/load_text_only.py` 剥离视觉塔成 text-only |
| CUDA 11.8+ / GPU | 训练硬件 | 单卡 A800-80GB 实测跑通（SFT 数小时 + GRPO 15 epoch ≈ 12 小时） |

> ⚠️ **不用 FlashAttention-2**：Qwen3.5 的 Gated DeltaNet 线性注意力层依赖 `fla`，对 Triton/torch 版本极敏感；训练/部署统一用 `sdpa` 实现，不装 flash-attn 本体（版本坑见 `docs/训练踩坑记录.md`）。

## 快速开始

### 1. 无 GPU 先验证流水线（smoke test + 单测）

```bash
pip install -r requirements.txt
python3 smoke_test/run_smoke.py       # 端到端 mock 烟雾测试（种子→清洗→采集→拆分→reward→评测全链路）
bash smoke_test/run_smoke.sh

# 单元测试（⚠️ 各子目录分开跑，不要合并成一次 pytest——同名 conftest.py 冲突坑，见 docs/README.md）
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

> 本仓库实测结果：**439 个测试通过、5 个跳过**（跳过项是依赖真实外部环境的用例，走文档化降级路径，不是失败）。

### 2. 数据侧完整链路（可复现）

```bash
# 种子生成（会覆盖 data/seeds/generated/，注意先备份）
python3 -m data.seeds.generate_seeds --count 47 --out-dir data/seeds/generated
# 清洗（质量过滤 → 双层去重 → 分层切分）
python3 -m data.clean.filter --in-dir data/seeds/generated --out-dir data/clean/filtered
python3 -m data.clean.dedup ...
python3 -m data.clean.split ...
# 冷启轨迹采集（mock 模式，真实模式需 Opus API + 真实 docker 环境）
python3 -m data.cold_start.collect_trajectories --mode mock
# 拒绝采样 + 前缀拆分 + SFT 格式转换
python3 -m data.cold_start.rejection_sample ...
python3 -m data.cold_start.prefix_split ...
python3 -m data.cold_start.to_llamafactory_format ...
```

### 3. 真实训练（租卡机器，详见 docs/复现指南.md）

```bash
# 0. 固定版本组合（见上表）+ 准备 text-only checkpoint
pip install torch==2.14.0 triton==3.8.0 flash-linear-attention==0.5.2
pip uninstall -y deepspeed
python3 sft/load_text_only.py --model-dir <原始多模态ckpt> --output-dir /data/models/qwen3.5-9b-text-only
# 记得把 preprocessor_config.json / video_preprocessor_config.json 复制回 text-only 目录

# 1. Cold Start SFT（LLaMA-Factory）
sft/train_sft.sh    # LLAMAFACTORY_DIR=<clone目录>；产出 sft-v9 的 LoRA checkpoint

# 2. 合并 LoRA → GRPO 起点
python3 deploy/merge_lora.py --adapter <sft-v9> --base <text-only> --output-dir /data/models/aiops-qwen3.5-9b-merged

# 3. GRPO（veRL，单卡 A800-80GB，15 epoch ≈ 12 小时）
#    先把 train/held_out 转成 veRL 数据集，再起训练
python3 -m grpo.to_verl_dataset ...
bash grpo/train_grpo.sh
# 产出 grpo/outputs/aiops-qwen3.5-9b-grpo-step{N}/（N=5/10/15，save_freq=5 / max_ckpt_to_keep=2）

# 4. 部署（vLLM，单卡 4090/A10/L20 24GB 即可）——vLLM 原生实现 Anthropic Messages API
MODEL_DIR=/data/models/aiops-qwen3.5-9b-merged deploy/vllm_serve.sh
# 5. 接入 AIops-agent：见 deploy/switch_aiops_model.md（AIOPS_MODEL / AIOPS_LLM_BASE_URL）
```

> **rollout 桥到真实环境**：GRPO rollout 跑在租的云 GPU 上（嵌套容器、无 CAP_NET_ADMIN、跑不了 docker-compose），Mac 侧用 `ssh -N -R 2222:localhost:22 <gpu>` 反向隧道，GPU 侧经 `localhost:2222` 反向 SSH 回 Mac 执行真实 kubectl/docker 命令。隧道 keepalive 参数（`ServerAliveInterval=30` + `ExitOnForwardFailure=yes`）是踩坑后补的，详见 `docs/训练踩坑记录.md`。

### 4. 评测

```bash
# 训练侧 mock 评测（reward 调参期快速迭代，一条 ~10 秒）
python3 -m eval.run_baseline_comparison
python3 -m eval.ablations.no_prefix_split          # mechanism-level 消融
python3 -m eval.ablations.no_credit_assignment     # mechanism-level 消融

# 真环境 headline（在 AIops-agent 仓库，加载真实系统提示词 + 白名单 hook + 真 Milvus + 真 docker 故障栈）
# 见本仓 submodule AIops-agent/ 的 eval/run.py 与 reports/ 下四档报告
```

## 关键模块说明

| 模块 | 一句话职责 | 对应教学文档 |
|---|---|---|
| `data/seeds/generate_seeds.py` | 四路合成种子告警（OTel 场景参数化 + flagd 组合 + 历史工单反演 + 手写兜底）+ v9 A/B/C/D 四家族定点增强 | 数据流水线 |
| `data/clean/dedup.py` | MD5 精确去重 + BGE(bge-small-zh-v1.5) 语义去重（cosine 阈值 0.92）+ 词法兜底降级 | 数据流水线 |
| `data/cold_start/prefix_split.py` | 按 tool_call 边界把 63 条轨迹前缀拆成 667 条工具级样本——撑起 SFT 最小训练规模的必要机制 | Cold Start SFT |
| `reward/step_reward.py` | 每步 7 条稠密信号（schema/hook/进展/重复/末步惩罚），直接 import AIops-agent 的 hook 判定 | Reward 工程 |
| `reward/credit_assignment.py` | kind × signal_type 覆盖度加权求和，替代 λ 指数衰减（防 GRPO 早期梯度塌陷） | Reward 工程 |
| `reward/outcome_reward.py` | 结构化五项打分 + `_assert_no_fix_result_leak` tripwire（诊断 reward 绝不消费 FixResult） | Reward 工程 |
| `reward/anti_hacking.py` | 四类结构性反 hacking：双源覆盖门 / evidence 可追溯 / real-health-only / 组方差降采样 | Reward 工程 |
| `grpo/reward_router.py` | 把三层 reward 组装成训练环信号：per_step = step + credit 直接相加，terminal 只加到末步（无 mixing 系数） | Reward 工程 |
| `verl_adapter/rollout_worker.py` | `RealHookRoutedTool`（真 hook 路由）+ SSH 反向隧道 bash 执行器，veRL rollout 桥到真实环境 | veRL 适配层 |
| `verl_adapter/rag_service.py` | Milvus + BGE 记忆检索服务（`threading.Lock` 常驻串行化 + 启动预热防冷启 spike） | veRL 适配层 |
| `grpo/verl_config/aiops_grpo.yaml` | veRL GRPO 配置（字段逐条标 [VERIFIED]/[ASSUMED]/[JUDGMENT] 状态） | GRPO 深度解析 |
| `sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml` | SFT 配置（LoRA r=16 α=32 bf16、全线性层 12 proj、sdpa） | Cold Start SFT |
| `deploy/merge_lora.py` / `vllm_serve.sh` | LoRA 合并 + vLLM 服务化（原生 Anthropic Messages API） | 部署 |
| `eval/ablations/*.py` | mechanism-level 消融：证明前缀展开 / credit assignment 是必要机制 | 评测口径 |

## 与 AIops-agent 的关系（三条硬边界）

1. **训练/应用侧边界**：只训诊断 Agent，修复 Agent 不训——`_assert_no_fix_result_leak` tripwire 在运行时强制。
2. **只读消费边界**：本仓库对 AIops-agent 的消费是纯"读"，唯一改动是一条纯新增 commit（`Diagnosis` schema 加 `suspect_repo` / `suspect_commit_hint` / `suspect_file_hint` 三个可选字段，向后兼容，未推到上游）。
3. **评测边界**：训练侧 mock harness 只做 reward 调参快速迭代；对外数字全部来自 AIops-agent 真环境 harness（13 个真实 docker-compose 故障栈）。

## 测试

```bash
# 单元测试 439 passed / 5 skipped（各子目录分开跑，见 docs/README.md 的 conftest 说明）
pytest reward/tests && pytest data/seeds/tests && pytest data/clean/tests && pytest data/cold_start/tests
pytest verl_adapter/tests && pytest sft/tests && pytest grpo/tests && pytest eval/tests && pytest deploy/tests

# 端到端 mock 烟雾测试
bash smoke_test/run_smoke.sh
```