#!/usr/bin/env bash
# GRPO training launch script for the AIOps diagnosis agent
# (Qwen3.5-9B text-only + LoRA).
#
# *** NOT EXECUTED THIS SESSION — DO NOT RUN AS-IS ***
# This sandbox has no veRL install, no GPU, and no live AIOps-agent
# environment (kind cluster / docker-compose OTel Demo / remediation.py
# whitelist runtime). This script is a structurally-correct-but-unverified
# invocation.
#
# CLI dotted-overrides (this exact form) were chosen over
# `--config-path=grpo/verl_config --config-name=aiops_grpo` because the
# latter requires aiops_grpo.yaml to correctly compose against veRL's own
# Hydra `defaults:` list (verl/trainer/config/ppo_trainer.yaml's own
# `defaults:` block, e.g. `- rollout@actor_rollout_ref.rollout: rollout`) —
# that composition mechanics is exactly the one thing this session could
# NOT verify without a real veRL install. The CLI-override form below is
# the lower-risk path until someone with a veRL environment confirms
# whether --config-name=aiops_grpo also works standalone.
#
# Hardware requirements (doc §5.1 经济档, comment-only — see
# grpo/verl_config/aiops_grpo.yaml's trainer: block for why this is a
# comment and not a functional config field):
#   - 2x RTX 4090 24GB (or 1x A100 40GB) rented from AutoDL/Featurize/
#     潞晨云/恒源云 or similar. GPU 0 runs the vLLM rollout worker
#     (Qwen3.5-9B text-only, gpu_memory_utilization~0.6); GPU 1 runs the
#     LoRA policy update. Estimated GRPO wall-clock per doc: ~70h economical
#     tier / ~45h comfort tier (1x A100/A800 80GB, both stages on one GPU).
#   - SFT cold-start (LLaMA-Factory, not this script) must have already
#     produced the checkpoint this script's MODEL_PATH points at — GRPO
#     starting from the base model with no cold start is explicitly called
#     out in the design doc (§三) as non-convergent for this task (hook
#     denies every malformed tool call, exploration is all invalid
#     trajectories).
#   - veRL installed with the vllm extra (`uv sync --extra vllm` or
#     `pip install -e .[vllm]` per veRL's own install docs — not verified
#     this session).
#   - The real AIOps-agent runtime (kind/docker-compose OTel Demo +
#     remediation.py whitelist + hooks.py) reachable from the rollout
#     worker, wired up by verl_adapter/rollout_worker.py (sibling task, not
#     built by this task) so the multi_turn tool-calling loop actually
#     drives real tool calls, not a mock.

set -xeuo pipefail

# [2026-09-09 实测] update_weights 阶段 load_fsdp_model_to_gpu 把 offload 到 CPU
# 的 actor 权重整个搬回 GPU 跟 vLLM 常驻显存撞车，实测卡在"还差 834MiB"这种量级
# 的差距——PyTorch 自己的报错信息建议的 expandable_segments 能减少 caching
# allocator 的碎片浪费，先加上这个免费的余量，不够再动 GPU_MEMORY_UTILIZATION。
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

########################### user-adjustable ###########################
# [2026-09-08 落地] 默认值指向【SFT 合并后】的 checkpoint（deploy/merge_lora.py 的
# 产物），不是 text-only 原始底座。原因（veRL 源码 verl/trainer/ppo/ray_trainer.py
# :356 实测核实）：LoRA 场景下 KL 的 ref = "actor 去掉 LoRA"，所以——
#   路线A（text-only 底座 + lora_adapter_path=SFT adapter）：初始策略=SFT 模型✓，
#     但 KL ref=裸底座 ✗（把策略往预训练分布拉，违背设计文档 §5.4"对 SFT 之后的
#     分布做 KL 惩罚"）；
#   路线B（本配置，合并模型 + 全新零初始化 LoRA）：初始策略=合并SFT模型+恒等LoRA
#     =精确等于 SFT 策略 ✓，KL ref=合并SFT模型 ✓，两个语义都对，且避开
#     LLaMA-Factory adapter 往 veRL FSDP 路径加载的格式兼容风险。
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/models/aiops-qwen3.5-9b-v9-merged}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}                        # doc §5.1 经济档写的是 2x RTX 4090 24GB，但本机实际是 1x A800-80GB（未曾也不会有第二张卡），默认值改成跟实际硬件一致
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}            # [2026-09-09 实测] rollout.tensor_model_parallel_size 若不显式传，veRL 走 rollout.yaml 自身默认值 2，跟单卡拓扑冲突会在 init_model 时报 "rollout world_size: 1 is not divisible by infer_world_size: 2"；单卡必须显式设 1（对齐 aiops_grpo.yaml 里早就写好的 [JUDGMENT] 值，这里补上真正生效的 CLI override）
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.3}      # [2026-09-09 实测] 0.6 是照抄 doc §5.1"vLLM offload 到另一张卡"的两卡经济档算的；本机是单卡 colocate（actor+ref FSDP 和 vLLM rollout 挤同一张卡），FSDP 模型一加载完，79.25GB 里已经只剩 ~44GB 空闲——0.6*79.25=47.55GB 超过空闲量，vLLM 启动直接报 "Free memory ... is less than desired GPU memory utilization"。0.35（≈27.7GB）能让 vLLM 启动，但 update_weights 阶段 actor 权重从 CPU offload 搬回 GPU 时跟 vLLM 常驻显存一起会差着几百 MiB OOM；曾经把它降到 0.25 想让路，结果 vLLM 自己那 ~18GB 权重几乎吃满 19.8GB 预算，KV cache 只剩 0.88GB 反而在 engine 启动时报 KV cache 不够——这两个 OOM 分别在不同阶段触发，不能只调这一个值互相覆盖。加了 MAX_MODEL_LEN 显式限制后 KV cache 需求已经很小，0.3（≈23.8GB）在两边都留出余量，配合上面的 expandable_segments 和下面的 actor offload 一起解决
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}                      # [2026-09-09 实测] 不显式设时 veRL 把它留 null，vLLM 直接退回模型 config 的原生 max_position_embeddings（Qwen3.5 是 262144），KV cache 预算按"至少要能服务一条 262144 token 的请求"来算，8.05GiB 需求 vs 0.25 utilization 时只有 0.88GiB 可用直接报错；本项目 max_prompt_length(4096)+max_response_length(2048)、多轮工具调用按 doc §5.1 上限 12 轮算，16384 留了足够余量又不至于让 vLLM 白白按 262144 去要 KV cache 预算

# --- rollout 工具执行的环境接线（verl_adapter/rollout_worker.py 读取）---
# veRL 从 tool_config.yaml 实例化 RealHookRoutedTool 时只传 config/tool_schema，
# executor 默认就是隧道实现；隧道端口/用户/host 走 AIOPS_TUNNEL_*，RAG 桥的
# python/仓库路径走 AIOPS_MAC_*——rollout_worker.py 里这些变量的硬编码默认值
# 现在只是占位符（不含真实机器信息），必须显式 export 成当前部署的真实值才能跑：
#   AIOPS_TUNNEL_USER / AIOPS_TUNNEL_HOST / AIOPS_TUNNEL_PORT（隧道账号/host/端口）
#   AIOPS_MAC_PYTHON / AIOPS_MAC_REPO_ROOT（Mac 上仓库 venv python 路径 / 仓库根路径）
# 工具命令的工作目录同样必须显式 export：Mac 上 deploys.log / workspace/<svc>/
# 所在的【根目录】AIops-agent checkout（不是本仓库嵌套的子模块副本——活环境挂载在那份上）。
export AIOPS_MAC_AGENT_DIR=${AIOPS_MAC_AGENT_DIR:-/path/to/AIops-agent}

train_batch_size=${TRAIN_BATCH_SIZE:-74}                   # [2026-09-09 修复] 原默认 256 是按 doc 估算的 150 条查询池写的
                                                             # [JUDGMENT] 值；真实池子经语义去重+分层切分后只有 74 条
                                                             # （见 data/clean/split/train/），verl 的 train dataloader
                                                             # drop_last=True（ray_trainer.py:409），256>74 会导致每个
                                                             # epoch 唯一的 batch 都被整批丢弃——15 个 epoch 一步都不训练，
                                                             # 静默空转。改成 74（用满整个训练集，不丢数据）。
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}                # [JUDGMENT] AIOps alert + tool history context
max_response_length=${MAX_RESPONSE_LENGTH:-2048}            # [JUDGMENT] no doc number

# [2026-09-09 实测修复] 官方 batch=74 首次启动实测：以下 4 个字段在 train_batch_size
# 从之前小规模验证的 8 改到 74 之后，若不显式传，Hydra config 的 __post_init__/
# validate() 会直接拒绝启动（进程秒退，从未真正跑起来，GPU 显存全程 0）：
#   1) actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu —— ActorConfig.__post_init__
#      （verl/workers/config/actor.py:207-210）：use_dynamic_bsz=False 时必须显式设
#      ppo_micro_batch_size 或 *_per_gpu 二者之一，两个都不设直接 AssertionError。
#   2) actor_rollout_ref.actor.ppo_mini_batch_size —— 同文件 ActorConfig 的
#      dataclass 默认值是 256（第 151 行），validate()（第 224-228 行）要求
#      train_batch_size >= ppo_mini_batch_size；74 < 256 会在这道检查上报另一个
#      ValueError（跟上面那条不是同一个错，是这次没实测到但读源码确认的第二道坑）。
#   3/4) actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu /
#      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu —— rollout.py 里
#      同名字段（209-211 行）默认都是 None，跟 actor 侧同款"二选一"要求（此前
#      batch=8 真实验证时就是靠显式传这三个 *_per_gpu=1 才跑通多轮工具调用）。
# 取值判断：
#   - 三个 *_micro_batch_size_per_gpu=1（"一次 forward/backward 只处理 1 条"）
#     控制的是单次前向/反向的显存占用，跟外层 train_batch_size 总数无关——74 只是
#     让这个循环多跑几次，不会推高单次显存峰值，batch=8 时验证过的安全值可以直接
#     沿用，不需要因为总数变大而调大。
#   - ppo_mini_batch_size 选择等于 train_batch_size（74）：
#     verl/trainer/ppo/ray_trainer.py:1352-1353 显示实际生效的
#     mini_batch(=ppo_mini_batch_size * rollout.n) 只是用来在 make_minibatch_iterator
#     里切 DataProto chunk，源码里唯一的整除性检查（actor.py:232-236）只作用于已
#     deprecate 的 ppo_micro_batch_size（非 per_gpu 变体），per_gpu 变体没有整除
#     约束；让 ppo_mini_batch_size==train_batch_size 使 mini_batch 正好等于当前
#     step 的全部样本（74*6=444），单 step 内只有一个 mini-batch，不引入新的分片/
#     整除边界条件，跟 batch=8 验证时"mini_batch=train_batch_size"的比例关系完全
#     一致，只是等比放大，不是从 8 那套值抄一遍不假思索地搬过来。
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-${train_batch_size}}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

# data.train_files/data.val_files: veRL 真实必填字段（RLHFDataset 加载入口），
# 之前这个脚本完全没设置——`grpo/to_verl_dataset.py` 是补上的转换步骤，运行方式见
# 该脚本模块 docstring；跑这个训练脚本前先跑：
#   python3 -m grpo.to_verl_dataset --alerts-dir data/clean/split/train --split train --out grpo/verl_data/train.jsonl
#   python3 -m grpo.to_verl_dataset --alerts-dir data/clean/split/held_out --split held_out --out grpo/verl_data/held_out.jsonl
# [VERIFIED] RLHFDataset 按文件名后缀分支加载，`.jsonl` 走
# `datasets.load_dataset("json", ...)`，跟 `.parquet` 对 veRL 是等价输入
# （已用 WebFetch 对照 verl/utils/dataset/rl_dataset.py 真实源码核实，见
# `to_verl_dataset.py` 模块 docstring）。
TRAIN_FILES=${TRAIN_FILES:-grpo/verl_data/train.jsonl}
VAL_FILES=${VAL_FILES:-grpo/verl_data/held_out.jsonl}

actor_lr=${ACTOR_LR:-1e-6}                                  # [JUDGMENT] veRL example default, no doc number
kl_loss_coef=${KL_LOSS_COEF:-0.001}                         # [JUDGMENT] veRL example default; doc only says "KL 惩罚", no coefficient
rollout_n=${ROLLOUT_N:-6}                                   # doc §5.1: group_size = 6, VERIFIED field name actor_rollout_ref.rollout.n
max_assistant_turns=${MAX_ASSISTANT_TURNS:-12}              # doc §5.1: "最长交互 12 轮"; see aiops_grpo.yaml's [ASSUMED] note on this field mapping

lora_rank=${LORA_RANK:-16}                                  # doc §5.1: r=16
lora_alpha=${LORA_ALPHA:-32}                                # doc §5.1: alpha=32

total_epochs=${TOTAL_EPOCHS:-15}                            # [JUDGMENT] no doc number
save_freq=${SAVE_FREQ:-5}                                   # [2026-09-09 修复] 原默认 20 在本项目 15 epoch=15 step 的场景下永远
                                                             # 打不满，实际效果是只有最后一步（第 15 步）落一份 checkpoint——
                                                             # 即使 test_freq=5 会在第 5/10/15 步都跑 held-out 打分写进日志，
                                                             # 中间两次对应的权重从未落盘，事后没法回退到分数更好的中间
                                                             # checkpoint（用户担心过拟合，需要能事后择优）。改成 5，跟
                                                             # test_freq 对齐，让第 5/10/15 步都各存一份，配合下面的
                                                             # max_actor_ckpt_to_keep 一起保留够用的历史版本。
test_freq=${TEST_FREQ:-5}                                   # doc §5.4: periodic held-out spot-check cadence (exact number not given)

# [VERIFIED 2026-09-07, 训练机 aiops-gpu:/root/autodl-tmp/code/verl] veRL 按
# 【文件路径】加载 custom_reward_function（verl/trainer/ppo/reward.py::
# get_custom_reward_fn -> verl/utils/import_utils.py::load_module，相对进程
# CWD 解析），并以逐样本关键字签名调用（naive.py L131-135）——所以这里默认指向
# 逐样本适配层 grpo/verl_reward_adapter.py::compute_score（从 solution_str
# 还原 Trajectory、调 compute_trajectory_reward 出分；组方差由模块内
# _GroupAccumulator 攒够 6 条后调 compute_group_rewards，其"同进程收到整组"
# 的假设未在真实 veRL 环境验证，见该模块 docstring）。本脚本必须在仓库根目录
# 下启动（path 的 CWD 相对解析 + 模块自身的 sys.path 自举都依赖文件真实存在）。
REWARD_FN_PATH=${REWARD_FN_PATH:-grpo/verl_reward_adapter.py}
REWARD_FN_NAME=${REWARD_FN_NAME:-compute_score}

PROJECT_NAME=${PROJECT_NAME:-aiops-diagnosis-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_9b_lora_grpo_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# [VERIFIED] entrypoint + dotted-override invocation convention.
#
# [2026-09-09 实测] actor/ref 都显式传 fsdp_config.model_dtype=bfloat16 的原因：
# verl/workers/config/engine.py 的 FSDPEngineConfig.model_dtype 默认是 "fp32"——
# 跟顶层 actor.fsdp_config.dtype: bfloat16（那是 autocast 用的混合精度计算
# dtype）是两个独立开关，model_dtype 才是 _build_module 里真正 from_pretrained
# 时传给 torch_dtype 的值，决定整个 FSDP flat_param（含冻结的 base 权重）在显存
# 里以什么精度常驻。9B 模型按 fp32 存储 = 36GB，是预期 bf16(18GB) 的 2 倍——这才
# 是 load_fsdp_model_to_gpu 阶段实测 actor 进程摸到 ~60GB（而不是预想的 ~18GB）
# 的真正原因；之前两轮调 gpu_memory_utilization / expandable_segments 都是在错
# 的层面找余量。LoRA 场景冻结 base 权重用 bf16 存储是标准做法（不影响 LoRA
# adapter 本身的训练精度），ref 也要设，否则算 logprob 时把 fp32 模型搬上 GPU
# 会在别处复现同一个 OOM。
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.use_shm=True \
    \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${rollout_n} \
    `# [2026-09-09 实测修复，比下面的 multi_turn.format 更根本的一层] veRL 的
     # actor_rollout_ref.rollout.agent.default_agent_loop 默认值是
     # "single_turn_agent"（verl/workers/config/rollout.py:74），只在数据的
     # non_tensor_batch 里显式带 "agent_name" 字段时才会换成别的 agent loop
     # （verl/experimental/agent_loop/agent_loop.py:580-582，没带时整批直接退回
     # default_agent_loop）——grpo/to_verl_dataset.py 从未写这个字段，所以之前
     # 就算 multi_turn.enable=True 且 tool_config_path 配好，实际调用的是
     # SingleTurnAgentLoop（只生成一次就结束，完全不看 multi_turn/tools 配置），
     # ToolAgentLoop 和它内部的 tool parser（不管 hermes 还是 qwen3_coder）从未
     # 被实例化过——用临时 debug patch 在 Qwen3XMLToolParser.extract_tool_calls
     # 里加 logger.warning 实测验证：patch 后跑两个真实 step，一行都没打印，
     # 反证 extract_tool_calls 根本没被调用。这是 num_turns 恒为 2.0 的真正
     # 第一道根因；下面的 multi_turn.format=qwen3_coder 是第二道（parser 语法
     # 不匹配），两个都要修才能让工具调用真的跑起来。` \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_assistant_turns} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=verl_adapter/tool_config.yaml \
    `# [2026-09-09 实测修复] veRL 的 multi_turn.format 默认值是 "hermes"
     # (verl/workers/config/rollout.py:61)，其 HermesToolParser 只认
     # <tool_call>{json}</tool_call>。但 SFT 用的 qwen3_5_nothink 模板
     # （对照 LLaMA-Factory src/llamafactory/data/tool_utils.py 里
     # Qwen35ToolUtils.function_formatter，云端
     # /root/autodl-tmp/code/LLaMA-Factory 实测核实）产出的是原生嵌套语法：
     # <tool_call>\n<function=NAME>\n<parameter=k>v</parameter>\n</function>\n</tool_call>。
     # 两者不匹配导致 HermesToolParser.extract_tool_calls 永远解析不出
     # tool_calls，每次 rollout 被误判为"未发起工具调用"直接终止
     # (training/num_turns 恒为 2.0 的症状)。
     # 修复：veRL 自带的 Qwen3XMLToolParser（注册名 "qwen3_coder"，定义在
     # verl/experimental/agent_loop/tool_parser.py:190-375，docstring 写明
     # "Tool parser for qwen3_coder/qwen3.5 model"）的四条正则
     # (tool_call_regex / tool_call_function_regex / tool_call_parameter_regex)
     # 跟上述 Qwen35ToolUtils 输出语法逐字节匹配，无需新写 parser，直接切换
     # format 即可。` \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu} \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu} \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name="${REWARD_FN_NAME}" \
    \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    `# [2026-09-09 复核，维持 2 不变] 用户提出想固定保留 3 份 checkpoint（配合
     # save_freq=5，第 5/10/15 步都各留一份，方便事后对着 held-out 分数择优）——
     # 实测 df -h /root/autodl-tmp 当时可用 48G，单份 checkpoint（9B LoRA + optimizer
     # state）约 17-18G，3 份需要 51-54G，超过可用空间；改成 3 有跑到一半磁盘写满
     # 拖垮训练的风险，按预案退回保留最近 2 份（会滚动丢掉第 5 步那份，只保留
     # 第 10、15 步），先保证训练能跑完不会因为磁盘写满而中断。` \
    "$@"
