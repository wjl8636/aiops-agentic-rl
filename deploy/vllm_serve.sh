#!/usr/bin/env bash
# 用 vLLM 把 deploy/merge_lora.py 产出的合并 checkpoint 起成 OpenAI-compatible endpoint
# （服务化部署：vLLM + 单卡 4090 / A10 / L20 24GB），已在真实 GPU 上跑通并接入
# AIops-agent。下面用到的 CLI 参数已经对照 vllm 官方仓库 main 分支源码核实过
# （不是凭训练记忆瞎写），核实方式见文件末尾的「核实记录」。
#
# ---------------------------------------------------------------------------
# 重要提醒（读完这段再往下看）：
# AIops-agent 用的是 Claude Agent SDK，SDK 底层是启动本机 `claude` CLI 子进程、通过
# ANTHROPIC_BASE_URL 把请求路由出去，`claude` CLI 只认 Anthropic Messages API 格式
# （POST /v1/messages）。实测装的 vLLM 0.28.0 已经原生实现了这个端点（不再是只有
# OpenAI Chat Completions 格式），`ANTHROPIC_BASE_URL` 可以直接指向下面这个 vLLM 服务
# 的地址，不需要额外接一层协议转译（LiteLLM Proxy 等）。接入 AIops-agent 侧的完整环境变量
# 见 deploy/switch_aiops_model.md，本脚本只负责起 vLLM 本身。
# ---------------------------------------------------------------------------

set -euo pipefail

# 合并后的单体 checkpoint 目录：deploy/merge_lora.py 的 --output-dir 产物，通过环境变量覆盖：
#   MODEL_DIR=/data/models/aiops-qwen3.5-9b-merged deploy/vllm_serve.sh
MODEL_DIR="${MODEL_DIR:-/data/models/aiops-qwen3.5-9b-merged}"

# 对外暴露给客户端（LiteLLM / AIops-agent 侧）的模型名，对应下面 --served-model-name。
# 建议跟 deploy/switch_aiops_model.md 里 AIOPS_MODEL 建议值保持一致，避免多处改名字漏改。
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-aiops-qwen3.5-9b}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# 2026-09-10 真机评测事故修订：32768 不够用，改成可覆盖的变量，默认值大幅上调。
# 真实 AIops-agent eval（claude_code Agent SDK 完整系统提示词 + skills + RAG 命中的历史
# 工单文本 + 多轮工具调用历史）实测单轮 prompt 达到 109007 字符；claude CLI 对
# `[claude-code:unrecognized_model]`（本仓库自定义 served-model-name 不在 CLI 已知模型表里）
# 会默认按 32000 output tokens 预留预算，32768 - 32000 = 768 tokens 根本不够放这份 prompt，
# 每一轮诊断都 100% 必然触发 400 context-length 错误（不是模型能力问题，是纯 infra 配置过小）。
# 模型原生 max_position_embeddings=262144，GPU 显存也远够（KV cache 预算由
# --gpu-memory-utilization 决定，不随 max-model-len 线性增加），改成 131072 留足余量。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# 可选：给服务加一层 Bearer token 鉴权（vLLM `--api-key`）。留空则不启用鉴权——
# 仅建议在租卡机器有防火墙/仅内网可达时这样做，对公网暴露必须设置。
VLLM_API_KEY="${VLLM_API_KEY:-}"

echo "[vllm_serve] MODEL_DIR           = ${MODEL_DIR}"
echo "[vllm_serve] SERVED_MODEL_NAME   = ${SERVED_MODEL_NAME}"
echo "[vllm_serve] HOST:PORT           = ${HOST}:${PORT}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[vllm_serve] 错误: 找不到合并后的 checkpoint 目录 '${MODEL_DIR}'。" >&2
  echo "  先跑 deploy/merge_lora.py 产出合并 checkpoint，再通过 MODEL_DIR=<path> 环境变量" \
       "指定这个目录，或者放在默认路径 /data/models/aiops-qwen3.5-9b-merged 。" >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "[vllm_serve] 错误: 找不到 vllm 命令。请先 pip install vllm（配合目标 GPU 的 CUDA 版本）。" >&2
  exit 1
fi

API_KEY_ARGS=()
if [[ -n "${VLLM_API_KEY}" ]]; then
  API_KEY_ARGS=(--api-key "${VLLM_API_KEY}")
fi

# ---------------------------------------------------------------------------
# 参数选型说明（design doc §5.1「服务化部署」+「关键显存优化组合」，单卡 4090/A10/L20 24GB）：
#
#   --dtype bfloat16
#     跟 SFT 训练、merge_lora.py 保存时的精度保持一致（design doc §5.1 明确要求 bf16）。
#     不用 fp16：Qwen3.5 系列官方发布即以 bf16 为主，混用精度容易在数值上引入不必要的偏差。
#
#   --max-model-len（默认 131072，见上方 MAX_MODEL_LEN 变量定义处的事故记录）
#     这里原先写的是「32768 在训练轨迹体量下够用」的估算，已被真机评测数据证伪：
#     完整 AIops-agent Claude Agent SDK 系统提示词 + skills + RAG 检索结果 + 多轮工具调用
#     历史，单轮 prompt 实测能到 10 万+字符，远超训练时单条样本（cutoff_len=8192）的量级，
#     不能用训练时的样本长度估算服务化推理阶段的真实上下文占用。
#
#   --gpu-memory-utilization 0.85
#     vLLM 默认是 0.9；24GB 消费级卡（4090/A10/L20）单卡既要装模型权重（9B bf16 约 18GB）
#     又要留 KV cache + CUDA context 开销，0.85 留出比默认更多余量，规避 OOM。如果部署时
#     发现显存仍然吃紧（比如同卡上还跑着别的进程），可以进一步调低；如果整卡独占且吃紧的是
#     KV cache 容量（并发数不够），可以适当调高但要先确认没有其它进程共享这张卡。
#
#   --tensor-parallel-size 1
#     design doc §5.1 明确是「单卡 4090 / A10 / L20 24GB」承载 text-only Qwen3.5-9B 推理，
#     不需要张量并行；显式写出来只是为了避免有人在多卡机器上误跑成多卡分片。
#
#   --trust-remote-code
#     跟 sft/load_text_only.py、deploy/merge_lora.py 保持一致的假设：Qwen3.5 发布初期
#     大概率仍需要 trust_remote_code 才能加载自定义 modeling 代码。
# ---------------------------------------------------------------------------
echo "[vllm_serve] 启动: vllm serve ${MODEL_DIR} --served-model-name ${SERVED_MODEL_NAME} ..."
vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  "${API_KEY_ARGS[@]}"

# ---------------------------------------------------------------------------
# 核实记录（WebFetch 对照 vllm-project/vllm 官方仓库 main 分支源码，非训练记忆）：
#
#   - `vllm serve <model>` 是当前推荐入口。design doc / 常见旧教程写的
#     `python3 -m vllm.entrypoints.openai.api_server --model ...` 已核实为**deprecated**：
#     该模块文件（vllm/entrypoints/openai/api_server.py）主分支源码里直接
#     `warnings.warn("...is deprecated...Use `vllm server` instead.", DeprecationWarning)`
#     （注：源码警告文案写的是 `vllm server`，但 CLI 子命令注册的真实名字是 `vllm serve`，
#     `vllm/entrypoints/cli/serve.py` 里 `class ServeSubcommand: name = "serve"`）。
#     旧调用方式目前仍能跑（只是打警告），但本脚本改用官方现在维护的入口，避免教一个
#     即将被移除的调用方式。
#   - --dtype / --max-model-len / --served-model-name / --trust-remote-code：
#     vllm/engine/arg_utils.py 里 `model_group.add_argument("--dtype", ...)` /
#     `("--max-model-len", ...)` / `("--served-model-name", ...)` /
#     `("--trust-remote-code", ...)` 均能在源码里定位到。
#   - --gpu-memory-utilization：同文件 `cache_group.add_argument("--gpu-memory-utilization",
#     ...)`，对应 dataclass 字段 `CacheConfig.gpu_memory_utilization`。
#   - --tensor-parallel-size（含短别名 -tp）：同文件
#     `parallel_group.add_argument("--tensor-parallel-size", "-tp", ...)`。
#   - --host / --port / --api-key：定义在 vllm/entrypoints/launchers/cli_args.py 的
#     `FrontendArgs` dataclass（`host: str | None`、`port: int = 8000`、
#     `api_key: list[str] | None`），由同文件的 `add_cli_args()` 自动转成
#     `--host` / `--port` / `--api-key` argparse 参数。
#   - 本脚本没有用到 --enable-lora / --lora-modules：那组参数是「让 vLLM 直接加载原始
#     LoRA adapter、不落地合并权重」的另一条路径（同样在 cli_args.py 里核实到存在），
#     跟 design doc §5.1「训练完把 LoRA 权重 merge 回底座」这句话描述的路径不同。
#     本脚本走「先 merge_lora.py 合并、再服务合并后的单体 checkpoint」这条设计文档主线，
#     --enable-lora 路径列在这里只是留给读者知道这是一个真实存在的替代方案。
# ---------------------------------------------------------------------------
