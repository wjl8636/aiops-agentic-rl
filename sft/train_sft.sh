#!/usr/bin/env bash
# AIOps 故障诊断处置 Agent · Cold Start SFT 训练入口脚本。
#
# 本脚本在本次会话里【没有被实际执行过】——这台机器没有 GPU、没有装 llamafactory、
# 没有真实的 Qwen3.5-9B checkpoint。脚本本身的调用约定（`llamafactory-cli train <config.yaml>`、
# `FORCE_TORCHRUN=1` 多卡启动方式）已经对照 LLaMA-Factory 官方 README.md 和 examples/README.md
# 核实过，不是凭训练记忆瞎写；但「在真实租的 GPU 机器上跑起来」
# 这件事本身没有被验证，跑之前请先过一遍下面的 Preflight 检查清单。
#
# ---------------------------------------------------------------------------
# Preflight 检查清单（真正在租的 GPU 机器上跑之前，人工确认一遍）：
#   1. 装好 LLaMA-Factory：
#        git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
#        cd LLaMA-Factory && pip install -e . && pip install -r requirements/metrics.txt
#      （install 命令已对照官方 README.md 核实；extras 具体写法以你 clone 到的版本 README 为准）
#   2. 装好 GPU 相关依赖：torch（配对目标机器的 CUDA 版本）、flash-attn（如果 YAML 里
#      flash_attn: fa2，机器上必须能装上 flash-attn，装不上就把 YAML 里改成 flash_attn: sdpa）。
#   3. 先跑 sft/load_text_only.py 把原始 Qwen3.5-9B 多模态 checkpoint 剥离视觉塔、
#      re-save 成 text-only 版本，再把 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 里
#      的 model_name_or_path 占位路径改成这份 text-only checkpoint 的真实绝对路径。
#   4. 确认下面 LLAMAFACTORY_DIR 指向你实际 clone LLaMA-Factory 的目录。
# ---------------------------------------------------------------------------

set -euo pipefail

# 本仓库根目录（此脚本假定从任意 cwd 调用，用脚本自身路径推出 REPO_ROOT，不依赖调用者的 cwd）。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# LLaMA-Factory 的安装/clone 目录，按需通过环境变量覆盖：
#   LLAMAFACTORY_DIR=/path/to/LLaMA-Factory sft/train_sft.sh
LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-${REPO_ROOT}/../LLaMA-Factory}"

TRAIN_CONFIG="${REPO_ROOT}/sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml"
DATASET_INFO_SNIPPET="${REPO_ROOT}/data/cold_start/dataset_info_snippet_v9.json"

echo "[train_sft] REPO_ROOT           = ${REPO_ROOT}"
echo "[train_sft] LLAMAFACTORY_DIR    = ${LLAMAFACTORY_DIR}"
echo "[train_sft] TRAIN_CONFIG        = ${TRAIN_CONFIG}"

if [[ ! -d "${LLAMAFACTORY_DIR}" ]]; then
  echo "[train_sft] 错误: 找不到 LLaMA-Factory 目录 '${LLAMAFACTORY_DIR}'。" >&2
  echo "  clone 官方仓库后通过 LLAMAFACTORY_DIR=<path> 环境变量指定，或放在 ${REPO_ROOT}/../LLaMA-Factory 。" >&2
  exit 1
fi

LF_DATASET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"
if [[ ! -f "${LF_DATASET_INFO}" ]]; then
  echo "[train_sft] 错误: 没找到 ${LF_DATASET_INFO}，这不像一份正常的 LLaMA-Factory 安装目录。" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 关键前置步骤：把 data/cold_start/dataset_info_snippet_v9.json 里的
# "aiops_cold_start_prefix_split" 注册项合并进 LLaMA-Factory 的 data/dataset_info.json，
# 否则 LLaMA-Factory 根本不知道 `dataset: aiops_cold_start_prefix_split` 这个名字对应哪份数据文件。
#
# 这里选择「用 python 做一次 JSON merge 并直接改写 LLaMA-Factory 目录里的文件」而不是符号链接整个
# dataset_info.json，原因是 LLaMA-Factory 自带的 dataset_info.json 本身还注册了 identity/alpaca 等
# 官方 demo 数据集，直接整体替换会破坏那些注册项；merge 是唯一不破坏现状的做法。
# 如果你不想让脚本改写 LLaMA-Factory 安装目录里的文件，可以注释掉下面这段，转而手动执行：
#   把 data/cold_start/dataset_info_snippet_v9.json 的内容手工粘贴合并进
#   <LLAMAFACTORY_DIR>/data/dataset_info.json 顶层 dict 里。
# ---------------------------------------------------------------------------
echo "[train_sft] 合并 dataset_info_snippet_v9.json 到 LLaMA-Factory 的 data/dataset_info.json ..."
python3 - "${LF_DATASET_INFO}" "${DATASET_INFO_SNIPPET}" "${REPO_ROOT}/data/cold_start" <<'PYEOF'
import json
import sys
from pathlib import Path

lf_dataset_info_path = Path(sys.argv[1])
snippet_path = Path(sys.argv[2])
cold_start_dir = Path(sys.argv[3])

lf_info = json.loads(lf_dataset_info_path.read_text(encoding="utf-8"))
snippet = json.loads(snippet_path.read_text(encoding="utf-8"))

for name, entry in snippet.items():
    entry = dict(entry)
    # snippet 里的 file_name 是相对 data/cold_start/ 目录的文件名（如
    # "aiops_cold_start_sft.mock.jsonl"）；LLaMA-Factory 的 dataset_info.json 里的 file_name
    # 是相对它自己 data/ 目录解析的，所以这里改写成能从 LLaMA-Factory data/ 目录定位到本仓库
    # 真实数据文件的绝对路径，避免用户还要再手动 cp/ln 一次数据文件。
    entry["file_name"] = str(cold_start_dir / entry["file_name"])
    if name in lf_info and lf_info[name] != entry:
        print(f"  [WARN] 覆盖已存在的同名 dataset_info 注册项: {name}")
    lf_info[name] = entry

lf_dataset_info_path.write_text(
    json.dumps(lf_info, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"  已合并 {list(snippet.keys())} 到 {lf_dataset_info_path}")
PYEOF

# ---------------------------------------------------------------------------
# 真正的训练调用。已对照官方仓库 README.md / examples/README.md 核实：
#   - 单卡: llamafactory-cli train <config.yaml>
#   - 多卡单机: FORCE_TORCHRUN=1 llamafactory-cli train <config.yaml>
#             （可选 CUDA_VISIBLE_DEVICES=0,1 限定可见 GPU）
# design doc §5.1「经济档」是 2x RTX 4090 24GB，SFT 阶段单卡即可（GRPO 才要双卡分离
# rollout/训练），所以这里默认走单卡调用；如果要多卡数据并行，取消下面 FORCE_TORCHRUN 那行的注释。
# ---------------------------------------------------------------------------
cd "${LLAMAFACTORY_DIR}"

echo "[train_sft] 启动训练: llamafactory-cli train ${TRAIN_CONFIG}"
# FORCE_TORCHRUN=1 \
llamafactory-cli train "${TRAIN_CONFIG}"

echo "[train_sft] 训练完成。LoRA 权重产出目录见 YAML 里的 output_dir。"
echo "[train_sft] 部署前需要 merge LoRA 回 text-only 底座（design doc §5.1「服务化部署」），"
echo "  用 llamafactory-cli export <merge_lora 配置>，本脚本不包含这一步。"
