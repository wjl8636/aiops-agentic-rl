#!/usr/bin/env bash
# 从 ModelScope 拉取 AIOps 诊断 Agent 的 4 个 LoRA adapter checkpoint
# (sft-v8 / sft-v9 / grpo-step10 / grpo-step15)，公开仓库，无需登录/token。
#
# 用法: bash artifacts/download_weights.sh [目标目录，默认 ./weights]
set -euo pipefail

TARGET_DIR="${1:-./weights}"
REPO_ID="HuaiNan54321/aiops-diagnosis-agent"

if ! python3 -c "import modelscope" >/dev/null 2>&1; then
    echo "未检测到 modelscope 包，请先安装: pip install modelscope" >&2
    exit 1
fi

mkdir -p "$TARGET_DIR"

python3 - "$TARGET_DIR" "$REPO_ID" <<'PYEOF'
import sys
from modelscope.hub.snapshot_download import snapshot_download

target_dir, repo_id = sys.argv[1], sys.argv[2]
local_dir = snapshot_download(repo_id, local_dir=target_dir)
print(f"下载完成: {local_dir}")
PYEOF

echo "4 个 adapter (sft-v8 / sft-v9 / grpo-step10 / grpo-step15) 已下载到 ${TARGET_DIR}"
