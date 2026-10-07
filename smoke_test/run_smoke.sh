#!/usr/bin/env bash
# 端到端小规模烟雾测试的入口——实际逻辑在 run_smoke.py（用 Python 串联更方便复用
# reward/grpo 的真实函数，不必每一步都过一次 subprocess+序列化）。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 smoke_test/run_smoke.py "$@"
