#!/usr/bin/env bash
# Mac 侧常驻 RAG 服务的启动包装（restart-loop，崩了 5 秒自动拉起）。
# 用法：screen -dmS rag_service bash verl_adapter/start_rag_service.sh
# 日志：~/tunnel_logs/rag_service.log（跟反向隧道同一个管理惯例）。
# 服务本体：verl_adapter/rag_service.py（协议/部署形态见其模块 docstring）。
set -u
cd "$(dirname "$0")/.."
mkdir -p ~/tunnel_logs
while true; do
    AIops-agent/.venv/bin/python verl_adapter/rag_service.py >> ~/tunnel_logs/rag_service.log 2>&1
    echo "[$(date)] rag_service exited ($?), restarting in 5s" >> ~/tunnel_logs/rag_service.log
    sleep 5
done
