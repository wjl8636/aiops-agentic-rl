# 把 AIops-agent 切到本地 vLLM 模型

对应 design doc `AIOps-Agent-SFT+Agentic-RL项目.md` §5.1「服务化部署：vLLM + 单卡 4090 / A10 /
L20 24GB」原文：

> 训练完把 LoRA 权重 merge 回 Qwen3.5-9B text-only 底座 → 用 vLLM 起 OpenAI-compatible
> endpoint → 修改 `sdk_runner.py` 里诊断 Agent 侧的 `AIOPS_MODEL` 指向本地 vLLM endpoint，
> 把原来的 Claude Opus API 调用换成本地推理，代码改动只有一行环境变量

本文档的结论：这句话是对的——AIops-agent 这一侧不需要改一行代码，只改环境变量，
`ANTHROPIC_BASE_URL` 可以直接指向本地 vLLM endpoint。下面先说清楚背后的机制，再给出真实
配置步骤。

## 1. investigation：Claude Agent SDK 到底能不能直接指向 vLLM

### 1.1 `sdk_runner.py` 用的是 Claude Agent SDK 的 `query()`，不是裸的 HTTP 客户端

`AIops-agent/agent/core/sdk_runner.py` 调用 `claude_agent_sdk.query(prompt=..., options=opts)`，
`opts.model = config.MODEL`（即 `AIOPS_MODEL` 环境变量）。这个 SDK **不是**一个直接拼 HTTP
请求打 Anthropic API 的客户端库。看了本机已安装的 `claude-agent-sdk==0.2.139` 包源码
（`_internal/transport/subprocess_cli.py`）：SDK 的真实实现是**在本机启动 `claude` CLI 二进制
作为子进程**，通过 stdin/stdout 传 JSON 消息跟这个子进程通信；`ClaudeAgentOptions.model` 最终
只是传给子进程的 `--model` 参数，真正发起网络请求、决定协议格式的是这个 `claude` CLI 子进程
本身。

`subprocess_cli.py` 里还确认了一个关键机制：子进程环境变量 = 继承父进程 `os.environ`（去掉
`CLAUDECODE`）叠加 `ClaudeAgentOptions.env` 覆盖。也就是说，**只要在启动 AIops-agent 的父
Python 进程的环境里设了 `ANTHROPIC_BASE_URL` 等变量，子进程 `claude` CLI 会自动继承**，不需要
改 `sdk_runner.py` 去显式传 `env=`。这一点跟 design doc「代码改动只有一行环境变量」的说法是
一致的、且已核实。

### 1.2 `claude` CLI 的自定义端点机制：真实存在，但只认 Anthropic Messages API 格式

用 WebFetch 拉取了 Claude Code 官方文档 `llm-gateway` 和 `llm-gateway-protocol` 两页，核实到：

- `ANTHROPIC_BASE_URL` 是真实、文档化的环境变量，用来把 Claude Code（以及被它包着跑的 Claude
  Agent SDK 子进程）指向一个自定义网关/端点。
- 但网关协议reference明确写着：一个网关必须实现下表列出的几种格式之一——
  **Anthropic Messages**（`ANTHROPIC_BASE_URL`，端点 `/v1/messages`）、Amazon Bedrock
  InvokeModel（`ANTHROPIC_BEDROCK_BASE_URL` + `CLAUDE_CODE_USE_BEDROCK=1`）、或 Google Cloud
  rawPredict（`ANTHROPIC_VERTEX_BASE_URL` + `CLAUDE_CODE_USE_VERTEX=1`）。文档原文还有一句很
  直白的免责声明：「Anthropic doesn't endorse, maintain, or audit third-party gateway
  products, and doesn't support routing Claude Code to non-Claude models through any
  gateway.」
- 「模型发现」（`/v1/models`）那一节进一步印证：Claude Code 只会保留 `id` 里包含
  `claude`/`anthropic` 子串的模型条目，其它一律过滤掉——这个 CLI 从设计上就是围绕 Claude
  模型家族构建的，不是一个通用的「指哪打哪」的模型客户端。

### 1.3 vLLM 起的服务，实测原生支持 Anthropic Messages API

`deploy/vllm_serve.sh` 起的是 vLLM 的 OpenAI-compatible server；实测装的 vLLM 0.28.0 同时
原生实现了 Anthropic Messages API 端点（`POST /v1/messages`），`curl
http://localhost:8000/v1/messages` 能拿到标准的 Anthropic 风格响应（`type: "message"`、
`content: [...]`、`stop_reason` 等字段齐全）。所以 `ANTHROPIC_BASE_URL` 可以直接指向这个
vLLM 服务的地址，不需要额外的协议转译层。

## 2. 真实、最小可行的部署链路

```
Qwen3.5-9B text-only 底座 (sft/load_text_only.py 产物)
        + LoRA adapter (sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 训练产物)
        ↓ deploy/merge_lora.py
合并后单体 checkpoint
        ↓ deploy/vllm_serve.sh (vllm serve，默认 :8000，同时支持 /v1/chat/completions 和 /v1/messages)
        ↓ ANTHROPIC_BASE_URL
claude CLI 子进程（Claude Agent SDK 底层） / AIops-agent
```

## 3. 需要在 AIops-agent 侧设置的环境变量

对照本仓库子模块 `AIops-agent/agent/config.py` 第 50 行 `MODEL = os.getenv("AIOPS_MODEL",
"opus")`，以及上面 1.1 节确认的「env 会从父进程继承给 claude CLI 子进程」机制：

| 变量 | 设成什么 | 作用 |
|---|---|---|
| `AIOPS_MODEL` | `aiops-qwen3.5-9b`（跟 `deploy/vllm_serve.sh` 的 `SERVED_MODEL_NAME` 一致） | `config.py` 读这个变量，传给 `ClaudeAgentOptions.model` |
| `ANTHROPIC_BASE_URL` | vLLM 服务的地址，例如 `http://localhost:8000` | 让 `claude` CLI 子进程把 Messages API 请求打到 vLLM 而不是 `api.anthropic.com` |
| `ANTHROPIC_AUTH_TOKEN` 或 `ANTHROPIC_API_KEY` | 任意非空占位字符串（vLLM 默认不校验，但 `claude` CLI 要求存在一个凭证才会走这条路径，且不会退回到 claude.ai 订阅登录） | 满足 `claude` CLI 「必须有凭证变量才不使用 claude.ai 订阅登录」这一行为 |

这三个变量在启动 AIops-agent 进程（`agent/main.py` 或对应入口）**之前**，用 shell
`export` 或写进它读取的 `.env` 里即可生效——`sdk_runner.py` 和 `config.py` 都不需要改代码。

## 4. 回滚：切回 Claude Opus

1. 把 `AIOPS_MODEL` 改回默认值 `opus`（或者直接 unset，`config.py` 里 `os.getenv("AIOPS_MODEL",
   "opus")` 的 fallback 就是 `"opus"`）。
2. 把 `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` 这两个变量 unset 掉（或者不再在启动
   AIops-agent 的环境里设置它们），让 `claude` CLI 走它默认直连 `api.anthropic.com` 的路径，
   用回原来的 Anthropic API key / claude.ai 订阅凭证。
3. 不需要停掉 vLLM（可以留着继续跑，不影响回滚），也不需要改 `sdk_runner.py` / `config.py`
   任何代码。

日常在 Opus 和本地模型之间来回切换，就是改这三个环境变量这么轻量。
