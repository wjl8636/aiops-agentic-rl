# 训练产物：SFT / GRPO Adapter 与训练曲线

汇总 AIOps 诊断 Agent 底座模型（Qwen3.5-9B + LoRA）目前已经跑完的训练产物，供不想自己重训一遍、只想直接拿权重/曲线看效果的读者使用。

## 权重（LoRA adapter）

4 个 checkpoint 的 LoRA adapter 都传到了 ModelScope，**公开、免登录下载**：

**仓库地址**：https://www.modelscope.cn/models/HuaiNan54321/aiops-diagnosis-agent

| checkpoint | 说明 | 大小 |
|---|---|---|
| `sft-v8` | Cold Start SFT 初版（早期数据集，作为基线对比保留） | ~165MB |
| `sft-v9` | Cold Start SFT 最终版（v9 数据集，3 epoch），是 GRPO 训练的起点 | ~165MB |
| `grpo-step10` | 在 sft-v9 基础上跑 GRPO 强化学习，第 10 步 checkpoint | ~83MB |
| `grpo-step15` | GRPO 第 15 步（15-epoch 训练的最终 checkpoint，即文档里的 `aiops-qwen3.5-9b-grpo-step15`） | ~83MB |

一键下载：

```bash
pip install modelscope
bash artifacts/download_weights.sh
```

会把 4 个 adapter 拉到本地 `./weights/` 目录。

**怎么用**：这些都是 LoRA adapter，不是合并后的完整模型，需要配合 base 模型加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("/path/to/qwen3.5-9b-text-only", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "./weights/sft-v9")
tokenizer = AutoTokenizer.from_pretrained("/path/to/qwen3.5-9b-text-only")
```

base 模型不是官方原版 Qwen3.5-9B（那是个多模态模型），是剥掉视觉塔之后的 text-only 版本，准备步骤见 [`docs/复现指南.md`](../docs/复现指南.md) 第 1 节（`sft/load_text_only.py`）。

## 训练曲线

- [`curves/sft-v8/`](curves/sft-v8/) —— `trainer_state.json` / `trainer_log.jsonl` / `training_loss.png`（LLaMA-Factory 训练输出，177 step，3 epoch，最终 train_loss 0.322）
- [`curves/sft-v9/`](curves/sft-v9/) —— 同上，252 step，3 epoch，最终 train_loss 0.336

**⚠️ GRPO 训练曲线缺失**：这一轮 GRPO 训练（`grpo_official_run_20260909_1740`）当时没开 wandb / swanlab / tensorboard 中任何一种日志，训练脚本自身的 stdout 也没有落盘（对应的 `main_ppo.log` 全部是 0 字节）。checkpoint 里唯一保存的 `data.pt` 只是 dataloader 迭代器状态，不含 reward / kl / loss 这些训练指标。也就是说 **reward 随 step 变化的曲线数据已经彻底丢失，没法从现有产物里恢复**——后续要么重跑一段 GRPO 时记得开日志，要么退而求其次对 `sft-v9` / `grpo-step10` / `grpo-step15` 逐 checkpoint 跑一遍评测，折算成离散对比点代替连续曲线。

## 效果对比（真实评测数字）

zero-shot / SFT / GRPO / Opus 四档完整对比已经在 [`docs/复现指南.md`](../docs/复现指南.md) 里记录，原始评测报告本仓库也存了一份镜像在 [`artifacts/eval_reports/`](eval_reports/)（复制自 `AIops-agent/reports/`，不依赖 submodule 权限也能直接打开核对）：

| 档位 | service_accuracy | kind_accuracy | route_accuracy |
|---|---|---|---|
| zero-shot（未训练 base，25 行） | 0.48 | 0.52 | 0.36 |
| SFT（v9，37 行） | 0.5676 | 0.6757 | 0.4595 |
| GRPO step15（25 行） | 0.80 | 0.72 | 0.64 |
| Opus 参考（23 行，非同口径，粗对比） | ~0.913 | — | — |

注意 SFT 档是 37 行、其余是 23~25 行，不是严格同口径，`docs/复现指南.md` 第 9 节讲了这个差异和排查方法，不要直接拿这几个数字线性外推对比。

## 拿 AIops-agent submodule 的提醒

`.gitmodules` 里 `AIops-agent` 用的是 SSH 地址（`git@github.com:...`），本机没配 GitHub SSH key 的话，`git clone --recurse-submodules` 会在这一步失败。两个办法：配好 SSH key，或者在 clone 前执行：

```bash
git config --global url."https://github.com/".insteadOf "git@github.com:"
```

用 HTTPS 匿名访问公开仓库。
