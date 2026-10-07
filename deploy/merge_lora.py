"""LoRA merge 脚本（服务化部署：vLLM + 单卡 4090 / A10 / L20 24GB）。

## 这个脚本要做什么
训练完把 LoRA 权重 merge 回 Qwen3.5-9B text-only 底座 → 用 vLLM 起
OpenAI-compatible endpoint。本脚本就是「merge 回底座」这一步：
  1. 用 `transformers` 加载 `sft/load_text_only.py` 产出的 text-only Qwen3.5-9B 底座
     （纯文本因果 LM 类，见该文件顶部对 `Qwen3_5ForCausalLM` 类名的说明与未验证风险）；
  2. 用 `peft.PeftModel.from_pretrained(base_model, adapter_dir)` 把
     `sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml` 训练产出的 LoRA adapter
     （PEFT 标准格式：adapter_config.json + adapter_model.safetensors）接到底座上；
  3. 调用 `merge_and_unload()` 把 LoRA 权重线性合并进底座权重，得到一个不再依赖 `peft` 库、
     vLLM 可以直接 `--model <output_dir>` 加载的单一 checkpoint 目录。

**适用范围**：只适用于 LLaMA-Factory（SFT）产出的 PEFT 标准格式 adapter。veRL（GRPO）保存的
checkpoint 是原始 FSDP 权重（`global_step_N/actor/model_world_size_*_rank_*.pt`），不是 PEFT
adapter，本脚本对这种 checkpoint 无效——GRPO checkpoint 要用 `python -m verl.model_merger merge
--backend fsdp` 合并，见 `docs/复现指南.md` 第 8 节。

## `merge_and_unload()` 是真实核实过的 API，不是凭训练记忆
本次会话所在环境没有装 `peft`（`pip show peft` 为空），无法本地跑一遍确认。已通过 WebFetch
拉取 huggingface/peft 官方仓库 `docs/source/developer_guides/checkpoint.md`（"Merge the
weights" 一节）核实到官方给出的示例正是：
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(...)
其中 `model` 是 `PeftModel.from_pretrained(base_model, adapter_path)` 返回的对象。这与本脚本
`merge_lora_and_save()` 里的调用顺序一致。官方文档同时提示：merge 之后就是一个普通
`transformers` 模型，不再有 PEFT 相关方法（不能再 unmerge / 加载多个 adapter / 临时禁用
adapter），这正是我们想要的「vLLM 能直接加载的单体 checkpoint」的形态。

## 本脚本没有验证过、必须诚实说明的部分
跟 `sft/load_text_only.py` 一样，这台机器没有网络权限下载 9B 权重、没有装
`transformers`/`peft`/`torch`，所以下面这些事实上没有被验证：
1. `sft/load_text_only.py` 里同样未验证的 `Qwen3_5ForCausalLM` 类名假设——本脚本复用同一个
   假设（因为 merge 的 base_model 必须用跟 SFT 训练时一致的类去加载，否则模块命名/权重形状
   对不上，LoRA adapter 的 `target_modules` 匹配会失败）。
2. `qwen3_5_9b_lora_sft.yaml` 里 `lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,
   down_proj` 这组模块名，在 Gated DeltaNet 混合架构的线性注意力层上是否真的一一对应
   （该 YAML 文件里已经用行内注释标注了这个风险，并给出 `lora_target: all` 作为 fallback）。
   如果训练时实际用的是 fallback，merge 阶段不需要改任何代码——`PeftModel.from_pretrained`
   会从 adapter_dir 里的 `adapter_config.json` 自己读出真实训练时的 target_modules，本脚本
   不需要、也不应该重复指定。
3. merge 后模型的显存/磁盘占用是否如预期（BF16 9B 底座 + 合并后的 LoRA 增量，理论上跟
   底座本身大小几乎相同，因为 LoRA 增量被吸收进原始权重矩阵，不额外增加参数量）。

因此本脚本的定位是**结构上正确、按 PEFT 官方文档核实过调用约定，但从未在真实 checkpoint 上
跑通**。跑之前必须先过下面的运行时环境检查；检查失败会给出具体该怎么查证的路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 跟 sft/load_text_only.py 保持同一份假设：这是 design doc 对纯文本因果 LM 类名的唯一依据
# （文档原文，非训练记忆）。base_model 必须用这个类加载，否则跟 LoRA adapter 训练时的模块
# 结构不匹配，PeftModel.from_pretrained 会报 target_modules 找不到。
EXPECTED_CAUSAL_LM_CLASS = "Qwen3_5ForCausalLM"


def _check_merge_dependencies():
    """运行时守卫：在真正尝试加载 9B 权重之前，先确认 `transformers` 认得
    `Qwen3_5ForCausalLM`，并且 `peft` 已安装。找不到就立刻用清晰、可执行的报错信息退出，
    而不是让 `from_pretrained` 在几十 GB 权重下载到一半之后才炸掉。

    风格跟 sft/load_text_only.py 的 `_check_transformers_support()` 保持一致（同一个项目的
    两个「本次会话没跑通、必须给出可执行排查路径」的脚本，报错信息的排查步骤应该长得一样）。

    返回：`transformers` 模块、目标 causal LM 类对象、`peft` 模块。
    """
    try:
        import transformers  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "[merge_lora] 当前环境没有安装 `transformers`，本脚本无法运行。\n"
            "  pip install -U transformers\n"
            "装好之后重新运行本脚本。"
        ) from exc

    causal_lm_cls = getattr(transformers, EXPECTED_CAUSAL_LM_CLASS, None)
    if causal_lm_cls is None:
        available_qwen3 = sorted(name for name in dir(transformers) if "Qwen3" in name)
        qwen3_symbols_msg = str(available_qwen3) if available_qwen3 else (
            "(一个都没有，说明这个 transformers 版本还完全不认识 Qwen3.5，可能太旧)"
        )
        raise SystemExit(
            f"[merge_lora] 本仓库的设计假设是 transformers 提供 `{EXPECTED_CAUSAL_LM_CLASS}` 这个"
            "纯文本因果语言模型类（跟 sft/load_text_only.py 剥离视觉塔时用的类必须是同一个，"
            "否则 base_model 的模块结构跟 LoRA adapter 训练时不一致），但当前安装的 "
            f"transformers=={transformers.__version__} 里找不到这个类。\n"
            f"  当前环境里名字含 'Qwen3' 的类/符号: {qwen3_symbols_msg}\n"
            "  排查步骤同 sft/load_text_only.py 的报错信息：先确认 transformers 版本，再查 "
            "Qwen3.5 官方 model card 上的真实类名，如果类名变了就同步改这里的 "
            "EXPECTED_CAUSAL_LM_CLASS（以及 sft/load_text_only.py 里的同名常量，两处必须一致）。"
        )

    try:
        import peft  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "[merge_lora] 当前环境没有安装 `peft`，本脚本无法运行。\n"
            "  pip install -U peft\n"
            "装好之后重新运行本脚本。"
        ) from exc

    if not hasattr(peft, "PeftModel"):
        # 极端情况下的兜底：一个自称 peft 的包但没有 PeftModel（例如损坏的安装）。
        raise SystemExit(
            f"[merge_lora] 当前安装的 `peft`（版本 {getattr(peft, '__version__', '未知')}）"
            "里找不到 `PeftModel`，这个安装可能损坏或者版本极其古老。\n"
            "  pip install -U --force-reinstall peft"
        )

    return transformers, causal_lm_cls, peft


def _import_torch_or_die():
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:
        raise SystemExit(
            "[merge_lora] 没有安装 `torch`，请先 `pip install torch`（配合目标 GPU 的 CUDA 版本）。"
        ) from exc


def _validate_paths(base_checkpoint: str, adapter_dir: Path, output_dir: Path) -> None:
    """CLI 参数的路径校验：能在没有 GPU/权重的环境里就做的检查，尽早给出可执行的报错。

    `base_checkpoint` 允许是本地路径也允许是 HF Hub 模型 id（跟 sft/load_text_only.py 的
    `--source-checkpoint` 一样），所以只有它「看起来像本地路径」（存在同名父目录或本身就是
    绝对/相对路径且带路径分隔符）时才检查是否存在；不对 Hub id 做存在性校验（`from_pretrained`
    自己会在真正下载时报错，那才是权威判断）。
    """
    adapter_dir = adapter_dir.expanduser()
    if not adapter_dir.exists():
        raise SystemExit(
            f"[merge_lora] --adapter-dir 指向的目录不存在: {adapter_dir}\n"
            "  这里应该填 LLaMA-Factory LoRA SFT 训练产出的 output_dir（PEFT 标准格式，"
            "目录里应该有 adapter_config.json + adapter_model.safetensors），"
            "见 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 的 `output_dir` 字段。"
        )
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.exists():
        raise SystemExit(
            f"[merge_lora] --adapter-dir={adapter_dir} 存在，但里面没有 adapter_config.json，"
            "不像一份 PEFT 标准格式的 LoRA adapter 目录。检查是不是指错了目录"
            "（比如指到了训练 output_dir 的上一级，或者指到了某个 checkpoint-N 子目录才对）。"
        )

    # 只有明显长得像「文件系统路径」的写法（绝对路径 / ./ ../ / ~）才做存在性校验；HF Hub 模型
    # id（如 "Qwen/Qwen3.5-9B"）里也带 "/"，不能拿「包含斜杠」当本地路径的判定依据，否则会把
    # 合法的 Hub id 误判成本地路径要求它必须存在于磁盘上。
    looks_like_local_path = base_checkpoint.startswith(("/", "./", "../", "~"))
    if looks_like_local_path and not Path(base_checkpoint).expanduser().exists():
        raise SystemExit(
            f"[merge_lora] --base-checkpoint 看起来是本地路径但不存在: {base_checkpoint}\n"
            "  这里应该填 sft/load_text_only.py --output-dir 产出的 text-only checkpoint 目录，"
            "不是原始的 Qwen3.5-9B 多模态 checkpoint。"
        )

    if output_dir.expanduser().exists() and any(output_dir.expanduser().iterdir()):
        raise SystemExit(
            f"[merge_lora] --output-dir={output_dir} 已存在且非空，为避免覆盖已有 merge 产物，"
            "请换一个空目录，或者先手动清空/搬走这个目录再重跑。"
        )


def merge_lora_and_save(
    base_checkpoint: str,
    adapter_dir: Path,
    output_dir: Path,
    dtype: str,
    attn_implementation: str,
    trust_remote_code: bool,
) -> None:
    """核心步骤：加载 text-only 底座 + LoRA adapter，merge 后 save 成单体 checkpoint。

    调用顺序对照 PEFT 官方文档 `checkpoint.md`「Merge the weights」一节核实过：
        base_model = CausalLM.from_pretrained(base_checkpoint, ...)
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)

    这个函数在本次会话里**从未被实际执行过**（没有网络、没有 9B 权重、没有装
    transformers/peft/torch），只做到「代码逻辑结构正确、调用约定跟 PEFT/transformers 的
    标准接口一致」。
    """
    transformers, causal_lm_cls, peft = _check_merge_dependencies()
    torch = _import_torch_or_die()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise SystemExit(f"[merge_lora] 不支持的 --dtype={dtype!r}，可选: {list(dtype_map)}")

    print(
        f"[merge_lora] 用 {EXPECTED_CAUSAL_LM_CLASS}.from_pretrained('{base_checkpoint}', "
        f"torch_dtype={dtype}, attn_implementation={attn_implementation!r}) 加载 text-only 底座……"
    )
    base_model = causal_lm_cls.from_pretrained(
        base_checkpoint,
        torch_dtype=dtype_map[dtype],
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    print(f"[merge_lora] 用 PeftModel.from_pretrained 加载 LoRA adapter: {adapter_dir} ……")
    peft_model = peft.PeftModel.from_pretrained(base_model, str(adapter_dir))

    print(
        "[merge_lora] 调用 merge_and_unload() 把 LoRA 权重合并进底座权重……\n"
        "  预期行为：合并后返回一个普通 transformers 模型（不再有 PEFT 专属方法），"
        "参数量跟底座本身基本相同（LoRA 增量被吸收进原始权重矩阵，不额外增加参数）。"
    )
    merged_model = peft_model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[merge_lora] 保存合并后的单体 checkpoint 到 {output_dir} ……")
    merged_model.save_pretrained(output_dir)

    print("[merge_lora] 保存 tokenizer（跟底座一致，不需要改动）……")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_checkpoint, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_dir)

    print(
        f"[merge_lora] 完成。合并后的 checkpoint 已写入 {output_dir}\n"
        "  下一步：把这个目录路径填进 deploy/vllm_serve.sh 的 MODEL_DIR，用 vLLM 起服务。"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 训练产出的 LoRA adapter "
            "merge 回 text-only Qwen3.5-9B 底座（design doc §5.1「服务化部署」第一步），"
            "得到一份 vLLM 可以直接 --model 加载的单体 checkpoint。"
        )
    )
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help=(
            "text-only 底座路径，即 sft/load_text_only.py --output-dir 的产物目录（绝对路径），"
            "不是原始 Qwen3.5-9B 多模态 checkpoint。"
        ),
    )
    parser.add_argument(
        "--adapter-dir",
        required=True,
        type=Path,
        help=(
            "LoRA adapter 目录（PEFT 标准格式），即 "
            "sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 里 `output_dir` 字段指向的训练产出。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="merge 后的单体 checkpoint 输出目录，交给 deploy/vllm_serve.sh 使用。",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="加载/保存精度，默认 bf16（跟 SFT 训练时的 bf16 保持一致，design doc §5.1）。",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help=(
            "transformers `from_pretrained(attn_implementation=...)` 的取值，默认 "
            "'flash_attention_2'。如果目标机器没装 flash-attn，改成 'sdpa'。"
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Qwen3.5 是新架构，发布初期大概率仍需要 trust_remote_code=True（默认开启）。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _validate_paths(args.base_checkpoint, args.adapter_dir, args.output_dir)

    merge_lora_and_save(
        base_checkpoint=args.base_checkpoint,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(
            "[merge_lora] 未提供参数，仅做一次 transformers/peft 环境自检：\n"
            "  用法: python3 deploy/merge_lora.py --base-checkpoint <text-only 底座路径> "
            "--adapter-dir <LoRA adapter 目录> --output-dir <merge 产物输出目录>"
        )
        _check_merge_dependencies()
        print("[merge_lora] 环境自检通过：transformers 里找到了 Qwen3_5ForCausalLM，且 peft 已安装。")
        sys.exit(0)
    main()
