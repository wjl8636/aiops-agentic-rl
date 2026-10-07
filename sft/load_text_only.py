"""视觉塔剥离脚本（design doc §5.1「底座选型、训练与部署算力方案」）。

## 这个脚本要做什么
Qwen3.5-9B 官方发布的是一个「原生多模态统一模型」（文本 + 图像 + 视频，早期融合架构）checkpoint。
本项目（AIOps 故障诊断决策）是纯文本场景，视觉塔（vision tower + 图文融合层）对我们是纯粹的显存
和磁盘冗余。设计文档给出的做法是：**用 `Qwen3_5ForCausalLM`（纯文本因果语言模型类）而不是
`Qwen3_5ForConditionalGeneration`（多模态类）去 `from_pretrained` 同一份 checkpoint**，只实例化
语言主干，然后把结果 `save_pretrained` 成一份新的、不含视觉权重的 text-only checkpoint 目录，
后续 SFT / GRPO / vLLM 部署全部指向这份 text-only 目录。

## 本脚本没有验证过、必须诚实说明的部分
本次会话所在环境**没有网络权限下载 9B 级别的真实权重，也没有装 `transformers`**（见下方运行时检查），
所以下面这些事实上没有被验证：
1. ~~`Qwen3_5ForCausalLM` 是否是真实类名~~ —— 已核实（对照 transformers v5.15.1 官方源码
   `modular_qwen3_5.py` 的 `__all__` 导出列表 + 官方文档 https://huggingface.co/docs/transformers/model_doc/qwen3_5
   的 Quickstart，两处都确认 `Qwen3_5ForCausalLM` 是真实存在的纯文本类）。不同 transformers 版本仍可能有差异，
   但类名本身不再是未知假设。
2. 官方文档 Quickstart 给出的示例是直接
   `Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", device_map="auto")`——即直接拿多模态 checkpoint 的
   Hub id 用纯文本类加载，文档层面认可这条路径。但**本脚本仍未在真实 9B checkpoint 上跑过**，
   加载时视觉权重具体是以 unexpected_keys 警告的形式被丢弃、还是需要先手动摘掉 config 里的 `vision_config`，
   仍待真实跑一次确认（已知 transformers 有一个相关 issue 谈到 `AutoModelForCausalLM`——注意不是
   `Qwen3_5ForCausalLM` 本身——在 auto 映射时不会自动展开嵌套的 `text_config`：
   https://github.com/huggingface/transformers/issues/45759 ，本脚本用的是显式类名不走 Auto 映射，
   预期不受影响，但值得留意）。
3. Gated DeltaNet 混合架构（config.text_config.layer_types 决定每层是 linear_attention 还是
   full_attention）在 `save_pretrained` 之后是否会带来额外的自定义权重命名/量化问题，仍未验证。
   另外线性注意力路径依赖可选的 `causal_conv1d`（Dao-AILab）和 `fla` 包，没装的话会静默回退到更慢、
   更吃显存的 PyTorch 实现——不影响本脚本的正确性，但影响后续训练/推理速度，值得在装环境时一并装上。

因此本脚本的定位是**结构上正确、按文档+官方证据要求实现，但从未在真实 9B checkpoint 上跑通**。跑之前
必须先过下面的运行时环境检查；如果检查失败，报错信息里给出了具体该怎么查证的路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 「纯文本处理（关键）...用 Qwen3_5ForCausalLM（而非 Qwen3_5ForConditionalGeneration）
# 加载 checkpoint 时只实例化语言主干」。这是我们对纯文本因果 LM 类名的唯一依据。
EXPECTED_CAUSAL_LM_CLASS = "Qwen3_5ForCausalLM"
# 多模态类名，仅用于在报错信息里提示「如果你只找到了这个类，说明装的是纯多模态 API，还没有拆分出
# text-only 类」，不在本脚本里实际使用。
EXPECTED_CONDITIONAL_GEN_CLASS = "Qwen3_5ForConditionalGeneration"


def _check_transformers_support():
    """运行时守卫：在真正尝试加载 9B 权重之前，先确认当前环境的 transformers 版本认得
    `Qwen3_5ForCausalLM` 这个类。找不到就立刻用清晰、可执行的报错信息退出，而不是让
    `from_pretrained` 在几十 GB 权重下载到一半之后才炸掉。

    返回：`transformers` 模块本身（调用方用它取 `AutoTokenizer` 等）以及目标类对象。
    """
    try:
        import transformers  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "[load_text_only] 当前环境没有安装 `transformers`，本脚本无法运行。\n"
            "  pip install -U transformers\n"
            "装好之后重新运行本脚本；如果装的版本仍然没有 Qwen3_5ForCausalLM，"
            "见下面『版本不匹配』分支的排查建议。"
        ) from exc

    causal_lm_cls = getattr(transformers, EXPECTED_CAUSAL_LM_CLASS, None)
    if causal_lm_cls is not None:
        return transformers, causal_lm_cls

    available_qwen3 = sorted(name for name in dir(transformers) if "Qwen3" in name)
    has_conditional_gen_only = EXPECTED_CONDITIONAL_GEN_CLASS in available_qwen3

    hint = ""
    if has_conditional_gen_only:
        hint = (
            f"\n  检测到 `{EXPECTED_CONDITIONAL_GEN_CLASS}`（多模态类）存在但纯文本类不存在——"
            "说明这个 transformers 版本目前只暴露了多模态入口，还没有（或者改名成了别的）"
            "对应的纯文本因果 LM 类。"
        )

    if available_qwen3:
        qwen3_symbols_msg = str(available_qwen3)
    else:
        qwen3_symbols_msg = "(一个都没有，说明这个 transformers 版本还完全不认识 Qwen3.5，可能太旧)"

    raise SystemExit(
        f"[load_text_only] 本仓库的设计假设是 transformers 提供 `{EXPECTED_CAUSAL_LM_CLASS}` 这个"
        "纯文本因果语言模型类（用于剥离 Qwen3.5-9B 的视觉塔），但当前安装的 "
        f"transformers=={transformers.__version__} 里找不到这个类。{hint}\n"
        f"  当前环境里名字含 'Qwen3' 的类/符号: {qwen3_symbols_msg}\n"
        "  这份设计文档写于 Qwen3.5 正式发布前后，官方最终定案的类名/模块路径有被调整的风险。"
        "排查步骤：\n"
        "    1) 确认 transformers.__version__ 是否 >= Qwen3.5 发布时对应的版本号；\n"
        "    2) 查 Qwen3.5 官方 model card / HF Hub 页面上给出的 `from_pretrained` 示例代码，"
        "确认现在的纯文本类到底叫什么；\n"
        "    3) 如果类名变了，把本文件顶部的 `EXPECTED_CAUSAL_LM_CLASS` 改成新的真实类名再重跑，"
        "本脚本其余逻辑（config 补丁 / save_pretrained 流程）不需要改。"
    )


def strip_vision_and_resave(
    source_checkpoint: str,
    output_dir: Path,
    dtype: str,
    attn_implementation: str,
    trust_remote_code: bool,
    delete_source_after_load: bool = False,
) -> None:
    """核心步骤：用纯文本因果 LM 类加载 `source_checkpoint`，只保留语言主干，
    re-save 成一份 text-only checkpoint 到 `output_dir`。

    `delete_source_after_load`（2026-09-10 真机首跑追加，磁盘空间受限时的可选优化）：
    原始多模态 checkpoint（~19GB）+ text-only 输出（~18GB）同时落盘在小容量分区上时可能放不下。
    开启这个选项后，会在**模型权重和 tokenizer 都已经完整加载进内存/显存之后**、`save_pretrained`
    之前，把 `source_checkpoint`（必须是本地目录，Hub id 不受影响）从磁盘删除，从而把峰值磁盘占用
    从「原始 + 输出」降到「max(原始, 输出)」。加载步骤本身（类名、dtype、attn_implementation 的
    选型依据）不受这个选项影响，纯粹是加载完成之后的磁盘腾挪时序调整。

    2026-09-10 真机首跑血泪教训（务必保留，不要在没理解原因前删掉下面的显式 clone 步骤）：
    第一次实现 `delete_source_after_load` 时天真地以为 `shutil.rmtree(source)` 之后磁盘立刻腾出
    空间，结果真机上 `df` 实时监控显示：`rmtree` 之后目录确实从文件系统里消失了（`ls`/`df <path>`
    都报「不存在」），但底层磁盘块完全没有释放——`save_pretrained` 写输出文件时磁盘继续被两份数据
    同时占用，直到写到 100% 塞满报 `ENOSPC` 崩溃退出、Python 进程真正退出后，块才被内核一次性释放。
    根因是 `safetensors` 库的 zero-copy 设计：`safe_open(...).get_tensor(k)` 返回的张量底层 storage
    直接是对源文件 mmap 区域的视图（不是独立内存拷贝），并且这个视图会让 mmap 映射保持存活，
    直到所有引用它的张量都被释放为止——Linux 语义下，一个文件被 `mmap()` 之后即使 `unlink()`，
    只要还有存活的映射，底层 inode/数据块就是「幽灵占用」状态，不会真正释放，`rm`/`rmtree` 报的
    「成功」只是删掉了目录项，不代表磁盘空间到手。**先尝试过给 `from_pretrained` 传
    `low_cpu_mem_usage=False`，真机复现结果完全一样（同样的 100% 写满 + 进程退出才释放），
    说明这个开关根本不影响 safetensors 底层的 zero-copy 视图，不是有效修复，别再走这条路。**
    真正有效的修复：模型加载完成后，显式遍历所有 parameters/buffers 逐个 `.clone()` 替换成独立
    内存拷贝，彻底切断跟源文件 mmap 的引用关系，这样 `rmtree(source)` 之后磁盘块才会立刻真正释放。
    代价是加载阶段峰值 RAM 需求更高（clone 时两份权重同时在内存里），但这台机器 RAM 是几百 GB
    量级，完全够用。
    """
    transformers, causal_lm_cls = _check_transformers_support()
    torch = _import_torch_or_die()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype not in dtype_map:
        raise SystemExit(f"[load_text_only] 不支持的 --dtype={dtype!r}，可选: {list(dtype_map)}")

    print(
        f"[load_text_only] 用 {EXPECTED_CAUSAL_LM_CLASS}.from_pretrained('{source_checkpoint}', "
        f"torch_dtype={dtype}, attn_implementation={attn_implementation!r}) 加载纯文本主干……\n"
        "  预期行为：checkpoint 里属于视觉塔/图文融合层的权重会被忽略/报 unexpected keys 警告——"
        "这是『视觉塔剥离』本身预期要发生的事，不是 bug。如果反而报 missing keys（语言主干权重缺失），"
        "说明这份 checkpoint 的分片布局跟纯文本类的期望不匹配，需要先手动检查 checkpoint 的 "
        "`model.safetensors.index.json`。"
    )
    model = causal_lm_cls.from_pretrained(
        source_checkpoint,
        torch_dtype=dtype_map[dtype],
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    if delete_source_after_load:
        # 见上方 2026-09-10 真机首跑教训：safetensors 的 zero-copy 视图会让模型参数底层
        # storage 一直 mmap 着 source_checkpoint 的原始文件，rmtree 之前必须先把每个
        # parameter/buffer 显式 clone 成独立内存拷贝，否则磁盘空间删不出来。
        print(
            "[load_text_only] --delete-source-after-load 已开启：clone 所有 parameters/buffers "
            "切断跟源文件 mmap 的引用关系（否则 rmtree 之后磁盘空间不会真正释放）……"
        )
        with torch.no_grad():
            for _name, _param in model.named_parameters():
                _param.data = _param.data.clone()
            for _name, _buf in model.named_buffers():
                if _buf is not None:
                    _buf.data = _buf.data.clone()

    # 只保存 tokenizer，不保存 image processor / processor_config（那些是多模态才需要的，
    # text-only 版本不应该带着它们，否则 vLLM / LLaMA-Factory 加载时可能误以为这仍是个多模态模型）。
    # 必须在（可能）删除 source_checkpoint 之前完成，因为 tokenizer 也是从 source 读的。
    print("[load_text_only] 加载纯文本 tokenizer（不加载 image processor/processor_config）……")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_checkpoint, trust_remote_code=trust_remote_code
    )

    if delete_source_after_load:
        source_path = Path(source_checkpoint)
        if source_path.is_dir():
            import gc  # noqa: PLC0415
            import shutil  # noqa: PLC0415

            # 保险起见显式 gc.collect()：确保 from_pretrained/AutoTokenizer.from_pretrained
            # 内部任何已经不再需要、但还没被引用计数回收的中间对象（可能持有对源文件的 mmap/fd）
            # 尽快释放，再执行 rmtree。核心修复是上面对每个 parameter/buffer 的显式 .clone()。
            gc.collect()
            print(
                f"[load_text_only] --delete-source-after-load 已开启：模型+tokenizer 已完整加载"
                f"进内存，现在删除本地源目录 {source_path} 以腾出磁盘空间……"
            )
            shutil.rmtree(source_path)
        else:
            print(
                f"[load_text_only] --delete-source-after-load 已开启，但 '{source_checkpoint}' "
                "不是本地目录（可能是 Hub id），跳过删除。"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load_text_only] 保存 text-only 语言主干到 {output_dir} ……")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(
        f"[load_text_only] 完成。text-only checkpoint 已写入 {output_dir}\n"
        "  下一步：把 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 里的 "
        "`model_name_or_path` 占位路径改成这个目录的真实绝对路径。"
    )


def _import_torch_or_die():
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as exc:
        raise SystemExit(
            "[load_text_only] 没有安装 `torch`，请先 `pip install torch`（配合目标 GPU 的 CUDA 版本）。"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "把一份 Qwen3.5-9B 多模态 checkpoint 用 Qwen3_5ForCausalLM 剥离视觉塔后 "
            "re-save 成 text-only checkpoint（design doc §5.1）。"
        )
    )
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help=(
            "原始 Qwen3.5-9B 多模态 checkpoint 路径或 HF Hub 模型 id，例如 "
            "'/data/models/Qwen3.5-9B' 或 'Qwen/Qwen3.5-9B'（占位，具体 Hub id 以官方发布为准）。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="剥离视觉塔后的 text-only checkpoint 输出目录。",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="加载/保存精度，默认 bf16（design doc §5.1『关键显存优化组合』之一）。",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help=(
            "transformers `from_pretrained(attn_implementation=...)` 的取值，默认 'flash_attention_2'。"
            "如果目标机器没装 flash-attn，改成 'sdpa'。"
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Qwen3.5 是新架构，发布初期大概率仍需要 trust_remote_code=True（默认开启）。",
    )
    parser.add_argument(
        "--delete-source-after-load",
        action="store_true",
        default=False,
        help=(
            "模型+tokenizer 完整加载进内存后，删除本地 --source-checkpoint 目录再落盘 output-dir。"
            "磁盘空间不够同时容纳『原始 checkpoint + text-only 输出』时使用，Hub id 来源会自动跳过。"
        ),
    )
    args = parser.parse_args()

    strip_vision_and_resave(
        source_checkpoint=args.source_checkpoint,
        output_dir=args.output_dir,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        delete_source_after_load=args.delete_source_after_load,
    )


if __name__ == "__main__":
    # 提前跑一次环境检查并打印结果，即使用户不带任何参数直接执行本文件，也能立刻看到
    # 当前 sandbox/训练机上 transformers 到底支不支持本脚本的核心假设。
    if len(sys.argv) == 1:
        print(
            "[load_text_only] 未提供参数，仅做一次 transformers/Qwen3_5ForCausalLM 环境自检：\n"
            "  用法: python3 sft/load_text_only.py --source-checkpoint <path_or_hub_id> "
            "--output-dir <path>"
        )
        _check_transformers_support()
        print("[load_text_only] 环境自检通过：transformers 里找到了 Qwen3_5ForCausalLM。")
        sys.exit(0)
    main()
