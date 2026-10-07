"""校验 sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 跟设计文档 §5.1 声明的字段值不会静默漂移。

只做「YAML 能被正常解析 + 关键字段值跟文档一致」的静态校验，不涉及真的跑训练
（这台机器没有 llamafactory/transformers/GPU，跑不了真训练，见 sft/ 目录下其它文件的说明）。
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "llamafactory_config" / "qwen3_5_9b_lora_sft.yaml"
)


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_yaml_parses():
    config = _load_config()
    assert isinstance(config, dict)
    assert config, "配置文件不应为空"


def test_lora_hyperparams_match_design_doc():
    # design doc §5.1: "LoRA（r=16、alpha=32、target = q/k/v/o + gate/up/down）"
    # 模块名列表已按真实 transformers 源码（modular_qwen3_5.py, v5.15.1）补全：Qwen3.5
    # 的 full_attention 层用标准命名（q/k/v/o_proj），linear_attention/Gated DeltaNet 层
    # 用不同命名（in_proj_qkv/z/b/a + out_proj），两者不冲突，需要都列出才能覆盖全部 32 层。
    config = _load_config()
    assert config["finetuning_type"] == "lora"
    assert config["lora_rank"] == 16
    assert config["lora_alpha"] == 32

    target_modules = {m.strip() for m in config["lora_target"].split(",")}
    expected_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    }
    assert target_modules == expected_modules


def test_precision_and_memory_optimizations_match_design_doc():
    # design doc §5.1: "关键显存优化组合：LoRA + gradient checkpointing + bf16 + FlashAttention-2"
    config = _load_config()
    assert config["bf16"] is True
    assert config["disable_gradient_checkpointing"] is False
    # 2026-09-07 实测改判：flash_attn 从设计文档的 "fa2"（FlashAttention-2）改为
    # "sdpa"——标准注意力层走 PyTorch 自带 SDPA、Gated DeltaNet 线性注意力层走
    # flash-linear-attention 的 Triton kernel（fla==0.5.2 需要 Triton>=3.8，与 fa2
    # 的编译链约束冲突）。真实训练机上 SFT 已用 sdpa 跑通，依据见
    # docs/训练踩坑记录.md §3.5"最终跑通的稳定组合"。LLaMA-Factory 的
    # AttentionFunction 枚举取值是字符串（"sdpa"/"fa2"），不是布尔值。
    assert config["flash_attn"] == "sdpa"


def test_dataset_matches_sibling_task_registration():
    # 必须跟 data/cold_start/dataset_info_snippet.json 里注册的 key 完全一致，
    # 否则 LLaMA-Factory 找不到这个 dataset 名字。
    config = _load_config()
    assert config["dataset"] == "aiops_cold_start_prefix_split"
    assert config["stage"] == "sft"
    assert config["do_train"] is True


def test_model_path_is_placeholder_for_text_only_checkpoint():
    config = _load_config()
    model_path = config["model_name_or_path"]
    assert isinstance(model_path, str) and model_path
    # 不要求路径真实存在（这台机器没有 9B checkpoint），但要求路径名能体现「text-only」，
    # 提醒使用者这里必须填 sft/load_text_only.py 剥离视觉塔之后的产物目录，不是原始多模态 checkpoint。
    assert "text-only" in model_path or "text_only" in model_path


def test_dataset_info_snippet_registers_same_dataset_name():
    """跨文件一致性检查：本 YAML 引用的 dataset 名字必须真的存在于
    data/cold_start/dataset_info_snippet_v9.json 里（该文件由 sibling task 产出，本任务只读不改）。

    文件名带 `_v9` 后缀——v9 训练数据落地时新增的版本化命名约定，这份测试原来引用的是
    没有后缀的旧文件名，已同步更新，不要再改回去。
    """
    import json

    snippet_path = (
        CONFIG_PATH.resolve().parent.parent.parent
        / "data"
        / "cold_start"
        / "dataset_info_snippet_v9.json"
    )
    snippet = json.loads(snippet_path.read_text(encoding="utf-8"))

    config = _load_config()
    assert config["dataset"] in snippet
    assert snippet[config["dataset"]]["formatting"] == "sharegpt"
