"""校验 deploy/merge_lora.py 的 CLI 参数解析、路径校验逻辑、以及缺依赖时的报错信息。

这台机器没有装 transformers/peft/torch（也没有 GPU、没有真实的 9B checkpoint），所以真正的
merge 逻辑（`merge_lora_and_save()` 内部 `from_pretrained` / `merge_and_unload` 调用）没法端到
端跑通——这一点跟 sft/tests/test_config_shape.py 面对同类问题的态度一致：只做「这台机器上能做的
静态校验」，不假装验证了跑不了的部分。

本文件覆盖三类可以离线验证的东西：
  1. argparse 参数解析：默认值、choices 校验（不依赖任何重量级库）。
  2. `_validate_paths()` 的路径校验逻辑：用 tmp_path 构造真实的目录结构去触发每条校验分支。
  3. `_check_merge_dependencies()` 的缺依赖报错信息：用 `sys.modules` 注入假的
     `transformers`/`peft` 模块（而不是依赖这台机器「恰好没装」这件事——那样测试结果会随环境
     漂移，不可复现），覆盖「完全没装」「装了但没有目标类」「transformers 齐了但没装 peft」
     三种分支，确认每种分支都给出可执行的排查线索（而不是一句「出错了」就退出）。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

# deploy/ 本身不是一个带 __init__.py 的 package（跟 sft/ 的历史写法不同，本任务只在 deploy/
# 内部新增文件，不去改其它目录的既有约定），所以用 sys.path 注入 + 直接 import 模块名，
# 不依赖 pytest 的 rootdir 自动插入行为。
_DEPLOY_DIR = Path(__file__).resolve().parent.parent
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

import merge_lora  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. argparse 参数解析
# --------------------------------------------------------------------------- #


def test_parser_requires_all_three_paths():
    parser = merge_lora._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # 三个 --xxx-dir/checkpoint 都是 required=True


def test_parser_defaults_match_design_doc():
    parser = merge_lora._build_parser()
    args = parser.parse_args(
        [
            "--base-checkpoint",
            "/data/models/qwen3.5-9b-text-only",
            "--adapter-dir",
            "/data/saves/lora",
            "--output-dir",
            "/data/models/merged",
        ]
    )
    # design doc §5.1「关键显存优化组合」之一是 bf16；跟 sft/load_text_only.py、
    # sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml 的默认精度保持一致。
    assert args.dtype == "bf16"
    assert args.attn_implementation == "flash_attention_2"
    assert args.trust_remote_code is True
    assert args.base_checkpoint == "/data/models/qwen3.5-9b-text-only"
    assert args.adapter_dir == Path("/data/saves/lora")
    assert args.output_dir == Path("/data/models/merged")


def test_parser_rejects_unsupported_dtype():
    parser = merge_lora._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--base-checkpoint",
                "x",
                "--adapter-dir",
                "y",
                "--output-dir",
                "z",
                "--dtype",
                "int8",  # 不在 choices=["bf16", "fp16", "fp32"] 里
            ]
        )


# --------------------------------------------------------------------------- #
# 2. _validate_paths() 路径校验逻辑
# --------------------------------------------------------------------------- #


def _make_valid_adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "lora_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    return adapter_dir


def test_validate_paths_rejects_missing_adapter_dir(tmp_path):
    with pytest.raises(SystemExit, match="adapter-dir 指向的目录不存在"):
        merge_lora._validate_paths(
            base_checkpoint="Qwen/Qwen3.5-9B",
            adapter_dir=tmp_path / "does_not_exist",
            output_dir=tmp_path / "out",
        )


def test_validate_paths_rejects_adapter_dir_without_config(tmp_path):
    adapter_dir = tmp_path / "lora_adapter"
    adapter_dir.mkdir()  # 目录存在，但没有 adapter_config.json
    with pytest.raises(SystemExit, match="没有 adapter_config.json"):
        merge_lora._validate_paths(
            base_checkpoint="Qwen/Qwen3.5-9B",
            adapter_dir=adapter_dir,
            output_dir=tmp_path / "out",
        )


def test_validate_paths_accepts_hub_id_base_checkpoint_without_requiring_it_on_disk(tmp_path):
    """HF Hub 模型 id（如 'Qwen/Qwen3.5-9B'）也带 '/'，不应该被误判成本地路径而要求存在于磁盘。"""
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    # 不应抛异常：base_checkpoint 是 Hub id 写法，output_dir 不存在（首次运行的正常情况）。
    merge_lora._validate_paths(
        base_checkpoint="Qwen/Qwen3.5-9B",
        adapter_dir=adapter_dir,
        output_dir=tmp_path / "brand_new_output_dir",
    )


def test_validate_paths_rejects_missing_local_base_checkpoint(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    missing_local_path = str(tmp_path / "no_such_checkpoint_dir")
    with pytest.raises(SystemExit, match="base-checkpoint 看起来是本地路径但不存在"):
        merge_lora._validate_paths(
            base_checkpoint=missing_local_path,
            adapter_dir=adapter_dir,
            output_dir=tmp_path / "out",
        )


def test_validate_paths_accepts_existing_local_base_checkpoint(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    base_checkpoint_dir = tmp_path / "text_only_base"
    base_checkpoint_dir.mkdir()
    merge_lora._validate_paths(
        base_checkpoint=str(base_checkpoint_dir),
        adapter_dir=adapter_dir,
        output_dir=tmp_path / "out",
    )


def test_validate_paths_rejects_nonempty_output_dir(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "leftover_file.bin").write_text("stale", encoding="utf-8")
    with pytest.raises(SystemExit, match="已存在且非空"):
        merge_lora._validate_paths(
            base_checkpoint="Qwen/Qwen3.5-9B",
            adapter_dir=adapter_dir,
            output_dir=output_dir,
        )


def test_validate_paths_accepts_empty_existing_output_dir(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()  # 存在但是空的：允许（比如用户提前 mkdir 占位）
    merge_lora._validate_paths(
        base_checkpoint="Qwen/Qwen3.5-9B",
        adapter_dir=adapter_dir,
        output_dir=output_dir,
    )


# --------------------------------------------------------------------------- #
# 3. _check_merge_dependencies() 缺依赖时的报错信息
# --------------------------------------------------------------------------- #


@pytest.fixture
def clean_sys_modules(monkeypatch):
    """确保测试注入的假 transformers/peft 不会污染其它测试；测试结束后恢复原状。"""
    for name in ("transformers", "peft"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield monkeypatch


def test_missing_transformers_gives_actionable_error(clean_sys_modules):
    clean_sys_modules.setitem(sys.modules, "transformers", None)  # import 会得到 ModuleNotFoundError
    # 用 setitem(..., None) 会让 `import transformers` 抛 ImportError，这是 CPython 官方支持的
    # 「显式标记某个模块不可用」的写法（PEP 328 / importlib 文档：sys.modules[name] = None
    # 会让后续 import 直接抛 ImportError），不需要真的卸载包。
    with pytest.raises(SystemExit) as excinfo:
        merge_lora._check_merge_dependencies()
    msg = str(excinfo.value)
    assert "pip install -U transformers" in msg


def test_transformers_without_causal_lm_class_gives_actionable_error(clean_sys_modules):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.0.0-fake"
    # 故意不设置 Qwen3_5ForCausalLM，模拟「版本太旧/类名对不上」的场景。
    clean_sys_modules.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(SystemExit) as excinfo:
        merge_lora._check_merge_dependencies()
    msg = str(excinfo.value)
    assert merge_lora.EXPECTED_CAUSAL_LM_CLASS in msg
    assert "4.0.0-fake" in msg


def test_transformers_ok_but_peft_missing_gives_actionable_error(clean_sys_modules):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.99.0-fake"
    setattr(fake_transformers, merge_lora.EXPECTED_CAUSAL_LM_CLASS, object())
    clean_sys_modules.setitem(sys.modules, "transformers", fake_transformers)
    clean_sys_modules.setitem(sys.modules, "peft", None)

    with pytest.raises(SystemExit) as excinfo:
        merge_lora._check_merge_dependencies()
    msg = str(excinfo.value)
    assert "pip install -U peft" in msg


def test_all_dependencies_present_returns_modules_and_class(clean_sys_modules):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.99.0-fake"
    fake_causal_lm_cls = object()
    setattr(fake_transformers, merge_lora.EXPECTED_CAUSAL_LM_CLASS, fake_causal_lm_cls)
    clean_sys_modules.setitem(sys.modules, "transformers", fake_transformers)

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = object()
    clean_sys_modules.setitem(sys.modules, "peft", fake_peft)

    transformers_mod, causal_lm_cls, peft_mod = merge_lora._check_merge_dependencies()
    assert transformers_mod is fake_transformers
    assert causal_lm_cls is fake_causal_lm_cls
    assert peft_mod is fake_peft
