"""Shape test for ``grpo/verl_config/aiops_grpo.yaml``.

This does NOT validate that veRL can actually load/run this file (veRL is
not installed in this sandbox — see the YAML's own header comment for what
is [VERIFIED] vs [ASSUMED]/[JUDGMENT]). It only proves two things a pure-Python test CAN
prove without veRL installed:

1. The file parses as valid YAML.
2. The handful of numbers the design doc pins exactly (group_size=6,
   max_turns=12, KL loss enabled) are present and correct, AND match the
   same constants ``grpo/reward_router.py`` uses for its own turn-budget
   penalties — so the two files can't silently drift apart.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from grpo.reward_router import MAX_TURNS

CONFIG_PATH = Path(__file__).resolve().parents[1] / "verl_config" / "aiops_grpo.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_file_exists_and_parses_as_yaml():
    config = _load_config()
    assert isinstance(config, dict)
    assert config  # non-empty


def test_group_size_is_six():
    """doc §5.1: "每个 query 采 group size = 6 条独立轨迹"."""
    config = _load_config()
    assert config["actor_rollout_ref"]["rollout"]["n"] == 6


def test_max_turns_is_twelve_and_matches_reward_router():
    """doc §5.1: "最长交互 12 轮，超过 12 轮还没收敛的截断给负奖" — this
    number must match ``grpo.reward_router.MAX_TURNS`` (the constant the
    router's ``max_turns_truncation_penalty`` uses), so the YAML's rollout
    turn budget and the reward function's truncation penalty can't drift
    apart from each other over time.
    """
    config = _load_config()
    multi_turn = config["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert multi_turn["max_assistant_turns"] == 12
    assert multi_turn["max_assistant_turns"] == MAX_TURNS


def test_kl_penalty_against_sft_policy_is_enabled():
    """doc §5.1/§5.4: "对 SFT 之后的分布做 KL 惩罚"."""
    config = _load_config()
    actor = config["actor_rollout_ref"]["actor"]
    assert actor["use_kl_loss"] is True
    assert isinstance(actor["kl_loss_coef"], (int, float))
    assert actor["kl_loss_coef"] > 0


def test_adv_estimator_is_grpo():
    config = _load_config()
    assert config["algorithm"]["adv_estimator"] == "grpo"


def test_reward_function_points_at_verl_reward_adapter():
    """The per-sample adapter (``grpo/verl_reward_adapter.py::
    compute_score``) is what real veRL's ``NaiveRewardManager`` can actually
    call — the old whole-group ``compute_group_rewards`` entry had a calling
    convention veRL never invokes (see the adapter module's docstring for the
    verified call shape).
    """
    config = _load_config()
    custom_fn = config["reward"]["custom_reward_function"]
    assert custom_fn["path"] == "grpo/verl_reward_adapter.py"
    assert custom_fn["name"] == "compute_score"


def test_adapter_group_size_matches_rollout_n():
    """``grpo.verl_reward_adapter.DEFAULT_GROUP_SIZE`` (when its in-process
    accumulator flushes a group and calls ``compute_group_rewards``) must
    equal the configured GRPO group size, or the accumulator would never
    complete a group (or flush partial ones) — same drift-guard idea as the
    max_turns test above.
    """
    from grpo.verl_reward_adapter import DEFAULT_GROUP_SIZE

    config = _load_config()
    assert config["actor_rollout_ref"]["rollout"]["n"] == DEFAULT_GROUP_SIZE


def test_lora_hyperparameters_match_doc_5_1():
    """doc §5.1: "LoRA（r=16、alpha=32、...）"."""
    config = _load_config()
    model = config["actor_rollout_ref"]["model"]
    assert model["lora_rank"] == 16
    assert model["lora_alpha"] == 32
