"""``grpo`` — the veRL-facing GRPO wiring layer for the AIOps diagnosis agent.

This package does not implement any reward *logic* (that's the sibling
``reward/`` package's job — see its module docstring). It only *composes*
``reward/``'s functions into the single scalar/per-step reward signal a
GRPO training loop actually consumes, and holds the veRL-facing config/
launch scripts. See ``grpo/reward_router.py`` for the composition and
``grpo/verl_config/aiops_grpo.yaml`` / ``grpo/train_grpo.sh`` for the
(unexecuted-this-session, veRL-not-installed) training wiring.
"""
