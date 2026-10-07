"""``reward`` — pure rule-based, zero-annotation reward functions for the AIOps
diagnosis-agent GRPO training loop.

Everything under this package is pure Python: no GPU, no network, no live API
calls. It is fully unit-testable on a laptop. See ``reward/trajectory.py`` for
the shared ``Trajectory``/``Step`` data contract that every other module in
this package (and the future ``verl_adapter/`` rollout layer) is built around.
"""
