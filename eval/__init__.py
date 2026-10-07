"""``eval`` — RL-specific evaluation harness for the AIOps diagnosis-agent
SFT + GRPO project.

This package is intentionally NOT a copy of ``AIops-agent/eval/`` (which
evaluates one already-deployed Claude-Opus-backed agent against a fixed
``expected.json``). This package instead answers the questions the design
doc's 六/评测方法 section asks about *training progress*:

- how does a candidate policy compare against zero-shot / SFT / SFT+GRPO /
  Opus-reference tiers on the same labeled eval set? (``run_baseline_comparison.py``)
- does the policy generalize to out-of-distribution fault types, or did it
  just memorize the training distribution? (``metrics.ood_generalization_ratio``)
- do the two training-time mechanisms the doc credits with fixing concrete
  failure modes (progressive prefix split, fine-grained credit assignment)
  actually move the numbers they claim to move? (``ablations/``)

Everything here runs against a **pluggable model-endpoint interface**
(``model_endpoints.DiagnosisEndpoint``) so the mock endpoints used for
testing in this repo can later be swapped for real vLLM/Opus adapters
without touching ``run_eval.py``/``run_baseline_comparison.py``.

IMPORTANT: the only labeled ground-truth data available in this repo today
is the 5 trajectories under ``data/cold_start/trajectories/mock/`` (paired
with ``data/cold_start/ground_truth/``). Every accuracy number this package
produces against that 5-item set is illustrative and proves the harness
*works*, not a reproduction of the design doc's reported percentages (those
require real trained checkpoints and the doc's full 100/50/40-item eval
sets). See each module's docstring for the same disclaimer inline.
"""
