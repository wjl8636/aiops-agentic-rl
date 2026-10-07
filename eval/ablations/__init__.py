"""Ablations reproducing, with real (if small-scale) computation, the two
training-time mechanisms the design doc credits with fixing concrete
failure modes (doc §6.4):

- ``no_prefix_split``: progressive prefix split (doc §4.4) vs. training on
  full trajectories only.
- ``no_credit_assignment``: fine-grained per-step credit assignment (doc
  §5.2 layer 3) vs. naive outcome-reward/T redistribution (doc §5.3 案例四).

Neither script retrains anything (no GPU/data at that scale exists in this
repo) — each measures the *exact mechanism* the doc blames for the
regression, on real data/real reward functions already in this repo, and
reports the measurable proxy honestly labeled as a proxy.
"""
