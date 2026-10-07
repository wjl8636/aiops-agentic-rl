"""让 tests/ 目录能直接 `import collect_trajectories` / `rejection_sample` / `prefix_split` /
`to_llamafactory_format` / `signals`——它们都在 tests/ 的上一级目录（data/cold_start/），
不是一个安装过的 package，所以手动把上一级目录塞进 sys.path。
"""
from __future__ import annotations

import sys
from pathlib import Path

COLD_START_DIR = Path(__file__).resolve().parent.parent
if str(COLD_START_DIR) not in sys.path:
    sys.path.insert(0, str(COLD_START_DIR))
