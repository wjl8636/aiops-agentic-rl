"""pytest 配置：把 data/clean/ 加入 sys.path（这样测试文件可以直接 `import dedup` /
`import filter as filter_mod` / `import split` / `import common`，跟被测模块自身用的
sys.path 自举方式保持一致），并提供加载 tests/fixtures/*.json 的小工具。

之所以不用 `data/clean/__init__.py` + 包相对导入：data/ 目录本身不归本任务管，不能往
data/__init__.py 这种会影响到 data/ 兄弟目录（seeds/ood_fixtures/cold_start）的地方加文件，
所以整个 data/clean/ 就设计成一组通过 sys.path 自举互相 `import` 的扁平模块，不组包。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CLEAN_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

if str(CLEAN_DIR) not in sys.path:
    sys.path.insert(0, str(CLEAN_DIR))


def load_fixture(name: str) -> dict:
    """按文件名（不含 .json 后缀也行）加载 tests/fixtures/ 下的一条合成告警。"""
    if not name.endswith(".json"):
        name = f"{name}.json"
    with open(FIXTURES_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_fixtures() -> dict[str, dict]:
    """加载 tests/fixtures/ 下的所有告警，key 是不带 .json 的文件名。"""
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(FIXTURES_DIR.glob("*.json"))}


@pytest.fixture
def fixtures() -> dict[str, dict]:
    return load_all_fixtures()
