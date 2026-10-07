"""数据清洗第一步：两层去重（对应 4.2 节「双层去重」）。

1. **MD5 精确去重**：把告警 payload 序列化成 canonical JSON（`sort_keys=True`），保证字段
   顺序不同但内容相同的两条告警会算出同一个哈希，完全一致的只保留先出现的一条。
2. **BGE 语义去重**：用 `sentence-transformers` 加载中文 BGE 模型
   （`BAAI/bge-small-zh-v1.5`）把 `annotations.summary + annotations.description` 编码成
   向量，两两算 cosine 相似度，> 0.92 判定语义重复，只保留先出现的一条。

## 降级说明（务必读）

如果当前环境装不上 `sentence-transformers`，或者装上了但模型下载不下来（无网络 / 无本地
缓存），本模块会**自动降级**成纯标准库的词法相似度（字符 bigram Jaccard 与
`difflib.SequenceMatcher.ratio` 取较大值），并用 `logging` 明确打一条 WARNING 说明走了哪条路径。

降级路径**只是为了让离线开发环境**（比如没有网络的 CI 容器、教学用的最小环境）也能跑通开发
迭代和单测，**不是生产替代品**——生产环境的语义去重效果依赖真实 embedding 模型捕捉同义改写 /
语序变化的能力，词法相似度做不到这一点，只能兜底措辞几乎照抄的明显重复。生产部署前必须确认
`sentence-transformers` + BGE 模型（或等价中文 embedding 模型）真的可用，参见
`docs/复现指南.md`（另一个任务产出）。

本环境实测记录：`sentence-transformers==5.6.0` + `torch==2.12.1` 已经在
`AIops-agent/.venv` 里装好，`BAAI/bge-small-zh-v1.5` 命中本地缓存
（`~/.cache/huggingface` / `~/.cache/modelscope`），真实加载 + 编码可以跑通（45s 首次加载，
后续走缓存更快）。但 `data/clean` 默认跑在系统 `/usr/bin/python3`（没装这些依赖），走的是
词法兜底路径——这正是本模块要优雅处理、而且默认单测要覆盖到的场景。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Optional, Sequence

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_SIM_THRESHOLD = 0.92


def canonical_json(alert: dict) -> str:
    """排序 key 的 canonical JSON 序列化：字段顺序不同、内容相同的 payload 会得到同一个字符串。"""
    return json.dumps(alert, sort_keys=True, ensure_ascii=False)


def alert_md5(alert: dict) -> str:
    return hashlib.md5(canonical_json(alert).encode("utf-8")).hexdigest()


def md5_dedup(alerts: list[dict]) -> tuple[list[dict], list[dict]]:
    """MD5 精确去重，保留先出现的一条。返回 (kept, dropped)。"""
    seen: set[str] = set()
    kept: list[dict] = []
    dropped: list[dict] = []
    for a in alerts:
        h = alert_md5(a)
        if h in seen:
            dropped.append(a)
        else:
            seen.add(h)
            kept.append(a)
    return kept, dropped


# ---------------------------------------------------------------------------
# 语义去重后端：真实 BGE 优先，失败时降级词法兜底
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingBackend:
    name: str  # "bge-real" | "lexical-fallback"
    encode: Optional[Callable[[Sequence[str]], object]] = None  # 只有 bge-real 有意义


_backend_cache: dict[str, EmbeddingBackend] = {}


def _char_ngram_jaccard(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 相似度：标准库实现，中文场景下比整词 Jaccard 更抗分词误差
    （不需要分词器），但也只能捕捉表面字符重叠，捕捉不到同义改写——这正是它只能当兜底、
    不能替代真实语义 embedding 的原因。"""

    def grams(s: str) -> set[str]:
        s = s.strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga and not gb:
        return 1.0
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _lexical_similarity(a: str, b: str) -> float:
    """词法相似度兜底：字符 bigram Jaccard 与 SequenceMatcher.ratio 取较大值，减少漏判。
    只在 BGE 不可用时启用，见模块 docstring 的降级说明。"""
    return max(_char_ngram_jaccard(a, b), SequenceMatcher(None, a, b).ratio())


def load_embedding_backend(
    model_name: str = DEFAULT_MODEL_NAME, force_fallback: bool = False
) -> EmbeddingBackend:
    """加载语义去重后端：优先尝试真实 BGE 模型，任何失败（import 失败 / 下载失败 /
    模型加载异常）都优雅降级为词法相似度兜底，并记录一条 WARNING 说明走了哪条路径。"""
    cache_key = "__fallback__" if force_fallback else model_name
    cached = _backend_cache.get(cache_key)
    if cached is not None:
        return cached

    if not force_fallback:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)

            def _encode(texts: Sequence[str]):
                return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

            backend = EmbeddingBackend(name="bge-real", encode=_encode)
            logger.info("语义去重后端：真实 BGE 模型加载成功 (%s)", model_name)
            _backend_cache[cache_key] = backend
            return backend
        except Exception as exc:  # noqa: BLE001 - 任何原因都要能优雅降级，不能让清洗流程崩掉
            logger.warning(
                "BGE 模型加载失败（%s: %s），降级为词法相似度兜底 —— 仅供离线开发使用，"
                "生产环境请确认 sentence-transformers + %s 可用（见 docs/复现指南.md）",
                type(exc).__name__,
                exc,
                model_name,
            )

    backend = EmbeddingBackend(name="lexical-fallback", encode=None)
    _backend_cache[cache_key] = backend
    return backend


def semantic_dedup(
    alerts: list[dict],
    threshold: float = DEFAULT_SIM_THRESHOLD,
    text_fn: Callable[[dict], str] = common.alert_text,
    backend: Optional[EmbeddingBackend] = None,
) -> tuple[list[dict], list[dict]]:
    """语义去重：两两相似度 > threshold 判定语义重复，只保留先出现的一条。返回 (kept, dropped)。"""
    backend = backend or load_embedding_backend()
    texts = [text_fn(a) for a in alerts]

    kept: list[dict] = []
    dropped: list[dict] = []

    if backend.name == "bge-real" and backend.encode is not None:
        import numpy as np

        vectors = np.asarray(backend.encode(texts))
        kept_vec_idx: list[int] = []
        for i, a in enumerate(alerts):
            is_dup = any(float(np.dot(vectors[i], vectors[j])) > threshold for j in kept_vec_idx)
            if is_dup:
                dropped.append(a)
            else:
                kept_vec_idx.append(i)
                kept.append(a)
        return kept, dropped

    kept_texts: list[str] = []
    for a, t in zip(alerts, texts):
        is_dup = any(_lexical_similarity(t, kt) > threshold for kt in kept_texts)
        if is_dup:
            dropped.append(a)
        else:
            kept.append(a)
            kept_texts.append(t)
    return kept, dropped


def dedup_pipeline(
    alerts: list[dict],
    threshold: float = DEFAULT_SIM_THRESHOLD,
    text_fn: Callable[[dict], str] = common.alert_text,
    backend: Optional[EmbeddingBackend] = None,
) -> dict:
    """完整双层去重管线：先 MD5 精确去重，再语义去重。返回各阶段明细，方便审计砍了哪些。"""
    after_md5, dropped_md5 = md5_dedup(alerts)
    backend = backend or load_embedding_backend()
    kept, dropped_semantic = semantic_dedup(after_md5, threshold=threshold, text_fn=text_fn, backend=backend)
    return {
        "kept": kept,
        "dropped_md5_exact": dropped_md5,
        "dropped_semantic": dropped_semantic,
        "backend_used": backend.name,
    }


def main() -> None:
    """真实 CLI：读 --in-dir 下的种子告警，跑双层去重，把幸存的写到 --out-dir。

    docs/复现指南.md 第 3 节给的正是 `python3 -m data.clean.dedup --in-dir ... --out-dir ...`
    这条命令——此前 `__main__` 只是打印一下当前会走哪条语义去重后端的 smoke test，
    并不会真的读写目录。
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", required=True, help="输入目录，读取其中的 seed_*.json")
    parser.add_argument("--out-dir", required=True, help="输出目录，写入幸存的 seed_*.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIM_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    backend = load_embedding_backend()
    logger.info("embedding backend in this environment: %s", backend.name)

    pairs = common.read_alerts_dir(args.in_dir)
    logger.info("读取 %d 条种子告警自 %s", len(pairs), args.in_dir)

    alerts = [a for _, a in pairs]
    result = dedup_pipeline(alerts, threshold=args.threshold, backend=backend)

    kept_ids = {id(a) for a in result["kept"]}
    kept_pairs = [(fname, a) for fname, a in pairs if id(a) in kept_ids]
    common.write_alerts_dir(kept_pairs, args.out_dir)

    logger.info(
        "去重完成：保留 %d 条，砍掉 MD5 精确重复 %d 条、语义重复 %d 条（后端=%s）-> 写入 %s",
        len(kept_pairs),
        len(result["dropped_md5_exact"]),
        len(result["dropped_semantic"]),
        result["backend_used"],
        args.out_dir,
    )


if __name__ == "__main__":
    main()
