"""Per-worker namespace/resource-prefix derivation + a file-lock-based
mutual-exclusion helper, standing in for "give each GRPO rollout worker its
own namespace or a scheduling lock" (transcript §十: "GRPO 一组 6 条轨迹是
并发跑的，多个 rollout worker 同时打真 kind 集群和真 Milvus 会有资源竞
争——同一个 pod 被 rollout A restart 的时候 rollout B 读到的状态是脏的，得
给每个 worker 起独立 namespace 或者加一层调度锁").

Two independent tools, meant to be used together or separately depending on
the resource:
- ``derive_k8s_namespace`` / ``derive_docker_prefix`` /
  ``derive_milvus_collection_suffix``: give each worker a deterministic,
  collision-resistant, backend-valid-charset name so concurrent workers
  never touch the same k8s namespace / docker container name / Milvus
  collection.
- ``acquire_worker_lock``: for resources that CAN'T be trivially
  namespaced (e.g. a single shared kind cluster's node capacity, or a
  shared Milvus connection pool) a real ``flock``-based mutual-exclusion
  lock a worker holds for the duration of a critical section.

Neither of these needs a live cluster to be correct — they're pure naming
logic + a real (but locally testable) file lock, which is exactly what the
task calls for: "doesn't need a live cluster to be correctly implemented
and unit tested".
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_LOCK_DIR = Path(__file__).resolve().parent / ".locks"

#: k8s namespace hard limit (DNS label): 63 chars, [a-z0-9]([-a-z0-9]*[a-z0-9])?
_K8S_NAMESPACE_MAX_LEN = 63
#: Conservative shared cap for docker container-name prefixes / Milvus
#: collection names too (docker allows up to 128 chars; Milvus's limit is
#: implementation-defined but comfortably above 63) — reuse one constant so
#: all three derivations behave consistently.
_MAX_LEN = 63


def _slug(value: object, sep: str = "-") -> str:
    """Lowercase, ``[a-z0-9]`` + ``sep`` only, never empty. ``sep`` must
    itself already be a valid separator char for the target charset (``-``
    for k8s/docker, ``_`` for Milvus) — using the caller's separator here
    (instead of always ``-``) keeps the whole derived name in one
    consistent, valid charset rather than mixing ``-`` and ``_``.
    """
    s = re.sub(r"[^a-z0-9]+", sep, str(value).strip().lower()).strip(sep)
    return s or "x"


def _stable_short_hash(*parts: object, length: int = 8) -> str:
    """Deterministic short hash of ``parts`` — guarantees two different
    ``(worker_id, rollout_id)`` inputs get different derived names even if
    their slugs happen to collapse to the same string (e.g. ``"Worker-1"``
    and ``"worker_1"`` both slug to ``"worker-1"``, but their raw
    representations differ, so the hash differs too).
    """
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:length]


def _derive(prefix: str, worker_id: object, rollout_id: Optional[object], sep: str, max_len: int) -> str:
    slug = _slug(worker_id, sep) if rollout_id is None else f"{_slug(worker_id, sep)}{sep}{_slug(rollout_id, sep)}"
    digest = _stable_short_hash(worker_id, rollout_id)
    name = f"{prefix}{sep}{slug}{sep}{digest}"
    if len(name) > max_len:
        # Fall back to a shorter, still-collision-resistant form: prefix + hash only.
        name = f"{prefix}{sep}{digest}"[:max_len]
    return name.strip(sep)


def derive_k8s_namespace(worker_id: object, rollout_id: Optional[object] = None, prefix: str = "aiops-rl") -> str:
    """Deterministic per-(worker_id[, rollout_id]) k8s namespace name.

    Valid as a k8s namespace (DNS label charset, <=63 chars, no leading/
    trailing ``-``) and collision-resistant across worker ids because a
    stable hash of the *raw* inputs is folded into the name (see
    ``_stable_short_hash``), not just the human-readable slug.
    """
    return _derive(prefix, worker_id, rollout_id, "-", _K8S_NAMESPACE_MAX_LEN)


def derive_docker_prefix(worker_id: object, rollout_id: Optional[object] = None, prefix: str = "aiops-rl") -> str:
    """Deterministic per-worker docker container-name prefix. Docker names
    allow a wider charset than k8s namespaces, but we reuse the same
    conservative DNS-safe slug so one naming scheme covers both backends
    (matches ``remediation.py``'s "两套后端...对后端无感知" design principle).
    """
    return _derive(prefix, worker_id, rollout_id, "-", _MAX_LEN)


def derive_milvus_collection_suffix(worker_id: object, rollout_id: Optional[object] = None, prefix: str = "aiops_rl") -> str:
    """Deterministic per-worker Milvus collection-name suffix. Milvus
    collection names must start with a letter/underscore and contain only
    ``[A-Za-z0-9_]`` (no hyphens) — same derivation as the other two, with
    ``_`` as the separator instead of ``-``, and a forced-underscore prefix
    if the result would otherwise start with a digit.
    """
    name = _derive(prefix, worker_id, rollout_id, "_", _MAX_LEN)
    if not re.match(r"^[A-Za-z_]", name):
        name = "_" + name
    return name


class WorkerLockTimeout(TimeoutError):
    """Raised by ``acquire_worker_lock`` when ``timeout`` elapses before the
    lock could be acquired."""


def _lock_path(resource_key: object, lock_dir: Optional[Path]) -> Path:
    lock_dir = Path(lock_dir) if lock_dir is not None else DEFAULT_LOCK_DIR
    lock_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(resource_key)) or "resource"
    return lock_dir / f"{safe}.lock"


@contextlib.contextmanager
def acquire_worker_lock(
    worker_id: object,
    resource_key: object,
    lock_dir: Optional[Path] = None,
    timeout: Optional[float] = None,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    """Real, file-lock-based (``fcntl.flock``) mutual exclusion for
    concurrent GRPO rollout workers contending over the SAME shared
    resource (e.g. ``resource_key="kind-cluster"`` or ``"milvus"``).

    Locking is keyed on ``resource_key``, not ``worker_id``: any two callers
    (regardless of which worker they are) that pass the same
    ``resource_key`` block each other; the same worker locking two
    *different* ``resource_key``s does not conflict. ``worker_id`` is
    recorded into the lock file purely for debugging (so a stuck lock file
    can be inspected to see who's holding it) — it plays no role in which
    resource is exclusive.

    Usage::

        with acquire_worker_lock(worker_id, "kind-cluster"):
            ...  # critical section: only one worker at a time runs this

    Raises ``WorkerLockTimeout`` if ``timeout`` is given and elapses before
    the lock is acquired; blocks indefinitely if ``timeout is None``
    (default).
    """
    import fcntl  # POSIX-only; this project's target environments (mac/linux CI, GRPO training boxes) are POSIX.

    path = _lock_path(resource_key, lock_dir)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise WorkerLockTimeout(
                        f"worker {worker_id!r} timed out after {timeout}s waiting for lock on {resource_key!r}"
                    )
                time.sleep(poll_interval)
        os.ftruncate(fd, 0)
        os.write(fd, str(worker_id).encode("utf-8"))
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
