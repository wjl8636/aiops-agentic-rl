"""namespace_isolation.py: naming must be deterministic + collision-free
across worker ids, and the lock helper must actually exclude concurrent
access (tested with real threads, not just logic inspection).
"""
from __future__ import annotations

import re
import shutil
import threading
import time

import pytest

from verl_adapter import namespace_isolation as ns


# --------------------------------------------------------------------------
# Naming: deterministic + collision-free.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "derive",
    [ns.derive_k8s_namespace, ns.derive_docker_prefix, ns.derive_milvus_collection_suffix],
)
def test_derive_is_deterministic(derive):
    assert derive(3) == derive(3)
    assert derive(3, rollout_id="r1") == derive(3, rollout_id="r1")


@pytest.mark.parametrize(
    "derive",
    [ns.derive_k8s_namespace, ns.derive_docker_prefix, ns.derive_milvus_collection_suffix],
)
def test_derive_is_collision_free_across_worker_ids(derive):
    names = {derive(i) for i in range(200)}
    assert len(names) == 200


@pytest.mark.parametrize(
    "derive",
    [ns.derive_k8s_namespace, ns.derive_docker_prefix, ns.derive_milvus_collection_suffix],
)
def test_derive_is_collision_free_with_rollout_id(derive):
    names = {derive(worker_id=w, rollout_id=r) for w in range(20) for r in range(20)}
    assert len(names) == 400


def test_k8s_namespace_is_valid_dns_label():
    for worker_id in (0, 41, "worker-A", "Worker_1"):
        name = ns.derive_k8s_namespace(worker_id, rollout_id="rollout-99")
        assert len(name) <= 63
        assert re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", name), name


def test_milvus_collection_suffix_valid_charset():
    for worker_id in (0, 7, "worker-A"):
        suffix = ns.derive_milvus_collection_suffix(worker_id, rollout_id="rollout-1")
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", suffix), suffix


def test_worker_id_and_rollout_id_both_affect_the_name():
    assert ns.derive_k8s_namespace(1) != ns.derive_k8s_namespace(2)
    assert ns.derive_k8s_namespace(1, rollout_id="a") != ns.derive_k8s_namespace(1, rollout_id="b")


# --------------------------------------------------------------------------
# Lock: real mutual exclusion under concurrent threads.
# --------------------------------------------------------------------------

@pytest.fixture()
def lock_dir(tmp_path):
    d = tmp_path / "locks"
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_lock_excludes_concurrent_access(lock_dir):
    """Two threads race to increment a shared counter inside the lock's
    critical section. Without real exclusion, interleaved
    read-sleep-increment-write would corrupt the final count; with real
    exclusion, it must land on exactly n_iterations * n_threads.
    """
    counter = {"value": 0}
    n_threads = 6
    n_iterations = 20

    def worker(worker_id: int) -> None:
        for _ in range(n_iterations):
            with ns.acquire_worker_lock(worker_id, "shared-resource", lock_dir=lock_dir):
                current = counter["value"]
                time.sleep(0.0005)  # widen the race window so a broken lock would show corruption
                counter["value"] = current + 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter["value"] == n_threads * n_iterations


def test_lock_is_reentrant_safe_for_sequential_acquisitions(lock_dir):
    for _ in range(3):
        with ns.acquire_worker_lock("w0", "res", lock_dir=lock_dir):
            pass  # must not deadlock on repeated sequential acquire/release


def test_lock_different_resource_keys_do_not_block_each_other(lock_dir):
    """Locking resource A does not block a concurrent acquisition of
    resource B (matches the docstring: locking is keyed on resource_key,
    not worker_id)."""
    order: list[str] = []
    release_a = threading.Event()
    entered_b = threading.Event()

    def hold_a():
        with ns.acquire_worker_lock("w0", "resource-a", lock_dir=lock_dir):
            order.append("a-enter")
            release_a.wait(timeout=2)
            order.append("a-exit")

    def take_b():
        entered_b.wait(timeout=0.1)  # let hold_a start first
        with ns.acquire_worker_lock("w1", "resource-b", lock_dir=lock_dir):
            order.append("b-enter")

    t_a = threading.Thread(target=hold_a)
    t_b = threading.Thread(target=take_b)
    t_a.start()
    time.sleep(0.05)
    entered_b.set()
    t_b.start()
    t_b.join(timeout=2)
    assert "b-enter" in order  # resource-b's lock was NOT blocked by resource-a's holder
    release_a.set()
    t_a.join(timeout=2)


def test_lock_timeout_raises_when_contended(lock_dir):
    holder_ready = threading.Event()
    release = threading.Event()

    def holder():
        with ns.acquire_worker_lock("w0", "contended", lock_dir=lock_dir):
            holder_ready.set()
            release.wait(timeout=2)

    t = threading.Thread(target=holder)
    t.start()
    holder_ready.wait(timeout=2)
    try:
        with pytest.raises(ns.WorkerLockTimeout):
            with ns.acquire_worker_lock("w1", "contended", lock_dir=lock_dir, timeout=0.2):
                pass
    finally:
        release.set()
        t.join(timeout=2)
