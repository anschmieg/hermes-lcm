"""Opt-in asynchronous compaction preparation and atomic publication.

This module owns the feature's sidecar tables and its bounded worker.  It never
writes a canonical summary while preparing a batch.  Publication is deliberately
implemented with one connection and one ``BEGIN IMMEDIATE`` transaction so the
summary rows, FTS trigger effects, lifecycle frontier, and batch state cannot
become partially visible.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping

from .db_bootstrap import (
    configure_connection,
    ensure_foreground_compaction_claim_tables,
    ensure_async_compaction_tables,
    ensure_temporal_rollup_invalidation_triggers,
)
from .fresh_tail import resolve_fresh_tail_boundary
from .message_content import stored_text_content_for_pattern_matching
from .message_patterns import compile_message_patterns, matches_message_pattern
from .escalation import SummaryCircuitBreaker, SummarySpendGuard
from .dag import SummaryDAG, SummaryNode
from .store import MessageStore
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

_PLUGIN_VERSION: str | None = None

def _read_plugin_version() -> str:
    global _PLUGIN_VERSION
    if _PLUGIN_VERSION is not None:
        return _PLUGIN_VERSION
    try:
        manifest_path = Path(__file__).parent.parent / "plugin.yaml"
        text = manifest_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("version:"):
                _PLUGIN_VERSION = line.split(":", 1)[1].strip()
                break
        if _PLUGIN_VERSION is None:
            _PLUGIN_VERSION = "unknown"
    except Exception:
        _PLUGIN_VERSION = "unknown"
    return _PLUGIN_VERSION

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "async_compaction_protocol_v1"
_ACTIVE_BATCH_STATES = ("pending", "preparing", "ready")
_WORKER_CLOSE_TIMEOUT_SECONDS = 0.25


class _ManagerOperation:
    """Lease that keeps a manager connection usable until its caller returns."""

    def __init__(self, manager: "AsyncCompactionManager"):
        self._manager = manager
        self._released = False

    def release(self, *, completed_worker: bool = False) -> None:
        if self._released:
            return
        self._released = True
        self._manager._end_operation(completed_worker=completed_worker)

    def __enter__(self) -> "_ManagerOperation":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


@dataclass
class CompactionBatch:
    batch_id: str
    conversation_id: str
    session_id: str
    state: str
    frontier_start_store_id: int
    frontier_end_store_id: int
    fresh_tail_count: int
    leaf_chunk_tokens: int
    policy_fingerprint: str
    summary_route_fingerprint: str
    source_coverage_hash: str
    expected_leaf_count: int
    prepared_leaf_count: int = 0
    source_ids: List[int] = field(default_factory=list)
    source_identity_hashes: List[str] = field(default_factory=list)
    failure_count: int = 0
    next_retry_at: float | None = None
    last_error: str = ""
    rejected_reason: str = ""
    lease_owner: str = ""
    lease_expires_at: float | None = None


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    reason: str = ""
    batch_id: str = ""
    node_ids: tuple[int, ...] = ()
    frontier_store_id: int = 0


@dataclass(frozen=True)
class _BackgroundSnapshot:
    messages: List[Dict[str, Any]]
    session_id: str
    conversation_id: str
    upper_store_id: int = 0
    config: Any = None
    model: str = ""
    provider: str = ""
    base_url: str = ""
    api_mode: str = ""
    hermes_home: str = ""
    session_ignored: bool = False
    session_stateless: bool = False
    threshold_tokens: int = 0
    raw_context_length: int = 0
    binding_generation: int = 0
    manager: Any = None
    store: Any = None
    dag: Any = None
    lifecycle: Any = None


@dataclass(frozen=True)
class _BackgroundTrigger:
    """Small immutable reply-path signal; durable state is read by the worker."""

    session_id: str
    conversation_id: str


@dataclass(frozen=True)
class _PublishedNodeMaintenance:
    """Immutable publication data handed to post-commit maintenance."""

    node_id: int
    session_id: str
    depth: int
    summary: str
    token_count: int
    source_token_count: int
    source_ids: tuple[int, ...]
    created_at: float
    earliest_at: float | None
    latest_at: float | None
    expand_hint: str


@dataclass(frozen=True)
class _PromotionMaintenancePayload:
    """Detached, immutable inputs for maintenance after publication."""

    nodes: tuple[_PublishedNodeMaintenance, ...]
    source_messages: tuple[tuple[int, Mapping[str, Any]], ...]
    snapshot: _BackgroundSnapshot
    db_path: str
    publication_dag: Any = None
    publication_store: Any = None


def _freeze_maintenance_value(value: Any) -> Any:
    """Copy JSON-like row data into immutable containers for maintenance."""
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_maintenance_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_maintenance_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_maintenance_value(item) for item in value)
    return value


class _BoundedBackgroundWorker:
    """One daemon worker with non-blocking bounded enqueue and safe draining."""

    def __init__(
        self,
        callback: Callable[[Any], None],
        max_items: int,
        on_exit: Callable[[], None] | None = None,
    ):
        self._callback = callback
        self._on_exit = on_exit
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, max_items))
        self._condition = threading.Condition()
        self._active = False
        self._stopping = False
        self._thread: threading.Thread | None = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def active(self) -> bool:
        with self._condition:
            return self._active

    def _try_enqueue(self, trigger: Any) -> bool:
        """Append without waiting on Queue's internal mutex."""
        if not self._queue.mutex.acquire(blocking=False):
            return False
        try:
            if self._queue.maxsize > 0 and len(self._queue.queue) >= self._queue.maxsize:
                return False
            self._queue.queue.append(trigger)
            self._queue.unfinished_tasks += 1
            self._queue.not_empty.notify()
            return True
        finally:
            self._queue.mutex.release()

    def enqueue(self, trigger: Any) -> bool:
        if not self._condition.acquire(blocking=False):
            return False
        try:
            if self._stopping:
                return False
            if not self._try_enqueue(trigger):
                return False
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name="lcm-async-compaction",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify_all()
            return True
        finally:
            self._condition.release()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._queue.empty() and not self._stopping:
                        self._condition.wait(0.1)
                    if self._stopping and self._queue.empty():
                        return
                    # Reserve the next queue item before releasing the condition.
                    # A drain racing the actual dequeue now observes the worker as
                    # busy instead of mistaking the empty queue for idle.
                    self._active = True
                try:
                    snapshot = self._queue.get_nowait()
                except queue.Empty:
                    with self._condition:
                        self._active = False
                        self._condition.notify_all()
                    continue
                try:
                    self._callback(snapshot)
                except Exception:
                    logger.warning("LCM async compaction worker job failed", exc_info=True)
                finally:
                    self._queue.task_done()
                    with self._condition:
                        self._active = False
                        self._condition.notify_all()
        finally:
            if self._on_exit is not None:
                try:
                    self._on_exit()
                except Exception:
                    logger.warning("LCM async compaction worker finalizer failed", exc_info=True)

    def drain(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._queue.empty() or self._active:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout: float | None = None) -> bool:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=None if timeout is None else max(0.0, timeout))
        return not thread.is_alive()


class AsyncCompactionManager:
    """Persistent batch lifecycle plus optional background preparation."""

    def __init__(self, engine: Any):
        self._engine = engine
        self._lock = threading.RLock()
        self._owner_id = uuid.uuid4().hex
        self._db_path = str(engine._store.db_path)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(engine._store.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        self._background_tables_enabled = bool(
            getattr(engine._config, "async_background_compaction_enabled", False)
        )
        if self._background_tables_enabled:
            ensure_async_compaction_tables(self._conn)
        else:
            ensure_foreground_compaction_claim_tables(self._conn)
        if bool(getattr(engine._config, "temporal_rollups_enabled", False)):
            ensure_temporal_rollup_invalidation_triggers(self._conn)
        if self._background_tables_enabled:
            self._recover_incomplete_batches()
        worker_enabled = bool(
            getattr(engine._config, "async_background_compaction_worker_enabled", False)
        )
        self._worker: _BoundedBackgroundWorker | None = (
            _BoundedBackgroundWorker(
                self._run_trigger,
                max_items=max(
                    1,
                    int(getattr(engine._config, "async_background_compaction_max_batches", 2) or 2),
                ),
                on_exit=self._finalize_worker_exit,
            )
            if worker_enabled and self._background_tables_enabled
            else None
        )
        self._enqueued_jobs = 0
        self._dropped_jobs = 0
        self._closed = False
        self._close_requested = False
        self._operation_inflight = 0
        self._foreground_claim = threading.local()

    @property
    def connection(self) -> sqlite3.Connection | None:
        return self._conn

    def _recover_incomplete_batches(self) -> None:
        conn = self._conn
        assert conn is not None
        now = time.time()
        conn.execute("BEGIN IMMEDIATE")
        try:
            stale_rows = conn.execute(
                """
                SELECT batch_id FROM compaction_batches
                WHERE state IN ('preparing', 'promoting')
                  AND (
                      lease_owner IS NULL
                      OR lease_expires_at IS NULL
                      OR lease_expires_at <= ?
                  )
                """,
                (now,),
            ).fetchall()
            stale_ids = [str(row[0]) for row in stale_rows]
            if stale_ids:
                placeholders = ",".join("?" for _ in stale_ids)
                conn.execute(
                    f"""
                    UPDATE compaction_batches
                    SET state = 'pending', prepared_leaf_count = 0,
                        lease_owner = NULL, lease_expires_at = NULL,
                        last_error = 'stale async preparation lease reclaimed',
                        updated_at = ?
                    WHERE batch_id IN ({placeholders})
                    """,
                    [now, *stale_ids],
                )
                conn.execute(
                    f"DELETE FROM pending_summary_nodes WHERE batch_id IN ({placeholders})",
                    stale_ids,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _hash_json(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _setting(config: Any, name: str, default: Any) -> Any:
        value = getattr(config, name, None)
        return default if value is None else value

    def _policy_fingerprint(self, snapshot: _BackgroundSnapshot | None = None) -> str:
        config = snapshot.config if snapshot is not None and snapshot.config is not None else self._engine._config
        runtime_threshold_tokens = getattr(self._engine, "threshold_tokens", 0)
        raw_context_length = getattr(self._engine, "raw_context_length", 0)
        if snapshot is not None:
            runtime_threshold_tokens = getattr(snapshot, "threshold_tokens", runtime_threshold_tokens)
            raw_context_length = getattr(snapshot, "raw_context_length", raw_context_length)
        policy = {
            "protocol": _PROTOCOL_VERSION,
            "fresh_tail_count": int(getattr(config, "fresh_tail_count", 0) or 0),
            "fresh_tail_max_tokens": int(getattr(config, "fresh_tail_max_tokens", 0) or 0),
            "leaf_chunk_tokens": int(getattr(config, "leaf_chunk_tokens", 0) or 0),
            "context_threshold": float(getattr(config, "context_threshold", 0.0) or 0.0),
            "runtime_threshold_tokens": int(runtime_threshold_tokens or 0),
            "raw_context_length": int(raw_context_length or 0),
            "dynamic_leaf_chunk_enabled": bool(getattr(config, "dynamic_leaf_chunk_enabled", False)),
            "dynamic_leaf_chunk_max": int(getattr(config, "dynamic_leaf_chunk_max", 0) or 0),
            "ignore_message_patterns": list(getattr(config, "ignore_message_patterns", []) or []),
            "ignore_message_patterns_source": str(
                getattr(config, "ignore_message_patterns_source", "default") or "default"
            ),
            "sensitive_patterns_enabled": bool(getattr(config, "sensitive_patterns_enabled", False)),
            "sensitive_patterns": list(getattr(config, "sensitive_patterns", []) or []),
            "sensitive_patterns_source": str(
                getattr(config, "sensitive_patterns_source", "default") or "default"
            ),
            "large_output_externalization_enabled": bool(
                getattr(config, "large_output_externalization_enabled", False)
            ),
            "large_output_externalization_threshold_chars": int(
                getattr(config, "large_output_externalization_threshold_chars", 0) or 0
            ),
            "large_output_externalization_path": str(
                getattr(config, "large_output_externalization_path", "") or ""
            ),
            "custom_instructions": str(getattr(config, "custom_instructions", "") or ""),
            "l2_budget_ratio": float(getattr(config, "l2_budget_ratio", 0.0) or 0.0),
            "l3_truncate_tokens": int(getattr(config, "l3_truncate_tokens", 0) or 0),
            "summary_circuit_breaker_failure_threshold": int(
                self._setting(config, "summary_circuit_breaker_failure_threshold", 2)
            ),
            "summary_circuit_breaker_cooldown_seconds": int(
                self._setting(config, "summary_circuit_breaker_cooldown_seconds", 300)
            ),
            "summary_spend_max_calls": int(
                self._setting(config, "summary_spend_max_calls", 24)
            ),
            "summary_spend_window_seconds": float(
                self._setting(config, "summary_spend_window_seconds", 600.0)
            ),
            "summary_spend_backoff_seconds": float(
                self._setting(config, "summary_spend_backoff_seconds", 1800.0)
            ),
            "ignore_session_patterns": list(getattr(config, "ignore_session_patterns", []) or []),
            "ignore_session_patterns_source": str(
                getattr(config, "ignore_session_patterns_source", "default") or "default"
            ),
            "stateless_session_patterns": list(getattr(config, "stateless_session_patterns", []) or []),
            "stateless_session_patterns_source": str(
                getattr(config, "stateless_session_patterns_source", "default") or "default"
            ),
            "temporal_rollups_enabled": bool(
                getattr(config, "temporal_rollups_enabled", False)
            ),
            "large_output_transcript_gc_enabled": bool(
                getattr(config, "large_output_transcript_gc_enabled", False)
            ),
            "embeddings_enabled": bool(getattr(config, "embeddings_enabled", False)),
        }
        return self._hash_json(policy)

    def _summary_route_fingerprint(self, snapshot: _BackgroundSnapshot | None = None) -> str:
        config = snapshot.config if snapshot is not None and snapshot.config is not None else self._engine._config
        route = {
            "protocol": _PROTOCOL_VERSION,
            "summary_model": str(getattr(config, "summary_model", "") or ""),
            "summary_fallback_models": list(getattr(config, "summary_fallback_models", []) or []),
            "provider": str(
                getattr(snapshot, "provider", "") if snapshot is not None else getattr(self._engine, "provider", "")
            ) or "",
            "model": str(
                getattr(snapshot, "model", "") if snapshot is not None else getattr(self._engine, "model", "")
            ) or "",
            "base_url": str(
                getattr(snapshot, "base_url", "") if snapshot is not None else getattr(self._engine, "base_url", "")
            ) or "",
            "api_mode": str(
                getattr(snapshot, "api_mode", "") if snapshot is not None else getattr(self._engine, "api_mode", "")
            ) or "",
            "summary_timeout_ms": int(getattr(config, "summary_timeout_ms", 0) or 0),
            "plugin_version": _read_plugin_version(),
        }
        return self._hash_json(route)

    def _lease_seconds(self, config: Any) -> float:
        configured = float(
            getattr(config, "async_background_compaction_lease_seconds", 0.0) or 0.0
        )
        summary_window = float(getattr(config, "summary_timeout_ms", 60_000) or 60_000) / 1000.0 + 30.0
        return max(1.0, configured, summary_window)

    def claim_foreground_sources(
        self,
        *,
        conversation_id: str,
        session_id: str,
        source_ids: list[int],
        config: Any,
    ) -> bool:
        """Claim one synchronous source window across managers/processes."""
        if self._closed or not source_ids:
            return False
        conn = self._conn
        assert conn is not None
        now = time.time()
        lease_expires_at = now + self._lease_seconds(config)
        source_ids = sorted(dict.fromkeys(int(value) for value in source_ids))
        placeholders = ",".join("?" for _ in source_ids)
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    """
                    SELECT owner_id, lease_expires_at, fencing_token
                    FROM foreground_compaction_claims
                    WHERE conversation_id = ? AND session_id = ?
                    """,
                    (conversation_id, session_id),
                ).fetchone()
                if existing and str(existing["owner_id"] or "") != self._owner_id:
                    if existing["lease_expires_at"] is not None and float(
                        existing["lease_expires_at"]
                    ) > now:
                        conn.rollback()
                        return False

                overlap = conn.execute(
                    f"""
                    SELECT 1
                    FROM summary_nodes AS node, json_each(node.source_ids) AS source
                    WHERE node.session_id = ?
                      AND node.source_type = 'messages'
                      AND CAST(source.value AS INTEGER) IN ({placeholders})
                    LIMIT 1
                    """,
                    [session_id, *source_ids],
                ).fetchone()
                if overlap:
                    conn.rollback()
                    return False

                existing_owner = str(existing["owner_id"] or "") if existing else ""
                existing_expires = (
                    float(existing["lease_expires_at"])
                    if existing and existing["lease_expires_at"] is not None
                    else 0.0
                )
                existing_token = int(existing["fencing_token"] or 0) if existing else 0
                if (
                    existing_owner == self._owner_id
                    and existing_token > 0
                    and existing_expires > now
                ):
                    fencing_token = existing_token
                else:
                    conn.execute(
                        "UPDATE foreground_compaction_fence "
                        "SET next_token = next_token + 1 WHERE fence_id = 1"
                    )
                    token_row = conn.execute(
                        "SELECT next_token FROM foreground_compaction_fence WHERE fence_id = 1"
                    ).fetchone()
                    if token_row is None:
                        raise RuntimeError("foreground compaction fence sequence is missing")
                    fencing_token = int(token_row[0])

                conn.execute(
                    """
                    INSERT INTO foreground_compaction_claims(
                        conversation_id, session_id, owner_id, lease_expires_at,
                        fencing_token
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id, session_id) DO UPDATE SET
                        owner_id = excluded.owner_id,
                        lease_expires_at = excluded.lease_expires_at,
                        fencing_token = excluded.fencing_token
                    """,
                    (
                        conversation_id,
                        session_id,
                        self._owner_id,
                        lease_expires_at,
                        fencing_token,
                    ),
                )
                conn.commit()
                self._foreground_claim.value = (
                    conversation_id,
                    session_id,
                    fencing_token,
                )
                return True
            except Exception:
                conn.rollback()
                raise

    def foreground_claim_token(self) -> int | None:
        claim = getattr(self._foreground_claim, "value", None)
        return int(claim[2]) if claim is not None else None

    def capture_foreground_source_identity_hashes(
        self,
        *,
        conversation_id: str,
        session_id: str,
        source_ids: list[int],
    ) -> dict[int, str] | None:
        """Capture source identities that publication must still match."""
        if self._closed or self._conn is None or not source_ids:
            return None
        source_ids = sorted(dict.fromkeys(int(value) for value in source_ids))
        placeholders = ",".join("?" for _ in source_ids)
        with self._lock:
            conn = self._conn
            if conn is None:
                return None
            rows = conn.execute(
                f"""
                SELECT * FROM messages
                WHERE store_id IN ({placeholders})
                ORDER BY store_id
                """,
                source_ids,
            ).fetchall()
            if [int(row["store_id"]) for row in rows] != source_ids:
                return None
            if any(
                str(row["session_id"] or "") != session_id
                or (
                    str(row["conversation_id"] or "")
                    and str(row["conversation_id"] or "") != conversation_id
                )
                for row in rows
            ):
                return None
            return {
                int(row["store_id"]): self._source_identity_hash(row)
                for row in rows
            }

    def release_foreground_claims(self, claim_token: int | None = None) -> None:
        if self._conn is None:
            return
        with self._lock:
            try:
                token = claim_token
                claim = getattr(self._foreground_claim, "value", None)
                if token is None and claim is not None:
                    token = int(claim[2])
                if token is None:
                    self._conn.execute(
                        "DELETE FROM foreground_compaction_claims WHERE owner_id = ?",
                        (self._owner_id,),
                    )
                else:
                    self._conn.execute(
                        "DELETE FROM foreground_compaction_claims "
                        "WHERE owner_id = ? AND fencing_token = ?",
                        (self._owner_id, int(token)),
                    )
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()
            finally:
                if claim is None or token is None or int(claim[2]) == int(token):
                    self._foreground_claim.value = None

    def _begin_operation(self) -> bool:
        with self._lock:
            if self._closed or self._conn is None:
                return False
            self._operation_inflight += 1
            return True

    def _acquire_operation(self) -> _ManagerOperation | None:
        if not self._begin_operation():
            return None
        return _ManagerOperation(self)

    def _end_operation(self, *, completed_worker: bool = False) -> None:
        with self._lock:
            self._operation_inflight = max(0, self._operation_inflight - 1)
            self._finish_close_locked(completed_worker=completed_worker)

    @contextmanager
    def _publication_lock(self):
        """Acquire engine state before manager state for atomic publication."""
        state_lock = getattr(self._engine, "_async_state_lock", None)
        lock = state_lock if state_lock is not None else threading.Lock()
        with lock:
            with self._lock:
                yield

    def _snapshot_binding_is_current(self, snapshot: _BackgroundSnapshot) -> bool:
        """Check the binding captured with a snapshot while state is fenced."""
        return bool(
            not self._closed
            and not bool(getattr(self._engine, "_shutdown_started", False))
            and int(getattr(self._engine, "_binding_generation", -1))
            == snapshot.binding_generation
            and snapshot.manager is self
            and getattr(self._engine, "_async_compaction", None) is self
            and getattr(self._engine, "_store", None) is snapshot.store
            and getattr(self._engine, "_dag", None) is snapshot.dag
            and getattr(self._engine, "_lifecycle", None) is snapshot.lifecycle
            and str(getattr(self._engine, "_session_id", "") or "")
            == snapshot.session_id
            and str(getattr(self._engine, "_conversation_id", "") or "")
            == snapshot.conversation_id
        )

    def begin_foreground_operation(self) -> bool:
        """Compatibility wrapper over the shared manager operation lifetime."""
        return self._begin_operation()

    def invalidate_foreground_claim(
        self,
        *,
        conversation_id: str,
        session_id: str,
    ) -> None:
        """Fence claims for an identity that is leaving this engine binding."""
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM foreground_compaction_claims "
                    "WHERE conversation_id = ? AND session_id = ?",
                    (conversation_id, session_id),
                )
                self._conn.commit()
            except sqlite3.Error:
                self._conn.rollback()

    def end_foreground_operation(self) -> None:
        self._end_operation()

    def _begin_worker_operation(self) -> bool:
        """Reserve this manager before a worker callback can read SQLite."""
        return self._begin_operation()

    def _end_worker_operation(self) -> None:
        """Release the worker's connection lease after its final SQLite read."""
        self._end_operation(completed_worker=True)

    def publish_foreground_node(
        self,
        node: SummaryNode,
        *,
        conversation_id: str,
        session_id: str,
        claim_token: int | None,
        frontier_store_id: int = 0,
        expected_generation: int | None = None,
        config: Any | None = None,
        expected_source_identity_hashes: list[str] | None = None,
    ) -> bool:
        """Publish one foreground node only while its SQLite fence is valid."""
        if self._closed or self._conn is None or claim_token is None:
            return False
        source_ids = sorted(dict.fromkeys(int(value) for value in node.source_ids))
        if not source_ids:
            return False
        placeholders = ",".join("?" for _ in source_ids)
        now = time.time()
        with self._lock:
            conn = self._conn
            if conn is None or self._closed:
                return False
            conn.execute("BEGIN IMMEDIATE")
            try:
                if (
                    expected_generation is not None
                    and int(getattr(self._engine, "_binding_generation", -1))
                    != int(expected_generation)
                ):
                    conn.rollback()
                    return False
                claim = conn.execute(
                    """
                    SELECT owner_id, fencing_token, lease_expires_at
                    FROM foreground_compaction_claims
                    WHERE conversation_id = ? AND session_id = ?
                    """,
                    (conversation_id, session_id),
                ).fetchone()
                if (
                    claim is None
                    or str(claim["owner_id"] or "") != self._owner_id
                    or int(claim["fencing_token"] or 0) != int(claim_token)
                    or claim["lease_expires_at"] is None
                    or float(claim["lease_expires_at"]) <= now
                ):
                    conn.rollback()
                    return False
                renewed = conn.execute(
                    """
                    UPDATE foreground_compaction_claims
                    SET lease_expires_at = ?
                    WHERE conversation_id = ? AND session_id = ?
                      AND owner_id = ? AND fencing_token = ?
                      AND lease_expires_at > ?
                    """,
                    (
                        now
                        + self._lease_seconds(
                            config if config is not None else self._engine._config
                        ),
                        conversation_id,
                        session_id,
                        self._owner_id,
                        int(claim_token),
                        now,
                    ),
                )
                if renewed.rowcount != 1:
                    conn.rollback()
                    return False

                if node.source_type == "messages":
                    if expected_source_identity_hashes is None or len(
                        expected_source_identity_hashes
                    ) != len(source_ids):
                        conn.rollback()
                        return False
                    source_rows = conn.execute(
                        f"""
                        SELECT * FROM messages
                        WHERE store_id IN ({placeholders})
                        ORDER BY store_id
                        """,
                        source_ids,
                    ).fetchall()
                    if [int(row["store_id"]) for row in source_rows] != source_ids:
                        conn.rollback()
                        return False
                    if any(
                        str(row["session_id"] or "") != session_id
                        or (
                            str(row["conversation_id"] or "")
                            and str(row["conversation_id"] or "") != conversation_id
                        )
                        for row in source_rows
                    ):
                        conn.rollback()
                        return False
                    if [
                        self._source_identity_hash(row) for row in source_rows
                    ] != [str(value) for value in expected_source_identity_hashes]:
                        conn.rollback()
                        return False

                overlap = conn.execute(
                    f"""
                    SELECT 1 FROM summary_nodes AS node, json_each(node.source_ids) AS source
                    WHERE node.session_id = ? AND node.source_type = ?
                      AND CAST(source.value AS INTEGER) IN ({placeholders})
                    LIMIT 1
                    """,
                    [session_id, node.source_type, *source_ids],
                ).fetchone()
                if overlap:
                    conn.rollback()
                    return False

                cur = conn.execute(
                    """
                    INSERT INTO summary_nodes(
                        session_id, depth, summary, token_count, source_token_count,
                        source_ids, source_type, created_at, earliest_at, latest_at,
                        expand_hint
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        node.depth,
                        node.summary,
                        node.token_count,
                        node.source_token_count,
                        json.dumps(source_ids),
                        node.source_type,
                        node.created_at or now,
                        node.earliest_at,
                        node.latest_at,
                        node.expand_hint,
                    ),
                )
                if node.source_type == "messages":
                    conn.execute(
                        """
                        UPDATE lcm_lifecycle_state
                        SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                            updated_at = ?
                        WHERE conversation_id = ? AND current_session_id = ?
                        """,
                        (int(frontier_store_id), now, conversation_id, session_id),
                    )
                    # Legacy/manual ContextEngine callers may have persisted
                    # messages without binding lifecycle state. Preserve the
                    # historical best-effort frontier behavior while keeping
                    # the canonical node insert in this same transaction.
                conn.commit()
                node.node_id = int(cur.lastrowid)
                return True
            except Exception:
                conn.rollback()
                raise

    def capture_snapshot(
        self,
        trigger: _BackgroundTrigger | None = None,
        *,
        _operation: _ManagerOperation | None = None,
    ) -> _BackgroundSnapshot | None:
        operation = _operation
        owns_operation = operation is None
        if operation is None:
            operation = self._acquire_operation()
            if operation is None:
                return None
        try:
            return self._capture_snapshot(trigger)
        finally:
            if owns_operation:
                operation.release()

    def _capture_snapshot(
        self,
        trigger: _BackgroundTrigger | None = None,
    ) -> _BackgroundSnapshot | None:
        """Capture small immutable runtime state and a durable message upper bound."""
        state_lock = getattr(self._engine, "_async_state_lock", None)
        lock = state_lock if state_lock is not None else threading.Lock()
        with lock:
            if self._closed or self._conn is None:
                return None
            config = copy.deepcopy(self._engine._config)
            session_id = str(getattr(self._engine, "_session_id", "") or "")
            conversation_id = str(getattr(self._engine, "_conversation_id", "") or "")
            if trigger is not None and (
                session_id != trigger.session_id
                or conversation_id != trigger.conversation_id
            ):
                return None
            model = str(getattr(self._engine, "model", "") or "")
            provider = str(getattr(self._engine, "provider", "") or "")
            base_url = str(getattr(self._engine, "base_url", "") or "")
            api_mode = str(getattr(self._engine, "api_mode", "") or "")
            hermes_home = str(getattr(self._engine, "_hermes_home", "") or "")
            session_ignored = bool(getattr(self._engine, "_session_ignored", False))
            session_stateless = bool(getattr(self._engine, "_session_stateless", False))
            threshold_tokens = int(getattr(self._engine, "threshold_tokens", 0) or 0)
            raw_context_length = int(getattr(self._engine, "raw_context_length", 0) or 0)
            binding_generation = int(getattr(self._engine, "_binding_generation", 0) or 0)
            store = getattr(self._engine, "_store", None)
            dag = getattr(self._engine, "_dag", None)
            lifecycle = getattr(self._engine, "_lifecycle", None)
            with self._lock:
                conn = self._conn
                assert conn is not None
                upper_row = conn.execute(
                    """
                    SELECT COALESCE(MAX(store_id), 0)
                    FROM messages
                    WHERE session_id = ? AND conversation_id = ?
                    """,
                    (session_id, conversation_id),
                ).fetchone()
            upper_store_id = int(upper_row[0] or 0) if upper_row else 0
        return _BackgroundSnapshot(
            messages=[],
            session_id=session_id,
            conversation_id=conversation_id,
            upper_store_id=upper_store_id,
            config=config,
            model=model,
            provider=provider,
            base_url=base_url,
            api_mode=api_mode,
            hermes_home=hermes_home,
            session_ignored=session_ignored,
            session_stateless=session_stateless,
            threshold_tokens=threshold_tokens,
            raw_context_length=raw_context_length,
            binding_generation=binding_generation,
            manager=self,
            store=store,
            dag=dag,
            lifecycle=lifecycle,
        )

    @classmethod
    def _source_identity_hash(cls, row: sqlite3.Row | Dict[str, Any]) -> str:
        def value(name: str, default: Any = "") -> Any:
            try:
                return row[name]
            except (KeyError, IndexError):
                return default

        content = value("content")
        tool_calls = value("tool_calls")
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        identity = [
            int(value("store_id", 0) or 0),
            str(value("session_id") or ""),
            str(value("conversation_id") or ""),
            str(value("role") or ""),
            hashlib.sha256(str(content or "").encode("utf-8")).hexdigest(),
            str(value("tool_call_id") or ""),
            cls._hash_json(tool_calls),
            str(value("tool_name") or ""),
            float(value("timestamp", 0.0) or 0.0),
        ]
        return cls._hash_json(identity)

    @classmethod
    def _source_coverage_hash(cls, source_ids: List[int], identity_hashes: List[str]) -> str:
        return cls._hash_json({"source_ids": source_ids, "identity_hashes": identity_hashes})

    def _row_to_batch(self, row: sqlite3.Row) -> CompactionBatch:
        def load_list(name: str) -> list[Any]:
            try:
                value = json.loads(row[name] or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                return []
            return value if isinstance(value, list) else []

        return CompactionBatch(
            batch_id=str(row["batch_id"]),
            conversation_id=str(row["conversation_id"]),
            session_id=str(row["session_id"]),
            state=str(row["state"]),
            frontier_start_store_id=int(row["frontier_start_store_id"] or 0),
            frontier_end_store_id=int(row["frontier_end_store_id"] or 0),
            fresh_tail_count=int(row["fresh_tail_count"] or 0),
            leaf_chunk_tokens=int(row["leaf_chunk_tokens"] or 0),
            policy_fingerprint=str(row["policy_fingerprint"] or ""),
            summary_route_fingerprint=str(row["summary_route_fingerprint"] or ""),
            source_coverage_hash=str(row["source_coverage_hash"] or ""),
            expected_leaf_count=int(row["expected_leaf_count"] or 0),
            prepared_leaf_count=int(row["prepared_leaf_count"] or 0),
            source_ids=[int(value) for value in load_list("source_ids")],
            source_identity_hashes=[str(value) for value in load_list("source_identity_hashes")],
            failure_count=int(row["failure_count"] or 0),
            next_retry_at=row["next_retry_at"],
            last_error=str(row["last_error"] or ""),
            rejected_reason=str(row["rejected_reason"] or ""),
            lease_owner=str(row["lease_owner"] or ""),
            lease_expires_at=row["lease_expires_at"],
        )

    def _get_batch(self, batch_id: str) -> CompactionBatch | None:
        conn = self._conn
        assert conn is not None
        row = conn.execute(
            "SELECT * FROM compaction_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        return self._row_to_batch(row) if row else None

    def get_batch(self, batch_id: str) -> CompactionBatch | None:
        with self._lock:
            return self._get_batch(batch_id)

    def _current_lifecycle_frontier(self, conversation_id: str, session_id: str) -> int | None:
        conn = self._conn
        assert conn is not None
        row = conn.execute(
            """
            SELECT current_frontier_store_id
            FROM lcm_lifecycle_state
            WHERE conversation_id = ? AND current_session_id = ?
            """,
            (conversation_id, session_id),
        ).fetchone()
        return int(row[0] or 0) if row else None

    def _candidate_rows(
        self,
        snapshot: _BackgroundSnapshot,
        *,
        frontier: int,
    ) -> list[sqlite3.Row]:
        conn = self._conn
        assert conn is not None
        rows = conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ? AND conversation_id = ? AND store_id <= ?
            ORDER BY store_id
            """,
            (snapshot.session_id, snapshot.conversation_id, snapshot.upper_store_id),
        ).fetchall()
        if not rows:
            return []
        boundary = resolve_fresh_tail_boundary(
            [self._message_from_row(row) for row in rows],
            fresh_tail_count=int(getattr(snapshot.config, "fresh_tail_count", 0) or 0),
            fresh_tail_max_tokens=int(getattr(snapshot.config, "fresh_tail_max_tokens", 0) or 0),
        )
        leading_anchor_count = int(
            bool(rows and str(rows[0]["role"] or "") == "system")
        )
        candidates = rows[leading_anchor_count:boundary.start]
        return [row for row in candidates if int(row["store_id"] or 0) > frontier]

    def _filter_candidate_rows(
        self,
        rows: list[sqlite3.Row],
        snapshot: _BackgroundSnapshot,
    ) -> list[sqlite3.Row]:
        compiled_patterns = compile_message_patterns(
            getattr(snapshot.config, "ignore_message_patterns", []) or []
        )
        if not compiled_patterns:
            return rows
        filtered: list[sqlite3.Row] = []
        for row in rows:
            message = dict(row)
            text = stored_text_content_for_pattern_matching(message.get("content")) or ""
            if not matches_message_pattern(text, compiled_patterns):
                filtered.append(row)
        return filtered

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        message = dict(row)
        if message.get("tool_calls") is None:
            message.pop("tool_calls", None)
        elif isinstance(message.get("tool_calls"), str):
            try:
                message["tool_calls"] = json.loads(message["tool_calls"])
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return message

    def _chunk_rows(
        self,
        rows: list[sqlite3.Row],
        snapshot: _BackgroundSnapshot,
    ) -> list[list[sqlite3.Row]]:
        config = snapshot.config
        leaf_tokens = max(1, int(getattr(config, "leaf_chunk_tokens", 1) or 1))
        chunks: list[list[sqlite3.Row]] = []
        remaining = list(rows)
        while remaining:
            working_limit = leaf_tokens
            if bool(getattr(config, "dynamic_leaf_chunk_enabled", False)):
                ceiling = max(leaf_tokens, int(getattr(config, "dynamic_leaf_chunk_max", leaf_tokens) or leaf_tokens))
                working_limit = leaf_tokens
                raw_tokens = sum(int(row["token_estimate"] or 0) for row in remaining)
                while working_limit < ceiling and raw_tokens > working_limit * 2:
                    working_limit = min(ceiling, working_limit * 2)
            selected: list[sqlite3.Row] = []
            used = 0
            for row in remaining:
                item = dict(row)
                item_tokens = count_message_tokens(item)
                if selected and used + item_tokens > working_limit:
                    break
                selected.append(row)
                used += item_tokens
            if not selected:
                break
            chunks.append(selected)
            remaining = remaining[len(selected):]
        return chunks

    def _insert_batch(
        self,
        *,
        snapshot: _BackgroundSnapshot,
        frontier: int,
        rows: list[sqlite3.Row],
        chunks: list[list[sqlite3.Row]],
    ) -> CompactionBatch | None:
        conn = self._conn
        assert conn is not None
        now = time.time()
        session_id = snapshot.session_id
        conversation_id = snapshot.conversation_id
        source_ids = [int(row["store_id"]) for row in rows]
        identity_hashes = [self._source_identity_hash(row) for row in rows]
        policy_fingerprint = self._policy_fingerprint(snapshot)
        route_fingerprint = self._summary_route_fingerprint(snapshot)
        active_sql = ",".join("?" for _ in _ACTIVE_BATCH_STATES)
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = conn.execute(
                f"""
                SELECT * FROM compaction_batches
                WHERE conversation_id = ? AND session_id = ?
                  AND state IN ({active_sql})
                ORDER BY created_at LIMIT 1
                """,
                [conversation_id, session_id, *_ACTIVE_BATCH_STATES],
            ).fetchone()
            if existing:
                existing_batch = self._row_to_batch(existing)
                if existing_batch.state == "ready":
                    conn.commit()
                    return existing_batch
                lease_owner = existing["lease_owner"]
                lease_expires_at = existing["lease_expires_at"]
                can_claim = (
                    existing_batch.state == "pending"
                    or not lease_owner
                    or lease_expires_at is None
                    or float(lease_expires_at) <= now
                    or str(lease_owner) == self._owner_id
                )
                if can_claim:
                    conn.execute(
                        f"""
                        UPDATE compaction_batches
                        SET state = 'preparing', lease_owner = ?,
                            lease_expires_at = ?, updated_at = ?
                        WHERE batch_id = ? AND state IN ({active_sql})
                          AND (
                              lease_owner IS NULL
                              OR lease_expires_at IS NULL
                              OR lease_expires_at <= ?
                              OR lease_owner = ?
                          )
                        """,
                        (
                            self._owner_id,
                            now + self._lease_seconds(snapshot.config),
                            now,
                            existing_batch.batch_id,
                            *_ACTIVE_BATCH_STATES,
                            now,
                            self._owner_id,
                        ),
                    )
                    existing = conn.execute(
                        "SELECT * FROM compaction_batches WHERE batch_id = ?",
                        (existing_batch.batch_id,),
                    ).fetchone()
                conn.commit()
                return self._row_to_batch(existing) if existing else existing_batch
            failed = conn.execute(
                """
                SELECT * FROM compaction_batches
                WHERE conversation_id = ? AND session_id = ?
                  AND state = 'failed'
                  AND source_coverage_hash = ?
                  AND policy_fingerprint = ?
                  AND summary_route_fingerprint = ?
                  AND next_retry_at > ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (
                    conversation_id,
                    session_id,
                    self._source_coverage_hash(source_ids, identity_hashes),
                    policy_fingerprint,
                    route_fingerprint,
                    now,
                ),
            ).fetchone()
            if failed:
                conn.commit()
                return self._row_to_batch(failed)
            max_batches = max(
                1,
                int(getattr(snapshot.config, "async_background_compaction_max_batches", 2) or 2),
            )
            active_count = conn.execute(
                f"""
                SELECT COUNT(*) FROM compaction_batches
                WHERE conversation_id = ? AND state IN ({active_sql})
                """,
                [conversation_id, *_ACTIVE_BATCH_STATES],
            ).fetchone()[0]
            if int(active_count or 0) >= max_batches:
                conn.commit()
                return None
            batch_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO compaction_batches(
                    batch_id, conversation_id, session_id, state,
                    frontier_start_store_id, frontier_end_store_id,
                    fresh_tail_count, leaf_chunk_tokens,
                    policy_fingerprint, summary_route_fingerprint,
                    source_coverage_hash, source_ids, source_identity_hashes,
                    expected_leaf_count, prepared_leaf_count,
                    created_at, updated_at, lease_owner, lease_expires_at
                ) VALUES (?, ?, ?, 'preparing', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    conversation_id,
                    session_id,
                    frontier,
                    max(source_ids),
                    int(getattr(snapshot.config, "fresh_tail_count", 0) or 0),
                    int(getattr(snapshot.config, "leaf_chunk_tokens", 0) or 0),
                    policy_fingerprint,
                    route_fingerprint,
                    self._source_coverage_hash(source_ids, identity_hashes),
                    json.dumps(source_ids),
                    json.dumps(identity_hashes),
                    len(chunks),
                    now,
                    now,
                    self._owner_id,
                    now + self._lease_seconds(snapshot.config),
                ),
            )
            conn.commit()
            return self._get_batch(batch_id)
        except Exception:
            conn.rollback()
            raise

    def _mark_failed(
        self,
        batch_id: str,
        exc: Exception,
        *,
        owner_id: str | None = None,
        config: Any | None = None,
    ) -> CompactionBatch:
        conn = self._conn
        assert conn is not None
        row = self._get_batch(batch_id)
        failure_count = (row.failure_count if row else 0) + 1
        backoff = max(
            0.0,
            float(
                getattr(
                    config if config is not None else self._engine._config,
                    "async_background_compaction_retry_backoff_seconds",
                    300.0,
                )
                or 0.0
            ),
        )
        message = f"{type(exc).__name__}: {exc}"[:500]
        conn.execute("BEGIN IMMEDIATE")
        try:
            current = conn.execute(
                "SELECT * FROM compaction_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if current is None:
                conn.commit()
                raise LookupError(f"async compaction batch disappeared: {batch_id}")
            if owner_id and str(current["lease_owner"] or "") != owner_id:
                # A lease may have been reclaimed while this worker was still
                # finishing its provider call. Never remove rows belonging to
                # the replacement owner; return its current batch unchanged.
                conn.commit()
                return self._row_to_batch(current)
            failure_count = int(current["failure_count"] or 0) + 1
            conn.execute(
                "DELETE FROM pending_summary_nodes WHERE batch_id = ?",
                (batch_id,),
            )
            owner_clause = ""
            owner_args: list[Any] = []
            if owner_id:
                owner_clause = " AND lease_owner = ?"
                owner_args.append(owner_id)
            conn.execute(
                f"""
                UPDATE compaction_batches
                SET state = 'failed', failure_count = ?, next_retry_at = ?,
                    last_error = ?, updated_at = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE batch_id = ?{owner_clause}
                """,
                [failure_count, time.time() + backoff, message, time.time(), batch_id, *owner_args],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        result = self._get_batch(batch_id)
        assert result is not None
        return result

    def prepare(
        self,
        messages: List[Dict[str, Any]],
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
        leave_state: str | None = None,
        snapshot: _BackgroundSnapshot | None = None,
    ) -> CompactionBatch | None:
        if self._closed:
            return None
        snapshot = snapshot or self.capture_snapshot()
        if not bool(getattr(snapshot.config, "async_background_compaction_enabled", False)):
            return None
        session_id = str(session_id or snapshot.session_id or "")
        conversation_id = str(conversation_id or snapshot.conversation_id or "")
        if not session_id or not conversation_id:
            return None
        if not messages and not int(snapshot.upper_store_id or 0):
            return None
        if (
            session_id != snapshot.session_id
            or conversation_id != snapshot.conversation_id
            or snapshot.session_ignored
            or snapshot.session_stateless
        ):
            return None
        if snapshot.session_id != session_id or snapshot.conversation_id != conversation_id:
            return None
        with self._lock:
            frontier = self._current_lifecycle_frontier(conversation_id, session_id)
            if frontier is None:
                return None
            rows = self._filter_candidate_rows(
                self._candidate_rows(
                    snapshot,
                    frontier=frontier,
                ),
                snapshot,
            )
            if not rows:
                return None
            total_tokens = count_messages_tokens([dict(row) for row in rows])
            leaf_limit = max(1, int(getattr(snapshot.config, "leaf_chunk_tokens", 1) or 1))
            if total_tokens < leaf_limit:
                return None
            chunks = self._chunk_rows(rows, snapshot)
            if not chunks:
                return None
            batch = self._insert_batch(
                snapshot=snapshot,
                frontier=frontier,
                rows=rows,
                chunks=chunks,
            )
            if batch is None:
                return batch
            if (
                batch is None
                or batch.state != "preparing"
                or self._lease_owner(batch.batch_id) != self._owner_id
            ):
                return batch
            if leave_state == "preparing":
                return batch

            prepared_rows: list[dict[str, Any]] = []
            summary_circuit_breaker = SummaryCircuitBreaker(
                failure_threshold=int(
                    self._setting(
                        snapshot.config, "summary_circuit_breaker_failure_threshold", 2
                    )
                ),
                cooldown_seconds=int(
                    self._setting(
                        snapshot.config, "summary_circuit_breaker_cooldown_seconds", 300
                    )
                ),
            )
            summary_spend_guard = SummarySpendGuard(
                max_calls=int(
                    self._setting(snapshot.config, "summary_spend_max_calls", 24)
                ),
                window_seconds=float(
                    self._setting(snapshot.config, "summary_spend_window_seconds", 600.0)
                ),
                backoff_seconds=float(
                    self._setting(snapshot.config, "summary_spend_backoff_seconds", 1800.0)
                ),
            )
            try:
                for chunk in chunks:
                    self._renew_lease(batch.batch_id, snapshot.config)
                    chunk_messages = [self._message_from_row(row) for row in chunk]
                    (
                        summarized_chunk,
                        source_tokens,
                        summary_text,
                        _level,
                        _attempts,
                    ) = self._engine._summarize_leaf_chunk_with_rescue(
                        chunk_messages,
                        _config_override=snapshot.config,
                        _session_id_override=session_id,
                        _hermes_home_override=snapshot.hermes_home,
                        _circuit_breaker_override=summary_circuit_breaker,
                        _spend_guard_override=summary_spend_guard,
                    )
                    if self._closed:
                        return None
                    if not summary_text or not summarized_chunk:
                        raise RuntimeError("background summary returned no content")
                    chunk_ids = [int(row["store_id"]) for row in chunk]
                    source_ids = [int(message["store_id"]) for message in summarized_chunk]
                    if source_ids != chunk_ids:
                        raise RuntimeError(
                            "background summary rescue changed batch coverage"
                        )
                    source_rows = list(chunk)
                    source_hashes = [self._source_identity_hash(row) for row in source_rows]
                    earliest = min(float(row["timestamp"]) for row in source_rows)
                    latest = max(float(row["timestamp"]) for row in source_rows)
                    pending_id = uuid.uuid4().hex
                    prepared_rows.append(
                        {
                            "pending_id": pending_id,
                            "batch_id": batch.batch_id,
                            "conversation_id": conversation_id,
                            "session_id": session_id,
                            "summary": str(summary_text),
                            "token_count": count_tokens(str(summary_text)),
                            "source_token_count": int(source_tokens),
                            "source_ids": json.dumps(source_ids),
                            "source_identity_hashes": json.dumps(source_hashes),
                            "source_range_start_store_id": min(source_ids),
                            "source_range_end_store_id": max(source_ids),
                            "created_at": time.time(),
                            "earliest_at": earliest,
                            "latest_at": latest,
                            "expand_hint": self._engine._extract_expand_hint(str(summary_text)),
                        }
                    )
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    lease_row = self._conn.execute(
                        """
                        SELECT state, lease_owner, lease_expires_at
                        FROM compaction_batches WHERE batch_id = ?
                        """,
                        (batch.batch_id,),
                    ).fetchone()
                    if (
                        lease_row is None
                        or str(lease_row["state"]) != "preparing"
                        or str(lease_row["lease_owner"] or "") != self._owner_id
                        or (
                            lease_row["lease_expires_at"] is not None
                            and float(lease_row["lease_expires_at"]) <= time.time()
                        )
                    ):
                        self._conn.rollback()
                        return self._get_batch(batch.batch_id)
                    for pending in prepared_rows:
                        self._conn.execute(
                            """
                            INSERT INTO pending_summary_nodes(
                                pending_id, batch_id, conversation_id, session_id,
                                depth, summary, token_count, source_token_count,
                                source_ids, source_identity_hashes,
                                source_range_start_store_id, source_range_end_store_id,
                                created_at, earliest_at, latest_at, expand_hint
                            ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            tuple(pending[key] for key in (
                                "pending_id", "batch_id", "conversation_id", "session_id",
                                "summary", "token_count", "source_token_count", "source_ids",
                                "source_identity_hashes", "source_range_start_store_id",
                                "source_range_end_store_id", "created_at", "earliest_at",
                                "latest_at", "expand_hint",
                            )),
                        )
                    self._conn.execute(
                        """
                        UPDATE compaction_batches
                        SET state = 'ready', prepared_leaf_count = ?, updated_at = ?,
                            lease_owner = NULL, lease_expires_at = NULL
                        WHERE batch_id = ? AND state = 'preparing' AND lease_owner = ?
                        """,
                        (len(prepared_rows), time.time(), batch.batch_id, self._owner_id),
                    )
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise
                return self._get_batch(batch.batch_id)
            except Exception as exc:
                logger.warning("LCM async compaction preparation failed: %s", exc)
                return self._mark_failed(
                    batch.batch_id,
                    exc,
                    owner_id=self._owner_id,
                    config=snapshot.config,
                )

    def _lease_owner(self, batch_id: str) -> str:
        conn = self._conn
        assert conn is not None
        row = conn.execute(
            "SELECT lease_owner FROM compaction_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        return str(row[0] or "") if row else ""

    def _renew_lease(self, batch_id: str, config: Any) -> bool:
        conn = self._conn
        assert conn is not None
        now = time.time()
        cur = conn.execute(
            """
            UPDATE compaction_batches
            SET lease_expires_at = ?, updated_at = ?
            WHERE batch_id = ? AND state = 'preparing' AND lease_owner = ?
            """,
            (now + self._lease_seconds(config), now, batch_id, self._owner_id),
        )
        conn.commit()
        return cur.rowcount == 1

    def enqueue_trigger(self, session_id: str, conversation_id: str) -> bool:
        if self._closed or self._worker is None or not session_id or not conversation_id:
            return False
        trigger = _BackgroundTrigger(session_id, conversation_id)
        accepted = self._worker.enqueue(trigger)
        if accepted:
            self._enqueued_jobs += 1
        else:
            self._dropped_jobs += 1
        return accepted

    def enqueue(self, messages: List[Dict[str, Any]] | None = None) -> bool:
        """Compatibility wrapper that enqueues only the current identity."""
        return self.enqueue_trigger(
            str(getattr(self._engine, "_session_id", "") or ""),
            str(getattr(self._engine, "_conversation_id", "") or ""),
        )

    def _finalize_worker_exit(self) -> None:
        """Close deferred resources once this manager's worker has stopped."""
        with self._lock:
            self._finish_close_locked(completed_worker=True)

    def _run_trigger(self, trigger: _BackgroundTrigger) -> None:
        operation = self._acquire_operation()
        if operation is None:
            return
        try:
            snapshot = self.capture_snapshot(trigger, _operation=operation)
            if snapshot is None:
                return
            self._run_snapshot(snapshot)
        finally:
            operation.release(completed_worker=True)

    def _finish_close_locked(self, *, completed_worker: bool = False) -> None:
        """Close only after workers and foreground operations release the manager."""
        if (
            not self._close_requested
            or self._operation_inflight
        ):
            return
        worker = self._worker
        if not completed_worker and worker is not None and (
            worker.active or worker.queue_depth
        ):
            return
        conn = self._conn
        if conn is None:
            return
        try:
            conn.execute(
                "DELETE FROM foreground_compaction_claims WHERE owner_id = ?",
                (self._owner_id,),
            )
            if self._background_tables_enabled:
                owned_rows = conn.execute(
                    """
                    SELECT batch_id FROM compaction_batches
                    WHERE state IN ('preparing', 'promoting') AND lease_owner = ?
                    """,
                    (self._owner_id,),
                ).fetchall()
                owned_ids = [str(row[0]) for row in owned_rows]
                conn.execute(
                    """
                    UPDATE compaction_batches
                    SET state = 'pending', prepared_leaf_count = 0,
                        lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = ?
                    WHERE state IN ('preparing', 'promoting') AND lease_owner = ?
                    """,
                    (time.time(), self._owner_id),
                )
                if owned_ids:
                    placeholders = ",".join("?" for _ in owned_ids)
                    conn.execute(
                        f"DELETE FROM pending_summary_nodes WHERE batch_id IN ({placeholders})",
                        owned_ids,
                    )
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
        try:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.Error:
            pass
        conn.close()
        self._conn = None
        self._close_requested = False

    def _run_snapshot(self, snapshot: _BackgroundSnapshot) -> None:
        try:
            self.prepare(
                snapshot.messages,
                snapshot=snapshot,
            )
        except Exception:
            logger.warning("LCM async compaction snapshot failed", exc_info=True)

    def _reject_in_transaction(
        self,
        batch: CompactionBatch,
        reason: str,
    ) -> PromotionResult:
        conn = self._conn
        assert conn is not None
        try:
            conn.execute(
                """
                UPDATE compaction_batches
                SET state = 'rejected', rejected_reason = ?, updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE batch_id = ? AND state = 'ready'
                """,
                (reason, time.time(), batch.batch_id),
            )
            conn.execute("DELETE FROM pending_summary_nodes WHERE batch_id = ?", (batch.batch_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return PromotionResult(False, reason, batch.batch_id)

    @staticmethod
    def _maintenance_node(node: SummaryNode) -> _PublishedNodeMaintenance:
        return _PublishedNodeMaintenance(
            node_id=int(node.node_id),
            session_id=str(node.session_id),
            depth=int(node.depth),
            summary=str(node.summary),
            token_count=int(node.token_count),
            source_token_count=int(node.source_token_count),
            source_ids=tuple(int(source_id) for source_id in node.source_ids),
            created_at=float(node.created_at),
            earliest_at=node.earliest_at,
            latest_at=node.latest_at,
            expand_hint=str(node.expand_hint),
        )

    def _run_post_publication_maintenance(
        self,
        payload: _PromotionMaintenancePayload,
    ) -> None:
        """Run detached maintenance without holding the manager or state lock."""
        snapshot = payload.snapshot
        maintenance_dag = payload.publication_dag
        maintenance_store = payload.publication_store
        owns_dag = False
        owns_store = False
        if bool(getattr(snapshot.config, "temporal_rollups_enabled", False)):
            if maintenance_dag is None:
                maintenance_dag = SummaryDAG(payload.db_path)
                owns_dag = True
        if bool(getattr(snapshot.config, "large_output_transcript_gc_enabled", False)):
            if maintenance_store is None:
                maintenance_store = MessageStore(
                    payload.db_path,
                    ingest_protection_config=snapshot.config,
                    hermes_home=snapshot.hermes_home,
                )
                owns_store = True

        nodes = tuple(
            SummaryNode(
                node_id=node.node_id,
                session_id=node.session_id,
                depth=node.depth,
                summary=node.summary,
                token_count=node.token_count,
                source_token_count=node.source_token_count,
                source_ids=list(node.source_ids),
                source_type="messages",
                created_at=node.created_at,
                earliest_at=node.earliest_at,
                latest_at=node.latest_at,
                expand_hint=node.expand_hint,
            )
            for node in payload.nodes
        )
        source_by_id = {
            source_id: dict(message)
            for source_id, message in payload.source_messages
        }
        try:
            for node in nodes:
                try:
                    self._engine._invalidate_rollups_for_published_node(
                        node,
                        config=snapshot.config,
                        dag=maintenance_dag,
                        session_id=snapshot.session_id,
                    )
                except Exception:
                    logger.warning(
                        "LCM async rollup invalidation failed after publication",
                        exc_info=True,
                    )
                try:
                    self._engine._maybe_gc_compacted_tool_results(
                        [
                            source_by_id[source_id]
                            for source_id in node.source_ids
                            if source_id in source_by_id
                        ],
                        list(node.source_ids),
                        config=snapshot.config,
                        session_id=snapshot.session_id,
                        hermes_home=snapshot.hermes_home,
                        store=maintenance_store,
                    )
                except Exception:
                    logger.warning(
                        "LCM async transcript maintenance failed after publication",
                        exc_info=True,
                    )
        finally:
            if owns_store and maintenance_store is not None:
                maintenance_store.close()
            if owns_dag and maintenance_dag is not None:
                maintenance_dag.close()

    def promote(
        self,
        batch_id: str,
        messages: List[Dict[str, Any]] | None = None,
    ) -> PromotionResult:
        operation = self._acquire_operation()
        if operation is None:
            return PromotionResult(False, "closed", batch_id)
        try:
            snapshot = self.capture_snapshot(_operation=operation)
            if snapshot is None:
                return PromotionResult(False, "closed", batch_id)
            return self._promote_with_snapshot(batch_id, messages, snapshot)
        finally:
            operation.release()

    def _promote_with_snapshot(
        self,
        batch_id: str,
        messages: List[Dict[str, Any]] | None = None,
        snapshot: _BackgroundSnapshot | None = None,
    ) -> PromotionResult:
        if snapshot is None:
            return PromotionResult(False, "closed", batch_id)
        if not bool(getattr(snapshot.config, "async_background_compaction_enabled", False)):
            return PromotionResult(False, "disabled", batch_id)
        maintenance_payload: _PromotionMaintenancePayload | None = None
        with self._publication_lock():
            conn = self._conn
            if conn is None or not self._snapshot_binding_is_current(snapshot):
                return PromotionResult(False, "stale_binding", batch_id)
            conn.execute("BEGIN IMMEDIATE")
            try:
                batch = self._get_batch(batch_id)
                if batch is None:
                    conn.rollback()
                    return PromotionResult(False, "unknown_batch", batch_id)
                if batch.state == "promoted":
                    conn.rollback()
                    return PromotionResult(
                        True,
                        "already_promoted",
                        batch.batch_id,
                        frontier_store_id=batch.frontier_end_store_id,
                    )
                if batch.state != "ready":
                    conn.rollback()
                    return PromotionResult(False, batch.rejected_reason or batch.state, batch.batch_id)
                session_id = snapshot.session_id
                conversation_id = snapshot.conversation_id
                if batch.session_id != session_id or batch.conversation_id != conversation_id:
                    return self._reject_in_transaction(batch, "session_identity_mismatch")
                if snapshot.session_ignored or snapshot.session_stateless:
                    return self._reject_in_transaction(batch, "session_policy_mismatch")
                if self._policy_fingerprint(snapshot) != batch.policy_fingerprint:
                    return self._reject_in_transaction(batch, "policy_fingerprint_mismatch")
                if self._summary_route_fingerprint(snapshot) != batch.summary_route_fingerprint:
                    return self._reject_in_transaction(batch, "summary_route_fingerprint_mismatch")
                current_frontier = self._current_lifecycle_frontier(conversation_id, session_id)
                if current_frontier is None or current_frontier != batch.frontier_start_store_id:
                    return self._reject_in_transaction(batch, "frontier_mismatch")

                source_ids = list(batch.source_ids)
                if not source_ids or source_ids != sorted(source_ids):
                    return self._reject_in_transaction(batch, "source_coverage_mismatch")
                foreground_claim = conn.execute(
                    """
                    SELECT 1
                    FROM foreground_compaction_claims
                    WHERE conversation_id = ? AND session_id = ?
                      AND owner_id != ? AND lease_expires_at > ?
                    LIMIT 1
                    """,
                    (conversation_id, session_id, self._owner_id, time.time()),
                ).fetchone()
                if foreground_claim:
                    return self._reject_in_transaction(
                        batch, "foreground_compaction_in_progress"
                    )
                placeholders = ",".join("?" for _ in source_ids)
                source_rows = conn.execute(
                    f"""
                    SELECT * FROM messages
                    WHERE store_id IN ({placeholders})
                    ORDER BY store_id
                    """,
                    source_ids,
                ).fetchall()
                if [int(row["store_id"]) for row in source_rows] != source_ids:
                    return self._reject_in_transaction(batch, "source_identity_mismatch")
                if any(
                    str(row["session_id"] or "") != session_id
                    or str(row["conversation_id"] or "") != conversation_id
                    for row in source_rows
                ):
                    return self._reject_in_transaction(batch, "source_identity_mismatch")
                if [self._source_identity_hash(row) for row in source_rows] != batch.source_identity_hashes:
                    return self._reject_in_transaction(batch, "source_identity_mismatch")
                if self._source_coverage_hash(source_ids, batch.source_identity_hashes) != batch.source_coverage_hash:
                    return self._reject_in_transaction(batch, "source_coverage_mismatch")

                full_rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ? AND conversation_id = ? AND store_id <= ?
                    ORDER BY store_id
                    """,
                    (session_id, conversation_id, snapshot.upper_store_id),
                ).fetchall()
                tail_boundary = resolve_fresh_tail_boundary(
                    [self._message_from_row(row) for row in full_rows],
                    fresh_tail_count=int(getattr(snapshot.config, "fresh_tail_count", 0) or 0),
                    fresh_tail_max_tokens=int(
                        getattr(snapshot.config, "fresh_tail_max_tokens", 0) or 0
                    ),
                )
                tail_ids = {
                    int(row["store_id"])
                    for row in full_rows[tail_boundary.start:]
                }
                if set(source_ids) & tail_ids:
                    return self._reject_in_transaction(batch, "fresh_tail_mismatch")

                overlap = conn.execute(
                    f"""
                    SELECT 1
                    FROM summary_nodes AS node, json_each(node.source_ids) AS source
                    WHERE node.session_id = ?
                      AND node.source_type = 'messages'
                      AND CAST(source.value AS INTEGER) IN ({placeholders})
                    LIMIT 1
                    """,
                    [session_id, *source_ids],
                ).fetchone()
                if overlap:
                    return self._reject_in_transaction(batch, "canonical_source_overlap")

                pending_rows = conn.execute(
                    """
                    SELECT * FROM pending_summary_nodes
                    WHERE batch_id = ? ORDER BY source_range_start_store_id
                    """,
                    (batch.batch_id,),
                ).fetchall()
                if len(pending_rows) != batch.expected_leaf_count or len(pending_rows) != batch.prepared_leaf_count:
                    return self._reject_in_transaction(batch, "pending_coverage_mismatch")
                pending_source_ids: list[int] = []
                for pending in pending_rows:
                    pending_ids = json.loads(pending["source_ids"] or "[]")
                    pending_hashes = json.loads(pending["source_identity_hashes"] or "[]")
                    if not isinstance(pending_ids, list) or not isinstance(pending_hashes, list):
                        return self._reject_in_transaction(batch, "pending_coverage_mismatch")
                    pending_source_ids.extend(int(value) for value in pending_ids)
                    expected_hashes = [
                        self._source_identity_hash(source_rows[source_ids.index(int(value))])
                        for value in pending_ids
                        if int(value) in source_ids
                    ]
                    if [str(value) for value in pending_hashes] != expected_hashes:
                        return self._reject_in_transaction(batch, "source_identity_mismatch")
                if pending_source_ids != source_ids:
                    return self._reject_in_transaction(batch, "pending_coverage_mismatch")

                node_ids: list[int] = []
                published_nodes: list[SummaryNode] = []
                for pending in pending_rows:
                    cur = conn.execute(
                        """
                        INSERT INTO summary_nodes(
                            session_id, depth, summary, token_count,
                            source_token_count, source_ids, source_type,
                            created_at, earliest_at, latest_at, expand_hint
                        ) VALUES (?, ?, ?, ?, ?, ?, 'messages', ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            int(pending["depth"]),
                            pending["summary"],
                            int(pending["token_count"]),
                            int(pending["source_token_count"]),
                            pending["source_ids"],
                            float(pending["created_at"]),
                            pending["earliest_at"],
                            pending["latest_at"],
                            pending["expand_hint"] or "",
                        ),
                    )
                    node_ids.append(int(cur.lastrowid))
                    published_nodes.append(
                        SummaryNode(
                            node_id=int(cur.lastrowid),
                            session_id=session_id,
                            depth=int(pending["depth"]),
                            summary=str(pending["summary"]),
                            token_count=int(pending["token_count"]),
                            source_token_count=int(pending["source_token_count"]),
                            source_ids=[int(value) for value in json.loads(pending["source_ids"] or "[]")],
                            source_type="messages",
                            created_at=float(pending["created_at"]),
                            earliest_at=pending["earliest_at"],
                            latest_at=pending["latest_at"],
                            expand_hint=pending["expand_hint"] or "",
                        )
                    )
                    if getattr(self._engine, "_async_compaction_publish_failure_hook", "") == "after_canonical_insert":
                        raise RuntimeError("injected async promotion failure")

                lifecycle_cursor = conn.execute(
                    """
                    UPDATE lcm_lifecycle_state
                    SET current_frontier_store_id = MAX(current_frontier_store_id, ?),
                        updated_at = ?
                    WHERE conversation_id = ? AND current_session_id = ?
                    """,
                    (batch.frontier_end_store_id, time.time(), conversation_id, session_id),
                )
                if lifecycle_cursor.rowcount != 1:
                    raise RuntimeError("async promotion lifecycle frontier update lost its session")
                conn.execute(
                    """
                    UPDATE compaction_batches
                    SET state = 'promoted', promoted_at = ?, updated_at = ?
                    WHERE batch_id = ? AND state = 'ready'
                    """,
                    (time.time(), time.time(), batch.batch_id),
                )
                superseded = conn.execute(
                    """
                    SELECT batch_id FROM compaction_batches
                    WHERE conversation_id = ? AND session_id = ?
                      AND batch_id != ?
                      AND created_at < (SELECT created_at FROM compaction_batches WHERE batch_id = ?)
                      AND state IN ('pending', 'preparing', 'ready')
                    """,
                    (conversation_id, session_id, batch.batch_id, batch.batch_id),
                ).fetchall()
                superseded_ids = [str(row[0]) for row in superseded]
                if superseded_ids:
                    superseded_placeholders = ",".join("?" for _ in superseded_ids)
                    conn.execute(
                        f"""
                        UPDATE compaction_batches
                        SET state = 'superseded', rejected_reason = 'older_batch_promoted', updated_at = ?
                        WHERE batch_id IN ({superseded_placeholders})
                        """,
                        [time.time(), *superseded_ids],
                    )
                    conn.execute(
                        f"DELETE FROM pending_summary_nodes WHERE batch_id IN ({superseded_placeholders})",
                        superseded_ids,
                    )
                conn.execute("DELETE FROM pending_summary_nodes WHERE batch_id = ?", (batch.batch_id,))
                conn.commit()
                maintenance_payload = _PromotionMaintenancePayload(
                    nodes=tuple(self._maintenance_node(node) for node in published_nodes),
                    source_messages=tuple(
                        (
                            int(row["store_id"]),
                            _freeze_maintenance_value(self._message_from_row(row)),
                        )
                        for row in source_rows
                    ),
                    snapshot=snapshot,
                    db_path=self._db_path,
                    publication_dag=(
                        self._engine._dag if self._db_path == ":memory:" else None
                    ),
                    publication_store=(
                        self._engine._store if self._db_path == ":memory:" else None
                    ),
                )
                promotion_result = PromotionResult(
                    True,
                    "promoted",
                    batch.batch_id,
                    tuple(node_ids),
                    batch.frontier_end_store_id,
                )
            except Exception:
                conn.rollback()
                raise

        # The durable publication transaction and immutable maintenance capture
        # are complete before the manager lock is released. Maintenance never
        # runs under that lock and therefore cannot invert the state -> manager
        # order used by snapshot capture and session rebind.
        if maintenance_payload is not None:
            self._run_post_publication_maintenance(maintenance_payload)

        state_lock = getattr(self._engine, "_async_state_lock", None)
        lock = state_lock if state_lock is not None else threading.Lock()
        with lock:
            if (
                int(getattr(self._engine, "_binding_generation", -1))
                == snapshot.binding_generation
                and getattr(self._engine, "_async_compaction", None) is self
                and str(getattr(self._engine, "_session_id", "") or "") == session_id
                and str(getattr(self._engine, "_conversation_id", "") or "") == conversation_id
            ):
                self._engine._last_compacted_store_id = max(
                    int(getattr(self._engine, "_last_compacted_store_id", 0) or 0),
                    batch.frontier_end_store_id,
                )
        return promotion_result

    def reject(self, batch_id: str, reason: str) -> PromotionResult:
        if self._closed:
            return PromotionResult(False, "closed", batch_id)
        with self._lock:
            conn = self._conn
            assert conn is not None
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._get_batch(batch_id)
                if row is None:
                    conn.rollback()
                    return PromotionResult(False, "unknown_batch", batch_id)
                if row.state == "ready":
                    conn.execute(
                        """
                        UPDATE compaction_batches
                        SET state='rejected', rejected_reason=?, updated_at=?,
                            lease_owner=NULL, lease_expires_at=NULL
                        WHERE batch_id=?
                        """,
                        (reason, time.time(), batch_id),
                    )
                    conn.execute("DELETE FROM pending_summary_nodes WHERE batch_id = ?", (batch_id,))
                conn.commit()
                return PromotionResult(False, reason, batch_id)
            except Exception:
                conn.rollback()
                raise

    def promote_next(self, messages: List[Dict[str, Any]]) -> PromotionResult | None:
        operation = self._acquire_operation()
        if operation is None:
            return PromotionResult(False, "closed", "")
        try:
            snapshot = self.capture_snapshot(_operation=operation)
            if snapshot is None:
                return PromotionResult(False, "closed", "")
            if not bool(getattr(snapshot.config, "async_background_compaction_enabled", False)):
                return None
            with self._lock:
                conn = self._conn
                if self._closed or conn is None:
                    return PromotionResult(False, "closed", "")
                session_id = snapshot.session_id
                conversation_id = snapshot.conversation_id
                row = conn.execute(
                    """
                    SELECT batch_id FROM compaction_batches
                    WHERE session_id = ? AND conversation_id = ? AND state = 'ready'
                    ORDER BY created_at LIMIT 1
                    """,
                    (session_id, conversation_id),
                ).fetchone()
            if row is None:
                return None
            return self._promote_with_snapshot(str(row[0]), messages, snapshot)
        finally:
            operation.release()

    def status(self, conversation_id: str | None = None) -> Dict[str, Any]:
        operation = self._acquire_operation()
        if operation is None:
            return self._closed_status()
        try:
            return self._status_with_operation(conversation_id)
        finally:
            operation.release()

    @staticmethod
    def _closed_status() -> Dict[str, Any]:
        return {
            "enabled": False,
            "pending_batches": 0,
            "prepared_batches": 0,
            "promoted_batches": 0,
            "rejected_batches": 0,
            "failed_batches": 0,
            "superseded_batches": 0,
            "preparing_batches": 0,
            "worker_enabled": False,
            "queue_depth": 0,
            "worker_active": False,
            "enqueued_jobs": 0,
            "dropped_jobs": 0,
            "pending_summaries": 0,
            "oldest_pending_age_seconds": None,
            "last_rejected_reason": None,
            "last_error": None,
        }

    def _status_with_operation(self, conversation_id: str | None = None) -> Dict[str, Any]:
        with self._lock:
            conn = self._conn
            if self._closed or conn is None or not self._background_tables_enabled:
                return self._closed_status()
            where = ""
            args: list[Any] = []
            if conversation_id:
                where = "WHERE conversation_id = ?"
                args.append(conversation_id)
            rows = conn.execute(
                f"SELECT state, COUNT(*) AS count FROM compaction_batches {where} GROUP BY state",
                args,
            ).fetchall()
            counts = {str(row[0]): int(row[1] or 0) for row in rows}
            pending_rows = conn.execute(
                f"""
                SELECT COUNT(*) AS count, MIN(p.created_at) AS oldest_created_at
                FROM pending_summary_nodes AS p
                JOIN compaction_batches AS b ON b.batch_id = p.batch_id
                WHERE b.state IN ('pending', 'preparing', 'ready')
                  {"AND p.conversation_id = ?" if conversation_id else ""}
                """,
                [conversation_id] if conversation_id else [],
            ).fetchone()
            latest_rejected = conn.execute(
                f"""
                SELECT rejected_reason
                FROM compaction_batches
                {where + (" AND" if where else "WHERE")} state = 'rejected'
                ORDER BY updated_at DESC LIMIT 1
                """,
                [*args],
            ).fetchone()
            latest_error = conn.execute(
                f"""
                SELECT last_error
                FROM compaction_batches
                {where + (" AND" if where else "WHERE")} last_error IS NOT NULL
                  AND last_error != ''
                ORDER BY updated_at DESC LIMIT 1
                """,
                [*args],
            ).fetchone()
            worker = self._worker
            oldest_created_at = pending_rows[1] if pending_rows else None
            return {
                "enabled": self._background_tables_enabled,
                "worker_enabled": worker is not None,
                "pending_batches": counts.get("pending", 0) + counts.get("preparing", 0),
                "preparing_batches": counts.get("preparing", 0),
                "prepared_batches": counts.get("ready", 0),
                "promoted_batches": counts.get("promoted", 0),
                "rejected_batches": counts.get("rejected", 0),
                "failed_batches": counts.get("failed", 0),
                "superseded_batches": counts.get("superseded", 0),
                "queue_depth": worker.queue_depth if worker else 0,
                "worker_active": worker.active if worker else False,
                "enqueued_jobs": self._enqueued_jobs,
                "dropped_jobs": self._dropped_jobs,
                "pending_summaries": int(pending_rows[0] or 0) if pending_rows else 0,
                "oldest_pending_age_seconds": (
                    max(0.0, time.time() - float(oldest_created_at))
                    if oldest_created_at is not None
                    else None
                ),
                "last_rejected_reason": str(latest_rejected[0] or "") if latest_rejected else None,
                "last_error": str(latest_error[0] or "") if latest_error else None,
            }

    def drain(self, timeout: float | None = None) -> bool:
        worker = self._worker
        return True if worker is None else worker.drain(timeout)

    def close(self) -> None:
        if self._closed and not self._close_requested:
            return
        # Do not take the manager lock here: preparation may hold it across a
        # provider call, and shutdown must remain bounded while that call is
        # still live. The worker/foreground guards below reconcile the close
        # request once the current operation releases the connection.
        self._closed = True
        self._close_requested = True
        worker = self._worker
        if worker is not None:
            # A live daemon callback still owns this manager connection. Wait
            # only for a bounded interval; the worker closes the connection from
            # its own completion path if it outlives this call.
            if not worker.close(timeout=_WORKER_CLOSE_TIMEOUT_SECONDS):
                logger.warning("LCM async compaction worker did not stop before shutdown")
                return
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._finish_close_locked()
        finally:
            self._lock.release()
