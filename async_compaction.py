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
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from .db_bootstrap import configure_connection, ensure_async_compaction_tables
from .tokens import count_message_tokens, count_messages_tokens, count_tokens

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "async_compaction_protocol_v1"
_ACTIVE_BATCH_STATES = ("pending", "preparing", "ready")


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


class _BoundedBackgroundWorker:
    """One daemon worker with non-blocking bounded enqueue and safe draining."""

    def __init__(self, callback: Callable[[_BackgroundSnapshot], None], max_items: int):
        self._callback = callback
        self._queue: queue.Queue[_BackgroundSnapshot] = queue.Queue(maxsize=max(1, max_items))
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

    def enqueue(self, snapshot: _BackgroundSnapshot) -> bool:
        with self._condition:
            if self._stopping:
                return False
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
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

    def _run(self) -> None:
        while True:
            with self._condition:
                if self._stopping and self._queue.empty():
                    return
            try:
                snapshot = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._condition:
                self._active = True
            try:
                self._callback(snapshot)
            except Exception:
                logger.warning("LCM async compaction worker job failed", exc_info=True)
            finally:
                self._queue.task_done()
                with self._condition:
                    self._active = False
                    self._condition.notify_all()

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
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            str(engine._store.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        configure_connection(self._conn)
        self._conn.row_factory = sqlite3.Row
        ensure_async_compaction_tables(self._conn)
        self._recover_incomplete_batches()
        self._worker: _BoundedBackgroundWorker | None = None
        self._enqueued_jobs = 0
        self._dropped_jobs = 0
        self._closed = False

    @property
    def connection(self) -> sqlite3.Connection | None:
        return self._conn

    def _recover_incomplete_batches(self) -> None:
        conn = self._conn
        assert conn is not None
        conn.execute(
            """
            UPDATE compaction_batches
            SET state = 'rejected',
                rejected_reason = 'restart_recovery',
                last_error = 'incomplete async batch recovered after restart',
                updated_at = ?
            WHERE state IN ('pending', 'preparing', 'promoting')
            """,
            (time.time(),),
        )
        conn.execute(
            """
            DELETE FROM pending_summary_nodes
            WHERE batch_id IN (
                SELECT batch_id FROM compaction_batches
                WHERE state = 'rejected' AND rejected_reason = 'restart_recovery'
            )
            """
        )

    @staticmethod
    def _hash_json(value: Any) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _policy_fingerprint(self) -> str:
        config = self._engine._config
        policy = {
            "protocol": _PROTOCOL_VERSION,
            "fresh_tail_count": int(getattr(config, "fresh_tail_count", 0) or 0),
            "fresh_tail_max_tokens": int(getattr(config, "fresh_tail_max_tokens", 0) or 0),
            "leaf_chunk_tokens": int(getattr(config, "leaf_chunk_tokens", 0) or 0),
            "context_threshold": float(getattr(config, "context_threshold", 0.0) or 0.0),
            "runtime_threshold_tokens": int(getattr(self._engine, "threshold_tokens", 0) or 0),
            "raw_context_length": int(getattr(self._engine, "raw_context_length", 0) or 0),
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
            "custom_instructions": str(getattr(config, "custom_instructions", "") or ""),
            "l2_budget_ratio": float(getattr(config, "l2_budget_ratio", 0.0) or 0.0),
            "l3_truncate_tokens": int(getattr(config, "l3_truncate_tokens", 0) or 0),
        }
        return self._hash_json(policy)

    def _summary_route_fingerprint(self) -> str:
        config = self._engine._config
        route = {
            "protocol": _PROTOCOL_VERSION,
            "summary_model": str(getattr(config, "summary_model", "") or ""),
            "summary_fallback_models": list(getattr(config, "summary_fallback_models", []) or []),
            "provider": str(getattr(self._engine, "provider", "") or ""),
            "model": str(getattr(self._engine, "model", "") or ""),
            "summary_timeout_ms": int(getattr(config, "summary_timeout_ms", 0) or 0),
            "plugin_version": "hermes-lcm",
        }
        return self._hash_json(route)

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
        messages: List[Dict[str, Any]],
        *,
        session_id: str,
        conversation_id: str,
        frontier: int,
    ) -> list[sqlite3.Row]:
        engine = self._engine
        if str(getattr(engine, "_session_id", "") or "") != session_id:
            return []
        if str(getattr(engine, "_conversation_id", "") or "") != conversation_id:
            return []
        raw_messages = engine._raw_backlog_messages(messages)
        if not raw_messages:
            return []
        previous_map = engine._current_compress_store_ids_by_message_id
        engine._current_compress_store_ids_by_message_id = engine._get_store_id_map_for_messages(raw_messages)
        try:
            source_ids = [
                int(store_id)
                for message in raw_messages
                for store_id in [engine._current_compress_store_ids_by_message_id.get(id(message))]
                if store_id is not None and int(store_id) > frontier
            ]
        finally:
            engine._current_compress_store_ids_by_message_id = previous_map
        source_ids = sorted(dict.fromkeys(source_ids))
        if not source_ids:
            return []
        conn = self._conn
        assert conn is not None
        placeholders = ",".join("?" for _ in source_ids)
        rows = conn.execute(
            f"""
            SELECT * FROM messages
            WHERE store_id IN ({placeholders})
              AND session_id = ? AND conversation_id = ?
            ORDER BY store_id
            """,
            [*source_ids, session_id, conversation_id],
        ).fetchall()
        if [int(row["store_id"]) for row in rows] != source_ids:
            return []
        return rows

    def _filter_candidate_rows(self, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        engine = self._engine
        filtered: list[sqlite3.Row] = []
        for row in rows:
            message = dict(row)
            try:
                ignored = bool(engine._matches_ignore_message_patterns(message))
            except (AttributeError, TypeError, ValueError):
                ignored = False
            if not ignored:
                filtered.append(row)
        return filtered

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        message = dict(row)
        if message.get("tool_calls") is None:
            message.pop("tool_calls", None)
        return message

    def _chunk_rows(self, rows: list[sqlite3.Row]) -> list[list[sqlite3.Row]]:
        engine = self._engine
        config = engine._config
        leaf_tokens = max(1, int(getattr(config, "leaf_chunk_tokens", 1) or 1))
        chunks: list[list[sqlite3.Row]] = []
        remaining = list(rows)
        while remaining:
            working_limit = leaf_tokens
            if bool(getattr(config, "dynamic_leaf_chunk_enabled", False)):
                working_limit = engine._working_leaf_chunk_tokens(
                    sum(int(row["token_estimate"] or 0) for row in remaining)
                )
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
        session_id: str,
        conversation_id: str,
        frontier: int,
        rows: list[sqlite3.Row],
        chunks: list[list[sqlite3.Row]],
    ) -> CompactionBatch | None:
        conn = self._conn
        assert conn is not None
        now = time.time()
        source_ids = [int(row["store_id"]) for row in rows]
        identity_hashes = [self._source_identity_hash(row) for row in rows]
        policy_fingerprint = self._policy_fingerprint()
        route_fingerprint = self._summary_route_fingerprint()
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
                conn.commit()
                return self._row_to_batch(existing)
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
                int(getattr(self._engine._config, "async_background_compaction_max_batches", 2) or 2),
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
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    batch_id,
                    conversation_id,
                    session_id,
                    frontier,
                    max(source_ids),
                    int(getattr(self._engine._config, "fresh_tail_count", 0) or 0),
                    int(getattr(self._engine._config, "leaf_chunk_tokens", 0) or 0),
                    policy_fingerprint,
                    route_fingerprint,
                    self._source_coverage_hash(source_ids, identity_hashes),
                    json.dumps(source_ids),
                    json.dumps(identity_hashes),
                    len(chunks),
                    now,
                    now,
                ),
            )
            conn.commit()
            return self._get_batch(batch_id)
        except Exception:
            conn.rollback()
            raise

    def _mark_failed(self, batch_id: str, exc: Exception) -> CompactionBatch:
        conn = self._conn
        assert conn is not None
        row = self._get_batch(batch_id)
        failure_count = (row.failure_count if row else 0) + 1
        backoff = max(
            0.0,
            float(
                getattr(
                    self._engine._config,
                    "async_background_compaction_retry_backoff_seconds",
                    300.0,
                )
                or 0.0
            ),
        )
        message = f"{type(exc).__name__}: {exc}"[:500]
        conn.execute(
            "DELETE FROM pending_summary_nodes WHERE batch_id = ?",
            (batch_id,),
        )
        conn.execute(
            """
            UPDATE compaction_batches
            SET state = 'failed', failure_count = ?, next_retry_at = ?,
                last_error = ?, updated_at = ?
            WHERE batch_id = ?
            """,
            (failure_count, time.time() + backoff, message, time.time(), batch_id),
        )
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
    ) -> CompactionBatch | None:
        if self._closed or not bool(getattr(self._engine._config, "async_background_compaction_enabled", False)):
            return None
        session_id = str(session_id or getattr(self._engine, "_session_id", "") or "")
        conversation_id = str(
            conversation_id or getattr(self._engine, "_conversation_id", "") or ""
        )
        if not session_id or not conversation_id or not messages:
            return None
        with self._lock:
            frontier = self._current_lifecycle_frontier(conversation_id, session_id)
            if frontier is None:
                return None
            rows = self._filter_candidate_rows(
                self._candidate_rows(
                    messages,
                    session_id=session_id,
                    conversation_id=conversation_id,
                    frontier=frontier,
                )
            )
            if not rows:
                return None
            total_tokens = count_messages_tokens([dict(row) for row in rows])
            leaf_limit = max(1, int(getattr(self._engine._config, "leaf_chunk_tokens", 1) or 1))
            if total_tokens < leaf_limit:
                return None
            chunks = self._chunk_rows(rows)
            if not chunks:
                return None
            batch = self._insert_batch(
                session_id=session_id,
                conversation_id=conversation_id,
                frontier=frontier,
                rows=rows,
                chunks=chunks,
            )
            if batch is None:
                return batch
            if batch.state == "pending":
                self._conn.execute(
                    "UPDATE compaction_batches SET state = 'preparing', updated_at = ? WHERE batch_id = ?",
                    (time.time(), batch.batch_id),
                )
                self._conn.commit()
                batch = self._get_batch(batch.batch_id)
            if batch is None or batch.state != "preparing":
                return batch
            if leave_state == "preparing":
                return batch

            pending_ids: list[str] = []
            try:
                for chunk in chunks:
                    chunk_messages = [self._message_from_row(row) for row in chunk]
                    (
                        summarized_chunk,
                        source_tokens,
                        summary_text,
                        _level,
                        _attempts,
                    ) = self._engine._summarize_leaf_chunk_with_rescue(chunk_messages)
                    if not summary_text or not summarized_chunk:
                        raise RuntimeError("background summary returned no content")
                    source_ids = [int(message["store_id"]) for message in summarized_chunk]
                    source_rows = [row for row in chunk if int(row["store_id"]) in source_ids]
                    source_hashes = [self._source_identity_hash(row) for row in source_rows]
                    earliest = min(float(row["timestamp"]) for row in source_rows)
                    latest = max(float(row["timestamp"]) for row in source_rows)
                    pending_id = uuid.uuid4().hex
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
                        (
                            pending_id,
                            batch.batch_id,
                            conversation_id,
                            session_id,
                            str(summary_text),
                            count_tokens(str(summary_text)),
                            int(source_tokens),
                            json.dumps(source_ids),
                            json.dumps(source_hashes),
                            min(source_ids),
                            max(source_ids),
                            time.time(),
                            earliest,
                            latest,
                            self._engine._extract_expand_hint(str(summary_text)),
                        ),
                    )
                    pending_ids.append(pending_id)
                self._conn.execute(
                    """
                    UPDATE compaction_batches
                    SET state = 'ready', prepared_leaf_count = ?, updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (len(pending_ids), time.time(), batch.batch_id),
                )
                return self._get_batch(batch.batch_id)
            except Exception as exc:
                logger.warning("LCM async compaction preparation failed: %s", exc)
                return self._mark_failed(batch.batch_id, exc)

    def enqueue(self, messages: List[Dict[str, Any]]) -> bool:
        if self._closed:
            return False
        config = self._engine._config
        if not (
            bool(getattr(config, "async_background_compaction_enabled", False))
            and bool(getattr(config, "async_background_compaction_worker_enabled", False))
        ):
            return False
        snapshot = _BackgroundSnapshot(
            messages=copy.deepcopy(messages),
            session_id=str(getattr(self._engine, "_session_id", "") or ""),
            conversation_id=str(getattr(self._engine, "_conversation_id", "") or ""),
        )
        with self._lock:
            if self._worker is None:
                self._worker = _BoundedBackgroundWorker(
                    self._run_snapshot,
                    max_items=max(
                        1,
                        int(getattr(config, "async_background_compaction_max_batches", 2) or 2),
                    ),
                )
            accepted = self._worker.enqueue(snapshot)
            if accepted:
                self._enqueued_jobs += 1
            else:
                self._dropped_jobs += 1
            return accepted

    def _run_snapshot(self, snapshot: _BackgroundSnapshot) -> None:
        try:
            self.prepare(
                snapshot.messages,
                session_id=snapshot.session_id,
                conversation_id=snapshot.conversation_id,
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
        conn.execute(
            """
            UPDATE compaction_batches
            SET state = 'rejected', rejected_reason = ?, updated_at = ?
            WHERE batch_id = ? AND state = 'ready'
            """,
            (reason, time.time(), batch.batch_id),
        )
        conn.execute("DELETE FROM pending_summary_nodes WHERE batch_id = ?", (batch.batch_id,))
        conn.commit()
        return PromotionResult(False, reason, batch.batch_id)

    def promote(
        self,
        batch_id: str,
        messages: List[Dict[str, Any]] | None = None,
    ) -> PromotionResult:
        if self._closed or not bool(getattr(self._engine._config, "async_background_compaction_enabled", False)):
            return PromotionResult(False, "disabled", batch_id)
        with self._lock:
            conn = self._conn
            assert conn is not None
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
                session_id = str(getattr(self._engine, "_session_id", "") or "")
                conversation_id = str(getattr(self._engine, "_conversation_id", "") or "")
                if batch.session_id != session_id or batch.conversation_id != conversation_id:
                    return self._reject_in_transaction(batch, "session_identity_mismatch")
                if self._policy_fingerprint() != batch.policy_fingerprint:
                    return self._reject_in_transaction(batch, "policy_fingerprint_mismatch")
                if self._summary_route_fingerprint() != batch.summary_route_fingerprint:
                    return self._reject_in_transaction(batch, "summary_route_fingerprint_mismatch")
                current_frontier = self._current_lifecycle_frontier(conversation_id, session_id)
                if current_frontier is None or current_frontier != batch.frontier_start_store_id:
                    return self._reject_in_transaction(batch, "frontier_mismatch")

                source_ids = list(batch.source_ids)
                if not source_ids or source_ids != sorted(source_ids):
                    return self._reject_in_transaction(batch, "source_coverage_mismatch")
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

                tail_count = max(0, int(getattr(self._engine._config, "fresh_tail_count", 0) or 0))
                tail_rows = conn.execute(
                    """
                    SELECT store_id FROM messages
                    WHERE session_id = ? AND conversation_id = ?
                    ORDER BY store_id DESC LIMIT ?
                    """,
                    (session_id, conversation_id, tail_count),
                ).fetchall()
                if set(source_ids) & {int(row[0]) for row in tail_rows}:
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
                self._engine._last_compacted_store_id = max(
                    int(getattr(self._engine, "_last_compacted_store_id", 0) or 0),
                    batch.frontier_end_store_id,
                )
                return PromotionResult(
                    True,
                    "promoted",
                    batch.batch_id,
                    tuple(node_ids),
                    batch.frontier_end_store_id,
                )
            except Exception:
                conn.rollback()
                raise

    def reject(self, batch_id: str, reason: str) -> PromotionResult:
        if self._closed:
            return PromotionResult(False, "closed", batch_id)
        with self._lock:
            conn = self._conn
            assert conn is not None
            conn.execute("BEGIN IMMEDIATE")
            row = self._get_batch(batch_id)
            if row is None:
                conn.rollback()
                return PromotionResult(False, "unknown_batch", batch_id)
            if row.state == "ready":
                conn.execute(
                    "UPDATE compaction_batches SET state='rejected', rejected_reason=?, updated_at=? WHERE batch_id=?",
                    (reason, time.time(), batch_id),
                )
                conn.execute("DELETE FROM pending_summary_nodes WHERE batch_id = ?", (batch_id,))
            conn.commit()
            return PromotionResult(False, reason, batch_id)

    def promote_next(self, messages: List[Dict[str, Any]]) -> PromotionResult | None:
        if not bool(getattr(self._engine._config, "async_background_compaction_enabled", False)):
            return None
        with self._lock:
            conn = self._conn
            assert conn is not None
            session_id = str(getattr(self._engine, "_session_id", "") or "")
            conversation_id = str(getattr(self._engine, "_conversation_id", "") or "")
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
        return self.promote(str(row[0]), messages)

    def status(self, conversation_id: str | None = None) -> Dict[str, Any]:
        if self._closed or self._conn is None:
            return {
                "enabled": bool(getattr(self._engine._config, "async_background_compaction_enabled", False)),
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
        where = ""
        args: list[Any] = []
        if conversation_id:
            where = "WHERE conversation_id = ?"
            args.append(conversation_id)
        rows = self._conn.execute(
            f"SELECT state, COUNT(*) AS count FROM compaction_batches {where} GROUP BY state",
            args,
        ).fetchall()
        counts = {str(row[0]): int(row[1] or 0) for row in rows}
        pending_rows = self._conn.execute(
            f"""
            SELECT COUNT(*) AS count, MIN(p.created_at) AS oldest_created_at
            FROM pending_summary_nodes AS p
            JOIN compaction_batches AS b ON b.batch_id = p.batch_id
            WHERE b.state IN ('pending', 'preparing', 'ready')
              {"AND p.conversation_id = ?" if conversation_id else ""}
            """,
            [conversation_id] if conversation_id else [],
        ).fetchone()
        latest_rejected = self._conn.execute(
            f"""
            SELECT rejected_reason
            FROM compaction_batches
            {where + (" AND" if where else "WHERE")} state = 'rejected'
            ORDER BY updated_at DESC LIMIT 1
            """,
            [*args],
        ).fetchone()
        latest_error = self._conn.execute(
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
            "enabled": bool(getattr(self._engine._config, "async_background_compaction_enabled", False)),
            "worker_enabled": bool(getattr(self._engine._config, "async_background_compaction_worker_enabled", False)),
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
        if self._closed:
            return
        self._closed = True
        worker = self._worker
        if worker is not None:
            timeout = max(
                5.0,
                float(getattr(self._engine._config, "summary_timeout_ms", 60000) or 60000) / 1000.0
                + 5.0,
            )
            if not worker.close(timeout):
                logger.warning("LCM async compaction worker did not stop before shutdown timeout")
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                try:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass
                conn.close()
