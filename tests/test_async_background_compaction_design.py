"""Contract tests for opt-in async/background compaction."""

from __future__ import annotations

import json
import sqlite3
import time
from queue import Queue
from threading import Event, Thread

import pytest

import hermes_lcm.async_compaction as async_compaction_module
from hermes_lcm.async_compaction import _BackgroundSnapshot, _BoundedBackgroundWorker
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _engine(tmp_path, *, session_id="async-session", conversation_id="async-conversation"):
    config = LCMConfig(
        database_path=str(tmp_path / f"{session_id}.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=False,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=conversation_id,
        platform="test",
        context_length=1_000,
    )
    return engine


def _messages(count=10, *, prefix="message"):
    messages = [{"role": "system", "content": "system prompt"}]
    for idx in range(count):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append(
            {
                "role": role,
                "content": f"{prefix} {idx} " + ("x " * 12),
            }
        )
    return messages


def test_default_disabled_async_compaction_is_inert(tmp_path):
    """Given default config, background prep is disabled and reports zero async debt."""
    config = LCMConfig(
        database_path=str(tmp_path / "disabled.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "disabled-session",
        conversation_id="disabled-conversation",
        platform="test",
        context_length=1_000,
    )
    try:
        messages = _messages()
        engine.ingest(messages)

        result = engine.prepare_background_compaction_once(messages)

        assert result is None or result.state == "disabled"
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert status["async_compaction"]["enabled"] is False
        assert status["async_compaction"]["pending_batches"] == 0
        assert status["async_compaction"]["prepared_batches"] == 0
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_pending_summaries_are_invisible_until_atomic_promotion(tmp_path):
    """Given prepared pending leaves, active context/readers ignore them until promotion."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)

        batch = engine.prepare_background_compaction_once(messages)

        assert batch.state == "ready"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        assert status["async_compaction"]["prepared_batches"] == 1
        assert status["dag"]["total_nodes"] == 0
        grep = json.loads(engine.handle_tool_call("lcm_grep", {"query": "message"}))
        assert all(result.get("kind") != "pending_summary" for result in grep.get("results", []))
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_stale_source_identity(tmp_path):
    """Given source rows changed after prep, promotion rejects without canonical mutation."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        first_source_id = batch.source_ids[0]
        engine._store._conn.execute(
            "UPDATE messages SET content = content || ' reconciled late' WHERE store_id = ?",
            (first_source_id,),
        )
        engine._store._conn.commit()

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "source_identity_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        assert engine.get_async_compaction_status()["rejected_batches"] == 1
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_live_config_change(tmp_path):
    """Given live config changes after prep, live policy wins over stale persisted metadata."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.fresh_tail_count = 6

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_summary_route_change(tmp_path):
    """Given summary model changes after prep, promotion rejects stale route output."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.summary_model = "different-summary-model"

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "summary_route_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_atomic_promotion_rejects_live_threshold_policy_change(tmp_path):
    """Given threshold changes after prep, live config beats persisted batch policy."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        engine._config.context_threshold = 0.75

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
    finally:
        engine.shutdown()


def test_foreground_compaction_race_supersedes_pending_batch(tmp_path, monkeypatch):
    """Given foreground compaction lands first, stale pending work is rejected/superseded."""
    engine = _engine(tmp_path)
    try:
        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: ("foreground summary", 0),
        )
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        compacted = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
        result = engine.promote_prepared_compaction(batch.batch_id, compacted)

        assert engine._dag.get_session_node_count(engine.current_session_id) >= 1
        assert result.promoted is False
        assert result.reason in {"frontier_mismatch", "canonical_source_overlap"}
        async_status = engine.get_async_compaction_status()
        assert async_status["superseded_batches"] + async_status["rejected_batches"] >= 1
    finally:
        engine.shutdown()


def test_summary_failure_backoff_does_not_wedge_foreground_compaction(tmp_path, monkeypatch):
    """Given background summary failure, backoff is visible but foreground can still compact."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)

        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("summary spend backoff open")),
        )
        batch = engine.prepare_background_compaction_once(messages)
        assert batch.state == "failed"
        assert engine.get_async_compaction_status()["failed_batches"] == 1

        monkeypatch.setattr(
            "hermes_lcm.engine.summarize_with_escalation",
            lambda **kwargs: ("foreground recovery summary", 0),
        )
        compacted = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)

        assert compacted != messages
        assert engine._last_compression_status == "compacted"
    finally:
        engine.shutdown()


def test_restart_recovers_or_discards_pending_batches_safely(tmp_path):
    """Given pending/preparing rows at shutdown, restart never treats them as canonical."""
    db_path = tmp_path / "restart.db"
    config = LCMConfig(
        database_path=str(db_path),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
    )

    engine = LCMEngine(config=config)
    engine.on_session_start("restart-session", conversation_id="restart-conversation", context_length=1_000)
    messages = _messages()
    try:
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages, leave_state="preparing")
        assert batch.state == "preparing"
    finally:
        engine.shutdown()

    restarted = LCMEngine(config=config)
    try:
        restarted.on_session_start("restart-session", conversation_id="restart-conversation", context_length=1_000)
        status = restarted.get_async_compaction_status()

        assert restarted._dag.get_session_node_count(restarted.current_session_id) == 0
        assert status["preparing_batches"] == 0
        assert status["pending_batches"] + status["rejected_batches"] + status["failed_batches"] >= 1
    finally:
        restarted.shutdown()


def test_successful_atomic_promotion_is_all_or_nothing(tmp_path):
    """Given a valid ready batch, node insert/frontier advance/batch state commit together."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        old_frontier = engine.get_status()["lifecycle"]["current_frontier_store_id"]

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is True
        assert engine._dag.get_session_node_count(engine.current_session_id) == batch.expected_leaf_count
        lifecycle = engine.get_status()["lifecycle"]
        assert lifecycle["current_frontier_store_id"] > old_frontier
        assert lifecycle["current_frontier_store_id"] == batch.frontier_end_store_id
        assert engine.get_async_compaction_status()["promoted_batches"] == 1
    finally:
        engine.shutdown()


def test_preflight_promotes_ready_batch_before_duplicate_leaf_summary(tmp_path, monkeypatch):
    """Normal threshold compaction adopts a ready batch without duplicate leaf work."""
    engine = _engine(tmp_path)
    calls = []

    def summarize(**kwargs):
        calls.append(kwargs)
        return "prepared summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        prepared_call_count = len(calls)

        assert engine.should_compress_preflight(messages)
        compacted = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)

        assert len([call for call in calls if call["depth"] == 0]) == prepared_call_count
        assert compacted != messages
        assert engine._dag.get_session_node_count(engine.current_session_id) >= batch.expected_leaf_count
        assert engine.get_async_compaction_status()["promoted_batches"] == 1
    finally:
        engine.shutdown()


def test_atomic_promotion_rolls_back_partial_publish_failure(tmp_path):
    """Given a mid-promotion failure, no canonical node/frontier/batch half-state remains."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        old_frontier = engine.get_status()["lifecycle"]["current_frontier_store_id"]
        engine._async_compaction_publish_failure_hook = "after_canonical_insert"

        with pytest.raises(RuntimeError, match="injected async promotion failure"):
            engine.promote_prepared_compaction(batch.batch_id, messages)

        lifecycle = engine.get_status()["lifecycle"]
        assert lifecycle["current_frontier_store_id"] == old_frontier
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        async_status = engine.get_async_compaction_status()
        assert async_status["promoted_batches"] == 0
        assert async_status["prepared_batches"] == 1
    finally:
        engine.shutdown()


def test_status_and_doctor_report_async_compaction_counts(tmp_path):
    """Given mixed async states, status and doctor expose pending/prepared/promoted/rejected counts."""
    engine = _engine(tmp_path)
    try:
        messages = _messages()
        engine.ingest(messages)
        ready = engine.prepare_background_compaction_once(messages)
        engine.reject_prepared_compaction(ready.batch_id, reason="policy_fingerprint_mismatch")
        engine.prepare_background_compaction_once(messages)

        status = json.loads(engine.handle_tool_call("lcm_status", {}))
        doctor = json.loads(engine.handle_tool_call("lcm_doctor", {}))

        assert status["async_compaction"]["prepared_batches"] == 1
        assert status["async_compaction"]["rejected_batches"] == 1
        async_checks = [check for check in doctor["checks"] if check["check"].startswith("async_compaction")]
        assert async_checks
        assert any("prepared_batches" in check["detail"] for check in async_checks)
    finally:
        engine.shutdown()


def test_on_turn_complete_enqueues_without_waiting_for_summary(tmp_path, monkeypatch):
    """The worker owns slow preparation; the completed-turn seam stays quick."""
    config = LCMConfig(
        database_path=str(tmp_path / "worker.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "worker-session",
        conversation_id="worker-conversation",
        platform="test",
        context_length=1_000,
    )
    started = Event()
    release = Event()

    def blocked_summary(**kwargs):
        started.set()
        assert release.wait(2.0)
        return "background summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
    try:
        messages = _messages()
        engine.ingest(messages)
        started_at = time.perf_counter()
        assert engine.on_turn_complete(
            messages,
            usage={"input_tokens": 123, "output_tokens": 45},
            platform="telegram",
            conversation_id="worker-conversation",
        ) is True
        assert time.perf_counter() - started_at < 0.5
        assert started.wait(2.0)
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
        release.set()
        assert engine.drain_async_compaction(timeout=3.0)
        assert engine.get_async_compaction_status()["prepared_batches"] == 1
    finally:
        release.set()
        engine.shutdown()


def test_concurrent_managers_claim_one_preparer_for_a_batch(tmp_path, monkeypatch):
    """Two SQLite-backed managers must never summarize the same batch."""
    engine_a = _engine(tmp_path, session_id="shared-session")
    engine_b = _engine(tmp_path, session_id="shared-session")
    started = Event()
    release = Event()
    calls = []
    results = []
    errors = []

    def blocked_summary(**kwargs):
        calls.append(kwargs)
        started.set()
        assert release.wait(3.0)
        return "one preparer", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
    messages = _messages()
    try:
        engine_a.ingest(messages)

        def prepare(engine):
            try:
                results.append(engine.prepare_background_compaction_once(messages))
            except Exception as exc:  # make a duplicate-writer failure observable
                errors.append(exc)

        first = Thread(target=prepare, args=(engine_a,))
        second = Thread(target=prepare, args=(engine_b,))
        first.start()
        assert started.wait(2.0)
        second.start()
        time.sleep(0.15)

        assert len(calls) == 1
        release.set()
        first.join(3.0)
        second.join(3.0)
        assert not errors
        assert len(results) == 2
        assert sorted(result.state for result in results) == ["preparing", "ready"]
        assert engine_a.get_async_compaction_status()["pending_summaries"] == results[0].expected_leaf_count
    finally:
        release.set()
        engine_b.shutdown()
        engine_a.shutdown()


def test_recovery_does_not_reject_another_live_preparer(tmp_path):
    """Opening a second manager must leave a non-expired lease untouched."""
    config = LCMConfig(
        database_path=str(tmp_path / "live-recovery.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
    )
    first = LCMEngine(config=config)
    second = None
    messages = _messages()
    try:
        first.on_session_start("live-session", conversation_id="live-conversation", context_length=1_000)
        first.ingest(messages)
        batch = first.prepare_background_compaction_once(messages, leave_state="preparing")

        second = LCMEngine(config=config)

        recovered = second._async_compaction.get_batch(batch.batch_id)
        assert recovered.state == "preparing"
        assert recovered.rejected_reason == ""
    finally:
        if second is not None:
            second.shutdown()
        first.shutdown()


def test_recovery_reclaims_only_an_expired_lease(tmp_path):
    """An expired preparation lease is reset for a later manager to claim."""
    config = LCMConfig(
        database_path=str(tmp_path / "stale-recovery.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
    )
    first = LCMEngine(config=config)
    second = None
    messages = _messages()
    try:
        first.on_session_start("stale-session", conversation_id="stale-conversation", context_length=1_000)
        first.ingest(messages)
        batch = first.prepare_background_compaction_once(messages, leave_state="preparing")
        first._async_compaction.connection.execute(
            "UPDATE compaction_batches SET lease_expires_at = ? WHERE batch_id = ?",
            (time.time() - 1.0, batch.batch_id),
        )
        first._async_compaction.connection.commit()

        second = LCMEngine(config=config)

        recovered = second._async_compaction.get_batch(batch.batch_id)
        assert recovered.state == "pending"
        assert recovered.rejected_reason == ""
        assert second._async_compaction.connection.execute(
            "SELECT COUNT(*) FROM pending_summary_nodes WHERE batch_id = ?",
            (batch.batch_id,),
        ).fetchone()[0] == 0
    finally:
        if second is not None:
            second.shutdown()
        first.shutdown()


def test_stale_worker_failure_cannot_delete_new_owner_pending_rows(tmp_path, monkeypatch):
    """A worker that lost its lease cannot clean up a newer owner's batch."""
    config = LCMConfig(
        database_path=str(tmp_path / "stale-worker.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
    )
    first = LCMEngine(config=config)
    second = None
    messages = _messages()
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("new owner summary", 0),
    )
    try:
        first.on_session_start("stale-worker-session", conversation_id="stale-worker-conversation", context_length=1_000)
        first.ingest(messages)
        original = first.prepare_background_compaction_once(messages, leave_state="preparing")
        first._async_compaction.connection.execute(
            "UPDATE compaction_batches SET lease_expires_at = ? WHERE batch_id = ?",
            (time.time() - 1.0, original.batch_id),
        )
        first._async_compaction.connection.commit()

        second = LCMEngine(config=config)
        second.on_session_start("stale-worker-session", conversation_id="stale-worker-conversation", context_length=1_000)
        replacement = second.prepare_background_compaction_once(messages)
        pending_count = replacement.expected_leaf_count
        assert replacement.state == "ready"
        assert first._async_compaction._mark_failed(
            original.batch_id,
            RuntimeError("old worker failed after lease loss"),
            owner_id=first._async_compaction._owner_id,
        ).state == "ready"
        assert first._async_compaction.connection.execute(
            "SELECT COUNT(*) FROM pending_summary_nodes WHERE batch_id = ?",
            (original.batch_id,),
        ).fetchone()[0] == pending_count
    finally:
        if second is not None:
            second.shutdown()
        first.shutdown()


def test_promotion_reuses_token_and_tool_group_fresh_tail_boundary(tmp_path, monkeypatch):
    """Promotion must reject a source moved into a protected assistant/tool group."""
    config = LCMConfig(
        database_path=str(tmp_path / "fresh-group.db"),
        fresh_tail_count=1,
        leaf_chunk_tokens=10,
        async_background_compaction_enabled=True,
    )
    engine = LCMEngine(config=config)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("group-safe summary", 0),
    )
    initial = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "old " + ("x " * 20)},
        {
            "role": "assistant",
            "content": "calling lookup",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "latest user turn"},
    ]
    try:
        engine.on_session_start("tail-session", conversation_id="tail-conversation", context_length=1_000)
        engine.ingest(initial)
        batch = engine.prepare_background_compaction_once(initial)
        assistant_id = engine._store.get_session_messages("tail-session")[2]["store_id"]
        assert assistant_id in batch.source_ids

        # A late reconciliation can turn the newest stored row into the tool
        # result that belongs to the preceding assistant call. Keep the batch's
        # source rows unchanged while moving the live tail boundary.
        latest_id = engine._store.get_session_messages("tail-session")[-1]["store_id"]
        engine._store._conn.execute(
            """
            UPDATE messages
            SET role = 'tool', tool_call_id = 'call-1', content = 'lookup result'
            WHERE store_id = ?
            """,
            (latest_id,),
        )
        engine._store._conn.commit()
        updated = initial[:-1] + [
            {"role": "tool", "tool_call_id": "call-1", "content": "lookup result"}
        ]

        result = engine.promote_prepared_compaction(batch.batch_id, updated)

        assert result.promoted is False
        assert result.reason == "fresh_tail_mismatch"
    finally:
        engine.shutdown()


def test_promotion_does_not_use_count_tail_when_token_tail_is_narrower(tmp_path, monkeypatch):
    """A token-bounded tail may protect fewer rows than its count limit."""
    config = LCMConfig(
        database_path=str(tmp_path / "fresh-token.db"),
        fresh_tail_count=4,
        fresh_tail_max_tokens=10,
        leaf_chunk_tokens=10,
        async_background_compaction_enabled=True,
    )
    engine = LCMEngine(config=config)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("token-safe summary", 0),
    )
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "old one " + ("x " * 20)},
        {"role": "assistant", "content": "old two " + ("y " * 20)},
        {"role": "user", "content": "newest"},
    ]
    try:
        engine.on_session_start("token-session", conversation_id="token-conversation", context_length=1_000)
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is True
    finally:
        engine.shutdown()


def test_worker_snapshot_survives_session_rebind(tmp_path, monkeypatch):
    """Preparation uses the captured session identity after the engine rebinds."""
    engine = _engine(tmp_path, session_id="snapshot-a", conversation_id="conversation-a")
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("captured session summary", 0),
    )
    messages = _messages()
    try:
        engine.ingest(messages)
        snapshot = engine._async_compaction.capture_snapshot()
        engine.on_session_start("snapshot-b", conversation_id="conversation-b", context_length=1_000)

        engine._async_compaction._run_snapshot(snapshot)

        row = engine._async_compaction.connection.execute(
            "SELECT state FROM compaction_batches WHERE session_id = ?",
            ("snapshot-a",),
        ).fetchone()
        assert row[0] == "ready"
    finally:
        engine.shutdown()


def test_worker_snapshot_uses_captured_batch_policy_after_config_rebind(tmp_path, monkeypatch):
    """Preparation persists the policy captured by the worker, not later config state."""
    engine = _engine(tmp_path, session_id="snapshot-policy")
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("captured policy summary", 0),
    )
    messages = _messages()
    try:
        engine.ingest(messages)
        snapshot = engine._async_compaction.capture_snapshot()
        expected_tail = snapshot.config.fresh_tail_count
        expected_leaf = snapshot.config.leaf_chunk_tokens

        engine._config.fresh_tail_count = expected_tail + 10
        engine._config.leaf_chunk_tokens = expected_leaf + 10
        engine._async_compaction._run_snapshot(snapshot)

        row = engine._async_compaction.connection.execute(
            "SELECT fresh_tail_count, leaf_chunk_tokens FROM compaction_batches "
            "WHERE session_id = ?",
            ("snapshot-policy",),
        ).fetchone()
        assert tuple(row) == (expected_tail, expected_leaf)
    finally:
        engine.shutdown()


def test_turn_callback_does_not_deepcopy_transcript(tmp_path, monkeypatch):
    """The callback copies small runtime state, not the host's full history."""
    config = LCMConfig(
        database_path=str(tmp_path / "bounded-callback.db"),
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("bounded-session", conversation_id="bounded-conversation", context_length=1_000)
    messages = [{"role": "user", "content": "x" * 10_000} for _ in range(500)]
    original_deepcopy = async_compaction_module.copy.deepcopy
    copied = []

    def record_deepcopy(value, *args, **kwargs):
        copied.append(value)
        return original_deepcopy(value, *args, **kwargs)

    monkeypatch.setattr(async_compaction_module.copy, "deepcopy", record_deepcopy)
    try:
        assert engine.on_turn_complete(messages) is True
        assert not any(value is messages for value in copied)
    finally:
        engine.shutdown()


def test_worker_drain_tracks_a_dequeued_job_as_in_flight():
    """Drain cannot report idle in the dequeue/active-state handoff window."""
    release = Event()
    dequeued = Event()
    worker = _BoundedBackgroundWorker(lambda snapshot: release.wait(2.0), max_items=1)

    def controlled_get(*args, **kwargs):
        if not args and not kwargs:
            item = Queue.get(worker._queue, block=False)
        else:
            item = Queue.get(worker._queue, *args, **kwargs)
        dequeued.set()
        assert release.wait(2.0)
        return item

    worker._queue.get = controlled_get
    worker._queue.get_nowait = controlled_get
    try:
        assert worker.enqueue(_BackgroundSnapshot("", "", ""))
        assert dequeued.wait(2.0)
        assert worker.drain(timeout=0.05) is False
    finally:
        release.set()
        assert worker.close(timeout=2.0)


def test_manager_does_not_close_connection_if_worker_did_not_stop(tmp_path):
    """A timed-out worker leaves its connection open for the live worker."""
    engine = _engine(tmp_path, session_id="close-contract")
    manager = engine._async_compaction

    class LiveWorker:
        queue_depth = 0
        active = True

        def close(self, timeout=None):
            return False

    manager._worker = LiveWorker()
    connection = manager.connection
    try:
        manager.close()
        assert manager.connection is connection
    finally:
        engine._async_compaction = None
        if connection is not None:
            connection.close()
        engine.shutdown()


def test_failed_preparation_rolls_back_pending_rows(tmp_path):
    """Failure cleanup is one transaction and preserves data when cleanup fails."""
    engine = _engine(tmp_path, session_id="failure-rollback")
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages, leave_state="preparing")
        manager = engine._async_compaction
        manager.connection.execute(
            """
            CREATE TRIGGER fail_async_batch_update
            BEFORE UPDATE OF state ON compaction_batches
            WHEN NEW.state = 'failed'
            BEGIN SELECT RAISE(ABORT, 'injected failure cleanup error'); END
            """
        )

        with pytest.raises(sqlite3.DatabaseError, match="injected failure cleanup error"):
            manager._mark_failed(batch.batch_id, RuntimeError("summary failed"))

        assert manager.connection.execute(
            "SELECT COUNT(*) FROM pending_summary_nodes WHERE batch_id = ?",
            (batch.batch_id,),
        ).fetchone()[0] == 0
        assert manager.get_batch(batch.batch_id).state == "preparing"
    finally:
        engine.shutdown()


def test_reject_rolls_back_when_pending_cleanup_fails(tmp_path):
    """Reject leaves a ready batch intact if its pending-row delete fails."""
    engine = _engine(tmp_path, session_id="reject-rollback")
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        manager = engine._async_compaction
        manager.connection.execute(
            """
            CREATE TRIGGER fail_async_pending_delete
            BEFORE DELETE ON pending_summary_nodes
            BEGIN SELECT RAISE(ABORT, 'injected reject cleanup error'); END
            """
        )

        with pytest.raises(sqlite3.DatabaseError, match="injected reject cleanup error"):
            manager.reject(batch.batch_id, "operator_rejected")

        assert manager.get_batch(batch.batch_id).state == "ready"
        assert manager.connection.execute(
            "SELECT COUNT(*) FROM pending_summary_nodes WHERE batch_id = ?",
            (batch.batch_id,),
        ).fetchone()[0] == batch.expected_leaf_count
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    ("field", "value"),
    (("base_url", "https://changed.example"), ("api_mode", "responses")),
)
def test_route_fingerprint_rejects_endpoint_changes(tmp_path, field, value):
    engine = _engine(tmp_path, session_id=f"route-{field}")
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        setattr(engine, field, value)

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "summary_route_fingerprint_mismatch"
    finally:
        engine.shutdown()


def test_policy_fingerprint_rejects_session_filter_changes(tmp_path):
    engine = _engine(tmp_path, session_id="policy-session")
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        engine._config.ignore_session_patterns = ["new-policy"]

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
    finally:
        engine.shutdown()


def test_rescue_shrinkage_cannot_publish_partial_batch(tmp_path, monkeypatch):
    engine = _engine(tmp_path, session_id="rescue-session")
    try:
        messages = _messages()
        engine.ingest(messages)

        def shrink(chunk, *args, **kwargs):
            return chunk[:-1], 1, "partial summary", 0, 2

        monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", shrink)
        batch = engine.prepare_background_compaction_once(messages)

        assert batch.state != "ready"
        assert engine._async_compaction.connection.execute(
            "SELECT COUNT(*) FROM pending_summary_nodes WHERE batch_id = ?",
            (batch.batch_id,),
        ).fetchone()[0] == 0
    finally:
        engine.shutdown()


def test_async_publication_runs_rollup_and_transcript_gc_hooks_after_commit(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "maintenance.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
        temporal_rollups_enabled=True,
        large_output_transcript_gc_enabled=True,
    )
    engine = LCMEngine(config=config)
    rollup_nodes = []
    gc_calls = []
    monkeypatch.setattr(engine, "_invalidate_rollups_for_published_node", lambda node: rollup_nodes.append(node))
    monkeypatch.setattr(
        engine,
        "_maybe_gc_compacted_tool_results",
        lambda compacted, source_ids, **kwargs: gc_calls.append((compacted, source_ids)),
    )
    try:
        messages = _messages()
        engine.on_session_start("maintenance-session", conversation_id="maintenance-conversation", context_length=1_000)
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is True
        assert len(rollup_nodes) == batch.expected_leaf_count
        assert len(gc_calls) == batch.expected_leaf_count
    finally:
        engine.shutdown()


def test_promotion_does_not_acquire_engine_state_lock_under_manager_lock(tmp_path, monkeypatch):
    """Promotion and session rebind use one lock order instead of deadlocking."""
    engine = _engine(tmp_path, session_id="lock-order")
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("lock-order summary", 0),
    )
    messages = _messages()
    try:
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        manager = engine._async_compaction

        class StateLockProbe:
            def __enter__(self):
                assert not manager._lock._is_owned(), (
                    "promotion must release its manager lock before taking engine state lock"
                )
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        engine._async_state_lock = StateLockProbe()
        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is True
    finally:
        engine.shutdown()
