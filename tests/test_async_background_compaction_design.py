"""Contract tests for opt-in async/background compaction."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from queue import Queue
from pathlib import Path
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


def test_two_engines_do_not_foreground_publish_after_promotion_miss(tmp_path, monkeypatch):
    """A promotion miss must not let a second engine publish overlapping leaves."""
    engine_a = _engine(tmp_path, session_id="shared-foreground-session")
    engine_b = _engine(tmp_path, session_id="shared-foreground-session")
    promoted = Event()
    original_promote_next = engine_a._async_compaction.promote_next

    def promote_a(messages):
        result = original_promote_next(messages)
        if result is not None and result.promoted:
            promoted.set()
        return result

    engine_a._async_compaction.promote_next = promote_a
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("foreground race summary", 0),
    )
    messages = _messages()
    try:
        engine_a.ingest(messages)
        batch = engine_a.prepare_background_compaction_once(messages)
        assert batch is not None

        engine_a._last_compression_status = "pending"
        engine_b._last_compression_status = "pending"
        results = []

        first = Thread(
            target=lambda: results.append(
                engine_a.compress(messages, current_tokens=engine_a.threshold_tokens + 1)
            )
        )

        def run_second():
            assert promoted.wait(3.0)
            results.append(
                engine_b.compress(messages, current_tokens=engine_b.threshold_tokens + 1)
            )

        second = Thread(target=run_second)
        first.start()
        second.start()
        first.join(5.0)
        second.join(5.0)

        assert len(results) == 2
        nodes = [
            node
            for node in engine_a._dag.get_session_nodes("shared-foreground-session")
            if node.source_type == "messages"
        ]
        source_sets = [set(node.source_ids) for node in nodes]
        assert len(nodes) == batch.expected_leaf_count
        assert all(not (left & right) for index, left in enumerate(source_sets) for right in source_sets[index + 1:])
    finally:
        engine_b.shutdown()
        engine_a.shutdown()


def test_expired_foreground_claim_fence_discards_stale_provider_result(tmp_path, monkeypatch):
    """A stolen foreground claim must fence the old provider result at publish."""
    db_path = tmp_path / "foreground-fence.db"
    config = LCMConfig(
        database_path=str(db_path),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=False,
        async_background_compaction_lease_seconds=1.0,
    )
    engine_a = LCMEngine(config=config)
    engine_b = LCMEngine(config=config)
    started = Event()
    release_first = Event()
    calls = []
    errors = []

    def summarize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            started.set()
            assert release_first.wait(3.0)
        return "fenced foreground summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
    messages = _messages()
    first_result = []
    second_result = []
    try:
        engine_a.on_session_start("fence-session", conversation_id="fence-conversation", context_length=1_000)
        engine_b.on_session_start("fence-session", conversation_id="fence-conversation", context_length=1_000)
        engine_a.ingest(messages)

        first = Thread(
            target=lambda: first_result.append(
                engine_a.compress(messages, current_tokens=engine_a.threshold_tokens + 1)
            )
        )
        first.start()
        assert started.wait(2.0)

        external = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            external.execute(
                "UPDATE foreground_compaction_claims SET lease_expires_at = ? "
                "WHERE conversation_id = ? AND session_id = ?",
                (time.time() - 1.0, "fence-conversation", "fence-session"),
            )
            external.commit()
        finally:
            external.close()

        second = Thread(
            target=lambda: second_result.append(
                engine_b.compress(messages, current_tokens=engine_b.threshold_tokens + 1)
            )
        )
        second.start()
        second.join(3.0)
        assert not second.is_alive()
        release_first.set()
        first.join(3.0)
        assert not first.is_alive()

        nodes = [
            node
            for node in engine_a._dag.get_session_nodes("fence-session")
            if node.source_type == "messages"
        ]
        assert len(calls) == 2
        assert len(first_result) == 1
        assert len(second_result) == 1
        assert len(nodes) == 1
        assert len(nodes[0].source_ids) == len(set(nodes[0].source_ids))
        assert engine_a._async_compaction.connection.execute(
            "SELECT next_token FROM foreground_compaction_fence WHERE fence_id = 1"
        ).fetchone()[0] >= 2
    except Exception as exc:
        errors.append(exc)
    finally:
        release_first.set()
        if first.is_alive():
            first.join(3.0)
        if 'second' in locals() and second.is_alive():
            second.join(3.0)
        engine_b.shutdown()
        engine_a.shutdown()
    assert not errors


def test_foreground_publication_is_excluded_across_managers_when_async_disabled(tmp_path, monkeypatch):
    """The foreground duplicate guard remains active without background prep."""
    db_path = tmp_path / "foreground-disabled.db"
    config = LCMConfig(
        database_path=str(db_path),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=False,
        async_background_compaction_worker_enabled=False,
    )
    engine_a = LCMEngine(config=config)
    engine_b = LCMEngine(config=config)
    started = Event()
    release = Event()
    calls = []
    errors = []

    def summarize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            started.set()
            assert release.wait(3.0)
        return "disabled async foreground summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", summarize)
    messages = _messages()
    results = []
    try:
        engine_a.on_session_start("disabled-race-session", conversation_id="disabled-race-conversation", context_length=1_000)
        engine_b.on_session_start("disabled-race-session", conversation_id="disabled-race-conversation", context_length=1_000)
        engine_a.ingest(messages)

        first = Thread(
            target=lambda: results.append(
                engine_a.compress(messages, current_tokens=engine_a.threshold_tokens + 1)
            )
        )
        second = Thread(
            target=lambda: results.append(
                engine_b.compress(messages, current_tokens=engine_b.threshold_tokens + 1)
            )
        )
        first.start()
        assert started.wait(2.0)
        second.start()
        time.sleep(0.15)
        assert len(calls) == 1
        release.set()
        first.join(3.0)
        second.join(3.0)
        assert not first.is_alive()
        assert not second.is_alive()
        nodes = [
            node
            for node in engine_a._dag.get_session_nodes("disabled-race-session")
            if node.source_type == "messages"
        ]
        assert len(results) == 2
        assert len(nodes) == 1
        assert engine_a.get_async_compaction_status()["enabled"] is False
    except Exception as exc:
        errors.append(exc)
    finally:
        release.set()
        engine_b.shutdown()
        engine_a.shutdown()
    assert not errors


def test_foreground_rebind_discards_hung_provider_without_writing_new_db(tmp_path, monkeypatch):
    """A foreground operation finishing after rebind cannot publish into the new profile."""
    home_a = tmp_path / "foreground-home-a"
    home_b = tmp_path / "foreground-home-b"
    config = LCMConfig(
        database_path="",
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=False,
    )
    engine = LCMEngine(config=config, hermes_home=str(home_a))
    started = Event()
    release = Event()
    errors = []

    def blocked_summary(**kwargs):
        started.set()
        assert release.wait(3.0)
        return "stale rebind summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
    messages = _messages()
    result = []
    try:
        engine.on_session_start(
            "old-session",
            conversation_id="old-conversation",
            hermes_home=str(home_a),
            context_length=1_000,
        )
        engine.ingest(messages)
        old_db = home_a / "lcm.db"
        worker = Thread(
            target=lambda: result.append(
                engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
            )
        )
        worker.start()
        assert started.wait(2.0)

        rebound = Thread(
            target=lambda: engine.on_session_start(
                "new-session",
                conversation_id="new-conversation",
                hermes_home=str(home_b),
                context_length=1_000,
            )
        )
        rebound.start()
        rebound.join(1.0)
        assert not rebound.is_alive()
        assert Path(engine._store.db_path) == home_b / "lcm.db"

        release.set()
        worker.join(3.0)
        assert not worker.is_alive()
        assert len(result) == 1
        assert result[0] == messages

        new_conn = sqlite3.connect(str(home_b / "lcm.db"))
        try:
            assert new_conn.execute(
                "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?",
                ("new-session",),
            ).fetchone()[0] == 0
        finally:
            new_conn.close()
        old_conn = sqlite3.connect(str(old_db))
        try:
            assert old_conn.execute(
                "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?",
                ("old-session",),
            ).fetchone()[0] == 0
        finally:
            old_conn.close()
    except Exception as exc:
        errors.append(exc)
    finally:
        release.set()
        engine.shutdown()
    assert not errors


def test_foreground_shutdown_is_bounded_and_discards_hung_provider_result(tmp_path, monkeypatch):
    """Shutdown does not close a foreground operation's resources or leak its result."""
    db_path = tmp_path / "foreground-shutdown.db"
    config = LCMConfig(
        database_path=str(db_path),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=False,
    )
    engine = LCMEngine(config=config)
    started = Event()
    release = Event()
    errors = []

    def blocked_summary(**kwargs):
        started.set()
        assert release.wait(3.0)
        return "stale shutdown summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
    messages = _messages()
    result = []
    worker = None
    try:
        engine.on_session_start("shutdown-session", conversation_id="shutdown-conversation", context_length=1_000)
        engine.ingest(messages)
        worker = Thread(
            target=lambda: result.append(
                engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
            )
        )
        worker.start()
        assert started.wait(2.0)

        shutdown = Thread(target=engine.shutdown)
        shutdown.start()
        shutdown.join(1.0)
        assert not shutdown.is_alive()

        release.set()
        worker.join(3.0)
        assert not worker.is_alive()
        assert len(result) == 1
        assert result[0] == messages
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?",
                ("shutdown-session",),
            ).fetchone()[0] == 0
        finally:
            conn.close()
    except Exception as exc:
        errors.append(exc)
    finally:
        release.set()
        if worker is not None and worker.is_alive():
            worker.join(3.0)
        engine.shutdown()
    assert not errors


def test_promotion_yields_to_live_foreground_claim(tmp_path, monkeypatch):
    """A foreground claimant that wins first must block async promotion."""
    engine_a = _engine(tmp_path, session_id="promotion-foreground-session")
    engine_b = _engine(tmp_path, session_id="promotion-foreground-session")
    claim_entered = Event()
    release_claim = Event()
    original_claim = engine_b._async_compaction.claim_foreground_sources

    def claim_and_hold(**kwargs):
        assert original_claim(**kwargs)
        claim_entered.set()
        assert release_claim.wait(3.0)
        return True

    engine_b._async_compaction.claim_foreground_sources = claim_and_hold
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("foreground claim wins", 0),
    )
    messages = _messages()
    worker = Thread(
        target=lambda: engine_b.compress(
            messages,
            current_tokens=engine_b.threshold_tokens + 1,
        )
    )
    try:
        engine_a.ingest(messages)
        batch = engine_a.prepare_background_compaction_once(messages)
        assert batch is not None
        engine_b._last_compression_status = "running"
        worker.start()
        assert claim_entered.wait(3.0)

        result = engine_a.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "foreground_compaction_in_progress"
    finally:
        release_claim.set()
        worker.join(5.0)
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


@pytest.mark.parametrize("lock_name", ("state", "manager"))
def test_on_turn_complete_does_not_wait_for_runtime_locks(tmp_path, lock_name):
    """Reply completion stays bounded while lifecycle or manager locks are held."""
    config = LCMConfig(
        database_path=str(tmp_path / f"callback-lock-{lock_name}.db"),
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "callback-lock-session",
        conversation_id="callback-lock-conversation",
        context_length=1_000,
    )
    lock = engine._async_state_lock if lock_name == "state" else engine._async_compaction._lock
    returned = Event()
    result = []
    messages = [{"role": "user", "content": "reply completed"}]
    lock.acquire()
    try:
        callback_thread = Thread(
            target=lambda: (result.append(engine.on_turn_complete(messages)), returned.set())
        )
        callback_thread.start()
        assert returned.wait(0.2)
        assert result == [True]
        callback_thread.join(1.0)
    finally:
        lock.release()
        engine.shutdown()


def test_on_turn_complete_does_not_capture_snapshot_under_sqlite_contention(tmp_path, monkeypatch):
    """SQLite/snapshot work is deferred until after the cheap trigger enqueue."""
    config = LCMConfig(
        database_path=str(tmp_path / "callback-sqlite-lock.db"),
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "callback-sqlite-session",
        conversation_id="callback-sqlite-conversation",
        context_length=1_000,
    )
    snapshot_started = Event()
    release_snapshot = Event()

    def blocked_snapshot(*args, **kwargs):
        snapshot_started.set()
        assert release_snapshot.wait(2.0)
        return None

    monkeypatch.setattr(engine._async_compaction, "capture_snapshot", blocked_snapshot)
    sqlite_lock = sqlite3.connect(config.database_path, timeout=0.1)
    sqlite_lock.execute("BEGIN IMMEDIATE")
    returned = Event()
    result = []
    try:
        callback_thread = Thread(
            target=lambda: (result.append(engine.on_turn_complete([{"role": "user", "content": "done"}]),), returned.set())
        )
        callback_thread.start()
        assert returned.wait(0.2)
        assert result == [True]
        assert snapshot_started.wait(1.0)
    finally:
        release_snapshot.set()
        callback_thread.join(2.0)
        sqlite_lock.rollback()
        sqlite_lock.close()
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


def test_shutdown_is_bounded_with_hung_summarizer(tmp_path, monkeypatch):
    """Shutdown returns before a live daemon worker releases its provider call."""
    config = LCMConfig(
        database_path=str(tmp_path / "hung-shutdown.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start("hung-shutdown-session", conversation_id="hung-shutdown-conversation", context_length=1_000)
    started = Event()
    release = Event()

    def blocked_summary(**kwargs):
        started.set()
        assert release.wait(3.0)
        return "eventual summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
    manager = engine._async_compaction
    messages = _messages()
    engine.ingest(messages)
    assert engine.on_turn_complete(messages) is True
    assert started.wait(2.0)
    connection = manager.connection
    shutdown_returned = Event()
    shutdown_thread = Thread(target=lambda: (engine.shutdown(), shutdown_returned.set()))
    shutdown_thread.start()
    try:
        assert shutdown_returned.wait(0.4)
        assert connection is not None
        assert connection.execute("SELECT 1").fetchone()[0] == 1
        assert manager._worker._thread.daemon is True
    finally:
        release.set()
        shutdown_thread.join(3.0)
        manager.drain(timeout=3.0)


def test_storage_rebind_is_bounded_with_hung_summarizer(tmp_path, monkeypatch):
    """Profile rebind leaves a live worker's SQLite connection untouched."""
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    config = LCMConfig(
        database_path="",
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config, hermes_home=str(home_a))
    engine.on_session_start(
        "hung-rebind-session",
        conversation_id="hung-rebind-conversation",
        hermes_home=str(home_a),
        context_length=1_000,
    )
    started = Event()
    release = Event()

    def blocked_summary(**kwargs):
        started.set()
        assert release.wait(3.0)
        return "eventual rebind summary", 0

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
    old_manager = engine._async_compaction
    old_connection = old_manager.connection
    messages = _messages()
    engine.ingest(messages)
    assert engine.on_turn_complete(messages) is True
    assert started.wait(2.0)
    rebind_returned = Event()
    rebind_thread = Thread(
        target=lambda: (
            engine.on_session_start(
                "rebound-session",
                conversation_id="rebound-conversation",
                hermes_home=str(home_b),
                context_length=1_000,
            ),
            rebind_returned.set(),
        )
    )
    rebind_thread.start()
    try:
        assert rebind_returned.wait(0.4)
        assert Path(engine._store.db_path) == home_b / "lcm.db"
        assert old_connection is not None
        assert old_connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        release.set()
        rebind_thread.join(3.0)
        old_manager.drain(timeout=3.0)
        engine.shutdown()


def test_worker_capture_snapshot_survives_rebind_while_foreground_finishes(
    tmp_path, monkeypatch
):
    """A starting worker keeps the retired manager connection alive."""
    for iteration in range(20):
        home_a = tmp_path / f"worker-lifecycle-a-{iteration}"
        home_b = tmp_path / f"worker-lifecycle-b-{iteration}"
        config = LCMConfig(
            database_path="",
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            async_background_compaction_enabled=True,
            async_background_compaction_worker_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(home_a))
        engine.on_session_start(
            "lifecycle-old",
            conversation_id="lifecycle-old-conversation",
            hermes_home=str(home_a),
            context_length=1_000,
        )
        messages = _messages(prefix=f"lifecycle-{iteration}")
        engine.ingest(messages)
        old_manager = engine._async_compaction
        assert old_manager is not None
        snapshot_entered = Event()
        allow_snapshot = Event()
        capture_errors = []
        original_capture = old_manager.capture_snapshot

        def delayed_capture(*args, **kwargs):
            snapshot_entered.set()
            assert allow_snapshot.wait(3.0)
            try:
                return original_capture(*args, **kwargs)
            except BaseException as exc:  # pragma: no cover - red-path evidence
                capture_errors.append(exc)
                raise

        monkeypatch.setattr(old_manager, "capture_snapshot", delayed_capture)
        provider_started = Event()
        allow_provider = Event()

        def blocked_summary(**kwargs):
            provider_started.set()
            assert allow_provider.wait(3.0)
            return "foreground lifecycle summary", 0

        monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
        foreground_result = []
        foreground = Thread(
            target=lambda: foreground_result.append(
                engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
            )
        )
        foreground.start()
        assert provider_started.wait(2.0)
        assert engine.on_turn_complete(messages) is True
        assert snapshot_entered.wait(2.0)

        rebound = Event()
        rebind = Thread(
            target=lambda: (
                engine.on_session_start(
                    "lifecycle-old",
                    conversation_id="lifecycle-old-conversation",
                    hermes_home=str(home_b),
                    context_length=1_000,
                ),
                rebound.set(),
            )
        )
        rebind.start()
        try:
            assert rebound.wait(1.0)
            allow_provider.set()
            foreground.join(3.0)
            assert not foreground.is_alive()
            allow_snapshot.set()
            assert old_manager.drain(timeout=3.0)
            assert not capture_errors
            assert foreground_result == [messages]
        finally:
            allow_provider.set()
            allow_snapshot.set()
            rebind.join(3.0)
            foreground.join(3.0)
            old_manager.drain(timeout=3.0)
            engine.shutdown()


@pytest.mark.parametrize("worker_stage", ("queued", "dequeued", "active"))
def test_manager_close_defers_connection_for_every_worker_stage(tmp_path, worker_stage):
    """A bounded close cannot invalidate queued or in-flight worker state."""
    config = LCMConfig(
        database_path=str(tmp_path / f"worker-stage-{worker_stage}.db"),
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        "worker-stage-session",
        conversation_id="worker-stage-conversation",
        context_length=1_000,
    )
    manager = engine._async_compaction
    assert manager is not None and manager._worker is not None
    worker = manager._worker
    connection = manager.connection
    release = Event()
    entered = Event()

    def callback(_item):
        entered.set()
        if worker_stage in {"active", "queued"}:
            assert release.wait(3.0)

    worker._callback = callback
    if worker_stage == "dequeued":
        def controlled_get(*args, **kwargs):
            item = Queue.get(worker._queue, *args, **kwargs)
            entered.set()
            assert release.wait(3.0)
            return item

        worker._queue.get = controlled_get
        worker._queue.get_nowait = controlled_get

    try:
        assert worker.enqueue("first")
        assert entered.wait(2.0)
        if worker_stage == "queued":
            assert worker.enqueue("second")
        manager.close()
        assert connection is not None
        assert manager.connection is connection
    finally:
        release.set()
        assert worker.close(timeout=3.0)
        manager.close()
        assert manager.connection is None
        engine.shutdown()


def test_foreground_publish_rejects_rewritten_source_inside_publish_fence(
    tmp_path, monkeypatch
):
    """A source rewrite during the provider call invalidates the old result."""
    for iteration in range(20):
        db_path = tmp_path / f"foreground-source-fence-{iteration}.db"
        config = LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            context_threshold=0.10,
            async_background_compaction_enabled=True,
            async_background_compaction_worker_enabled=False,
        )
        engine_a = LCMEngine(config=config)
        engine_b = LCMEngine(config=config)
        engine_a.on_session_start(
            "source-fence-session",
            conversation_id="source-fence-conversation",
            context_length=1_000,
        )
        engine_b.on_session_start(
            "source-fence-session",
            conversation_id="source-fence-conversation",
            context_length=1_000,
        )
        messages = _messages(prefix=f"source-fence-{iteration}")
        engine_a.ingest(messages)
        started = Event()
        release = Event()

        def blocked_summary(**kwargs):
            started.set()
            assert release.wait(3.0)
            return "stale source summary", 0

        monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", blocked_summary)
        result = []
        worker = Thread(
            target=lambda: result.append(
                engine_a.compress(messages, current_tokens=engine_a.threshold_tokens + 1)
            )
        )
        worker.start()
        try:
            assert started.wait(2.0)
            source_id = int(
                engine_a._async_compaction.connection.execute(
                    "SELECT store_id FROM messages WHERE session_id = ? AND role != 'system' "
                    "ORDER BY store_id LIMIT 1",
                    ("source-fence-session",),
                ).fetchone()[0]
            )
            engine_b._store._conn.execute(
                "UPDATE messages SET content = content || ' rewritten by manager b' "
                "WHERE store_id = ?",
                (source_id,),
            )
            engine_b._store._conn.commit()
            release.set()
            worker.join(3.0)
            assert not worker.is_alive()
            assert engine_a._dag.get_session_nodes("source-fence-session") == []
        finally:
            release.set()
            worker.join(3.0)
            engine_b.shutdown()
            engine_a.shutdown()


def test_foreground_marker_does_not_cross_profile_rebind_after_publish(
    tmp_path, monkeypatch
):
    """A published old-profile result cannot advance the new profile marker."""
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("marker rebind summary", 0),
    )
    for iteration in range(20):
        home_a = tmp_path / f"marker-old-{iteration}"
        home_b = tmp_path / f"marker-new-{iteration}"
        config = LCMConfig(
            database_path="",
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            context_threshold=0.10,
            async_background_compaction_enabled=False,
        )
        engine = LCMEngine(config=config, hermes_home=str(home_a))
        engine.on_session_start(
            "marker-old-session",
            conversation_id="marker-old-conversation",
            hermes_home=str(home_a),
            context_length=1_000,
        )
        messages = _messages(prefix=f"marker-{iteration}")
        engine.ingest(messages)
        published = Event()
        rebound = Event()

        def rebind_after_publish(*args, **kwargs):
            published.set()
            assert rebound.wait(3.0)

        monkeypatch.setattr(engine, "_invalidate_rollups_for_published_node", rebind_after_publish)

        def rebind():
            assert published.wait(3.0)
            engine.on_session_start(
                "marker-new-session",
                conversation_id="marker-new-conversation",
                hermes_home=str(home_b),
                context_length=1_000,
            )
            rebound.set()

        rebind_thread = Thread(target=rebind)
        rebind_thread.start()
        try:
            result = engine.compress(messages, current_tokens=engine.threshold_tokens + 1)
            rebind_thread.join(3.0)
            assert not rebind_thread.is_alive()
            assert result == messages
            assert engine._last_compacted_store_id == 0
            new_message = {"role": "user", "content": f"new profile message {iteration}"}
            new_store_id = engine._store.append(
                "marker-new-session",
                new_message,
                token_estimate=8,
                source="test",
                conversation_id="marker-new-conversation",
            )
            assert engine._get_store_id_map_for_messages([new_message]) == {id(new_message): new_store_id}
            old_conn = sqlite3.connect(str(home_a / "lcm.db"))
            new_conn = sqlite3.connect(str(home_b / "lcm.db"))
            try:
                assert old_conn.execute(
                    "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?",
                    ("marker-old-session",),
                ).fetchone()[0] == 1
                assert new_conn.execute(
                    "SELECT COUNT(*) FROM summary_nodes WHERE session_id = ?",
                    ("marker-new-session",),
                ).fetchone()[0] == 0
            finally:
                old_conn.close()
                new_conn.close()
        finally:
            rebound.set()
            rebind_thread.join(3.0)
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("large_output_externalization_path", "/tmp/changed-lcm-payloads"),
        ("summary_circuit_breaker_failure_threshold", 7),
        ("summary_circuit_breaker_cooldown_seconds", 17),
        ("summary_spend_max_calls", 3),
        ("summary_spend_window_seconds", 17.5),
        ("summary_spend_backoff_seconds", 27.5),
    ),
)
def test_policy_fingerprint_rejects_preparation_setting_changes(tmp_path, field, value):
    """Every preparation route/guard setting participates in promotion identity."""
    engine = _engine(tmp_path, session_id=f"fingerprint-{field}")
    try:
        messages = _messages()
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        setattr(engine._config, field, value)

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is False
        assert result.reason == "policy_fingerprint_mismatch"
        assert engine._dag.get_session_node_count(engine.current_session_id) == 0
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
    monkeypatch.setattr(
        engine,
        "_invalidate_rollups_for_published_node",
        lambda node, **kwargs: rollup_nodes.append(node),
    )
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


def test_post_publish_maintenance_uses_publication_snapshot_after_rebind(tmp_path, monkeypatch):
    """Maintenance keeps the published batch binding when the engine rebinds."""
    config = LCMConfig(
        database_path=str(tmp_path / "maintenance-race.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        async_background_compaction_enabled=True,
        temporal_rollups_enabled=True,
        large_output_transcript_gc_enabled=True,
    )
    engine = LCMEngine(config=config)
    hook_started = Event()
    rebound = Event()
    rollup_calls = []
    gc_calls = []

    def rollup_hook(node, **kwargs):
        hook_started.set()
        assert rebound.wait(3.0)
        rollup_calls.append((node, kwargs))

    def gc_hook(compacted, source_ids, **kwargs):
        gc_calls.append((compacted, source_ids, kwargs))

    monkeypatch.setattr(engine, "_invalidate_rollups_for_published_node", rollup_hook)
    monkeypatch.setattr(engine, "_maybe_gc_compacted_tool_results", gc_hook)
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("snapshot maintenance summary", 0),
    )
    messages = _messages()

    def rebind():
        assert hook_started.wait(3.0)
        engine.on_session_start(
            "maintenance-new-session",
            conversation_id="maintenance-new-conversation",
            context_length=1_000,
        )
        engine._config.temporal_rollups_enabled = False
        engine._config.large_output_transcript_gc_enabled = False
        rebound.set()

    rebind_thread = Thread(target=rebind)
    rebind_thread.start()
    try:
        engine.on_session_start(
            "maintenance-session",
            conversation_id="maintenance-conversation",
            context_length=1_000,
        )
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)

        result = engine.promote_prepared_compaction(batch.batch_id, messages)

        assert result.promoted is True
        rebind_thread.join(3.0)
        assert rebound.is_set()
        assert rollup_calls
        assert gc_calls
        rollup_kwargs = rollup_calls[0][1]
        gc_kwargs = gc_calls[0][2]
        assert rollup_kwargs["session_id"] == "maintenance-session"
        assert rollup_kwargs["config"].temporal_rollups_enabled is True
        assert gc_kwargs["session_id"] == "maintenance-session"
        assert gc_kwargs["config"].large_output_transcript_gc_enabled is True
        assert gc_kwargs["hermes_home"] == engine._hermes_home
    finally:
        rebound.set()
        rebind_thread.join(3.0)
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


def test_same_identity_rebind_during_blocked_post_publish_maintenance_is_bounded(
    tmp_path, monkeypatch
):
    """A profile rebind with the same session identity cannot deadlock promotion."""
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **kwargs: ("same-identity maintenance summary", 0),
    )
    leaked_threads = []
    for iteration in range(50):
        home_a = tmp_path / f"maintenance-lock-a-{iteration}"
        home_b = tmp_path / f"maintenance-lock-b-{iteration}"
        config = LCMConfig(
            database_path="",
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            async_background_compaction_enabled=True,
            temporal_rollups_enabled=True,
            large_output_transcript_gc_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(home_a))
        maintenance_started = Event()
        rebind_returned = Event()
        allow_maintenance = Event()
        promotion_result = []
        errors = []

        def blocked_rollup(node, **kwargs):
            maintenance_started.set()
            if not rebind_returned.wait(1.0):
                errors.append(RuntimeError("same-identity rebind did not return"))
            if not allow_maintenance.wait(1.0):
                errors.append(RuntimeError("maintenance was not released"))

        monkeypatch.setattr(engine, "_invalidate_rollups_for_published_node", blocked_rollup)
        try:
            session_id = "same-identity-session"
            conversation_id = "same-identity-conversation"
            engine.on_session_start(
                session_id,
                conversation_id=conversation_id,
                hermes_home=str(home_a),
                context_length=1_000,
            )
            messages = _messages(prefix=f"same-identity-{iteration}")
            engine.ingest(messages)
            batch = engine.prepare_background_compaction_once(messages)

            promotion = Thread(
                target=lambda: promotion_result.append(
                    engine.promote_prepared_compaction(batch.batch_id, messages)
                ),
                name=f"lcm-test-promotion-{iteration}",
            )
            rebind = Thread(
                target=lambda: (
                    engine.on_session_start(
                        session_id,
                        conversation_id=conversation_id,
                        hermes_home=str(home_b),
                        context_length=1_000,
                    ),
                    rebind_returned.set(),
                ),
                name=f"lcm-test-rebind-{iteration}",
            )
            promotion.start()
            assert maintenance_started.wait(1.0)
            rebind.start()
            rebind.join(1.0)
            allow_maintenance.set()
            promotion.join(1.0)
            rebind.join(1.0)
            # Break the expected red-path cycle so this repro never leaves
            # non-daemon test threads behind after documenting the failure.
            if promotion.is_alive() or rebind.is_alive():
                rebind_returned.set()
                allow_maintenance.set()
                promotion.join(1.0)
                rebind.join(1.0)
            assert not promotion.is_alive()
            assert not rebind.is_alive()
            assert not errors
            assert promotion_result and promotion_result[0].promoted is True
            assert engine._last_compacted_store_id == 0
        finally:
            allow_maintenance.set()
            leaked_threads.extend(
                thread
                for thread in threading.enumerate()
                if thread.name in {
                    f"lcm-test-promotion-{iteration}",
                    f"lcm-test-rebind-{iteration}",
                }
                and thread.is_alive()
            )
            engine.shutdown()

    assert not leaked_threads


def test_status_close_race_is_coherent_and_does_not_use_closed_connection(
    tmp_path,
):
    """Status and close share a safe connection lifetime under repeated races."""

    class ConnectionProxy:
        def __init__(self, connection, status_thread_id):
            self._connection = connection
            self._status_thread_id = status_thread_id
            self.query_started = Event()
            self.allow_query = Event()
            self._blocked = False

        def execute(self, sql, *args):
            if (
                threading.get_ident() == self._status_thread_id
                and not self._blocked
                and str(sql).lstrip().upper().startswith("SELECT STATE")
            ):
                self._blocked = True
                self.query_started.set()
                assert self.allow_query.wait(1.0)
            return self._connection.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._connection, name)

    for iteration in range(100):
        config = LCMConfig(
            database_path=str(tmp_path / f"status-close-{iteration}.db"),
            async_background_compaction_enabled=True,
        )
        engine = LCMEngine(config=config)
        manager = engine._async_compaction
        assert manager is not None
        original_connection = manager.connection
        assert original_connection is not None
        status_errors = []
        status_values = []
        close_returned = Event()
        status_thread_id = [None]
        proxy = ConnectionProxy(original_connection, None)
        manager._conn = proxy

        def run_status():
            status_thread_id[0] = threading.get_ident()
            proxy._status_thread_id = status_thread_id[0]
            try:
                status_values.append(manager.status())
            except BaseException as exc:  # pragma: no cover - red-path evidence
                status_errors.append(exc)

        status = Thread(target=run_status, name=f"lcm-test-status-{iteration}")
        closer = Thread(
            target=lambda: (manager.close(), close_returned.set()),
            name=f"lcm-test-close-{iteration}",
        )
        try:
            status.start()
            assert proxy.query_started.wait(1.0)
            closer.start()
            # The unfixed status path lets close invalidate the connection while
            # the first query is paused. A lock-protected path keeps close behind
            # this query, so release it after giving the old path a chance to race.
            close_returned.wait(0.05)
            proxy.allow_query.set()
            status.join(1.0)
            closer.join(1.0)
            assert not status.is_alive()
            assert not closer.is_alive()
            assert not status_errors
            assert status_values
            assert status_values[0]["pending_batches"] >= 0
        finally:
            proxy.allow_query.set()
            status.join(1.0)
            closer.join(1.0)
            engine.shutdown()


def test_foreground_preflight_drops_ready_batch_after_exact_profile_rebind(
    tmp_path, monkeypatch
):
    """A ready batch from a retired profile cannot truncate the new context."""
    for iteration in range(20):
        home_a = tmp_path / f"preflight-rebind-a-{iteration}"
        home_b = tmp_path / f"preflight-rebind-b-{iteration}"
        config = LCMConfig(
            database_path="",
            fresh_tail_count=2,
            leaf_chunk_tokens=20,
            context_threshold=0.10,
            async_background_compaction_enabled=True,
        )
        engine = LCMEngine(config=config, hermes_home=str(home_a))
        session_id = "preflight-rebind-session"
        conversation_id = "preflight-rebind-conversation"
        messages = _messages(prefix=f"preflight-rebind-{iteration}")
        engine.on_session_start(
            session_id,
            conversation_id=conversation_id,
            hermes_home=str(home_a),
            context_length=1_000,
        )
        engine.ingest(messages)
        batch = engine.prepare_background_compaction_once(messages)
        assert batch is not None and batch.state == "ready"
        old_manager = engine._async_compaction
        assert old_manager is not None

        preflight_started = Event()
        allow_preflight = Event()
        old_manager_used = Event()
        original_get_store_id_map = engine._get_store_id_map_for_messages
        original_promote_next = old_manager.promote_next

        def blocked_store_id_map(current_messages):
            preflight_started.set()
            assert allow_preflight.wait(3.0)
            return original_get_store_id_map(current_messages)

        def unexpected_promote_next(current_messages):
            old_manager_used.set()
            return original_promote_next(current_messages)

        monkeypatch.setattr(engine, "_get_store_id_map_for_messages", blocked_store_id_map)
        monkeypatch.setattr(old_manager, "promote_next", unexpected_promote_next)
        engine._last_compression_status = "pending"
        results = []
        errors = []

        def compress():
            try:
                results.append(
                    engine.compress(
                        messages,
                        current_tokens=engine.threshold_tokens + 1,
                    )
                )
            except BaseException as exc:  # pragma: no cover - red-path evidence
                errors.append(exc)

        foreground = Thread(target=compress, name=f"lcm-test-preflight-{iteration}")
        foreground.start()
        assert preflight_started.wait(2.0)

        rebound = Event()

        def rebind():
            engine.on_session_start(
                session_id,
                conversation_id=conversation_id,
                hermes_home=str(home_b),
                context_length=1_000,
            )
            rebound.set()

        rebind_thread = Thread(target=rebind, name=f"lcm-test-preflight-rebind-{iteration}")
        rebind_thread.start()
        try:
            assert rebound.wait(2.0)
            allow_preflight.set()
            foreground.join(3.0)
            rebind_thread.join(3.0)
            assert not foreground.is_alive()
            assert not rebind_thread.is_alive()
            assert not errors
            assert results == [messages]
            assert not old_manager_used.is_set()
            assert engine._dag.get_session_node_count(session_id) == 0
        finally:
            allow_preflight.set()
            foreground.join(3.0)
            rebind_thread.join(3.0)
            engine.shutdown()


def test_public_promotion_close_race_returns_non_promoted_without_assertion(
    tmp_path, monkeypatch
):
    """Closing a manager cannot invalidate a public promotion snapshot."""
    for iteration in range(200):
        config = LCMConfig(
            database_path=str(tmp_path / f"promote-close-{iteration}.db"),
            async_background_compaction_enabled=True,
        )
        engine = LCMEngine(config=config)
        manager = engine._async_compaction
        assert manager is not None
        capture_started = Event()
        allow_capture = Event()
        original_capture = manager.capture_snapshot

        def blocked_capture(*args, **kwargs):
            capture_started.set()
            assert manager._operation_inflight == 1
            assert allow_capture.wait(2.0)
            return original_capture(*args, **kwargs)

        monkeypatch.setattr(manager, "capture_snapshot", blocked_capture)
        results = []
        errors = []

        def promote():
            try:
                results.append(manager.promote("missing-batch", []))
            except BaseException as exc:  # pragma: no cover - red-path evidence
                errors.append(exc)

        promoter = Thread(target=promote, name=f"lcm-test-promote-{iteration}")
        closer = Thread(target=manager.close, name=f"lcm-test-promote-close-{iteration}")
        try:
            promoter.start()
            assert capture_started.wait(1.0)
            closer.start()
            closer.join(1.0)
            assert not closer.is_alive()
            allow_capture.set()
            promoter.join(2.0)
            assert not promoter.is_alive()
            assert not errors
            assert results and results[0].promoted is False
        finally:
            allow_capture.set()
            promoter.join(2.0)
            closer.join(2.0)
            engine.shutdown()


def test_status_and_tool_status_rebind_race_keeps_one_coherent_resource(
    tmp_path, monkeypatch
):
    """Direct and tool status reads never cross a detached profile bundle."""
    for iteration in range(200):
        home_a = tmp_path / f"status-rebind-a-{iteration}"
        home_b = tmp_path / f"status-rebind-b-{iteration}"
        config = LCMConfig(database_path="", async_background_compaction_enabled=True)
        engine = LCMEngine(config=config, hermes_home=str(home_a))
        session_id = "status-rebind-session"
        conversation_id = "status-rebind-conversation"
        engine.on_session_start(
            session_id,
            conversation_id=conversation_id,
            hermes_home=str(home_a),
            context_length=1_000,
        )
        old_store = engine._store
        assert old_store is not None
        read_started = Event()
        allow_read = Event()
        errors = []
        original_read = old_store.read_compaction_telemetry
        original_count = old_store.get_session_count

        def blocked_read(*args, **kwargs):
            read_started.set()
            assert allow_read.wait(2.0)
            return original_read(*args, **kwargs)

        def blocked_count(*args, **kwargs):
            read_started.set()
            assert allow_read.wait(2.0)
            return original_count(*args, **kwargs)

        monkeypatch.setattr(old_store, "read_compaction_telemetry", blocked_read)
        monkeypatch.setattr(old_store, "get_session_count", blocked_count)
        values = []
        use_tool = iteration % 2 == 1

        def read_status():
            try:
                values.append(
                    json.loads(engine.handle_tool_call("lcm_status", {}))
                    if use_tool
                    else engine.get_status()
                )
            except BaseException as exc:  # pragma: no cover - red-path evidence
                errors.append(exc)

        status_thread = Thread(target=read_status, name=f"lcm-test-status-rebind-{iteration}")
        status_thread.start()
        assert read_started.wait(1.0)
        rebind_thread = Thread(
            target=lambda: engine.on_session_start(
                session_id,
                conversation_id=conversation_id,
                hermes_home=str(home_b),
                context_length=1_000,
            ),
            name=f"lcm-test-status-rebind-switch-{iteration}",
        )
        rebind_thread.start()
        try:
            allow_read.set()
            status_thread.join(2.0)
            rebind_thread.join(2.0)
            assert not status_thread.is_alive()
            assert not rebind_thread.is_alive()
            assert not errors
            assert values
            if use_tool:
                assert values[0].get("error") is None
            else:
                assert values[0].get("engine") == "lcm"
        finally:
            allow_read.set()
            status_thread.join(2.0)
            rebind_thread.join(2.0)
            engine.shutdown()
