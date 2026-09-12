"""Tests for the LCMEngine.select_context() hook.

``select_context()`` is the *selection* verb — distinct from compression:
  - ``compress()``      : context is too long → shrink it.
  - ``select_context()``: DAG has compacted history → use that instead.

The hook should:
1. Return ``None`` when no DAG summary nodes exist (cache-preserving no-op).
2. Return an assembled context (DAG summaries + fresh tail) when summaries exist.
3. Respect ``budget_tokens`` by capping the assembled context.
4. Not mutate the original ``request_messages``.
5. Return ``None`` when the assembled context is structurally identical to the
   original request (cache-preserving no-op).
"""

import copy

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG, SummaryNode
from hermes_lcm.engine import LCMEngine


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.fresh_tail_count = 4
    config.leaf_chunk_tokens = 100
    config.database_path = str(tmp_path / "lcm_select_context_test.db")
    e = LCMEngine(config=config)
    e._session_id = "test-session"
    e.context_length = 128000
    e.threshold_tokens = int(128000 * config.context_threshold)
    try:
        yield e
    finally:
        e.shutdown()


def _make_messages(n: int, start: int = 0) -> list[dict]:
    """Create a simple alternating user/assistant message list."""
    msgs = []
    for i in range(start, start + n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"Message {i}"})
    return msgs


def _make_request(system: bool = True, n_messages: int = 10) -> list[dict]:
    """Build a request-style message list (optional system + history)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    msgs.extend(_make_messages(n_messages))
    return msgs


class TestSelectContextNoOp:
    """When no DAG summaries exist, select_context returns None (no replacement)."""

    def test_returns_none_no_dag_nodes(self, engine):
        request = _make_request()
        result = engine.select_context(request, budget_tokens=128000)
        assert result is None

    def test_returns_none_empty_session(self, engine):
        request = _make_request(n_messages=0)
        result = engine.select_context(request, budget_tokens=128000)
        assert result is None

    def test_does_not_mutate_request(self, engine):
        request = _make_request()
        original = copy.deepcopy(request)
        engine.select_context(request, budget_tokens=128000)
        assert request == original


class TestSelectContextWithSummaries:
    """When DAG summaries exist, select_context assembles and returns them."""

    @staticmethod
    def _seed_summary_node(engine, summary_text: str, depth: int = 1):
        """Insert a summary node directly into the DAG for testing."""
        engine._dag.add_node(
            SummaryNode(
                session_id=engine._session_id,
                depth=depth,
                summary=summary_text,
                source_ids=[],
                expand_hint="test",
            )
        )

    def test_returns_assembled_context_with_summaries(self, engine):
        self._seed_summary_node(engine, "Earlier conversation about testing.")
        request = _make_request()
        result = engine.select_context(request, budget_tokens=128000)
        assert result is not None
        # Result should contain the summary somewhere in the content
        all_content = " ".join(
            m.get("content", "") for m in result if isinstance(m.get("content"), str)
        )
        assert "testing" in all_content

    def test_preserves_system_message(self, engine):
        self._seed_summary_node(engine, "Summary content.")
        request = _make_request(system=True)
        result = engine.select_context(request, budget_tokens=128000)
        assert result is not None
        assert result[0]["role"] == "system"
        assert "helpful assistant" in result[0]["content"]

    def test_preserves_fresh_tail(self, engine):
        self._seed_summary_node(engine, "Summary of old messages.")
        request = _make_request(system=True, n_messages=8)
        result = engine.select_context(request, budget_tokens=128000)
        assert result is not None
        # The fresh tail should include recent messages
        all_content = [m.get("content", "") for m in result]
        # At least the last message should be preserved
        assert any("Message 7" in c for c in all_content if isinstance(c, str))

    def test_assembly_contains_summary_content(self, engine):
        """When a DAG summary exists, select_context returns a list that
        includes the summary content alongside the fresh tail."""
        self._seed_summary_node(engine, "A" * 5000)  # Large summary
        request = _make_request(system=True, n_messages=20)
        result = engine.select_context(request, budget_tokens=128000)
        assert result is not None
        # The summary content should be present in the assembled context
        all_content = " ".join(
            m.get("content", "") for m in result if isinstance(m.get("content"), str)
        )
        assert "AAAA" in all_content

    def test_no_system_message(self, engine):
        self._seed_summary_node(engine, "Summary without system prompt.")
        request = _make_request(system=False, n_messages=6)
        result = engine.select_context(request, budget_tokens=128000)
        assert result is not None
        # Should not start with a system message
        if result:
            assert result[0]["role"] != "system"


class TestSelectContextCachePreservation:
    """select_context should return None when the result would be identical."""

    def test_returns_none_when_assembly_matches_request(self, engine):
        # No summaries → no replacement → None
        request = _make_request(system=True, n_messages=5)
        result = engine.select_context(request, budget_tokens=128000)
        assert result is None

    def test_does_not_mutate_request_with_summaries(self, engine):
        TestSelectContextWithSummaries._seed_summary_node(
            engine, "Test summary."
        )
        request = _make_request(system=True, n_messages=6)
        original = copy.deepcopy(request)
        engine.select_context(request, budget_tokens=128000)
        assert request == original


class TestSelectContextSignature:
    """Verify the method signature matches the ContextEngine ABC."""

    def test_accepts_keyword_args(self, engine):
        request = _make_request()
        # Should not raise — all kwargs are optional
        result = engine.select_context(
            request,
            conversation_messages=request,
            incoming_message={"role": "user", "content": "hi"},
            budget_tokens=128000,
        )
        # No DAG nodes → None
        assert result is None

    def test_budget_tokens_zero_falls_back_to_assembly_cap(self, engine):
        TestSelectContextWithSummaries._seed_summary_node(
            engine, "Summary with zero budget."
        )
        request = _make_request(system=True, n_messages=4)
        # budget_tokens=0 means no assembly cap override
        result = engine.select_context(request, budget_tokens=0)
        assert result is not None
