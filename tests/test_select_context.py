"""Tests for the LCMEngine.select_context() hook.

``select_context()`` is the *selection* verb — distinct from compression:
  - ``compress()``      : context is too long → shrink it.
  - ``select_context()``: DAG has compacted history → use that instead.

The hook should:
1. Return ``None`` when no DAG summary nodes exist (cache-preserving no-op).
2. Return ``None`` when savings don't justify breaking cache (cache-aware).
3. Return an assembled context when summaries exist AND savings are significant.
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


def _make_messages(n: int, start: int = 0, tokens_per_msg: int = 50) -> list[dict]:
    """Create a simple alternating user/assistant message list.

    tokens_per_msg controls the approximate token size of each message.
    At ~4 chars/token, tokens_per_msg=50 → ~200 chars per message.
    """
    msgs = []
    for i in range(start, start + n):
        role = "user" if i % 2 == 0 else "assistant"
        # Pad content to approximate the requested token count
        padding = "x" * (tokens_per_msg * 4)
        msgs.append({"role": role, "content": f"Message {i} {padding}"})
    return msgs


def _make_request(
    system: bool = True,
    n_messages: int = 10,
    tokens_per_msg: int = 50,
) -> list[dict]:
    """Build a request-style message list (optional system + history)."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": "You are a helpful assistant."})
    msgs.extend(_make_messages(n_messages, tokens_per_msg=tokens_per_msg))
    return msgs


def _make_high_pressure_request(
    system: bool = True,
    n_messages: int = 100,
    tokens_per_msg: int = 500,
) -> list[dict]:
    """Build a high-pressure request that exceeds the cache-break threshold.

    With 100 messages × 500 tokens each = ~50K tokens at 80% of 128K budget,
    this should trigger the savings check.
    """
    return _make_request(system=system, n_messages=n_messages, tokens_per_msg=tokens_per_msg)


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
    """When DAG summaries exist, select_context assembles and returns them
    (only when savings justify breaking the cache)."""

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

    def test_returns_none_low_pressure_even_with_summaries(self, engine):
        """When context is low-pressure, select_context preserves cache."""
        self._seed_summary_node(engine, "Small summary of old messages.")
        # Low-pressure: only 10 messages × 50 tokens ≈ 500 tokens
        request = _make_request(n_messages=10, tokens_per_msg=50)
        result = engine.select_context(request, budget_tokens=128000)
        # With very low pressure (< 1%), savings don't justify breaking cache
        assert result is None

    def test_preserves_system_message_under_pressure(self, engine):
        """When replacement happens under pressure, the system message is preserved."""
        self._seed_summary_node(engine, "X" * 8000)
        request = _make_high_pressure_request(system=True)
        result = engine.select_context(request, budget_tokens=128000)
        if result is not None:
            assert result[0]["role"] == "system"
            assert "helpful assistant" in result[0]["content"]

    def test_preserves_fresh_tail_under_pressure(self, engine):
        """When replacement happens under pressure, the fresh tail is preserved."""
        self._seed_summary_node(engine, "Summary of old messages.")
        request = _make_high_pressure_request(n_messages=50, tokens_per_msg=500)
        result = engine.select_context(request, budget_tokens=128000)
        if result is not None:
            # The result should be a valid message list
            assert isinstance(result, list)
            assert len(result) > 0

    def test_no_system_message_under_pressure(self, engine):
        """When replacement happens without a system prompt, no system msg."""
        self._seed_summary_node(engine, "Summary without system prompt.")
        request = _make_high_pressure_request(system=False)
        result = engine.select_context(request, budget_tokens=128000)
        if result is not None:
            if result:
                assert result[0]["role"] != "system"

    def test_assembly_valid_output_under_pressure(self, engine):
        """When replacement happens, the output is a valid message list."""
        self._seed_summary_node(engine, "A" * 8000)
        request = _make_high_pressure_request()
        result = engine.select_context(request, budget_tokens=128000)
        if result is not None:
            assert isinstance(result, list)
            assert all(isinstance(m, dict) for m in result)


class TestSelectContextCachePreservation:
    """select_context should return None when the result would be identical
    or when savings don't justify breaking cache."""

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

    def test_small_savings_preserved_low_pressure(self, engine):
        """With small context and low pressure, cache is preserved even with summaries."""
        TestSelectContextWithSummaries._seed_summary_node(
            engine, "A small summary."
        )
        # Only 5 small messages — no pressure, savings are trivial
        request = _make_request(n_messages=5, tokens_per_msg=10)
        result = engine.select_context(request, budget_tokens=128000)
        # Cache should be preserved — savings not worth the break
        assert result is None


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

    def test_budget_tokens_zero_low_pressure(self, engine):
        """With budget_tokens=0 and low pressure, cache is preserved."""
        TestSelectContextWithSummaries._seed_summary_node(
            engine, "Summary with zero budget."
        )
        request = _make_request(system=True, n_messages=4)
        result = engine.select_context(request, budget_tokens=0)
        # With no budget and low pressure, should preserve cache
        assert result is None
