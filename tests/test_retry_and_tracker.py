import os
import tempfile
from unittest.mock import MagicMock
import pytest
from src.llm.client import LLM, Tracker


def test_tracker_initialization_and_accumulation():
    tr = Tracker()
    assert tr.llm_calls == 0
    assert tr.in_tokens == 0
    assert tr.out_tokens == 0
    assert tr.latency == 0.0
    assert tr.cache_hits == 0
    assert tr.retrievals == 0
    assert tr.probe_searches == 0
    assert tr.failures == 0

    tr.llm_calls += 2
    tr.in_tokens += 150
    tr.out_tokens += 50
    tr.latency += 0.85
    tr.cache_hits += 1
    tr.retrievals += 3
    tr.probe_searches += 4
    tr.failures += 1

    assert tr.llm_calls == 2
    assert tr.in_tokens == 150
    assert tr.cache_hits == 1
    assert tr.retrievals == 3
    assert tr.failures == 1


def test_cache_hit_and_stats():
    with tempfile.NamedTemporaryFile("w", suffix=".sqlite", delete=False) as f:
        tmp_db = f.name

    try:
        cfg = {"model": {"name": "test-model", "max_tokens": 100}}
        llm = LLM(cfg, use_cache=True, namespace="test_ns", cache_path=tmp_db)

        # Mock the external groq client
        mock_choice = MagicMock()
        mock_choice.message.content = "Test answer"
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage.prompt_tokens = 25
        mock_resp.usage.completion_tokens = 10

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = mock_resp
        llm._client = mock_groq

        tr1 = Tracker()
        res1 = llm.chat([{"role": "user", "content": "hello"}], tr1)
        assert res1 == "Test answer"
        assert tr1.llm_calls == 1
        assert tr1.cache_hits == 0
        assert tr1.in_tokens == 25
        assert tr1.out_tokens == 10
        assert mock_groq.chat.completions.create.call_count == 1

        # Second identical call should hit the SQLite cache
        tr2 = Tracker()
        res2 = llm.chat([{"role": "user", "content": "hello"}], tr2)
        assert res2 == "Test answer"
        assert tr2.llm_calls == 1
        assert tr2.cache_hits == 1
        assert tr2.in_tokens == 25
        assert tr2.out_tokens == 10
        # External API should NOT be called on cache hit
        assert mock_groq.chat.completions.create.call_count == 1
    finally:
        llm.close()
        if os.path.exists(tmp_db):
            os.remove(tmp_db)


def test_retry_exhaustion_raises_runtime_error():
    with tempfile.NamedTemporaryFile("w", suffix=".sqlite", delete=False) as f:
        tmp_db = f.name

    try:
        cfg = {"model": {"name": "test-model", "max_tokens": 100}}
        llm = LLM(cfg, use_cache=True, namespace="test_ns", cache_path=tmp_db)

        mock_groq = MagicMock()
        mock_groq.chat.completions.create.side_effect = ConnectionError("API connection failed")
        llm._client = mock_groq

        tr = Tracker()
        # Mock time.sleep to avoid waiting during test
        from unittest.mock import patch
        with patch("time.sleep", return_value=None):
            with pytest.raises(RuntimeError, match="LLM call failed after 6 attempts"):
                llm.chat([{"role": "user", "content": "fail"}], tr)

        assert tr.failures == 6
        assert mock_groq.chat.completions.create.call_count == 6
    finally:
        llm.close()
        if os.path.exists(tmp_db):
            os.remove(tmp_db)

