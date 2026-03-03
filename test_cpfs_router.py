#!/usr/bin/env python3
"""
Unit tests for cpfs_router.py
Tests all identified bugs and their fixes.
"""

import asyncio
import json
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))

# We need to mock __file__ for load_env since it runs at import time
# and looks for .env relative to the script
import cpfs_router as router


class TestUnusedImports(unittest.TestCase):
    """Bug #7: Verify unused imports (math, field, Any) are removed."""

    def test_no_math_import(self):
        import inspect
        source = inspect.getsource(router)
        # math should not be imported at the top level
        self.assertNotIn("import math", source.split("\n")[0:10].__repr__())

    def test_re_imported(self):
        """re module should be available for word-boundary matching."""
        self.assertTrue(hasattr(router, 're'))


class TestEndomorphism(unittest.TestCase):
    """Bug #2: error_count should reset on success."""

    def test_error_count_resets_on_success(self):
        agent = router.MarkovLLMAgent("test", "Test Agent", ["reasoning"])
        # Simulate 2 errors
        agent.error_count = 2
        agent.endomorphism(False, 1000)
        self.assertEqual(agent.error_count, 2)  # error_count incremented in call(), not endomorphism
        self.assertTrue(agent.available)  # still < 3

        # Now simulate success — error_count should reset
        agent.error_count = 5  # pretend we had 5 errors
        agent.endomorphism(True, 500)
        self.assertEqual(agent.error_count, 0)
        self.assertTrue(agent.available)

    def test_agent_unavailable_after_3_consecutive_errors(self):
        agent = router.MarkovLLMAgent("test", "Test Agent", ["reasoning"])
        agent.error_count = 3
        agent.endomorphism(False, 5000)
        self.assertFalse(agent.available)

    def test_agent_recovers_after_success(self):
        agent = router.MarkovLLMAgent("test", "Test Agent", ["reasoning"])
        agent.error_count = 10
        agent.available = False
        agent.endomorphism(True, 500)
        self.assertTrue(agent.available)
        self.assertEqual(agent.error_count, 0)


class TestDetectTaskType(unittest.TestCase):
    """Bug #1 & #12: keyword conflicts and substring matching."""

    def test_napysh_goes_to_writing_not_code(self):
        """'напиш' was in both code and writing — should now be only in writing."""
        result = router.detect_task_type("напиши мені текст")
        self.assertEqual(result, "writing")

    def test_code_detection(self):
        result = router.detect_task_type("напиши код на python")
        # "код" and "python" match code; "напиши" matches writing
        # code should win with 2 keywords vs 1
        self.assertEqual(result, "code")

    def test_no_false_positive_local(self):
        """Bug #12: 'local' in 'localhost' should NOT trigger privacy."""
        # "localhost" contains "local" as substring but not as word boundary
        result = router.detect_task_type("connect to localhost:3000")
        self.assertNotEqual(result, "privacy")

    def test_privacy_detection(self):
        """Actual privacy keywords should still work."""
        result = router.detect_task_type("зроби це приватно і локально")
        self.assertEqual(result, "privacy")

    def test_math_detection(self):
        result = router.detect_task_type("calculate the integral of x^2")
        self.assertEqual(result, "math")

    def test_default_reasoning(self):
        """No keywords matched → reasoning."""
        result = router.detect_task_type("як справи?")
        self.assertEqual(result, "reasoning")

    def test_multimodal_detection(self):
        result = router.detect_task_type("опиши це зображення")
        self.assertEqual(result, "multimodal")

    def test_substring_no_false_positive_число(self):
        """'число' inside 'зменшити число помилок' — this is tricky since
        'число' is a standalone word here, so it SHOULD match math."""
        result = router.detect_task_type("зменшити число помилок у коді")
        # "число" matches math, "код" matches code — could be either
        # The important thing is it doesn't crash
        self.assertIn(result, ["math", "code"])


class TestBayesianRouter(unittest.TestCase):
    """Test router routing logic."""

    def test_no_available_agents(self):
        agents = [router.MarkovLLMAgent("a", "A", ["code"])]
        agents[0].available = False
        r = router.BayesianRouter(agents)
        self.assertIsNone(r.route("hello"))

    def test_force_model(self):
        a1 = router.MarkovLLMAgent("claude", "Claude", ["code"])
        a1.available = True
        a1.stability_score = 0.8
        a2 = router.MarkovLLMAgent("gpt4", "GPT-4", ["code"])
        a2.available = True
        a2.stability_score = 0.9
        r = router.BayesianRouter([a1, a2])
        result = r.route("test", force="claude")
        self.assertEqual(result.agent_id, "claude")

    def test_force_unavailable_falls_back(self):
        a1 = router.MarkovLLMAgent("claude", "Claude", ["code"])
        a1.available = False
        a2 = router.MarkovLLMAgent("gpt4", "GPT-4", ["code"])
        a2.available = True
        a2.stability_score = 0.7
        r = router.BayesianRouter([a1, a2])
        result = r.route("test", force="claude")
        self.assertEqual(result.agent_id, "gpt4")

    def test_attractor_no_available(self):
        agents = [router.MarkovLLMAgent("a", "A", ["code"])]
        agents[0].available = False
        r = router.BayesianRouter(agents)
        att = r.attractor()
        self.assertEqual(att["stability"], 0.0)

    def test_attractor_with_agents(self):
        a1 = router.MarkovLLMAgent("a", "A", ["code"])
        a1.available = True
        a1.stability_score = 0.8
        r = router.BayesianRouter([a1])
        att = r.attractor()
        self.assertIn("mean", att)
        self.assertIn("stability", att)
        self.assertEqual(att["available_agents"], 1)


class TestStateDistribution(unittest.TestCase):
    def test_liquidity_unavailable(self):
        sd = router.StateDistribution("test", 0.5, 0.1, 0.8, ["code"], available=False)
        self.assertEqual(sd.liquidity(), 0.0)

    def test_liquidity_available(self):
        sd = router.StateDistribution("test", 0.5, 0.1, 0.8, ["code"],
                                       latency_ms=500, cost_per_1k=1.0, available=True)
        li = sd.liquidity()
        self.assertGreater(li, 0.0)
        # L_I = (0.8 * 0.9) / (0.5 * 1.0) = 1.44
        self.assertAlmostEqual(li, 1.44, places=2)

    def test_liquidity_zero_cost(self):
        """Ollama has cost_per_1k=0.0 — should use max(0.1, 0.0) = 0.1."""
        sd = router.StateDistribution("test", 0.5, 0.1, 0.8, ["code"],
                                       latency_ms=500, cost_per_1k=0.0, available=True)
        li = sd.liquidity()
        self.assertGreater(li, 0.0)


class TestClaudeSystemMessage(unittest.TestCase):
    """Bug #5: Claude should handle system messages properly."""

    def test_system_message_extraction(self):
        """Verify Claude agent separates system messages from user/assistant."""
        agent = router.ClaudeAgent()
        agent.api_key = "test-key"

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        # We can't call the API, but we can verify the logic by
        # checking the code path. Let's mock _post_json.
        mock_response = {
            "content": [{"text": "Hi there!"}],
            "usage": {"output_tokens": 10}
        }

        with patch.object(router, '_post_json', return_value=mock_response) as mock_post:
            result = asyncio.run(agent.call(messages))
            # Verify _post_json was called with correct body
            call_args = mock_post.call_args
            body = call_args[0][2]  # third positional arg is body
            # Should have "system" field
            self.assertEqual(body["system"], "You are helpful.")
            # Messages should NOT contain the system message
            for m in body["messages"]:
                self.assertNotEqual(m["role"], "system")
            self.assertEqual(len(body["messages"]), 1)
            self.assertEqual(body["messages"][0]["role"], "user")

    def test_no_system_message(self):
        """When no system message, body should not have 'system' key."""
        agent = router.ClaudeAgent()
        agent.api_key = "test-key"

        messages = [{"role": "user", "content": "Hello"}]
        mock_response = {
            "content": [{"text": "Hi!"}],
            "usage": {"output_tokens": 5}
        }

        with patch.object(router, '_post_json', return_value=mock_response) as mock_post:
            asyncio.run(agent.call(messages))
            body = mock_post.call_args[0][2]
            self.assertNotIn("system", body)


class TestGeminiApiKeyNotInUrl(unittest.TestCase):
    """Bug #4: Gemini API key should be in header, not URL."""

    def test_api_key_in_header(self):
        agent = router.GeminiAgent()
        agent.api_key = "test-gemini-key"

        messages = [{"role": "user", "content": "Hello"}]
        mock_response = {
            "candidates": [{"content": {"parts": [{"text": "Hi!"}]}}],
            "usageMetadata": {"candidatesTokenCount": 5}
        }

        with patch.object(router, '_post_json', return_value=mock_response) as mock_post:
            asyncio.run(agent.call(messages))
            call_args = mock_post.call_args
            url = call_args[0][0]
            headers = call_args[0][1]
            # API key should NOT be in URL
            self.assertNotIn("test-gemini-key", url)
            self.assertNotIn("key=", url)
            # API key should be in headers
            self.assertEqual(headers["x-goog-api-key"], "test-gemini-key")

    def test_gemini_skips_system_messages(self):
        """System messages should be filtered out for Gemini."""
        agent = router.GeminiAgent()
        agent.api_key = "test-key"

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        mock_response = {
            "candidates": [{"content": {"parts": [{"text": "Hi!"}]}}],
            "usageMetadata": {"candidatesTokenCount": 5}
        }

        with patch.object(router, '_post_json', return_value=mock_response) as mock_post:
            asyncio.run(agent.call(messages))
            body = mock_post.call_args[0][2]
            # Only 1 content (user), system should be filtered
            self.assertEqual(len(body["contents"]), 1)
            self.assertEqual(body["contents"][0]["role"], "user")


class TestAsyncioGetRunningLoop(unittest.TestCase):
    """Bug #6: get_event_loop() → get_running_loop()."""

    def test_claude_uses_running_loop(self):
        import inspect
        source = inspect.getsource(router.ClaudeAgent.call)
        self.assertIn("get_running_loop", source)
        self.assertNotIn("get_event_loop", source)

    def test_gpt_uses_running_loop(self):
        import inspect
        source = inspect.getsource(router.GPTAgent.call)
        self.assertIn("get_running_loop", source)
        self.assertNotIn("get_event_loop", source)

    def test_gemini_uses_running_loop(self):
        import inspect
        source = inspect.getsource(router.GeminiAgent.call)
        self.assertIn("get_running_loop", source)
        self.assertNotIn("get_event_loop", source)

    def test_ollama_uses_running_loop(self):
        import inspect
        source_call = inspect.getsource(router.OllamaAgent.call)
        source_check = inspect.getsource(router.OllamaAgent.check)
        self.assertIn("get_running_loop", source_call)
        self.assertIn("get_running_loop", source_check)
        self.assertNotIn("get_event_loop", source_call)
        self.assertNotIn("get_event_loop", source_check)

    def test_nanoai_uses_running_loop(self):
        import inspect
        source = inspect.getsource(router.NanoAIAgent.call)
        self.assertIn("get_running_loop", source)
        self.assertNotIn("get_event_loop", source)


class TestFallbackHistoryManagement(unittest.TestCase):
    """Bug #3: Fallback should preserve history consistency."""

    def test_fallback_failure_preserves_user_message(self):
        """When both primary and fallback fail, user message should remain in history."""
        # We test the logic by checking the code contains the fix
        import inspect
        source = inspect.getsource(router.chat_loop)
        # After fallback exception, history should get the user message back
        self.assertIn("history.append({\"role\": \"user\", \"content\": prompt})", source)

    def test_no_fallback_preserves_user_message(self):
        """When primary fails and no fallback agent exists, user message should remain."""
        import inspect
        source = inspect.getsource(router.chat_loop)
        # There should be an else branch after 'if fallback:' that re-adds user message
        self.assertIn("Немає доступних fallback-агентів", source)


class TestMarkovLLMAgentStats(unittest.TestCase):
    def test_stats_keys(self):
        agent = router.MarkovLLMAgent("test", "Test", ["code"])
        s = agent.stats()
        expected_keys = {"name", "available", "stability", "latency_ms",
                         "calls", "errors", "tokens", "liquidity"}
        self.assertEqual(set(s.keys()), expected_keys)

    def test_endomorphism_stability_bounds(self):
        """Stability score should always be in [0, 1]."""
        agent = router.MarkovLLMAgent("test", "Test", ["code"])
        # Many successes
        for _ in range(100):
            agent.endomorphism(True, 100)
        self.assertLessEqual(agent.stability_score, 1.0)
        self.assertGreaterEqual(agent.stability_score, 0.0)

        # Many failures
        agent.error_count = 0  # reset for test
        for _ in range(100):
            agent.error_count = 0  # keep available
            agent.endomorphism(False, 5000)
        self.assertLessEqual(agent.stability_score, 1.0)
        self.assertGreaterEqual(agent.stability_score, 0.0)

    def test_variance_bounded_above(self):
        """Variance should never exceed 1.0 even after many failures."""
        agent = router.MarkovLLMAgent("test", "Test", ["code"])
        agent.variance = 0.9
        for _ in range(100):
            agent.error_count = 0  # keep available for testing
            agent.endomorphism(False, 5000)
        self.assertLessEqual(agent.variance, 1.0)
        self.assertGreaterEqual(agent.variance, 0.01)

    def test_mean_bounds(self):
        """Mean should always be in [0, 1]."""
        agent = router.MarkovLLMAgent("test", "Test", ["code"])
        for _ in range(1000):
            agent.endomorphism(True, 100)
        self.assertLessEqual(agent.mean, 1.0)
        self.assertGreaterEqual(agent.mean, 0.0)


class TestLoadEnv(unittest.TestCase):
    def test_load_env_no_file(self):
        """Should not crash when .env doesn't exist."""
        # This is already tested implicitly by importing the module
        # but let's be explicit
        with patch('pathlib.Path.exists', return_value=False):
            router.load_env()  # should not raise

    def test_load_env_strips_quotes(self):
        """Values wrapped in quotes should have quotes stripped."""
        import tempfile
        env_content = 'TEST_QUOTED_KEY="my-secret-value"\nTEST_SINGLE_QUOTED=\'another-value\'\n'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            f.flush()
            tmp_path = f.name
        try:
            # Remove from env if already set
            os.environ.pop("TEST_QUOTED_KEY", None)
            os.environ.pop("TEST_SINGLE_QUOTED", None)
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.read_text', return_value=env_content):
                router.load_env()
            self.assertEqual(os.environ.get("TEST_QUOTED_KEY"), "my-secret-value")
            self.assertEqual(os.environ.get("TEST_SINGLE_QUOTED"), "another-value")
        finally:
            os.unlink(tmp_path)
            os.environ.pop("TEST_QUOTED_KEY", None)
            os.environ.pop("TEST_SINGLE_QUOTED", None)


class TestRecheckCommand(unittest.TestCase):
    """Test /recheck command exists in help and chat_loop."""

    def test_recheck_in_help(self):
        import inspect
        source = inspect.getsource(router.print_help)
        self.assertIn("/recheck", source)

    def test_recheck_in_chat_loop(self):
        import inspect
        source = inspect.getsource(router.chat_loop)
        self.assertIn("/recheck", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
