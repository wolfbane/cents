"""Tests for LLM-enhanced sentiment agent functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cents.agents import SentimentAgent
from cents.agents.sentiment import clear_sentiment_cache, _article_score_cache
from cents.models import Thesis, EvidenceType


class MockAnthropicResponse:
    """Mock response from anthropic API."""

    def __init__(self, text: str):
        self.content = [MagicMock(text=text)]


class TestFallbackToKeywordScoring:
    """Test fallback to keyword-based scoring when no API key."""

    @patch("cents.agents.sentiment.get_settings")
    def test_no_anthropic_key_uses_keyword_scoring(self, mock_settings):
        """Falls back to keyword scoring when ANTHROPIC_API_KEY not configured."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.default_api_timeout = 10

        agent = SentimentAgent()

        # Should return None when no API key
        client = agent._get_anthropic_client()
        assert client is None

    @patch("cents.agents.sentiment.urlopen")
    @patch("cents.agents.sentiment.get_settings")
    def test_keyword_scoring_without_anthropic(self, mock_settings, mock_urlopen):
        """Uses keyword scoring and reports keyword method when no Anthropic key."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = None
        mock_settings.return_value.default_api_timeout = 10

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "articles": [
                {"title": "Stock surges on strong earnings", "source": {"name": "News"}},
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        agent = SentimentAgent()
        result = agent.research("TEST")

        # Should use keyword scoring
        assert result.evidence[0].metadata.get("scoring_method") == "keyword"
        assert "LLM-enhanced" not in result.summary


class TestLLMScoring:
    """Test LLM-based article scoring."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_sentiment_cache()

    @patch("cents.agents.sentiment.urlopen")
    @patch("cents.agents.sentiment.get_settings")
    def test_llm_scoring_bullish_article(self, mock_settings, mock_urlopen):
        """LLM scores bullish article with high confidence."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        # Mock news API response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "articles": [
                {
                    "title": "Company beats earnings, raises guidance",
                    "description": "Strong quarterly results",
                    "source": {"name": "News"},
                    "url": "https://example.com/article1",
                },
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # Create mock anthropic client
        mock_client = MagicMock()

        # Mock filter response (return index 0)
        filter_response = MockAnthropicResponse("0")
        # Mock scoring response (bullish)
        score_response = MockAnthropicResponse('{"score": 0.8, "reasoning": "Strong earnings beat"}')

        mock_client.messages.create.side_effect = [filter_response, score_response]

        agent = SentimentAgent(anthropic_client=mock_client)
        result = agent.research("TEST")

        # Should use LLM scoring
        assert result.evidence[0].metadata.get("scoring_method") == "llm"
        assert result.evidence[0].metadata.get("llm_score") == 0.8
        assert result.evidence[0].confidence >= 0.7
        assert "LLM-enhanced" in result.summary

    @patch("cents.agents.sentiment.urlopen")
    @patch("cents.agents.sentiment.get_settings")
    def test_llm_scoring_bearish_article(self, mock_settings, mock_urlopen):
        """LLM scores bearish article correctly."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "articles": [
                {
                    "title": "Company under investigation",
                    "description": "SEC probe announced",
                    "source": {"name": "News"},
                    "url": "https://example.com/article2",
                },
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        mock_client = MagicMock()
        filter_response = MockAnthropicResponse("0")
        score_response = MockAnthropicResponse('{"score": -0.7, "reasoning": "Regulatory risk"}')
        mock_client.messages.create.side_effect = [filter_response, score_response]

        agent = SentimentAgent(anthropic_client=mock_client)
        result = agent.research("TEST")

        assert result.evidence[0].metadata.get("llm_score") == -0.7
        assert result.evidence[0].type == EvidenceType.CONTRADICTING

    @patch("cents.agents.sentiment.urlopen")
    @patch("cents.agents.sentiment.get_settings")
    def test_llm_scoring_with_thesis_context(self, mock_settings, mock_urlopen):
        """LLM scoring uses thesis hypothesis for context."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "articles": [
                {
                    "title": "AI spending increases",
                    "description": "Cloud providers boost AI investments",
                    "source": {"name": "News"},
                    "url": "https://example.com/article3",
                },
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        mock_client = MagicMock()
        filter_response = MockAnthropicResponse("0")
        score_response = MockAnthropicResponse('{"score": 0.9, "reasoning": "Supports AI growth thesis"}')
        mock_client.messages.create.side_effect = [filter_response, score_response]

        thesis = Thesis(title="AI Growth", hypothesis="NVDA will benefit from AI infrastructure spending")

        agent = SentimentAgent(anthropic_client=mock_client)
        result = agent.research("NVDA", thesis)

        # Verify thesis hypothesis was used in prompt
        calls = mock_client.messages.create.call_args_list
        assert len(calls) == 2
        # Check scoring call includes thesis
        scoring_call = calls[1]
        prompt = scoring_call.kwargs["messages"][0]["content"]
        assert "AI infrastructure spending" in prompt


class TestCaching:
    """Test article score caching behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_sentiment_cache()

    @patch("cents.agents.sentiment.get_settings")
    def test_cache_stores_results(self, mock_settings):
        """Cache stores LLM scoring results."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        score_response = MockAnthropicResponse('{"score": 0.5, "reasoning": "Neutral"}')
        mock_client.messages.create.return_value = score_response

        agent = SentimentAgent(anthropic_client=mock_client)

        article = {
            "title": "Test article",
            "description": "Test description",
            "url": "https://example.com/cached",
        }

        # First call should hit LLM
        agent._score_with_llm(article, "TEST", None)
        assert mock_client.messages.create.call_count == 1

        # Check cache was populated
        assert "https://example.com/cached" in _article_score_cache

    @patch("cents.agents.sentiment.get_settings")
    def test_cache_prevents_duplicate_llm_calls(self, mock_settings):
        """Cached results prevent duplicate LLM calls."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        score_response = MockAnthropicResponse('{"score": 0.5, "reasoning": "Neutral"}')
        mock_client.messages.create.return_value = score_response

        agent = SentimentAgent(anthropic_client=mock_client)

        article = {
            "title": "Test article",
            "description": "Test description",
            "url": "https://example.com/dedup",
        }

        # First call
        agent._score_with_llm(article, "TEST", None)
        # Second call with same URL
        agent._score_with_llm(article, "TEST", None)

        # Should only call LLM once
        assert mock_client.messages.create.call_count == 1

    def test_clear_cache(self):
        """clear_sentiment_cache() empties the cache."""
        _article_score_cache["test_url"] = {"score": 0.5}
        assert len(_article_score_cache) == 1

        clear_sentiment_cache()
        assert len(_article_score_cache) == 0


class TestArticleFiltering:
    """Test LLM-based article filtering."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_sentiment_cache()

    @patch("cents.agents.sentiment.get_settings")
    def test_filter_returns_relevant_indices(self, mock_settings):
        """Filter correctly parses LLM response for relevant indices."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        # LLM returns indices 1 and 3 as relevant
        filter_response = MockAnthropicResponse("1\n3")
        mock_client.messages.create.return_value = filter_response

        agent = SentimentAgent(anthropic_client=mock_client)

        articles = [
            {"title": "PyPI release v1.2.3", "description": "Package update"},
            {"title": "NVDA earnings beat", "description": "Strong results"},
            {"title": "Job posting: Engineer", "description": "We're hiring"},
            {"title": "NVDA AI momentum", "description": "Continued growth"},
        ]

        filtered = agent._filter_relevant_articles(articles, "NVDA", None)

        # Should return articles at indices 1 and 3
        assert len(filtered) == 2
        assert filtered[0]["title"] == "NVDA earnings beat"
        assert filtered[1]["title"] == "NVDA AI momentum"

    @patch("cents.agents.sentiment.get_settings")
    def test_filter_fallback_on_empty_response(self, mock_settings):
        """Filter falls back to first 5 articles when LLM returns nothing parseable."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        # LLM returns unparseable response
        filter_response = MockAnthropicResponse("I cannot determine relevant articles.")
        mock_client.messages.create.return_value = filter_response

        agent = SentimentAgent(anthropic_client=mock_client)

        articles = [{"title": f"Article {i}"} for i in range(10)]

        filtered = agent._filter_relevant_articles(articles, "TEST", None)

        # Should fall back to first 5
        assert len(filtered) == 5

    @patch("cents.agents.sentiment.get_settings")
    def test_filter_handles_exception(self, mock_settings):
        """Filter falls back to first 5 on exception."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        agent = SentimentAgent(anthropic_client=mock_client)

        articles = [{"title": f"Article {i}"} for i in range(10)]

        filtered = agent._filter_relevant_articles(articles, "TEST", None)

        # Should fall back to first 5
        assert len(filtered) == 5


class TestConfidence:
    """Test confidence scoring for LLM results."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_sentiment_cache()

    @patch("cents.agents.sentiment.get_settings")
    def test_high_score_high_confidence(self, mock_settings):
        """High magnitude scores yield higher confidence (0.7-0.9 range)."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        # Score of 1.0 should give confidence of 0.9
        score_response = MockAnthropicResponse('{"score": 1.0, "reasoning": "Very bullish"}')
        mock_client.messages.create.return_value = score_response

        agent = SentimentAgent(anthropic_client=mock_client)

        article = {"title": "Test", "url": "https://example.com/high"}
        _, _, confidence, _ = agent._score_with_llm(article, "TEST", None)

        assert confidence == pytest.approx(0.9, rel=0.01)

    @patch("cents.agents.sentiment.get_settings")
    def test_neutral_score_lower_confidence(self, mock_settings):
        """Neutral scores yield lower confidence (around 0.7)."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        # Score of 0.0 should give confidence of 0.7
        score_response = MockAnthropicResponse('{"score": 0.0, "reasoning": "Neutral"}')
        mock_client.messages.create.return_value = score_response

        agent = SentimentAgent(anthropic_client=mock_client)

        article = {"title": "Test", "url": "https://example.com/neutral"}
        _, _, confidence, _ = agent._score_with_llm(article, "TEST", None)

        assert confidence == pytest.approx(0.7, rel=0.01)


class TestLLMErrorHandling:
    """Test error handling for LLM operations."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_sentiment_cache()

    @patch("cents.agents.sentiment.get_settings")
    def test_llm_error_falls_back_to_keyword(self, mock_settings):
        """Falls back to keyword scoring on LLM error."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        agent = SentimentAgent(anthropic_client=mock_client)

        article = {
            "title": "Stock surges",
            "description": "Strong gains",
            "url": "https://example.com/error",
        }
        ev_type, score, confidence, metadata = agent._score_with_llm(article, "TEST", None)

        # Should fall back to keyword scoring
        assert metadata.get("scoring_method") == "keyword"
        assert confidence == 0.5  # Keyword confidence

    @patch("cents.agents.sentiment.get_settings")
    def test_malformed_json_falls_back(self, mock_settings):
        """Falls back to keyword scoring on malformed JSON response."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        # Return malformed JSON
        score_response = MockAnthropicResponse("This is not JSON at all")
        mock_client.messages.create.return_value = score_response

        agent = SentimentAgent(anthropic_client=mock_client)

        article = {
            "title": "Stock drops",
            "description": "Weak earnings",
            "url": "https://example.com/malformed",
        }
        _, _, _, metadata = agent._score_with_llm(article, "TEST", None)

        # Should fall back to keyword scoring
        assert metadata.get("scoring_method") == "keyword"
