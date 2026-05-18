"""Lookahead-leak audit for the sentiment agent (cents-ekd).

NewsAPI returns articles published at any time, including same-day pieces
that explicitly reference today's price action ("NVDA up 8% after ..."). If
the LLM scorer is materially moved by the price-mention itself rather than
the underlying news, the agent is implicitly forward-looking on intraday
horizons and contaminates outcome labels.

This file documents the methodology. The cheap, always-on path runs both
headlines through a *mocked* Anthropic and asserts only that the scoring
plumbing produced two numbers — it cannot prove leakage on its own.

The on-demand path (``--runlookahead``) hits the real Anthropic API and
asserts the LLM is not heavily moved by the price-mention. Failure of the
live assertion is the documented "leakage detected" signal.

Run the live audit with::

    pytest tests/test_lookahead_audit.py --runlookahead

The live test skips automatically when ``ANTHROPIC_API_KEY`` is not set.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from cents.agents import SentimentAgent
from cents.agents.sentiment import clear_sentiment_cache


# Two headlines that differ ONLY in whether they reference the day's price
# move. A well-behaved sentiment scorer should land within a small window
# of itself on both — the underlying news is identical.
HEADLINE_NEUTRAL = {
    "title": "NVDA announces new datacenter partnership",
    "description": (
        "NVIDIA Corporation announced a strategic datacenter partnership "
        "expected to expand its enterprise AI footprint."
    ),
    "url": "https://example.test/nvda-neutral",
}
HEADLINE_PRICE_LADEN = {
    "title": "NVDA stock surges 8% after datacenter partnership announcement",
    "description": (
        "NVIDIA Corporation announced a strategic datacenter partnership "
        "expected to expand its enterprise AI footprint. Shares jumped 8% "
        "in early trading on the news."
    ),
    "url": "https://example.test/nvda-price-laden",
}

# Score delta that, when exceeded by the LIVE LLM, indicates the price
# mention is moving the score more than the underlying news.
_LOOKAHEAD_DELTA_THRESHOLD = 0.3


class MockAnthropicResponse:
    """Minimal mock of an Anthropic messages.create response."""

    def __init__(self, text: str):
        self.content = [MagicMock(text=text)]


@pytest.mark.lookahead_audit
class TestLookaheadAuditMocked:
    """Always-on smoke version — exercises the scoring plumbing with mocked LLM.

    This cannot detect leakage. Its only job is to document the methodology
    in code form: the same agent, the same prompt path, two headlines that
    differ only in whether they mention today's price. The live variant
    (below) is the one that actually measures the contamination.
    """

    def setup_method(self):
        clear_sentiment_cache()

    @patch("cents.agents.sentiment.get_settings")
    def test_methodology_runs_both_headlines(self, mock_settings):
        mock_settings.return_value.news_api_key = "test_key"
        mock_settings.return_value.anthropic_api_key = "test_anthropic_key"
        mock_settings.return_value.default_api_timeout = 10

        mock_client = MagicMock()
        # Same score for both — the mocked LLM is by definition deterministic.
        mock_client.messages.create.return_value = MockAnthropicResponse(
            '{"score": 0.5, "reasoning": "datacenter partnership is positive"}'
        )
        agent = SentimentAgent(anthropic_client=mock_client)

        _, score_neutral, _, _, _ = agent._score_with_llm(
            HEADLINE_NEUTRAL, "NVDA", None
        )
        clear_sentiment_cache()
        _, score_price_laden, _, _, _ = agent._score_with_llm(
            HEADLINE_PRICE_LADEN, "NVDA", None
        )

        # Both should produce a numeric score in [-1, 1]. The mocked LLM
        # returns the same value for both — this only verifies the plumbing.
        assert -1.0 <= score_neutral <= 1.0
        assert -1.0 <= score_price_laden <= 1.0
        assert mock_client.messages.create.call_count == 2


@pytest.mark.lookahead_audit
class TestLookaheadAuditLive:
    """Live audit — hits the real Anthropic API. Opt-in via --runlookahead.

    Failure of ``assert abs(score_price_laden - score_neutral) < THRESHOLD``
    is the documented "leakage detected" signal. The remediation is the
    ``news_cutoff_time`` config knob (filtering articles by publishedAt
    before market open) — see ``scope.mdx`` and ``FactoryConfig``.
    """

    def setup_method(self):
        clear_sentiment_cache()

    def test_price_mention_does_not_dominate_score(self, request):
        if not request.config.getoption("--runlookahead", default=False):
            pytest.skip("live audit; pass --runlookahead to run")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        agent = SentimentAgent()
        _, score_neutral, _, _, _ = agent._score_with_llm(
            HEADLINE_NEUTRAL, "NVDA", None
        )
        clear_sentiment_cache()
        _, score_price_laden, _, _, _ = agent._score_with_llm(
            HEADLINE_PRICE_LADEN, "NVDA", None
        )

        delta = abs(score_price_laden - score_neutral)
        assert delta < _LOOKAHEAD_DELTA_THRESHOLD, (
            f"Lookahead leakage detected: price-laden headline scored "
            f"{score_price_laden:+.3f}, neutral scored {score_neutral:+.3f} "
            f"(|delta|={delta:.3f} >= {_LOOKAHEAD_DELTA_THRESHOLD}). "
            "The LLM is being moved by the price-mention itself rather than "
            "the underlying news — sentiment scores on same-day articles "
            "are implicitly forward-looking. Mitigation: set "
            "factory.toml's news_cutoff_time to filter NewsAPI articles by "
            "publishedAt before market open."
        )
