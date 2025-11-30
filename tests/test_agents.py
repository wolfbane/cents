"""Tests for research agents with mocked external dependencies."""

from unittest.mock import MagicMock, patch, PropertyMock
import json

import pytest
import pandas as pd

from cents.agents import (
    FundamentalsAgent,
    TechnicalAgent,
    MacroAgent,
    SentimentAgent,
    OrchestratorAgent,
    AgentResult,
)
from cents.models import Thesis, EvidenceType


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_create_agent_result(self):
        """AgentResult can be created with required fields."""
        result = AgentResult(evidence=[], conviction_delta=5.0, summary="Test")
        assert result.evidence == []
        assert result.conviction_delta == 5.0
        assert result.summary == "Test"


class TestFundamentalsAgent:
    """Tests for FundamentalsAgent with mocked yfinance."""

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_low_pe_bullish(self, mock_ticker_class):
        """Low P/E ratio generates bullish signal."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 12.0,
            "shortName": "Test Corp",
        }
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "Low P/E" in result.summary
        assert len(result.evidence) >= 1

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_high_pe_bearish(self, mock_ticker_class):
        """High P/E ratio generates bearish signal."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 50.0,
            "shortName": "Expensive Corp",
        }
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "High P/E" in result.summary

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_strong_growth(self, mock_ticker_class):
        """Strong revenue growth is bullish."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "revenueGrowth": 0.30,  # 30%
            "shortName": "Growth Corp",
        }
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "Strong revenue growth" in result.summary

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_negative_growth(self, mock_ticker_class):
        """Negative revenue growth is bearish."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "revenueGrowth": -0.10,  # -10%
            "shortName": "Declining Corp",
        }
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "Negative revenue growth" in result.summary

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_high_debt(self, mock_ticker_class):
        """High debt-to-equity is bearish."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "debtToEquity": 250.0,
            "shortName": "Leveraged Corp",
        }
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "High debt" in result.summary

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_analyst_buy(self, mock_ticker_class):
        """Buy recommendation is bullish."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "recommendationKey": "strong_buy",
            "shortName": "Hot Stock",
        }
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        evidence_content = [e.content for e in result.evidence]
        assert any("Strong Buy" in c for c in evidence_content)

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_with_thesis(self, mock_ticker_class):
        """Research uses thesis ID when provided."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"trailingPE": 15.0, "shortName": "Test"}
        mock_ticker_class.return_value = mock_ticker

        thesis = Thesis(title="Test thesis")
        agent = FundamentalsAgent()
        result = agent.research("TEST", thesis)

        for e in result.evidence:
            assert e.thesis_id == thesis.id

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_api_error(self, mock_ticker_class):
        """Handles API errors gracefully."""
        mock_ticker = MagicMock()
        type(mock_ticker).info = PropertyMock(side_effect=Exception("API Error"))
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "Failed to fetch" in result.summary

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_retries_on_transient_failure(self, mock_ticker_class):
        """Retries yfinance info fetch before failing."""
        mock_ticker = MagicMock()
        type(mock_ticker).info = PropertyMock(
            side_effect=[Exception("temporary"), {"shortName": "Retry Corp"}]
        )
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert "Retry Corp" in result.summary

    @patch("cents.agents.fundamentals.yf.Ticker")
    def test_research_no_signals(self, mock_ticker_class):
        """No significant signals when data is neutral."""
        mock_ticker = MagicMock()
        mock_ticker.info = {
            "trailingPE": 20.0,  # Neutral
            "shortName": "Average Corp",
        }
        mock_ticker_class.return_value = mock_ticker

        agent = FundamentalsAgent()
        result = agent.research("TEST")

        assert "No significant signals" in result.summary


class TestTechnicalAgent:
    """Tests for TechnicalAgent with mocked yfinance."""

    def _create_mock_history(self, prices, volumes=None):
        """Helper to create mock price history DataFrame."""
        n = len(prices)
        if volumes is None:
            volumes = [1000000] * n
        return pd.DataFrame({
            "Close": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Volume": volumes,
        })

    @patch("cents.agents.technical.yf.Ticker")
    def test_research_strong_momentum(self, mock_ticker_class):
        """Strong upward momentum is bullish."""
        mock_ticker = MagicMock()
        # Price went from 100 to 120 over 30 days (+20%)
        prices = [100 + (i * 0.67) for i in range(30)]
        mock_ticker.history.return_value = self._create_mock_history(prices)
        mock_ticker_class.return_value = mock_ticker

        agent = TechnicalAgent()
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "Strong momentum" in result.summary

    @patch("cents.agents.technical.yf.Ticker")
    def test_research_weak_momentum(self, mock_ticker_class):
        """Strong downward momentum is bearish."""
        mock_ticker = MagicMock()
        # Price went from 100 to 80 over 30 days (-20%)
        prices = [100 - (i * 0.67) for i in range(30)]
        mock_ticker.history.return_value = self._create_mock_history(prices)
        mock_ticker_class.return_value = mock_ticker

        agent = TechnicalAgent()
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "Weak momentum" in result.summary

    @patch("cents.agents.technical.yf.Ticker")
    def test_research_above_moving_averages(self, mock_ticker_class):
        """Price above MAs is bullish."""
        mock_ticker = MagicMock()
        # Steadily rising prices - current price above both MAs
        prices = [100 + i for i in range(100)]  # 100 days of uptrend
        mock_ticker.history.return_value = self._create_mock_history(prices)
        mock_ticker_class.return_value = mock_ticker

        agent = TechnicalAgent()
        result = agent.research("TEST")

        assert "Above MAs" in result.summary

    @patch("cents.agents.technical.yf.Ticker")
    def test_research_below_moving_averages(self, mock_ticker_class):
        """Price below MAs is bearish."""
        mock_ticker = MagicMock()
        # Steadily falling prices - current price below both MAs
        prices = [200 - i for i in range(100)]  # 100 days of downtrend
        mock_ticker.history.return_value = self._create_mock_history(prices)
        mock_ticker_class.return_value = mock_ticker

        agent = TechnicalAgent()
        result = agent.research("TEST")

        assert "Below MAs" in result.summary

    @patch("cents.agents.technical.yf.Ticker")
    def test_research_high_volume(self, mock_ticker_class):
        """High volume with price increase is bullish."""
        mock_ticker = MagicMock()
        prices = [100 + i for i in range(30)]  # Rising
        # Recent volume 2x average
        volumes = [1000000] * 25 + [2000000] * 5
        mock_ticker.history.return_value = self._create_mock_history(prices, volumes)
        mock_ticker_class.return_value = mock_ticker

        agent = TechnicalAgent()
        result = agent.research("TEST")

        assert "High volume" in result.summary

    @patch("cents.agents.technical.yf.Ticker")
    def test_research_near_52w_high(self, mock_ticker_class):
        """Price near 52-week high is bullish."""
        mock_ticker = MagicMock()
        # Price at all-time high
        prices = [100] * 50 + [150]  # Jump to new high
        mock_ticker.history.return_value = self._create_mock_history(prices)
        mock_ticker_class.return_value = mock_ticker

        agent = TechnicalAgent()
        result = agent.research("TEST")

        assert "Near 52w high" in result.summary

    @patch("cents.agents.technical.yf.Ticker")
    def test_research_empty_history(self, mock_ticker_class):
        """Handles empty history gracefully."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        agent = TechnicalAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "No historical data" in result.summary


class TestMacroAgent:
    """Tests for MacroAgent with mocked FRED API."""

    @patch.dict("os.environ", {"FRED_API_KEY": ""}, clear=True)
    def test_research_no_api_key(self):
        """Returns guidance when no API key configured."""
        agent = MacroAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "not configured" in result.summary
        assert result.evidence[0].type == EvidenceType.CONTRADICTING

    @patch.dict("os.environ", {"FRED_API_KEY": "test_key"})
    def test_research_high_rates_bearish(self):
        """High fed funds rate is bearish."""
        agent = MacroAgent()

        # Directly test interpretation logic for high rates
        ev_type, delta, note = agent._interpret_indicator("DFF", 5.5)

        assert ev_type == EvidenceType.CONTRADICTING
        assert delta < 0
        assert "High rates" in note

    @patch("cents.agents.macro.urlopen")
    @patch.dict("os.environ", {"FRED_API_KEY": "test_key"})
    def test_research_low_rates_bullish(self, mock_urlopen):
        """Low fed funds rate is bullish."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "observations": [{"value": "1.5", "date": "2024-01-01"}]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        agent = MacroAgent()
        result = agent.research("TEST")

        assert result.conviction_delta > 0

    def test_interpret_inverted_yield_curve(self):
        """Inverted yield curve is very bearish."""
        agent = MacroAgent()
        ev_type, delta, note = agent._interpret_indicator("T10Y2Y", -0.5)

        assert ev_type == EvidenceType.CONTRADICTING
        assert delta < 0
        assert "Inverted" in note

    def test_interpret_high_unemployment(self):
        """High unemployment is bearish."""
        agent = MacroAgent()
        ev_type, delta, note = agent._interpret_indicator("UNRATE", 7.0)

        assert ev_type == EvidenceType.CONTRADICTING
        assert delta < 0

    def test_interpret_low_unemployment(self):
        """Low unemployment is bullish."""
        agent = MacroAgent()
        ev_type, delta, note = agent._interpret_indicator("UNRATE", 3.5)

        assert ev_type == EvidenceType.SUPPORTING
        assert delta > 0

    def test_interpret_high_vix(self):
        """High VIX is bearish."""
        agent = MacroAgent()
        ev_type, delta, note = agent._interpret_indicator("VIXCLS", 35.0)

        assert ev_type == EvidenceType.CONTRADICTING
        assert delta < 0
        assert "High VIX" in note

    def test_interpret_low_vix(self):
        """Low VIX is bullish."""
        agent = MacroAgent()
        ev_type, delta, note = agent._interpret_indicator("VIXCLS", 12.0)

        assert ev_type == EvidenceType.SUPPORTING
        assert delta > 0


class TestSentimentAgent:
    """Tests for SentimentAgent with mocked News API."""

    @patch.dict("os.environ", {"NEWS_API_KEY": ""}, clear=True)
    def test_research_no_api_key(self):
        """Returns guidance when no API key configured."""
        agent = SentimentAgent()
        result = agent.research("AAPL")

        assert result.conviction_delta == 0
        assert "not configured" in result.summary

    @patch("cents.agents.sentiment.urlopen")
    @patch.dict("os.environ", {"NEWS_API_KEY": "test_key"})
    def test_research_positive_news(self, mock_urlopen):
        """Positive news generates bullish signal."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "articles": [
                {"title": "Company beats earnings, stock surges", "source": {"name": "News"}},
                {"title": "Analysts upgrade to buy rating", "source": {"name": "News"}},
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        agent = SentimentAgent()
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "Positive news sentiment" in result.summary

    @patch("cents.agents.sentiment.urlopen")
    @patch.dict("os.environ", {"NEWS_API_KEY": "test_key"})
    def test_research_negative_news(self, mock_urlopen):
        """Negative news generates bearish signal."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "articles": [
                {"title": "Company misses earnings, stock falls", "source": {"name": "News"}},
                {"title": "Investigation into company practices", "source": {"name": "News"}},
                {"title": "Analysts downgrade amid losses", "source": {"name": "News"}},
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        agent = SentimentAgent()
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "Negative news sentiment" in result.summary

    @patch("cents.agents.sentiment.urlopen")
    @patch.dict("os.environ", {"NEWS_API_KEY": "test_key"})
    def test_research_no_articles(self, mock_urlopen):
        """Handles no articles gracefully."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"articles": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        agent = SentimentAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "No recent news" in result.summary

    @patch.dict("os.environ", {}, clear=True)
    def test_research_missing_news_api_key_warns(self):
        """Explicit warning is returned when NEWS_API_KEY is absent."""
        agent = SentimentAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert result.evidence[0].type == EvidenceType.CONTRADICTING
        assert "WARNING" in result.summary


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent."""

    @patch.object(FundamentalsAgent, "research")
    @patch.object(TechnicalAgent, "research")
    @patch.object(MacroAgent, "research")
    @patch.object(SentimentAgent, "research")
    def test_research_aggregates_results(
        self, mock_sentiment, mock_macro, mock_technical, mock_fundamentals
    ):
        """Orchestrator aggregates all agent results."""
        # Setup mock returns
        mock_fundamentals.return_value = AgentResult(
            evidence=[], conviction_delta=5.0, summary="Fundamentals: bullish"
        )
        mock_technical.return_value = AgentResult(
            evidence=[], conviction_delta=3.0, summary="Technical: bullish"
        )
        mock_macro.return_value = AgentResult(
            evidence=[], conviction_delta=-2.0, summary="Macro: bearish"
        )
        mock_sentiment.return_value = AgentResult(
            evidence=[], conviction_delta=1.0, summary="Sentiment: neutral"
        )

        agent = OrchestratorAgent()
        result = agent.research("TEST")

        # Total should be 5 + 3 - 2 + 1 = 7
        assert result.conviction_delta == 7.0
        assert "fundamentals: +5" in result.summary
        assert "technical: +3" in result.summary
        assert "macro: -2" in result.summary

    @patch.object(FundamentalsAgent, "research")
    @patch.object(TechnicalAgent, "research")
    @patch.object(MacroAgent, "research")
    @patch.object(SentimentAgent, "research")
    def test_research_synthesizes_evidence(
        self, mock_sentiment, mock_macro, mock_technical, mock_fundamentals
    ):
        """Orchestrator creates synthesis evidence."""
        from cents.models import Evidence

        supporting_evidence = Evidence(
            thesis_id="test",
            agent="fundamentals",
            type=EvidenceType.SUPPORTING,
            content="Good metric",
            source="test",
        )
        contradicting_evidence = Evidence(
            thesis_id="test",
            agent="technical",
            type=EvidenceType.CONTRADICTING,
            content="Bad metric",
            source="test",
        )

        mock_fundamentals.return_value = AgentResult(
            evidence=[supporting_evidence, supporting_evidence],
            conviction_delta=5.0,
            summary="Bullish",
        )
        mock_technical.return_value = AgentResult(
            evidence=[contradicting_evidence],
            conviction_delta=-2.0,
            summary="Bearish",
        )
        mock_macro.return_value = AgentResult(
            evidence=[], conviction_delta=0, summary="Neutral"
        )
        mock_sentiment.return_value = AgentResult(
            evidence=[], conviction_delta=0, summary="Neutral"
        )

        agent = OrchestratorAgent()
        result = agent.research("TEST")

        # Should have synthesis evidence
        synthesis = [e for e in result.evidence if e.agent == "orchestrator"]
        assert len(synthesis) == 1
        assert "2 supporting" in synthesis[0].content or "supporting" in synthesis[0].content.lower()

    @patch.object(FundamentalsAgent, "research")
    @patch.object(TechnicalAgent, "research")
    @patch.object(MacroAgent, "research")
    @patch.object(SentimentAgent, "research")
    def test_research_all_bullish_consensus(
        self, mock_sentiment, mock_macro, mock_technical, mock_fundamentals
    ):
        """Strong consensus when all agents bullish."""
        for mock in [mock_fundamentals, mock_technical, mock_macro, mock_sentiment]:
            mock.return_value = AgentResult(
                evidence=[], conviction_delta=5.0, summary="Bullish"
            )

        agent = OrchestratorAgent()
        result = agent.research("TEST")

        # All 4 agents bullish = 20 total
        assert result.conviction_delta == 20.0
        # Synthesis should note agreement
        synthesis = [e for e in result.evidence if e.agent == "orchestrator"]
        assert any("agreement" in e.content.lower() for e in synthesis)
