"""Tests for research agents with mocked external dependencies."""

from datetime import datetime, timedelta
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
from cents.data import PriceBar, PriceHistory
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
    """Tests for FundamentalsAgent with mocked fundamentals provider."""

    def _create_mock_provider(self, **kwargs):
        """Create a mock fundamentals provider returning given data."""
        from cents.data import FundamentalsData
        mock_provider = MagicMock()
        data = FundamentalsData(symbol="TEST", **kwargs)
        mock_provider.get_fundamentals.return_value = data
        return mock_provider

    def test_research_low_pe_bullish(self):
        """Low P/E ratio generates bullish signal."""
        mock_provider = self._create_mock_provider(
            pe_ratio=12.0,
            name="Test Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "Low P/E" in result.summary
        assert len(result.evidence) >= 1

    def test_research_high_pe_bearish(self):
        """High P/E ratio generates bearish signal."""
        mock_provider = self._create_mock_provider(
            pe_ratio=50.0,
            name="Expensive Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "High P/E" in result.summary

    def test_research_strong_growth(self):
        """Strong revenue growth is bullish."""
        mock_provider = self._create_mock_provider(
            revenue_growth=0.30,  # 30%
            name="Growth Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "Strong revenue growth" in result.summary

    def test_research_negative_growth(self):
        """Negative revenue growth is bearish."""
        mock_provider = self._create_mock_provider(
            revenue_growth=-0.10,  # -10%
            name="Declining Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "Negative revenue growth" in result.summary

    def test_research_high_debt(self):
        """High debt-to-equity is bearish."""
        mock_provider = self._create_mock_provider(
            debt_to_equity=250.0,
            name="Leveraged Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "High debt" in result.summary

    def test_research_analyst_buy(self):
        """Buy recommendation is bullish."""
        mock_provider = self._create_mock_provider(
            recommendation="strong_buy",
            name="Hot Stock",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        evidence_content = [e.content for e in result.evidence]
        assert any("Strong Buy" in c for c in evidence_content)

    def test_research_with_thesis(self):
        """Research uses thesis ID when provided."""
        mock_provider = self._create_mock_provider(
            pe_ratio=15.0,
            name="Test",
        )

        thesis = Thesis(title="Test thesis")
        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST", thesis)

        for e in result.evidence:
            assert e.thesis_id == thesis.id

    def test_research_api_error(self):
        """Handles API errors gracefully."""
        mock_provider = MagicMock()
        mock_provider.get_fundamentals.side_effect = Exception("API Error")

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "failed" in result.summary.lower()
        assert "API Error" in result.summary

    def test_research_retries_on_transient_failure(self):
        """Retries fetch before failing."""
        from cents.data import FundamentalsData
        mock_provider = MagicMock()
        mock_provider.get_fundamentals.side_effect = [
            Exception("temporary"),
            FundamentalsData(symbol="TEST", name="Retry Corp", pe_ratio=20.0),
        ]

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert "Retry Corp" in result.summary

    def test_research_no_signals(self):
        """No significant signals when data is neutral."""
        mock_provider = self._create_mock_provider(
            pe_ratio=20.0,  # Neutral
            name="Average Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert "No significant signals" in result.summary


class TestTechnicalAgent:
    """Tests for TechnicalAgent with mocked price provider."""

    def _create_price_history(self, prices, volumes=None):
        """Helper to create PriceHistory from price list."""
        n = len(prices)
        if volumes is None:
            volumes = [1000000] * n
        base_time = datetime.now() - timedelta(days=n)
        bars = [
            PriceBar(
                timestamp=base_time + timedelta(days=i),
                open=prices[i],
                high=prices[i] * 1.01,
                low=prices[i] * 0.99,
                close=prices[i],
                volume=volumes[i],
            )
            for i in range(n)
        ]
        return PriceHistory(symbol="TEST", bars=bars)

    def _create_mock_provider(self, prices, volumes=None):
        """Create a mock price provider returning given price history."""
        mock_provider = MagicMock()
        mock_provider.get_history.return_value = self._create_price_history(prices, volumes)
        return mock_provider

    def test_research_strong_momentum(self):
        """Strong upward momentum is bullish."""
        # Price went from 100 to 120 over 30 days (+20%)
        prices = [100 + (i * 0.67) for i in range(30)]
        mock_provider = self._create_mock_provider(prices)

        agent = TechnicalAgent(price_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "Strong momentum" in result.summary

    def test_research_weak_momentum(self):
        """Strong downward momentum is bearish."""
        # Price went from 100 to 80 over 30 days (-20%)
        prices = [100 - (i * 0.67) for i in range(30)]
        mock_provider = self._create_mock_provider(prices)

        agent = TechnicalAgent(price_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "Weak momentum" in result.summary

    def test_research_above_moving_averages(self):
        """Price above MAs is bullish."""
        # Steadily rising prices - current price above both MAs
        prices = [100 + i for i in range(100)]  # 100 days of uptrend
        mock_provider = self._create_mock_provider(prices)

        agent = TechnicalAgent(price_provider=mock_provider)
        result = agent.research("TEST")

        assert "Above MAs" in result.summary

    def test_research_below_moving_averages(self):
        """Price below MAs is bearish."""
        # Steadily falling prices - current price below both MAs
        prices = [200 - i for i in range(100)]  # 100 days of downtrend
        mock_provider = self._create_mock_provider(prices)

        agent = TechnicalAgent(price_provider=mock_provider)
        result = agent.research("TEST")

        assert "Below MAs" in result.summary

    def test_research_high_volume(self):
        """High volume with price increase is bullish."""
        prices = [100 + i for i in range(30)]  # Rising
        # Recent volume 2x average
        volumes = [1000000] * 25 + [2000000] * 5
        mock_provider = self._create_mock_provider(prices, volumes)

        agent = TechnicalAgent(price_provider=mock_provider)
        result = agent.research("TEST")

        assert "High volume" in result.summary

    def test_research_near_52w_high(self):
        """Price near 52-week high is bullish."""
        # Price at all-time high
        prices = [100] * 50 + [150]  # Jump to new high
        mock_provider = self._create_mock_provider(prices)

        agent = TechnicalAgent(price_provider=mock_provider)
        result = agent.research("TEST")

        assert "Near 52w high" in result.summary

    def test_research_empty_history(self):
        """Handles empty history gracefully."""
        mock_provider = MagicMock()
        mock_provider.get_history.return_value = PriceHistory(symbol="TEST", bars=[])

        agent = TechnicalAgent(price_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "No historical data" in result.summary


class TestMacroAgent:
    """Tests for MacroAgent with mocked FRED API."""

    @patch("cents.agents.macro.get_settings")
    def test_research_no_api_key(self, mock_settings):
        """Returns guidance when no API key configured."""
        mock_settings.return_value.fred_api_key = None
        agent = MacroAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "not configured" in result.summary
        # Missing API key should be neutral, not contradicting (it's not counter-evidence)
        assert result.evidence[0].type == EvidenceType.NEUTRAL

    def test_research_high_rates_bearish(self):
        """High fed funds rate is bearish."""
        agent = MacroAgent()

        # Directly test interpretation logic for high rates
        ev_type, delta, note = agent._interpret_indicator("DFF", 5.5)

        assert ev_type == EvidenceType.CONTRADICTING
        assert delta < 0
        assert "High rates" in note

    @patch("cents.agents.macro.urlopen")
    @patch("cents.agents.macro.get_settings")
    def test_research_low_rates_bullish(self, mock_settings, mock_urlopen):
        """Low fed funds rate is bullish."""
        mock_settings.return_value.fred_api_key = "test_key"
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

    @patch("cents.agents.sentiment.get_settings")
    def test_research_no_api_key(self, mock_settings):
        """Returns guidance when no API key configured."""
        mock_settings.return_value.news_api_key = None
        agent = SentimentAgent()
        result = agent.research("AAPL")

        assert result.conviction_delta == 0
        assert "not configured" in result.summary

    @patch("cents.agents.sentiment.urlopen")
    @patch("cents.agents.sentiment.get_settings")
    def test_research_positive_news(self, mock_settings, mock_urlopen):
        """Positive news generates bullish signal."""
        mock_settings.return_value.news_api_key = "test_key"
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
    @patch("cents.agents.sentiment.get_settings")
    def test_research_negative_news(self, mock_settings, mock_urlopen):
        """Negative news generates bearish signal."""
        mock_settings.return_value.news_api_key = "test_key"
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
    @patch("cents.agents.sentiment.get_settings")
    def test_research_no_articles(self, mock_settings, mock_urlopen):
        """Handles no articles gracefully."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"articles": []}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        agent = SentimentAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "No recent news" in result.summary

    @patch("cents.agents.sentiment.get_settings")
    def test_research_missing_news_api_key_warns(self, mock_settings):
        """Explicit warning is returned when NEWS_API_KEY is absent."""
        mock_settings.return_value.news_api_key = None
        agent = SentimentAgent()
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        # Missing API key should be neutral, not contradicting (it's not counter-evidence)
        assert result.evidence[0].type == EvidenceType.NEUTRAL
        assert "WARNING" in result.summary

    def test_negation_flips_positive_to_negative(self):
        """Negation words before positive keywords flip sentiment."""
        agent = SentimentAgent()

        # "failed to beat" should be negative, not positive
        pos, neg = agent._count_sentiment_words("Company failed to beat earnings")
        assert neg > pos, "Negated positive should count as negative"

        # "not bullish" should be negative
        pos, neg = agent._count_sentiment_words("Not bullish on this stock")
        assert neg > pos

    def test_negation_flips_negative_to_positive(self):
        """Negation words before negative keywords flip sentiment."""
        agent = SentimentAgent()

        # "no decline" should be positive
        pos, neg = agent._count_sentiment_words("No decline in revenue")
        assert pos > neg, "Negated negative should count as positive"

        # "never bearish" should be positive
        pos, neg = agent._count_sentiment_words("Never bearish on this name")
        assert pos > neg

    def test_negation_after_keyword(self):
        """Negation immediately after keyword also flips sentiment."""
        agent = SentimentAgent()

        # "upgrade unlikely" should be negative
        pos, neg = agent._count_sentiment_words("Upgrade unlikely given headwinds")
        assert neg > pos, "Post-keyword negation should flip sentiment"

    def test_no_negation_preserves_sentiment(self):
        """Without negation, sentiment words count normally."""
        agent = SentimentAgent()

        # Plain positive
        pos, neg = agent._count_sentiment_words("Company beats expectations")
        assert pos > neg

        # Plain negative
        pos, neg = agent._count_sentiment_words("Stock drops amid concerns")
        assert neg > pos

    @patch("cents.agents.sentiment.urlopen")
    @patch("cents.agents.sentiment.get_settings")
    def test_research_negated_positive_is_bearish(self, mock_settings, mock_urlopen):
        """News with negated positive words generates bearish signal."""
        mock_settings.return_value.news_api_key = "test_key"
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "articles": [
                {"title": "Company failed to beat earnings", "source": {"name": "News"}},
                {"title": "Upgrade unlikely after weak quarter", "source": {"name": "News"}},
            ]
        }).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        agent = SentimentAgent()
        result = agent.research("TEST")

        # Should be bearish despite having "beat" and "upgrade" in text
        assert result.conviction_delta < 0


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

        # Total should be 5 + 3 - 2 + 1 = 7 (no evidence = raw deltas used)
        assert result.conviction_delta == 7.0
        assert "fundamentals: +5.0" in result.summary
        assert "technical: +3.0" in result.summary
        assert "macro: -2.0" in result.summary

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

    @patch.object(FundamentalsAgent, "research")
    @patch.object(TechnicalAgent, "research")
    @patch.object(MacroAgent, "research")
    @patch.object(SentimentAgent, "research")
    def test_confidence_weighting_high_confidence(
        self, mock_sentiment, mock_macro, mock_technical, mock_fundamentals
    ):
        """High-confidence evidence weighs conviction more heavily."""
        from cents.models import Evidence

        # High confidence evidence (0.9 avg)
        high_conf_evidence = Evidence(
            thesis_id="test",
            agent="fundamentals",
            type=EvidenceType.SUPPORTING,
            content="Strong signal",
            source="test",
            confidence=0.9,
        )

        mock_fundamentals.return_value = AgentResult(
            evidence=[high_conf_evidence],
            conviction_delta=10.0,  # Raw delta
            summary="High confidence bullish",
        )
        # Other agents return zero
        for mock in [mock_technical, mock_macro, mock_sentiment]:
            mock.return_value = AgentResult(
                evidence=[], conviction_delta=0, summary="Neutral"
            )

        agent = OrchestratorAgent()
        result = agent.research("TEST")

        # Weighted: 10.0 * 0.9 = 9.0
        assert result.conviction_delta == pytest.approx(9.0, rel=0.01)

    @patch.object(FundamentalsAgent, "research")
    @patch.object(TechnicalAgent, "research")
    @patch.object(MacroAgent, "research")
    @patch.object(SentimentAgent, "research")
    def test_confidence_weighting_low_confidence(
        self, mock_sentiment, mock_macro, mock_technical, mock_fundamentals
    ):
        """Low-confidence evidence weighs conviction less."""
        from cents.models import Evidence

        # Low confidence evidence (0.3 avg)
        low_conf_evidence = Evidence(
            thesis_id="test",
            agent="sentiment",
            type=EvidenceType.SUPPORTING,
            content="Weak signal",
            source="test",
            confidence=0.3,
        )

        mock_sentiment.return_value = AgentResult(
            evidence=[low_conf_evidence],
            conviction_delta=10.0,  # Raw delta
            summary="Low confidence bullish",
        )
        # Other agents return zero
        for mock in [mock_fundamentals, mock_technical, mock_macro]:
            mock.return_value = AgentResult(
                evidence=[], conviction_delta=0, summary="Neutral"
            )

        agent = OrchestratorAgent()
        result = agent.research("TEST")

        # Weighted: 10.0 * 0.3 = 3.0
        assert result.conviction_delta == pytest.approx(3.0, rel=0.01)

    @patch.object(FundamentalsAgent, "research")
    @patch.object(TechnicalAgent, "research")
    @patch.object(MacroAgent, "research")
    @patch.object(SentimentAgent, "research")
    def test_confidence_weighting_no_evidence_uses_raw(
        self, mock_sentiment, mock_macro, mock_technical, mock_fundamentals
    ):
        """When no evidence, raw conviction delta is used."""
        # Agent with delta but no evidence
        mock_fundamentals.return_value = AgentResult(
            evidence=[],  # No evidence
            conviction_delta=5.0,
            summary="Bullish",
        )
        for mock in [mock_technical, mock_macro, mock_sentiment]:
            mock.return_value = AgentResult(
                evidence=[], conviction_delta=0, summary="Neutral"
            )

        agent = OrchestratorAgent()
        result = agent.research("TEST")

        # No evidence = use raw delta (5.0)
        assert result.conviction_delta == 5.0


class TestFundamentalsForwardMetrics:
    """Tests for forward-looking metrics in FundamentalsAgent."""

    def _create_mock_provider(self, **kwargs):
        """Create a mock fundamentals provider returning given data."""
        from cents.data import FundamentalsData
        mock_provider = MagicMock()
        data = FundamentalsData(symbol="TEST", **kwargs)
        mock_provider.get_fundamentals.return_value = data
        return mock_provider

    def test_forward_pe_lower_than_trailing_bullish(self):
        """Forward P/E significantly lower than trailing is bullish."""
        mock_provider = self._create_mock_provider(
            pe_ratio=25.0,
            forward_pe=18.0,  # 28% lower
            name="Growth Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "growth expected" in result.summary.lower()

    def test_forward_pe_higher_than_trailing_bearish(self):
        """Forward P/E significantly higher than trailing is bearish."""
        mock_provider = self._create_mock_provider(
            pe_ratio=20.0,
            forward_pe=28.0,  # 40% higher
            name="Declining Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "decline expected" in result.summary.lower()

    def test_forward_pe_similar_to_trailing_neutral(self):
        """Forward P/E similar to trailing is neutral."""
        mock_provider = self._create_mock_provider(
            pe_ratio=20.0,
            forward_pe=21.0,  # Only 5% higher
            name="Stable Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # No forward P/E delta added (within 20% threshold)
        # Check evidence was created but delta is 0 for forward PE
        forward_evidence = [e for e in result.evidence if "Forward P/E" in e.content]
        assert len(forward_evidence) == 1
        assert forward_evidence[0].type == EvidenceType.NEUTRAL

    def test_earnings_growth_positive_bullish(self):
        """Expected earnings growth > 15% is bullish."""
        mock_provider = self._create_mock_provider(
            earnings_growth=0.25,  # 25%
            name="High Growth Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 0
        assert "earnings growth" in result.summary.lower()

    def test_earnings_decline_bearish(self):
        """Expected earnings decline > 10% is bearish."""
        mock_provider = self._create_mock_provider(
            earnings_growth=-0.20,  # -20%
            name="Declining Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert "earnings decline" in result.summary.lower()

    def test_no_forward_metrics_no_change(self):
        """When forward metrics are None, behavior unchanged."""
        mock_provider = self._create_mock_provider(
            pe_ratio=20.0,
            forward_pe=None,
            earnings_growth=None,
            name="Normal Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Only trailing P/E evidence, no forward evidence
        forward_evidence = [e for e in result.evidence if "Forward P/E" in e.content]
        assert len(forward_evidence) == 0


class TestFundamentalsSectorRelativeScoring:
    """Tests for sector-relative valuation scoring in FundamentalsAgent."""

    def _create_mock_provider(self, **kwargs):
        """Create a mock fundamentals provider returning given data."""
        from cents.data import FundamentalsData
        mock_provider = MagicMock()
        data = FundamentalsData(symbol="TEST", **kwargs)
        mock_provider.get_fundamentals.return_value = data
        return mock_provider

    def test_tech_high_pe_acceptable(self):
        """Tech sector P/E of 35 is acceptable (median 28, threshold 36.4)."""
        mock_provider = self._create_mock_provider(
            pe_ratio=35.0,
            sector="Technology",
            name="Tech Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 35 is below 36.4 (28 * 1.3) - should not be bearish
        assert result.conviction_delta >= 0
        assert "High P/E" not in result.summary

    def test_financial_pe_35_overvalued(self):
        """Financial sector P/E of 35 is overvalued (median 14, threshold 18.2)."""
        mock_provider = self._create_mock_provider(
            pe_ratio=35.0,
            sector="Financial Services",
            name="Bank Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 35 is well above 18.2 (14 * 1.3) - should be bearish
        assert result.conviction_delta < 0
        assert "High P/E" in result.summary
        assert "Financial Services" in result.summary

    def test_tech_low_pe_undervalued(self):
        """Tech sector P/E of 15 is undervalued (threshold 19.6)."""
        mock_provider = self._create_mock_provider(
            pe_ratio=15.0,
            sector="Technology",
            name="Cheap Tech",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 15 is below 19.6 (28 * 0.7) - should be bullish
        assert result.conviction_delta > 0
        assert "Low P/E" in result.summary
        assert "Technology" in result.summary

    def test_utility_high_debt_acceptable(self):
        """Utilities sector D/E of 200% is acceptable (norm 150%, threshold 225%)."""
        mock_provider = self._create_mock_provider(
            debt_to_equity=2.0,  # 200%
            sector="Utilities",
            name="Power Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 200% is below 225% (150 * 1.5) - should not be bearish for high debt
        assert "High debt" not in result.summary

    def test_tech_high_debt_risky(self):
        """Tech sector D/E of 200% is risky (norm 50%, threshold 75%)."""
        mock_provider = self._create_mock_provider(
            debt_to_equity=2.0,  # 200%
            sector="Technology",
            name="Leveraged Tech",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 200% is well above 75% (50 * 1.5) - should be bearish
        assert result.conviction_delta < 0
        assert "High debt" in result.summary
        assert "Technology" in result.summary

    def test_unknown_sector_uses_defaults(self):
        """Unknown sector uses default thresholds."""
        mock_provider = self._create_mock_provider(
            pe_ratio=35.0,
            sector="Unknown Industry",
            name="Mystery Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 35 is above default high (30) - should be bearish
        assert result.conviction_delta < 0
        assert "High P/E" in result.summary

    def test_no_sector_uses_defaults(self):
        """No sector (None) uses default thresholds."""
        mock_provider = self._create_mock_provider(
            pe_ratio=35.0,
            sector=None,
            name="No Sector Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 35 is above default high (30) - should be bearish
        assert result.conviction_delta < 0
        assert "High P/E" in result.summary

    def test_sector_margin_thresholds(self):
        """Financial sector high margin (30%) is excellent (median 22%)."""
        mock_provider = self._create_mock_provider(
            profit_margin=0.30,  # 30%
            sector="Financial Services",
            name="Profitable Bank",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 30% is above 33% (22% * 1.5) - check evidence for sector note
        evidence = [e for e in result.evidence if "Profit Margin" in e.content]
        assert len(evidence) == 1
        assert "Financial Services" in evidence[0].content

    def test_real_estate_pe_threshold(self):
        """Real estate sector has highest P/E threshold (median 35)."""
        mock_provider = self._create_mock_provider(
            pe_ratio=40.0,
            sector="Real Estate",
            name="REIT Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 40 is below 45.5 (35 * 1.3) - should not be bearish
        assert "High P/E" not in result.summary

    def test_energy_low_pe_expected(self):
        """Energy sector has low P/E threshold (median 12)."""
        mock_provider = self._create_mock_provider(
            pe_ratio=8.0,
            sector="Energy",
            name="Oil Corp",
        )

        agent = FundamentalsAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 8 is below 8.4 (12 * 0.7) - should be bullish
        assert result.conviction_delta > 0
        assert "Low P/E" in result.summary
        assert "Energy" in result.summary
