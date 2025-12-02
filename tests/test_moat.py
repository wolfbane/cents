"""Tests for MoatAgent with mocked FMP provider."""

from unittest.mock import MagicMock

import pytest

from cents.agents import MoatAgent, AgentResult
from cents.data import FundamentalsData
from cents.models import Thesis, EvidenceType, ThesisDimension


class TestMoatAgent:
    """Tests for MoatAgent."""

    def _create_mock_provider(
        self,
        historical_ratios=None,
        fundamentals_kwargs=None,
    ):
        """Create a mock FMP provider with given historical ratios."""
        mock_provider = MagicMock()

        # Set up historical ratios
        if historical_ratios is None:
            historical_ratios = []
        mock_provider.get_historical_ratios.return_value = historical_ratios

        # Set up fundamentals for sector info
        if fundamentals_kwargs is None:
            fundamentals_kwargs = {}
        mock_provider.get_fundamentals.return_value = FundamentalsData(
            symbol="TEST",
            **fundamentals_kwargs,
        )

        return mock_provider

    def test_research_strong_moat(self):
        """High ROIC with stable margins indicates strong moat."""
        # 5 years of high, consistent ROIC and margins
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.20, "returnOnEquity": 0.25,
             "grossProfitMargin": 0.55, "operatingProfitMargin": 0.30, "netProfitMargin": 0.20},
            {"date": "2022-12-31", "roic": 0.19, "returnOnEquity": 0.24,
             "grossProfitMargin": 0.54, "operatingProfitMargin": 0.29, "netProfitMargin": 0.19},
            {"date": "2021-12-31", "roic": 0.21, "returnOnEquity": 0.26,
             "grossProfitMargin": 0.55, "operatingProfitMargin": 0.31, "netProfitMargin": 0.21},
            {"date": "2020-12-31", "roic": 0.18, "returnOnEquity": 0.23,
             "grossProfitMargin": 0.53, "operatingProfitMargin": 0.28, "netProfitMargin": 0.18},
            {"date": "2019-12-31", "roic": 0.20, "returnOnEquity": 0.25,
             "grossProfitMargin": 0.54, "operatingProfitMargin": 0.30, "netProfitMargin": 0.20},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Strong Moat Corp", "sector": "Technology"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Should have strong positive conviction
        assert result.conviction_delta > 5
        assert "Strong returns on capital" in result.summary
        assert "moat" in result.dimension_scores
        assert result.dimension_scores["moat"] > 0

    def test_research_weak_moat(self):
        """Low ROIC with volatile margins indicates weak moat."""
        # 5 years of low, inconsistent ROIC
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.05, "returnOnEquity": 0.08,
             "grossProfitMargin": 0.30, "operatingProfitMargin": 0.10, "netProfitMargin": 0.05},
            {"date": "2022-12-31", "roic": 0.12, "returnOnEquity": 0.15,
             "grossProfitMargin": 0.40, "operatingProfitMargin": 0.18, "netProfitMargin": 0.12},
            {"date": "2021-12-31", "roic": 0.03, "returnOnEquity": 0.04,
             "grossProfitMargin": 0.25, "operatingProfitMargin": 0.05, "netProfitMargin": 0.02},
            {"date": "2020-12-31", "roic": 0.08, "returnOnEquity": 0.10,
             "grossProfitMargin": 0.35, "operatingProfitMargin": 0.12, "netProfitMargin": 0.08},
            {"date": "2019-12-31", "roic": 0.06, "returnOnEquity": 0.09,
             "grossProfitMargin": 0.28, "operatingProfitMargin": 0.08, "netProfitMargin": 0.04},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Weak Moat Corp", "sector": "Consumer Cyclical"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Should have negative conviction
        assert result.conviction_delta < 0
        assert "moat" in result.dimension_scores

    def test_research_high_roic_variance_penalized(self):
        """High variance in ROIC reduces moat score."""
        # High average ROIC but very volatile
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.30, "returnOnEquity": 0.35,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.20},
            {"date": "2022-12-31", "roic": 0.05, "returnOnEquity": 0.08,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.20},
            {"date": "2021-12-31", "roic": 0.25, "returnOnEquity": 0.30,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.20},
            {"date": "2020-12-31", "roic": 0.08, "returnOnEquity": 0.10,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.20},
            {"date": "2019-12-31", "roic": 0.22, "returnOnEquity": 0.27,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.20},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Volatile Corp", "sector": "Technology"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Should have evidence about high variance
        variance_evidence = [e for e in result.evidence if "variance" in e.content.lower()]
        assert len(variance_evidence) >= 1
        assert variance_evidence[0].type == EvidenceType.CONTRADICTING

    def test_research_stable_margins_bullish(self):
        """Low margin variance indicates stable business."""
        # Very stable gross margins over 5 years
        historical_ratios = [
            {"date": f"202{i}-12-31", "roic": 0.12, "returnOnEquity": 0.15,
             "grossProfitMargin": 0.40, "operatingProfitMargin": 0.20, "netProfitMargin": 0.12}
            for i in range(5)
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Stable Corp", "sector": "Consumer Defensive"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Low variance should be supporting
        stability_evidence = [e for e in result.evidence if "stability" in e.content.lower()]
        assert len(stability_evidence) >= 1
        assert stability_evidence[0].type == EvidenceType.SUPPORTING

    def test_research_expanding_margins_bullish(self):
        """Expanding margins indicate strengthening moat."""
        # Margins improving over time (older to newer)
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.15, "returnOnEquity": 0.18,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.15},
            {"date": "2022-12-31", "roic": 0.14, "returnOnEquity": 0.17,
             "grossProfitMargin": 0.48, "operatingProfitMargin": 0.24, "netProfitMargin": 0.14},
            {"date": "2021-12-31", "roic": 0.13, "returnOnEquity": 0.16,
             "grossProfitMargin": 0.45, "operatingProfitMargin": 0.22, "netProfitMargin": 0.13},
            {"date": "2020-12-31", "roic": 0.12, "returnOnEquity": 0.15,
             "grossProfitMargin": 0.42, "operatingProfitMargin": 0.20, "netProfitMargin": 0.12},
            {"date": "2019-12-31", "roic": 0.11, "returnOnEquity": 0.14,
             "grossProfitMargin": 0.40, "operatingProfitMargin": 0.18, "netProfitMargin": 0.11},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Improving Corp", "sector": "Technology"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Expanding margins should be noted
        expand_evidence = [e for e in result.evidence if "expanding" in e.content.lower()]
        assert len(expand_evidence) >= 1
        assert expand_evidence[0].type == EvidenceType.SUPPORTING

    def test_research_contracting_margins_bearish(self):
        """Contracting margins indicate deteriorating moat."""
        # Margins declining over time
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.10, "returnOnEquity": 0.12,
             "grossProfitMargin": 0.30, "operatingProfitMargin": 0.12, "netProfitMargin": 0.08},
            {"date": "2022-12-31", "roic": 0.11, "returnOnEquity": 0.13,
             "grossProfitMargin": 0.32, "operatingProfitMargin": 0.14, "netProfitMargin": 0.09},
            {"date": "2021-12-31", "roic": 0.12, "returnOnEquity": 0.15,
             "grossProfitMargin": 0.35, "operatingProfitMargin": 0.16, "netProfitMargin": 0.10},
            {"date": "2020-12-31", "roic": 0.14, "returnOnEquity": 0.17,
             "grossProfitMargin": 0.38, "operatingProfitMargin": 0.18, "netProfitMargin": 0.12},
            {"date": "2019-12-31", "roic": 0.15, "returnOnEquity": 0.18,
             "grossProfitMargin": 0.40, "operatingProfitMargin": 0.20, "netProfitMargin": 0.14},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Declining Corp", "sector": "Consumer Cyclical"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Contracting margins should be noted
        contract_evidence = [e for e in result.evidence if "contracting" in e.content.lower()]
        assert len(contract_evidence) >= 1
        assert contract_evidence[0].type == EvidenceType.CONTRADICTING

    def test_research_premium_margins_pricing_power(self):
        """Gross margins well above sector indicate pricing power."""
        # Tech sector median is 0.55, this company has 0.70
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.15, "returnOnEquity": 0.18,
             "grossProfitMargin": 0.70, "operatingProfitMargin": 0.35, "netProfitMargin": 0.25},
            {"date": "2022-12-31", "roic": 0.14, "returnOnEquity": 0.17,
             "grossProfitMargin": 0.68, "operatingProfitMargin": 0.34, "netProfitMargin": 0.24},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Premium Corp", "sector": "Technology"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Should note premium margins vs sector
        pricing_evidence = [e for e in result.evidence if "vs sector" in e.content.lower()]
        assert len(pricing_evidence) >= 1
        assert pricing_evidence[0].type == EvidenceType.SUPPORTING

    def test_research_below_sector_margins(self):
        """Gross margins below sector indicate weak pricing."""
        # Tech sector median is 0.55, this company has 0.40
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.08, "returnOnEquity": 0.10,
             "grossProfitMargin": 0.40, "operatingProfitMargin": 0.15, "netProfitMargin": 0.08},
            {"date": "2022-12-31", "roic": 0.07, "returnOnEquity": 0.09,
             "grossProfitMargin": 0.38, "operatingProfitMargin": 0.14, "netProfitMargin": 0.07},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Commodity Corp", "sector": "Technology"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Below sector should be noted as weak
        pricing_evidence = [e for e in result.evidence if "vs sector" in e.content.lower()]
        assert len(pricing_evidence) >= 1
        assert pricing_evidence[0].type == EvidenceType.CONTRADICTING

    def test_research_no_historical_data(self):
        """Handles missing historical data gracefully."""
        mock_provider = self._create_mock_provider(
            historical_ratios=[],
            fundamentals_kwargs={"name": "New Corp"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "No historical data" in result.summary

    def test_research_partial_data(self):
        """Handles partial data (some fields None) gracefully."""
        historical_ratios = [
            {"date": "2023-12-31", "roic": None, "returnOnEquity": 0.15,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.15},
            {"date": "2022-12-31", "roic": None, "returnOnEquity": 0.14,
             "grossProfitMargin": 0.48, "operatingProfitMargin": 0.24, "netProfitMargin": 0.14},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Partial Data Corp"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Should fall back to ROE when ROIC is missing
        roe_evidence = [e for e in result.evidence if "ROE" in e.content]
        assert len(roe_evidence) >= 1

    def test_research_with_thesis(self):
        """Research uses thesis ID when provided."""
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.15, "returnOnEquity": 0.18,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.15},
            {"date": "2022-12-31", "roic": 0.14, "returnOnEquity": 0.17,
             "grossProfitMargin": 0.48, "operatingProfitMargin": 0.24, "netProfitMargin": 0.14},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Test Corp"},
        )

        thesis = Thesis(title="Test thesis")
        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST", thesis)

        for e in result.evidence:
            assert e.thesis_id == thesis.id

    def test_research_api_error(self):
        """Handles API errors gracefully."""
        mock_provider = MagicMock()
        mock_provider.get_historical_ratios.side_effect = ValueError("API Error")

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "failed" in result.summary.lower()
        assert "API Error" in result.summary

    def test_evidence_uses_moat_dimension(self):
        """All evidence should use MOAT dimension."""
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.15, "returnOnEquity": 0.18,
             "grossProfitMargin": 0.50, "operatingProfitMargin": 0.25, "netProfitMargin": 0.15},
            {"date": "2022-12-31", "roic": 0.14, "returnOnEquity": 0.17,
             "grossProfitMargin": 0.48, "operatingProfitMargin": 0.24, "netProfitMargin": 0.14},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "Test Corp", "sector": "Technology"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # All evidence should have MOAT dimension
        for e in result.evidence:
            assert e.dimension == ThesisDimension.MOAT

    def test_dimension_scores_includes_moat_and_quality(self):
        """Dimension scores should include both moat and quality."""
        historical_ratios = [
            {"date": "2023-12-31", "roic": 0.20, "returnOnEquity": 0.25,
             "grossProfitMargin": 0.55, "operatingProfitMargin": 0.30, "netProfitMargin": 0.20},
            {"date": "2022-12-31", "roic": 0.19, "returnOnEquity": 0.24,
             "grossProfitMargin": 0.54, "operatingProfitMargin": 0.29, "netProfitMargin": 0.19},
            {"date": "2021-12-31", "roic": 0.21, "returnOnEquity": 0.26,
             "grossProfitMargin": 0.55, "operatingProfitMargin": 0.31, "netProfitMargin": 0.21},
        ]

        mock_provider = self._create_mock_provider(
            historical_ratios=historical_ratios,
            fundamentals_kwargs={"name": "High Quality Corp", "sector": "Technology"},
        )

        agent = MoatAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Should have both moat and quality scores
        assert "moat" in result.dimension_scores
        assert "quality" in result.dimension_scores
        # High ROIC should contribute to both
        assert result.dimension_scores["moat"] > 0
        assert result.dimension_scores["quality"] > 0
