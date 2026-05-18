"""Tests for InsiderAgent with mocked FMP provider."""

from unittest.mock import MagicMock

import pytest

from cents.agents import InsiderAgent, AgentResult
from cents.models import Thesis, EvidenceType, ThesisDimension


class TestInsiderAgent:
    """Tests for InsiderAgent."""

    def _create_mock_provider(self, trades=None):
        """Create a mock FMP provider with given insider trades."""
        mock_provider = MagicMock()
        mock_provider.get_insider_trades.return_value = trades or []
        return mock_provider

    def _make_trade(
        self,
        tx_type="S-Sale",
        name="John Doe",
        role="officer: CEO",
        shares=1000,
        price=100.0,
        date="2025-01-15",
    ):
        """Helper to create a trade record."""
        return {
            "symbol": "TEST",
            "transactionDate": date,
            "transactionType": tx_type,
            "reportingName": name,
            "typeOfOwner": role,
            "securitiesTransacted": shares,
            "price": price,
            "acquisitionOrDisposition": "A" if tx_type == "P-Purchase" else "D",
        }

    def test_research_no_trades(self):
        """Returns neutral when no trades found."""
        mock_provider = self._create_mock_provider(trades=[])
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "No informative insider trades" in result.summary

    def test_research_filters_routine_trades(self):
        """Filters out gifts, awards, option exercises."""
        trades = [
            self._make_trade(tx_type="G-Gift", name="CEO", price=0),
            self._make_trade(tx_type="M-Exempt", name="CFO", price=0),
            self._make_trade(tx_type="A-Award", name="CTO", price=0),
            self._make_trade(tx_type="F-InKind", name="COO", price=100),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # All trades filtered out
        assert result.conviction_delta == 0
        assert "No informative insider trades" in result.summary

    def test_research_cluster_buying_bullish(self):
        """Cluster buying by multiple insiders is very bullish."""
        trades = [
            self._make_trade(
                tx_type="P-Purchase", name="Alice CEO", role="officer: CEO",
                shares=5000, price=100, date="2025-01-15"
            ),
            self._make_trade(
                tx_type="P-Purchase", name="Bob CFO", role="officer: CFO",
                shares=3000, price=100, date="2025-01-14"
            ),
            self._make_trade(
                tx_type="P-Purchase", name="Carol COO", role="officer: COO",
                shares=2000, price=100, date="2025-01-13"
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # 3 C-suite buying = strong bullish signal
        assert result.conviction_delta > 5
        assert "Cluster buying" in result.summary or "buys" in result.summary

        # Check for cluster evidence
        cluster_ev = [e for e in result.evidence if "cluster" in e.content.lower()]
        assert len(cluster_ev) >= 1
        assert cluster_ev[0].type == EvidenceType.SUPPORTING

    def test_research_two_insiders_buying(self):
        """Two insiders buying is moderately bullish."""
        trades = [
            self._make_trade(
                tx_type="P-Purchase", name="Alice CEO", role="officer: CEO",
                shares=5000, price=100, date="2025-01-15"
            ),
            self._make_trade(
                tx_type="P-Purchase", name="Bob CFO", role="officer: CFO",
                shares=3000, price=100, date="2025-01-14"
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 3
        evidence_content = " ".join(e.content for e in result.evidence)
        assert "Multiple insiders buying" in evidence_content or "C-suite" in evidence_content

    def test_research_large_ceo_purchase_bullish(self):
        """Large CEO purchase (>$500k) is very bullish."""
        trades = [
            self._make_trade(
                tx_type="P-Purchase", name="John CEO", role="officer: Chief Executive Officer",
                shares=10000, price=100,  # $1M purchase
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta > 3
        assert any("Large C-suite purchase" in e.content for e in result.evidence)

    def test_research_cluster_selling_bearish(self):
        """Multiple insiders selling is bearish."""
        trades = [
            self._make_trade(
                tx_type="S-Sale", name="Alice CEO", role="officer: CEO",
                shares=10000, price=100, date="2025-01-15"
            ),
            self._make_trade(
                tx_type="S-Sale", name="Bob CFO", role="officer: CFO",
                shares=8000, price=100, date="2025-01-14"
            ),
            self._make_trade(
                tx_type="S-Sale", name="Carol VP", role="officer: VP Sales",
                shares=5000, price=100, date="2025-01-13"
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        # Cluster selling should be bearish
        assert result.conviction_delta < 0
        cluster_ev = [e for e in result.evidence if "cluster" in e.content.lower() or "Multiple insiders selling" in e.content]
        if cluster_ev:
            assert cluster_ev[0].type == EvidenceType.CONTRADICTING

    def test_research_large_ceo_sale_bearish(self):
        """Large CEO sale (>$1M) is moderately bearish."""
        trades = [
            self._make_trade(
                tx_type="S-Sale", name="John CEO", role="officer: Chief Executive Officer",
                shares=20000, price=100,  # $2M sale
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta < 0
        assert any("Large C-suite sale" in e.content for e in result.evidence)

    def test_repeated_sales_by_same_insider_aggregated(self):
        """One CEO running a 10b5-1 program produces one row, not N rows.

        Regression for the KVYO case: 14 separate Bialecki sales over recent
        months were each emitting their own Evidence row and each penalizing
        conviction_delta, inflating the orchestrator's contradicting count.
        These all trace to one underlying decision (the trading plan), so
        they should aggregate to a single row.
        """
        # Eight separate $3M sales by the same CEO — a typical 10b5-1 cadence.
        trades = [
            self._make_trade(
                tx_type="S-Sale",
                name="Same CEO",
                role="officer: Chief Executive Officer",
                shares=30000,
                price=100,  # $3M per filing
                date=f"2025-01-{day:02d}",
            )
            for day in (2, 5, 8, 11, 14, 17, 20, 23)
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        large_sale_rows = [e for e in result.evidence if "Large C-suite sale" in e.content]
        assert len(large_sale_rows) == 1
        # Aggregated value should reflect all 8 filings ($24M)
        assert "$24,000,000" in large_sale_rows[0].content
        # And the content should disclose that this is a multi-filing aggregate
        assert "8 filings" in large_sale_rows[0].content

    def test_research_filters_zero_price_trades(self):
        """Filters out trades with $0 price (non-market)."""
        trades = [
            self._make_trade(tx_type="S-Sale", name="CEO", price=0),
            self._make_trade(tx_type="P-Purchase", name="CFO", price=0),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "No informative insider trades" in result.summary

    def test_research_with_thesis(self):
        """Research uses thesis ID when provided."""
        trades = [
            self._make_trade(
                tx_type="P-Purchase", name="CEO", role="officer: CEO",
                shares=1000, price=100,
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        thesis = Thesis(title="Test thesis")
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST", thesis)

        for e in result.evidence:
            assert e.thesis_id == thesis.id

    def test_research_api_error(self):
        """Handles API errors gracefully."""
        mock_provider = MagicMock()
        mock_provider.get_insider_trades.side_effect = ValueError("API Error")

        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert result.conviction_delta == 0
        assert "failed" in result.summary.lower()
        assert "API Error" in result.summary

    def test_evidence_uses_sentiment_dimension(self):
        """Insider evidence uses SENTIMENT dimension."""
        trades = [
            self._make_trade(
                tx_type="P-Purchase", name="CEO", role="officer: CEO",
                shares=1000, price=100,
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        for e in result.evidence:
            assert e.dimension == ThesisDimension.SENTIMENT

    def test_dimension_scores_sentiment(self):
        """Dimension scores should include sentiment."""
        trades = [
            self._make_trade(
                tx_type="P-Purchase", name="CEO", role="officer: CEO",
                shares=5000, price=100,
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert "sentiment" in result.dimension_scores
        assert result.dimension_scores["sentiment"] > 0

    def test_role_weighting_csuite_highest(self):
        """C-suite trades weighted higher than directors."""
        # CEO purchase
        ceo_trades = [
            self._make_trade(
                tx_type="P-Purchase", name="CEO", role="officer: CEO",
                shares=1000, price=100,
            ),
        ]
        ceo_provider = self._create_mock_provider(trades=ceo_trades)
        ceo_agent = InsiderAgent(fundamentals_provider=ceo_provider)
        ceo_result = ceo_agent.research("TEST")

        # Director purchase (same size)
        dir_trades = [
            self._make_trade(
                tx_type="P-Purchase", name="Director", role="director",
                shares=1000, price=100,
            ),
        ]
        dir_provider = self._create_mock_provider(trades=dir_trades)
        dir_agent = InsiderAgent(fundamentals_provider=dir_provider)
        dir_result = dir_agent.research("TEST")

        # CEO should have higher conviction impact
        assert ceo_result.conviction_delta > dir_result.conviction_delta

    def test_mixed_buys_and_sells(self):
        """Summary reflects mixed activity."""
        trades = [
            self._make_trade(
                tx_type="P-Purchase", name="CEO", role="officer: CEO",
                shares=5000, price=100,
            ),
            self._make_trade(
                tx_type="S-Sale", name="VP", role="officer: VP",
                shares=3000, price=100,
            ),
        ]
        mock_provider = self._create_mock_provider(trades=trades)
        agent = InsiderAgent(fundamentals_provider=mock_provider)
        result = agent.research("TEST")

        assert "buys" in result.summary and "sells" in result.summary
