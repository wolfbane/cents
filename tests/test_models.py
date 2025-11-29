"""Tests for domain models."""

from datetime import date

import pytest

from cents.models import (
    Thesis,
    ThesisStatus,
    Position,
    PositionSide,
    PositionStatus,
    Outcome,
    ThesisAccuracy,
)


class TestThesis:
    """Tests for Thesis model."""

    def test_create_thesis_defaults(self):
        """New thesis has correct defaults."""
        t = Thesis(title="Test thesis")
        assert t.title == "Test thesis"
        assert t.status == ThesisStatus.OPEN
        assert t.conviction == 50.0
        assert t.hypothesis == ""
        assert t.tags == []
        assert len(t.id) == 8

    def test_update_conviction_clamps_high(self):
        """Conviction clamps at 100."""
        t = Thesis(title="Test", conviction=90.0)
        t.update_conviction(20.0)
        assert t.conviction == 100.0

    def test_update_conviction_clamps_low(self):
        """Conviction clamps at 0."""
        t = Thesis(title="Test", conviction=10.0)
        t.update_conviction(-20.0)
        assert t.conviction == 0.0

    def test_update_conviction_normal(self):
        """Normal conviction updates work."""
        t = Thesis(title="Test", conviction=50.0)
        t.update_conviction(15.0)
        assert t.conviction == 65.0

    def test_close(self):
        """Closing thesis sets status."""
        t = Thesis(title="Test")
        t.close()
        assert t.status == ThesisStatus.CLOSED

    def test_invalidate(self):
        """Invalidating thesis sets status."""
        t = Thesis(title="Test")
        t.invalidate()
        assert t.status == ThesisStatus.INVALIDATED


class TestPosition:
    """Tests for Position model."""

    def test_create_position_defaults(self):
        """New position has correct defaults."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=150.0,
            size=100,
        )
        assert p.symbol == "AAPL"
        assert p.side == PositionSide.LONG
        assert p.entry_price == 150.0
        assert p.size == 100
        assert p.status == PositionStatus.OPEN
        assert p.exit_price is None
        assert p.paper is True

    def test_close_position(self):
        """Closing position sets exit price and status."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=150.0,
            size=100,
        )
        p.close(160.0)
        assert p.status == PositionStatus.CLOSED
        assert p.exit_price == 160.0
        assert p.exit_date == date.today()

    def test_pnl_long_profit(self):
        """P&L calculation for profitable long."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=10,
        )
        p.close(110.0)
        assert p.pnl == 100.0  # 10 * (110 - 100)
        assert p.pnl_pct == 10.0  # 10%

    def test_pnl_long_loss(self):
        """P&L calculation for losing long."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=10,
        )
        p.close(90.0)
        assert p.pnl == -100.0  # 10 * (90 - 100)
        assert p.pnl_pct == -10.0

    def test_pnl_short_profit(self):
        """P&L calculation for profitable short."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.SHORT,
            entry_price=100.0,
            size=10,
        )
        p.close(90.0)
        assert p.pnl == 100.0  # 10 * (100 - 90) for short
        assert p.pnl_pct == 10.0

    def test_pnl_short_loss(self):
        """P&L calculation for losing short."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.SHORT,
            entry_price=100.0,
            size=10,
        )
        p.close(110.0)
        assert p.pnl == -100.0
        assert p.pnl_pct == -10.0

    def test_pnl_open_position(self):
        """Open position has no P&L."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=10,
        )
        assert p.pnl is None
        assert p.pnl_pct is None


class TestOutcome:
    """Tests for Outcome model."""

    def test_create_outcome(self):
        """Outcome creation with defaults."""
        o = Outcome(
            position_id="abc123",
            pnl=100.0,
            pnl_pct=10.0,
        )
        assert o.position_id == "abc123"
        assert o.pnl == 100.0
        assert o.pnl_pct == 10.0
        assert o.thesis_accuracy == ThesisAccuracy.UNCLEAR
        assert o.retrospective == ""
        assert o.agent_performance == {}


# --- Edge Case Tests ---


class TestThesisEdgeCases:
    """Edge case tests for Thesis model."""

    def test_conviction_at_exact_zero(self):
        """Conviction can be exactly 0."""
        t = Thesis(title="Test", conviction=0.0)
        assert t.conviction == 0.0
        # Can't go lower
        t.update_conviction(-10.0)
        assert t.conviction == 0.0

    def test_conviction_at_exact_hundred(self):
        """Conviction can be exactly 100."""
        t = Thesis(title="Test", conviction=100.0)
        assert t.conviction == 100.0
        # Can't go higher
        t.update_conviction(10.0)
        assert t.conviction == 100.0

    def test_conviction_from_zero_to_hundred(self):
        """Large update from 0 clamps at 100."""
        t = Thesis(title="Test", conviction=0.0)
        t.update_conviction(150.0)
        assert t.conviction == 100.0

    def test_conviction_from_hundred_to_zero(self):
        """Large negative update from 100 clamps at 0."""
        t = Thesis(title="Test", conviction=100.0)
        t.update_conviction(-150.0)
        assert t.conviction == 0.0

    def test_zero_delta_no_change(self):
        """Zero delta doesn't change conviction."""
        t = Thesis(title="Test", conviction=50.0)
        t.update_conviction(0.0)
        assert t.conviction == 50.0

    def test_negative_conviction_input_clamps(self):
        """Negative initial conviction gets handled."""
        # Note: dataclass doesn't validate, but update_conviction clamps
        t = Thesis(title="Test", conviction=-10.0)
        t.update_conviction(0.0)
        assert t.conviction == 0.0  # Clamped by update

    def test_empty_tags_list(self):
        """Empty tags list is valid."""
        t = Thesis(title="Test", tags=[])
        assert t.tags == []

    def test_updated_at_changes_on_conviction_update(self):
        """Updated timestamp changes when conviction updated."""
        t = Thesis(title="Test")
        original_updated = t.updated_at
        import time
        time.sleep(0.01)  # Small delay
        t.update_conviction(5.0)
        assert t.updated_at > original_updated


class TestPositionEdgeCases:
    """Edge case tests for Position model."""

    def test_zero_size_position(self):
        """Zero size position has zero P&L."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=0,
        )
        p.close(200.0)
        assert p.pnl == 0.0  # 0 * (200 - 100)

    def test_fractional_shares(self):
        """Fractional share positions work."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=0.5,  # Half share
        )
        p.close(110.0)
        assert p.pnl == 5.0  # 0.5 * 10

    def test_very_small_price_change(self):
        """Small price changes calculate correctly."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=100,
        )
        p.close(100.01)
        assert abs(p.pnl - 1.0) < 0.01  # ~$1 profit
        assert abs(p.pnl_pct - 0.01) < 0.001  # ~0.01%

    def test_close_at_same_price(self):
        """Closing at entry price gives zero P&L."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=100,
        )
        p.close(100.0)
        assert p.pnl == 0.0
        assert p.pnl_pct == 0.0

    def test_close_with_custom_date(self):
        """Can close with custom exit date."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=100,
        )
        custom_date = date(2024, 6, 15)
        p.close(110.0, exit_date=custom_date)
        assert p.exit_date == custom_date

    def test_large_position_size(self):
        """Large position sizes calculate correctly."""
        p = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            entry_price=100.0,
            size=1000000,  # 1M shares
        )
        p.close(101.0)
        assert p.pnl == 1000000.0  # $1M profit

    def test_penny_stock_precision(self):
        """Low-priced stocks maintain precision."""
        p = Position(
            symbol="PENNY",
            side=PositionSide.LONG,
            entry_price=0.01,
            size=10000,
        )
        p.close(0.02)
        assert p.pnl == 100.0  # 10000 * 0.01
        assert p.pnl_pct == 100.0  # 100% gain

    def test_short_sell_100_percent_gain(self):
        """Short position with stock going to near-zero."""
        p = Position(
            symbol="FAIL",
            side=PositionSide.SHORT,
            entry_price=100.0,
            size=10,
        )
        p.close(1.0)  # Stock crashed
        assert p.pnl == 990.0  # 10 * (100 - 1)
        assert p.pnl_pct == 99.0  # 99% gain


class TestOutcomeEdgeCases:
    """Edge case tests for Outcome model."""

    def test_zero_pnl(self):
        """Zero P&L outcome is valid."""
        o = Outcome(position_id="test", pnl=0.0, pnl_pct=0.0)
        assert o.pnl == 0.0
        assert o.pnl_pct == 0.0

    def test_large_negative_pnl(self):
        """Large losses are recorded correctly."""
        o = Outcome(position_id="test", pnl=-1000000.0, pnl_pct=-90.0)
        assert o.pnl == -1000000.0
        assert o.pnl_pct == -90.0

    def test_all_accuracy_values(self):
        """All accuracy enum values work."""
        for accuracy in ThesisAccuracy:
            o = Outcome(
                position_id="test",
                pnl=100.0,
                pnl_pct=10.0,
                thesis_accuracy=accuracy,
            )
            assert o.thesis_accuracy == accuracy

    def test_agent_performance_dict(self):
        """Agent performance dictionary stores correctly."""
        o = Outcome(
            position_id="test",
            pnl=100.0,
            pnl_pct=10.0,
            agent_performance={
                "fundamentals": 0.8,
                "technical": 0.6,
                "macro": -0.2,
            },
        )
        assert o.agent_performance["fundamentals"] == 0.8
        assert len(o.agent_performance) == 3
