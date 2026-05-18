"""Regression tests for the new cents.finance package."""

from __future__ import annotations

import math
import pytest

from cents.finance import (
    Cost,
    apply_close_cost,
    apply_open_cost,
    average_daily_volume,
    beta_match_ratio,
    check_kill_switch,
    compute_drawdown,
    estimate_beta,
    passes_borrow_gate,
    passes_liquidity_gate,
    realized_vol_pct,
    stop_hit,
    target_hit,
    vol_scaled_shares,
)
from cents.finance.portfolio import DrawdownState
from cents.models import PositionSide


# ---- sizing (cents-wiz) -------------------------------------------------


class TestRealizedVol:
    def test_constant_series_returns_zero_vol(self):
        closes = [100.0] * 30
        v = realized_vol_pct(closes, lookback=20)
        assert v == pytest.approx(0.0)

    def test_alternating_pct_moves_produce_positive_vol(self):
        # 1% up then 1% down repeated.
        closes = [100.0]
        for _ in range(25):
            closes.append(closes[-1] * 1.01)
            closes.append(closes[-1] * 0.99)
        v = realized_vol_pct(closes, lookback=20)
        assert v is not None and v > 0

    def test_insufficient_data_returns_none(self):
        assert realized_vol_pct([100.0, 101.0], lookback=20) is None


class TestVolScaledShares:
    def test_higher_vol_gets_smaller_position(self):
        low_v, _ = vol_scaled_shares(
            price=100.0, annual_vol_pct=15.0, budget_usd=100_000,
            target_vol_pct_per_position=0.5, max_position_pct=5.0,
        )
        high_v, _ = vol_scaled_shares(
            price=100.0, annual_vol_pct=60.0, budget_usd=100_000,
            target_vol_pct_per_position=0.5, max_position_pct=5.0,
        )
        # 4x vol → ~1/4 the shares (subject to caps).
        assert high_v < low_v

    def test_max_position_cap_engages(self):
        # Very low vol → enormous target shares; cap should engage.
        shares, method = vol_scaled_shares(
            price=10.0, annual_vol_pct=1.0, budget_usd=100_000,
            target_vol_pct_per_position=0.5, max_position_pct=5.0,
        )
        # Max dollar = $5,000 → 500 shares.
        assert shares == pytest.approx(500.0)
        assert method == "max_capped"

    def test_no_vol_falls_back_to_equal_dollar(self):
        shares, method = vol_scaled_shares(
            price=100.0, annual_vol_pct=None, budget_usd=100_000,
            target_vol_pct_per_position=0.5, max_position_pct=5.0,
            fallback_position_usd=3_000.0,
        )
        assert shares == pytest.approx(30.0)
        assert method == "equal_dollar"


# ---- costs (cents-5s7) --------------------------------------------------


class TestCosts:
    def test_open_cost_components(self):
        c = apply_open_cost(
            shares=100, price=50.0,
            commission_per_share_usd=0.005, slippage_bps=5,
        )
        assert c.commission == pytest.approx(0.5)
        # Slippage: 100 * 50 * 0.0005 = 2.50
        assert c.slippage == pytest.approx(2.5)
        assert c.total == pytest.approx(3.0)

    def test_close_cost_includes_short_borrow(self):
        c = apply_close_cost(
            side="short", shares=100, entry_price=50.0, exit_price=48.0,
            days_held=30,
            commission_per_share_usd=0.0, slippage_bps=0,
            borrow_rate_pa_pct=5.0,
        )
        # avg_notional ~ 100 * 49 = 4900; 5%pa * 30/365 = ~0.00411
        # 4900 * 0.00411 ≈ 20.13
        assert c.borrow_carry == pytest.approx(20.137, rel=0.01)

    def test_long_has_no_borrow_carry(self):
        c = apply_close_cost(
            side="long", shares=100, entry_price=50.0, exit_price=55.0,
            days_held=30,
            commission_per_share_usd=0.0, slippage_bps=0, borrow_rate_pa_pct=5.0,
        )
        assert c.borrow_carry == 0.0

    def test_gap_penalty_applies_on_exit_notional(self):
        c = apply_close_cost(
            side="long", shares=100, entry_price=100.0, exit_price=90.0,
            days_held=10,
            commission_per_share_usd=0.0, slippage_bps=0, borrow_rate_pa_pct=0.0,
            gap_penalty_bps=25,
        )
        # 100 * 90 * 0.0025 = 22.5
        assert c.gap_penalty == pytest.approx(22.5)


# ---- portfolio drawdown + kill switch (cents-59r) ----------------------


class _PriceProviderStub:
    def __init__(self, prices: dict[str, float]):
        self._p = prices

    def get_latest_price(self, symbol: str):
        return self._p.get(symbol)


def _make_pos(symbol, side_value, entry_price, size, pnl=None, thesis_id=None):
    """Minimal duck-typed Position for the compute_drawdown tests."""
    class _Pos:
        pass
    p = _Pos()
    p.symbol = symbol
    p.entry_price = entry_price
    p.size = size
    p.side = type("S", (), {"value": side_value})()
    p.pnl = pnl
    p.thesis_id = thesis_id
    return p


class TestKillSwitch:
    def test_no_positions_means_no_drawdown(self):
        state = compute_drawdown(
            open_positions=[], closed_today=[],
            price_provider=_PriceProviderStub({}),
            budget_usd=100_000.0,
        )
        assert state.gate_open is True
        assert state.unrealized_drawdown_pct == 0.0

    def test_unrealized_loss_trips_gate(self):
        # Long 100 @ 50 → cost basis 5000; mark @ 40 → unrealized PnL -1000.
        # Against a $5k budget that is -20% — trips a 10% cap.
        pos = _make_pos("AAA", "long", entry_price=50.0, size=100)
        state = compute_drawdown(
            open_positions=[pos], closed_today=[],
            price_provider=_PriceProviderStub({"AAA": 40.0}),
            budget_usd=5_000.0,
        )
        gated = check_kill_switch(
            state, max_portfolio_drawdown_pct=10.0, max_daily_loss_pct=3.0,
        )
        assert gated.gate_open is False
        assert "drawdown" in (gated.gate_reason or "")

    def test_small_loss_against_large_budget_does_not_trip(self):
        """Regression: a -$50 loss on a $1k position against a $100k budget
        is -0.05% DD, not -5% — must NOT trip a 3% daily cap.

        Pre-fix the denominator was open cost basis, so the gate became more
        sensitive as the book shrank; this asserts the new budget-based math.
        """
        # Long 10 @ 100 (cost basis $1,000); mark @ 95 → unrealized -$50.
        pos = _make_pos("AAA", "long", entry_price=100.0, size=10)
        state = compute_drawdown(
            open_positions=[pos], closed_today=[],
            price_provider=_PriceProviderStub({"AAA": 95.0}),
            budget_usd=100_000.0,
        )
        assert state.unrealized_drawdown_pct == pytest.approx(-0.05)
        gated = check_kill_switch(
            state, max_portfolio_drawdown_pct=3.0, max_daily_loss_pct=3.0,
        )
        assert gated.gate_open is True
        assert gated.gate_reason is None

    def test_paired_legs_are_netted_in_cost_basis(self):
        """A neutral-cohort thesis owns long + short on the same thesis_id —
        their cost basis should net, not double-count."""
        long_leg = _make_pos("AAA", "long", entry_price=50.0, size=100, thesis_id="t1")
        short_leg = _make_pos("XLK", "short", entry_price=100.0, size=45, thesis_id="t1")
        state = compute_drawdown(
            open_positions=[long_leg, short_leg], closed_today=[],
            price_provider=_PriceProviderStub({"AAA": 50.0, "XLK": 100.0}),
            budget_usd=100_000.0,
        )
        # Gross would be $5,000 + $4,500 = $9,500.
        # Netted: abs(5000 - 4500) = $500.
        assert state.open_cost_basis_usd == pytest.approx(500.0)

    def test_daily_realized_loss_trips_gate(self):
        # closed today with -$1000 realized loss against a $10k budget → -10%.
        closed = _make_pos("BBB", "long", entry_price=100.0, size=10, pnl=-1000.0)
        state = compute_drawdown(
            open_positions=[], closed_today=[closed],
            price_provider=_PriceProviderStub({}),
            budget_usd=10_000.0,
        )
        gated = check_kill_switch(
            state, max_portfolio_drawdown_pct=50.0, max_daily_loss_pct=3.0,
        )
        assert gated.gate_open is False
        assert "realized" in (gated.gate_reason or "")


# ---- liquidity + borrow gates (cents-hz0) ------------------------------


class TestLiquidity:
    def test_passes_when_adv_well_above_required(self):
        closes = [100.0] * 30
        volumes = [1_000_000] * 30
        chk = passes_liquidity_gate(
            symbol="AAA", position_size_usd=10_000,
            closes=closes, volumes=volumes,
            adv_multiple=50,
        )
        # ADV ≈ $100M, required = 500k. Passes.
        assert chk.passes is True

    def test_fails_when_adv_below_required(self):
        closes = [10.0] * 30
        volumes = [10_000] * 30  # ADV = $100k
        chk = passes_liquidity_gate(
            symbol="ZZZ", position_size_usd=10_000,
            closes=closes, volumes=volumes,
            adv_multiple=50,  # required = $500k
        )
        assert chk.passes is False
        assert "ADV" in (chk.reason or "")

    def test_no_history_fails_closed(self):
        chk = passes_liquidity_gate(
            symbol="???", position_size_usd=10_000,
            closes=None, volumes=None,
            adv_multiple=50,
        )
        assert chk.passes is False


class TestBorrow:
    def test_long_always_passes_with_zero_borrow(self):
        chk = passes_borrow_gate(symbol="AAA", side="long", default_borrow_rate_pa_pct=3.0)
        assert chk.passes is True
        assert chk.borrow_rate_pa_pct == 0.0

    def test_short_records_synthetic_rate_for_paper(self):
        chk = passes_borrow_gate(symbol="AAA", side="short", default_borrow_rate_pa_pct=3.0)
        assert chk.passes is True
        assert chk.borrow_rate_pa_pct == 3.0


# ---- hedging (cents-t8r) ------------------------------------------------


class TestHedging:
    def test_beta_above_one_returns_larger_hedge(self):
        # underlying moves 1.5x hedge in the same direction
        hedge = [100.0]
        under = [100.0]
        for i in range(70):
            hedge.append(hedge[-1] * (1 + 0.001 * (1 if i % 2 == 0 else -1)))
            under.append(under[-1] * (1 + 0.0015 * (1 if i % 2 == 0 else -1)))
        beta = estimate_beta(under, hedge, lookback=60)
        assert beta is not None and beta > 1.2

    def test_beta_clamping(self):
        # Even with a wild estimate, the ratio clamps to [min, max].
        assert beta_match_ratio(beta=10.0, default_beta=1.0, max_beta=3.0) == 3.0
        assert beta_match_ratio(beta=0.01, default_beta=1.0, min_beta=0.25) == 0.25

    def test_none_beta_uses_default(self):
        assert beta_match_ratio(beta=None, default_beta=1.0) == 1.0

    def test_nan_beta_uses_default(self):
        assert beta_match_ratio(beta=float("nan"), default_beta=1.0) == 1.0

    def test_low_r_squared_returns_none(self):
        """When R² of the underlying-on-hedge regression is below the
        threshold the relationship is too weak to hedge with, so we refuse
        the estimate entirely (rather than clamp it)."""
        import random

        # Independent random walks: the regression should explain almost
        # nothing, R² ≈ 0.
        rng = random.Random(7)
        hedge = [100.0]
        under = [100.0]
        for _ in range(80):
            hedge.append(hedge[-1] * (1 + rng.uniform(-0.01, 0.01)))
            under.append(under[-1] * (1 + rng.uniform(-0.01, 0.01)))
        # Without the gate, a beta would be returned — just probably small.
        unguarded = estimate_beta(under, hedge, lookback=60)
        assert unguarded is not None
        # With a strict R² gate it must return None.
        assert estimate_beta(under, hedge, lookback=60, min_r_squared=0.5) is None

    def test_high_r_squared_passes_gate(self):
        """A tightly-coupled series passes a strict R² gate."""
        # underlying = 1.5 * hedge with tiny additive noise — R² near 1.
        import random
        rng = random.Random(13)
        hedge = [100.0]
        under = [100.0]
        for _ in range(80):
            step = rng.uniform(-0.01, 0.01)
            hedge.append(hedge[-1] * (1 + step))
            # Co-moving with small idiosyncratic noise.
            under.append(under[-1] * (1 + 1.5 * step + rng.uniform(-1e-5, 1e-5)))
        beta = estimate_beta(under, hedge, lookback=60, min_r_squared=0.5)
        assert beta is not None and beta > 1.0


class TestTriggers:
    """Direction-aware target/stop predicates (cents/finance/triggers.py)."""

    def test_long_target_hit_when_price_above(self):
        assert target_hit(PositionSide.LONG, price=110.0, target=105.0)
        assert not target_hit(PositionSide.LONG, price=100.0, target=105.0)

    def test_long_stop_hit_when_price_below(self):
        assert stop_hit(PositionSide.LONG, price=95.0, stop=100.0)
        assert not stop_hit(PositionSide.LONG, price=105.0, stop=100.0)

    def test_short_target_hit_when_price_below(self):
        """Short wins when price drops — target sits below entry."""
        assert target_hit(PositionSide.SHORT, price=90.0, target=95.0)
        assert not target_hit(PositionSide.SHORT, price=100.0, target=95.0)

    def test_short_stop_hit_when_price_rises(self):
        """Short loses when price rises — stop sits above entry."""
        assert stop_hit(PositionSide.SHORT, price=110.0, stop=105.0)
        assert not stop_hit(PositionSide.SHORT, price=100.0, stop=105.0)

    def test_none_thresholds_never_hit(self):
        assert not target_hit(PositionSide.LONG, price=110.0, target=None)
        assert not stop_hit(PositionSide.SHORT, price=110.0, stop=None)
