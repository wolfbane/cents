"""Tests for screener registry + individual strategies."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from cents.screeners import SCREENERS, get_screener
from cents.screeners.growth import GrowthScreener
from cents.screeners.insider_cluster import InsiderClusterScreener
from cents.screeners.mean_reversion import MeanReversionScreener
from cents.screeners.momentum import MomentumScreener
from cents.screeners.value import ValueScreener


# ---- shared fixtures -------------------------------------------------------


def _fundamentals(symbol: str, **kwargs):
    """Build a stub FundamentalsData with overrides."""
    stub = MagicMock()
    stub.symbol = symbol
    stub.pe_ratio = kwargs.get("pe_ratio")
    stub.debt_to_equity = kwargs.get("debt_to_equity")
    stub.return_on_equity = kwargs.get("return_on_equity")
    stub.profit_margin = kwargs.get("profit_margin")
    return stub


def _history(closes: list[float], volumes: list[int] | None = None):
    stub = MagicMock()
    stub.closes = closes
    stub.volumes = volumes or [1_000_000] * len(closes)
    stub.bars = [object()] * len(closes)
    return stub


# ---- registry --------------------------------------------------------------


class TestScreenerRegistry:
    def test_exposes_all_five_strategies(self):
        assert set(SCREENERS) == {
            "value",
            "growth",
            "momentum",
            "mean_reversion",
            "insider_cluster",
        }

    def test_get_screener_returns_instance(self):
        s = get_screener("value")
        assert s.name == "value"

    def test_get_screener_unknown_raises(self):
        with pytest.raises(KeyError):
            get_screener("nonexistent")

    def test_each_describe_returns_rules(self):
        for name, screener in SCREENERS.items():
            payload = screener.describe()
            assert "description" in payload, name
            assert "rules" in payload, name
            assert payload["rules"], name


# ---- value -----------------------------------------------------------------


class TestValueScreener:
    def _provider(self, fundamentals_by_symbol, revenue_pairs):
        """Build a provider mock returning per-symbol fundamentals + revenue history."""
        p = MagicMock()
        p.get_fundamentals.side_effect = lambda sym: fundamentals_by_symbol[sym]

        def fake_fetch_json(endpoint, **kwargs):
            sym = kwargs.get("symbol")
            return revenue_pairs.get(sym)

        p._fetch_json.side_effect = fake_fetch_json
        return p

    def test_passes_when_all_filters_met(self):
        provider = self._provider(
            {"GOOD": _fundamentals("GOOD", pe_ratio=10.0, debt_to_equity=0.3, return_on_equity=0.20)},
            {"GOOD": [{"revenue": 120}, {"revenue": 100}]},
        )
        s = ValueScreener(fundamentals_provider=provider)
        assert s.screen(["GOOD"]) == ["GOOD"]

    def test_fails_when_pe_too_high(self):
        provider = self._provider(
            {"BAD": _fundamentals("BAD", pe_ratio=40.0, debt_to_equity=0.3, return_on_equity=0.20)},
            {"BAD": [{"revenue": 120}, {"revenue": 100}]},
        )
        assert ValueScreener(fundamentals_provider=provider).screen(["BAD"]) == []

    def test_fails_on_declining_revenue(self):
        provider = self._provider(
            {"DEC": _fundamentals("DEC", pe_ratio=10.0, debt_to_equity=0.3, return_on_equity=0.20)},
            {"DEC": [{"revenue": 80}, {"revenue": 100}]},
        )
        assert ValueScreener(fundamentals_provider=provider).screen(["DEC"]) == []

    def test_ranks_cheaper_pe_first(self):
        provider = self._provider(
            {
                "A": _fundamentals("A", pe_ratio=8.0, debt_to_equity=0.3, return_on_equity=0.20),
                "B": _fundamentals("B", pe_ratio=12.0, debt_to_equity=0.3, return_on_equity=0.20),
            },
            {
                "A": [{"revenue": 120}, {"revenue": 100}],
                "B": [{"revenue": 120}, {"revenue": 100}],
            },
        )
        assert ValueScreener(fundamentals_provider=provider).screen(["B", "A"]) == ["A", "B"]

    def test_empty_candidate_list_returns_empty(self):
        s = ValueScreener(fundamentals_provider=MagicMock())
        assert s.screen([]) == []

    def test_graceful_per_symbol_failure(self):
        provider = MagicMock()

        def get_fundamentals(sym):
            if sym == "FAIL":
                raise RuntimeError("boom")
            return _fundamentals(sym, pe_ratio=10.0, debt_to_equity=0.3, return_on_equity=0.20)

        provider.get_fundamentals.side_effect = get_fundamentals
        provider._fetch_json.side_effect = lambda endpoint, **kwargs: [{"revenue": 120}, {"revenue": 100}]
        s = ValueScreener(fundamentals_provider=provider)
        assert s.screen(["FAIL", "OK"]) == ["OK"]

    def test_default_limit_truncates(self):
        provider = MagicMock()
        provider.get_fundamentals.side_effect = lambda sym: _fundamentals(
            sym, pe_ratio=10.0, debt_to_equity=0.3, return_on_equity=0.20
        )
        provider._fetch_json.side_effect = lambda endpoint, **kwargs: [{"revenue": 120}, {"revenue": 100}]
        syms = [f"S{i:03d}" for i in range(50)]
        result = ValueScreener(fundamentals_provider=provider, limit=10).screen(syms)
        assert len(result) == 10


# ---- growth ----------------------------------------------------------------


class TestGrowthScreener:
    def _provider(self, history_by_symbol):
        p = MagicMock()
        p._fetch_json.side_effect = lambda endpoint, **kwargs: history_by_symbol.get(kwargs.get("symbol"))
        return p

    def test_passes_with_strong_cagr_and_margins(self):
        # 100 → 200 over 3 years = ~26% CAGR; gross margins climbing 40 → 50.
        rows = [
            {"revenue": 200, "grossProfit": 100},
            {"revenue": 160, "grossProfit": 76},
            {"revenue": 130, "grossProfit": 58},
            {"revenue": 100, "grossProfit": 40},
        ]
        p = self._provider({"WIN": rows})
        assert GrowthScreener(fundamentals_provider=p).screen(["WIN"]) == ["WIN"]

    def test_fails_when_cagr_below_threshold(self):
        rows = [
            {"revenue": 110, "grossProfit": 55},
            {"revenue": 108, "grossProfit": 54},
            {"revenue": 104, "grossProfit": 52},
            {"revenue": 100, "grossProfit": 50},
        ]
        p = self._provider({"SLOW": rows})
        assert GrowthScreener(fundamentals_provider=p).screen(["SLOW"]) == []

    def test_fails_when_margin_compresses(self):
        rows = [
            {"revenue": 200, "grossProfit": 90},   # 45%
            {"revenue": 160, "grossProfit": 80},   # 50%
            {"revenue": 130, "grossProfit": 65},   # 50%
            {"revenue": 100, "grossProfit": 50},   # 50%
        ]
        p = self._provider({"COMPRESS": rows})
        assert GrowthScreener(fundamentals_provider=p).screen(["COMPRESS"]) == []

    def test_short_history_skipped(self):
        p = self._provider({"NEW": [{"revenue": 100, "grossProfit": 50}]})
        assert GrowthScreener(fundamentals_provider=p).screen(["NEW"]) == []

    def test_empty_candidate_list(self):
        assert GrowthScreener(fundamentals_provider=MagicMock()).screen([]) == []


# ---- momentum --------------------------------------------------------------


class TestMomentumScreener:
    def _provider(self, history_by_symbol):
        p = MagicMock()
        p.get_history.side_effect = lambda sym, **kwargs: history_by_symbol[sym]
        return p

    def test_passes_when_trend_volume_3m_change(self):
        # Linear uptrend across 80 days; volume jumps in last 5 days.
        closes = [100 + i * 0.5 for i in range(80)]
        volumes = [1_000_000] * 75 + [5_000_000] * 5
        p = self._provider({"UP": _history(closes, volumes)})
        assert MomentumScreener(price_provider=p).screen(["UP"]) == ["UP"]

    def test_fails_below_50d_ma(self):
        # Downtrend — latest close < MA50.
        closes = [200 - i * 0.5 for i in range(80)]
        volumes = [1_000_000] * 75 + [5_000_000] * 5
        p = self._provider({"DOWN": _history(closes, volumes)})
        assert MomentumScreener(price_provider=p).screen(["DOWN"]) == []

    def test_fails_when_volume_quiet(self):
        closes = [100 + i * 0.5 for i in range(80)]
        volumes = [1_000_000] * 80  # no volume surge
        p = self._provider({"QUIET": _history(closes, volumes)})
        assert MomentumScreener(price_provider=p).screen(["QUIET"]) == []

    def test_empty_candidate_list(self):
        assert MomentumScreener(price_provider=MagicMock()).screen([]) == []

    def test_graceful_per_symbol_failure(self):
        p = MagicMock()

        def get_history(sym, **kwargs):
            if sym == "FAIL":
                raise RuntimeError("boom")
            closes = [100 + i * 0.5 for i in range(80)]
            volumes = [1_000_000] * 75 + [5_000_000] * 5
            return _history(closes, volumes)

        p.get_history.side_effect = get_history
        assert MomentumScreener(price_provider=p).screen(["FAIL", "OK"]) == ["OK"]


# ---- mean reversion --------------------------------------------------------


class TestMeanReversionScreener:
    def test_passes_oversold_with_quality(self):
        # 30 closes ending in a sharp drop → RSI < 30.
        closes = [100.0] * 15 + [100 - i * 1.0 for i in range(15)]
        price = MagicMock()
        price.get_history.side_effect = lambda sym, **kwargs: _history(closes)
        fund = MagicMock()
        fund.get_fundamentals.side_effect = lambda sym: _fundamentals(sym, return_on_equity=0.18)
        s = MeanReversionScreener(price_provider=price, fundamentals_provider=fund)
        assert s.screen(["DROP"]) == ["DROP"]

    def test_fails_when_not_oversold(self):
        # Flat tape → RSI ~ 50.
        closes = [100.0] * 30
        price = MagicMock()
        price.get_history.side_effect = lambda sym, **kwargs: _history(closes)
        fund = MagicMock()
        fund.get_fundamentals.side_effect = lambda sym: _fundamentals(sym, return_on_equity=0.20)
        s = MeanReversionScreener(price_provider=price, fundamentals_provider=fund)
        assert s.screen(["FLAT"]) == []

    def test_fails_when_quality_gate_misses(self):
        closes = [100.0] * 15 + [100 - i * 1.0 for i in range(15)]
        price = MagicMock()
        price.get_history.side_effect = lambda sym, **kwargs: _history(closes)
        fund = MagicMock()
        fund.get_fundamentals.side_effect = lambda sym: _fundamentals(sym, return_on_equity=-0.05)
        s = MeanReversionScreener(price_provider=price, fundamentals_provider=fund)
        assert s.screen(["LOWQ"]) == []

    def test_empty_candidate_list(self):
        s = MeanReversionScreener(price_provider=MagicMock(), fundamentals_provider=MagicMock())
        assert s.screen([]) == []


# ---- insider cluster -------------------------------------------------------


class TestInsiderClusterScreener:
    def _trade(self, name: str, tx: str, days_ago: int):
        when = (datetime(2025, 5, 1) - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        return {
            "transactionDate": when,
            "transactionType": tx,
            "reportingName": name,
            "price": 100.0,
            "securitiesTransacted": 100,
        }

    def test_passes_three_distinct_buyers_no_sells(self):
        provider = MagicMock()
        provider.get_insider_trades.return_value = [
            self._trade("Alice", "P-Purchase", 5),
            self._trade("Bob", "P-Purchase", 8),
            self._trade("Carol", "P-Purchase", 10),
        ]
        s = InsiderClusterScreener(
            fundamentals_provider=provider,
            now=datetime(2025, 5, 1),
        )
        assert s.screen(["CLUSTER"]) == ["CLUSTER"]

    def test_passes_when_buys_outweigh_offsetting_sell(self):
        """A scheduled 10b5-1 sale shouldn't kill a clear net-buy cluster."""
        provider = MagicMock()
        provider.get_insider_trades.return_value = [
            self._trade("Alice", "P-Purchase", 5),     # $10k buy
            self._trade("Bob", "P-Purchase", 8),       # $10k buy
            self._trade("Carol", "P-Purchase", 10),    # $10k buy
            self._trade("Dave", "S-Sale", 7),          # $10k sell (10b5-1 plan)
        ]
        s = InsiderClusterScreener(
            fundamentals_provider=provider, now=datetime(2025, 5, 1)
        )
        # 3 buyers, $30k buys vs $10k sells → net +$20k. Passes.
        assert s.screen(["NET_POS"]) == ["NET_POS"]

    def test_fails_when_sells_outweigh_buys(self):
        """Net-negative dollar flow disqualifies even if 3+ insiders bought."""
        provider = MagicMock()
        big_sell = self._trade("Dave", "S-Sale", 5)
        big_sell["securitiesTransacted"] = 1000  # $100k sell
        provider.get_insider_trades.return_value = [
            self._trade("Alice", "P-Purchase", 5),   # $10k
            self._trade("Bob", "P-Purchase", 8),     # $10k
            self._trade("Carol", "P-Purchase", 10),  # $10k
            big_sell,                                # $100k sell
        ]
        s = InsiderClusterScreener(
            fundamentals_provider=provider, now=datetime(2025, 5, 1)
        )
        # $30k buys vs $100k sells → net -$70k. Fails.
        assert s.screen(["NET_NEG"]) == []

    def test_fails_below_min_buyers(self):
        provider = MagicMock()
        provider.get_insider_trades.return_value = [
            self._trade("Alice", "P-Purchase", 5),
            self._trade("Bob", "P-Purchase", 8),
        ]
        s = InsiderClusterScreener(
            fundamentals_provider=provider, now=datetime(2025, 5, 1)
        )
        assert s.screen(["FEW"]) == []

    def test_ignores_buys_outside_window(self):
        provider = MagicMock()
        provider.get_insider_trades.return_value = [
            self._trade("Alice", "P-Purchase", 5),
            self._trade("Bob", "P-Purchase", 8),
            self._trade("Carol", "P-Purchase", 90),  # outside 30d window
        ]
        s = InsiderClusterScreener(
            fundamentals_provider=provider, now=datetime(2025, 5, 1)
        )
        assert s.screen(["STALE"]) == []

    def test_ranks_by_distinct_buyer_count(self):
        provider = MagicMock()

        def get_trades(symbol, **kwargs):
            if symbol == "MORE":
                return [
                    self._trade(n, "P-Purchase", 5)
                    for n in ("A", "B", "C", "D")
                ]
            return [
                self._trade(n, "P-Purchase", 5) for n in ("A", "B", "C")
            ]

        provider.get_insider_trades.side_effect = get_trades
        s = InsiderClusterScreener(
            fundamentals_provider=provider, now=datetime(2025, 5, 1)
        )
        assert s.screen(["FEW", "MORE"]) == ["MORE", "FEW"]
