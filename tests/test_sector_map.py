"""Tests for the sector → SPDR ETF map."""

from unittest.mock import MagicMock, patch

from cents.data.providers import FundamentalsData
from cents.factory.sector_map import SECTOR_ETF_MAP, hedge_etf_for


def _settings(api_key: str | None = "key"):
    s = MagicMock()
    s.fmp_api_key = api_key
    return s


class TestSectorEtfMap:
    def test_known_sectors_map_correctly(self):
        assert SECTOR_ETF_MAP["Technology"] == "XLK"
        assert SECTOR_ETF_MAP["Financial Services"] == "XLF"
        assert SECTOR_ETF_MAP["Energy"] == "XLE"
        assert SECTOR_ETF_MAP["Healthcare"] == "XLV"


class TestHedgeEtfFor:
    def test_falls_back_to_spy_without_fmp_key(self):
        with patch("cents.factory.sector_map.get_settings", return_value=_settings(None)):
            assert hedge_etf_for("NVDA") == "SPY"

    def test_maps_via_fmp_sector(self):
        provider = MagicMock()
        provider.get_fundamentals.return_value = FundamentalsData(
            symbol="NVDA", sector="Technology"
        )
        with patch("cents.factory.sector_map.get_settings", return_value=_settings("k")):
            with patch("cents.data.get_fundamentals_provider", return_value=provider):
                assert hedge_etf_for("NVDA") == "XLK"

    def test_unknown_sector_falls_back_to_spy(self):
        provider = MagicMock()
        provider.get_fundamentals.return_value = FundamentalsData(
            symbol="WEIRD", sector="Aerospace & Defense"
        )
        with patch("cents.factory.sector_map.get_settings", return_value=_settings("k")):
            with patch("cents.data.get_fundamentals_provider", return_value=provider):
                assert hedge_etf_for("WEIRD") == "SPY"

    def test_provider_failure_falls_back_to_spy(self):
        def boom():
            raise RuntimeError("FMP down")

        with patch("cents.factory.sector_map.get_settings", return_value=_settings("k")):
            with patch("cents.data.get_fundamentals_provider", side_effect=boom):
                assert hedge_etf_for("X") == "SPY"
