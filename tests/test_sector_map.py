"""Tests for the sector → SPDR ETF map."""

from unittest.mock import MagicMock, patch

import pytest

from cents.data.providers import FundamentalsData
from cents.factory.sector_map import (
    SECTOR_ETF_MAP,
    TransientSectorLookupError,
    hedge_etf_for,
)


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

    def test_provider_failure_raises_transient_does_not_fall_back_to_spy(self):
        """A raised provider error is transient — surface it; don't hedge with SPY.

        Falling back to SPY for a network-degraded sector lookup silently
        produces a "neutral" thesis hedged against the broad market instead
        of the symbol's actual sector ETF — contaminating the paired-neutral
        cohort with what is really a directional bet.
        """
        def boom():
            raise RuntimeError("FMP down")

        with patch("cents.factory.sector_map.get_settings", return_value=_settings("k")):
            with patch("cents.data.get_fundamentals_provider", side_effect=boom):
                with pytest.raises(TransientSectorLookupError):
                    hedge_etf_for("X")

    def test_degraded_response_with_no_sector_raises_transient(self):
        """FMP responded but profile/ratios failed AND sector is empty → transient.

        Repro of the 2026-05-22 06:35 ET failure: FMP _fetch_json swallows
        URLError and returns None for `profile`, so FundamentalsData comes
        back with sector=None and degraded=True. Without the degraded check,
        the caller can't distinguish "we couldn't reach FMP" from "this
        symbol genuinely has no sector entry" — both used to silently route
        to SPY.
        """
        provider = MagicMock()
        provider.get_fundamentals.return_value = FundamentalsData(
            symbol="BRK.B", sector=None, degraded=True
        )
        with patch("cents.factory.sector_map.get_settings", return_value=_settings("k")):
            with patch("cents.data.get_fundamentals_provider", return_value=provider):
                with pytest.raises(TransientSectorLookupError):
                    hedge_etf_for("BRK.B")

    def test_clean_response_with_no_sector_still_falls_back_to_spy(self):
        """FMP responded cleanly but the symbol has no sector → SPY is legit.

        This is the terminal "we know there is no sector" case (some
        symbols genuinely lack sector metadata in FMP). It is meaningfully
        distinct from the degraded case above and should keep the existing
        SPY-fallback behavior.
        """
        provider = MagicMock()
        provider.get_fundamentals.return_value = FundamentalsData(
            symbol="OBSCURE", sector=None, degraded=False
        )
        with patch("cents.factory.sector_map.get_settings", return_value=_settings("k")):
            with patch("cents.data.get_fundamentals_provider", return_value=provider):
                assert hedge_etf_for("OBSCURE") == "SPY"
