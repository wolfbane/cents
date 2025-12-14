"""Macro agent - analyzes economic environment using rate of change signals."""

import re
from datetime import date, timedelta
from urllib.request import urlopen
from urllib.error import URLError
import json
import logging

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.config import get_settings
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension

logger = logging.getLogger(__name__)

# Rate of change thresholds (percentage points over lookback period)
FED_RATE_CUT_THRESHOLD = -0.25    # Rate cut of 25bp+ = bullish
FED_RATE_HIKE_THRESHOLD = 0.25   # Rate hike of 25bp+ = bearish

# Unemployment change thresholds
UNEMPLOYMENT_RISING_THRESHOLD = 0.3   # Rising 0.3%+ = bearish
UNEMPLOYMENT_FALLING_THRESHOLD = -0.2  # Falling 0.2%+ = bullish
UNEMPLOYMENT_LOW = 4.5                 # Below 4.5% = healthy labor market

# Yield curve - only flag extreme inversions (reduced weight)
YIELD_CURVE_DEEPLY_INVERTED = -0.5  # Below -0.5% = recession warning

# VIX thresholds
VIX_SPIKE_THRESHOLD = 25      # Above 25 = elevated fear
VIX_COMPLACENT = 13           # Below 13 = complacency


def _sanitize_url(url: str) -> str:
    """Remove API keys from URL for safe logging."""
    return re.sub(r"(api_key=)[^&]+", r"\1***", url, flags=re.IGNORECASE)


class MacroAgent(BaseAgent):
    """Agent that analyzes macroeconomic indicators using rate of change."""

    name = "macro"

    # FRED series IDs for key indicators
    INDICATORS = {
        "DFF": "Fed Funds Rate",
        "T10Y2Y": "10Y-2Y Spread (Yield Curve)",
        "UNRATE": "Unemployment Rate",
        "VIXCLS": "VIX (Volatility Index)",
    }

    # Lookback period for rate of change (in days)
    LOOKBACK_DAYS = 90

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.api_key = settings.fred_api_key

    def research(
        self, symbol: str, thesis: Thesis | None = None, as_of: date | None = None
    ) -> AgentResult:
        """Research macro environment using rate of change signals."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []

        thesis_id = thesis.id if thesis else None

        if not self.api_key:
            return self._research_without_fred(thesis_id, summaries)

        # Fetch current and historical data for each indicator
        for series_id, name in self.INDICATORS.items():
            try:
                current, current_date = self._with_retries(
                    lambda s=series_id: self._fetch_fred_series(s, as_of=as_of)
                )
                if current is None:
                    continue

                # Fetch historical value for rate of change
                lookback_date = (as_of or date.today()) - timedelta(days=self.LOOKBACK_DAYS)
                historical, hist_date = self._with_retries(
                    lambda s=series_id: self._fetch_fred_series(s, as_of=lookback_date)
                )

                # Calculate change
                change = (current - historical) if historical is not None else None

                ev_type, delta, note, metadata = self._interpret_with_change(
                    series_id, current, change
                )
                conviction_delta += delta
                dimension_scores["macro"] = dimension_scores.get("macro", 0) + delta
                if note:
                    summaries.append(note)

                # Build evidence content
                if change is not None:
                    change_str = f"{change:+.2f}" if series_id != "VIXCLS" else f"{change:+.1f}"
                    content = f"{name}: {current:.2f} ({change_str} over 3mo)"
                else:
                    content = f"{name}: {current:.2f}"

                evidence.append(
                    self.create_evidence(
                        thesis_id=thesis_id,
                        content=content,
                        source=f"FRED:{series_id}",
                        evidence_type=ev_type,
                        confidence=0.7,
                        dimension=ThesisDimension.MACRO,
                        metadata={
                            "series": series_id,
                            "value": current,
                            "date": current_date,
                            "change_3mo": change,
                            **metadata,
                        },
                    )
                )
            except RECOVERABLE_EXCEPTIONS as e:
                evidence.append(
                    self.create_evidence(
                        thesis_id=thesis_id,
                        content=f"FRED fetch failed for {series_id}: {e}",
                        source=f"FRED:{series_id}",
                        evidence_type=EvidenceType.NEUTRAL,
                        confidence=0.0,
                        dimension=ThesisDimension.MACRO,
                        metadata={"error": "fred_fetch_failed", "series": series_id},
                    )
                )

        if summaries:
            summary = "Macro: " + "; ".join(summaries)
        else:
            summary = "Macro: Neutral environment"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
            dimension_scores=dimension_scores,
        )

    def _fetch_fred_series(
        self, series_id: str, as_of: date | None = None
    ) -> tuple[float | None, str | None]:
        """Fetch latest value from FRED API."""
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={self.api_key}"
            f"&file_type=json&sort_order=desc&limit=1"
        )
        if as_of:
            url += f"&observation_end={as_of.isoformat()}"
        try:
            with urlopen(url, timeout=10) as response:
                data = json.loads(response.read())
                obs = data.get("observations", [])
                if obs and obs[0].get("value") != ".":
                    return float(obs[0]["value"]), obs[0]["date"]
            return None, None
        except URLError as e:
            logger.warning("FRED API request failed for %s: %s", series_id, e)
            logger.debug("Failed URL: %s", _sanitize_url(url))
            raise

    def _interpret_with_change(
        self, series_id: str, value: float, change: float | None
    ) -> tuple[EvidenceType, float, str | None, dict]:
        """Interpret indicator using rate of change, not just absolute level.

        Returns: (evidence_type, delta, summary_note, metadata)
        """
        metadata = {}

        if series_id == "DFF":  # Fed Funds Rate - focus on direction
            if change is not None:
                if change <= FED_RATE_CUT_THRESHOLD:
                    # Fed cutting rates = bullish for equities
                    metadata["signal"] = "rate_cut"
                    return EvidenceType.SUPPORTING, 4, f"Fed cutting rates ({change:+.2f}%)", metadata
                elif change >= FED_RATE_HIKE_THRESHOLD:
                    # Fed hiking rates = bearish
                    metadata["signal"] = "rate_hike"
                    return EvidenceType.CONTRADICTING, -3, f"Fed hiking rates ({change:+.2f}%)", metadata
            # Rates stable = neutral (already priced in)
            metadata["signal"] = "stable"
            return EvidenceType.NEUTRAL, 0, None, metadata

        elif series_id == "T10Y2Y":  # Yield curve - reduced weight, only flag deep inversion
            if value < YIELD_CURVE_DEEPLY_INVERTED:
                # Deep inversion = recession warning (but lower weight)
                metadata["signal"] = "deep_inversion"
                return EvidenceType.CONTRADICTING, -2, f"Deeply inverted curve ({value:.2f}%)", metadata
            elif change is not None and change > 0.3:
                # Curve steepening = improving outlook
                metadata["signal"] = "steepening"
                return EvidenceType.SUPPORTING, 2, "Yield curve steepening", metadata
            metadata["signal"] = "neutral"
            return EvidenceType.NEUTRAL, 0, None, metadata

        elif series_id == "UNRATE":  # Unemployment - both level and direction matter
            if change is not None:
                if change >= UNEMPLOYMENT_RISING_THRESHOLD:
                    # Rising unemployment = weakening economy = bearish
                    metadata["signal"] = "rising"
                    return EvidenceType.CONTRADICTING, -4, f"Rising unemployment ({change:+.1f}%)", metadata
                elif change <= UNEMPLOYMENT_FALLING_THRESHOLD:
                    # Falling unemployment = strengthening economy = bullish
                    metadata["signal"] = "falling"
                    return EvidenceType.SUPPORTING, 3, f"Falling unemployment ({change:+.1f}%)", metadata

            # Level check: low unemployment = healthy economy
            if value < UNEMPLOYMENT_LOW:
                metadata["signal"] = "low_stable"
                return EvidenceType.SUPPORTING, 2, f"Low unemployment ({value:.1f}%)", metadata
            metadata["signal"] = "neutral"
            return EvidenceType.NEUTRAL, 0, None, metadata

        elif series_id == "VIXCLS":  # VIX - focus on direction and extremes
            if change is not None:
                if value > VIX_SPIKE_THRESHOLD and change < -5:
                    # VIX was high but falling = fear receding = bullish
                    metadata["signal"] = "fear_receding"
                    return EvidenceType.SUPPORTING, 3, f"Fear receding (VIX {value:.0f}, falling)", metadata
                elif value < VIX_COMPLACENT and change > 3:
                    # VIX was low but rising = complacency ending = warning
                    metadata["signal"] = "complacency_ending"
                    return EvidenceType.CONTRADICTING, -2, f"VIX rising from lows ({value:.0f})", metadata

            if value > 30:
                # Extreme fear - often contrarian bullish
                metadata["signal"] = "extreme_fear"
                return EvidenceType.NEUTRAL, 1, f"Elevated VIX ({value:.0f})", metadata
            metadata["signal"] = "neutral"
            return EvidenceType.NEUTRAL, 0, None, metadata

        return EvidenceType.NEUTRAL, 0, None, {}

    def _research_without_fred(
        self, thesis_id: str, summaries: list
    ) -> AgentResult:
        """Provide guidance when FRED API key not configured."""
        evidence = [
            self.create_evidence(
                thesis_id=thesis_id,
                content=(
                    "FRED API key missing - macro data retrieval skipped. "
                    "Set FRED_API_KEY env var for richer macro context."
                ),
                source="system",
                evidence_type=EvidenceType.NEUTRAL,
                confidence=0.0,
                metadata={"error": "missing_fred_api_key"},
            )
        ]
        return AgentResult(
            evidence=evidence,
            conviction_delta=0,
            summary=(
                "WARNING: Macro signals limited - FRED_API_KEY not configured "
                "(get a free key at fred.stlouisfed.org)"
            ),
        )
