"""Macro agent - analyzes economic environment."""

import re
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError
import json
import logging

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.config import get_settings
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension

logger = logging.getLogger(__name__)


def _sanitize_url(url: str) -> str:
    """Remove API keys from URL for safe logging."""
    return re.sub(r"(api_key=)[^&]+", r"\1***", url, flags=re.IGNORECASE)


class MacroAgent(BaseAgent):
    """Agent that analyzes macroeconomic indicators."""

    name = "macro"

    # FRED series IDs for key indicators
    INDICATORS = {
        "DFF": "Fed Funds Rate",
        "T10Y2Y": "10Y-2Y Spread (Yield Curve)",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "CPI (Inflation)",
        "VIXCLS": "VIX (Volatility Index)",
    }

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.api_key = settings.fred_api_key

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Research macro environment (symbol-agnostic)."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []

        thesis_id = thesis.id if thesis else "standalone"

        if not self.api_key:
            # Provide general macro context without FRED
            return self._research_without_fred(thesis_id, summaries)

        # Fetch FRED data
        for series_id, name in self.INDICATORS.items():
            try:
                value, date = self._with_retries(lambda s=series_id: self._fetch_fred_series(s))
                if value is None:
                    continue

                ev_type, delta, note = self._interpret_indicator(series_id, value)
                conviction_delta += delta
                dimension_scores["macro"] = dimension_scores.get("macro", 0) + delta
                if note:
                    summaries.append(note)

                evidence.append(
                    self.create_evidence(
                        thesis_id=thesis_id,
                        content=f"{name}: {value:.2f} (as of {date})",
                        source=f"FRED:{series_id}",
                        evidence_type=ev_type,
                        confidence=0.7,
                        dimension=ThesisDimension.MACRO,
                        metadata={"series": series_id, "value": value, "date": date},
                    )
                )
            except RECOVERABLE_EXCEPTIONS as e:
                evidence.append(
                    self.create_evidence(
                        thesis_id=thesis_id,
                        content=(
                            f"FRED fetch failed for {series_id} after retries: {e}"),
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
            summary = "Macro: No significant signals"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
            dimension_scores=dimension_scores,
        )

    def _fetch_fred_series(self, series_id: str) -> tuple[Optional[float], Optional[str]]:
        """Fetch latest value from FRED API."""
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={self.api_key}"
            f"&file_type=json&sort_order=desc&limit=1"
        )
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

    def _interpret_indicator(
        self, series_id: str, value: float
    ) -> tuple[EvidenceType, float, Optional[str]]:
        """Interpret indicator value for equity investing."""
        if series_id == "DFF":  # Fed Funds Rate
            if value > 5:
                return EvidenceType.CONTRADICTING, -3, f"High rates ({value:.2f}%)"
            elif value < 2:
                return EvidenceType.SUPPORTING, 3, f"Low rates ({value:.2f}%)"
            return EvidenceType.NEUTRAL, 0, None

        elif series_id == "T10Y2Y":  # Yield curve
            if value < 0:
                return EvidenceType.CONTRADICTING, -5, "Inverted yield curve"
            elif value > 1:
                return EvidenceType.SUPPORTING, 2, "Steep yield curve"
            return EvidenceType.NEUTRAL, 0, None

        elif series_id == "UNRATE":  # Unemployment
            if value > 6:
                return EvidenceType.CONTRADICTING, -2, f"High unemployment ({value:.1f}%)"
            elif value < 4:
                return EvidenceType.SUPPORTING, 2, f"Low unemployment ({value:.1f}%)"
            return EvidenceType.NEUTRAL, 0, None

        elif series_id == "VIXCLS":  # VIX
            if value > 30:
                return EvidenceType.CONTRADICTING, -3, f"High VIX ({value:.0f})"
            elif value < 15:
                return EvidenceType.SUPPORTING, 2, f"Low VIX ({value:.0f})"
            return EvidenceType.NEUTRAL, 0, None

        return EvidenceType.NEUTRAL, 0, None

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
