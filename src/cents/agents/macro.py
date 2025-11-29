"""Macro agent - analyzes economic environment."""

import os
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError
import json

from cents.agents.base import BaseAgent, AgentResult
from cents.models import Evidence, EvidenceType, Thesis


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
        self.api_key = os.environ.get("FRED_API_KEY")

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Research macro environment (symbol-agnostic)."""
        evidence = []
        conviction_delta = 0.0
        summaries = []

        thesis_id = thesis.id if thesis else "standalone"

        if not self.api_key:
            # Provide general macro context without FRED
            return self._research_without_fred(thesis_id, summaries)

        # Fetch FRED data
        for series_id, name in self.INDICATORS.items():
            try:
                value, date = self._fetch_fred_series(series_id)
                if value is None:
                    continue

                ev_type, delta, note = self._interpret_indicator(series_id, value)
                conviction_delta += delta
                if note:
                    summaries.append(note)

                evidence.append(
                    self.create_evidence(
                        thesis_id=thesis_id,
                        content=f"{name}: {value:.2f} (as of {date})",
                        source=f"FRED:{series_id}",
                        evidence_type=ev_type,
                        confidence=0.7,
                        metadata={"series": series_id, "value": value, "date": date},
                    )
                )
            except Exception as e:
                continue

        if summaries:
            summary = "Macro: " + "; ".join(summaries)
        else:
            summary = "Macro: No significant signals"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
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
        except (URLError, json.JSONDecodeError, KeyError, ValueError):
            pass
        return None, None

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
                content="FRED API key not configured. Set FRED_API_KEY env var for macro data.",
                source="system",
                evidence_type=EvidenceType.NEUTRAL,
                confidence=0.0,
                metadata={"error": "no_api_key"},
            )
        ]
        return AgentResult(
            evidence=evidence,
            conviction_delta=0,
            summary="Macro: FRED API key not configured (get free key at fred.stlouisfed.org)",
        )
