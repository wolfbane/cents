"""Macro agent - analyzes economic environment using rate of change signals."""

import re
from datetime import date, timedelta
from urllib.request import urlopen
from urllib.error import URLError
import json
import logging

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.cache import cached_request
from cents.config import get_settings
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension

logger = logging.getLogger(__name__)

MACRO_SIGNAL_CONFIG = {
    "DFF": {
        "thresholds": {
            "cut": -0.25,   # Rate cut of 25bp+ = bullish
            "hike": 0.25,   # Rate hike of 25bp+ = bearish
        },
        "rules": [
            {
                "signal": "rate_cut",
                "condition": lambda value, change, t: change is not None and change <= t["cut"],
                "evidence": EvidenceType.SUPPORTING,
                "delta": 4,
                "note": lambda value, change: f"Fed cutting rates ({change:+.2f}%)",
            },
            {
                "signal": "rate_hike",
                "condition": lambda value, change, t: change is not None and change >= t["hike"],
                "evidence": EvidenceType.CONTRADICTING,
                "delta": -3,
                "note": lambda value, change: f"Fed hiking rates ({change:+.2f}%)",
            },
        ],
        "fallback": {
            "signal": "stable",
            "evidence": EvidenceType.NEUTRAL,
            "delta": 0,
            "note": None,
        },
    },
    "T10Y2Y": {
        "thresholds": {
            "deep_inversion": -0.5,  # Below -0.5% = recession warning
            "steepening": 0.3,
        },
        "rules": [
            {
                "signal": "deep_inversion",
                "condition": lambda value, change, t: value < t["deep_inversion"],
                "evidence": EvidenceType.CONTRADICTING,
                "delta": -2,
                "note": lambda value, change: f"Deeply inverted curve ({value:.2f}%)",
                "level_only": True,
            },
            {
                "signal": "steepening",
                "condition": lambda value, change, t: change is not None and change > t["steepening"],
                "evidence": EvidenceType.SUPPORTING,
                "delta": 2,
                "note": lambda value, change: "Yield curve steepening",
            },
        ],
        "fallback": {
            "signal": "neutral",
            "evidence": EvidenceType.NEUTRAL,
            "delta": 0,
            "note": None,
        },
    },
    "UNRATE": {
        "thresholds": {
            "rising": 0.3,   # Rising 0.3%+ = bearish
            "falling": -0.2,  # Falling 0.2%+ = bullish
            "low": 4.5,       # Below 4.5% = healthy labor market
        },
        "rules": [
            {
                "signal": "rising",
                "condition": lambda value, change, t: change is not None and change >= t["rising"],
                "evidence": EvidenceType.CONTRADICTING,
                "delta": -4,
                "note": lambda value, change: f"Rising unemployment ({change:+.1f}%)",
            },
            {
                "signal": "falling",
                "condition": lambda value, change, t: change is not None and change <= t["falling"],
                "evidence": EvidenceType.SUPPORTING,
                "delta": 3,
                "note": lambda value, change: f"Falling unemployment ({change:+.1f}%)",
            },
            {
                "signal": "low_level",
                "condition": lambda value, change, t: value < t["low"],
                "evidence": EvidenceType.SUPPORTING,
                "delta": 2,
                "note": lambda value, change: f"Low unemployment level ({value:.1f}%, level-based signal)",
                "level_only": True,
            },
        ],
        "fallback": {
            "signal": "neutral",
            "evidence": EvidenceType.NEUTRAL,
            "delta": 0,
            "note": None,
        },
    },
    "VIXCLS": {
        "thresholds": {
            "spike": 25,      # Above 25 = elevated fear
            "complacent": 13, # Below 13 = complacency
            "extreme": 30,
        },
        "rules": [
            {
                "signal": "fear_receding",
                "condition": lambda value, change, t: change is not None and value > t["spike"] and change < -5,
                "evidence": EvidenceType.SUPPORTING,
                "delta": 3,
                "note": lambda value, change: f"Fear receding (VIX {value:.0f}, falling)",
            },
            {
                "signal": "complacency_ending",
                "condition": lambda value, change, t: change is not None and value < t["complacent"] and change > 3,
                "evidence": EvidenceType.CONTRADICTING,
                "delta": -2,
                "note": lambda value, change: f"VIX rising from lows ({value:.0f})",
            },
            {
                "signal": "extreme_fear",
                "condition": lambda value, change, t: value > t["extreme"],
                "evidence": EvidenceType.NEUTRAL,
                "delta": 1,
                "note": lambda value, change: f"Elevated VIX ({value:.0f})",
                "level_only": True,
            },
        ],
        "fallback": {
            "signal": "neutral",
            "evidence": EvidenceType.NEUTRAL,
            "delta": 0,
            "note": None,
        },
    },
}


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

                # Build evidence content. When the fired rule is level-only
                # (the rule dict marks itself with `"level_only": True`),
                # suppress the decorative 3mo change figure so a reader
                # doesn't mistake it for the driver.
                if change is not None and not metadata.get("level_only"):
                    change_str = f"{change:+.2f}" if series_id != "VIXCLS" else f"{change:+.1f}"
                    content = f"{name}: {current:.2f} ({change_str} over 3mo)"
                else:
                    content = f"{name}: {current:.2f}"
                # Append the fired-rule reason so the [+]/[-]/[~] tag is self-
                # explanatory — without this, readers see the change figure and
                # assume that's what drove the signal, even when the rule fired
                # on absolute level.
                if note:
                    content = f"{content} — {note}"

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
        """Fetch latest value from FRED API with caching for historical data."""
        # Build cache key params
        cache_params = {
            "series_id": series_id,
            "as_of": as_of.isoformat() if as_of else "latest",
        }

        def do_fetch():
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
                        # Return tuple as list for JSON serialization
                        return [float(obs[0]["value"]), obs[0]["date"]]
                return None
            except URLError as e:
                logger.warning("FRED API request failed for %s: %s", series_id, e)
                logger.debug("Failed URL: %s", _sanitize_url(url))
                raise

        # Only cache historical data (when as_of is in the past)
        use_cache = as_of is not None and as_of < date.today()

        if use_cache:
            result = cached_request("fred", "observations", cache_params, do_fetch)
        else:
            result = do_fetch()

        if result:
            return result[0], result[1]
        return None, None

    def _interpret_with_change(
        self, series_id: str, value: float, change: float | None
    ) -> tuple[EvidenceType, float, str | None, dict]:
        """Interpret indicator using rate of change, not just absolute level.

        Returns: (evidence_type, delta, summary_note, metadata)
        """
        config = MACRO_SIGNAL_CONFIG.get(series_id)
        if not config:
            return EvidenceType.NEUTRAL, 0, None, {}

        thresholds = config["thresholds"]
        for rule in config["rules"]:
            if rule["condition"](value, change, thresholds):
                metadata = {
                    "signal": rule["signal"],
                    "level_only": rule.get("level_only", False),
                }
                note = rule["note"](value, change)
                return rule["evidence"], rule["delta"], note, metadata

        fallback = config["fallback"]
        metadata = {"signal": fallback["signal"]}
        return fallback["evidence"], fallback["delta"], fallback["note"], metadata

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
