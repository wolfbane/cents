"""Insider trading agent - analyzes SEC Form 4 filings for informative signals."""

import math
from collections import defaultdict
from datetime import date, datetime, timedelta

from cents.agents.base import (
    MAX_CONVICTION_DELTA,
    BaseAgent,
    AgentResult,
    RECOVERABLE_EXCEPTIONS,
    sanitize_metadata_string,
)
from cents.models import EvidenceType, Thesis, ThesisDimension

# Transaction types to include (informative open market trades)
INFORMATIVE_TYPES = {"S-Sale", "P-Purchase"}

# Transaction types to exclude (routine/compensation-related)
ROUTINE_TYPES = {"G-Gift", "M-Exempt", "F-InKind", "A-Award"}

# Role keywords for weighting
C_SUITE_KEYWORDS = {"ceo", "cfo", "coo", "chief"}
VP_KEYWORDS = {"vp", "vice president", "director"}
OWNER_KEYWORDS = {"10%", "owner"}

# Thresholds
LARGE_PURCHASE_VALUE = 500_000  # $500k
LARGE_SALE_VALUE = 1_000_000    # $1M
CLUSTER_WINDOW_DAYS = 30
# Window for "discretionary overlay" on top of a 10b5-1 program — if a meaningful
# share of the trades in an aggregate cluster in the most recent fortnight, it's
# more likely an opportunistic decision than a scheduled tranche.
DISCRETIONARY_WINDOW_DAYS = 14
DISCRETIONARY_CONCENTRATION_PCT = 0.70

# FMP exposes the Form 4 10b5-1 indicator under several names depending on
# endpoint/plan; accept any of them. Truthy values are interpreted as "plan
# trade." When absent the suffix degrades to "(across N filings)" without
# claiming 10b5-1 from the count alone.
_RULE_10B5_1_FIELDS = ("rule10b5-1", "rule10b5_1", "isRule10b5_1", "ruleSection10b5_1")


def _trade_is_10b5_1(trade: dict) -> bool:
    for field in _RULE_10B5_1_FIELDS:
        val = trade.get(field)
        if val is True or (isinstance(val, str) and val.strip().lower() in {"true", "1", "y", "yes"}):
            return True
    return False


def _parse_trade_date(trade: dict) -> date | None:
    raw = trade.get("transactionDate")
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _scaled_role_delta(
    value: float, weight: float, sign: int, threshold: float, base: float
) -> float:
    """Log-scaled delta so a $200M sale isn't indistinguishable from a $1.5M
    sale once both clear the threshold. Baseline at the threshold is
    ``base × weight``; each 10× above adds another ``base × weight``. Capped
    at the per-agent ``MAX_CONVICTION_DELTA``.
    """
    scale = max(0.0, math.log10(value / threshold)) if value > 0 else 0.0
    raw = sign * weight * (base + base * scale)
    return max(-MAX_CONVICTION_DELTA, min(MAX_CONVICTION_DELTA, raw))


def _get_fundamentals_provider():
    """Lazy import to avoid circular dependencies."""
    from cents.data.fmp import get_fundamentals_provider
    return get_fundamentals_provider()


def _parse_role(type_of_owner: str) -> tuple[str, float]:
    """Parse insider role and return (role_name, weight).

    Args:
        type_of_owner: FMP field like "officer: CEO" or "director"

    Returns:
        Tuple of (simplified role, weight multiplier)
    """
    if not type_of_owner:
        return "unknown", 0.5

    lower = type_of_owner.lower()

    if any(kw in lower for kw in C_SUITE_KEYWORDS):
        return "c-suite", 1.0
    elif any(kw in lower for kw in OWNER_KEYWORDS):
        return "10% owner", 0.8
    elif any(kw in lower for kw in VP_KEYWORDS):
        return "vp/director", 0.7
    else:
        return "other", 0.5


class InsiderAgent(BaseAgent):
    """Agent that analyzes insider trading patterns from SEC Form 4 filings."""

    name = "insider"

    def __init__(self, fundamentals_provider=None):
        """
        Initialize insider agent.

        Args:
            fundamentals_provider: FMP provider instance (defaults to singleton)
        """
        super().__init__()
        self._provider = fundamentals_provider

    @property
    def provider(self):
        """Get fundamentals data provider, creating default if needed."""
        if self._provider is None:
            self._provider = _get_fundamentals_provider()
        return self._provider

    def research(
        self, symbol: str, thesis: Thesis | None = None, as_of: date | None = None
    ) -> AgentResult:
        """Research insider trading activity for a symbol."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []

        try:
            trades = self._with_retries(
                lambda: self.provider.get_insider_trades(symbol, limit=100, as_of=as_of)
            )
        except RECOVERABLE_EXCEPTIONS as e:
            return self._error_result(symbol, e)

        thesis_id = thesis.id if thesis else None

        # Filter to informative trades only
        informative = self._filter_informative_trades(trades)

        if not informative:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"{symbol}: No informative insider trades in recent filings",
            )

        # Separate buys and sells
        buys = [t for t in informative if t["transactionType"] == "P-Purchase"]
        sells = [t for t in informative if t["transactionType"] == "S-Sale"]

        # Analyze cluster patterns
        cluster_ev, cluster_delta = self._analyze_clusters(buys, sells, thesis_id)
        evidence.extend(cluster_ev)
        conviction_delta += cluster_delta
        if cluster_delta > 3:
            summaries.append("Cluster buying")
        elif cluster_delta < -3:
            summaries.append("Cluster selling")

        # Analyze significant individual trades
        trade_ev, trade_delta = self._analyze_significant_trades(
            buys, sells, thesis_id
        )
        evidence.extend(trade_ev)
        conviction_delta += trade_delta

        # Summarize activity
        if buys and not sells:
            summaries.append(f"{len(buys)} insider buys")
        elif sells and not buys:
            summaries.append(f"{len(sells)} insider sells")
        elif buys and sells:
            summaries.append(f"{len(buys)} buys, {len(sells)} sells")

        # All insider signals go to sentiment dimension
        dimension_scores["sentiment"] = conviction_delta

        if summaries:
            summary = f"{symbol}: " + "; ".join(summaries)
        else:
            summary = f"{symbol}: Mixed insider activity"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
            dimension_scores=dimension_scores,
        )

    def _filter_informative_trades(self, trades: list[dict]) -> list[dict]:
        """Filter to only informative open market trades.

        Excludes:
        - Gifts, awards, option exercises, tax withholding
        - Trades with $0 price (non-market)
        - Records with missing ``reportingName`` (FMP data-quality dropouts;
          filtered here so cluster detection and per-insider aggregation see
          the same set instead of disagreeing on what counts as a trade).
        """
        informative = []
        for t in trades:
            tx_type = t.get("transactionType", "")
            price = t.get("price", 0)
            name = t.get("reportingName") or ""

            # Only include P-Purchase or S-Sale with actual price + named filer
            if tx_type in INFORMATIVE_TYPES and price and price > 0 and name:
                informative.append(t)

        return informative

    def _analyze_clusters(
        self, buys: list[dict], sells: list[dict], thesis_id: str
    ) -> tuple[list, float]:
        """Detect cluster buying/selling patterns.

        Multiple insiders trading in same direction within 30 days
        is a strong signal.
        """
        evidence = []
        delta = 0.0

        # Analyze buy clusters
        if len(buys) >= 2:
            buy_cluster = self._find_cluster(buys)
            if buy_cluster:
                # Key on CIK (with normalized-name fallback) so the same
                # person under two name spellings doesn't inflate the count.
                unique_insiders = len({self._insider_dedup_key(t) for t in buy_cluster})
                if unique_insiders >= 3:
                    delta += 5.0
                    evidence.append(self.create_evidence(
                        thesis_id=thesis_id,
                        content=f"Cluster buying: {unique_insiders} insiders purchased within 30 days",
                        source="fmp",
                        evidence_type=EvidenceType.SUPPORTING,
                        confidence=0.85,
                        dimension=ThesisDimension.SENTIMENT,
                        metadata={
                            "pattern": "cluster_buy",
                            "insider_count": unique_insiders,
                            "trades": len(buy_cluster),
                        },
                    ))
                elif unique_insiders >= 2:
                    delta += 3.0
                    evidence.append(self.create_evidence(
                        thesis_id=thesis_id,
                        content=f"Multiple insiders buying: {unique_insiders} executives purchased recently",
                        source="fmp",
                        evidence_type=EvidenceType.SUPPORTING,
                        confidence=0.80,
                        dimension=ThesisDimension.SENTIMENT,
                        metadata={
                            "pattern": "multiple_buy",
                            "insider_count": unique_insiders,
                        },
                    ))

        # Analyze sell clusters
        if len(sells) >= 3:
            sell_cluster = self._find_cluster(sells)
            if sell_cluster:
                unique_insiders = len({self._insider_dedup_key(t) for t in sell_cluster})
                if unique_insiders >= 3:
                    delta -= 3.0
                    evidence.append(self.create_evidence(
                        thesis_id=thesis_id,
                        content=f"Multiple insiders selling: {unique_insiders} executives sold within 30 days",
                        source="fmp",
                        evidence_type=EvidenceType.CONTRADICTING,
                        confidence=0.75,
                        dimension=ThesisDimension.SENTIMENT,
                        metadata={
                            "pattern": "cluster_sell",
                            "insider_count": unique_insiders,
                        },
                    ))

        return evidence, delta

    def _find_cluster(self, trades: list[dict]) -> list[dict]:
        """Find trades within CLUSTER_WINDOW_DAYS of each other."""
        if not trades:
            return []

        # Sort by date
        sorted_trades = sorted(
            trades,
            key=lambda t: t.get("transactionDate", ""),
            reverse=True
        )

        # Get most recent trade date
        try:
            recent_date = datetime.strptime(
                sorted_trades[0]["transactionDate"], "%Y-%m-%d"
            )
        except (ValueError, KeyError):
            return []

        # Find all trades within window
        cutoff = recent_date - timedelta(days=CLUSTER_WINDOW_DAYS)
        cluster = []
        for t in sorted_trades:
            try:
                trade_date = datetime.strptime(t["transactionDate"], "%Y-%m-%d")
                if trade_date >= cutoff:
                    cluster.append(t)
            except (ValueError, KeyError):
                continue

        return cluster

    def _insider_dedup_key(self, trade: dict) -> str:
        """Stable per-insider identifier.

        Prefer FMP's ``reportingCik`` — the SEC's immutable insider ID — and
        fall back to a normalised reportingName when CIK is absent (older
        records, partial-payload responses). The normalisation collapses
        casing/punctuation differences so that "BIALECKI ANDREW" and
        "Bialecki, Andrew J" don't dedupe as two different people.
        """
        cik = trade.get("reportingCik") or trade.get("cik")
        if cik:
            return f"cik:{cik}"
        name = trade.get("reportingName") or ""
        normalised = "".join(c.upper() for c in name if c.isalnum())
        return f"name:{normalised}"

    def _aggregate_by_insider(self, trades: list[dict]) -> list[dict]:
        """Aggregate trades by a stable insider key so a 10b5-1 program (one
        decision, many filings) becomes a single row rather than N rows that
        each get individually weighted into conviction_delta.

        Each synthetic bucket carries enough information to render an
        attribution-honest evidence row:

        - ``is_10b5_1``: any trade in the bucket carries FMP's Form 4
          rule10b5-1 flag (the SEC-mandated checkbox). Without this the
          suffix used to claim "(likely 10b5-1)" from filing count alone.
        - ``min_date`` / ``max_date``: lets the suffix show the date range
          so a 8-month plan is distinguishable from an 8-day burst.
        - ``discretionary_overlay``: when more than DISCRETIONARY_CONCENTRATION_PCT
          of the trades happen in the trailing DISCRETIONARY_WINDOW_DAYS, mark
          the bucket as showing a discretionary overlay on top of any plan.
        - ``_role_weight``: highest role weight observed — CFO→CEO
          transitions land on the senior role, not whichever filing FMP
          returned first.
        """
        if not trades:
            return []
        agg: dict[str, dict] = {}
        for t in trades:
            # reportingName is guaranteed non-empty by _filter_informative_trades.
            # Sanitize the FMP-supplied strings here so all downstream evidence
            # content interpolations are safe — Form 4 reportingName is filer-
            # self-typed and an injection vector if it ever flows raw into a
            # prompt the orchestrator's LLM consumes.
            name = sanitize_metadata_string(t["reportingName"])
            type_of_owner = sanitize_metadata_string(t.get("typeOfOwner", ""))
            _, weight = _parse_role(type_of_owner)
            key = self._insider_dedup_key(t)
            value = (t.get("securitiesTransacted", 0) or 0) * (t.get("price", 0) or 0)
            shares = t.get("securitiesTransacted", 0) or 0
            tx_date = _parse_trade_date(t)
            is_plan = _trade_is_10b5_1(t)
            bucket = agg.setdefault(
                key,
                {
                    "reportingName": name,
                    "typeOfOwner": type_of_owner,
                    "_role_weight": weight,
                    "value": 0.0,
                    "shares": 0.0,
                    "trade_count": 0,
                    "is_10b5_1": False,
                    "min_date": None,
                    "max_date": None,
                    "_dates": [],
                },
            )
            # Promote to the higher-weight role if a later filing shows one
            # — role transitions happen mid-window (CFO → CEO, etc).
            if weight > bucket["_role_weight"]:
                bucket["_role_weight"] = weight
                bucket["typeOfOwner"] = type_of_owner
            bucket["value"] += value
            bucket["shares"] += shares
            bucket["trade_count"] += 1
            bucket["is_10b5_1"] = bucket["is_10b5_1"] or is_plan
            if tx_date is not None:
                bucket["_dates"].append(tx_date)
                if bucket["min_date"] is None or tx_date < bucket["min_date"]:
                    bucket["min_date"] = tx_date
                if bucket["max_date"] is None or tx_date > bucket["max_date"]:
                    bucket["max_date"] = tx_date

        # Post-pass: compute the discretionary-overlay flag now that all dates
        # are in. "Discretionary overlay" = a sufficient share of trades fall
        # inside the trailing window relative to the bucket's most recent
        # filing, which an analyst would read as opportunistic on top of the
        # baseline plan cadence.
        for bucket in agg.values():
            dates = bucket.pop("_dates")
            if bucket["max_date"] is None or len(dates) < 2:
                bucket["discretionary_overlay"] = False
                continue
            cutoff = bucket["max_date"] - timedelta(days=DISCRETIONARY_WINDOW_DAYS)
            recent = sum(1 for d in dates if d >= cutoff)
            bucket["discretionary_overlay"] = (
                recent / len(dates) > DISCRETIONARY_CONCENTRATION_PCT
            )
        return list(agg.values())

    def _format_aggregate_suffix(self, bucket: dict, is_sale: bool) -> str:
        """Build the trailing "(across N filings, dates, plan-flag, overlay)"
        annotation for an aggregated evidence row.

        Empty string when the bucket only holds one filing — the original
        evidence row already names the date implicitly.
        """
        count = bucket["trade_count"]
        if count <= 1:
            return ""
        parts: list[str] = [f"across {count} filings"]
        if bucket.get("min_date") and bucket.get("max_date"):
            parts.append(f"{bucket['min_date']}–{bucket['max_date']}")
        if bucket.get("is_10b5_1"):
            parts.append("10b5-1 plan")
        if bucket.get("discretionary_overlay"):
            parts.append(f"recent {DISCRETIONARY_WINDOW_DAYS}d burst")
        return f" ({', '.join(parts)})"

    def _analyze_significant_trades(
        self, buys: list[dict], sells: list[dict], thesis_id: str
    ) -> tuple[list, float]:
        """Analyze individual significant trades (large value or C-suite).

        Trades are aggregated per insider before scoring so that a single
        person executing a multi-tranche 10b5-1 program produces one
        Evidence row, not N. Otherwise one CEO's tax-planned sales can
        spam the orchestrator's contradicting count with rows that all
        trace to one decision.
        """
        evidence = []
        delta = 0.0

        # Analyze purchases by role (aggregated per insider)
        for buy in self._aggregate_by_insider(buys):
            role, weight = _parse_role(buy.get("typeOfOwner", ""))
            value = buy["value"]
            suffix = self._format_aggregate_suffix(buy, is_sale=False)
            metadata_base = {
                "insider": buy["reportingName"],
                "role": buy.get("typeOfOwner"),
                "value": value,
                "shares": buy["shares"],
                "trade_count": buy["trade_count"],
                "is_10b5_1": buy["is_10b5_1"],
                "min_date": str(buy["min_date"]) if buy["min_date"] else None,
                "max_date": str(buy["max_date"]) if buy["max_date"] else None,
                "discretionary_overlay": buy["discretionary_overlay"],
            }

            if role == "c-suite" and value >= LARGE_PURCHASE_VALUE:
                d = _scaled_role_delta(value, weight, +1, LARGE_PURCHASE_VALUE, 4.0)
                delta += d
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=(
                        f"Large C-suite purchase: {buy['reportingName']} "
                        f"bought ${value:,.0f}{suffix}"
                    ),
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.85,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata=metadata_base,
                ))
            elif role == "c-suite":
                # Smaller C-suite buy still meaningful
                delta += 2.0 * weight
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"C-suite purchase: {buy['reportingName']} ({role}){suffix}",
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.75,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata=metadata_base,
                ))
            elif role in ("vp/director", "10% owner"):
                # VP/Director buy is moderately bullish
                delta += 1.5 * weight
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Insider purchase: {buy['reportingName']} ({role}){suffix}",
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.70,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata=metadata_base,
                ))

        # Check for large non-routine sales. The scaled delta means a $200M
        # CEO sale carries more weight than a $1.5M one; the suffix names the
        # 10b5-1 plan flag when FMP exposes it (gated on the actual Form 4
        # checkbox, not the filing count) and surfaces a discretionary-overlay
        # note when a meaningful share of activity sits in the trailing 14d.
        for sell in self._aggregate_by_insider(sells):
            role, weight = _parse_role(sell.get("typeOfOwner", ""))
            value = sell["value"]
            suffix = self._format_aggregate_suffix(sell, is_sale=True)
            metadata_base = {
                "insider": sell["reportingName"],
                "role": sell.get("typeOfOwner"),
                "value": value,
                "trade_count": sell["trade_count"],
                "is_10b5_1": sell["is_10b5_1"],
                "min_date": str(sell["min_date"]) if sell["min_date"] else None,
                "max_date": str(sell["max_date"]) if sell["max_date"] else None,
                "discretionary_overlay": sell["discretionary_overlay"],
            }

            if role == "c-suite" and value >= LARGE_SALE_VALUE:
                # Large C-suite sale is notable but not as negative
                # (could be diversification, estate planning, etc.)
                delta += _scaled_role_delta(value, weight, -1, LARGE_SALE_VALUE, 2.0)
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=(
                        f"Large C-suite sale: {sell['reportingName']} "
                        f"sold ${value:,.0f}{suffix}"
                    ),
                    source="fmp",
                    evidence_type=EvidenceType.CONTRADICTING,
                    confidence=0.70,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata=metadata_base,
                ))

        return evidence, delta
