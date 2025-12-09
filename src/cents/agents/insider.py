"""Insider trading agent - analyzes SEC Form 4 filings for informative signals."""

from collections import defaultdict
from datetime import datetime, timedelta

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
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

    def research(self, symbol: str, thesis: Thesis | None = None) -> AgentResult:
        """Research insider trading activity for a symbol."""
        evidence = []
        conviction_delta = 0.0
        dimension_scores: dict[str, float] = {}
        summaries = []

        try:
            trades = self._with_retries(
                lambda: self.provider.get_insider_trades(symbol, limit=100)
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
        """
        informative = []
        for t in trades:
            tx_type = t.get("transactionType", "")
            price = t.get("price", 0)

            # Only include P-Purchase or S-Sale with actual price
            if tx_type in INFORMATIVE_TYPES and price and price > 0:
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
                unique_insiders = len(set(t["reportingName"] for t in buy_cluster))
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
                unique_insiders = len(set(t["reportingName"] for t in sell_cluster))
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

    def _analyze_significant_trades(
        self, buys: list[dict], sells: list[dict], thesis_id: str
    ) -> tuple[list, float]:
        """Analyze individual significant trades (large value or C-suite)."""
        evidence = []
        delta = 0.0

        # Analyze purchases by role
        for buy in buys:
            role, weight = _parse_role(buy.get("typeOfOwner", ""))
            value = (buy.get("securitiesTransacted", 0) or 0) * (buy.get("price", 0) or 0)

            if role == "c-suite" and value >= LARGE_PURCHASE_VALUE:
                delta += 4.0 * weight
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Large C-suite purchase: {buy['reportingName']} bought ${value:,.0f}",
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.85,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata={
                        "insider": buy["reportingName"],
                        "role": buy.get("typeOfOwner"),
                        "value": value,
                        "shares": buy.get("securitiesTransacted"),
                    },
                ))
            elif role == "c-suite":
                # Smaller C-suite buy still meaningful
                delta += 2.0 * weight
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"C-suite purchase: {buy['reportingName']} ({role})",
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.75,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata={
                        "insider": buy["reportingName"],
                        "role": buy.get("typeOfOwner"),
                        "value": value,
                    },
                ))
            elif role in ("vp/director", "10% owner"):
                # VP/Director buy is moderately bullish
                delta += 1.5 * weight
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Insider purchase: {buy['reportingName']} ({role})",
                    source="fmp",
                    evidence_type=EvidenceType.SUPPORTING,
                    confidence=0.70,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata={
                        "insider": buy["reportingName"],
                        "role": buy.get("typeOfOwner"),
                        "value": value,
                    },
                ))

        # Check for large non-routine sales (only flag very large ones)
        for sell in sells:
            role, weight = _parse_role(sell.get("typeOfOwner", ""))
            value = (sell.get("securitiesTransacted", 0) or 0) * (sell.get("price", 0) or 0)

            if role == "c-suite" and value >= LARGE_SALE_VALUE:
                # Large C-suite sale is notable but not as negative
                # (could be diversification, estate planning, etc.)
                delta -= 2.0 * weight
                evidence.append(self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"Large C-suite sale: {sell['reportingName']} sold ${value:,.0f}",
                    source="fmp",
                    evidence_type=EvidenceType.CONTRADICTING,
                    confidence=0.70,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata={
                        "insider": sell["reportingName"],
                        "role": sell.get("typeOfOwner"),
                        "value": value,
                    },
                ))

        return evidence, delta
