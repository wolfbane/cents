"""Sentiment agent - analyzes news and market sentiment."""

import re
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import quote
import json

from cents.agents.base import BaseAgent, AgentResult
from cents.config import get_settings
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension


class SentimentAgent(BaseAgent):
    """Agent that analyzes news sentiment for a symbol."""

    name = "sentiment"

    # Simple keyword-based sentiment (fallback when no NLP available)
    POSITIVE_WORDS = {
        "beat", "beats", "exceeds", "exceeded", "surge", "surges", "rally",
        "upgrade", "upgraded", "buy", "bullish", "growth", "profit", "gains",
        "outperform", "record", "breakthrough", "strong", "positive", "optimistic",
    }
    NEGATIVE_WORDS = {
        "miss", "misses", "missed", "fall", "falls", "drop", "drops", "decline",
        "downgrade", "downgraded", "sell", "bearish", "loss", "losses", "weak",
        "underperform", "warning", "concern", "risk", "negative", "pessimistic",
        "lawsuit", "investigation", "recall", "layoffs", "bankruptcy",
    }

    def __init__(self):
        super().__init__()
        settings = get_settings()
        self.news_api_key = settings.news_api_key

    def research(self, symbol: str, thesis: Optional[Thesis] = None) -> AgentResult:
        """Analyze news sentiment for a symbol."""
        thesis_id = thesis.id if thesis else "standalone"

        if not self.news_api_key:
            return self._research_without_api(symbol, thesis_id)

        try:
            articles = self._with_retries(lambda: self._fetch_news(symbol))
            if not articles:
                return AgentResult(
                    evidence=[],
                    conviction_delta=0,
                    summary=f"{symbol}: No recent news found",
                )
            return self._analyze_articles(articles, symbol, thesis_id)
        except Exception as e:
            return AgentResult(
                evidence=[],
                conviction_delta=0,
                summary=f"{symbol}: Failed to fetch news after retries - {e}",
            )

    def _fetch_news(self, symbol: str) -> list[dict]:
        """Fetch news from NewsAPI."""
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={quote(symbol)}&language=en&sortBy=publishedAt&pageSize=10"
            f"&apiKey={self.news_api_key}"
        )
        req = Request(url, headers={"User-Agent": "cents/0.1"})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            return data.get("articles", [])

    def _analyze_articles(
        self, articles: list[dict], symbol: str, thesis_id: str
    ) -> AgentResult:
        """Analyze sentiment of news articles."""
        evidence = []
        total_score = 0
        summaries = []

        for article in articles[:5]:  # Analyze top 5
            title = article.get("title", "")
            description = article.get("description", "") or ""
            source = article.get("source", {}).get("name", "Unknown")
            url = article.get("url", "")

            # Simple keyword sentiment
            text = f"{title} {description}".lower()
            pos_count = sum(1 for w in self.POSITIVE_WORDS if w in text)
            neg_count = sum(1 for w in self.NEGATIVE_WORDS if w in text)

            if pos_count > neg_count:
                ev_type = EvidenceType.SUPPORTING
                score = min(pos_count - neg_count, 3)
            elif neg_count > pos_count:
                ev_type = EvidenceType.CONTRADICTING
                score = -min(neg_count - pos_count, 3)
            else:
                ev_type = EvidenceType.NEUTRAL
                score = 0

            total_score += score

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"{title[:80]}..." if len(title) > 80 else title,
                    source=f"{source}: {url}" if url else source,
                    evidence_type=ev_type,
                    confidence=0.5,  # Keyword analysis is low confidence
                    dimension=ThesisDimension.SENTIMENT,
                    metadata={
                        "positive_words": pos_count,
                        "negative_words": neg_count,
                    },
                )
            )

        # Overall sentiment
        conviction_delta = total_score * 0.5  # Scale down
        dimension_scores = {"sentiment": conviction_delta}

        if total_score > 3:
            summaries.append("Positive news sentiment")
        elif total_score < -3:
            summaries.append("Negative news sentiment")
        else:
            summaries.append("Mixed/neutral news sentiment")

        summary = f"{symbol}: " + "; ".join(summaries) + f" ({len(articles)} articles)"

        return AgentResult(
            evidence=evidence,
            conviction_delta=conviction_delta,
            summary=summary,
            dimension_scores=dimension_scores,
        )

    def _research_without_api(self, symbol: str, thesis_id: str) -> AgentResult:
        """Provide guidance when News API key not configured."""
        evidence = [
            self.create_evidence(
                thesis_id=thesis_id,
                content=(
                    "News API key missing - sentiment scan skipped. "
                    "Set NEWS_API_KEY env var for NewsAPI access."
                ),
                source="system",
                evidence_type=EvidenceType.CONTRADICTING,
                confidence=0.0,
                metadata={"error": "missing_news_api_key"},
            )
        ]
        return AgentResult(
            evidence=evidence,
            conviction_delta=0,
            summary=(
                f"WARNING: {symbol} sentiment skipped - NEWS_API_KEY not configured "
                "(get a free key at newsapi.org)"
            ),
        )
