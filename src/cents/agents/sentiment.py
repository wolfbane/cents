"""Sentiment agent - analyzes news and market sentiment."""

import json
from typing import Optional
from urllib.request import urlopen, Request
from urllib.parse import quote

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.config import get_settings
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension


class SentimentAgent(BaseAgent):
    """Agent that analyzes news sentiment for a symbol."""

    name = "sentiment"

    # Simple keyword-based sentiment with negation detection
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
    # Words that negate sentiment within a 3-word window
    NEGATION_WORDS = {
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "fail", "fails", "failed", "failing",
        "unlikely", "unable", "without", "lack", "lacks", "lacking",
        "doubt", "doubts", "doubted", "doubtful",
        "hardly", "barely", "scarcely",
    }
    NEGATION_WINDOW = 3  # Check this many words before the sentiment word

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
        except RECOVERABLE_EXCEPTIONS as e:
            return self._error_result(symbol, e)

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

    def _is_negated(self, words: list[str], keyword_index: int) -> bool:
        """Check if a keyword is negated (before or immediately after)."""
        # Check before the keyword (within window)
        start = max(0, keyword_index - self.NEGATION_WINDOW)
        for i in range(start, keyword_index):
            if words[i] in self.NEGATION_WORDS:
                return True

        # Check immediately after (1 word) for patterns like "upgrade unlikely"
        if keyword_index + 1 < len(words):
            if words[keyword_index + 1] in self.NEGATION_WORDS:
                return True

        return False

    def _count_sentiment_words(self, text: str) -> tuple[int, int]:
        """Count positive and negative sentiment words, accounting for negation.

        Returns (positive_count, negative_count) after flipping negated sentiments.
        """
        # Tokenize: split on non-alphanumeric, keep only words
        words = [w for w in text.lower().replace("'", " ").split() if w.isalpha()]

        pos_count = 0
        neg_count = 0

        for i, word in enumerate(words):
            is_negated = self._is_negated(words, i)

            if word in self.POSITIVE_WORDS:
                if is_negated:
                    neg_count += 1  # "not bullish" → negative
                else:
                    pos_count += 1
            elif word in self.NEGATIVE_WORDS:
                if is_negated:
                    pos_count += 1  # "not bearish" → positive
                else:
                    neg_count += 1

        return pos_count, neg_count

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

            # Keyword sentiment with negation detection
            text = f"{title} {description}"
            pos_count, neg_count = self._count_sentiment_words(text)

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
                evidence_type=EvidenceType.NEUTRAL,
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
