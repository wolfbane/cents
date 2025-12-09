"""Sentiment agent - analyzes news and market sentiment."""

import json
import logging
from urllib.request import urlopen, Request
from urllib.parse import quote

from cents.agents.base import BaseAgent, AgentResult, RECOVERABLE_EXCEPTIONS
from cents.config import get_settings
from cents.models import Evidence, EvidenceType, Thesis, ThesisDimension


logger = logging.getLogger(__name__)

# Module-level cache for LLM article scores (keyed by URL)
_article_score_cache: dict[str, dict] = {}


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

    def __init__(self, anthropic_client=None):
        super().__init__()
        settings = get_settings()
        self.news_api_key = settings.news_api_key
        self.anthropic_api_key = settings.anthropic_api_key
        self._timeout = settings.default_api_timeout
        self._anthropic_client = anthropic_client

    def _get_anthropic_client(self):
        """Get or create anthropic client."""
        if self._anthropic_client is not None:
            return self._anthropic_client
        if not self.anthropic_api_key:
            return None
        try:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            return self._anthropic_client
        except ImportError:
            logger.warning("anthropic package not installed")
            return None

    def research(self, symbol: str, thesis: Thesis | None = None) -> AgentResult:
        """Analyze news sentiment for a symbol."""
        thesis_id = thesis.id if thesis else None

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
            return self._analyze_articles(articles, symbol, thesis, thesis_id)
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
        with urlopen(req, timeout=self._timeout) as response:
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

    def _score_article(self, article: dict) -> tuple[EvidenceType, int, float, dict]:
        """Score an article using keyword-based sentiment (fallback).

        Returns (evidence_type, score, confidence, metadata).
        """
        title = article.get("title", "")
        description = article.get("description", "") or ""

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

        return ev_type, score, 0.5, {
            "positive_words": pos_count,
            "negative_words": neg_count,
            "scoring_method": "keyword",
        }

    def _filter_relevant_articles(
        self, articles: list[dict], symbol: str, thesis: Thesis | None
    ) -> list[dict]:
        """Use LLM to filter relevant articles. Returns relevant articles."""
        client = self._get_anthropic_client()
        if not client or len(articles) == 0:
            return articles[:5]

        # Build article list for prompt
        article_list = []
        for i, article in enumerate(articles[:10]):
            title = article.get("title", "No title")
            snippet = (article.get("description", "") or "")[:200]
            article_list.append(f"{i}. {title}\n   {snippet}")

        hypothesis = thesis.hypothesis if thesis else "General investment analysis"

        prompt = f"""Given these news articles about {symbol}, which are relevant to evaluating this investment thesis?
Thesis: {hypothesis}

Articles:
{chr(10).join(article_list)}

Return only the indices (0-based) of relevant articles, one per line. Filter out:
- PyPI/npm package releases
- Job postings
- Unrelated companies with similar names
- Press releases with no real news

Return 3-5 relevant indices, or fewer if less are relevant."""

        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse indices from response
            indices = []
            for line in text.split("\n"):
                line = line.strip()
                # Extract digits from each line
                for word in line.split():
                    if word.isdigit():
                        idx = int(word)
                        if 0 <= idx < len(articles):
                            indices.append(idx)
                        break

            if indices:
                return [articles[i] for i in indices[:5]]
        except Exception as e:
            logger.warning(f"LLM filter failed: {e}")

        # Fallback to first 5
        return articles[:5]

    def _score_with_llm(
        self, article: dict, symbol: str, thesis: Thesis | None
    ) -> tuple[EvidenceType, float, float, dict]:
        """Score an article using LLM. Returns (evidence_type, score, confidence, metadata)."""
        url = article.get("url", "")

        # Check cache
        if url and url in _article_score_cache:
            cached = _article_score_cache[url]
            return (
                cached["evidence_type"],
                cached["score"],
                cached["confidence"],
                cached["metadata"],
            )

        client = self._get_anthropic_client()
        if not client:
            return self._score_article(article)

        title = article.get("title", "No title")
        snippet = (article.get("description", "") or "")[:500]
        hypothesis = thesis.hypothesis if thesis else "General investment"

        prompt = f"""Score the sentiment of this news for the investment thesis.
Symbol: {symbol}
Thesis: {hypothesis}

Article: {title} - {snippet}

Return a JSON object: {{"score": <-1 to 1>, "reasoning": "<brief explanation>"}}
Score meaning: -1 = very bearish for thesis, 0 = neutral, +1 = very bullish for thesis."""

        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON from response (may have text before/after)
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                score = float(result.get("score", 0))
                reasoning = result.get("reasoning", "")

                # Clamp score to [-1, 1]
                score = max(-1.0, min(1.0, score))

                if score > 0.2:
                    ev_type = EvidenceType.SUPPORTING
                elif score < -0.2:
                    ev_type = EvidenceType.CONTRADICTING
                else:
                    ev_type = EvidenceType.NEUTRAL

                # Higher confidence for LLM scoring (0.7-0.9 based on score magnitude)
                confidence = 0.7 + 0.2 * abs(score)

                metadata = {
                    "llm_score": score,
                    "reasoning": reasoning,
                    "scoring_method": "llm",
                }

                # Cache result
                if url:
                    _article_score_cache[url] = {
                        "evidence_type": ev_type,
                        "score": score,
                        "confidence": confidence,
                        "metadata": metadata,
                    }

                return ev_type, score, confidence, metadata

        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")

        # Fallback to keyword scoring
        return self._score_article(article)

    def _analyze_articles(
        self, articles: list[dict], symbol: str, thesis: Thesis | None, thesis_id: str
    ) -> AgentResult:
        """Analyze sentiment of news articles."""
        evidence = []
        total_score = 0.0
        summaries = []

        # Use LLM to filter relevant articles if available
        client = self._get_anthropic_client()
        if client:
            filtered_articles = self._filter_relevant_articles(articles, symbol, thesis)
        else:
            filtered_articles = articles[:5]

        for article in filtered_articles:
            title = article.get("title", "")
            source = article.get("source", {}).get("name", "Unknown")
            url = article.get("url", "")

            # Use LLM scoring if available, otherwise keyword scoring
            if client:
                ev_type, score, confidence, metadata = self._score_with_llm(
                    article, symbol, thesis
                )
                # Scale LLM score (-1 to 1) to match keyword scale (-3 to 3)
                score = score * 3
            else:
                ev_type, score, confidence, metadata = self._score_article(article)

            total_score += score

            evidence.append(
                self.create_evidence(
                    thesis_id=thesis_id,
                    content=f"{title[:80]}..." if len(title) > 80 else title,
                    source=f"{source}: {url}" if url else source,
                    evidence_type=ev_type,
                    confidence=confidence,
                    dimension=ThesisDimension.SENTIMENT,
                    metadata=metadata,
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

        # Note if LLM was used
        if client:
            summaries.append("LLM-enhanced analysis")

        summary = f"{symbol}: " + "; ".join(summaries) + f" ({len(filtered_articles)} articles)"

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


def clear_sentiment_cache():
    """Clear the article score cache. Useful for testing."""
    _article_score_cache.clear()
