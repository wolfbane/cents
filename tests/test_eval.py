"""Tests for the LLM eval harness.

We don't hit the real Anthropic API here — every test injects a fake client
that returns scripted responses, so we can verify the metric math without
network or cost.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cents.cli import cli
from cents.eval import runner as eval_runner
from cents.eval.runner import (
    _BAND_TARGET,
    _bucket_score,
    _confusion_matrix,
    load_premise_golden,
    load_sentiment_golden,
    run_premise_eval,
    run_sentiment_eval,
)
from cents.models import EVENT_TAGS


# --- Shared fake Anthropic client ---


class _FakeAnthropic:
    """Returns the next scripted response on each `messages.create` call.

    Mirrors the shape used in tests/test_premise.py / test_sentiment_llm.py so
    we don't introduce a third "what does a fake anthropic client look like"
    pattern.
    """

    def __init__(self, response_texts):
        self._responses = list(response_texts)
        self._idx = 0
        self.messages = self
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._responses[self._idx]
        self._idx = min(self._idx + 1, len(self._responses) - 1)
        msg = MagicMock()
        msg.content = [MagicMock(text=text)]
        msg.model = "claude-haiku-4-5"
        msg.usage = MagicMock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        return msg


# --- Golden-set loading ---


class TestGoldenSets:
    def test_premise_golden_loads_and_validates(self):
        fixtures = load_premise_golden()
        assert len(fixtures) >= 30, "want at least 30 premise fixtures"
        for f in fixtures:
            assert {"id", "symbol", "thesis_summary", "expected_tags"} <= f.keys()
            for tag in f["expected_tags"]:
                assert tag in EVENT_TAGS, f"{f['id']} has bad tag {tag}"

    def test_premise_golden_respects_limit(self):
        assert len(load_premise_golden(limit=3)) == 3

    def test_sentiment_golden_loads_and_validates(self):
        fixtures = load_sentiment_golden()
        assert len(fixtures) >= 30, "want at least 30 sentiment fixtures"
        for f in fixtures:
            assert {"id", "symbol", "article_title", "expected_score_band"} <= f.keys()
            assert f["expected_score_band"] in {"bullish", "neutral", "bearish"}

    def test_sentiment_golden_respects_limit(self):
        assert len(load_sentiment_golden(limit=4)) == 4

    def test_premise_golden_rejects_invalid_tag(self, tmp_path, monkeypatch):
        # Patch the loader to read from a tampered file.
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({
            "id": "x", "symbol": "X", "thesis_summary": "",
            "expected_tags": ["not_a_real_tag"],
        }) + "\n")
        monkeypatch.setattr(
            eval_runner, "_golden_path", lambda _name: bad,
        )
        with pytest.raises(ValueError, match="not in EVENT_TAGS"):
            load_premise_golden()


# --- Bucketing + confusion matrix ---


class TestBucketing:
    def test_bullish_threshold(self):
        assert _bucket_score(0.4) == "bullish"
        assert _bucket_score(0.31) == "bullish"

    def test_bearish_threshold(self):
        assert _bucket_score(-0.4) == "bearish"
        assert _bucket_score(-0.31) == "bearish"

    def test_neutral_band(self):
        assert _bucket_score(0.0) == "neutral"
        assert _bucket_score(0.3) == "neutral"
        assert _bucket_score(-0.3) == "neutral"

    def test_confusion_matrix_shape_is_stable(self):
        m = _confusion_matrix([("bullish", "bullish"), ("bearish", "neutral")])
        # All bands appear in both dimensions.
        for b in ("bullish", "neutral", "bearish"):
            assert b in m
            for p in ("bullish", "neutral", "bearish"):
                assert p in m[b]
        assert m["bullish"]["bullish"] == 1
        assert m["bearish"]["neutral"] == 1
        assert m["neutral"]["neutral"] == 0


# --- Premise runner ---


class TestPremiseRunner:
    def test_perfect_predictions_score_full_marks(self, monkeypatch):
        # 2-fixture mini golden set; mocked LLM returns the expected tags
        # verbatim. Patch load_premise_golden so we don't pull the real file.
        mini = [
            {"id": "a", "symbol": "AAA", "thesis_summary": "x",
             "expected_tags": ["ai_capex"]},
            {"id": "b", "symbol": "BBB", "thesis_summary": "y",
             "expected_tags": ["fed_policy", "rates"]},
        ]
        monkeypatch.setattr(eval_runner, "load_premise_golden", lambda limit=None: mini)
        client = _FakeAnthropic([
            '{"tags": ["ai_capex"]}',
            '{"tags": ["fed_policy", "rates"]}',
        ])
        result = run_premise_eval(anthropic_client=client)
        assert result.fixtures_run == 2
        assert result.tp == 3
        assert result.fp == 0
        assert result.fn == 0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_partial_predictions_compute_pr_correctly(self, monkeypatch):
        mini = [
            {"id": "a", "symbol": "AAA", "thesis_summary": "x",
             "expected_tags": ["ai_capex", "fed_policy"]},
        ]
        monkeypatch.setattr(eval_runner, "load_premise_golden", lambda limit=None: mini)
        # Predict ai_capex (TP), tariffs.china (FP); miss fed_policy (FN).
        client = _FakeAnthropic(['{"tags": ["ai_capex", "tariffs.china"]}'])
        result = run_premise_eval(anthropic_client=client)
        assert result.tp == 1
        assert result.fp == 1
        assert result.fn == 1
        # precision = TP / (TP+FP) = 1/2; recall = TP / (TP+FN) = 1/2
        assert result.precision == 0.5
        assert result.recall == 0.5
        assert result.f1 == pytest.approx(0.5)

    def test_skips_with_clear_message_when_no_client(self, monkeypatch):
        # Force _build_anthropic_client to return None.
        monkeypatch.setattr(eval_runner, "_build_anthropic_client", lambda: None)
        result = run_premise_eval()
        assert result.fixtures_run == 0
        assert "ANTHROPIC_API_KEY" in result.skipped_reason


# --- Sentiment runner ---


class TestSentimentRunner:
    def test_perfect_bullish_scoring(self, monkeypatch):
        mini = [
            {"id": "a", "symbol": "AAA", "article_title": "Great news",
             "description": "Up big", "thesis": "moonshot",
             "expected_score_band": "bullish"},
        ]
        monkeypatch.setattr(eval_runner, "load_sentiment_golden", lambda limit=None: mini)
        # Score 0.8 → bullish (matches), squared err = (0.8-0.6)^2 = 0.04.
        client = _FakeAnthropic(['{"score": 0.8, "reasoning": "bullish"}'])
        result = run_sentiment_eval(anthropic_client=client)
        assert result.fixtures_run == 1
        assert result.correct_band == 1
        assert result.accuracy == 1.0
        assert result.brier_score == pytest.approx(0.04, abs=1e-6)
        assert result.confusion_matrix["bullish"]["bullish"] == 1

    def test_wrong_direction_hits_band_and_brier(self, monkeypatch):
        mini = [
            {"id": "a", "symbol": "AAA", "article_title": "Bad news",
             "description": "down", "thesis": "t",
             "expected_score_band": "bearish"},
        ]
        monkeypatch.setattr(eval_runner, "load_sentiment_golden", lambda limit=None: mini)
        # Score +0.7 → predicted bullish; expected bearish. Squared err
        # against pseudo-target -0.6 = (0.7 - (-0.6))^2 = 1.69.
        client = _FakeAnthropic(['{"score": 0.7, "reasoning": "wrong"}'])
        result = run_sentiment_eval(anthropic_client=client)
        assert result.correct_band == 0
        assert result.accuracy == 0.0
        assert result.brier_score == pytest.approx(1.69, abs=1e-6)
        assert result.confusion_matrix["bearish"]["bullish"] == 1

    def test_score_clamped_to_unit_interval(self, monkeypatch):
        mini = [
            {"id": "a", "symbol": "AAA", "article_title": "t",
             "description": "d", "thesis": "t",
             "expected_score_band": "bullish"},
        ]
        monkeypatch.setattr(eval_runner, "load_sentiment_golden", lambda limit=None: mini)
        # LLM returns 1.5; SentimentAgent clamps to 1.0; band still bullish.
        client = _FakeAnthropic(['{"score": 1.5, "reasoning": "very"}'])
        result = run_sentiment_eval(anthropic_client=client)
        assert result.fixtures[0]["score"] == 1.0
        assert result.correct_band == 1
        # Squared error vs bullish-target 0.6 = (1.0 - 0.6)^2 = 0.16
        assert result.brier_score == pytest.approx(0.16, abs=1e-6)

    def test_keyword_fallback_is_penalized(self, monkeypatch):
        # When the LLM emits unparseable text, SentimentAgent falls back to
        # keyword scoring. The eval runner notices and penalizes — predicted
        # band is neutral and the squared-error term uses 0.0 against
        # whatever pseudo-target.
        mini = [
            {"id": "a", "symbol": "AAA", "article_title": "Plain title",
             "description": "no obvious sentiment words",
             "thesis": "t", "expected_score_band": "bullish"},
        ]
        monkeypatch.setattr(eval_runner, "load_sentiment_golden", lambda limit=None: mini)
        client = _FakeAnthropic(["no json here at all"])
        result = run_sentiment_eval(anthropic_client=client)
        assert result.fixtures[0]["scoring_method"] != "llm"
        assert result.fixtures[0]["predicted_band"] == "neutral"
        # (0.0 - 0.6)^2 = 0.36
        assert result.brier_score == pytest.approx(0.36, abs=1e-6)

    def test_skips_with_clear_message_when_no_client(self, monkeypatch):
        monkeypatch.setattr(eval_runner, "_build_anthropic_client", lambda: None)
        result = run_sentiment_eval()
        assert result.fixtures_run == 0
        assert "ANTHROPIC_API_KEY" in result.skipped_reason
        # Confusion matrix has stable shape even when skipped.
        assert "bullish" in result.confusion_matrix

    def test_band_target_constants(self):
        # Locks the pseudo-targets so a future change forces a rethink.
        assert _BAND_TARGET == {"bullish": 0.6, "neutral": 0.0, "bearish": -0.6}


# --- CLI smoke tests ---


@pytest.fixture
def cli_runner():
    return CliRunner()


class TestEvalCLI:
    def test_eval_golden_show_premise(self, cli_runner):
        result = cli_runner.invoke(cli, ["eval", "golden", "show", "--set", "premise"])
        assert result.exit_code == 0, result.output
        assert "Premise golden set" in result.output

    def test_eval_golden_show_sentiment_json(self, cli_runner):
        result = cli_runner.invoke(
            cli, ["eval", "golden", "show", "--set", "sentiment", "-o", "json"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["set"] == "sentiment"
        assert data["count"] >= 30

    def test_eval_run_skips_when_no_key(self, cli_runner, monkeypatch):
        # Both subevals should skip; CLI should exit non-zero so cron / CI can
        # detect the missing credential rather than thinking everything passed.
        monkeypatch.setattr(eval_runner, "_build_anthropic_client", lambda: None)
        result = cli_runner.invoke(cli, ["eval", "run", "--set", "all"])
        assert result.exit_code == 1
        assert "ANTHROPIC_API_KEY" in result.output

    def test_eval_run_with_mocked_runner_text(self, cli_runner, monkeypatch):
        # Patch run_eval at the CLI layer so we don't hit the LLM but still
        # exercise the printer + exit code.
        from cents.cli import eval as eval_cli
        from cents.eval.runner import (
            EvalResult, PremiseEvalResult, SentimentEvalResult,
        )

        fake = EvalResult(
            premise=PremiseEvalResult(
                fixtures_run=2, tp=2, fp=1, fn=0,
                precision=2 / 3, recall=1.0, f1=0.8,
                fixtures=[
                    {"id": "a", "symbol": "A", "expected": ["x"],
                     "predicted": ["x"], "tp": 1, "fp": 0, "fn": 0},
                    {"id": "b", "symbol": "B", "expected": ["y"],
                     "predicted": ["y", "z"], "tp": 1, "fp": 1, "fn": 0},
                ],
            ),
            sentiment=SentimentEvalResult(
                fixtures_run=1, correct_band=1, accuracy=1.0,
                brier_score=0.04,
                confusion_matrix={
                    "bullish": {"bullish": 1, "neutral": 0, "bearish": 0},
                    "neutral": {"bullish": 0, "neutral": 0, "bearish": 0},
                    "bearish": {"bullish": 0, "neutral": 0, "bearish": 0},
                },
                fixtures=[{"id": "a", "symbol": "A", "expected_band": "bullish",
                           "predicted_band": "bullish", "score": 0.8,
                           "scoring_method": "llm"}],
            ),
            model="claude-haiku-4-5",
        )
        monkeypatch.setattr(eval_cli, "run_eval", lambda **_kw: fake)
        result = cli_runner.invoke(cli, ["eval", "run", "--set", "all"])
        assert result.exit_code == 0, result.output
        assert "Premise classifier eval" in result.output
        assert "Sentiment scorer eval" in result.output
        assert "Confusion matrix" in result.output

    def test_eval_run_with_mocked_runner_json(self, cli_runner, monkeypatch):
        from cents.cli import eval as eval_cli
        from cents.eval.runner import EvalResult, PremiseEvalResult

        fake = EvalResult(
            premise=PremiseEvalResult(
                fixtures_run=1, tp=1, fp=0, fn=0,
                precision=1.0, recall=1.0, f1=1.0,
                fixtures=[{"id": "a", "symbol": "A", "expected": ["x"],
                           "predicted": ["x"], "tp": 1, "fp": 0, "fn": 0}],
            ),
            sentiment=None,
            model="claude-haiku-4-5",
        )
        monkeypatch.setattr(eval_cli, "run_eval", lambda **_kw: fake)
        result = cli_runner.invoke(
            cli, ["eval", "run", "--set", "premise", "-o", "json"]
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["premise"]["fixtures_run"] == 1
        assert data["premise"]["precision"] == 1.0
        assert data["sentiment"] is None
        assert data["model"] == "claude-haiku-4-5"
