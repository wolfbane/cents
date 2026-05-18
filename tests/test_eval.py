"""Tests for the LLM eval harness.

We don't hit the real Anthropic API here — every test injects a fake client
that returns scripted responses, so we can verify the metric math without
network or cost.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cents.cli import cli
from cents.eval import runner as eval_runner
from cents.eval import baseline as baseline_mod
from cents.eval.baseline import (
    detect_drift,
    evaluate_gate,
    load_baseline,
    load_history,
    load_thresholds,
    persist_baseline as _persist_baseline,
    persist_history_row,
    persist_thresholds,
)
from cents.eval.calibrate import calibrate_thresholds
from cents.eval.runner import (
    _BAND_TARGET,
    _bucket_score,
    _confusion_matrix,
    bootstrap_ci,
    load_premise_golden,
    load_sentiment_golden,
    run_premise_eval,
    run_sentiment_eval,
)
from cents.models import EVENT_TAGS, AlertType


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


# --- Golden-set schema + coverage validation -----------------------------


class TestGoldenSetValidation:
    """Schema-level invariants on the packaged golden fixtures.

    These tests guard against silent regressions in fixture quality — every
    tag in `expected_tags` must come from the controlled vocabulary, every
    sentiment fixture must have a valid `expected_score_band`, and the
    operator-targeted "200+ fixtures, every tag covered, multi-sector"
    promises must hold.
    """

    def test_premise_set_meets_target_size(self):
        fixtures = load_premise_golden()
        assert len(fixtures) >= 200, (
            f"premise golden set has {len(fixtures)} fixtures; need 200+"
        )

    def test_sentiment_set_meets_target_size(self):
        fixtures = load_sentiment_golden()
        assert len(fixtures) >= 200, (
            f"sentiment golden set has {len(fixtures)} fixtures; need 200+"
        )

    def test_premise_fixtures_have_required_keys(self):
        fixtures = load_premise_golden()
        for f in fixtures:
            assert {"id", "symbol", "thesis_summary", "expected_tags"} <= f.keys(), (
                f"fixture missing required keys: {f.get('id')}"
            )
            assert isinstance(f["expected_tags"], list), f["id"]

    def test_premise_fixture_ids_are_unique(self):
        fixtures = load_premise_golden()
        ids = [f["id"] for f in fixtures]
        assert len(ids) == len(set(ids)), "duplicate premise fixture IDs"

    def test_sentiment_fixtures_have_required_keys(self):
        fixtures = load_sentiment_golden()
        for f in fixtures:
            assert {"id", "symbol", "article_title", "expected_score_band"} <= f.keys(), (
                f"fixture missing required keys: {f.get('id')}"
            )

    def test_sentiment_fixture_ids_are_unique(self):
        fixtures = load_sentiment_golden()
        ids = [f["id"] for f in fixtures]
        assert len(ids) == len(set(ids)), "duplicate sentiment fixture IDs"

    def test_all_premise_tags_in_controlled_vocab(self):
        fixtures = load_premise_golden()
        for f in fixtures:
            for tag in f["expected_tags"]:
                assert tag in EVENT_TAGS, (
                    f"{f['id']} has tag {tag!r} not in EVENT_TAGS"
                )

    def test_all_sentiment_bands_valid(self):
        fixtures = load_sentiment_golden()
        for f in fixtures:
            assert f["expected_score_band"] in {"bullish", "neutral", "bearish"}, (
                f"{f['id']} has invalid band {f['expected_score_band']!r}"
            )

    def test_every_event_tag_has_some_premise_coverage(self):
        """Each EVENT_TAGS entry should appear in at least one fixture.

        Without this, a tag could drift out of the curated set silently.
        """
        fixtures = load_premise_golden()
        covered = set()
        for f in fixtures:
            for tag in f["expected_tags"]:
                covered.add(tag)
        missing = EVENT_TAGS - covered
        assert not missing, f"event tags with no fixture coverage: {sorted(missing)}"

    def test_sentiment_per_sector_coverage(self):
        """Roughly 10+ fixtures per major sector by symbol membership.

        This is a sanity check, not a perfect classifier — we accept that
        many of these symbols span multiple sectors.
        """
        sector_buckets = {
            "tech": {"AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "TSM", "INTC", "META",
                     "AVGO", "AMZN", "ARM", "MU", "QCOM", "MRVL", "ORCL", "CRM",
                     "CRWD", "PANW", "ZS", "DDOG", "SNOW", "MDB", "NOW", "ADBE",
                     "ANET", "CSCO", "IBM", "OKTA", "TWLO", "TXN", "ON", "WOLF",
                     "TEAM", "GTLB", "TM", "GE", "ARM", "LRCX", "AMAT", "KLAC",
                     "ASML"},
            "financials": {"JPM", "BAC", "WFC", "C", "GS", "BX", "KKR", "APO", "ARES",
                           "SCHW", "BLK", "SPGI", "MCO", "ICE", "CME", "AMP",
                           "TROW", "STT", "BK", "MET", "AIG", "PNC", "USB", "TFC",
                           "HBAN", "RF", "V", "MA", "SQ", "PYPL"},
            "energy": {"XOM", "CVX", "SLB", "OXY", "EOG", "DVN", "PSX", "VLO",
                       "MPC", "WMB", "ET", "EQT", "BKR", "HAL", "LIN", "GEV",
                       "CCJ", "CEG", "URA", "VST", "TLN", "NRG", "BTU", "FCX",
                       "ALB", "NEM", "RIO", "BHP", "MP"},
            "healthcare": {"LLY", "PFE", "JNJ", "ABBV", "MRK", "BMY", "VRTX",
                           "REGN", "GILD", "AMGN", "BIIB", "MRNA", "UNH", "CVS",
                           "ELV", "HCA", "ISRG", "WBA", "GEO", "CXW"},
            "consumer": {"WMT", "TGT", "COST", "HD", "LOW", "DG", "DLTR", "NKE",
                         "LULU", "ULTA", "EL", "KO", "PG", "PEP", "MCD", "SBUX",
                         "CMG", "RH", "F", "GM", "TSLA", "ABNB", "BKNG", "MAR",
                         "ROST", "BURL", "TJX", "BUD"},
        }
        fixtures = load_sentiment_golden()
        for sector, syms in sector_buckets.items():
            count = sum(1 for f in fixtures if f["symbol"] in syms)
            assert count >= 10, (
                f"sector {sector} has only {count} fixtures (want >=10)"
            )

    def test_sentiment_set_has_prompt_injection_negative_controls(self):
        """At least one bearish-style injection payload must be in the
        sentiment set — these are the negative controls that catch a model
        which acts on adversarial instructions embedded in untrusted text.
        """
        fixtures = load_sentiment_golden()

        def _has_injection(rec: dict) -> bool:
            text = (rec.get("article_title", "") + " " + rec.get("description", "")).lower()
            return (
                "ignore previous" in text
                or "ignore prior" in text
                or "</article>" in text
                or "system:" in text
            )

        injection_fixtures = [f for f in fixtures if _has_injection(f)]
        assert injection_fixtures, "no prompt-injection negative-control fixtures found"
        # The expected band for an injection payload must be neutral (we don't
        # want models to grant attackers their requested polarity).
        for f in injection_fixtures:
            assert f["expected_score_band"] == "neutral", (
                f"injection fixture {f['id']} has non-neutral expected band — "
                "this defeats the negative-control purpose."
            )


# --- Bootstrap CI -------------------------------------------------------


class TestBootstrapCI:
    def test_ci_brackets_the_mean_on_known_data(self):
        # All values 1.0 → mean is exactly 1.0 and CI collapses to (1.0, 1.0).
        lo, hi = bootstrap_ci([1.0] * 10, samples=200)
        assert lo == 1.0 == hi

    def test_ci_widens_with_variance(self):
        narrow = bootstrap_ci([0.5] * 50, samples=200)
        wide = bootstrap_ci([0.0, 1.0] * 25, samples=200)
        assert (narrow[1] - narrow[0]) <= (wide[1] - wide[0])

    def test_ci_is_reproducible_with_fixed_seed(self):
        data = [0.0, 0.3, 0.6, 1.0, 0.5, 0.7, 0.2, 0.9, 0.4, 0.55]
        a = bootstrap_ci(data, samples=200, seed=42)
        b = bootstrap_ci(data, samples=200, seed=42)
        assert a == b

    def test_ci_handles_empty_input(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_ci_handles_singleton(self):
        assert bootstrap_ci([0.7]) == (0.7, 0.7)

    def test_premise_run_populates_f1_ci(self, monkeypatch):
        from cents.eval.runner import _BAND_TARGET as _bt  # noqa
        mini = [
            {"id": "a", "symbol": "A", "thesis_summary": "x",
             "expected_tags": ["ai_capex"]},
            {"id": "b", "symbol": "B", "thesis_summary": "y",
             "expected_tags": ["fed_policy"]},
            {"id": "c", "symbol": "C", "thesis_summary": "z",
             "expected_tags": ["rates"]},
        ]
        monkeypatch.setattr(eval_runner, "load_premise_golden", lambda limit=None: mini)
        client = _FakeAnthropic([
            '{"tags": ["ai_capex"]}',
            '{"tags": ["fed_policy"]}',
            '{"tags": ["rates"]}',
        ])
        result = run_premise_eval(anthropic_client=client)
        # F1 is 1.0; CI tight on a perfect run.
        assert result.f1_ci[0] <= result.f1 <= result.f1_ci[1] + 1e-9

    def test_sentiment_run_populates_cis(self, monkeypatch):
        mini = [
            {"id": "a", "symbol": "A", "article_title": "great",
             "description": "x", "thesis": "t",
             "expected_score_band": "bullish"},
            {"id": "b", "symbol": "B", "article_title": "bad",
             "description": "y", "thesis": "t",
             "expected_score_band": "bearish"},
        ]
        monkeypatch.setattr(eval_runner, "load_sentiment_golden", lambda limit=None: mini)
        client = _FakeAnthropic([
            '{"score": 0.7, "reasoning": "g"}',
            '{"score": -0.7, "reasoning": "b"}',
        ])
        result = run_sentiment_eval(anthropic_client=client)
        assert result.accuracy_ci[0] <= result.accuracy <= result.accuracy_ci[1] + 1e-9
        assert result.brier_ci[0] <= result.brier_score <= result.brier_ci[1] + 1e-9


# --- Baseline + gate -----------------------------------------------------


class TestBaselineGate:
    @pytest.fixture(autouse=True)
    def _isolate_baseline(self, tmp_path, monkeypatch):
        # Redirect the packaged baseline.json path to a tmp file so tests
        # don't clobber the real shipping baseline.
        from cents.eval import baseline as bmod
        tmp_baseline = tmp_path / "baseline.json"
        tmp_thresholds = tmp_path / "thresholds.json"
        monkeypatch.setattr(bmod, "_packaged_baseline_path", lambda: tmp_baseline)
        monkeypatch.setattr(bmod, "_packaged_thresholds_path", lambda: tmp_thresholds)

    def test_permissive_when_baseline_unlocked(self):
        outcome = evaluate_gate(
            premise_f1=0.50,
            sentiment_brier=2.0,
            tolerance_pp=5.0,
            baseline={"premise_f1": 0.9, "sentiment_brier": 0.1, "locked_at": None},
        )
        assert outcome["permissive"] is True
        assert outcome["passed"] is True

    def test_passes_when_within_tolerance(self):
        outcome = evaluate_gate(
            premise_f1=0.87,
            sentiment_brier=0.12,
            tolerance_pp=5.0,
            baseline={"premise_f1": 0.90, "sentiment_brier": 0.10, "locked_at": "x"},
        )
        assert outcome["passed"] is True

    def test_fails_when_premise_regresses_beyond_tolerance(self):
        outcome = evaluate_gate(
            premise_f1=0.80,  # 10pp below baseline 0.90
            sentiment_brier=0.10,
            tolerance_pp=5.0,
            baseline={"premise_f1": 0.90, "sentiment_brier": 0.10, "locked_at": "x"},
        )
        assert outcome["passed"] is False
        # one of the messages mentions the regression
        assert any("regression" in m for m in outcome["messages"])

    def test_fails_when_brier_worsens_beyond_tolerance(self):
        outcome = evaluate_gate(
            premise_f1=0.90,
            sentiment_brier=0.30,  # 20pp worse than baseline 0.10
            tolerance_pp=5.0,
            baseline={"premise_f1": 0.90, "sentiment_brier": 0.10, "locked_at": "x"},
        )
        assert outcome["passed"] is False

    def test_persist_baseline_writes_complete_record(self):
        rec = _persist_baseline(
            premise_f1=0.85, sentiment_brier=0.15,
            model_snapshot="claude-haiku-4-5-20251001",
        )
        assert rec["premise_f1"] == 0.85
        assert rec["sentiment_brier"] == 0.15
        assert rec["locked_at"]  # non-empty timestamp
        assert rec["model_snapshot"] == "claude-haiku-4-5-20251001"
        reloaded = load_baseline()
        assert reloaded["premise_f1"] == 0.85

    def test_persist_baseline_preserves_unwritten_half(self):
        _persist_baseline(premise_f1=0.7, sentiment_brier=0.2, model_snapshot="m")
        # Now persist only premise — sentiment_brier should carry over.
        rec = _persist_baseline(premise_f1=0.8, sentiment_brier=None, model_snapshot="m")
        assert rec["premise_f1"] == 0.8
        assert rec["sentiment_brier"] == 0.2


# --- History persistence + drift -----------------------------------------


class TestHistoryDrift:
    def test_persist_history_appends_row(self, tmp_path):
        # CENTS_DB_PATH is autouse-set to tmp_path/test.db by conftest, so
        # history lands in tmp_path/eval_history/.
        path = persist_history_row(
            premise_f1=0.8,
            sentiment_brier=0.1,
            premise_f1_ci=(0.78, 0.82),
            sentiment_brier_ci=(0.08, 0.12),
            model_snapshot="m1",
            n_fixtures_premise=200,
            n_fixtures_sentiment=200,
        )
        assert path.exists()
        rows = load_history()
        assert len(rows) == 1
        assert rows[0]["premise_f1"] == 0.8
        # second call appends.
        persist_history_row(
            premise_f1=0.81,
            sentiment_brier=0.11,
            premise_f1_ci=(0.79, 0.83),
            sentiment_brier_ci=(0.09, 0.13),
            model_snapshot="m1",
            n_fixtures_premise=200,
            n_fixtures_sentiment=200,
        )
        rows = load_history()
        assert len(rows) == 2

    def test_drift_check_no_history_returns_no_drift(self):
        outcome = detect_drift()
        assert outcome["drift_detected"] is False
        assert "No premise_f1 row" in outcome["messages"][0]

    def test_drift_check_fires_when_below_threshold(self):
        today = date.today()
        # Build a 7-row history at F1 ~0.85, then today at 0.75 → 10pp below median.
        for i in range(7):
            d = today - timedelta(days=i + 1)
            persist_history_row(
                premise_f1=0.85,
                sentiment_brier=0.1,
                premise_f1_ci=None,
                sentiment_brier_ci=None,
                model_snapshot="m",
                n_fixtures_premise=200,
                n_fixtures_sentiment=200,
                when=datetime(d.year, d.month, d.day, 12, 0, 0),
            )
        # Today's row.
        persist_history_row(
            premise_f1=0.75,
            sentiment_brier=0.1,
            premise_f1_ci=None,
            sentiment_brier_ci=None,
            model_snapshot="m",
            n_fixtures_premise=200,
            n_fixtures_sentiment=200,
        )
        outcome = detect_drift(threshold_pp=5.0)
        assert outcome["drift_detected"] is True
        assert outcome["today_f1"] == 0.75
        assert abs(outcome["median_f1"] - 0.85) < 1e-9
        assert outcome["delta_pp"] < -5

    def test_drift_check_does_not_fire_when_within_band(self):
        today = date.today()
        for i in range(7):
            d = today - timedelta(days=i + 1)
            persist_history_row(
                premise_f1=0.85,
                sentiment_brier=0.1,
                premise_f1_ci=None,
                sentiment_brier_ci=None,
                model_snapshot="m",
                n_fixtures_premise=200,
                n_fixtures_sentiment=200,
                when=datetime(d.year, d.month, d.day, 12, 0, 0),
            )
        persist_history_row(
            premise_f1=0.83,
            sentiment_brier=0.1,
            premise_f1_ci=None,
            sentiment_brier_ci=None,
            model_snapshot="m",
            n_fixtures_premise=200,
            n_fixtures_sentiment=200,
        )
        outcome = detect_drift(threshold_pp=5.0)
        assert outcome["drift_detected"] is False


# --- Threshold calibration ---------------------------------------------


class TestThresholdCalibration:
    def test_picks_obvious_thresholds_on_synthetic_data(self):
        # Linearly separable: bullish at +0.5, bearish at -0.5, neutral at 0.
        fixtures = []
        for _ in range(20):
            fixtures.append({"score": 0.5, "expected_score_band": "bullish"})
            fixtures.append({"score": -0.5, "expected_score_band": "bearish"})
            fixtures.append({"score": 0.0, "expected_score_band": "neutral"})
        result = calibrate_thresholds(fixtures)
        # Any threshold pair where -0.5 < neg_t < 0 < pos_t < 0.5 works;
        # the search should find one with accuracy 1.0.
        assert result.accuracy == 1.0
        assert result.negative_threshold > -0.5
        assert result.positive_threshold < 0.5
        assert result.n_fixtures == 60

    def test_breaks_ties_in_favour_of_band_balance(self):
        # A degenerate set that scores 0.0 → expected neutral, no bullish/bearish data.
        # ALL threshold pairs would yield accuracy 1.0 by predicting neutral, but
        # the balance tie-breaker should still pick something sane (not crash).
        fixtures = [{"score": 0.0, "expected_score_band": "neutral"}] * 10
        result = calibrate_thresholds(fixtures)
        assert result.accuracy == 1.0
        # All predictions are neutral; balance is 0 because only one band fired.
        assert result.balance == 0.0

    def test_empty_inputs_returns_defaults(self):
        result = calibrate_thresholds([])
        assert result.n_fixtures == 0
        assert result.accuracy == 0.0
        assert result.positive_threshold == 0.3
        assert result.negative_threshold == -0.2

    def test_skips_fixtures_missing_required_fields(self):
        fixtures = [
            {"score": 0.5, "expected_score_band": "bullish"},
            {"expected_score_band": "bullish"},  # missing score
            {"score": 0.5},  # missing band
            {"score": -0.5, "expected_score_band": "bearish"},
        ]
        result = calibrate_thresholds(fixtures)
        assert result.n_fixtures == 2

    def test_persist_thresholds_writes_file(self, tmp_path, monkeypatch):
        from cents.eval import baseline as bmod
        tmp_thresh = tmp_path / "thresholds.json"
        monkeypatch.setattr(bmod, "_packaged_thresholds_path", lambda: tmp_thresh)
        persist_thresholds(
            positive_threshold=0.25,
            negative_threshold=-0.15,
            accuracy=0.84,
            model_snapshot="m",
        )
        reloaded = load_thresholds()
        assert reloaded["positive_threshold"] == 0.25
        assert reloaded["negative_threshold"] == -0.15
        assert reloaded["calibrated_accuracy"] == 0.84

    def test_sentiment_agent_uses_calibrated_thresholds(self, tmp_path, monkeypatch):
        """The agent's _resolve_llm_thresholds() should pick up the calibrated file."""
        from cents.eval import baseline as bmod
        from cents.agents.sentiment import _resolve_llm_thresholds

        tmp_thresh = tmp_path / "thresholds.json"
        monkeypatch.setattr(bmod, "_packaged_thresholds_path", lambda: tmp_thresh)

        # No file → defaults.
        defaults = _resolve_llm_thresholds()
        assert defaults["positive_threshold"] == 0.2
        assert defaults["negative_threshold"] == -0.2

        # Persist calibrated, then read again.
        persist_thresholds(
            positive_threshold=0.35,
            negative_threshold=-0.25,
            accuracy=0.85,
            model_snapshot="m",
        )
        calibrated = _resolve_llm_thresholds()
        assert calibrated["positive_threshold"] == 0.35
        assert calibrated["negative_threshold"] == -0.25


# --- CLI surfaces --------------------------------------------------------


class TestEvalCLINew:
    def test_drift_check_emits_alert_on_regression(self, cli_runner):
        # Seed a clean history with >5pp drop today.
        today = date.today()
        for i in range(7):
            d = today - timedelta(days=i + 1)
            persist_history_row(
                premise_f1=0.85,
                sentiment_brier=0.1,
                premise_f1_ci=None,
                sentiment_brier_ci=None,
                model_snapshot="m",
                n_fixtures_premise=200,
                n_fixtures_sentiment=200,
                when=datetime(d.year, d.month, d.day, 12, 0, 0),
            )
        persist_history_row(
            premise_f1=0.70,
            sentiment_brier=0.1,
            premise_f1_ci=None,
            sentiment_brier_ci=None,
            model_snapshot="m",
            n_fixtures_premise=200,
            n_fixtures_sentiment=200,
        )
        result = cli_runner.invoke(cli, ["eval", "drift-check", "--threshold-pp", "5"])
        # exit code 2 signals drift detected.
        assert result.exit_code == 2, result.output
        assert "DRIFT" in result.output

        # The alert was persisted under the test-isolated DB.
        from cents.db import AlertRepository
        alerts = AlertRepository().list_all()
        assert any(a.alert_type == AlertType.MODEL_DRIFT for a in alerts)

    def test_drift_check_returns_clean_on_no_history(self, cli_runner):
        result = cli_runner.invoke(cli, ["eval", "drift-check", "--threshold-pp", "5"])
        assert result.exit_code == 0
        assert "No premise_f1 row" in result.output

    def test_eval_run_persist_history_writes_row(self, cli_runner, monkeypatch):
        from cents.cli import eval as eval_cli
        from cents.eval.runner import EvalResult, PremiseEvalResult, SentimentEvalResult

        fake = EvalResult(
            premise=PremiseEvalResult(
                fixtures_run=1, tp=1, fp=0, fn=0,
                precision=1.0, recall=1.0, f1=0.9, f1_ci=(0.85, 0.95),
                fixtures=[],
            ),
            sentiment=SentimentEvalResult(
                fixtures_run=1, correct_band=1, accuracy=1.0,
                brier_score=0.05, brier_ci=(0.04, 0.06),
                confusion_matrix={
                    "bullish": {"bullish": 1, "neutral": 0, "bearish": 0},
                    "neutral": {"bullish": 0, "neutral": 0, "bearish": 0},
                    "bearish": {"bullish": 0, "neutral": 0, "bearish": 0},
                },
                fixtures=[],
            ),
            model="claude-haiku-4-5",
        )
        monkeypatch.setattr(eval_cli, "run_eval", lambda **_kw: fake)
        result = cli_runner.invoke(
            cli, ["eval", "run", "--set", "all", "--persist-history"]
        )
        assert result.exit_code == 0, result.output
        rows = load_history()
        assert rows
        assert rows[-1]["premise_f1"] == 0.9
        assert rows[-1]["sentiment_brier"] == 0.05

    def test_eval_run_gate_fails_on_regression(self, cli_runner, monkeypatch, tmp_path):
        from cents.cli import eval as eval_cli
        from cents.eval import baseline as bmod
        from cents.eval.runner import EvalResult, PremiseEvalResult

        # Lock a baseline at 0.9 in an isolated location.
        tmp_baseline = tmp_path / "baseline.json"
        monkeypatch.setattr(bmod, "_packaged_baseline_path", lambda: tmp_baseline)
        _persist_baseline(premise_f1=0.9, sentiment_brier=0.1, model_snapshot="m")

        fake = EvalResult(
            premise=PremiseEvalResult(
                fixtures_run=1, tp=1, fp=0, fn=0,
                precision=1.0, recall=1.0, f1=0.5, f1_ci=(0.45, 0.55),
                fixtures=[],
            ),
            sentiment=None,
            model="claude-haiku-4-5",
        )
        monkeypatch.setattr(eval_cli, "run_eval", lambda **_kw: fake)
        result = cli_runner.invoke(
            cli, ["eval", "run", "--set", "premise", "--gate", "--tolerance-pp", "5"]
        )
        # Exit 2 = gate failed.
        assert result.exit_code == 2, result.output
        assert "regression" in result.output.lower()

    def test_eval_run_persist_baseline_locks(self, cli_runner, monkeypatch, tmp_path):
        from cents.cli import eval as eval_cli
        from cents.eval import baseline as bmod
        from cents.eval.runner import EvalResult, PremiseEvalResult

        tmp_baseline = tmp_path / "baseline.json"
        monkeypatch.setattr(bmod, "_packaged_baseline_path", lambda: tmp_baseline)

        fake = EvalResult(
            premise=PremiseEvalResult(
                fixtures_run=1, tp=1, fp=0, fn=0,
                precision=1.0, recall=1.0, f1=0.92, f1_ci=(0.89, 0.95),
                fixtures=[],
            ),
            sentiment=None,
            model="claude-haiku-4-5",
        )
        monkeypatch.setattr(eval_cli, "run_eval", lambda **_kw: fake)
        result = cli_runner.invoke(
            cli, ["eval", "run", "--set", "premise", "--persist-baseline"]
        )
        assert result.exit_code == 0, result.output
        rec = load_baseline()
        assert rec["premise_f1"] == 0.92
        assert rec["locked_at"]
