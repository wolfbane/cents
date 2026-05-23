"""Tests for cents.viz.queries.

The queries module is load-bearing for every chart — a bug in the
thesis ⋈ position join or the cost-attribution rule silently corrupts
every downstream visualization. The other viz modules (ascii, static,
sunburst) deliberately only do rendering, so they don't need their own
data-shape tests.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from cents.db.repository import (
    LLMUsageRepository,
    PositionRepository,
    ThesisRepository,
)
from cents.db.schema import reset_connection
from cents.models.llm_usage import LLMUsage
from cents.models.position import Position, PositionSide, PositionStatus
from cents.models.thesis import (
    HedgeBasis,
    PremiseSource,
    Thesis,
    ThesisCohort,
    ThesisOutcome,
    ThesisStatus,
)
from cents.viz import queries as q


# ---------------------------------------------------------------------------
# Fixtures — three theses spanning the cases each chart cares about.
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("CENTS_DB_PATH", str(tmp_path / "viz.db"))
    reset_connection()

    t_repo = ThesisRepository()
    p_repo = PositionRepository()
    u_repo = LLMUsageRepository()

    now = datetime(2026, 5, 22, 12, 0, 0)
    week_ago = now - timedelta(days=7)
    fortnight = now - timedelta(days=14)

    # LLM-arm directional, judged CORRECT.
    t_llm_win = Thesis(
        id="t_llm_win",
        title="bullish on NVDA",
        symbol="NVDA",
        cohort=ThesisCohort.DIRECTIONAL,
        orchestrator_label="llm",
        experiment_id="pilot_v1",
        premise_classification_source=PremiseSource.LLM,
        premise_tags=["ai_capex"],
        regime_snapshot={"policy": "risk-on", "vol": "low"},
        discovery_source="my_value",
        conviction=72.0,
        calibrated_p_correct=0.62,
        status=ThesisStatus.CLOSED,
        outcome=ThesisOutcome.CORRECT,
        created_at=fortnight,
        closed_at=week_ago,
    )
    t_repo.create(t_llm_win)
    p_repo.create(Position(
        thesis_id="t_llm_win",
        symbol="NVDA",
        side=PositionSide.LONG,
        size=10,
        entry_price=100.0,
        entry_date=fortnight.date(),
        exit_price=120.0,
        exit_date=week_ago.date(),
        status=PositionStatus.CLOSED,
        costs_applied_usd=20.0,
    ))

    # LLM-arm paired-neutral, judged INCORRECT, two legs.
    t_llm_loss = Thesis(
        id="t_llm_loss",
        title="bearish on F",
        symbol="F",
        hedge_symbol="XLI",
        cohort=ThesisCohort.NEUTRAL,
        hedge_basis=HedgeBasis.BETA,
        orchestrator_label="llm",
        experiment_id="pilot_v1",
        premise_classification_source=PremiseSource.LLM,
        premise_tags=["tariffs.china"],
        regime_snapshot={"policy": "risk-off", "vol": "high"},
        discovery_source="my_value",
        conviction=28.0,
        calibrated_p_correct=0.55,
        status=ThesisStatus.CLOSED,
        outcome=ThesisOutcome.INCORRECT,
        created_at=fortnight,
        closed_at=now - timedelta(days=2),
    )
    t_repo.create(t_llm_loss)
    p_repo.create(Position(
        thesis_id="t_llm_loss",
        symbol="F",
        side=PositionSide.SHORT,
        size=10,
        entry_price=10.0,
        entry_date=fortnight.date(),
        exit_price=12.0,
        exit_date=(now - timedelta(days=2)).date(),
        status=PositionStatus.CLOSED,
        costs_applied_usd=5.0,
    ))
    p_repo.create(Position(
        thesis_id="t_llm_loss",
        symbol="XLI",
        side=PositionSide.LONG,
        size=10,
        entry_price=120.0,
        entry_date=fortnight.date(),
        exit_price=118.0,
        exit_date=(now - timedelta(days=2)).date(),
        status=PositionStatus.CLOSED,
        costs_applied_usd=5.0,
    ))

    # Random-arm directional, judged CORRECT.
    t_random_win = Thesis(
        id="t_random_win",
        title="random: AAPL",
        symbol="AAPL",
        cohort=ThesisCohort.DIRECTIONAL,
        orchestrator_label="random",
        experiment_id="pilot_v1",
        premise_classification_source=PremiseSource.FALLBACK_SECTOR,
        premise_tags=["ai_capex"],
        regime_snapshot={"policy": "risk-on", "vol": "low"},
        conviction=58.0,
        calibrated_p_correct=0.50,
        status=ThesisStatus.CLOSED,
        outcome=ThesisOutcome.CORRECT,
        created_at=fortnight,
        closed_at=week_ago,
    )
    t_repo.create(t_random_win)
    p_repo.create(Position(
        thesis_id="t_random_win",
        symbol="AAPL",
        side=PositionSide.LONG,
        size=5,
        entry_price=200.0,
        entry_date=fortnight.date(),
        exit_price=210.0,
        exit_date=week_ago.date(),
        status=PositionStatus.CLOSED,
        costs_applied_usd=2.0,
    ))

    # LLM-cost rows: an attributable one on NVDA, an overhead one, and
    # an unattributable one.
    u_repo.create(LLMUsage(
        model="claude-haiku-4-5-20251001",
        agent="sentiment",
        operation="score",
        input_tokens=1000,
        output_tokens=200,
        context="NVDA",
        called_at=fortnight + timedelta(hours=1),
    ))
    u_repo.create(LLMUsage(
        model="claude-haiku-4-5-20251001",
        agent="premise",
        operation="classify_premise",
        input_tokens=500,
        output_tokens=50,
        context="NVDA",
        called_at=fortnight + timedelta(hours=1),
    ))
    u_repo.create(LLMUsage(
        model="claude-haiku-4-5-20251001",
        agent="event",
        operation="tag_event",
        input_tokens=2000,
        output_tokens=300,
        context=None,
        called_at=fortnight + timedelta(hours=1),
    ))

    yield {"now": now}

    reset_connection()


# ---------------------------------------------------------------------------
# list_theses — the join that backs every chart.
# ---------------------------------------------------------------------------


def test_list_theses_rolls_up_legs(populated_db):
    rows = q.list_theses()
    by_id = {r.id: r for r in rows}

    # Directional thesis: one leg.
    assert by_id["t_llm_win"].legs == 1
    assert by_id["t_llm_win"].pnl == pytest.approx(180.0)  # (120-100)*10 - 20

    # Paired-neutral: both legs sum into one pnl. SHORT F (entry 10 → exit
    # 12) loses 20; LONG XLI (entry 120 → exit 118) loses 20; costs 10.
    assert by_id["t_llm_loss"].legs == 2
    assert by_id["t_llm_loss"].pnl == pytest.approx(-50.0)


def test_list_theses_filters_by_experiment(populated_db):
    rows = q.list_theses(experiment_id="pilot_v1")
    assert {r.id for r in rows} == {"t_llm_win", "t_llm_loss", "t_random_win"}

    rows = q.list_theses(experiment_id="nonexistent")
    assert rows == []


def test_regime_label_compaction(populated_db):
    rows = q.list_theses()
    by_id = {r.id: r for r in rows}
    assert by_id["t_llm_win"].regime_label == "risk-on:low"
    assert by_id["t_llm_loss"].regime_label == "risk-off:high"


# ---------------------------------------------------------------------------
# cost_by_thesis — the attribution rule must match the factory analyzer.
# ---------------------------------------------------------------------------


def test_cost_attribution_matches_factory_rules(populated_db):
    rows = q.list_theses()
    cost, unattributable = q.cost_by_thesis(rows)

    # The NVDA `score` call (context=NVDA, within t_llm_win's lifetime)
    # attributes to t_llm_win. The two overhead ops (classify_premise,
    # tag_event) go to unattributable, regardless of context.
    assert "t_llm_win" in cost
    assert cost["t_llm_win"] > 0
    assert "t_llm_loss" not in cost
    assert "t_random_win" not in cost
    assert unattributable > 0


# ---------------------------------------------------------------------------
# cohort_metrics — the workhorse for charts 3, 5, 11.
# ---------------------------------------------------------------------------


def test_cohort_metrics_groups_by_orchestrator(populated_db):
    rows = q.list_theses()
    metrics = q.cohort_metrics(rows, by=["orchestrator"])
    by_key = {m.key: m for m in metrics}

    # llm arm: 2 opened, 2 judged, 1 correct.
    llm = by_key[("llm",)]
    assert llm.opened == 2
    assert llm.judged == 2
    assert llm.correct == 1
    assert llm.win_rate == pytest.approx(0.5)
    assert llm.win_rate_ci is not None
    lo, hi = llm.win_rate_ci
    assert 0.0 <= lo < 0.5 < hi <= 1.0

    # random arm: 1 opened, 1 judged, 1 correct.
    rnd = by_key[("random",)]
    assert rnd.win_rate == pytest.approx(1.0)


def test_cohort_metrics_2x2(populated_db):
    rows = q.list_theses()
    metrics = q.cohort_metrics(rows, by=["orchestrator", "cohort"])
    by_key = {m.key: m for m in metrics}

    # llm × directional: t_llm_win — 1 opened, 1 correct.
    cell = by_key[("llm", "directional")]
    assert cell.opened == 1 and cell.correct == 1

    # llm × neutral: t_llm_loss — 1 opened, 0 correct.
    cell = by_key[("llm", "neutral")]
    assert cell.opened == 1 and cell.correct == 0


def test_cohort_metrics_unknown_axis_raises(populated_db):
    with pytest.raises(ValueError):
        q.cohort_metrics([], by=["not_a_real_axis"])


# ---------------------------------------------------------------------------
# Other primitives — light sanity checks. The render layer can't catch
# shape bugs here, so we anchor each to one assertion.
# ---------------------------------------------------------------------------


def test_calibration_buckets_split_by_arm(populated_db):
    buckets = q.calibration_buckets(q.list_theses())
    labels = {b.label for b in buckets}
    assert labels == {"llm", "random"}


def test_pinball_points_only_closed(populated_db):
    pts = q.pinball_points(q.list_theses())
    # All three theses are closed in the fixture.
    assert len(pts) == 3
    assert {p.label for p in pts} == {"llm", "random"}
    # conviction_delta = conviction - 50
    nvda = next(p for p in pts if p.outcome == ThesisOutcome.CORRECT.value and p.label == "llm")
    assert nvda.conviction_delta == pytest.approx(22.0)


def test_tag_concentration_window(populated_db):
    pts = q.tag_concentration(q.list_theses(), days=30, top_n=3)
    # 30 days requested → 30 buckets.
    assert len(pts) == 30
    # ai_capex shows up in the totals (both NVDA-bullish and AAPL-bullish
    # have it as dominant tag).
    found = False
    for p in pts:
        if "ai_capex" in p.counts and p.counts["ai_capex"] > 0:
            found = True
            break
    assert found


def test_cumulative_pnl_split_by_arm(populated_db):
    rows = q.list_theses()
    pts = q.cumulative_pnl(rows, days=30)
    final = pts[-1]
    # llm net: +180 (NVDA) + (-50) (paired) = +130
    assert final.cum_pnl_by_label["llm"] == pytest.approx(130.0)
    # random net: +48 (AAPL: (210-200)*5 - 2)
    assert final.cum_pnl_by_label["random"] == pytest.approx(48.0)


def test_tag_regime_heatmap_min_n_greys_low_cells(populated_db):
    cells = q.tag_regime_heatmap(q.list_theses(), min_n=5)
    # Every cell in the fixture has n < 5, so win_rate should be None
    # regardless of how many wins there are.
    assert all(c.win_rate is None for c in cells if c.n > 0)


def test_eval_history_missing_dir_returns_empty(tmp_path):
    pts = q.eval_history(root=tmp_path / "does-not-exist")
    assert pts == []


def test_eval_history_reads_jsonl(tmp_path):
    today = date.today()
    p = tmp_path / f"{today.isoformat()}.jsonl"
    p.write_text(
        '{"date":"X","premise_f1":0.7,"sentiment_brier":0.2}\n'
        '{"date":"X","premise_f1":0.75,"sentiment_brier":0.18}\n'
    )
    pts = q.eval_history(days=1, root=tmp_path)
    assert len(pts) == 1
    # Latest row on the date wins.
    assert pts[0].premise_f1 == pytest.approx(0.75)


def test_bootstrap_diff_p_obvious_case():
    # Strongly different proportions → p near 0.
    a = [1] * 90 + [0] * 10
    b = [0] * 90 + [1] * 10
    p = q.bootstrap_diff_p(a, b, iters=200)
    assert p < 0.05


def test_bootstrap_diff_p_identical():
    a = [1, 0, 1, 0]
    b = [1, 0, 1, 0]
    # Observed diff = 0 → returns 1.0 by convention.
    assert q.bootstrap_diff_p(a, b, iters=50) == 1.0
