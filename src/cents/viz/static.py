"""Matplotlib renderers for ``cents report``.

Each function:
1. Takes already-prepared data from ``cents.viz.queries``.
2. Writes a PNG to ``out_dir / "<name>.png"``.
3. Writes a JSON sidecar to ``out_dir / "<name>.json"`` so the figure
   is reproducible from data alone (audit) and the Starlight site can
   pick up the underlying numbers for accessible tables.

Matplotlib is imported lazily — the module-level import is gated behind
``_lazy_plt()`` so ``cents.viz.queries`` can be imported and tested
without the ``[viz]`` extra installed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from cents.viz.queries import (
    CalibrationBucket,
    CohortMetrics,
    HeatCell,
    PinballPoint,
    PnlPoint,
    TagSeriesPoint,
    ThesisRow,
    bootstrap_diff_p,
)
from cents.models.thesis import ThesisOutcome, ThesisStatus


# Matplotlib import is wrapped so a missing extra fails at call time,
# not at import time. ``cents.viz.queries`` must remain importable in
# the no-viz path.
# Module-level cache. ``matplotlib.use("Agg")`` must be called BEFORE
# pyplot is imported, so a second call from a later render_* function
# warns (or errors on newer matplotlib). Cache on first hit and reuse.
_plt = None


def _lazy_plt():
    global _plt
    if _plt is not None:
        return _plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "cents report needs the [viz] extra: pip install -e '.[viz]'"
        ) from exc
    _plt = plt
    return _plt


def _write_sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, default=str))


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "value"):  # str-Enum
        return obj.value
    return obj


# ---------------------------------------------------------------------------
# Chart 3: 2×2 hit rate
# ---------------------------------------------------------------------------


def render_hit_rate_2x2(
    rows_by_orchestrator_cohort: dict[tuple[str, str], CohortMetrics],
    rows: Sequence[ThesisRow],
    *,
    out_dir: Path,
    name: str = "hit_rate_2x2",
) -> Path:
    """Four cells: (llm|random) × (directional|neutral).

    Each cell prints the hit rate, Wilson CI, and N. The footer shows
    the diff vs random and a bootstrap p-value computed from the row
    outcomes — this is the headline chart that answers the experiment's
    primary question, so we want the significance bookkeeping on it.
    """
    plt = _lazy_plt()
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    arms = ("llm", "random")
    cohorts = ("directional", "neutral")

    for i, arm in enumerate(arms):
        for j, cohort in enumerate(cohorts):
            ax = axes[i][j]
            cell = rows_by_orchestrator_cohort.get((arm, cohort))
            if cell is None or cell.judged == 0:
                ax.text(
                    0.5, 0.5,
                    f"{arm} · {cohort}\n(no data)",
                    ha="center", va="center", fontsize=11, color="#888",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            wr = cell.win_rate or 0.0
            ci = cell.win_rate_ci
            ax.barh(
                [0], [wr], color="#2c7fb8" if arm == "llm" else "#bdbdbd",
                height=0.5,
            )
            if ci is not None:
                lo, hi = ci
                ax.errorbar([wr], [0], xerr=[[wr - lo], [hi - wr]],
                            ecolor="black", capsize=4, fmt="none")
            ax.set_xlim(0.0, 1.0)
            ax.set_yticks([])
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.axvline(0.5, color="#cccccc", linestyle=":", linewidth=1)
            ax.set_title(
                f"{arm} · {cohort}\n{wr*100:.1f}%  (n={cell.judged})",
                fontsize=11,
            )

    # Compute the headline diff + p with the raw row outcomes so it's
    # not an artifact of the binned hit_rate.
    def _outcomes(arm: str, cohort: str) -> list[int]:
        out = []
        for r in rows:
            if r.status != ThesisStatus.CLOSED:
                continue
            if r.outcome not in (ThesisOutcome.CORRECT, ThesisOutcome.INCORRECT, ThesisOutcome.PARTIAL):
                continue
            if r.orchestrator_label != arm:
                continue
            if r.cohort.value != cohort:
                continue
            out.append(1 if r.outcome == ThesisOutcome.CORRECT else 0)
        return out

    pieces = []
    for cohort in cohorts:
        a = _outcomes("llm", cohort)
        b = _outcomes("random", cohort)
        if a and b:
            d = (sum(a) / len(a)) - (sum(b) / len(b))
            p = bootstrap_diff_p(a, b, iters=500)
            pieces.append(f"{cohort}: Δ={d*100:+.1f}pp  p≈{p:.2f}")
    if pieces:
        fig.suptitle("Hit rate: orchestrator × cohort", fontsize=13)
        fig.text(0.5, 0.02, "  ·  ".join(pieces), ha="center", fontsize=10, color="#444")

    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)

    _write_sidecar(
        out_dir / f"{name}.json",
        {"cells": {f"{k[0]}×{k[1]}": v for k, v in rows_by_orchestrator_cohort.items()}},
    )
    return out


# ---------------------------------------------------------------------------
# Chart 4: calibration reliability
# ---------------------------------------------------------------------------


def render_calibration_reliability(
    buckets: Sequence[CalibrationBucket],
    *,
    out_dir: Path,
    name: str = "calibration_reliability",
) -> Path:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(7, 6))

    groups: dict[str, list[CalibrationBucket]] = {}
    for b in buckets:
        groups.setdefault(b.label, []).append(b)

    colors = {"llm": "#2c7fb8", "random": "#969696"}
    for label, bs in groups.items():
        bs = sorted(bs, key=lambda x: x.bin_centre)
        xs = [b.bin_centre for b in bs]
        ys = [b.realized for b in bs]
        sizes = [20 + 8 * b.n for b in bs]  # marker size encodes N
        ax.scatter(xs, ys, s=sizes, alpha=0.7, label=label,
                   color=colors.get(label, "#fdae6b"))

    ax.plot([0, 1], [0, 1], linestyle="--", color="#888", linewidth=1, label="perfect")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("predicted p (calibrated_p_correct)")
    ax.set_ylabel("realized hit rate")
    ax.set_title("Calibration reliability")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    out = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    _write_sidecar(out_dir / f"{name}.json", {"buckets": list(buckets)})
    return out


# ---------------------------------------------------------------------------
# Chart 5: invalidation funnel
# ---------------------------------------------------------------------------


def render_invalidation_funnel(
    rows: Sequence[ThesisRow],
    *,
    out_dir: Path,
    name: str = "invalidation_funnel",
) -> Path:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(10, 5))

    arms = ("llm", "random")
    outcomes_order = (
        ThesisOutcome.INVALIDATED,
        ThesisOutcome.CORRECT,
        ThesisOutcome.INCORRECT,
        ThesisOutcome.PARTIAL,
        ThesisOutcome.UNCLEAR,
        ThesisOutcome.PREEMPTED,
    )
    colors = {
        ThesisOutcome.INVALIDATED: "#d7301f",
        ThesisOutcome.CORRECT: "#31a354",
        ThesisOutcome.INCORRECT: "#969696",
        ThesisOutcome.PARTIAL: "#fdae6b",
        ThesisOutcome.UNCLEAR: "#bdbdbd",
        ThesisOutcome.PREEMPTED: "#dadaeb",
    }

    sidecar: dict[str, Any] = {}
    for i, arm in enumerate(arms):
        arm_rows = [r for r in rows if r.orchestrator_label == arm]
        opened = len(arm_rows)
        counts = {o: sum(1 for r in arm_rows if r.outcome == o) for o in outcomes_order}
        # Still-open theses (outcome=None) don't fit into outcomes_order
        # but DO contribute to ``opened``. Surface them as a distinct
        # "open" bucket so the bar length reconciles with n=opened.
        open_n = sum(1 for r in arm_rows if r.outcome is None)
        sidecar[arm] = {
            "opened": opened,
            "open": open_n,
            **{k.value: v for k, v in counts.items()},
        }

        left = 0.0
        for outcome in outcomes_order:
            n = counts[outcome]
            if n == 0:
                continue
            ax.barh(i, n, left=left, color=colors[outcome], edgecolor="white", label=outcome.value if i == 0 else None)
            ax.text(left + n / 2, i, f"{outcome.value}\n{n}", ha="center", va="center", fontsize=8, color="white")
            left += n
        if open_n:
            ax.barh(i, open_n, left=left, color="#f7f7f7",
                    edgecolor="#999", linestyle=":", label="open" if i == 0 else None)
            ax.text(left + open_n / 2, i, f"open\n{open_n}",
                    ha="center", va="center", fontsize=8, color="#444")
        ax.text(-2, i, f"{arm}  n={opened}", ha="right", va="center", fontsize=10)

    ax.set_yticks([])
    ax.set_xlabel("theses (opened, by outcome)")
    ax.set_title("Invalidation funnel by arm")

    # Footer: INVALIDATED ratio (the CLAUDE.md "~2.5× expected" check).
    inv = {arm: sidecar[arm].get(ThesisOutcome.INVALIDATED.value, 0) for arm in arms}
    opened = {arm: sidecar[arm]["opened"] for arm in arms}
    if all(opened.values()) and inv["random"]:
        ratio = (inv["llm"] / opened["llm"]) / (inv["random"] / opened["random"])
        fig.text(
            0.5, 0.02,
            f"INVALIDATED share — llm {inv['llm']/opened['llm']*100:.1f}%  ·  random {inv['random']/opened['random']*100:.1f}%  →  {ratio:.1f}× (expected ≈2.5×)",
            ha="center", fontsize=10, color="#444",
        )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    _write_sidecar(out_dir / f"{name}.json", sidecar)
    return out


# ---------------------------------------------------------------------------
# Chart 6: tag concentration over time
# ---------------------------------------------------------------------------


def render_tag_concentration(
    points: Sequence[TagSeriesPoint],
    invalidations: Sequence[tuple[str, str]],
    *,
    out_dir: Path,
    name: str = "tag_concentration",
) -> Path:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(11, 6))

    if not points:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        out = out_dir / f"{name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        _write_sidecar(out_dir / f"{name}.json", {"points": []})
        return out

    tags: list[str] = []
    seen: set[str] = set()
    for p in points:
        for t in p.counts:
            if t not in seen:
                seen.add(t)
                tags.append(t)

    days = [p.day for p in points]
    series = {t: [p.counts.get(t, 0) for p in points] for t in tags}

    ax.stackplot(
        range(len(days)),
        *[series[t] for t in tags],
        labels=tags,
        alpha=0.85,
    )
    ax.set_xticks(range(0, len(days), max(1, len(days) // 8)))
    ax.set_xticklabels(
        [days[i] for i in range(0, len(days), max(1, len(days) // 8))],
        rotation=30, ha="right",
    )
    ax.set_ylabel("open theses (by dominant tag)")
    ax.legend(loc="upper left", fontsize=8)

    day_to_x = {d: i for i, d in enumerate(days)}
    for inv_day, msg in invalidations:
        if inv_day in day_to_x:
            ax.axvline(day_to_x[inv_day], color="#d7301f", alpha=0.5, linewidth=1)
            ax.text(day_to_x[inv_day], ax.get_ylim()[1] * 0.95, "⚡",
                    ha="center", va="top", fontsize=10, color="#d7301f")

    ax.set_title("Premise-tag concentration over time")
    fig.tight_layout()
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    _write_sidecar(
        out_dir / f"{name}.json",
        {"points": list(points), "invalidations": list(invalidations)},
    )
    return out


# ---------------------------------------------------------------------------
# Chart 7: cumulative P&L with regime overlay
# ---------------------------------------------------------------------------


def render_pnl_curve(
    points: Sequence[PnlPoint],
    *,
    regime_spans: Sequence[tuple[str, str, str]] = (),  # (start_day, end_day, label)
    out_dir: Path,
    name: str = "pnl_curve",
) -> Path:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(11, 5))

    if not points:
        ax.text(0.5, 0.5, "no closed theses yet", ha="center", va="center")
        out = out_dir / f"{name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        _write_sidecar(out_dir / f"{name}.json", {"points": []})
        return out

    labels: list[str] = []
    seen: set[str] = set()
    for p in points:
        for l in p.cum_pnl_by_label:
            if l not in seen:
                seen.add(l)
                labels.append(l)

    days = [p.day for p in points]
    colors = {"llm": "#2c7fb8", "random": "#969696"}
    for label in labels:
        ys = [p.cum_pnl_by_label.get(label, 0.0) for p in points]
        ax.plot(range(len(days)), ys, label=label,
                color=colors.get(label, "#fdae6b"), linewidth=2)

    day_to_x = {d: i for i, d in enumerate(days)}
    regime_colors = {"risk-on": "#c7e9c0", "risk-off": "#fee0d2", "neutral": "#f0f0f0"}
    for start, end, lbl in regime_spans:
        s = day_to_x.get(start)
        e = day_to_x.get(end)
        if s is not None and e is not None:
            ax.axvspan(s, e, color=regime_colors.get(lbl, "#eee"), alpha=0.4)

    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_xticks(range(0, len(days), max(1, len(days) // 8)))
    ax.set_xticklabels(
        [days[i] for i in range(0, len(days), max(1, len(days) // 8))],
        rotation=30, ha="right",
    )
    ax.set_ylabel("cumulative net P&L ($)")
    ax.set_title("Cumulative P&L with regime overlay")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    _write_sidecar(
        out_dir / f"{name}.json",
        {"points": list(points), "regime_spans": [list(s) for s in regime_spans]},
    )
    return out


# ---------------------------------------------------------------------------
# Chart 8: pinball scatter
# ---------------------------------------------------------------------------


def render_pinball(
    points: Sequence[PinballPoint],
    *,
    out_dir: Path,
    name: str = "pinball",
) -> Path:
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(10, 6))

    if not points:
        ax.text(0.5, 0.5, "no closed theses", ha="center", va="center")
        out = out_dir / f"{name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        _write_sidecar(out_dir / f"{name}.json", {"points": []})
        return out

    # y position by outcome bucket.
    y_for = {
        ThesisOutcome.CORRECT.value: 1.0,
        ThesisOutcome.PARTIAL.value: 0.5,
        ThesisOutcome.UNCLEAR.value: 0.0,
        ThesisOutcome.INCORRECT.value: -1.0,
        ThesisOutcome.INVALIDATED.value: -1.5,
        ThesisOutcome.PREEMPTED.value: -2.0,
    }
    for arm, marker, color in (("llm", "o", "#2c7fb8"), ("random", "o", "#bdbdbd")):
        xs = [p.conviction_delta for p in points if p.label == arm]
        ys = [y_for.get(p.outcome, 0.0) for p in points if p.label == arm]
        ss = [10 + min(200, abs(p.pnl)) for p in points if p.label == arm]
        if not xs:
            continue
        face = color if arm == "llm" else "none"
        ax.scatter(xs, ys, s=ss, alpha=0.55 if arm == "llm" else 0.8,
                   marker=marker, facecolors=face, edgecolors=color,
                   linewidths=1.0, label=arm)

    ax.axvline(0, color="#888", linewidth=0.5)
    ax.axhline(0, color="#888", linewidth=0.5)
    ax.set_yticks(list(y_for.values()))
    ax.set_yticklabels(list(y_for.keys()))
    ax.set_xlabel("conviction_delta (= conviction − 50)")
    ax.set_title("Pinball: outcome by conviction (dot size = |pnl|)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    _write_sidecar(out_dir / f"{name}.json", {"points": list(points)})
    return out


# ---------------------------------------------------------------------------
# Chart 10: tag × regime heatmap
# ---------------------------------------------------------------------------


def render_tag_regime_heatmap(
    cells: Sequence[HeatCell],
    *,
    out_dir: Path,
    name: str = "tag_regime_heatmap",
) -> Path:
    plt = _lazy_plt()

    tags = sorted({c.tag for c in cells})
    regimes = sorted({c.regime for c in cells})
    if not tags or not regimes:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "no judged theses with tags", ha="center", va="center")
        out = out_dir / f"{name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        _write_sidecar(out_dir / f"{name}.json", {"cells": []})
        return out

    grid: list[list[float | None]] = []
    n_grid: list[list[int]] = []
    cell_by_key = {(c.tag, c.regime): c for c in cells}
    for tag in tags:
        row: list[float | None] = []
        n_row: list[int] = []
        for regime in regimes:
            c = cell_by_key.get((tag, regime))
            row.append(c.win_rate if c else None)
            n_row.append(c.n if c else 0)
        grid.append(row)
        n_grid.append(n_row)

    import numpy as np
    arr = np.array([[float("nan") if v is None else v for v in row] for row in grid])

    # Low-N cells encode as NaN. Set the colormap's "bad" colour to a
    # distinct light grey so they read as "not enough data" instead of
    # rendering as white (which on the RdYlGn colormap looks like a
    # mid-rate value).
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#e0e0e0")

    fig, ax = plt.subplots(figsize=(1.4 * len(regimes) + 4, 0.55 * len(tags) + 2))
    im = ax.imshow(
        np.ma.masked_invalid(arr), cmap=cmap, vmin=0.3, vmax=0.7, aspect="auto",
    )
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels(tags)
    for i in range(len(tags)):
        for j in range(len(regimes)):
            wr = grid[i][j]
            n = n_grid[i][j]
            if wr is None:
                ax.text(j, i, f"n={n}", ha="center", va="center", fontsize=8, color="#888")
            else:
                ax.text(j, i, f"{wr*100:.0f}%\nn={n}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("hit rate")
    ax.set_title("Hit rate by premise tag × regime")
    fig.tight_layout()
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    _write_sidecar(out_dir / f"{name}.json", {"cells": list(cells)})
    return out
