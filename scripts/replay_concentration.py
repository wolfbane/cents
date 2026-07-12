#!/usr/bin/env python
"""Replay a gathered week of factory candidates through the v0.12 concentration
gate (per-arm ledger + ambient-tag exemption) to see how each arm *behaves*
once the cap stops degenerating the book — without re-running a single LLM call.

Why this works: ``shadow_opens`` already recorded every classified candidate the
engine rejected (with its premise tags, direction and timestamp), and ``theses``
holds the ones that opened. Together they ARE the per-arm candidate stream. The
v0.12 fix is strictly *more permissive* (it only ever removes a concentration
block), so every original open still opens and some ``concentration_cap`` rejects
now pass. We replay the concentration decision chronologically, growing a
simulated per-arm book and respecting ``max_new_per_run``, using the exact
functions the live engine now calls (imported below) so the replay can't drift
from production behaviour.

Scope / honesty caveats (printed in the footer too):
  * Only the concentration gate is re-decided. below_threshold / no_price /
    hedge_* rejects stay rejected (those gates are unchanged).
  * Second-order budget/preemption effects of the extra opens are ignored. With
    budget_usd=100k and ~$5k positions the gross cap never binds in this window,
    so max_new_per_run is the only binding constraint and it IS respected.
  * This is BEHAVIOUR, not realized P&L: the positions are days old, so the
    30/60-day forward-return horizon has not elapsed. No returns are claimed.

Read-only against ~/.cents/data/pilot.db. Usage: python scripts/replay_concentration.py
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from cents.factory.config import load_factory_config
from cents.factory.premise import compute_ambient_tags, premise_concentration_exceeded

DB = Path.home() / ".cents/data/pilot.db"
EXP = "b14be90a"  # pilot_v2


def _loads(raw, default):
    try:
        v = json.loads(raw) if raw else default
        return v if v is not None else default
    except (json.JSONDecodeError, TypeError):
        return default


def _date(ts: str) -> str:
    return ts[:10]


def load_candidates(con):
    """Unified candidate stream: opened theses + rejected shadow rows."""
    cands = []
    for r in con.execute(
        "SELECT symbol, orchestrator_label arm, premise_tags, premise_direction, "
        "created_at FROM theses WHERE experiment_id=? ",
        (EXP,),
    ):
        cands.append({
            "symbol": r["symbol"], "arm": r["arm"],
            "tags": _loads(r["premise_tags"], []),
            "dir": _loads(r["premise_direction"], {}),
            "created_at": r["created_at"], "disp": "opened",
        })
    for r in con.execute(
        "SELECT symbol, orchestrator_label arm, premise_tags, premise_direction, "
        "created_at, reason FROM shadow_opens WHERE experiment_id=? ",
        (EXP,),
    ):
        cands.append({
            "symbol": r["symbol"], "arm": r["arm"],
            "tags": _loads(r["premise_tags"], []),
            "dir": _loads(r["premise_direction"], {}),
            "created_at": r["created_at"], "disp": r["reason"],
        })
    cands.sort(key=lambda c: c["created_at"])
    return cands


def ambient_for(cands, arm, day, cfg):
    """Mirror engine._arm_ambient_tags: tags over the arm's classified
    candidates (opened + tagged shadow) in the trailing window ending `day`."""
    end = datetime.fromisoformat(day + "T23:59:59")
    cutoff = end - timedelta(days=cfg.ambient_lookback_days)
    tag_lists = [
        c["tags"] for c in cands
        if c["arm"] == arm and c["tags"]
        and cutoff <= datetime.fromisoformat(c["created_at"]) <= end
    ]
    return compute_ambient_tags(
        tag_lists,
        threshold=cfg.ambient_tag_prevalence,
        min_sample=cfg.ambient_min_sample,
    )


def main():
    cfg = load_factory_config()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cands = load_candidates(con)

    # One scheduled run per (arm, day). Group + keep chronological order.
    runs: dict[tuple[str, str], list] = defaultdict(list)
    for c in cands:
        runs[(c["arm"], _date(c["created_at"]))].append(c)

    sim_book: dict[str, list] = defaultdict(list)   # arm -> [(tags, dir)] cumulative
    actual: dict[tuple[str, str], int] = defaultdict(int)
    fixed: dict[tuple[str, str], int] = defaultdict(int)
    gained: dict[tuple[str, str], list] = defaultdict(list)
    ambient_seen: dict[str, frozenset] = {}

    # Real opens per run define the spare rate-limit capacity the fix can use:
    # the fix only ever UNLOCKS blocked candidates into a run's leftover slots,
    # so no run can exceed max_new_per_run (reality already opened up to it).
    real_opens = defaultdict(int)
    for c in cands:
        if c["disp"] == "opened":
            real_opens[(c["arm"], _date(c["created_at"]))] += 1

    for (arm, day) in sorted(runs):
        ambient = ambient_for(cands, arm, day, cfg)
        ambient_seen[arm] = ambient  # last (most recent) wins for the summary
        spare = max(0, cfg.max_new_per_run - real_opens[(arm, day)])
        unlocked = 0
        for c in runs[(arm, day)]:  # chronological within the run
            if c["disp"] == "opened":
                sim_book[arm].append((c["tags"], c["dir"]))
                actual[(arm, day)] += 1
                fixed[(arm, day)] += 1
            elif c["disp"] == "concentration_cap" and unlocked < spare:
                blocked = premise_concentration_exceeded(
                    c["tags"], c["dir"], sim_book[arm],
                    cfg.max_per_premise_tag, ambient_tags=ambient,
                )
                if not blocked:
                    sim_book[arm].append((c["tags"], c["dir"]))
                    fixed[(arm, day)] += 1
                    gained[(arm, day)].append(c["symbol"])
                    unlocked += 1
            # other reasons (below_threshold/no_price/hedge_*): gate unchanged

    arms = sorted({a for a, _ in runs})
    days = sorted({d for _, d in runs})

    print("=" * 72)
    print("CONCENTRATION-GATE REPLAY — pilot_v2 (b14be90a)")
    print(f"config: per_arm={cfg.concentration_per_arm}  cap={cfg.max_per_premise_tag}  "
          f"ambient>= {cfg.ambient_tag_prevalence:.0%} (min n={cfg.ambient_min_sample}, "
          f"{cfg.ambient_lookback_days}d)  max_new_per_run={cfg.max_new_per_run}")
    print("=" * 72)
    print(f"\n{'day':<12}" + "".join(f"{a+' act/fix':>16}" for a in arms))
    tot_act = defaultdict(int)
    tot_fix = defaultdict(int)
    for day in days:
        row = f"{day:<12}"
        for arm in arms:
            a, f = actual[(arm, day)], fixed[(arm, day)]
            tot_act[arm] += a
            tot_fix[arm] += f
            row += f"{f'{a}/{f}':>16}"
        print(row)
    print("-" * (12 + 16 * len(arms)))
    print(f"{'TOTAL':<12}" + "".join(f"{f'{tot_act[a]}/{tot_fix[a]}':>16}" for a in arms))

    for arm in arms:
        amb = sorted(ambient_seen.get(arm, []))
        print(f"\n[{arm}] ambient tags exempted (most recent run): "
              f"{', '.join(amb) if amb else '(none)'}")
        names = [s for (a, _), syms in gained.items() if a == arm for s in syms]
        if names:
            from collections import Counter
            cnt = Counter(names)
            print(f"[{arm}] +{len(names)} opens unlocked across the week "
                  f"(by name): {dict(cnt.most_common())}")
        else:
            print(f"[{arm}] no additional opens unlocked.")

    print("\n" + "=" * 72)
    print("BEHAVIOUR, not realized P&L: positions are days old; the 30/60d "
          "forward-return\nhorizon has not elapsed. Only the concentration gate "
          "is re-decided; other\ngates (threshold/price/hedge) and second-order "
          "budget effects are unchanged.")
    print("=" * 72)


if __name__ == "__main__":
    main()
