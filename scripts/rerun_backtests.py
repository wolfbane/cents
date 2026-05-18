"""Re-run cents backtests against a snapshot of (symbol, start, end) tuples.

Use case: a data-layer fix (e.g. switching Alpaca to split-adjusted bars in
commit bc6c140) invalidates every persisted `BacktestSignal.forward_returns`
row whose evaluation window crosses a corporate action. The repo-local SQLite
DB at ``~/.cents/data/cents.db`` is small enough that surgical detection is
not worth the code — nuke the whole `backtests` table and re-run.

Workflow:

1. ``python scripts/rerun_backtests.py --snapshot snapshot.json``
   Captures the current `cents backtest list` to ``snapshot.json``. Dedupes
   on ``(symbol, start_date, end_date)`` so a window run N times shows up
   once. Run this BEFORE deleting anything.

2. Nuke the existing rows (loop over ``snapshot['all_ids']`` calling
   ``cents backtest delete <id> --yes``). Left as a manual step on purpose
   so an operator must opt in to the destructive action.

3. ``python scripts/rerun_backtests.py --snapshot snapshot.json --rerun``
   Sequentially re-runs ``cents backtest run`` for each unique tuple.
   Writes per-tuple logs to ``--logdir`` and a results JSON to
   ``--results``. Hits live Alpaca + FMP, so runs sequentially with a
   configurable sleep between calls to avoid rate-limit pressure.

This script is intentionally a thin orchestrator around the CLI — it does
not import any ``cents`` Python internals.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def capture_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """Dump current `cents backtest list` and dedupe to unique tuples."""
    res = subprocess.run(
        ["cents", "backtest", "list", "--output", "json"],
        capture_output=True, text=True, check=True,
    )
    rows = json.loads(res.stdout)

    groups: dict[tuple[str, str, str], list[str]] = {}
    for r in rows:
        key = (r["symbol"], r["start_date"], r["end_date"])
        groups.setdefault(key, []).append(r["id"])

    unique = [
        {"symbol": s, "start_date": sd, "end_date": ed, "all_ids": ids}
        for (s, sd, ed), ids in sorted(groups.items())
    ]
    snapshot = {
        "total_rows": len(rows),
        "unique_tuples": len(unique),
        "unique": unique,
        "all_ids": [r["id"] for r in rows],
    }
    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    print(
        f"Snapshot: {len(rows)} rows -> {len(unique)} unique tuples "
        f"(written to {snapshot_path})"
    )
    return snapshot


def rerun(
    snapshot_path: Path,
    logdir: Path,
    results_path: Path,
    sleep_between: float,
) -> int:
    """Re-run every unique tuple in the snapshot via `cents backtest run`."""
    snapshot = json.loads(snapshot_path.read_text())
    unique = snapshot["unique"]
    logdir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for i, u in enumerate(unique, 1):
        sym, sd, ed = u["symbol"], u["start_date"], u["end_date"]
        tag = f"{sym}_{sd}_{ed}"
        log = logdir / f"{tag}.log"
        print(f"[{i}/{len(unique)}] {sym} {sd} -> {ed} ...", flush=True)
        t0 = time.time()
        res = subprocess.run(
            ["cents", "backtest", "run", sym, "--start", sd, "--end", ed,
             "--output", "json"],
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        log.write_text(
            f"STDOUT\n------\n{res.stdout}\n\n"
            f"STDERR\n------\n{res.stderr}\n"
        )
        entry: dict[str, Any] = {
            "symbol": sym,
            "start": sd,
            "end": ed,
            "returncode": res.returncode,
            "elapsed_s": round(elapsed, 1),
        }
        if res.returncode != 0:
            entry["error"] = res.stderr.strip()[-500:]
            print(
                f"  FAIL ({elapsed:.1f}s): {entry['error'][:200]}",
                flush=True,
            )
        else:
            print(f"  OK  ({elapsed:.1f}s)", flush=True)
        results.append(entry)
        results_path.write_text(json.dumps(results, indent=2))
        time.sleep(sleep_between)

    ok = sum(1 for r in results if r["returncode"] == 0)
    print(f"\nDone: {ok}/{len(results)} succeeded")
    return 0 if ok == len(results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("/tmp/cents_backtest_snapshot.json"),
        help="Path to the snapshot JSON file.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Re-run every tuple in the snapshot. Without this flag the "
            "script only captures the snapshot."
        ),
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("/tmp/cents_rerun_logs"),
        help="Directory for per-tuple run logs.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/tmp/cents_rerun_results.json"),
        help="Path for the cumulative results JSON.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between runs (bump if you see 429s).",
    )
    args = parser.parse_args(argv)

    if args.rerun:
        if not args.snapshot.exists():
            print(
                f"Snapshot not found at {args.snapshot}; run without "
                "--rerun first to capture it.",
                file=sys.stderr,
            )
            return 2
        return rerun(args.snapshot, args.logdir, args.results, args.sleep)

    capture_snapshot(args.snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
