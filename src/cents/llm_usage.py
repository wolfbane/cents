"""LLM usage recording, pre-flight cost cap, and blob-store provenance.

Every call site that hits Anthropic's API funnels through three helpers:

1. ``check_cost_cap(call_kwargs, agent, operation)`` — raises
   :class:`cents.exceptions.CostCapExceeded` if running this call would push
   the active per-run or per-day spend over its cap. Called PRE-call so the
   offending API request is never made.
2. ``record_llm_usage(response, ...)`` — best-effort POST-call write of one
   row to ``llm_usage``. Returns the row's id so call sites can stamp it onto
   downstream Evidence rows. Failures are logged at debug and swallowed.
3. ``persist_call_blob(call_id, prompt, input_text, output_text, model)`` —
   append-only snapshot of the full prompt/input/output to a gzipped JSON
   file at ``~/.cents/data/llm_calls/YYYYMMDD/<id>.json.gz``.

A ``cost_cap(...)`` context manager binds a per-run cap (e.g. from
``cents factory run --max-cost-usd 1.50``) into the module-level singleton
so the four wrapped LLM call sites pick it up without explicit plumbing.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import threading
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterator

from cents.db import LLMUsageRepository
from cents.exceptions import CostCapExceeded
from cents.models import LLMUsage
from cents.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


# --- Token estimation heuristics --------------------------------------------

# Anthropic's BPE produces ~4 input chars per token on English prose. This is a
# generous heuristic — we round UP so the cap-check trips before the real call
# rather than after.
_CHARS_PER_TOKEN = 4


def _estimate_input_tokens_from_messages(messages: list[dict]) -> int:
    """Estimate input token count from an Anthropic messages list."""
    char_count = 0
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, str):
            char_count += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    char_count += len(str(block.get("text") or ""))
                else:
                    char_count += len(str(block))
    return math.ceil(char_count / _CHARS_PER_TOKEN)


def peek_cost_usd(call_kwargs: dict[str, Any]) -> float:
    """Estimate the USD cost of an Anthropic call BEFORE it is made.

    ``call_kwargs`` is the kwargs dict that would be passed to
    ``client.messages.create(**call_kwargs)``. Uses ``max_tokens`` as the
    output ceiling and a 4-chars/token heuristic over message content for
    the input estimate.

    Always returns a positive float; falls back to a small non-zero value
    when the model is unknown so unknown-model calls still consume cap
    budget proportionally to their max_tokens.
    """
    model = call_kwargs.get("model") or ""
    max_tokens = int(call_kwargs.get("max_tokens") or 0)
    input_tokens = _estimate_input_tokens_from_messages(call_kwargs.get("messages") or [])

    cost = estimate_cost_usd(model, input_tokens, max_tokens)
    if cost is None:
        # Unknown model: charge as if it were the most expensive model we know
        # so an unbounded loop still trips the cap. Sonnet rates ($3 in / $15 out).
        cost = (input_tokens * 3.0 + max_tokens * 15.0) / 1_000_000.0
    # Floor at a tiny non-zero so tests asserting "positive" hold even for
    # zero-token estimates (which shouldn't happen in practice).
    return max(cost, 1e-9)


# --- Daily cap query --------------------------------------------------------


def today_cost_usd(*, today: date | None = None) -> float:
    """Sum priced cost over today's `llm_usage` rows. Returns 0.0 on error.

    Queries with ``since=midnight, limit=None`` so the cap remains accurate
    once daily call volume exceeds whatever in-memory limit a global scan
    would impose. Pre-Batch-I this used ``list_recent(limit=10000)`` with no
    ``since`` filter, which silently undercounted on the days that needed it
    most (high-volume runs are exactly when the cap should bite).
    """
    anchor = today or date.today()
    midnight = datetime.combine(anchor, time.min)
    try:
        rows = LLMUsageRepository().list_recent(since=midnight, limit=None)
    except Exception as e:  # noqa: BLE001 — bookkeeping must not crash callers
        logger.debug("today_cost_usd failed: %s", e)
        return 0.0
    total = 0.0
    for row in rows:
        cost = estimate_cost_usd(
            row.model,
            row.input_tokens or 0,
            row.output_tokens or 0,
            cache_read=row.cache_read_input_tokens or 0,
            cache_write=row.cache_creation_input_tokens or 0,
        )
        if cost is not None:
            total += cost
    return total


# --- Cost cap singleton -----------------------------------------------------


class _CostCapState:
    """Per-process running cost ledger + active per-run cap.

    Threads share the singleton, so the cumulative cost is guarded by a lock.
    Tests can call ``reset()`` between cases.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_cap_usd: float | None = None
        self._run_cost_usd: float = 0.0

    @property
    def run_cap_usd(self) -> float | None:
        return self._run_cap_usd

    @property
    def run_cost_usd(self) -> float:
        return self._run_cost_usd

    def snapshot(self) -> tuple[float | None, float]:
        """Atomic (cap, cost) read under the same lock that write paths use.

        cents-acb: previously check_cost_cap read cap + cost independently,
        admitting a torn read where two concurrent calls both saw stale costs
        and both passed the cap check. This collapses the read to one
        lock-acquire so a CAS-style check-and-decide is consistent.
        """
        with self._lock:
            return self._run_cap_usd, self._run_cost_usd

    def set_run_cap(self, cap_usd: float | None) -> None:
        with self._lock:
            self._run_cap_usd = cap_usd
            self._run_cost_usd = 0.0

    def reset(self) -> None:
        self.set_run_cap(None)

    def add_actual_cost(self, cost_usd: float) -> None:
        if cost_usd <= 0:
            return
        with self._lock:
            self._run_cost_usd += cost_usd


_state = _CostCapState()


def current_run_spend_usd() -> float:
    return _state.run_cost_usd


def current_run_cap_usd() -> float | None:
    return _state.run_cap_usd


def reset_cost_cap_state() -> None:
    """Clear the per-run cap and accumulator. Intended for test isolation."""
    _state.reset()


@contextmanager
def cost_cap(max_cost_usd: float | None) -> Iterator[None]:
    """Bind a per-run cost cap into the module singleton for the block.

    ``max_cost_usd=None`` disables the per-run cap (the daily cap still
    applies). Nested ``cost_cap`` blocks reset the cap to the innermost value
    and restore the outer one on exit; this matches how the CLI uses it.
    """
    previous_cap = _state.run_cap_usd
    previous_cost = _state.run_cost_usd
    with _state._lock:
        _state._run_cap_usd = max_cost_usd
        _state._run_cost_usd = 0.0
    try:
        yield
    finally:
        with _state._lock:
            _state._run_cap_usd = previous_cap
            _state._run_cost_usd = previous_cost


def check_cost_cap(call_kwargs: dict[str, Any], *, agent: str, operation: str) -> None:
    """Raise :class:`CostCapExceeded` if the next call would exceed the active cap.

    Checks (in order):

    1. Per-run cap from :func:`cost_cap` / ``--max-cost-usd`` CLI flag.
    2. Per-day cap from ``max_llm_spend_usd_per_day`` config / env.

    Both caps reuse the same estimate (no double-counting between the
    in-memory run total and today's DB total).
    """
    estimate = peek_cost_usd(call_kwargs)

    # 1. Per-run cap — atomic (cap, spent) read so concurrent calls can't
    # both see stale `spent` and both pass the cap check (cents-acb).
    cap, spent = _state.snapshot()
    if cap is not None:
        projected = spent + estimate
        if projected > cap:
            raise CostCapExceeded(
                f"LLM cost cap would be exceeded: agent={agent} op={operation} "
                f"cap=${cap:.4f} spent=${spent:.4f} "
                f"next≈${estimate:.4f}",
                cap_kind="run",
                cap_usd=cap,
                current_usd=spent,
                next_call_estimate_usd=estimate,
            )

    # 2. Daily cap — only query when configured, since it hits the DB.
    daily_cap = _get_daily_cap()
    if daily_cap is not None:
        spent_today = today_cost_usd()
        if spent_today + estimate > daily_cap:
            raise CostCapExceeded(
                f"Daily LLM cost cap would be exceeded: agent={agent} "
                f"op={operation} cap=${daily_cap:.4f} spent_today="
                f"${spent_today:.4f} next≈${estimate:.4f}",
                cap_kind="daily",
                cap_usd=daily_cap,
                current_usd=spent_today,
                next_call_estimate_usd=estimate,
            )


def get_daily_cap() -> float | None:
    """Public accessor for the configured daily LLM spend cap.

    Resolves from ``CENTS_MAX_LLM_SPEND_USD_PER_DAY`` env first, then
    ``max_llm_spend_usd_per_day`` in ``~/.cents/config.toml``. Returns
    ``None`` when no cap is configured. (The daily cap is a global config
    setting, NOT a per-experiment factory.toml setting — putting it in
    factory.toml silently does nothing.)
    """
    return _get_daily_cap()


def _get_daily_cap() -> float | None:
    """Resolve the daily cap from env first, then Settings — env wins for tests."""
    env_val = os.environ.get("CENTS_MAX_LLM_SPEND_USD_PER_DAY")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            return None
    # Avoid an import cycle (config.py is light, but be safe).
    from cents.config import get_settings

    try:
        return get_settings().max_llm_spend_usd_per_day
    except Exception:  # noqa: BLE001 — bad config must never crash callers
        return None


# --- Recording --------------------------------------------------------------


def record_llm_usage(
    response: Any,
    agent: str,
    operation: str,
    context: str | None = None,
) -> str | None:
    """Persist a usage record from an Anthropic Message response.

    Returns the row id so callers can stamp it onto downstream Evidence rows.
    Returns ``None`` on failure (DB locked, table missing, …) — bookkeeping
    must never break user-facing flows.

    Also adds the actual (priced) cost to the per-run cap accumulator so
    subsequent ``check_cost_cap`` calls reflect real spend, not estimates.
    """
    try:
        usage = response.usage
        row = LLMUsage(
            model=response.model or "",
            agent=agent,
            operation=operation,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
            context=context,
        )
        LLMUsageRepository().create(row)

        # Update the per-run cap accumulator with priced actuals.
        cost = estimate_cost_usd(
            row.model,
            row.input_tokens,
            row.output_tokens,
            cache_read=row.cache_read_input_tokens,
            cache_write=row.cache_creation_input_tokens,
        )
        if cost is not None:
            _state.add_actual_cost(cost)
        return row.id
    except Exception as e:  # noqa: BLE001 — bookkeeping must never raise
        logger.debug("record_llm_usage failed (agent=%s op=%s): %s", agent, operation, e)
        return None


# --- Blob store -------------------------------------------------------------


def _blob_store_root() -> Path:
    """Root path for append-only LLM call snapshots."""
    override = os.environ.get("CENTS_LLM_BLOB_DIR")
    if override:
        return Path(override)
    # Default sits alongside the DB at ~/.cents/data/llm_calls/.
    return Path.home() / ".cents" / "data" / "llm_calls"


def blob_path_for(call_id: str, *, when: datetime | None = None) -> Path:
    """Return the on-disk path for a given call_id's snapshot."""
    anchor = when or datetime.now()
    return _blob_store_root() / anchor.strftime("%Y%m%d") / f"{call_id}.json.gz"


def persist_call_blob(
    call_id: str | None,
    *,
    prompt: str,
    input_text: str,
    output_text: str,
    model: str,
    agent: str,
    operation: str,
    when: datetime | None = None,
) -> Path | None:
    """Snapshot the full prompt+input+output to the gzipped blob store.

    Best-effort: returns the written path on success, or None on failure.
    Never raises. No secret material is included — the helper takes raw text
    only.
    """
    if not call_id:
        return None
    try:
        path = blob_path_for(call_id, when=when)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "call_id": call_id,
            "agent": agent,
            "operation": operation,
            "model": model,
            "captured_at": (when or datetime.now()).isoformat(),
            "prompt": prompt,
            "input": input_text,
            "output": output_text,
        }
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path
    except Exception as e:  # noqa: BLE001 — must not crash callers
        logger.debug("persist_call_blob failed for %s: %s", call_id, e)
        return None


def load_call_blob(call_id: str) -> dict | None:
    """Read back a snapshot by id. Searches the date-stamped subdirectories."""
    root = _blob_store_root()
    if not root.exists():
        return None
    # Search each day-directory; small fan-out keeps it cheap.
    matches = list(root.glob(f"*/{call_id}.json.gz"))
    if not matches:
        return None
    try:
        with gzip.open(matches[0], "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:  # noqa: BLE001
        logger.debug("load_call_blob failed for %s: %s", call_id, e)
        return None
