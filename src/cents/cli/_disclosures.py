"""Shared disclosure footer for performance-reporting CLI commands.

cents is a research tool, not an investing tool. Real-money trading is out of
scope. Every CLI surface that emits performance numbers (P&L, win-rate, hit-
rate, forward returns, cohort spreads) must surface these caveats so they're
impossible to miss when readers copy numbers out of context.

This module also owns the sample-size warning threshold so it can't drift
across commands.
"""

from __future__ import annotations

# A bucket needs at least this many closed observations before its numbers
# are worth quoting. Below this we surface a strong warning.
LOW_N_THRESHOLD = 30


def disclosure_text(
    *,
    audience: str = "personal",
    costs_applied: bool = False,
) -> str:
    """Render the standard disclosure footer.

    Args:
        audience: Reserved for future variants (e.g. ``"third-party"``).
            Currently unused; the default ``"personal"`` block is always
            returned but the argument is part of the stable interface so
            callers don't need to be rewired when variants land.
        costs_applied: If ``False`` (the default), the disclosure states the
            numbers are *gross of costs*. When the analytics layer eventually
            models slippage / borrow / commission, callers can pass ``True``
            and the wording flips to "net of modeled costs".

    Returns:
        Multi-line plain-text block, no leading or trailing whitespace.
    """
    cost_line = (
        "- Returns shown are NET of modeled costs (slippage / borrow / commission)."
        if costs_applied
        else "- Returns shown are GROSS of costs (no slippage, borrow, or commission modeled)."
    )
    lines = [
        "Disclosures:",
        "- cents is a research tool, not an investing tool.",
        "- Real-money trading is out of scope; these numbers come from a model, not actual fills.",
        cost_line,
        "- Past performance does not predict future returns.",
        "- Time period reflects only the data persisted locally; results are sample-dependent.",
    ]
    return "\n".join(lines)


def low_n_warning(n: int, *, threshold: int = LOW_N_THRESHOLD) -> str | None:
    """Return a warning string when the sample size is below ``threshold``.

    Args:
        n: Observed sample size (closed theses, signals, etc.).
        threshold: Minimum sample size considered statistically meaningful.
            Defaults to :data:`LOW_N_THRESHOLD` (30).

    Returns:
        A human-readable warning when ``n < threshold``; ``None`` otherwise.
    """
    if n >= threshold:
        return None
    return (
        f"WARNING: low sample size (N={n}, threshold={threshold}). "
        "Results are not statistically meaningful and should be treated as anecdotal."
    )
