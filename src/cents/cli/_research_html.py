"""Self-contained HTML rendering for `cents research --export-html`.

Inline CSS only — the produced file has no external assets so it can be
iframe-embedded on the docs site or shared standalone.
"""
from __future__ import annotations

from html import escape
from typing import Any

from cents.models import EvidenceType


EVIDENCE_ICONS: dict[str, str] = {
    EvidenceType.SUPPORTING.value: "+",
    EvidenceType.CONTRADICTING.value: "-",
    EvidenceType.NEUTRAL.value: "~",
}


_HTML_STYLE = """
:root {
  color-scheme: light dark;
  --accent: #16a34a;
  --bg: #ffffff;
  --bg-card: #f8fafc;
  --fg: #0f172a;
  --fg-muted: #475569;
  --border: #e2e8f0;
  --pos: #16a34a;
  --neg: #dc2626;
  --neutral: #64748b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0b0f17;
    --bg-card: #111827;
    --fg: #e5e7eb;
    --fg-muted: #94a3b8;
    --border: #1f2937;
    --pos: #4ade80;
    --neg: #f87171;
    --neutral: #94a3b8;
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  margin: 0;
  padding: 2rem 1rem;
  line-height: 1.5;
}
.container { max-width: 920px; margin: 0 auto; }
header { border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
h1, h2, h3, .mono {
  font-family: ui-monospace, "SF Mono", Menlo, Monaco, "Cascadia Mono", monospace;
}
h1 { font-size: 2rem; margin: 0 0 0.5rem; letter-spacing: -0.02em; }
h2 { font-size: 1.25rem; margin: 2rem 0 1rem; color: var(--fg); }
.subtitle { color: var(--fg-muted); font-size: 0.9rem; margin: 0; }
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem;
  margin-top: 1.25rem;
}
.meta-item .label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--fg-muted);
}
.meta-item .value { font-family: ui-monospace, monospace; font-size: 1.1rem; font-weight: 600; }
.delta-pos { color: var(--pos); }
.delta-neg { color: var(--neg); }
.delta-neutral { color: var(--neutral); }
.accent { color: var(--accent); }
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  margin-bottom: 1rem;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}
.agent-name {
  font-family: ui-monospace, monospace;
  font-weight: 600;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.summary { color: var(--fg); margin: 0.5rem 0; }
.dimension-scores {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.75rem;
}
.dim-pill {
  font-family: ui-monospace, monospace;
  font-size: 0.75rem;
  padding: 0.2rem 0.55rem;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg);
}
.evidence-list { list-style: none; padding: 0; margin: 0; }
.evidence-item {
  border-left: 3px solid var(--neutral);
  padding: 0.6rem 0 0.6rem 0.75rem;
  margin-bottom: 0.6rem;
}
.evidence-item.supporting { border-left-color: var(--pos); }
.evidence-item.contradicting { border-left-color: var(--neg); }
.evidence-item.neutral { border-left-color: var(--neutral); }
.evidence-icon {
  display: inline-block;
  width: 1.1rem;
  font-family: ui-monospace, monospace;
  font-weight: 700;
}
.evidence-icon.supporting { color: var(--pos); }
.evidence-icon.contradicting { color: var(--neg); }
.evidence-icon.neutral { color: var(--neutral); }
.evidence-source {
  font-family: ui-monospace, monospace;
  font-size: 0.8rem;
  color: var(--fg-muted);
}
.evidence-content { margin: 0.25rem 0; }
.evidence-confidence { font-size: 0.75rem; color: var(--fg-muted); font-family: ui-monospace, monospace; }
footer { margin-top: 3rem; color: var(--fg-muted); font-size: 0.75rem; text-align: center; }
.disclaimer { margin-top: 0.5rem; font-style: italic; }
"""


def _esc(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def _delta_class(delta: float) -> str:
    if delta > 0.1:
        return "delta-pos"
    if delta < -0.1:
        return "delta-neg"
    return "delta-neutral"


def _render_evidence_item(evidence: dict[str, Any]) -> str:
    ev_type = evidence.get("type") or EvidenceType.NEUTRAL.value
    icon = EVIDENCE_ICONS.get(ev_type, "~")
    source = _esc(evidence.get("source") or evidence.get("agent") or "")
    content = _esc(evidence.get("content") or "")
    confidence = evidence.get("confidence")
    confidence_str = (
        f"confidence: {confidence:.2f}"
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        else ""
    )
    return f"""
    <li class="evidence-item {_esc(ev_type)}">
      <div>
        <span class="evidence-icon {_esc(ev_type)}">{icon}</span>
        <span class="evidence-source">{source}</span>
      </div>
      <div class="evidence-content">{content}</div>
      <div class="evidence-confidence">{_esc(confidence_str)}</div>
    </li>
    """.strip()


def _render_agent_card(agent_output: dict[str, Any]) -> str:
    name = _esc(agent_output.get("agent") or "")
    summary = _esc(agent_output.get("summary") or "")
    delta = float(agent_output.get("conviction_delta", 0.0) or 0.0)
    delta_str = f"{delta:+.1f}"
    dimension_scores = agent_output.get("dimension_scores") or {}
    dim_pills = "".join(
        f'<span class="dim-pill">{_esc(k)}: {float(v):+.1f}</span>'
        for k, v in dimension_scores.items()
    )
    evidence_items = "".join(
        _render_evidence_item(e) for e in agent_output.get("evidence") or []
    )
    evidence_block = (
        f'<ul class="evidence-list">{evidence_items}</ul>'
        if evidence_items
        else '<p class="subtitle">No evidence reported by this agent.</p>'
    )
    dim_block = f'<div class="dimension-scores">{dim_pills}</div>' if dim_pills else ""
    return f"""
    <section class="card">
      <div class="card-header">
        <span class="agent-name">{name}</span>
        <span class="mono {_delta_class(delta)}">Δ {delta_str}</span>
      </div>
      <p class="summary">{summary}</p>
      {dim_block}
      {evidence_block}
    </section>
    """.strip()


def render_research_html(payload: dict[str, Any]) -> str:
    """Render a research payload as a self-contained HTML document.

    The payload mirrors the JSON output shape of ``cents research`` plus a
    ``generated_at`` timestamp. No external assets are referenced.
    """
    symbol = _esc(payload.get("symbol") or "")
    price = payload.get("price")
    price_str = (
        f"${float(price):.2f}"
        if isinstance(price, (int, float)) and not isinstance(price, bool)
        else "—"
    )
    delta = float(payload.get("total_conviction_delta", 0.0) or 0.0)
    delta_str = f"{delta:+.1f}"
    generated_at = _esc(payload.get("generated_at") or "")
    as_of = _esc(payload.get("as_of") or "live")

    agent_cards = "".join(
        _render_agent_card(a) for a in payload.get("agents") or []
    )
    if not agent_cards:
        agent_cards = '<p class="subtitle">No agent results to display.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>cents research: {symbol}</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<div class="container">
  <header>
    <h1><span class="accent">cents</span> research / {symbol}</h1>
    <p class="subtitle">Thesis-driven, agent-orchestrated research report</p>
    <div class="meta-grid">
      <div class="meta-item">
        <div class="label">Symbol</div>
        <div class="value mono">{symbol}</div>
      </div>
      <div class="meta-item">
        <div class="label">Price</div>
        <div class="value mono">{_esc(price_str)}</div>
      </div>
      <div class="meta-item">
        <div class="label">Conviction Δ</div>
        <div class="value mono {_delta_class(delta)}">{delta_str}</div>
      </div>
      <div class="meta-item">
        <div class="label">As of</div>
        <div class="value mono">{as_of}</div>
      </div>
      <div class="meta-item">
        <div class="label">Generated</div>
        <div class="value mono">{generated_at}</div>
      </div>
    </div>
  </header>

  <h2>Agent results</h2>
  {agent_cards}

  <footer>
    Generated by <span class="accent">cents</span> — thesis-driven investment research CLI.
    <div class="disclaimer">Not financial advice. For research and educational use only.</div>
  </footer>
</div>
</body>
</html>
"""
