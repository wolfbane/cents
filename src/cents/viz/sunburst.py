"""Agent-contribution sunburst (chart 9), via plotly.

Two-level pie: outer ring is per-agent total ``conviction_delta`` on
correct theses, inner ring is the top contributing rule/metric per
agent (parsed from the evidence row's content suffix — see CLAUDE.md
"Agent evidence rows carry fired-rule attribution").

This is the only chart that uses plotly. The rest of the viz stack
sticks to matplotlib for static exports because PNGs are easier to
embed in the Starlight site. Sunbursts are genuinely better as
interactive HTML — the hover-to-reveal the inner-ring labels matters
on a chart with 7 agents × N rules each.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Sequence

from cents.db.repository import EvidenceRepository
from cents.viz.queries import ThesisRow
from cents.models.thesis import ThesisOutcome, ThesisStatus


# Rule attribution suffix: "Unemployment Rate 4.30 — low_level: UNRATE < 4.5%"
# We split on the em-dash and take the rule name before the colon.
_RULE_RE = re.compile(r"\s—\s*([a-z_][a-z0-9_]*)\s*:", re.IGNORECASE)


def _rule_from_content(content: str) -> str:
    m = _RULE_RE.search(content or "")
    if m:
        return m.group(1)
    # Older evidence rows have no attribution suffix — bucket them.
    return "unattributed"


def render_agent_contribution_sunburst(
    rows: Sequence[ThesisRow],
    *,
    out_dir: Path,
    name: str = "agent_contribution",
) -> Path:
    """Write ``<name>.html`` to ``out_dir`` and a JSON sidecar.

    Counts evidence rows on CORRECT theses, grouped by agent and rule.
    Returns the HTML path.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "agent-contribution sunburst needs the [viz] extra: pip install -e '.[viz]'"
        ) from exc

    correct_ids = {
        r.id for r in rows
        if r.status == ThesisStatus.CLOSED and r.outcome == ThesisOutcome.CORRECT
    }
    repo = EvidenceRepository()
    # Walk evidence per correct thesis. The repo has list_for_thesis;
    # avoid a full-table scan because evidence rows can be plentiful.
    by_agent_rule: dict[tuple[str, str], int] = defaultdict(int)
    for tid in correct_ids:
        for ev in repo.list_for_thesis(tid):
            rule = _rule_from_content(ev.content)
            by_agent_rule[(ev.agent, rule)] += 1

    # Plotly sunburst expects flat lists with parent pointers. Root
    # label is "correct theses (n=N)".
    root = f"correct theses (n={len(correct_ids)})"
    labels: list[str] = [root]
    parents: list[str] = [""]
    values: list[int] = [sum(by_agent_rule.values())]

    by_agent: dict[str, int] = defaultdict(int)
    for (agent, _rule), v in by_agent_rule.items():
        by_agent[agent] += v
    for agent, v in sorted(by_agent.items(), key=lambda kv: -kv[1]):
        labels.append(agent)
        parents.append(root)
        values.append(v)
    for (agent, rule), v in sorted(by_agent_rule.items(), key=lambda kv: -kv[1]):
        # Plotly requires unique labels in a tree; prefix rule with
        # agent so two agents can share a rule name.
        labels.append(f"{agent}/{rule}")
        parents.append(agent)
        values.append(v)

    fig = go.Figure(go.Sunburst(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>contribution: %{value}<extra></extra>",
    ))
    fig.update_layout(
        title="Agent contribution on correct theses (by fired rule)",
        margin=dict(t=60, l=10, r=10, b=10),
    )

    html_path = out_dir / f"{name}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")

    sidecar = {
        "root_n": len(correct_ids),
        "totals_by_agent": dict(by_agent),
        "totals_by_agent_rule": {f"{a}/{r}": v for (a, r), v in by_agent_rule.items()},
    }
    (out_dir / f"{name}.json").write_text(json.dumps(sidecar, indent=2))
    return html_path
