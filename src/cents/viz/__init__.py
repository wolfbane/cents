"""Visualization layer for cents.

Two render paths over a single data layer:

- ``cents.viz.queries`` — SQL → dataclasses. No third-party deps; always
  importable. Every chart pulls from here, so the same data backs the
  terminal dashboard and the static report exports.
- ``cents.viz.ascii`` — rich + plotext renderers for `cents pilot dashboard`.
- ``cents.viz.static`` — matplotlib renderers for `cents report`.
- ``cents.viz.sunburst`` — plotly sunburst (chart 9 only).

The ascii/static modules import their third-party deps lazily so a missing
``[viz]`` extra surfaces a friendly error at command time, not at import.
"""

from cents.viz import queries

__all__ = ["queries"]
