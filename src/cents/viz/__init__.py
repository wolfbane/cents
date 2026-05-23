"""Visualization layer for cents.

Two render paths over a single data layer:

- ``cents.viz.queries`` — SQL → dataclasses. No third-party deps; always
  importable. Every chart pulls from here, so the same data backs the
  terminal dashboard and the static report exports.
- ``cents.viz.ascii`` — rich + plotext renderers for `cents pilot dashboard`.
- ``cents.viz.static`` — matplotlib renderers for `cents report`.
- ``cents.viz.sunburst`` — plotly sunburst (chart 9 only).

``queries`` has no third-party deps and is always importable.
``ascii`` imports rich + plotext at module top — the CLI wrapper
(``cents.cli.pilot``) catches the ImportError. ``static`` and
``sunburst`` import matplotlib / plotly lazily inside their renderer
functions; ``cents.cli.report`` pre-flights those imports so a missing
``[viz]`` extra surfaces as a friendly install hint, not a mid-render
RuntimeError.
"""

from cents.viz import queries

__all__ = ["queries"]
