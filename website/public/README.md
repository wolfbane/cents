# /public assets

Static files copied verbatim into the build output.

## Placeholders

- `demo.cast` — placeholder. Replace with the real asciinema recording produced
  in Phase 2 (`examples/demo.cast` from the repo root). The
  `<AsciinemaPlayer>` component shows a "demo recording coming soon" message
  until the file is non-trivial.
- `examples/` — landing zone for `research-NVDA.html` (rendered via
  `cents research --export html`). Until that artifact lands, the iframe on
  `/agents` shows a fallback message.
