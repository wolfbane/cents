# Examples

Sample artifacts referenced by the docs site at `dollarsandcents.ai`. The site serves them out of `../website/public/` — regenerate **directly into those paths** so the homepage demo and `agents.mdx` iframe pick them up:

- **`../website/public/demo.cast`** — asciinema recording of an end-to-end session.

  ```bash
  asciinema rec ../website/public/demo.cast
  ```

- **`../website/public/examples/research-NVDA.html`** — sample HTML report.

  ```bash
  cents research NVDA --export-html ../website/public/examples/research-NVDA.html
  ```

Both require API keys configured (`~/.cents/config.toml`). Re-run whenever the CLI surface or output formatting changes meaningfully.
