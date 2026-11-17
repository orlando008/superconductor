# Project website

`index.html` is the public explainer, published to GitHub Pages by
`.github/workflows/pages.yml` whenever `site/` changes on `main`.

It is a single self-contained file: inline CSS and JavaScript, fonts from Google Fonts, no
build step. Open it directly in a browser to preview changes.

## Guidelines

- **Keep it accurate.** Every number on the page must be real and sourced (the data tables
  under each chart show them). Schematic parts are labelled as schematic, and the lattice
  animation is labelled as an illustration, not a simulation.
- **Keep it plain.** The audience is curious non-physicists. Explain terms where they first
  appear.
- **Keep project facts current.** The layer statuses mirror `docs/roadmap.md`; update both
  together.
- **All text and data go in the HTML.** JavaScript only draws the charts and runs the grid
  animation. Many AI agents read pages without running scripts, so anything written by script
  is invisible to them. Every chart's numbers also appear in its "Show the numbers" table.
- **`llms.txt`** is the short guide for AI agents (the [llms.txt](https://llmstxt.org/)
  convention). Keep its links in sync when files move.
- Lines ending in `<!--pages-only-->` are the HTML document wrapper. They are stripped when
  the page is previewed in hosts that supply their own wrapper; leave the markers in place.
