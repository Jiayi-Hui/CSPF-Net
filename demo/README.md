# Demo Frontend

This folder contains a presentation-oriented frontend prototype for CSPF-Net.

Open it locally with any static server, for example:

```bash
cd demo
python -m http.server 8000
```

Then visit `http://localhost:8000`.

Notes:

- `Analyze Text` currently uses a lightweight browser-side heuristic to mock the final interaction.
- The layout is designed so the analyzer panel can later be connected to a Python API endpoint.
- `Human or Machine?` is a classroom demo game based on curated sample cases.
