# Progress Plot Automation

This repo auto-generates daily progress plots from experiment logs.

## Data source

- Primary: `RESULTS.md` lines in format:
  - `DATE | TASK | MODEL_CONFIG | METRIC=... | NOTE | path`
- Fallback for `METRIC=TBD` / missing numeric metric:
  - load metric from referenced `results/daily/.../results.json` (`task.scores`)

Supported tasks:
- `matbench_mp_e_form`
- `matbench_mp_gap`

## Generator

- Script: `scripts/generate_progress_plot.py`
- Outputs:
  - `docs/assets/progress_matbench_mp_e_form.svg`
  - `docs/assets/progress_matbench_mp_gap.svg`
  - `docs/assets/progress_data.json`

The plot overlays our run trajectory with Top-1..Top-5 leaderboard reference lines.
Reference values are currently sourced from `docs/research/2026-02-21_strategy_top1.md`.

## GitHub Actions

Workflow: `.github/workflows/progress_plot.yml`

Triggers:
- push to `main` when results or plot script changes
- daily schedule
- manual dispatch

Behavior:
- regenerate plot assets
- commit only generated assets (`docs/assets/progress_*.svg`, `docs/assets/progress_data.json`) using bot identity
