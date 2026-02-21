# MALTbot🦞

## Documentation

- [Matbench Ops Rules](docs/MATBENCH_OPS.md)
- [Daily Colab Run](docs/colab.md)
- [Progress Plot Automation](docs/progress.md)

## CHGNet finetune track (`matbench_mp_e_form`)

- Runner: `scripts/run_chgnet_mp_e_form.py`
- Config: `configs/chgnet_mp_e_form.yaml`
- Canonical notebook: `notebooks/MALTbot_2.ipynb`
- Results artifact: `results/daily/YYYY-MM-DD/<run_name>/results.json`
- Daily summary log: `RESULTS.md`

## Daily progress (auto-updated)

`RESULTS.md` and `results/daily/*/results.json` are parsed to regenerate these charts.
Dashed lines are Top-1..Top-5 leaderboard references.

### `matbench_mp_e_form`

![Progress: matbench_mp_e_form](docs/assets/progress_matbench_mp_e_form.svg)

### `matbench_mp_gap`

![Progress: matbench_mp_gap](docs/assets/progress_matbench_mp_gap.svg)

`notebooks/route_b_chgnet_matbench_mp_e_form_colab.ipynb` is kept as a compatibility pointer to `MALTbot_2.ipynb`.
